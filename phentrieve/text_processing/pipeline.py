"""
Text processing pipeline orchestrator for Phentrieve.

This module provides pipeline components that manage the sequence
of text processing operations including chunking and assertion detection.
"""

import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from phentrieve.text_processing.assertion_detection import (
    AssertionDetector,
    AssertionStatus,
    CombinedAssertionDetector,
    _is_cue_match,
    get_spacy_model,
    load_context_cues,
    pseudo_cue_spans,
)
from phentrieve.text_processing.chunkers import (
    NoOpChunker,
    TextChunker,
)
from phentrieve.text_processing.cleaners import (
    clean_internal_newlines_and_extra_spaces,
    normalize_line_endings,
)
from phentrieve.text_processing.spans import find_span_in_text
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)

_SENTENCE_BREAKS = (".", "!", "?", ";", "\n")
_SUBJECT_DEPS = ("nsubj", "nsubjpass", "nsubj:pass", "sb")
FAMILY_EXPERIENCER = "family_history"


def _cue_spans(text_lower: str, cues: tuple[str, ...]) -> list[tuple[int, int]]:
    """All word-bounded matches of the cues, sorted by position."""
    spans = []
    for cue in cues:
        index = text_lower.find(cue)
        while index >= 0:
            if _is_cue_match(text_lower, cue, index):
                spans.append((index, index + len(cue)))
            index = text_lower.find(cue, index + 1)
    return sorted(spans)


class _SentenceContext:
    """Cue and parse lookups over the sentences of one document (#400)."""

    def __init__(self, text: str, language: str, pseudo_spans: list[tuple[int, int]]):
        self.text = text
        self.lower = text.lower()
        # "not only" / "cannot be ruled out" hold no usable cue
        for start, end in pseudo_spans:
            self.lower = self.lower[:start] + " " * (end - start) + self.lower[end:]
        self.nlp = get_spacy_model(language)
        self.forward_negation = load_context_cues(
            language, "NEGATED_EXISTENCE", "FORWARD"
        )
        self.terminate = load_context_cues(language, "TERMINATE")
        self.family = load_context_cues(language, "FAMILY")
        self._docs: dict[int, Any] = {}

    def sentence_start(self, pos: int) -> int:
        return max(self.text.rfind(t, 0, pos) for t in _SENTENCE_BREAKS) + 1

    def cues(
        self, cues: tuple[str, ...], start: int, end: int
    ) -> list[tuple[int, int]]:
        return [
            (start + s, start + e) for s, e in _cue_spans(self.lower[start:end], cues)
        ]

    def tokens(self, start: int, end: int) -> list[Any]:
        """Parsed tokens in [start, end); the whole sentence is parsed for context."""
        if self.nlp is None:
            return []
        sent_start = self.sentence_start(start)
        if sent_start not in self._docs:
            sent_end = min(
                (i for t in _SENTENCE_BREAKS if (i := self.text.find(t, start)) >= 0),
                default=len(self.text),
            )
            self._docs[sent_start] = self.nlp(self.text[sent_start:sent_end])
        return [
            t
            for t in self._docs[sent_start]
            if start <= sent_start + t.idx < end and not t.is_space
        ]

    def opens_clause(self, start: int, end: int) -> bool:
        """True if [start, end) holds a new clause: a finite verb or a subject."""
        return any(
            "Fin" in t.morph.get("VerbForm") or t.dep_ in _SUBJECT_DEPS
            for t in self.tokens(start, end)
        )


def _propagate_list_negation(
    context: _SentenceContext,
    chunks: list[dict[str, Any]],
    spans: list[tuple[int, int]],
) -> None:
    """Carry a forward negation cue over the coordinated list it introduces.

    Chunking splits "No A, B or C" into "A" | "B" | "C", and only the chunk next
    to the cue saw it (#400). A chunk inherits the negation of the chunk before
    it when both are in the same sentence and the text between holds no scope
    terminator (but, however, ...) and no new clause.
    """
    if context.nlp is None:
        return
    for idx in range(1, len(chunks)):
        prev, chunk = chunks[idx - 1], chunks[idx]
        (prev_start, prev_end), (start, end) = spans[idx - 1], spans[idx]
        if (
            prev["status"] is not AssertionStatus.NEGATED
            or chunk["status"] is not AssertionStatus.AFFIRMED
            or prev_start < 0
            or start < prev_end
        ):
            continue
        if not prev["assertion_details"].get("propagated_negation"):
            # The cue must open a list: a forward cue, and the chunk must end
            # in its object ("Seizures were not observed, ataxia" does not).
            cue_spans = context.cues(
                context.forward_negation, context.sentence_start(prev_start), prev_end
            )
            prev_tokens = [
                t for t in context.tokens(prev_start, prev_end) if not t.is_punct
            ]
            if (
                not cue_spans
                or context.cues(context.terminate, cue_spans[-1][1], prev_end)
                or not prev_tokens
                or prev_tokens[-1].pos_ in ("VERB", "AUX")
            ):
                continue
        between = context.text[prev_end:start]
        if (
            any(t in between for t in _SENTENCE_BREAKS)
            or context.cues(context.terminate, prev_end, end)
            or context.opens_clause(prev_end, end)
        ):
            continue
        chunk["status"] = AssertionStatus.NEGATED
        chunk["assertion_details"] = {
            **chunk["assertion_details"],
            "final_status": AssertionStatus.NEGATED.value,
            # The whole chunk is in scope, so the chunk-level status applies.
            "negated_scope_texts": [],
            "propagated_negation": True,
        }


def _mark_family_experiencer(
    context: _SentenceContext,
    chunks: list[dict[str, Any]],
    spans: list[tuple[int, int]],
) -> None:
    """Mark chunks about a relative ("His brother has scoliosis", #400).

    A family cue from the bundled ConText rules counts for the rest of its
    sentence, up to a scope terminator or another subject ("Mother reports
    that he has seizures" stays the patient's).
    shortcut: cue-after-finding ("scoliosis in his brother") and other people
    (donor, neighbour) are not detected; extend the rules if that matters.
    """
    if context.nlp is None or not context.family:
        return
    for chunk, (start, end) in zip(chunks, spans, strict=True):
        if start < 0:
            continue
        cue_spans = context.cues(context.family, context.sentence_start(start), end)
        if not cue_spans:
            continue
        cue_end = cue_spans[-1][1]
        if context.cues(context.terminate, cue_end, end) or any(
            t.dep_ in _SUBJECT_DEPS for t in context.tokens(cue_end, end)
        ):
            continue
        chunk["assertion_details"] = {
            **chunk["assertion_details"],
            "experiencer": FAMILY_EXPERIENCER,
        }


def _splits_pseudo_cue(
    chunk_span: tuple[int, int], pseudo_spans: list[tuple[int, int]]
) -> bool:
    """True if the chunk holds only part of a phrase like "cannot be ruled out".

    The detector cannot see such a phrase across chunks, so each half reads as
    negated. A chunk holding the whole phrase was already judged correctly.
    """
    chunk_start, chunk_end = chunk_span
    return any(
        chunk_start < pseudo_end
        and chunk_end > pseudo_start
        and not (chunk_start <= pseudo_start and pseudo_end <= chunk_end)
        for pseudo_start, pseudo_end in pseudo_spans
    )


class TextProcessingPipeline:
    """
    Pipeline orchestrator for text processing.

    This class manages the sequence of text processing steps,
    including text chunking and assertion detection.
    """

    def __init__(
        self,
        language: str,
        chunking_pipeline_config: list[dict],
        assertion_config: dict,
        sbert_model_for_semantic_chunking: SentenceTransformer | None = None,
    ):
        """
        Initialize the text processing pipeline.

        Args:
            language: ISO language code ('en', 'de', etc.)
            chunking_pipeline_config: List of dicts configuring the chunking pipeline.
                Example: [
                    {'type': 'paragraph'},
                    {'type': 'semantic', 'similarity_threshold': 0.4,
                     'min_sentences': 1, 'max_sentences': 3},
                    {'type': 'fine_grained_punctuation'}
                ]
            assertion_config: Dict configuring assertion detection.
                Example: {
                    'enable_keyword': True,
                    'enable_dependency': True,
                    'preference': 'dependency'
                }
            sbert_model_for_semantic_chunking: Pre-loaded SentenceTransformer model
                for semantic chunking (required if 'semantic' type is in the pipeline)
        """
        self.language = language
        self.chunking_pipeline_config = chunking_pipeline_config
        self.assertion_config = assertion_config
        self.sbert_model = sbert_model_for_semantic_chunking

        # Create chunkers list
        self.chunkers = self._create_chunkers()

        # Create assertion detector
        self.assertion_detector = self._create_assertion_detector()

        logger.info(
            "Initialized TextProcessingPipeline with %s chunking stages and assertion detection config: %s",
            len(self.chunkers),
            _sanitize(str(assertion_config)),
        )

    def _create_chunkers(self) -> list[TextChunker]:
        """Create chunker instances based on configuration.

        Delegates the per-type construction to the registry in
        _chunker_registry.py so this method stays a thin loop.
        """
        from phentrieve.text_processing._chunker_registry import build_chunker

        chunkers: list[TextChunker] = []
        for stage_config in self.chunking_pipeline_config:
            if isinstance(stage_config, dict):
                chunker_type = stage_config.get("type", "unknown")
                chunker_config = stage_config.get("config", {})
            else:
                chunker_type = stage_config
                chunker_config = {}

            chunker = build_chunker(
                chunker_type=chunker_type,
                chunker_config=chunker_config,
                language=self.language,
                sbert_model=self.sbert_model,
            )
            if chunker is None:
                logger.warning(
                    "Unknown chunker type '%s' in config, skipping",
                    _sanitize(chunker_type),
                )
                continue
            chunkers.append(chunker)

        if not chunkers:
            logger.warning(
                "No valid chunkers specified in config, using NoOpChunker as fallback."
            )
            chunkers.append(NoOpChunker(language=self.language))
        return chunkers

    def _create_assertion_detector(self) -> AssertionDetector:
        """
        Create assertion detector based on configuration.

        Returns:
            Initialized assertion detector
        """
        # Check if assertion detection is disabled
        if self.assertion_config.get("disable", False):
            # Return a placeholder detector that always returns AFFIRMED
            class PlaceholderDetector(AssertionDetector):
                def detect(self, text_chunk: str):
                    return AssertionStatus.AFFIRMED, {"detection_disabled": True}

            return PlaceholderDetector(language=self.language)

        # Extract configuration
        enable_keyword = self.assertion_config.get("enable_keyword", True)
        enable_dependency = self.assertion_config.get("enable_dependency", True)
        preference = self.assertion_config.get("preference", "dependency")

        # Create combined detector with the specified configuration
        return CombinedAssertionDetector(
            language=self.language,
            enable_keyword=enable_keyword,
            enable_dependency=enable_dependency,
            preference=preference,
        )

    def process(
        self, raw_text: str, include_positions: bool = False
    ) -> list[dict[str, Any]]:
        """
        Process raw text through the chunking pipeline and assertion detection.

        Args:
            raw_text: The raw text to process
            include_positions: If True, include start_char/end_char positions
                in output for each chunk (relative to raw_text)

        Returns:
            List of processed chunks with assertion information:
            [
                {
                    'text': str,                # The chunk text
                    'status': AssertionStatus,  # Enum value indicating assertion status
                    'assertion_details': Dict,  # Details about the assertion analysis
                    'source_indices': Dict,     # Tracking information about source
                    'start_char': int,          # Position in raw_text (if include_positions)
                    'end_char': int,            # Position in raw_text (if include_positions)
                },
                ...
            ]
        """
        if not raw_text or not raw_text.strip():
            logger.warning(
                "TextProcessingPipeline.process called with empty input text."
            )
            return []

        # Store original text for position tracking (BEFORE any normalization)
        original_text = raw_text

        normalized_text = normalize_line_endings(raw_text)
        current_segments_for_stage: list[str] = [normalized_text]

        # Simplified source tracking for this refactor:
        # Each element in current_source_info_list corresponds to a segment in current_segments_for_stage
        # It stores a list of strings describing the applied chunkers.
        current_source_info_list: list[list[str]] = [["initial_raw_text"]]

        for chunker_idx, chunker_instance in enumerate(self.chunkers):
            chunker_name = chunker_instance.__class__.__name__
            logger.debug(
                "Pipeline Stage %s: Applying %s to %s segment(s).",
                chunker_idx + 1,
                chunker_name,
                len(current_segments_for_stage),
            )

            input_segments_to_current_chunker = current_segments_for_stage
            input_source_info_to_current_chunker = current_source_info_list

            output_segments_from_chunker = chunker_instance.chunk(
                input_segments_to_current_chunker
            )

            # Basic source tracking update:
            # If a chunker doesn't change the number of segments, source info maps 1:1.
            # If it splits one segment into many, all new segments inherit and append the current chunker's name.
            # This is a simplification. True lineage is harder.
            # For now, we'll just track that this chunker was applied.
            # A more robust way would be for chunkers to also return source mapping.

            updated_source_info_list = []
            if (
                len(input_segments_to_current_chunker)
                == len(output_segments_from_chunker)
                and len(input_segments_to_current_chunker) > 0
            ):
                # Assume 1:1 mapping if segment count doesn't change
                for i, _ in enumerate(output_segments_from_chunker):
                    updated_source_info_list.append(
                        input_source_info_to_current_chunker[i] + [chunker_name]
                    )
            else:  # Segment count changed, apply new source info more generically
                for _ in output_segments_from_chunker:
                    # This is a simplification; ideally, we'd know which input segment(s) an output segment came from.
                    # For now, just indicate this chunker ran. A more complex approach would be needed for precise lineage.
                    updated_source_info_list.append([f"processed_by_{chunker_name}"])

            current_segments_for_stage = output_segments_from_chunker
            current_source_info_list = updated_source_info_list

            logger.debug(
                "Stage %s (%s) produced %s segment(s).",
                chunker_idx + 1,
                chunker_name,
                len(current_segments_for_stage),
            )

        final_raw_chunks_text_only: list[str] = current_segments_for_stage

        processed_chunks_with_assertion: list[dict[str, Any]] = []
        assertion_inputs: list[str] = []
        chunk_document_spans: list[tuple[int, int]] = []
        search_start = 0  # Track position for sequential search (handles duplicates)

        for idx, final_text_chunk in enumerate(final_raw_chunks_text_only):
            # Locate the chunk in the original text. We always attempt this (not
            # only when include_positions is set) because the assertion detector
            # needs the chunk's surrounding sentence context: the final-chunk
            # cleaner strips leading negation cues ("no", "does not have") from the
            # chunk text, so detecting on the cleaned chunk alone loses polarity
            # (the C1 false-positive class). The span lets us recover the cue from
            # original_text without changing the retrieval/display chunk text.
            start_char, end_char = -1, -1
            assertion_context_start = search_start
            span = find_span_in_text(final_text_chunk, original_text, search_start)
            if span:
                start_char, end_char = span.start_char, span.end_char
                # Restore the within-sentence context that precedes this chunk
                # (where a stripped leading cue lives), bounded by the previous
                # chunk (search_start) and the current sentence so a following
                # concept's cue is not pulled into this chunk's scope.
                gap = original_text[search_start:start_char]
                last_terminator = max(
                    (gap.rfind(t) for t in _SENTENCE_BREAKS),
                    default=-1,
                )
                if last_terminator >= 0:
                    assertion_context_start = search_start + last_terminator + 1
                search_start = span.end_char  # Continue from end of found chunk

            cleaned_final_chunk = clean_internal_newlines_and_extra_spaces(
                final_text_chunk
            )
            if not cleaned_final_chunk:
                continue

            # Detect assertion over the chunk plus its restored leading context so
            # prepositional negation ("no X", "does not have X") is honored. Fall
            # back to the cleaned chunk when the span could not be located.
            assertion_input = cleaned_final_chunk
            if span and end_char > assertion_context_start:
                context_text = original_text[assertion_context_start:end_char].strip()
                if context_text:
                    assertion_input = context_text

            assertion_inputs.append(assertion_input)

            source_info_for_chunk = (
                current_source_info_list[idx]
                if idx < len(current_source_info_list)
                else ["unknown_source"]
            )

            # "status" and "assertion_details" are filled in below, after one
            # batched assertion-detection call for all chunks (#410).
            chunk_document_spans.append((start_char, end_char))
            chunk_data: dict[str, Any] = {
                "text": cleaned_final_chunk,
                "status": None,
                "assertion_details": None,
                "source_indices": {"processing_stages": source_info_for_chunk},
            }

            # Only include positions when requested (backward compatibility)
            if include_positions:
                chunk_data["start_char"] = start_char
                chunk_data["end_char"] = end_char

            processed_chunks_with_assertion.append(chunk_data)

        assertion_results = self.assertion_detector.detect_batch(assertion_inputs)
        pseudo_spans = pseudo_cue_spans(original_text, self.language)
        for chunk_data, chunk_span, (assertion_status, assertion_details) in zip(
            processed_chunks_with_assertion,
            chunk_document_spans,
            assertion_results,
            strict=True,
        ):
            if assertion_status is AssertionStatus.NEGATED and _splits_pseudo_cue(
                chunk_span, pseudo_spans
            ):
                assertion_status = AssertionStatus.AFFIRMED
                assertion_details = {**assertion_details, "negated_scope_texts": []}
            chunk_data["status"] = assertion_status
            chunk_data["assertion_details"] = assertion_details

        if not self.assertion_config.get("disable", False):
            context = _SentenceContext(original_text, self.language, pseudo_spans)
            _propagate_list_negation(
                context, processed_chunks_with_assertion, chunk_document_spans
            )
            _mark_family_experiencer(
                context, processed_chunks_with_assertion, chunk_document_spans
            )

        logger.info(
            "TextProcessingPipeline processed text into %s final asserted chunks.",
            len(processed_chunks_with_assertion),
        )
        return processed_chunks_with_assertion
