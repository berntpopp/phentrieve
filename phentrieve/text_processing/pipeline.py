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
                    (gap.rfind(t) for t in (".", "!", "?", ";", "\n")),
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

            assertion_status, assertion_details = self.assertion_detector.detect(
                assertion_input
            )

            source_info_for_chunk = (
                current_source_info_list[idx]
                if idx < len(current_source_info_list)
                else ["unknown_source"]
            )

            chunk_data: dict[str, Any] = {
                "text": cleaned_final_chunk,
                "status": assertion_status,
                "assertion_details": assertion_details,
                "source_indices": {"processing_stages": source_info_for_chunk},
            }

            # Only include positions when requested (backward compatibility)
            if include_positions:
                chunk_data["start_char"] = start_char
                chunk_data["end_char"] = end_char

            processed_chunks_with_assertion.append(chunk_data)

        logger.info(
            "TextProcessingPipeline processed text into %s final asserted chunks.",
            len(processed_chunks_with_assertion),
        )
        return processed_chunks_with_assertion
