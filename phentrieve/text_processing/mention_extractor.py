"""
Mention extraction for clinical finding discovery.

This module identifies candidate clinical finding spans (mentions)
in text using NLP-based extraction and filtering. Mentions are the
core unit of processing in the mention-level HPO extraction pipeline.

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Span

from phentrieve.text_processing.document_structure import (
    DocumentStructure,
    DocumentStructureDetector,
)
from phentrieve.text_processing.mention import Mention

logger = logging.getLogger(__name__)

# Stopwords and patterns to filter out non-clinical mentions
STOPWORD_PATTERNS: list[str] = [
    r"^(?:the|a|an|this|that|these|those|it|its)$",
    r"^(?:he|she|they|we|i|you|his|her|their|our|my|your)$",
    r"^(?:is|are|was|were|be|been|being|have|has|had)$",
    r"^(?:and|or|but|if|then|so|because|although|however)$",
    r"^\d+$",  # Pure numbers
    r"^[.,:;!?\-\(\)]+$",  # Punctuation
]

# Minimum/maximum mention length in words
DEFAULT_MIN_MENTION_WORDS = 1
DEFAULT_MAX_MENTION_WORDS = 10
DEFAULT_MIN_MENTION_CHARS = 2
DEFAULT_MAX_MENTION_CHARS = 100

# POS tags for clinical findings (nouns, adjectives)
CLINICAL_POS_TAGS = {"NOUN", "PROPN", "ADJ"}

# Dependency labels for clinical finding heads
CLINICAL_DEP_LABELS = {"nsubj", "dobj", "pobj", "attr", "ROOT", "appos", "conj"}


@dataclass
class MentionExtractionConfig:
    """
    Configuration for mention extraction.

    Attributes:
        min_mention_words: Minimum words in a mention
        max_mention_words: Maximum words in a mention
        min_mention_chars: Minimum characters in a mention
        max_mention_chars: Maximum characters in a mention
        include_noun_phrases: Extract noun phrases
        include_adj_noun_phrases: Include adjective-noun combinations
        include_verb_phrases: Extract verb phrases (for symptom descriptions)
        filter_stopwords: Filter out stopword-only mentions
        filter_pronouns: Filter out pronoun-only mentions
        context_window_chars: Characters of context to include around mentions
    """

    min_mention_words: int = DEFAULT_MIN_MENTION_WORDS
    max_mention_words: int = DEFAULT_MAX_MENTION_WORDS
    min_mention_chars: int = DEFAULT_MIN_MENTION_CHARS
    max_mention_chars: int = DEFAULT_MAX_MENTION_CHARS
    include_noun_phrases: bool = True
    include_adj_noun_phrases: bool = True
    include_verb_phrases: bool = False
    filter_stopwords: bool = True
    filter_pronouns: bool = True
    context_window_chars: int = 50


class MentionExtractor:
    """
    Extract clinical finding mentions from text.

    Uses spaCy NLP for linguistic analysis and applies clinical
    domain filters to identify relevant text spans.

    Example:
        >>> extractor = MentionExtractor(language="en")
        >>> mentions = extractor.extract("Patient has seizures and headaches.")
        >>> [m.text for m in mentions]
        ['seizures', 'headaches']
    """

    # spaCy model mapping
    SPACY_MODELS: dict[str, str] = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "es": "es_core_news_sm",
        "fr": "fr_core_news_sm",
        "nl": "nl_core_news_sm",
    }

    def __init__(
        self,
        language: str = "en",
        config: MentionExtractionConfig | None = None,
        nlp: Language | None = None,
    ):
        """
        Initialize the mention extractor.

        Args:
            language: Language code (e.g., "en", "de")
            config: Extraction configuration
            nlp: Optional pre-loaded spaCy model
        """
        self.language = language
        self.config = config or MentionExtractionConfig()
        self._nlp = nlp
        self._structure_detector: DocumentStructureDetector | None = None

    @property
    def nlp(self) -> Language:
        """Get the spaCy model, loading if necessary."""
        if self._nlp is None:
            self._nlp = self._load_nlp()
        return self._nlp

    @property
    def structure_detector(self) -> DocumentStructureDetector:
        """Get the document structure detector."""
        if self._structure_detector is None:
            self._structure_detector = DocumentStructureDetector(
                language=self.language,
                nlp=self.nlp,
            )
        return self._structure_detector

    def _load_nlp(self) -> Language:
        """Load the appropriate spaCy model."""
        import spacy

        model_name = self.SPACY_MODELS.get(self.language, "en_core_web_sm")

        try:
            return spacy.load(model_name)
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found, falling back to 'en_core_web_sm'"
            )
            return spacy.load("en_core_web_sm")

    def extract(
        self,
        text: str,
        doc_structure: DocumentStructure | None = None,
    ) -> list[Mention]:
        """
        Extract clinical finding mentions from text.

        Args:
            text: Document text
            doc_structure: Optional pre-computed document structure

        Returns:
            List of Mention objects
        """
        logger.debug(f"Extracting mentions from text of length {len(text)}")

        # Get or create document structure
        if doc_structure is None:
            doc_structure = self.structure_detector.analyze(text)

        # Process with spaCy
        doc = self.nlp(text)

        # Extract candidate spans
        candidates = self._extract_candidates(doc)

        # Convert to Mention objects with structure info
        mentions = self._create_mentions(candidates, text, doc_structure)

        # Filter invalid mentions
        mentions = self._filter_mentions(mentions)

        # Remove overlapping mentions (keep longer ones)
        mentions = self._deduplicate_mentions(mentions)

        logger.debug(f"Extracted {len(mentions)} mentions")
        return mentions

    def extract_with_structure(
        self,
        text: str,
        doc_id: str = "unknown",
    ) -> tuple[list[Mention], DocumentStructure]:
        """
        Extract mentions along with document structure.

        Args:
            text: Document text
            doc_id: Document identifier

        Returns:
            Tuple of (mentions, document_structure)
        """
        doc_structure = self.structure_detector.analyze(text, doc_id=doc_id)
        mentions = self.extract(text, doc_structure=doc_structure)
        return mentions, doc_structure

    def _extract_candidates(self, doc: Doc) -> list[Span]:
        """Extract candidate spans from spaCy doc."""
        candidates: list[Span] = []

        # Extract noun phrases
        if self.config.include_noun_phrases:
            for chunk in doc.noun_chunks:
                candidates.append(chunk)

        # Extract adjective-noun phrases
        if self.config.include_adj_noun_phrases:
            adj_noun_spans = self._extract_adj_noun_phrases(doc)
            candidates.extend(adj_noun_spans)

        # Extract verb phrases (for symptom descriptions)
        if self.config.include_verb_phrases:
            verb_spans = self._extract_verb_phrases(doc)
            candidates.extend(verb_spans)

        return candidates

    def _extract_adj_noun_phrases(self, doc: Doc) -> list[Span]:
        """Extract adjective-noun combinations not covered by noun_chunks."""
        spans: list[Span] = []

        for token in doc:
            # Look for adjectives modifying nouns
            if token.pos_ == "ADJ" and token.dep_ == "amod":
                head = token.head
                if head.pos_ in {"NOUN", "PROPN"}:
                    # Create span from adjective to noun
                    start = min(token.i, head.i)
                    end = max(token.i, head.i) + 1
                    span = doc[start:end]

                    # Check if this span is already covered
                    if not any(
                        existing.start <= span.start and span.end <= existing.end
                        for existing in spans
                    ):
                        spans.append(span)

        return spans

    def _extract_verb_phrases(self, doc: Doc) -> list[Span]:
        """Extract verb phrases that might describe symptoms."""
        spans: list[Span] = []

        for token in doc:
            # Look for verbs with clinical significance
            if token.pos_ == "VERB" and token.dep_ in {"ROOT", "advcl", "relcl"}:
                # Get the verb and its direct objects
                start = token.i
                end = token.i + 1

                for child in token.children:
                    if child.dep_ in {"dobj", "attr", "prep"}:
                        end = max(end, child.i + 1)
                        # Include prepositional objects
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                end = max(end, grandchild.i + 1)

                if end > start + 1:  # Only if we have more than just the verb
                    spans.append(doc[start:end])

        return spans

    def _create_mentions(
        self,
        candidates: list[Span],
        full_text: str,
        structure: DocumentStructure,
    ) -> list[Mention]:
        """Convert candidate spans to Mention objects."""
        mentions: list[Mention] = []

        for span in candidates:
            text = span.text.strip()
            if not text:
                continue

            start_char = span.start_char
            end_char = span.end_char

            # Get sentence index
            sentence = structure.get_sentence_at_position(start_char)
            sentence_idx = sentence.idx if sentence else 0

            # Get section type
            section_type = structure.get_section_at_position(start_char)

            # Get context window
            context_start = max(0, start_char - self.config.context_window_chars)
            context_end = min(
                len(full_text), end_char + self.config.context_window_chars
            )
            context_window = full_text[context_start:context_end]

            mention = Mention(
                text=text,
                start_char=start_char,
                end_char=end_char,
                sentence_idx=sentence_idx,
                section_type=section_type,
                context_window=context_window,
            )
            mentions.append(mention)

        return mentions

    def _filter_mentions(self, mentions: list[Mention]) -> list[Mention]:
        """Filter out invalid mentions."""
        filtered: list[Mention] = []

        for mention in mentions:
            # Length checks
            if len(mention.text) < self.config.min_mention_chars:
                continue
            if len(mention.text) > self.config.max_mention_chars:
                continue

            word_count = len(mention.text.split())
            if word_count < self.config.min_mention_words:
                continue
            if word_count > self.config.max_mention_words:
                continue

            # Stopword filter
            if self.config.filter_stopwords:
                if self._is_stopword_only(mention.text):
                    continue

            # Pronoun filter
            if self.config.filter_pronouns:
                if self._is_pronoun_only(mention.text):
                    continue

            filtered.append(mention)

        return filtered

    def _is_stopword_only(self, text: str) -> bool:
        """Check if text consists only of stopwords."""
        text_lower = text.lower().strip()
        for pattern in STOPWORD_PATTERNS:
            if re.match(pattern, text_lower):
                return True
        return False

    def _is_pronoun_only(self, text: str) -> bool:
        """Check if text is only pronouns."""
        text_lower = text.lower().strip()
        pronoun_pattern = r"^(?:he|she|they|we|i|you|it|his|her|their|our|my|your|its)$"
        return bool(re.match(pronoun_pattern, text_lower))

    def _deduplicate_mentions(self, mentions: list[Mention]) -> list[Mention]:
        """Remove overlapping mentions, preferring longer ones."""
        if not mentions:
            return mentions

        # Sort by length (descending) then by start position
        sorted_mentions = sorted(
            mentions,
            key=lambda m: (-m.span_length, m.start_char),
        )

        kept: list[Mention] = []
        for mention in sorted_mentions:
            # Check if this mention overlaps with any kept mention
            overlaps = False
            for kept_mention in kept:
                if mention.overlaps_with(kept_mention):
                    overlaps = True
                    break

            if not overlaps:
                kept.append(mention)

        # Re-sort by position
        kept.sort(key=lambda m: m.start_char)
        return kept


def extract_mentions(
    text: str,
    language: str = "en",
    config: MentionExtractionConfig | None = None,
) -> list[Mention]:
    """
    Convenience function to extract mentions from text.

    Args:
        text: Document text
        language: Language code
        config: Extraction configuration

    Returns:
        List of Mention objects
    """
    extractor = MentionExtractor(language=language, config=config)
    return extractor.extract(text)
