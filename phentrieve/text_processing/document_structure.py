"""
Document structure detection for mention-level HPO extraction.

This module provides lightweight document structure analysis including
sentence segmentation and section detection. The structure is used for
context gating (e.g., separating family history from current findings)
during mention-level processing.

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spacy.language import Language

logger = logging.getLogger(__name__)

# Regex patterns for section header detection
SECTION_HEADER_PATTERNS: dict[str, list[str]] = {
    "family_history": [
        r"(?i)^#+\s*family\s+history",
        r"(?i)^family\s+history\s*[:;]",
        r"(?i)^\*{0,2}family\s+history\*{0,2}\s*[:;]?",
        r"(?i)familienanamnese\s*[:;]?",  # German
    ],
    "past_medical_history": [
        r"(?i)^#+\s*past\s+medical\s+history",
        r"(?i)^past\s+medical\s+history\s*[:;]",
        r"(?i)^pmh\s*[:;]",
        r"(?i)eigenanamnese\s*[:;]?",  # German
    ],
    "current_findings": [
        r"(?i)^#+\s*(?:present|current)\s+(?:illness|findings|symptoms)",
        r"(?i)^(?:present|current)\s+(?:illness|findings|symptoms)\s*[:;]",
        r"(?i)^hpi\s*[:;]",
        r"(?i)aktuelle\s+beschwerden\s*[:;]?",  # German
    ],
    "physical_examination": [
        r"(?i)^#+\s*physical\s+exam(?:ination)?",
        r"(?i)^physical\s+exam(?:ination)?\s*[:;]",
        r"(?i)^pe\s*[:;]",
        r"(?i)körperliche\s+untersuchung\s*[:;]?",  # German
    ],
    "assessment": [
        r"(?i)^#+\s*assessment",
        r"(?i)^assessment\s*[:;]",
        r"(?i)^impression\s*[:;]",
        r"(?i)beurteilung\s*[:;]?",  # German
    ],
    "plan": [
        r"(?i)^#+\s*plan",
        r"(?i)^plan\s*[:;]",
        r"(?i)^treatment\s+plan\s*[:;]",
    ],
}

# Section-ending patterns
SECTION_END_PATTERNS: list[str] = [
    r"^#+\s+",  # Markdown headers
    r"^\*{2,}\s*$",  # Separator lines
    r"^-{3,}\s*$",  # Separator lines
    r"^={3,}\s*$",  # Separator lines
]


@dataclass
class SentenceSpan:
    """
    A sentence span in the document.

    Attributes:
        idx: Sentence index (0-based)
        text: Sentence text
        start_char: Start character position
        end_char: End character position
        section_type: Section this sentence belongs to (if any)
    """

    idx: int
    text: str
    start_char: int
    end_char: int
    section_type: str | None = None

    def __len__(self) -> int:
        """Length of sentence in characters."""
        return self.end_char - self.start_char


@dataclass
class SectionSpan:
    """
    A section span in the document.

    Attributes:
        section_type: Type of section (e.g., "family_history")
        header_text: The header text that identified this section
        start_char: Start character position
        end_char: End character position (may be -1 if extends to document end)
        sentences: Indices of sentences in this section
    """

    section_type: str
    header_text: str
    start_char: int
    end_char: int = -1  # -1 means extends to document end or next section
    sentences: list[int] = field(default_factory=list)


@dataclass
class DocumentStructure:
    """
    Structural analysis of a document.

    Contains sentence and section boundaries for context gating
    during mention-level processing.

    Attributes:
        doc_id: Document identifier
        full_text: Original document text
        sentences: List of sentence spans
        sections: List of section spans
        metadata: Additional metadata

    Example:
        >>> structure = DocumentStructure.from_text(
        ...     "Patient presents with seizures. Family history: mother has epilepsy.",
        ...     doc_id="case_001",
        ... )
        >>> len(structure.sentences)
        2
    """

    doc_id: str
    full_text: str
    sentences: list[SentenceSpan] = field(default_factory=list)
    sections: list[SectionSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_sentences(self) -> int:
        """Number of sentences."""
        return len(self.sentences)

    @property
    def num_sections(self) -> int:
        """Number of detected sections."""
        return len(self.sections)

    def get_sentence(self, idx: int) -> SentenceSpan | None:
        """Get sentence by index."""
        if 0 <= idx < len(self.sentences):
            return self.sentences[idx]
        return None

    def get_section_at_position(self, char_pos: int) -> str | None:
        """
        Get the section type at a character position.

        Args:
            char_pos: Character position in document

        Returns:
            Section type string or None if not in a detected section
        """
        for section in self.sections:
            if section.start_char <= char_pos:
                if section.end_char == -1 or char_pos < section.end_char:
                    return section.section_type
        return None

    def get_sentence_at_position(self, char_pos: int) -> SentenceSpan | None:
        """
        Get the sentence containing a character position.

        Args:
            char_pos: Character position in document

        Returns:
            SentenceSpan or None if not found
        """
        for sentence in self.sentences:
            if sentence.start_char <= char_pos < sentence.end_char:
                return sentence
        return None

    def get_sentences_in_section(self, section_type: str) -> list[SentenceSpan]:
        """Get all sentences in a specific section type."""
        return [s for s in self.sentences if s.section_type == section_type]

    def is_in_family_history(self, char_pos: int) -> bool:
        """Check if a position is within a family history section."""
        section = self.get_section_at_position(char_pos)
        return section == "family_history"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "num_sentences": self.num_sentences,
            "num_sections": self.num_sections,
            "sentences": [
                {
                    "idx": s.idx,
                    "text": s.text,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                    "section_type": s.section_type,
                }
                for s in self.sentences
            ],
            "sections": [
                {
                    "section_type": sec.section_type,
                    "header_text": sec.header_text,
                    "start_char": sec.start_char,
                    "end_char": sec.end_char,
                    "sentences": sec.sentences,
                }
                for sec in self.sections
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_text(
        cls,
        text: str,
        doc_id: str = "unknown",
        language: str = "en",
        nlp: Language | None = None,
    ) -> DocumentStructure:
        """
        Create DocumentStructure by analyzing text.

        Args:
            text: Document text
            doc_id: Document identifier
            language: Language code
            nlp: Optional pre-loaded spaCy model

        Returns:
            DocumentStructure instance
        """
        detector = DocumentStructureDetector(language=language, nlp=nlp)
        return detector.analyze(text, doc_id=doc_id)


class DocumentStructureDetector:
    """
    Detector for document structure including sentences and sections.

    Uses spaCy for sentence segmentation and regex patterns for
    section header detection.

    Attributes:
        language: Language code for spaCy model
        nlp: Loaded spaCy model (lazy-loaded)
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
        nlp: Language | None = None,
    ):
        """
        Initialize the detector.

        Args:
            language: Language code
            nlp: Optional pre-loaded spaCy model
        """
        self.language = language
        self._nlp = nlp

    @property
    def nlp(self) -> Language:
        """Get the spaCy model, loading if necessary."""
        if self._nlp is None:
            self._nlp = self._load_nlp()
        return self._nlp

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

    def analyze(self, text: str, doc_id: str = "unknown") -> DocumentStructure:
        """
        Analyze document text and extract structure.

        Args:
            text: Document text
            doc_id: Document identifier

        Returns:
            DocumentStructure with sentences and sections
        """
        logger.debug(f"Analyzing document structure for {doc_id}")

        # First detect sections (before sentence segmentation)
        sections = self._detect_sections(text)

        # Segment sentences
        sentences = self._segment_sentences(text)

        # Assign section types to sentences
        self._assign_sections_to_sentences(sentences, sections)

        # Update section sentence lists
        for section in sections:
            section.sentences = [
                s.idx
                for s in sentences
                if s.section_type == section.section_type
                and section.start_char <= s.start_char
            ]

        structure = DocumentStructure(
            doc_id=doc_id,
            full_text=text,
            sentences=sentences,
            sections=sections,
        )

        logger.debug(
            f"Document structure: {structure.num_sentences} sentences, "
            f"{structure.num_sections} sections"
        )

        return structure

    def _segment_sentences(self, text: str) -> list[SentenceSpan]:
        """Segment text into sentences using spaCy."""
        doc = self.nlp(text)
        sentences: list[SentenceSpan] = []

        for idx, sent in enumerate(doc.sents):
            sentences.append(
                SentenceSpan(
                    idx=idx,
                    text=sent.text.strip(),
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                )
            )

        return sentences

    def _detect_sections(self, text: str) -> list[SectionSpan]:
        """Detect section headers in the text."""
        sections: list[SectionSpan] = []
        lines = text.split("\n")
        char_offset = 0

        for line in lines:
            line_start = char_offset
            line_end = char_offset + len(line)

            # Check for section headers
            for section_type, patterns in SECTION_HEADER_PATTERNS.items():
                for pattern in patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        # Close previous section if exists
                        if sections and sections[-1].end_char == -1:
                            sections[-1].end_char = line_start

                        sections.append(
                            SectionSpan(
                                section_type=section_type,
                                header_text=line.strip(),
                                start_char=line_start,
                            )
                        )
                        break
                else:
                    continue
                break

            char_offset = line_end + 1  # +1 for newline

        return sections

    def _assign_sections_to_sentences(
        self,
        sentences: list[SentenceSpan],
        sections: list[SectionSpan],
    ) -> None:
        """Assign section types to sentences based on position."""
        for sentence in sentences:
            for section in sections:
                if section.start_char <= sentence.start_char:
                    if section.end_char == -1 or sentence.end_char <= section.end_char:
                        sentence.section_type = section.section_type


def detect_family_history_spans(
    text: str,
    language: str = "en",
) -> list[tuple[int, int]]:
    """
    Detect character spans that are part of family history context.

    This is a convenience function for quick family history detection
    without full document structure analysis.

    Args:
        text: Document text
        language: Language code

    Returns:
        List of (start_char, end_char) tuples for family history regions
    """
    spans: list[tuple[int, int]] = []

    # In-line family history patterns
    inline_patterns = [
        r"(?i)family\s+history\s+(?:of|is\s+significant\s+for|includes?|positive\s+for)[:\s]+([^.!?\n]+[.!?\n]?)",
        r"(?i)(?:mother|father|sibling|brother|sister|parent|grandparent|uncle|aunt|cousin)\s+(?:has|had|with|diagnosed\s+with)\s+[^.!?\n]+[.!?\n]?",
        r"(?i)familienanamnese[:\s]+[^.!?\n]+[.!?\n]?",  # German
    ]

    for pattern in inline_patterns:
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end()))

    # Merge overlapping spans
    if spans:
        spans.sort()
        merged: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        spans = merged

    return spans
