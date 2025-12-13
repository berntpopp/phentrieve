"""
Unit tests for document structure detection.

Tests the DocumentStructureDetector and related functionality.
"""

import pytest

from phentrieve.text_processing.document_structure import (
    DocumentStructure,
    DocumentStructureDetector,
    SectionSpan,
    SentenceSpan,
    detect_family_history_spans,
)


class TestSentenceSpan:
    """Tests for SentenceSpan dataclass."""

    def test_basic_creation(self):
        """Test basic SentenceSpan creation."""
        span = SentenceSpan(
            idx=0,
            text="Patient presents with seizures.",
            start_char=0,
            end_char=31,
        )
        assert span.idx == 0
        assert span.text == "Patient presents with seizures."
        assert len(span) == 31

    def test_section_type_assignment(self):
        """Test section type can be assigned."""
        span = SentenceSpan(
            idx=0,
            text="Mother has epilepsy.",
            start_char=0,
            end_char=20,
            section_type="family_history",
        )
        assert span.section_type == "family_history"


class TestSectionSpan:
    """Tests for SectionSpan dataclass."""

    def test_basic_creation(self):
        """Test basic SectionSpan creation."""
        section = SectionSpan(
            section_type="family_history",
            header_text="Family History:",
            start_char=100,
        )
        assert section.section_type == "family_history"
        assert section.end_char == -1  # Not yet terminated


class TestDocumentStructure:
    """Tests for DocumentStructure dataclass."""

    def test_basic_creation(self):
        """Test basic DocumentStructure creation."""
        structure = DocumentStructure(
            doc_id="case_001",
            full_text="Patient presents with seizures.",
        )
        assert structure.doc_id == "case_001"
        assert structure.num_sentences == 0
        assert structure.num_sections == 0

    def test_get_section_at_position(self):
        """Test getting section at a character position."""
        structure = DocumentStructure(
            doc_id="test",
            full_text="Current findings: seizures. Family history: epilepsy.",
        )
        structure.sections = [
            SectionSpan(
                section_type="current_findings",
                header_text="Current findings:",
                start_char=0,
                end_char=27,
            ),
            SectionSpan(
                section_type="family_history",
                header_text="Family history:",
                start_char=28,
                end_char=-1,
            ),
        ]

        assert structure.get_section_at_position(10) == "current_findings"
        assert structure.get_section_at_position(35) == "family_history"

    def test_is_in_family_history(self):
        """Test family history position detection."""
        structure = DocumentStructure(
            doc_id="test",
            full_text="Findings: seizures. Family history: epilepsy.",
        )
        structure.sections = [
            SectionSpan(
                section_type="family_history",
                header_text="Family history:",
                start_char=20,
            ),
        ]

        assert structure.is_in_family_history(5) is False
        assert structure.is_in_family_history(25) is True

    def test_from_text_factory(self):
        """Test creating structure from text."""
        text = "Patient presents with seizures. No headaches noted."
        structure = DocumentStructure.from_text(
            text,
            doc_id="case_001",
            language="en",
        )

        assert structure.doc_id == "case_001"
        assert structure.num_sentences >= 1  # At least one sentence


class TestDocumentStructureDetector:
    """Tests for DocumentStructureDetector."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return DocumentStructureDetector(language="en")

    def test_analyze_simple_text(self, detector):
        """Test analyzing simple text."""
        text = "Patient presents with seizures. No headaches observed."
        structure = detector.analyze(text, doc_id="test")

        assert structure.num_sentences >= 1
        assert structure.full_text == text

    def test_detect_family_history_section(self, detector):
        """Test detection of family history section."""
        text = """Current findings: The patient has seizures.

Family history: Mother has epilepsy. Father has migraines."""

        structure = detector.analyze(text)

        # Should detect family history section
        fh_sections = [
            s for s in structure.sections if s.section_type == "family_history"
        ]
        assert len(fh_sections) > 0

    def test_sentence_segmentation(self, detector):
        """Test sentence segmentation."""
        text = "First sentence. Second sentence. Third sentence."
        structure = detector.analyze(text)

        assert structure.num_sentences == 3

    def test_assign_section_to_sentences(self, detector):
        """Test that sentences are assigned to sections."""
        text = """Family history:
Mother has epilepsy.
Father has diabetes."""

        structure = detector.analyze(text)

        # Sentences after "Family history:" should be in that section
        fh_sentences = structure.get_sentences_in_section("family_history")
        assert len(fh_sentences) >= 1


class TestDetectFamilyHistorySpans:
    """Tests for detect_family_history_spans function."""

    def test_inline_family_history(self):
        """Test detecting inline family history mentions."""
        text = "Patient has seizures. Family history of epilepsy in mother."
        spans = detect_family_history_spans(text)

        assert len(spans) >= 1
        # The span should cover the family history mention
        start, end = spans[0]
        assert "family history" in text[start:end].lower()

    def test_relative_mention(self):
        """Test detecting relative mentions."""
        text = "Patient presents with fever. Mother has epilepsy."
        spans = detect_family_history_spans(text)

        assert len(spans) >= 1

    def test_no_family_history(self):
        """Test text without family history."""
        text = "Patient presents with seizures and headaches."
        spans = detect_family_history_spans(text)

        assert len(spans) == 0

    def test_multiple_family_mentions(self):
        """Test multiple family history mentions."""
        text = "Mother has epilepsy. Father has diabetes. Uncle had heart disease."
        spans = detect_family_history_spans(text)

        # Should detect multiple spans or merge adjacent ones
        assert len(spans) >= 1

    def test_german_family_history(self):
        """Test German family history detection."""
        text = "Patient mit Krampfanfällen. Familienanamnese: Mutter mit Epilepsie."
        spans = detect_family_history_spans(text, language="de")

        assert len(spans) >= 1
