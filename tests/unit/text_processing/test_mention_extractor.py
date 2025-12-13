"""
Unit tests for mention extraction.

Tests the MentionExtractor and related functionality.
"""

import pytest

from phentrieve.text_processing.document_structure import DocumentStructure
from phentrieve.text_processing.mention import Mention
from phentrieve.text_processing.mention_extractor import (
    MentionExtractionConfig,
    MentionExtractor,
    extract_mentions,
)


class TestMentionExtractionConfig:
    """Tests for MentionExtractionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MentionExtractionConfig()

        assert config.min_mention_words == 1
        assert config.max_mention_words == 10
        assert config.include_noun_phrases is True
        assert config.filter_stopwords is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MentionExtractionConfig(
            min_mention_chars=5,
            max_mention_chars=50,
            context_window_chars=100,
        )

        assert config.min_mention_chars == 5
        assert config.max_mention_chars == 50
        assert config.context_window_chars == 100


class TestMentionExtractor:
    """Tests for MentionExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create an extractor for testing."""
        return MentionExtractor(language="en")

    def test_extract_single_mention(self, extractor):
        """Test extracting a single mention."""
        text = "Patient has seizures."
        mentions = extractor.extract(text)

        assert len(mentions) >= 1
        # Should find "seizures" or similar
        mention_texts = [m.text.lower() for m in mentions]
        assert any("seizure" in t for t in mention_texts)

    def test_extract_multiple_mentions(self, extractor):
        """Test extracting multiple mentions."""
        text = "Patient presents with seizures and headaches."
        mentions = extractor.extract(text)

        assert len(mentions) >= 2

    def test_mentions_have_spans(self, extractor):
        """Test that mentions have correct span information."""
        text = "Patient has seizures."
        mentions = extractor.extract(text)

        for mention in mentions:
            assert mention.start_char >= 0
            assert mention.end_char > mention.start_char
            assert mention.start_char < len(text)
            assert mention.end_char <= len(text)
            # The span should match the text
            assert text[mention.start_char : mention.end_char] == mention.text

    def test_mentions_have_sentence_idx(self, extractor):
        """Test that mentions have sentence indices."""
        text = "First sentence here. Second sentence with seizures."
        mentions = extractor.extract(text)

        for mention in mentions:
            assert mention.sentence_idx >= 0

    def test_filter_stopwords(self, extractor):
        """Test that stopwords are filtered."""
        text = "The patient has the seizures."
        mentions = extractor.extract(text)

        # "the" should not appear as a standalone mention
        mention_texts = [m.text.lower().strip() for m in mentions]
        assert "the" not in mention_texts

    def test_filter_pronouns(self, extractor):
        """Test that pronouns are filtered."""
        text = "He has seizures. She has headaches."
        mentions = extractor.extract(text)

        mention_texts = [m.text.lower().strip() for m in mentions]
        assert "he" not in mention_texts
        assert "she" not in mention_texts

    def test_context_window(self, extractor):
        """Test that context window is populated."""
        text = "The patient has severe recurring seizures in the morning."
        mentions = extractor.extract(text)

        for mention in mentions:
            # Context window should be populated
            assert mention.context_window is not None or mention.span_length <= 10

    def test_with_document_structure(self, extractor):
        """Test extraction with pre-computed document structure."""
        text = "Patient has seizures. Family history: mother has epilepsy."
        structure = DocumentStructure.from_text(text, language="en")

        mentions = extractor.extract(text, doc_structure=structure)

        assert len(mentions) >= 2

    def test_extract_with_structure_returns_both(self, extractor):
        """Test extract_with_structure returns mentions and structure."""
        text = "Patient presents with fever and seizures."
        mentions, structure = extractor.extract_with_structure(text, doc_id="test")

        assert len(mentions) >= 1
        assert structure.doc_id == "test"

    def test_deduplicate_overlapping(self, extractor):
        """Test that overlapping mentions are deduplicated."""
        text = "Patient has severe seizures."
        mentions = extractor.extract(text)

        # No two mentions should overlap
        for i, m1 in enumerate(mentions):
            for m2 in mentions[i + 1 :]:
                assert not m1.overlaps_with(m2)

    def test_min_length_filter(self):
        """Test minimum length filtering."""
        config = MentionExtractionConfig(min_mention_chars=5)
        extractor = MentionExtractor(language="en", config=config)

        text = "I am OK with seizures."
        mentions = extractor.extract(text)

        # Short mentions like "I", "am", "OK" should be filtered
        for mention in mentions:
            assert len(mention.text) >= 5


class TestExtractMentionsFunction:
    """Tests for extract_mentions convenience function."""

    def test_basic_usage(self):
        """Test basic usage of convenience function."""
        text = "Patient has seizures and headaches."
        mentions = extract_mentions(text, language="en")

        assert len(mentions) >= 2
        assert all(isinstance(m, Mention) for m in mentions)

    def test_with_config(self):
        """Test with custom configuration."""
        config = MentionExtractionConfig(
            min_mention_chars=3,
            include_noun_phrases=True,
        )
        text = "Patient has seizures."
        mentions = extract_mentions(text, language="en", config=config)

        assert len(mentions) >= 1
