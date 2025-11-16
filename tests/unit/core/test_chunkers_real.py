"""Real unit tests for chunkers (execute actual code for coverage).

These tests use real chunker instances with minimal mocking to achieve
actual code coverage. Unlike the existing tests which mock everything,
these tests execute the real chunking logic.
"""

from unittest.mock import Mock

import pytest

from phentrieve.text_processing.chunkers import (
    ConjunctionChunker,
    FinalChunkCleaner,
    FineGrainedPunctuationChunker,
    NoOpChunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowSemanticSplitter,
)

pytestmark = pytest.mark.unit


class TestParagraphChunkerReal:
    """Real unit tests for ParagraphChunker (actual code execution)."""

    def test_chunk_empty_list(self):
        """Test chunking empty list."""
        chunker = ParagraphChunker()
        result = chunker.chunk([])
        assert result == []
        assert isinstance(result, list)

    def test_chunk_single_paragraph(self):
        """Test chunking single paragraph."""
        chunker = ParagraphChunker()
        text = "This is a single paragraph with no breaks."
        result = chunker.chunk([text])
        assert len(result) == 1
        assert result[0] == text

    def test_chunk_multiple_paragraphs(self):
        """Test chunking multiple paragraphs."""
        chunker = ParagraphChunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = chunker.chunk([text])
        assert len(result) == 3


class TestSentenceChunkerReal:
    """Real unit tests for SentenceChunker (actual code execution)."""

    def test_chunk_empty_list(self):
        """Test chunking empty list."""
        chunker = SentenceChunker(language="en")
        assert chunker.chunk([]) == []

    def test_chunk_single_sentence(self):
        """Test chunking single sentence."""
        chunker = SentenceChunker(language="en")
        text = "Patient has fever."
        result = chunker.chunk([text])
        assert len(result) == 1

    def test_chunk_multiple_sentences(self):
        """Test chunking multiple sentences."""
        chunker = SentenceChunker(language="en")
        text = "First sentence. Second sentence. Third sentence."
        result = chunker.chunk([text])
        assert len(result) == 3


class TestConjunctionChunkerReal:
    """Real unit tests for ConjunctionChunker."""

    def test_chunk_empty_list(self):
        """Test chunking empty list."""
        chunker = ConjunctionChunker(language="en")
        assert chunker.chunk([]) == []

    def test_chunk_and_conjunction(self):
        """Test splitting at 'and' conjunction."""
        chunker = ConjunctionChunker(language="en")
        text = "Patient has fever and patient has chills"
        result = chunker.chunk([text])
        assert len(result) == 2


class TestFinalChunkCleanerReal:
    """Real unit tests for FinalChunkCleaner."""

    def test_chunk_empty_list(self):
        """Test cleaning empty list."""
        cleaner = FinalChunkCleaner(language="en")
        assert cleaner.chunk([]) == []

    def test_chunk_removes_leading_article(self):
        """Test removal of leading articles."""
        cleaner = FinalChunkCleaner(
            language="en",
            min_cleaned_chunk_length_chars=1,
            filter_short_low_value_chunks_max_words=1,
        )
        text = "The patient has fever"
        result = cleaner.chunk([text])
        assert len(result) == 1
        assert result[0] == "patient has fever"


class TestNoOpChunkerReal:
    """Real unit tests for NoOpChunker."""

    def test_chunk_returns_input_unchanged(self):
        """Test that input is returned unchanged."""
        chunker = NoOpChunker()
        inputs = [
            [],
            ["single"],
            ["multiple", "items"],
        ]
        for inp in inputs:
            result = chunker.chunk(inp)
            assert result == inp


class TestFineGrainedPunctuationChunkerReal:
    """Real unit tests for FineGrainedPunctuationChunker."""

    def test_chunk_empty_list(self):
        """Test chunking empty list."""
        chunker = FineGrainedPunctuationChunker(language="en")
        assert chunker.chunk([]) == []

    def test_chunk_basic_punctuation_split(self):
        """Test basic splitting at punctuation."""
        chunker = FineGrainedPunctuationChunker(language="en")
        text = "First part. Second part. Third part."
        result = chunker.chunk([text])
        assert len(result) >= 3

    def test_chunk_preserves_abbreviations(self):
        """Test that common abbreviations are preserved."""
        chunker = FineGrainedPunctuationChunker(language="en")
        text = "Dr. Smith examined the patient."
        result = chunker.chunk([text])
        # "Dr." should not cause a split
        assert len(result) == 1
        assert "Dr." in result[0] or "Dr" in result[0]

    def test_chunk_preserves_decimal_numbers(self):
        """Test that decimal numbers are preserved."""
        chunker = FineGrainedPunctuationChunker(language="en")
        text = "Temperature was 37.5 degrees."
        result = chunker.chunk([text])
        # The decimal should be preserved
        has_decimal = any("37" in chunk and "5" in chunk for chunk in result)
        assert has_decimal

    def test_chunk_comma_separation(self):
        """Test splitting at commas."""
        chunker = FineGrainedPunctuationChunker(language="en")
        text = "Patient has fever, chills, and headache."
        result = chunker.chunk([text])
        # Should split at commas
        assert len(result) >= 2

    def test_chunk_semicolon_separation(self):
        """Test splitting at semicolons."""
        chunker = FineGrainedPunctuationChunker(language="en")
        text = "First symptom; second symptom; third symptom."
        result = chunker.chunk([text])
        # Should split at semicolons
        assert len(result) >= 3

    def test_chunk_empty_segments_skipped(self):
        """Test that empty segments are skipped."""
        chunker = FineGrainedPunctuationChunker(language="en")
        result = chunker.chunk(["", "  ", "\n"])
        # Empty and whitespace-only segments should be skipped
        assert result == []

    def test_chunk_multiple_input_segments(self):
        """Test processing multiple input segments."""
        chunker = FineGrainedPunctuationChunker(language="en")
        segments = ["First text. Second text.", "Third text. Fourth text."]
        result = chunker.chunk(segments)
        # Should process all segments
        assert len(result) >= 4


class TestSlidingWindowSemanticSplitterReal:
    """Real unit tests for SlidingWindowSemanticSplitter (with mocked model)."""

    def test_init_with_model(self):
        """Test initialization with a model."""
        mock_model = Mock()
        splitter = SlidingWindowSemanticSplitter(
            language="en",
            model=mock_model,
            window_size_tokens=7,
            step_size_tokens=1,
            splitting_threshold=0.5,
        )
        assert splitter.model == mock_model
        assert splitter.window_size_tokens == 7
        assert splitter.step_size_tokens == 1
        assert splitter.splitting_threshold == 0.5

    def test_init_without_model_raises_error(self):
        """Test that initialization without a model raises ValueError."""
        with pytest.raises(ValueError, match="SentenceTransformer model is required"):
            SlidingWindowSemanticSplitter(language="en", model=None)

    def test_init_step_size_minimum_one(self):
        """Test that step size is enforced to be at least 1."""
        mock_model = Mock()
        splitter = SlidingWindowSemanticSplitter(
            language="en",
            model=mock_model,
            step_size_tokens=0,  # Try to set to 0
        )
        assert splitter.step_size_tokens == 1  # Should be enforced to 1
