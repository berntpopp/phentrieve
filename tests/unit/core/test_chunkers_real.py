"""Real unit tests for chunkers (execute actual code for coverage).

These tests use real chunker instances with minimal mocking to achieve
actual code coverage. Unlike the existing tests which mock everything,
these tests execute the real chunking logic.
"""

import pytest

from phentrieve.text_processing.chunkers import (
    ConjunctionChunker,
    FinalChunkCleaner,
    NoOpChunker,
    ParagraphChunker,
    SentenceChunker,
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
