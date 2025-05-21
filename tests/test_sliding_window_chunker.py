"""
Tests for the sliding window semantic chunker implementation.

This module tests the SlidingWindowSemanticSplitter functionality,
checking its ability to properly split text at semantic boundaries.
"""

import unittest
import logging
import string

from sentence_transformers import SentenceTransformer
from phentrieve.config import get_sliding_window_config_with_params
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.text_processing.chunkers import SlidingWindowSemanticSplitter


class TestSlidingWindowSplitter(unittest.TestCase):
    """Test cases for the SlidingWindowSemanticSplitter class directly."""

    def setUp(self):
        """Initialize resources needed for tests."""
        # Use a small model for testing
        self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        # Default settings for testing
        self.chunker = SlidingWindowSemanticSplitter(
            model=self.model,
            window_size_tokens=5,  # Smaller for quicker tests
            step_size_tokens=1,
            splitting_threshold=0.7,  # Higher threshold for splits
            min_split_segment_length_words=2,  # Small for test purposes
        )

    def test_empty_input(self):
        """Test that empty input produces empty output."""
        result = self.chunker.chunk([])
        self.assertEqual(result, [])

        # Empty string in list
        result = self.chunker.chunk([""])
        self.assertEqual(result, [])

        # Whitespace only
        result = self.chunker.chunk(["   "])
        self.assertEqual(result, [])

    def test_short_segment(self):
        """Test that segments shorter than window_size_tokens are returned as-is."""
        short_text = "Just a few words"  # 4 tokens, less than window_size
        result = self.chunker.chunk([short_text])
        self.assertEqual(result, [short_text])

    def test_semantic_splitting(self):
        """Test that text with distinct topics is split appropriately."""
        mixed_topic_text = (
            "Python is a popular programming language used for web development,"
            " data science, and AI. The Sahara desert is the largest hot desert"
            " in the world spanning 11 countries in North Africa."
        )

        result = self.chunker.chunk([mixed_topic_text])

        # We should get at least 2 chunks since there are two distinct topics
        self.assertGreater(len(result), 1)

        # Check that the content is preserved (all original words present in output)
        combined_result = " ".join(result)

        # Strip punctuation from words for comparison
        translator = str.maketrans("", "", string.punctuation)

        # Remove punctuation and convert to lowercase for comparison
        result_text_clean = combined_result.lower().translate(translator)
        result_words = set(result_text_clean.split())

        # All key words from the original should appear in the result
        for key_word in ["python", "programming", "sahara", "desert", "africa"]:
            self.assertIn(key_word, result_words)

    def test_multiple_segments(self):
        """Test that multiple input segments are processed correctly."""
        segment1 = "Dogs are mammals with four legs and a tail."
        segment2 = "A novel is a long, fictional narrative."

        result = self.chunker.chunk([segment1, segment2])

        # All original key words should be preserved in the output
        combined_result = " ".join(result)
        for key_word in ["dogs", "mammals", "novel", "fictional", "narrative"]:
            self.assertIn(key_word.lower(), combined_result.lower())


class TestSlidingWindowChunker(unittest.TestCase):
    """Test cases for the sliding window chunker functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used across all tests."""
        # Suppress verbose logging during tests
        logging.basicConfig(level=logging.ERROR)

        # Load the model once for all tests
        cls.model = SentenceTransformer("FremyCompany/BioLORD-2023-M")

    def test_german_clinical_note_chunking(self):
        """Test chunking of a German clinical note."""
        # Sample German clinical text
        text = """
        Trinkschwäche und Hypotonie. Fehlende Fixation und kein Blickkontakt.
        MRT: Kortikale Atrophie und Kleinhirnhypoplasie.
        Mikrozephalie und Krampfanfälle in der Vorgeschichte.
        """

        # Create a pipeline with sliding window chunking
        config = get_sliding_window_config_with_params(
            window_size=2, step_size=1, threshold=0.6, min_segment_length=1
        )

        pipeline = TextProcessingPipeline(
            language="de",
            chunking_pipeline_config=config,
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=self.model,
        )

        # Process the text
        chunks = pipeline.process(text)

        # Verify we get an appropriate number of chunks
        self.assertGreater(len(chunks), 1, "Should split into multiple chunks")

        # Verify chunks contain meaningful content
        for chunk in chunks:
            self.assertIsInstance(chunk, dict, "Chunks should be dictionary objects")
            self.assertIn("text", chunk, "Chunks should contain 'text' key")
            self.assertGreater(len(chunk["text"]), 0, "Chunk text should not be empty")

    def test_sliding_window_parameters(self):
        """Test the effect of different parameters on chunking behavior."""
        text = """
        Patient presents with severe headache, nausea, and vomiting.
        Symptoms started three days ago and have progressively worsened.
        Patient has a history of migraines but describes this as different.
        Neurological examination reveals no focal deficits.
        """

        # Test with different window sizes
        configs = [
            get_sliding_window_config_with_params(
                window_size=2, step_size=1, threshold=0.5
            ),
            get_sliding_window_config_with_params(
                window_size=3, step_size=1, threshold=0.5
            ),
            get_sliding_window_config_with_params(
                window_size=2, step_size=1, threshold=0.7
            ),
        ]

        chunk_counts = []
        for config in configs:
            pipeline = TextProcessingPipeline(
                language="en",
                chunking_pipeline_config=config,
                assertion_config={"disable": True},
                sbert_model_for_semantic_chunking=self.model,
            )
            chunks = pipeline.process(text)
            chunk_counts.append(len(chunks))

        # Verify that parameters affect chunking
        self.assertNotEqual(
            chunk_counts[0],
            chunk_counts[1],
            "Different window sizes should produce different chunk counts",
        )
        self.assertNotEqual(
            chunk_counts[0],
            chunk_counts[2],
            "Different thresholds should produce different chunk counts",
        )


if __name__ == "__main__":
    unittest.main()
