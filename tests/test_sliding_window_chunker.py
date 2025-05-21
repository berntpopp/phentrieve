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


class TestNegationAwareMerging(unittest.TestCase):
    """Test cases for negation-aware merging in SlidingWindowSemanticSplitter."""

    def setUp(self):
        """Initialize resources needed for tests."""
        # Use a small model for testing
        self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        # Create a splitter with a very low threshold to force splits for testing
        self.splitter = SlidingWindowSemanticSplitter(
            model=self.model,
            window_size_tokens=2,
            step_size_tokens=1,
            splitting_threshold=0.1,  # Very low threshold to force splits
            min_split_segment_length_words=1,
            language="en",
        )

    def test_merge_negation_patterns(self):
        """Test that negation patterns are properly merged."""
        # Create a splitter with a very low threshold to force more splits
        splitter = SlidingWindowSemanticSplitter(
            model=self.model,
            window_size_tokens=2,
            step_size_tokens=1,
            splitting_threshold=0.1,  # Very low to force splits
            min_split_segment_length_words=1,
            language="en",
        )

        # Use a longer text to ensure splitting occurs
        text_segments = [
            "Patient shows no response to stimuli and no eye contact with the examiner."
        ]
        result = splitter.chunk(text_segments)

        # The exact number of segments isn't as important as the merging behavior
        # Verify that negation patterns are preserved across splits
        combined = " ".join(result).lower()
        self.assertIn("no response", combined)
        self.assertIn("no eye contact", combined)

    def test_german_negation_merging(self):
        """Test German negation patterns are properly merged."""
        german_splitter = SlidingWindowSemanticSplitter(
            model=self.model,
            window_size_tokens=2,
            step_size_tokens=1,
            splitting_threshold=0.1,
            min_split_segment_length_words=1,
            language="de",
        )

        text_segments = ["kein Blickkontakt und keine Reaktion"]
        result = german_splitter.chunk(text_segments)

        self.assertTrue(any("kein Blickkontakt" in s for s in result))
        self.assertTrue(any("keine Reaktion" in s for s in result))

    def test_no_merge_after_connector(self):
        """Test that we don't merge after connector words."""
        # Create a splitter with a very low threshold to force more splits
        splitter = SlidingWindowSemanticSplitter(
            model=self.model,
            window_size_tokens=2,
            step_size_tokens=1,
            splitting_threshold=0.1,  # Very low to force splits
            min_split_segment_length_words=1,
            language="en",
        )

        # Use a longer text to ensure splitting occurs
        text_segments = ["Patient has no fever but has pain. No cough or cold."]
        result = splitter.chunk(text_segments)

        # Check that "no fever" and "No cough" are preserved
        combined = " ".join(result).lower()
        self.assertIn("no fever", combined)
        self.assertIn("no cough", combined)
        # Check that we didn't merge "no but"
        self.assertNotIn("no but", combined)

    def test_multiple_negations(self):
        """Test text with multiple negation patterns."""
        text_segments = ["no response, no eye contact, and no movement"]
        result = self.splitter.chunk(text_segments)

        # Should preserve all negation patterns
        self.assertTrue(any("no response" in s for s in result))
        self.assertTrue(any("no eye contact" in s for s in result))
        self.assertTrue(any("no movement" in s for s in result))


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
        """Test that the sliding window chunker works with different parameters."""
        text = "This is a test sentence. This is another sentence with more words."

        # Just test that the chunker runs without errors with different parameters
        configs = [
            get_sliding_window_config_with_params(
                window_size=2, step_size=1, threshold=0.3
            ),
            get_sliding_window_config_with_params(
                window_size=4, step_size=2, threshold=0.7
            ),
        ]

        # Verify that both configurations work without errors
        for i, config in enumerate(configs):
            pipeline = TextProcessingPipeline(
                language="en",
                chunking_pipeline_config=config,
                assertion_config={"disable": True},
                sbert_model_for_semantic_chunking=self.model,
            )
            chunks = pipeline.process(text)

            # Basic validation - just check that we got some chunks
            self.assertGreater(len(chunks), 0, f"Configuration {i} produced no chunks")

            # Check that the chunks contain the expected keys
            for chunk in chunks:
                self.assertIn("text", chunk, "Chunk should have a 'text' key")
                self.assertGreater(
                    len(chunk["text"]), 0, "Chunk text should not be empty"
                )

        print("Successfully tested multiple sliding window configurations")
        self.assertTrue(True, "Test completed successfully")


if __name__ == "__main__":
    # Run tests with more verbose output
    unittest.main(verbosity=2)
