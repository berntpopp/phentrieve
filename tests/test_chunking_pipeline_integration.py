"""
Tests for the chunking pipeline integration, especially for the new conjunction chunking strategy.

This module tests the integration of different chunkers in the text processing pipeline,
with particular focus on the sliding_window_punct_conj_cleaned strategy.
"""

import unittest

from sentence_transformers import SentenceTransformer

from phentrieve.config import (
    get_sliding_window_punct_cleaned_config,
    get_sliding_window_punct_conj_cleaned_config,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline


class TestChunkingPipelineIntegration(unittest.TestCase):
    """Test cases for chunking pipeline integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used across all tests."""
        # Load the model once for all tests
        cls.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    def test_sliding_window_punct_cleaned_strategy(self):
        """Test the sliding_window_punct_cleaned strategy."""
        # Sample text with punctuation
        text = """The patient reports fever, headache, and fatigue.
        No cough or shortness of breath was observed.
        Patient has history of hypertension but no diabetes."""

        # Create a pipeline with sliding_window_punct_cleaned strategy
        pipeline = TextProcessingPipeline(
            language="en",
            chunking_pipeline_config=get_sliding_window_punct_cleaned_config(),
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=self.model,
        )

        # Process the text
        chunks = pipeline.process(text)

        # Verify we get chunks
        self.assertGreater(len(chunks), 0, "Should produce chunks")

        # Verify all chunks have text
        for chunk in chunks:
            self.assertIn("text", chunk, "Chunk should have 'text' field")
            self.assertGreater(len(chunk["text"]), 0, "Chunk text should not be empty")

    def test_sliding_window_punct_conj_cleaned_strategy(self):
        """Test the new sliding_window_punct_conj_cleaned strategy."""
        # Sample text with conjunctions and punctuation
        text = """The patient reports fever, headache, and fatigue but no cough.
        She has a history of hypertension and diabetes, but denies smoking or alcohol use."""

        # Create a pipeline with sliding_window_punct_conj_cleaned strategy
        pipeline = TextProcessingPipeline(
            language="en",
            chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=self.model,
        )

        # Process the text
        chunks = pipeline.process(text)

        # Verify we get chunks
        self.assertGreater(len(chunks), 0, "Should produce chunks")

        # Verify all chunks have text
        for chunk in chunks:
            self.assertIn("text", chunk, "Chunk should have 'text' field")
            self.assertGreater(len(chunk["text"]), 0, "Chunk text should not be empty")

        # Extract all chunk texts for easier verification
        chunk_texts = [chunk["text"] for chunk in chunks]
        all_text = " ".join(chunk_texts)

        # Verify that key phrases are preserved after chunking
        # Note: Negation might be handled differently by assertion detectors later
        key_phrases = [
            "patient reports fever",
            "headache",
            "fatigue",
            "cough",  # 'no cough' might be split up
            "history of hypertension",
            "diabetes",
            "smoking",
            "alcohol use",
        ]

        for phrase in key_phrases:
            self.assertIn(
                phrase.lower(),
                all_text.lower(),
                f"Phrase '{phrase}' should be preserved in chunking output",
            )

    def test_language_specific_conjunction_splitting(self):
        """Test that conjunction splitting works with different languages."""
        # Test German text
        german_text = """Der Patient berichtet über Fieber und Kopfschmerzen,
        aber keine Husten oder Atemnot."""

        # Create a pipeline with sliding_window_punct_conj_cleaned strategy
        pipeline = TextProcessingPipeline(
            language="de",
            chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=self.model,
        )

        # Process the text
        chunks = pipeline.process(german_text)

        # Verify we get chunks
        self.assertGreater(len(chunks), 0, "Should produce chunks for German text")

        # Extract all chunk texts
        chunk_texts = [chunk["text"] for chunk in chunks]
        all_text = " ".join(chunk_texts)

        # Verify that key German phrases are preserved
        key_phrases = ["Fieber", "Kopfschmerzen", "keine Husten", "Atemnot"]
        for phrase in key_phrases:
            self.assertIn(
                phrase, all_text, f"German phrase '{phrase}' should be preserved"
            )

    def test_complex_clinical_note(self):
        """Test processing of a more complex clinical note."""
        clinical_note = """
        Patient is a 45-year-old male presenting with chest pain and shortness of breath.
        The pain is described as sharp and radiating to the left arm, but not to the jaw or back.
        Patient reports that the pain started yesterday after mild exercise and has been intermittent.

        Patient has a history of hypertension and hyperlipidemia, but no prior cardiac events.
        He takes lisinopril for blood pressure control and atorvastatin for cholesterol management.

        On examination, vital signs are stable with BP 130/85, HR 82, RR 18, and O2 sat 98% on room air.
        Heart sounds are regular without murmurs, gallops, or rubs. Lungs are clear to auscultation.
        No peripheral edema is noted, and pulses are strong and equal bilaterally.
        """

        # Create a pipeline with sliding_window_punct_conj_cleaned strategy
        pipeline = TextProcessingPipeline(
            language="en",
            chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=self.model,
        )

        # Process the text
        chunks = pipeline.process(clinical_note)

        # Verify we get chunks
        self.assertGreater(
            len(chunks), 5, "Complex clinical note should produce multiple chunks"
        )

        # Extract all chunk texts
        chunk_texts = [chunk["text"] for chunk in chunks]

        # Verify that semantically related information tends to stay together
        # by checking if certain related phrases appear in the same chunk

        # Find chunk containing "chest pain"
        chest_pain_chunks = [
            text for text in chunk_texts if "chest pain" in text.lower()
        ]
        self.assertGreater(
            len(chest_pain_chunks),
            0,
            "Should have at least one chunk with 'chest pain'",
        )

        # Check that all key clinical information is preserved somewhere in the chunks
        key_clinical_terms = [
            "chest pain",
            "shortness of breath",
            "sharp",
            "radiating",
            "left arm",
            "hypertension",
            "hyperlipidemia",
            "lisinopril",
            "atorvastatin",
            "vital signs",
            "BP",
            "lungs",
            "edema",
        ]

        for term in key_clinical_terms:
            self.assertTrue(
                any(term.lower() in chunk.lower() for chunk in chunk_texts),
                f"Clinical term '{term}' should be preserved in some chunk",
            )

        # Similarly, medication information should tend to stay together
        medication_chunks = [
            text
            for text in chunk_texts
            if "lisinopril" in text.lower() or "atorvastatin" in text.lower()
        ]
        self.assertGreater(
            len(medication_chunks),
            0,
            "Should have at least one chunk with medication information",
        )

    def test_custom_chunking_parameters(self):
        """Test that custom parameters for the sliding window can be applied."""
        text = "This is a test sentence with multiple parts and various conjunctions but not too long."

        # Create a pipeline with custom parameters
        custom_config = get_sliding_window_punct_conj_cleaned_config()

        # Customize some parameters - set a very high splitting threshold
        # This should prevent semantic splitting and rely more on punctuation and conjunctions
        for stage in custom_config:
            if stage.get("type") == "sliding_window" and "config" in stage:
                stage["config"]["splitting_threshold"] = 0.9  # Very high threshold
                stage["config"]["window_size_tokens"] = 3  # Small window
                stage["config"]["step_size_tokens"] = 1  # Small step

        pipeline = TextProcessingPipeline(
            language="en",
            chunking_pipeline_config=custom_config,
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=self.model,
        )

        # Process the text
        chunks = pipeline.process(text)

        # With a high threshold, we should see more splitting
        # particularly at conjunctions "and" and "but"
        self.assertGreater(
            len(chunks), 1, "Should produce multiple chunks with custom parameters"
        )

        # Extract chunk texts
        chunk_texts = [chunk["text"] for chunk in chunks]

        # Check that we get reasonable splits
        all_text = " ".join(chunk_texts)
        self.assertIn("test sentence", all_text, "Core content should be preserved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
