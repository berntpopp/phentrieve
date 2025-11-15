"""Integration tests for chunking pipeline (pytest style)."""

import pytest
from sentence_transformers import SentenceTransformer

from phentrieve.config import (
    get_sliding_window_punct_cleaned_config,
    get_sliding_window_punct_conj_cleaned_config,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline


# Mark all tests as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedding_model():
    """Real embedding model for integration tests (module-scoped for performance)."""
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")


class TestChunkingPipelineIntegration:
    """Integration tests for chunking pipeline with real models."""

    def test_sliding_window_punct_cleaned_strategy(self, embedding_model):
        """Test the sliding_window_punct_cleaned strategy."""
        text = """The patient reports fever, headache, and fatigue.
        No cough or shortness of breath was observed.
        Patient has history of hypertension but no diabetes."""

        pipeline = TextProcessingPipeline(
            language="en",
            chunking_pipeline_config=get_sliding_window_punct_cleaned_config(),
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=embedding_model,
        )

        chunks = pipeline.process(text)

        assert len(chunks) > 0, "Should produce chunks"

        for chunk in chunks:
            assert "text" in chunk, "Chunk should have 'text' field"
            assert len(chunk["text"]) > 0, "Chunk text should not be empty"

    def test_sliding_window_punct_conj_cleaned_strategy(self, embedding_model):
        """Test the sliding_window_punct_conj_cleaned strategy."""
        text = """The patient reports fever, headache, and fatigue but no cough.
        She has a history of hypertension and diabetes, but denies smoking or alcohol use."""

        pipeline = TextProcessingPipeline(
            language="en",
            chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=embedding_model,
        )

        chunks = pipeline.process(text)

        assert len(chunks) > 0, "Should produce chunks"

        for chunk in chunks:
            assert "text" in chunk, "Chunk should have 'text' field"
            assert len(chunk["text"]) > 0, "Chunk text should not be empty"

        # Extract all chunk texts
        chunk_texts = [chunk["text"] for chunk in chunks]
        all_text = " ".join(chunk_texts)

        # Verify key phrases are preserved
        key_phrases = [
            "patient reports fever",
            "headache",
            "fatigue",
            "cough",
            "history of hypertension",
            "diabetes",
            "smoking",
            "alcohol use",
        ]

        for phrase in key_phrases:
            assert phrase.lower() in all_text.lower(), \
                f"Phrase '{phrase}' should be preserved in chunking output"

    def test_language_specific_conjunction_splitting(self, embedding_model):
        """Test conjunction splitting with different languages."""
        german_text = """Der Patient berichtet über Fieber und Kopfschmerzen,
        aber keine Husten oder Atemnot."""

        pipeline = TextProcessingPipeline(
            language="de",
            chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=embedding_model,
        )

        chunks = pipeline.process(german_text)

        assert len(chunks) > 0, "Should produce chunks for German text"

        chunk_texts = [chunk["text"] for chunk in chunks]
        all_text = " ".join(chunk_texts)

        # Verify German phrases are preserved
        key_phrases = ["Fieber", "Kopfschmerzen", "keine Husten", "Atemnot"]
        for phrase in key_phrases:
            assert phrase in all_text, f"German phrase '{phrase}' should be preserved"

    def test_complex_clinical_note(self, embedding_model):
        """Test processing of complex clinical note."""
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

        pipeline = TextProcessingPipeline(
            language="en",
            chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=embedding_model,
        )

        chunks = pipeline.process(clinical_note)

        assert len(chunks) > 5, "Complex clinical note should produce multiple chunks"

        chunk_texts = [chunk["text"] for chunk in chunks]

        # Find chunk containing "chest pain"
        chest_pain_chunks = [text for text in chunk_texts if "chest pain" in text.lower()]
        assert len(chest_pain_chunks) > 0, "Should have at least one chunk with 'chest pain'"

        # Verify all key clinical information is preserved
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
            assert any(term.lower() in chunk.lower() for chunk in chunk_texts), \
                f"Clinical term '{term}' should be preserved in some chunk"

        # Medication information should be present
        medication_chunks = [
            text
            for text in chunk_texts
            if "lisinopril" in text.lower() or "atorvastatin" in text.lower()
        ]
        assert len(medication_chunks) > 0, \
            "Should have at least one chunk with medication information"

    def test_custom_chunking_parameters(self, embedding_model):
        """Test custom parameters for sliding window."""
        text = "This is a test sentence with multiple parts and various conjunctions but not too long."

        # Custom configuration
        custom_config = get_sliding_window_punct_conj_cleaned_config()

        # Customize parameters - high splitting threshold
        for stage in custom_config:
            if stage.get("type") == "sliding_window" and "config" in stage:
                stage["config"]["splitting_threshold"] = 0.9  # Very high
                stage["config"]["window_size_tokens"] = 3  # Small window
                stage["config"]["step_size_tokens"] = 1  # Small step

        pipeline = TextProcessingPipeline(
            language="en",
            chunking_pipeline_config=custom_config,
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=embedding_model,
        )

        chunks = pipeline.process(text)

        assert len(chunks) > 1, "Should produce multiple chunks with custom parameters"

        chunk_texts = [chunk["text"] for chunk in chunks]
        all_text = " ".join(chunk_texts)
        assert "test sentence" in all_text, "Core content should be preserved"
