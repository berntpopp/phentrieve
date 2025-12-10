"""Integration tests for pipeline position tracking.

Tests the include_positions parameter of TextProcessingPipeline.process().
"""

import pytest
from sentence_transformers import SentenceTransformer

from phentrieve.config import get_sliding_window_punct_conj_cleaned_config
from phentrieve.text_processing.pipeline import TextProcessingPipeline

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedding_model() -> SentenceTransformer:
    """Real embedding model for integration tests (module-scoped for performance)."""
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")


@pytest.fixture
def pipeline(embedding_model: SentenceTransformer) -> TextProcessingPipeline:
    """Create pipeline with semantic chunking for testing."""
    return TextProcessingPipeline(
        language="en",
        chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
        assertion_config={"disable": True},
        sbert_model_for_semantic_chunking=embedding_model,
    )


class TestPipelinePositions:
    """Integration tests for position tracking in TextProcessingPipeline."""

    def test_default_no_positions(self, pipeline: TextProcessingPipeline) -> None:
        """Test that positions are NOT included by default (backward compatibility)."""
        chunks = pipeline.process("Patient has seizures and headache.")

        assert len(chunks) > 0, "Should produce at least one chunk"

        for chunk in chunks:
            assert "start_char" not in chunk, (
                "start_char should not be present by default"
            )
            assert "end_char" not in chunk, "end_char should not be present by default"

    def test_with_positions(self, pipeline: TextProcessingPipeline) -> None:
        """Test that positions ARE included when requested."""
        text = "Patient has seizures. No fever was observed."
        chunks = pipeline.process(text, include_positions=True)

        assert len(chunks) > 0, "Should produce at least one chunk"

        for chunk in chunks:
            assert "start_char" in chunk, "start_char should be present when requested"
            assert "end_char" in chunk, "end_char should be present when requested"
            # Valid positions or -1 if not found
            if chunk["start_char"] >= 0:
                assert chunk["start_char"] < len(text), (
                    "start_char should be within text"
                )
                assert chunk["end_char"] <= len(text), "end_char should be within text"
                assert chunk["start_char"] < chunk["end_char"], (
                    "start should be before end"
                )

    def test_positions_are_valid_indices(
        self, pipeline: TextProcessingPipeline
    ) -> None:
        """Test that positions can be used to slice original text."""
        text = "The patient presents with headache, fever, and fatigue."
        chunks = pipeline.process(text, include_positions=True)

        for chunk in chunks:
            start = chunk["start_char"]
            end = chunk["end_char"]

            if start >= 0 and end >= 0:
                # The sliced text should contain the chunk text (possibly with whitespace diffs)
                sliced = text[start:end]
                # Normalize whitespace for comparison
                import re

                norm_sliced = re.sub(r"\s+", " ", sliced.strip())
                norm_chunk = re.sub(r"\s+", " ", chunk["text"].strip())
                # The chunk text should be derivable from the sliced region
                assert norm_chunk in norm_sliced or norm_sliced in norm_chunk, (
                    f"Chunk '{chunk['text']}' not found in slice '{sliced}'"
                )

    def test_sequential_positions_no_overlap(
        self, pipeline: TextProcessingPipeline
    ) -> None:
        """Test that chunk positions are sequential without overlaps."""
        text = "Pain noted today. Pain increased yesterday. Pain resolved this morning."
        chunks = pipeline.process(text, include_positions=True)

        # Filter chunks with valid positions
        valid_chunks = [
            c for c in chunks if c["start_char"] >= 0 and c["end_char"] >= 0
        ]

        if len(valid_chunks) > 1:
            # Sort by start position
            sorted_chunks = sorted(valid_chunks, key=lambda c: c["start_char"])

            for i in range(1, len(sorted_chunks)):
                prev_end = sorted_chunks[i - 1]["end_char"]
                curr_start = sorted_chunks[i]["start_char"]
                assert curr_start >= prev_end, (
                    f"Chunk {i} starts ({curr_start}) before chunk {i - 1} ends ({prev_end})"
                )

    def test_empty_text_returns_empty(self, pipeline: TextProcessingPipeline) -> None:
        """Test that empty text returns empty list."""
        chunks = pipeline.process("", include_positions=True)
        assert chunks == []

        chunks = pipeline.process("   ", include_positions=True)
        assert chunks == []

    def test_unicode_text_positions(self, pipeline: TextProcessingPipeline) -> None:
        """Test position tracking with Unicode text (German)."""
        text = "Der Patient hat Trinkschwäche und Muskelhypotonie."
        chunks = pipeline.process(text, include_positions=True)

        assert len(chunks) > 0
        for chunk in chunks:
            start = chunk["start_char"]
            end = chunk["end_char"]
            if start >= 0:
                # Unicode positions should be valid
                assert end <= len(text), "Position should not exceed text length"

    def test_multiline_text_positions(self, pipeline: TextProcessingPipeline) -> None:
        """Test position tracking with multiline text."""
        text = """First symptom: headache.
Second symptom: fever.
Third symptom: fatigue."""
        chunks = pipeline.process(text, include_positions=True)

        assert len(chunks) > 0
        for chunk in chunks:
            if chunk["start_char"] >= 0:
                assert chunk["end_char"] <= len(text)


class TestPipelinePositionsBackwardCompatibility:
    """Tests ensuring backward compatibility of position tracking."""

    def test_existing_fields_unchanged(self, pipeline: TextProcessingPipeline) -> None:
        """Test that existing chunk fields are still present."""
        chunks = pipeline.process("Patient has fever.", include_positions=True)

        for chunk in chunks:
            # All existing fields should still be present
            assert "text" in chunk
            assert "status" in chunk
            assert "assertion_details" in chunk
            assert "source_indices" in chunk

    def test_process_signature_backward_compatible(
        self, pipeline: TextProcessingPipeline
    ) -> None:
        """Test that process() works without include_positions argument."""
        # This should work without any arguments (backward compatible)
        chunks = pipeline.process("Patient has fever.")
        assert len(chunks) > 0
