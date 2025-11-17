"""Integration tests for chunk index alignment in text processing API.

These tests validate that chunk IDs remain consistent across all components
of the API response, preventing off-by-one errors and invalid references.
"""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from api.main import app

    return TestClient(app)


class TestChunkIdConsistency:
    """Test chunk ID consistency across API response components."""

    def test_chunk_ids_are_sequential_one_based(self, api_client):
        """Chunk IDs must be sequential starting from 1."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia. No heart disease.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        chunks = data["processed_chunks"]
        chunk_ids = [c["chunk_id"] for c in chunks]

        # Invariant: sequential 1-based IDs
        assert chunk_ids == list(range(1, len(chunks) + 1)), (
            f"Chunk IDs should be sequential 1-based, got {chunk_ids}"
        )

    def test_source_chunk_ids_reference_existing_chunks(self, api_client):
        """All source_chunk_ids must reference existing processed chunks."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia. No heart disease.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        valid_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}

        for term in data["aggregated_hpo_terms"]:
            for chunk_id in term["source_chunk_ids"]:
                assert chunk_id in valid_chunk_ids, (
                    f"Term {term['hpo_id']} references non-existent chunk {chunk_id}. "
                    f"Valid IDs: {valid_chunk_ids}"
                )

    def test_text_attribution_chunk_ids_valid(self, api_client):
        """All text_attribution chunk_ids must reference existing chunks."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        valid_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}

        for term in data["aggregated_hpo_terms"]:
            for attribution in term["text_attributions"]:
                assert attribution["chunk_id"] in valid_chunk_ids, (
                    f"Term {term['hpo_id']} attribution references "
                    f"non-existent chunk {attribution['chunk_id']}"
                )

    def test_top_evidence_chunk_id_valid_when_present(self, api_client):
        """top_evidence_chunk_id must reference existing chunk if not None."""
        # Arrange
        request = {
            "text_content": "Patient has seizures.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        valid_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}

        for term in data["aggregated_hpo_terms"]:
            top_chunk_id = term.get("top_evidence_chunk_id")
            if top_chunk_id is not None:
                assert top_chunk_id in valid_chunk_ids, (
                    f"Term {term['hpo_id']} top_evidence_chunk_id {top_chunk_id} "
                    f"does not reference an existing chunk"
                )


class TestChunkAlignmentAcrossStrategies:
    """Test chunk alignment for all chunking strategies."""

    @pytest.mark.parametrize(
        "strategy",
        [
            "simple",
            "semantic",
            "detailed",
            "sliding_window",
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_strategy_maintains_chunk_alignment(self, api_client, strategy):
        """All chunking strategies must maintain valid chunk references."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia. Tremor noted. No heart disease.",
            "language": "en",
            "chunking_strategy": strategy,
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate chunk ID consistency
        valid_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}

        for term in data["aggregated_hpo_terms"]:
            # Check source_chunk_ids
            for chunk_id in term["source_chunk_ids"]:
                assert chunk_id in valid_chunk_ids, (
                    f"Strategy '{strategy}': Term {term['hpo_id']} "
                    f"references invalid chunk {chunk_id}"
                )

            # Check text_attributions
            for attribution in term["text_attributions"]:
                assert attribution["chunk_id"] in valid_chunk_ids, (
                    f"Strategy '{strategy}': Term {term['hpo_id']} "
                    f"attribution has invalid chunk_id {attribution['chunk_id']}"
                )

            # Check top_evidence_chunk_id
            if term.get("top_evidence_chunk_id") is not None:
                assert term["top_evidence_chunk_id"] in valid_chunk_ids, (
                    f"Strategy '{strategy}': Term {term['hpo_id']} "
                    f"has invalid top_evidence_chunk_id"
                )


class TestEdgeCases:
    """Test edge cases for chunk index alignment."""

    def test_single_chunk_all_references_to_one(self, api_client):
        """Single chunk case - all references should be to chunk_id 1."""
        # Arrange
        request = {
            "text_content": "Seizures",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should have at least 1 chunk (may be more depending on processing)
        assert len(data["processed_chunks"]) >= 1
        assert data["processed_chunks"][0]["chunk_id"] == 1

        # All references should be valid chunk IDs
        valid_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}
        for term in data["aggregated_hpo_terms"]:
            assert all(cid in valid_chunk_ids for cid in term["source_chunk_ids"]), (
                f"Single/minimal chunk scenario: all source_chunk_ids should be valid, "
                f"got {term['source_chunk_ids']}, valid: {valid_chunk_ids}"
            )

    def test_empty_text_graceful_handling(self, api_client):
        """Empty text should be handled gracefully."""
        # Arrange
        request = {
            "text_content": "",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert - should handle gracefully (200 or 400 acceptable)
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            # Should have valid structure
            assert "processed_chunks" in data
            assert "aggregated_hpo_terms" in data
            assert isinstance(data["processed_chunks"], list)
            assert isinstance(data["aggregated_hpo_terms"], list)

    def test_whitespace_only_text_handling(self, api_client):
        """Whitespace-only text should be handled gracefully."""
        # Arrange
        request = {
            "text_content": "   \n\t   ",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            assert "processed_chunks" in data
            assert "aggregated_hpo_terms" in data

    def test_very_long_text_maintains_alignment(self, api_client):
        """Long text with many chunks maintains alignment."""
        # Arrange - create long text
        sentences = [
            "Patient has seizures.",
            "Ataxia noted.",
            "No heart disease.",
            "Tremor observed.",
            "Muscle weakness present.",
        ]
        long_text = " ".join(sentences * 5)  # 25 sentences

        request = {
            "text_content": long_text,
            "language": "en",
            "chunking_strategy": "simple",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should have multiple chunks
        num_chunks = len(data["processed_chunks"])
        assert num_chunks > 1

        # All chunk IDs should be valid
        valid_chunk_ids = set(range(1, num_chunks + 1))
        actual_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}
        assert actual_chunk_ids == valid_chunk_ids

        # All references should be valid
        for term in data["aggregated_hpo_terms"]:
            assert all(1 <= cid <= num_chunks for cid in term["source_chunk_ids"]), (
                f"Invalid source_chunk_ids in long text: {term['source_chunk_ids']}"
            )


class TestHPOMatchesInChunks:
    """Test HPO matches within processed chunks structure."""

    def test_hpo_matches_structure_valid(self, api_client):
        """HPO matches in chunks should have valid structure."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        for chunk in data["processed_chunks"]:
            # Each chunk should have valid structure
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "status" in chunk
            assert "hpo_matches" in chunk

            # hpo_matches should be a list
            assert isinstance(chunk["hpo_matches"], list)

            # Each match should have required fields
            for match in chunk["hpo_matches"]:
                assert "hpo_id" in match
                assert "name" in match
                assert "score" in match
                assert isinstance(match["score"], (int, float))
                assert match["score"] >= 0.0

    def test_chunk_id_consistency_in_response_structure(self, api_client):
        """Verify chunk_id consistency throughout entire response."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia. Tremor present.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Collect all chunk_id references from different parts of response
        chunk_ids_from_chunks = {c["chunk_id"] for c in data["processed_chunks"]}

        chunk_ids_from_sources = set()
        chunk_ids_from_attributions = set()
        chunk_ids_from_top_evidence = set()

        for term in data["aggregated_hpo_terms"]:
            chunk_ids_from_sources.update(term["source_chunk_ids"])

            for attr in term["text_attributions"]:
                chunk_ids_from_attributions.add(attr["chunk_id"])

            if term.get("top_evidence_chunk_id") is not None:
                chunk_ids_from_top_evidence.add(term["top_evidence_chunk_id"])

        # All referenced chunk IDs must be from processed_chunks
        assert chunk_ids_from_sources.issubset(chunk_ids_from_chunks), (
            f"source_chunk_ids contains invalid references: "
            f"{chunk_ids_from_sources - chunk_ids_from_chunks}"
        )

        assert chunk_ids_from_attributions.issubset(chunk_ids_from_chunks), (
            f"text_attribution chunk_ids contains invalid references: "
            f"{chunk_ids_from_attributions - chunk_ids_from_chunks}"
        )

        assert chunk_ids_from_top_evidence.issubset(chunk_ids_from_chunks), (
            f"top_evidence_chunk_id contains invalid references: "
            f"{chunk_ids_from_top_evidence - chunk_ids_from_chunks}"
        )
