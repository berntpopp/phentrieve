"""Unit tests for query_orchestrator pure functions.

Tests for helper functions used in query orchestration:
- convert_results_to_candidates: Result format conversion
- segment_text: Text segmentation
- format_results: Result formatting and filtering

Following best practices:
- Test pure functions directly (no complex mocking)
- Mock only external dependencies (file I/O, models)
- Comprehensive edge case coverage
- Clear Arrange-Act-Assert structure
"""

import pytest

from phentrieve.retrieval.query_orchestrator import (
    convert_results_to_candidates,
    format_results,
    segment_text,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for convert_results_to_candidates()
# =============================================================================


class TestConvertResultsToCandidates:
    """Test convert_results_to_candidates() function."""

    def test_converts_valid_chromadb_results_to_candidates(self):
        """Test converting valid ChromaDB results to candidate format."""
        # Arrange
        results = {
            "ids": [["HP:0001250", "HP:0002066"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001250", "label": "Seizure"},
                    {"hpo_id": "HP:0002066", "label": "Gait ataxia"},
                ]
            ],
            "documents": [["Abnormal electrical discharge", "Walking difficulty"]],
            "distances": [[0.15, 0.35]],
        }

        # Act
        candidates = convert_results_to_candidates(results)

        # Assert
        assert len(candidates) == 2
        assert candidates[0]["hpo_id"] == "HP:0001250"
        assert candidates[0]["english_doc"] == "Abnormal electrical discharge"
        assert candidates[0]["comparison_text"] == "Abnormal electrical discharge"
        assert candidates[0]["rank"] == 1
        assert "bi_encoder_score" in candidates[0]
        assert candidates[1]["rank"] == 2

    def test_uses_english_doc_for_comparison(self):
        """Test always uses English documents for cross-lingual comparison."""
        # Arrange
        results = {
            "ids": [["HP:0001250"]],
            "metadatas": [[{"hpo_id": "HP:0001250", "label": "Seizure"}]],
            "documents": [["English document"]],
            "distances": [[0.2]],
        }

        # Act
        candidates = convert_results_to_candidates(results)

        # Assert
        assert candidates[0]["comparison_text"] == "English document"
        assert candidates[0]["english_doc"] == "English document"

    def test_returns_empty_list_for_empty_results(self):
        """Test returns empty list when results are empty."""
        # Arrange
        results = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}

        # Act
        candidates = convert_results_to_candidates(results)

        # Assert
        assert candidates == []

    def test_returns_empty_list_when_ids_missing(self):
        """Test returns empty list when ids are missing."""
        # Arrange
        results = {}

        # Act
        candidates = convert_results_to_candidates(results)

        # Assert
        assert candidates == []

    def test_returns_empty_list_when_ids_none(self):
        """Test returns empty list when ids is None."""
        # Arrange
        results = {"ids": None}

        # Act
        candidates = convert_results_to_candidates(results)

        # Assert
        assert candidates == []

    def test_calculates_bi_encoder_similarity_from_distance(self):
        """Test bi-encoder similarity is calculated from distance."""
        # Arrange
        results = {
            "ids": [["HP:0001250"]],
            "metadatas": [[{"hpo_id": "HP:0001250", "label": "Seizure"}]],
            "documents": [["Doc"]],
            "distances": [[0.1]],  # Small distance = high similarity
        }

        # Act
        candidates = convert_results_to_candidates(results)

        # Assert
        # Distance 0.1 should convert to similarity ~0.9
        assert candidates[0]["bi_encoder_score"] > 0.8
        assert candidates[0]["bi_encoder_score"] <= 1.0


# =============================================================================
# Tests for segment_text()
# =============================================================================


class TestSegmentText:
    """Test segment_text() function."""

    def test_segments_single_sentence(self):
        """Test segmenting text with single sentence."""
        # Arrange
        text = "Patient has seizures."

        # Act
        sentences = segment_text(text, lang="en")

        # Assert
        assert len(sentences) == 1
        assert sentences[0] == "Patient has seizures."

    def test_segments_multiple_sentences(self):
        """Test segmenting text with multiple sentences."""
        # Arrange
        text = "Patient has seizures. They occur at night. Family history is negative."

        # Act
        sentences = segment_text(text, lang="en")

        # Assert
        assert len(sentences) == 3
        # Note: pysbd may add trailing spaces to sentences
        assert any("Patient has seizures" in s for s in sentences)
        assert any("They occur at night" in s for s in sentences)

    def test_detects_english_for_ascii_text_when_lang_none(self):
        """Test defaults to English for ASCII text when lang is None."""
        # Arrange
        text = "This is ASCII text with more than 20 characters."

        # Act
        sentences = segment_text(text, lang=None)

        # Assert - Should successfully segment (implies English detection)
        assert len(sentences) >= 1
        assert isinstance(sentences, list)

    def test_defaults_to_english_for_short_text(self):
        """Test defaults to English for short text."""
        # Arrange
        text = "Short."  # Less than 20 chars

        # Act
        sentences = segment_text(text, lang=None)

        # Assert
        assert len(sentences) == 1

    def test_handles_empty_text(self):
        """Test handles empty text gracefully."""
        # Arrange
        text = ""

        # Act
        sentences = segment_text(text, lang="en")

        # Assert
        # pysbd returns empty list or list with empty string
        assert isinstance(sentences, list)

    def test_preserves_punctuation_in_segments(self):
        """Test preserves punctuation in segmented text."""
        # Arrange
        text = "Question? Answer! Statement."

        # Act
        sentences = segment_text(text, lang="en")

        # Assert
        assert any("?" in s for s in sentences)
        assert any("!" in s for s in sentences)


# =============================================================================
# Tests for format_results()
# =============================================================================


class TestFormatResults:
    """Test format_results() function."""

    def test_formats_valid_results_with_defaults(self):
        """Test formatting valid results with default parameters."""
        # Arrange
        results = {
            "ids": [["HP:0001250", "HP:0002066"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001250", "label": "Seizure"},
                    {"hpo_id": "HP:0002066", "label": "Gait ataxia"},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        # Act
        formatted = format_results(results, query="seizures")

        # Assert
        assert "results" in formatted
        assert "query_text_processed" in formatted
        assert "header_info" in formatted
        assert len(formatted["results"]) == 2
        assert formatted["query_text_processed"] == "seizures"

    def test_filters_results_by_threshold(self):
        """Test filters results below similarity threshold."""
        # Arrange
        results = {
            "ids": [["HP:0001250", "HP:0002066", "HP:0003333"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001250", "label": "High score"},
                    {"hpo_id": "HP:0002066", "label": "Medium score"},
                    {"hpo_id": "HP:0003333", "label": "Low score"},
                ]
            ],
            "distances": [[0.1, 0.5, 0.9]],  # High, medium, low similarity
        }

        # Act
        formatted = format_results(results, threshold=0.5, max_results=10)

        # Assert
        # With threshold 0.5, only high and medium similarity should pass
        assert len(formatted["results"]) <= 2
        # Verify all results are above threshold
        for result in formatted["results"]:
            assert result["similarity"] >= 0.5

    def test_limits_results_by_max_results(self):
        """Test limits number of results to max_results."""
        # Arrange
        results = {
            "ids": [["HP:0001", "HP:0002", "HP:0003", "HP:0004", "HP:0005"]],
            "metadatas": [
                [{"hpo_id": f"HP:000{i}", "label": f"Term {i}"} for i in range(1, 6)]
            ],
            "distances": [[0.1, 0.15, 0.2, 0.25, 0.3]],
        }

        # Act
        formatted = format_results(results, threshold=0.0, max_results=3)

        # Assert
        assert len(formatted["results"]) == 3

    def test_returns_empty_results_for_empty_input(self):
        """Test returns empty results structure for empty input."""
        # Arrange
        results = {"ids": [[]], "metadatas": [[]], "distances": [[]]}

        # Act
        formatted = format_results(results, query="test")

        # Assert
        assert formatted["results"] == []
        assert formatted["query_text_processed"] == "test"
        assert "No matching" in formatted["header_info"]

    def test_handles_results_without_ids(self):
        """Test handles results when ids are missing."""
        # Arrange
        results = {}

        # Act
        formatted = format_results(results)

        # Assert
        assert formatted["results"] == []

    def test_handles_reranked_results(self):
        """Test handles reranked results with cross_encoder_score."""
        # Arrange
        results = {
            "ids": [["HP:0001250", "HP:0002066"]],
            "metadatas": [
                [
                    {
                        "hpo_id": "HP:0001250",
                        "label": "Term 1",
                        "cross_encoder_score": 0.95,
                        "original_rank": 2,
                    },
                    {
                        "hpo_id": "HP:0002066",
                        "label": "Term 2",
                        "cross_encoder_score": 0.85,
                        "original_rank": 1,
                    },
                ]
            ],
            "distances": [[0.2, 0.1]],
        }

        # Act
        formatted = format_results(results, max_results=10, reranked=True)

        # Assert
        # Results should be in order provided (reranked order), not sorted by distance
        assert len(formatted["results"]) == 2
        assert formatted["results"][0]["hpo_id"] == "HP:0001250"
        assert formatted["results"][0]["cross_encoder_score"] == 0.95

    def test_coerces_max_results_from_string(self):
        """Test coerces max_results from string to integer."""
        # Arrange
        results = {
            "ids": [["HP:0001", "HP:0002", "HP:0003"]],
            "metadatas": [
                [{"hpo_id": f"HP:000{i}", "label": f"Term {i}"} for i in range(1, 4)]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }

        # Act
        formatted = format_results(results, threshold=0.0, max_results="2")  # String!

        # Assert
        assert len(formatted["results"]) == 2  # Should coerce to int 2

    def test_uses_default_max_results_when_none(self):
        """Test uses default when max_results is None."""
        # Arrange
        results = {
            "ids": [["HP:0001"]],
            "metadatas": [[{"hpo_id": "HP:0001", "label": "Term"}]],
            "distances": [[0.1]],
        }

        # Act
        formatted = format_results(results, max_results=None)

        # Assert
        assert "results" in formatted
        assert len(formatted["results"]) >= 0  # Should not crash

    def test_handles_missing_distances(self):
        """Test handles results without distances field."""
        # Arrange
        results = {
            "ids": [["HP:0001250"]],
            "metadatas": [[{"hpo_id": "HP:0001250", "label": "Term"}]],
            # No "distances" key
        }

        # Act
        formatted = format_results(results)

        # Assert
        assert len(formatted["results"]) == 1
        # Should use default distance of 0.0 (similarity = 1.0)
        assert formatted["results"][0]["similarity"] == 1.0

    def test_sorts_non_reranked_results_by_similarity(self):
        """Test sorts non-reranked results by bi-encoder similarity."""
        # Arrange
        results = {
            "ids": [["HP:0001", "HP:0002", "HP:0003"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001", "label": "Low"},
                    {"hpo_id": "HP:0002", "label": "High"},
                    {"hpo_id": "HP:0003", "label": "Medium"},
                ]
            ],
            "distances": [[0.8, 0.1, 0.4]],  # Low, High, Medium similarity
        }

        # Act
        formatted = format_results(results, threshold=0.0, max_results=10)

        # Assert
        # Should be sorted by similarity (distance converted): High, Medium, Low
        assert formatted["results"][0]["hpo_id"] == "HP:0002"  # Highest similarity
        assert formatted["results"][1]["hpo_id"] == "HP:0003"  # Medium
        assert formatted["results"][2]["hpo_id"] == "HP:0001"  # Lowest
