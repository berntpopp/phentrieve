"""Real unit tests for reranker module (actual code execution)."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from phentrieve.retrieval.reranker import load_cross_encoder, rerank_with_cross_encoder

pytestmark = pytest.mark.unit


class TestLoadCrossEncoder:
    """Test load_cross_encoder function with real logic execution."""

    @patch("phentrieve.retrieval.reranker.CrossEncoder")
    @patch("phentrieve.retrieval.reranker.torch.cuda.is_available")
    def test_successful_load_with_auto_cpu(self, mock_cuda, mock_ce):
        """Test successful model loading with automatic CPU detection."""
        # Arrange
        mock_cuda.return_value = False
        mock_model = Mock()
        mock_ce.return_value = mock_model

        # Act
        result = load_cross_encoder("test-model")

        # Assert
        assert result == mock_model
        mock_ce.assert_called_once_with("test-model", device="cpu")

    @patch("phentrieve.retrieval.reranker.CrossEncoder")
    @patch("phentrieve.retrieval.reranker.torch.cuda.is_available")
    def test_successful_load_with_auto_cuda(self, mock_cuda, mock_ce):
        """Test successful model loading with automatic CUDA detection."""
        # Arrange
        mock_cuda.return_value = True
        mock_model = Mock()
        mock_ce.return_value = mock_model

        # Act
        result = load_cross_encoder("test-model")

        # Assert
        assert result == mock_model
        mock_ce.assert_called_once_with("test-model", device="cuda")

    @patch("phentrieve.retrieval.reranker.CrossEncoder")
    @patch("phentrieve.retrieval.reranker.torch.cuda.is_available")
    def test_successful_load_with_explicit_device(self, mock_cuda, mock_ce):
        """Test successful model loading with explicit device override."""
        # Arrange
        mock_cuda.return_value = True  # CUDA available but we want CPU
        mock_model = Mock()
        mock_ce.return_value = mock_model

        # Act
        result = load_cross_encoder("test-model", device="cpu")

        # Assert
        assert result == mock_model
        mock_ce.assert_called_once_with("test-model", device="cpu")

    @patch("phentrieve.retrieval.reranker.CrossEncoder")
    @patch("phentrieve.retrieval.reranker.torch.cuda.is_available")
    def test_load_failure_error_handling(self, mock_cuda, mock_ce):
        """Test error handling when model loading fails."""
        # Arrange
        mock_cuda.return_value = False
        mock_ce.side_effect = Exception("Model not found")

        # Act
        result = load_cross_encoder("invalid-model")

        # Assert
        assert result is None


class TestRerankWithCrossEncoder:
    """Test rerank_with_cross_encoder function with real logic execution."""

    def test_rerank_with_traditional_scores(self):
        """Test reranking with traditional single-value scores."""
        # Arrange
        mock_model = Mock()
        # Traditional cross-encoder returns single scores
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.7])

        query = "test query"
        candidates = [
            {"english_doc": "doc1", "id": "1"},
            {"english_doc": "doc2", "id": "2"},
            {"english_doc": "doc3", "id": "3"},
        ]

        # Act
        result = rerank_with_cross_encoder(query, candidates, mock_model)

        # Assert
        assert len(result) == 3
        assert result[0]["id"] == "1"  # Highest score 0.9
        assert result[0]["cross_encoder_score"] == 0.9
        assert result[1]["id"] == "3"  # Second highest 0.7
        assert result[1]["cross_encoder_score"] == 0.7
        assert result[2]["id"] == "2"  # Lowest 0.3
        assert result[2]["cross_encoder_score"] == 0.3

    def test_rerank_with_nli_scores(self):
        """Test reranking with NLI-style multi-value scores."""
        # Arrange
        mock_model = Mock()
        # NLI model returns array with [entailment, neutral, contradiction] probabilities
        mock_model.predict.return_value = np.array(
            [
                [0.8, 0.1, 0.1],  # doc1 - high entailment
                [0.2, 0.3, 0.5],  # doc2 - low entailment
                [0.6, 0.2, 0.2],  # doc3 - medium entailment
            ]
        )

        query = "test query"
        candidates = [
            {"english_doc": "doc1", "id": "1"},
            {"english_doc": "doc2", "id": "2"},
            {"english_doc": "doc3", "id": "3"},
        ]

        # Act
        result = rerank_with_cross_encoder(query, candidates, mock_model)

        # Assert
        assert len(result) == 3
        assert result[0]["id"] == "1"  # Highest entailment 0.8
        assert result[0]["cross_encoder_score"] == 0.8
        assert result[1]["id"] == "3"  # Second highest 0.6
        assert result[2]["id"] == "2"  # Lowest 0.2

    def test_rerank_with_comparison_text(self):
        """Test reranking uses comparison_text when available."""
        # Arrange
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9, 0.7])

        query = "test query"
        candidates = [
            {"comparison_text": "translated text 1", "english_doc": "orig1", "id": "1"},
            {"comparison_text": "translated text 2", "english_doc": "orig2", "id": "2"},
        ]

        # Act
        rerank_with_cross_encoder(query, candidates, mock_model)

        # Assert
        # Verify predict was called with comparison_text, not english_doc
        call_args = mock_model.predict.call_args[0][0]
        assert call_args[0] == (query, "translated text 1")
        assert call_args[1] == (query, "translated text 2")

    def test_rerank_fallback_to_english_doc(self):
        """Test reranking falls back to english_doc when comparison_text missing."""
        # Arrange
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9, 0.7])

        query = "test query"
        candidates = [
            {"english_doc": "doc1", "id": "1"},  # No comparison_text
            {"english_doc": "doc2", "id": "2"},
        ]

        # Act
        rerank_with_cross_encoder(query, candidates, mock_model)

        # Assert
        call_args = mock_model.predict.call_args[0][0]
        assert call_args[0] == (query, "doc1")
        assert call_args[1] == (query, "doc2")

    def test_rerank_empty_candidates(self):
        """Test reranking with empty candidates list."""
        # Arrange
        mock_model = Mock()
        query = "test query"
        candidates = []

        # Act
        result = rerank_with_cross_encoder(query, candidates, mock_model)

        # Assert
        assert result == []
        mock_model.predict.assert_not_called()

    def test_rerank_error_handling(self):
        """Test error handling during prediction."""
        # Arrange
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")

        query = "test query"
        candidates = [
            {"english_doc": "doc1", "id": "1"},
            {"english_doc": "doc2", "id": "2"},
        ]

        # Act
        result = rerank_with_cross_encoder(query, candidates, mock_model)

        # Assert
        # Should return original candidates unchanged on error
        assert result == candidates
        assert "cross_encoder_score" not in result[0]

    def test_rerank_preserves_original_fields(self):
        """Test that reranking preserves all original candidate fields."""
        # Arrange
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9, 0.7])

        query = "test query"
        candidates = [
            {"english_doc": "doc1", "id": "1", "metadata": {"key": "value1"}},
            {"english_doc": "doc2", "id": "2", "metadata": {"key": "value2"}},
        ]

        # Act
        result = rerank_with_cross_encoder(query, candidates, mock_model)

        # Assert
        assert result[0]["metadata"] == {"key": "value1"}
        assert result[1]["metadata"] == {"key": "value2"}
        assert "cross_encoder_score" in result[0]
        assert "cross_encoder_score" in result[1]

    def test_rerank_handles_missing_english_doc(self):
        """Test reranking handles candidates with missing english_doc field."""
        # Arrange
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9])

        query = "test query"
        candidates = [
            {"id": "1"},  # Missing both comparison_text and english_doc
        ]

        # Act
        result = rerank_with_cross_encoder(query, candidates, mock_model)

        # Assert
        # Should still work, using empty string as fallback
        assert len(result) == 1
        assert "cross_encoder_score" in result[0]
