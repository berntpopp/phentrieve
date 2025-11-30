"""Unit tests for API helper functions.

Tests for the API-specific retrieval orchestration:
- execute_hpo_retrieval_for_api: Main API query processing function

Following best practices:
- Mock external dependencies (retriever, cross-encoder, assertion detector)
- Test async function behavior
- Comprehensive path coverage (empty query, no results, reranking)
- Clear Arrange-Act-Assert structure
"""

import pytest

from phentrieve.retrieval.api_helpers import execute_hpo_retrieval_for_api

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for execute_hpo_retrieval_for_api()
# =============================================================================


class TestExecuteHpoRetrievalForApi:
    """Test execute_hpo_retrieval_for_api() function."""

    @pytest.mark.asyncio
    async def test_empty_query_text(self, mocker):
        """Test that empty query text returns error response."""
        # Arrange
        mock_retriever = mocker.Mock()
        empty_text = ""

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=empty_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
        )

        # Assert
        assert result["query_text_processed"] == empty_text
        assert "Error" in result["header"]
        assert result["results"] == []
        assert result["original_query_assertion_status"] is None
        # Retriever should not be called for empty text
        mock_retriever.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_query_text(self, mocker):
        """Test that whitespace-only text returns error response."""
        # Arrange
        mock_retriever = mocker.Mock()
        whitespace_text = "   \n\t  "

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=whitespace_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
        )

        # Assert
        assert result["query_text_processed"] == whitespace_text
        assert "Error" in result["header"]
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_successful_retrieval_without_reranking(self, mocker):
        """Test successful retrieval without reranking."""
        # Arrange
        mock_retriever = mocker.Mock()
        query_text = "Patient has fever and cough"

        # Mock retriever to return results
        mock_retriever.query.return_value = {
            "ids": [["HP:0001945", "HP:0012735"]],
            "documents": [["Fever", "Cough"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001945", "label": "Fever"},
                    {"hpo_id": "HP:0012735", "label": "Cough"},
                ]
            ],
            "similarities": [[0.92, 0.85]],
        }

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
            detect_query_assertion=False,
        )

        # Assert
        assert result["query_text_processed"] == query_text.strip()
        assert len(result["results"]) == 2
        assert result["results"][0]["hpo_id"] == "HP:0001945"
        assert result["results"][0]["label"] == "Fever"
        assert result["results"][0]["similarity"] == 0.92
        assert result["results"][1]["hpo_id"] == "HP:0012735"
        assert result["results"][1]["similarity"] == 0.85

        # Verify retriever called with correct parameters
        mock_retriever.query.assert_called_once_with(
            text=query_text.strip(), n_results=5, include_similarities=True
        )

    @pytest.mark.asyncio
    async def test_successful_retrieval_with_reranking(self, mocker):
        """Test successful retrieval with cross-encoder reranking."""
        # Arrange
        mock_retriever = mocker.Mock()
        mock_cross_encoder = mocker.Mock()
        query_text = "Patient has fever"

        # Mock retriever to return results
        mock_retriever.query.return_value = {
            "ids": [["HP:0001945", "HP:0011134"]],
            "documents": [["Fever", "Tetralogy of Fallot"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001945", "label": "Fever"},
                    {"hpo_id": "HP:0011134", "label": "Tetralogy of Fallot"},
                ]
            ],
            "similarities": [[0.85, 0.82]],
        }

        # Mock cross-encoder to return scores (reverse order to test reranking)
        mock_cross_encoder.predict.return_value = [0.95, 0.45]

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=True,
            cross_encoder=mock_cross_encoder,
            rerank_count=10,
            detect_query_assertion=False,
        )

        # Assert
        assert len(result["results"]) == 2
        # After reranking, Fever (0.95 score) should be first
        assert result["results"][0]["hpo_id"] == "HP:0001945"
        assert result["results"][0]["cross_encoder_score"] == 0.95
        # Tetralogy (0.45 score) should be second
        assert result["results"][1]["hpo_id"] == "HP:0011134"
        assert result["results"][1]["cross_encoder_score"] == 0.45

        # Verify cross-encoder called with sentence pairs
        mock_cross_encoder.predict.assert_called_once()
        call_args = mock_cross_encoder.predict.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == (query_text.strip(), "Fever")
        assert call_args[1] == (query_text.strip(), "Tetralogy of Fallot")

    @pytest.mark.asyncio
    async def test_no_results_found(self, mocker):
        """Test handling when no results are found."""
        # Arrange
        mock_retriever = mocker.Mock()
        query_text = "Some obscure medical term"

        # Mock retriever to return empty results
        mock_retriever.query.return_value = {"ids": [[]], "documents": [[]]}

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
            detect_query_assertion=False,
        )

        # Assert
        assert result["query_text_processed"] == query_text.strip()
        assert "No HPO terms found" in result["header"]
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, mocker):
        """Test that results below similarity threshold are filtered out."""
        # Arrange
        mock_retriever = mocker.Mock()
        query_text = "Patient has fever"

        # Mock retriever with one result above and one below threshold
        mock_retriever.query.return_value = {
            "ids": [["HP:0001945", "HP:0012735"]],
            "documents": [["Fever", "Cough"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001945", "label": "Fever"},
                    {"hpo_id": "HP:0012735", "label": "Cough"},
                ]
            ],
            "similarities": [[0.92, 0.45]],  # Second below 0.5 threshold
        }

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
            detect_query_assertion=False,
        )

        # Assert - only Fever should be included
        assert len(result["results"]) == 1
        assert result["results"][0]["hpo_id"] == "HP:0001945"
        assert result["results"][0]["similarity"] == 0.92

    @pytest.mark.asyncio
    async def test_result_limiting(self, mocker):
        """Test that results are limited to num_results."""
        # Arrange
        mock_retriever = mocker.Mock()
        query_text = "Patient symptoms"

        # Mock retriever with more results than requested
        mock_retriever.query.return_value = {
            "ids": [["HP:0001", "HP:0002", "HP:0003", "HP:0004", "HP:0005"]],
            "documents": [["Term1", "Term2", "Term3", "Term4", "Term5"]],
            "metadatas": [
                [{"hpo_id": f"HP:000{i}", "label": f"Term{i}"} for i in range(1, 6)]
            ],
            "similarities": [[0.95, 0.90, 0.85, 0.80, 0.75]],
        }

        # Act - request only 3 results
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
            detect_query_assertion=False,
        )

        # Assert - should only return 3 results
        assert len(result["results"]) == 3
        assert result["results"][0]["hpo_id"] == "HP:0001"
        assert result["results"][2]["hpo_id"] == "HP:0003"

    @pytest.mark.asyncio
    async def test_reranker_disabled_when_cross_encoder_missing(self, mocker):
        """Test that reranking is disabled if cross_encoder not provided."""
        # Arrange
        mock_retriever = mocker.Mock()
        mock_logger = mocker.patch("phentrieve.retrieval.api_helpers.logger")
        query_text = "Patient has fever"

        mock_retriever.query.return_value = {
            "ids": [["HP:0001945"]],
            "documents": [["Fever"]],
            "metadatas": [[{"hpo_id": "HP:0001945", "label": "Fever"}]],
            "similarities": [[0.92]],
        }

        # Act - enable_reranker=True but cross_encoder=None
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=True,  # Requested
            cross_encoder=None,  # But not provided
            rerank_count=10,
            detect_query_assertion=False,
        )

        # Assert
        assert len(result["results"]) == 1
        # Should not have cross_encoder_score
        assert "cross_encoder_score" not in result["results"][0]
        # Should log warning
        mock_logger.warning.assert_called_once()
        assert "no cross_encoder provided" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_assertion_detection_enabled(self, mocker):
        """Test query assertion detection when enabled."""
        # Arrange
        mock_retriever = mocker.Mock()
        mock_assertion_detector_class = mocker.patch(
            "phentrieve.retrieval.api_helpers.CombinedAssertionDetector"
        )
        mock_assertion_detector = mocker.Mock()
        mock_assertion_detector_class.return_value = mock_assertion_detector

        # Mock assertion status enum
        mock_assertion_status = mocker.Mock()
        mock_assertion_status.value = "NEGATED"
        mock_assertion_detector.detect.return_value = (mock_assertion_status, {})

        query_text = "No fever reported"

        mock_retriever.query.return_value = {
            "ids": [["HP:0001945"]],
            "documents": [["Fever"]],
            "metadatas": [[{"hpo_id": "HP:0001945", "label": "Fever"}]],
            "similarities": [[0.92]],
        }

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
            detect_query_assertion=True,
            query_assertion_language="en",
            query_assertion_preference="dependency",
        )

        # Assert
        assert result["original_query_assertion_status"] == "NEGATED"
        # Verify assertion detector created with correct params
        mock_assertion_detector_class.assert_called_once_with(
            language="en", preference="dependency"
        )
        mock_assertion_detector.detect.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_assertion_detection_disabled(self, mocker):
        """Test that assertion detection is skipped when disabled."""
        # Arrange
        mock_retriever = mocker.Mock()
        mock_assertion_detector_class = mocker.patch(
            "phentrieve.retrieval.api_helpers.CombinedAssertionDetector"
        )
        query_text = "Patient has fever"

        mock_retriever.query.return_value = {
            "ids": [["HP:0001945"]],
            "documents": [["Fever"]],
            "metadatas": [[{"hpo_id": "HP:0001945", "label": "Fever"}]],
            "similarities": [[0.92]],
        }

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
            detect_query_assertion=False,  # Disabled
        )

        # Assert
        assert result["original_query_assertion_status"] is None
        # Assertion detector should not be created
        mock_assertion_detector_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_assertion_detection_error_handling(self, mocker):
        """Test error handling in assertion detection."""
        # Arrange
        mock_retriever = mocker.Mock()
        mock_assertion_detector_class = mocker.patch(
            "phentrieve.retrieval.api_helpers.CombinedAssertionDetector"
        )
        mock_assertion_detector = mocker.Mock()
        mock_assertion_detector_class.return_value = mock_assertion_detector
        mock_assertion_detector.detect.side_effect = Exception(
            "Assertion detection failed"
        )

        mock_logger = mocker.patch("phentrieve.retrieval.api_helpers.logger")

        query_text = "Patient has fever"

        mock_retriever.query.return_value = {
            "ids": [["HP:0001945"]],
            "documents": [["Fever"]],
            "metadatas": [[{"hpo_id": "HP:0001945", "label": "Fever"}]],
            "similarities": [[0.92]],
        }

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
            detect_query_assertion=True,
        )

        # Assert - should continue with None assertion status
        assert result["original_query_assertion_status"] is None
        # Should log warning
        mock_logger.warning.assert_called()
        assert "Error in assertion detection" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_reranking_error_handling(self, mocker):
        """Test error handling during reranking."""
        # Arrange
        mock_retriever = mocker.Mock()
        mock_cross_encoder = mocker.Mock()
        mock_cross_encoder.predict.side_effect = Exception("Reranking failed")
        # Error is now caught and logged in reranker.py, not api_helpers.py
        mock_reranker_logger = mocker.patch("phentrieve.retrieval.reranker.logger")

        query_text = "Patient has fever"

        mock_retriever.query.return_value = {
            "ids": [["HP:0001945"]],
            "documents": [["Fever"]],
            "metadatas": [[{"hpo_id": "HP:0001945", "label": "Fever"}]],
            "similarities": [[0.92]],
        }

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=True,
            cross_encoder=mock_cross_encoder,
            rerank_count=10,
            detect_query_assertion=False,
        )

        # Assert - should continue with dense retrieval results (fallback)
        assert len(result["results"]) == 1
        assert result["results"][0]["hpo_id"] == "HP:0001945"
        # Should log error in reranker module
        mock_reranker_logger.error.assert_called_once()
        assert (
            "Error during protected re-ranking"
            in mock_reranker_logger.error.call_args[0][0]
        )

    @pytest.mark.asyncio
    async def test_debug_logging_enabled(self, mocker):
        """Test that debug logging is enabled when debug=True."""
        # Arrange
        mock_retriever = mocker.Mock()
        mock_logger = mocker.patch("phentrieve.retrieval.api_helpers.logger")
        query_text = "Patient has fever"

        mock_retriever.query.return_value = {
            "ids": [["HP:0001945"]],
            "documents": [["Fever"]],
            "metadatas": [[{"hpo_id": "HP:0001945", "label": "Fever"}]],
            "similarities": [[0.92]],
        }

        # Act
        result = await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=False,
            cross_encoder=None,
            rerank_count=10,
            detect_query_assertion=False,
            debug=True,  # Debug enabled
        )

        # Assert
        assert len(result["results"]) == 1
        # Logger setLevel should be called
        mock_logger.setLevel.assert_called_once()
        # Debug message should be logged
        mock_logger.debug.assert_called()
        assert "Processing API query" in mock_logger.debug.call_args_list[0][0][0]

    @pytest.mark.asyncio
    async def test_rerank_count_passed_to_retriever(self, mocker):
        """Test that rerank_count is used when reranking enabled."""
        # Arrange
        mock_retriever = mocker.Mock()
        mock_cross_encoder = mocker.Mock()
        query_text = "Patient has fever"

        mock_retriever.query.return_value = {
            "ids": [["HP:0001945"]],
            "documents": [["Fever"]],
            "metadatas": [[{"hpo_id": "HP:0001945", "label": "Fever"}]],
            "similarities": [[0.92]],
        }
        mock_cross_encoder.predict.return_value = [0.95]

        # Act - execute retrieval (result not needed, testing side effects)
        await execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            enable_reranker=True,
            cross_encoder=mock_cross_encoder,
            rerank_count=20,  # Different from num_results
            detect_query_assertion=False,
        )

        # Assert
        # When reranking enabled, should request rerank_count results
        mock_retriever.query.assert_called_once_with(
            text=query_text.strip(), n_results=20, include_similarities=True
        )
