"""Unit tests for API helper functions.

Tests for the API-specific retrieval orchestration:
- execute_hpo_retrieval_for_api: Main API query processing function

Following best practices:
- Mock external dependencies (retriever, assertion detector)
- Test synchronous function behavior (callers run it in a threadpool)
- Comprehensive path coverage (empty query, no results, filtering)
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

    def test_empty_query_text(self, mocker):
        """Test that empty query text returns error response."""
        # Arrange
        mock_retriever = mocker.Mock()
        empty_text = ""

        # Act
        result = execute_hpo_retrieval_for_api(
            text=empty_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
        )

        # Assert
        assert result["query_text_processed"] == empty_text
        assert "Error" in result["header"]
        assert result["results"] == []
        assert result["original_query_assertion_status"] is None
        # Retriever should not be called for empty text
        mock_retriever.query.assert_not_called()

    def test_whitespace_only_query_text(self, mocker):
        """Test that whitespace-only text returns error response."""
        # Arrange
        mock_retriever = mocker.Mock()
        whitespace_text = "   \n\t  "

        # Act
        result = execute_hpo_retrieval_for_api(
            text=whitespace_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
        )

        # Assert
        assert result["query_text_processed"] == whitespace_text
        assert "Error" in result["header"]
        assert result["results"] == []

    def test_successful_dense_retrieval(self, mocker):
        """Test successful dense retrieval."""
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
        result = execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            detect_query_assertion=False,
            multi_vector=False,
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

    def test_debug_logs_do_not_include_raw_query_text(self, mocker, caplog):
        """Debug logging should report metadata without submitted text."""
        mock_retriever = mocker.Mock()
        query_text = "Herr Bernt Popp ist dumm"
        mock_retriever.query.return_value = {"ids": [[]], "documents": [[]]}

        with caplog.at_level("DEBUG", logger="phentrieve.retrieval.api_helpers"):
            execute_hpo_retrieval_for_api(
                text=query_text,
                language="en",
                retriever=mock_retriever,
                num_results=5,
                similarity_threshold=0.5,
                detect_query_assertion=False,
                multi_vector=False,
                debug=True,
            )

        assert query_text not in caplog.text
        assert f"text_chars={len(query_text)}" in caplog.text

    def test_no_results_found(self, mocker):
        """Test handling when no results are found."""
        # Arrange
        mock_retriever = mocker.Mock()
        query_text = "Some obscure medical term"

        # Mock retriever to return empty results
        mock_retriever.query.return_value = {"ids": [[]], "documents": [[]]}

        # Act
        result = execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            detect_query_assertion=False,
            multi_vector=False,
        )

        # Assert
        assert result["query_text_processed"] == query_text.strip()
        assert "No HPO terms found" in result["header"]
        assert result["results"] == []

    def test_multi_vector_request_rejects_single_vector_index(self, mocker):
        """A requested multi-vector query must not silently downgrade."""
        mock_retriever = mocker.Mock()
        mock_retriever.detect_index_type.return_value = "single_vector"

        with pytest.raises(RuntimeError, match="Multi-vector mode requested"):
            execute_hpo_retrieval_for_api(
                text="Patient has fever",
                language="en",
                retriever=mock_retriever,
                num_results=5,
                similarity_threshold=0.5,
                detect_query_assertion=False,
                multi_vector=True,
            )

        mock_retriever.query.assert_not_called()
        mock_retriever.query_multi_vector.assert_not_called()

    def test_similarity_threshold_filtering(self, mocker):
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
        result = execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            detect_query_assertion=False,
            multi_vector=False,
        )

        # Assert - only Fever should be included
        assert len(result["results"]) == 1
        assert result["results"][0]["hpo_id"] == "HP:0001945"
        assert result["results"][0]["similarity"] == 0.92

    def test_result_limiting(self, mocker):
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
        result = execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            detect_query_assertion=False,
            multi_vector=False,
        )

        # Assert - should only return 3 results
        assert len(result["results"]) == 3
        assert result["results"][0]["hpo_id"] == "HP:0001"
        assert result["results"][2]["hpo_id"] == "HP:0003"

    def test_assertion_detection_enabled(self, mocker):
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
        result = execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            detect_query_assertion=True,
            query_assertion_language="en",
            query_assertion_preference="dependency",
            multi_vector=False,
        )

        # Assert
        assert result["original_query_assertion_status"] == "NEGATED"
        # Verify assertion detector created with correct params
        mock_assertion_detector_class.assert_called_once_with(
            language="en", preference="dependency"
        )
        mock_assertion_detector.detect.assert_called_once_with(query_text)

    @pytest.mark.parametrize(
        ("language", "query", "expected_embedded"),
        [
            ("en", "no seizures", "seizures"),
            ("de", "keine Krampfanfälle", "krampfanfälle"),
        ],
    )
    def test_negated_query_embeds_text_without_negation_cue(
        self, mocker, language, query, expected_embedded
    ):
        """Negation cue is stripped before embedding; status stays negated."""
        mock_retriever = mocker.Mock()
        mock_retriever.query.return_value = {"ids": [[]], "documents": [[]]}

        result = execute_hpo_retrieval_for_api(
            text=query,
            language=language,
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            query_assertion_preference="keyword",
            multi_vector=False,
        )

        assert mock_retriever.query.call_args.kwargs["text"] == expected_embedded
        assert result["query_text_processed"] == expected_embedded
        assert result["original_query_assertion_status"] == "negated"

    @pytest.mark.parametrize(
        ("query", "expected_status", "expected_embedded"),
        [
            # Short foreign negations are still found without a language
            ("keine Krampfanfälle", "negated", "krampfanfälle"),
            ("pas de fièvre", "negated", "fièvre"),
            # English words must not be re-read as foreign cues (French "non")
            ("non ambulatory", "affirmed", "non ambulatory"),
            ("non-progressive ataxia", "affirmed", "non-progressive ataxia"),
            ("short stature", "affirmed", "short stature"),
        ],
    )
    def test_auto_language_assertion(
        self, mocker, query, expected_status, expected_embedded
    ):
        """With no language given, every language is tried; English wins its words."""
        mock_retriever = mocker.Mock()
        mock_retriever.query.return_value = {"ids": [[]], "documents": [[]]}

        result = execute_hpo_retrieval_for_api(
            text=query,
            language=None,
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            query_assertion_preference="keyword",
            multi_vector=False,
        )

        assert mock_retriever.query.call_args.kwargs["text"] == expected_embedded
        assert result["original_query_assertion_status"] == expected_status

    def test_assertion_detection_disabled(self, mocker):
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
        result = execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            detect_query_assertion=False,  # Disabled
            multi_vector=False,
        )

        # Assert
        assert result["original_query_assertion_status"] is None
        # Assertion detector should not be created
        mock_assertion_detector_class.assert_not_called()

    def test_assertion_detection_error_handling(self, mocker):
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
        result = execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            detect_query_assertion=True,
            multi_vector=False,
        )

        # Assert - should continue with None assertion status
        assert result["original_query_assertion_status"] is None
        # Should log warning
        mock_logger.warning.assert_called()
        warning_messages = [
            call_args.args[0] for call_args in mock_logger.warning.call_args_list
        ]
        assert any(
            "Error in assertion detection" in message for message in warning_messages
        )

    def test_debug_logging_enabled(self, mocker):
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
        result = execute_hpo_retrieval_for_api(
            text=query_text,
            language="en",
            retriever=mock_retriever,
            num_results=5,
            similarity_threshold=0.5,
            detect_query_assertion=False,
            debug=True,  # Debug enabled
            multi_vector=False,
        )

        # Assert
        assert len(result["results"]) == 1
        # Logger setLevel should be called
        mock_logger.setLevel.assert_called_once()
        # Debug message should be logged
        mock_logger.debug.assert_called()
        assert "Processing API query" in mock_logger.debug.call_args_list[0][0][0]
