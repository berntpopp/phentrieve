"""Unit tests for query router helper functions."""

import pytest

from api.routers.query_router import (
    _resolve_query_language,
    _resolve_reranker_model_name,
    _resolve_translation_directory,
)

pytestmark = pytest.mark.unit


class TestResolveQueryLanguage:
    """Test _resolve_query_language helper function."""

    def test_returns_provided_language_when_specified(self, mocker):
        """Test returns provided language without detection."""
        # Arrange
        mock_detect = mocker.patch("api.routers.query_router.detect_language")

        # Act
        result = _resolve_query_language(
            text="Patient has seizures",
            language="de",
            default_language="en",
        )

        # Assert
        assert result == "de"
        mock_detect.assert_not_called()  # Should not detect if provided

    def test_auto_detects_language_when_not_provided(self, mocker):
        """Test auto-detects language when not specified."""
        # Arrange
        mock_detect = mocker.patch(
            "api.routers.query_router.detect_language",
            return_value="fr",
        )

        # Act
        result = _resolve_query_language(
            text="Le patient a des crises",
            language=None,
            default_language="en",
        )

        # Assert
        assert result == "fr"
        mock_detect.assert_called_once_with(
            "Le patient a des crises", default_lang="en"
        )

    def test_falls_back_to_default_on_detection_error(self, mocker):
        """Test falls back to default language if detection fails."""
        # Arrange
        mock_detect = mocker.patch(
            "api.routers.query_router.detect_language",
            side_effect=Exception("Detection failed"),
        )

        # Act
        result = _resolve_query_language(
            text="Some text",
            language=None,
            default_language="en",
        )

        # Assert
        assert result == "en"
        mock_detect.assert_called_once()

    def test_uses_custom_default_language(self, mocker):
        """Test uses custom default language on detection failure."""
        # Arrange
        mocker.patch(
            "api.routers.query_router.detect_language",
            side_effect=Exception("Failed"),
        )

        # Act
        result = _resolve_query_language(
            text="Text",
            language=None,
            default_language="de",
        )

        # Assert
        assert result == "de"


class TestResolveRerankerModelName:
    """Test _resolve_reranker_model_name helper function."""

    def test_returns_none_when_reranker_disabled(self):
        """Test returns None when reranking is disabled."""
        # Act
        result = _resolve_reranker_model_name(
            enable_reranker=False,
            reranker_mode="cross-lingual",
            reranker_model="some-model",
        )

        # Assert
        assert result is None

    def test_returns_cross_lingual_model_for_cross_lingual_mode(self):
        """Test returns cross-lingual model in cross-lingual mode."""
        # Act
        result = _resolve_reranker_model_name(
            enable_reranker=True,
            reranker_mode="cross-lingual",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
            monolingual_reranker_model="should-not-use-this",
        )

        # Assert
        assert result == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def test_returns_monolingual_model_for_monolingual_mode(self):
        """Test returns monolingual model in monolingual mode."""
        # Act
        result = _resolve_reranker_model_name(
            enable_reranker=True,
            reranker_mode="monolingual",
            reranker_model="should-not-use-this",
            monolingual_reranker_model="castorini/monot5-base-msmarco",
        )

        # Assert
        assert result == "castorini/monot5-base-msmarco"

    def test_uses_default_cross_lingual_model_when_none_provided(self):
        """Test uses default cross-lingual model when not specified."""
        # Act
        result = _resolve_reranker_model_name(
            enable_reranker=True,
            reranker_mode="cross-lingual",
            reranker_model=None,
        )

        # Assert
        assert result is not None  # Should use DEFAULT_RERANKER_MODEL
        # Verify it's the expected default model (BAAI/bge-reranker-v2-m3)
        assert "bge-reranker" in result or "BAAI" in result

    def test_uses_default_monolingual_model_when_none_provided(self):
        """Test uses default monolingual model when not specified."""
        # Act
        result = _resolve_reranker_model_name(
            enable_reranker=True,
            reranker_mode="monolingual",
            monolingual_reranker_model=None,
        )

        # Assert
        assert result is not None  # Should use DEFAULT_MONOLINGUAL_RERANKER_MODEL


class TestResolveTranslationDirectory:
    """Test _resolve_translation_directory helper function."""

    def test_returns_none_when_reranker_disabled(self):
        """Test returns None when reranking is disabled."""
        # Act
        result = _resolve_translation_directory(
            enable_reranker=False,
            reranker_mode="monolingual",
            language="en",
        )

        # Assert
        assert result is None

    def test_returns_none_for_cross_lingual_mode(self):
        """Test returns None for cross-lingual mode (no translations needed)."""
        # Act
        result = _resolve_translation_directory(
            enable_reranker=True,
            reranker_mode="cross-lingual",
            language="en",
        )

        # Assert
        assert result is None

    def test_builds_path_from_language_code(self):
        """Test builds translation directory path from language code."""
        # Act
        result = _resolve_translation_directory(
            enable_reranker=True,
            reranker_mode="monolingual",
            language="de",
        )

        # Assert
        assert result is not None
        assert "hpo_translations" in result
        assert result.endswith("de")

    def test_uses_custom_translation_dir_name(self):
        """Test uses custom translation directory name when provided."""
        # Act
        result = _resolve_translation_directory(
            enable_reranker=True,
            reranker_mode="monolingual",
            language="en",
            translation_dir_name="custom_translations",
        )

        # Assert
        assert result is not None
        assert "custom_translations" in result
        assert "en" not in result  # Should use custom name, not language

    def test_warns_if_directory_not_found(self, mocker, tmp_path):
        """Test logs warning if translation directory doesn't exist."""
        # Arrange
        mock_logger = mocker.patch("api.routers.query_router.logger")

        # Act
        result = _resolve_translation_directory(
            enable_reranker=True,
            reranker_mode="monolingual",
            language="nonexistent_lang",
        )

        # Assert
        assert result is not None  # Still returns path
        mock_logger.warning.assert_called_once()
        assert "not found" in mock_logger.warning.call_args[0][0]


class TestQueryRequestValidation:
    """Test QueryRequest schema validation."""

    def test_allows_num_results_below_limit_with_details(self):
        """Test that num_results <= 20 is allowed when include_details=True."""
        from api.schemas.query_schemas import QueryRequest

        # Should not raise
        request = QueryRequest(
            text="test query",
            num_results=20,
            include_details=True,
        )
        assert request.num_results == 20
        assert request.include_details is True

    def test_allows_high_num_results_without_details(self):
        """Test that num_results > 20 is allowed when include_details=False."""
        from api.schemas.query_schemas import QueryRequest

        # Should not raise (within le=50 constraint)
        request = QueryRequest(
            text="test query",
            num_results=50,
            include_details=False,
        )
        assert request.num_results == 50
        assert request.include_details is False

    def test_rejects_high_num_results_with_details(self):
        """Test that num_results > 20 raises error when include_details=True."""
        from pydantic import ValidationError

        from api.schemas.query_schemas import QueryRequest

        # Should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                text="test query",
                num_results=50,
                include_details=True,
            )

        # Verify error message is clear
        error = str(exc_info.value)
        assert "Maximum 20 results allowed when include_details=true" in error
        assert "Requested: 50" in error

    def test_error_message_provides_guidance(self):
        """Test that error message provides actionable guidance."""
        from pydantic import ValidationError

        from api.schemas.query_schemas import QueryRequest

        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                text="test query",
                num_results=25,
                include_details=True,
            )

        error = str(exc_info.value)
        assert "Reduce num_results or disable include_details" in error
