"""Unit tests for query router helper functions."""

import pytest

from api.routers.query_router import (
    _resolve_query_language,
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
