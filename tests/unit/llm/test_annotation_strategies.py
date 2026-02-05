"""Unit tests for annotation strategies."""

from unittest.mock import MagicMock, patch

import pytest

from phentrieve.llm.types import (
    AnnotationMode,
    AssertionStatus,
    LLMResponse,
)


class TestDirectTextStrategy:
    """Tests for DirectTextStrategy."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.model = "github/gpt-4o"
        provider.temperature = 0.0
        return provider

    @pytest.fixture
    def strategy(self, mock_provider):
        """Create a DirectTextStrategy with mock provider."""
        from phentrieve.llm.annotation.direct_text import DirectTextStrategy

        return DirectTextStrategy(mock_provider)

    def test_mode_is_direct(self, strategy):
        """Test that strategy has correct mode."""
        assert strategy.mode == AnnotationMode.DIRECT

    def test_parse_json_response(self, strategy):
        """Test parsing valid JSON response."""
        response_text = """
        ```json
        {
            "annotations": [
                {
                    "hpo_id": "HP:0001250",
                    "term_name": "Seizure",
                    "assertion": "affirmed",
                    "confidence": 0.95,
                    "evidence_text": "recurrent seizures"
                }
            ]
        }
        ```
        """
        annotations = strategy._parse_response(response_text)
        assert len(annotations) == 1
        assert annotations[0].hpo_id == "HP:0001250"
        assert annotations[0].term_name == "Seizure"
        assert annotations[0].assertion == AssertionStatus.AFFIRMED
        assert annotations[0].confidence == 0.95

    def test_parse_raw_json_response(self, strategy):
        """Test parsing JSON without code block."""
        response_text = (
            '{"annotations": [{"hpo_id": "HP:0001250", "term_name": "Seizure"}]}'
        )
        annotations = strategy._parse_response(response_text)
        assert len(annotations) == 1
        assert annotations[0].hpo_id == "HP:0001250"

    def test_parse_negated_assertion(self, strategy):
        """Test parsing negated assertion."""
        response_text = """{"annotations": [
            {"hpo_id": "HP:0001627", "term_name": "Cardiac", "assertion": "negated"}
        ]}"""
        annotations = strategy._parse_response(response_text)
        assert annotations[0].assertion == AssertionStatus.NEGATED

    def test_parse_uncertain_assertion(self, strategy):
        """Test parsing uncertain assertion."""
        response_text = """{"annotations": [
            {"hpo_id": "HP:0001250", "term_name": "Seizure", "assertion": "possible"}
        ]}"""
        annotations = strategy._parse_response(response_text)
        assert annotations[0].assertion == AssertionStatus.UNCERTAIN

    def test_normalize_hpo_id_formats(self, strategy):
        """Test normalization of various HPO ID formats."""
        # Standard format
        assert strategy._normalize_hpo_id("HP:0001250") == "HP:0001250"

        # Lowercase
        assert strategy._normalize_hpo_id("hp:0001250") == "HP:0001250"

        # Without prefix
        assert strategy._normalize_hpo_id("0001250") == "HP:0001250"

        # Short number (needs padding)
        assert strategy._normalize_hpo_id("HP:1250") == "HP:0001250"

        # Invalid format
        assert strategy._normalize_hpo_id("invalid") is None

    def test_parse_invalid_response(self, strategy):
        """Test parsing invalid response returns empty list."""
        annotations = strategy._parse_response("This is not JSON")
        assert annotations == []

    def test_parse_response_missing_annotations(self, strategy):
        """Test parsing response without annotations key."""
        annotations = strategy._parse_response('{"other": "data"}')
        assert annotations == []

    def test_annotate_calls_provider(self, strategy, mock_provider):
        """Test that annotate calls the provider."""
        mock_response = LLMResponse(
            content='{"annotations": []}',
            model="github/gpt-4o",
            provider="github",
        )
        mock_provider.complete.return_value = mock_response

        with patch(
            "phentrieve.llm.annotation.direct_text.get_prompt"
        ) as mock_get_prompt:
            mock_template = MagicMock()
            mock_template.get_messages.return_value = [
                {"role": "user", "content": "test"}
            ]
            mock_template.version = "v1.0.0"
            mock_get_prompt.return_value = mock_template

            result = strategy.annotate("Patient has seizures.", validate_hpo_ids=False)

            mock_provider.complete.assert_called_once()
            assert result.mode == AnnotationMode.DIRECT
            assert result.model == "github/gpt-4o"


class TestToolGuidedStrategy:
    """Tests for ToolGuidedStrategy."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.model = "github/gpt-4o"
        provider.temperature = 0.0
        provider.supports_tools.return_value = True
        return provider

    def test_invalid_mode_raises(self, mock_provider):
        """Test that invalid mode raises ValueError."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy

        with pytest.raises(ValueError) as exc_info:
            ToolGuidedStrategy(mock_provider, mode=AnnotationMode.DIRECT)
        assert "Invalid mode" in str(exc_info.value)

    def test_tool_term_mode(self, mock_provider):
        """Test creating strategy with TOOL_TERM mode."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy

        strategy = ToolGuidedStrategy(mock_provider, mode=AnnotationMode.TOOL_TERM)
        assert strategy.mode == AnnotationMode.TOOL_TERM

    def test_tool_text_mode(self, mock_provider):
        """Test creating strategy with TOOL_TEXT mode."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy

        strategy = ToolGuidedStrategy(mock_provider, mode=AnnotationMode.TOOL_TEXT)
        assert strategy.mode == AnnotationMode.TOOL_TEXT

    def test_default_mode_is_tool_text(self, mock_provider):
        """Test that default mode is TOOL_TEXT."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy

        strategy = ToolGuidedStrategy(mock_provider)
        assert strategy.mode == AnnotationMode.TOOL_TEXT

    def test_parse_response_same_as_direct(self, mock_provider):
        """Test that response parsing works like DirectTextStrategy."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy

        strategy = ToolGuidedStrategy(mock_provider)

        response_text = """{"annotations": [
            {"hpo_id": "HP:0001250", "term_name": "Seizure", "assertion": "affirmed"}
        ]}"""
        annotations = strategy._parse_response(response_text)
        assert len(annotations) == 1
        assert annotations[0].hpo_id == "HP:0001250"
