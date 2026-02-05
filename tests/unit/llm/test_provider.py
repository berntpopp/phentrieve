"""Unit tests for LLM provider module."""

from unittest.mock import MagicMock, patch

import pytest


class TestLLMProviderImport:
    """Tests for LiteLLM import detection."""

    def test_get_litellm_import_error(self):
        """Test helpful error when LiteLLM is not installed."""
        # If litellm is not installed, _get_litellm should raise ImportError
        # with a helpful message
        try:
            from phentrieve.llm.provider import _get_litellm

            _get_litellm()
            # If it succeeds, litellm is installed - skip test
            pytest.skip("LiteLLM is installed, cannot test import error")
        except ImportError as e:
            error_msg = str(e)
            assert "LiteLLM is not installed" in error_msg
            assert "pip install litellm" in error_msg
            assert "phentrieve[llm]" in error_msg


class TestLLMProvider:
    """Tests for LLMProvider class with mocked LiteLLM."""

    @pytest.fixture
    def mock_litellm(self):
        """Create a mock LiteLLM module."""
        mock = MagicMock()
        mock.suppress_debug_info = False
        return mock

    @pytest.fixture
    def provider_with_mock(self, mock_litellm):
        """Create a provider with mocked LiteLLM."""
        with patch("phentrieve.llm.provider._get_litellm", return_value=mock_litellm):
            from phentrieve.llm.provider import LLMProvider

            return LLMProvider(model="github/gpt-4o")

    def test_extract_provider(self, provider_with_mock):
        """Test provider extraction from model string."""
        assert provider_with_mock.provider == "github"

    def test_extract_provider_no_prefix(self, mock_litellm):
        """Test provider defaults to openai when no prefix."""
        with patch("phentrieve.llm.provider._get_litellm", return_value=mock_litellm):
            from phentrieve.llm.provider import LLMProvider

            provider = LLMProvider(model="gpt-4o")
            assert provider.provider == "openai"

    def test_default_settings(self, provider_with_mock):
        """Test default provider settings."""
        assert provider_with_mock.temperature == 0.0
        assert provider_with_mock.max_tokens == 4096
        assert provider_with_mock.timeout == 120

    def test_custom_settings(self, mock_litellm):
        """Test custom provider settings."""
        with patch("phentrieve.llm.provider._get_litellm", return_value=mock_litellm):
            from phentrieve.llm.provider import LLMProvider

            provider = LLMProvider(
                model="gemini/gemini-1.5-pro",
                temperature=0.5,
                max_tokens=8192,
                timeout=300,
            )
            assert provider.temperature == 0.5
            assert provider.max_tokens == 8192
            assert provider.timeout == 300
            assert provider.provider == "gemini"

    def test_supports_tools_common_models(self, provider_with_mock):
        """Test that common models support tools."""
        assert provider_with_mock.supports_tools() is True

    def test_supports_tools_legacy_model(self, mock_litellm):
        """Test that legacy models don't support tools."""
        with patch("phentrieve.llm.provider._get_litellm", return_value=mock_litellm):
            from phentrieve.llm.provider import LLMProvider

            provider = LLMProvider(model="text-davinci-003")
            assert provider.supports_tools() is False


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    def test_execute_unknown_tool(self):
        """Test that unknown tools raise ValueError."""
        from phentrieve.llm.provider import ToolExecutor

        executor = ToolExecutor()
        with pytest.raises(ValueError) as exc_info:
            executor.execute("unknown_tool", {})
        assert "Unknown tool" in str(exc_info.value)


class TestAvailableModels:
    """Tests for get_available_models function."""

    def test_get_available_models(self):
        """Test getting available model presets."""
        from phentrieve.llm.provider import get_available_models

        models = get_available_models()

        # Check required providers
        assert "github" in models
        assert "gemini" in models
        assert "anthropic" in models
        assert "openai" in models
        assert "ollama" in models

        # Check specific models
        assert "github/gpt-4o" in models["github"]
        assert "gemini/gemini-1.5-pro" in models["gemini"]


class TestPhentrieveTools:
    """Tests for Phentrieve tool definitions."""

    def test_tool_definitions_structure(self):
        """Test that tool definitions have correct structure."""
        from phentrieve.llm.provider import PHENTRIEVE_TOOLS

        assert len(PHENTRIEVE_TOOLS) == 2

        # Check query_hpo_terms
        query_tool = PHENTRIEVE_TOOLS[0]
        assert query_tool["type"] == "function"
        assert query_tool["function"]["name"] == "query_hpo_terms"
        assert "parameters" in query_tool["function"]
        assert "query" in query_tool["function"]["parameters"]["properties"]

        # Check process_clinical_text
        process_tool = PHENTRIEVE_TOOLS[1]
        assert process_tool["type"] == "function"
        assert process_tool["function"]["name"] == "process_clinical_text"
        assert "text" in process_tool["function"]["parameters"]["properties"]
