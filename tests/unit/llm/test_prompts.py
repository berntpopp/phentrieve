"""Unit tests for prompt template loading."""

import pytest

from phentrieve.llm.types import AnnotationMode


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_render_user_prompt(self):
        """Test rendering user prompt template."""
        from phentrieve.llm.prompts.loader import PromptTemplate

        template = PromptTemplate(
            system_prompt="You are a helpful assistant.",
            user_prompt_template="Analyze this text: {text}",
        )

        rendered = template.render_user_prompt("Patient has seizures.")
        assert rendered == "Analyze this text: Patient has seizures."

    def test_render_with_kwargs(self):
        """Test rendering with additional kwargs."""
        from phentrieve.llm.prompts.loader import PromptTemplate

        template = PromptTemplate(
            system_prompt="System prompt",
            user_prompt_template="Text: {text}, Language: {language}",
        )

        rendered = template.render_user_prompt("Test", language="en")
        assert rendered == "Text: Test, Language: en"

    def test_get_messages_basic(self):
        """Test building basic message list."""
        from phentrieve.llm.prompts.loader import PromptTemplate

        template = PromptTemplate(
            system_prompt="System instruction",
            user_prompt_template="User: {text}",
        )

        messages = template.get_messages("Hello")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System instruction"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User: Hello"

    def test_get_messages_with_examples(self):
        """Test building messages with few-shot examples."""
        from phentrieve.llm.prompts.loader import PromptTemplate

        template = PromptTemplate(
            system_prompt="System",
            user_prompt_template="{text}",
            few_shot_examples=[
                {"input": "Example input", "output": "Example output"},
            ],
        )

        messages = template.get_messages("Test", include_examples=True)
        assert len(messages) == 4  # system + example pair + user
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Example input"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Example output"

    def test_get_messages_without_examples(self):
        """Test building messages without examples."""
        from phentrieve.llm.prompts.loader import PromptTemplate

        template = PromptTemplate(
            system_prompt="System",
            user_prompt_template="{text}",
            few_shot_examples=[
                {"input": "Example", "output": "Output"},
            ],
        )

        messages = template.get_messages("Test", include_examples=False)
        assert len(messages) == 2  # system + user only


class TestPromptLoading:
    """Tests for prompt template loading functions."""

    def test_load_direct_text_en_prompt(self):
        """Test loading English direct text prompt."""
        from phentrieve.llm.prompts import load_prompt_template

        template = load_prompt_template(AnnotationMode.DIRECT, "en")

        assert template.mode == "direct_text"
        assert template.language == "en"
        assert "HPO" in template.system_prompt
        assert "{text}" in template.user_prompt_template

    def test_load_direct_text_de_prompt(self):
        """Test loading German direct text prompt."""
        from phentrieve.llm.prompts import load_prompt_template

        template = load_prompt_template(AnnotationMode.DIRECT, "de")

        assert template.language == "de"
        assert "HPO" in template.system_prompt or "Phänotyp" in template.system_prompt

    def test_get_prompt_direct(self):
        """Test get_prompt convenience function for direct mode."""
        from phentrieve.llm.prompts import get_prompt

        template = get_prompt(AnnotationMode.DIRECT, "en")
        assert template.mode == "direct_text"

    def test_get_prompt_tool_term(self):
        """Test get_prompt for tool term mode."""
        from phentrieve.llm.prompts import get_prompt

        template = get_prompt(AnnotationMode.TOOL_TERM, "en")
        assert "query_hpo_terms" in template.system_prompt

    def test_get_prompt_tool_text(self):
        """Test get_prompt for tool text mode."""
        from phentrieve.llm.prompts import get_prompt

        template = get_prompt(AnnotationMode.TOOL_TEXT, "en")
        assert "process_clinical_text" in template.system_prompt

    def test_list_available_prompts(self):
        """Test listing available prompts."""
        from phentrieve.llm.prompts import list_available_prompts

        available = list_available_prompts()

        # Should have at least direct_text and tool_guided
        assert "direct_text" in available
        assert "tool_guided" in available

        # Should have language variants
        assert "en" in available["direct_text"]

    def test_load_nonexistent_prompt(self):
        """Test loading prompt that doesn't exist raises FileNotFoundError."""
        from phentrieve.llm.prompts import load_prompt_template

        with pytest.raises(FileNotFoundError):
            load_prompt_template("nonexistent_mode", "xx")


class TestModeToDir:
    """Tests for _mode_to_dir helper function."""

    def test_direct_mode(self):
        """Test converting DIRECT mode to directory."""
        from phentrieve.llm.prompts.loader import _mode_to_dir

        assert _mode_to_dir(AnnotationMode.DIRECT) == "direct_text"

    def test_tool_term_mode(self):
        """Test converting TOOL_TERM mode to directory."""
        from phentrieve.llm.prompts.loader import _mode_to_dir

        assert _mode_to_dir(AnnotationMode.TOOL_TERM) == "tool_guided"

    def test_tool_text_mode(self):
        """Test converting TOOL_TEXT mode to directory."""
        from phentrieve.llm.prompts.loader import _mode_to_dir

        assert _mode_to_dir(AnnotationMode.TOOL_TEXT) == "tool_guided"

    def test_string_passthrough(self):
        """Test that strings pass through unchanged."""
        from phentrieve.llm.prompts.loader import _mode_to_dir

        assert _mode_to_dir("custom_mode") == "custom_mode"
