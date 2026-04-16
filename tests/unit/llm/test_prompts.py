from __future__ import annotations

from pathlib import Path

import pytest

from phentrieve.llm.config import DEFAULT_TOOL_QUERY_RESULTS
from phentrieve.llm.prompts import loader
from phentrieve.llm.types import AnnotationMode, PostProcessingStep


@pytest.fixture(autouse=True)
def clear_prompt_loader_cache() -> None:
    loader.load_prompt_template.cache_clear()
    yield
    loader.load_prompt_template.cache_clear()


def test_get_mapping_prompt_loads_packaged_template() -> None:
    template = loader.get_mapping_prompt("fr")

    assert template.language == "en"
    assert template.version == "v4.0.0"
    assert "You map clinical phenotype phrases to HPO terms." in template.system_prompt


def test_load_prompt_template_prefers_user_override(monkeypatch, tmp_path) -> None:
    user_templates = tmp_path / "user"
    package_templates = tmp_path / "package"
    user_path = user_templates / "two_phase" / "en.yaml"
    package_path = package_templates / "two_phase" / "en.yaml"

    user_path.parent.mkdir(parents=True)
    package_path.parent.mkdir(parents=True)
    user_path.write_text(
        "version: 'user'\nsystem_prompt: user prompt\nuser_prompt_template: '{text}'\n",
        encoding="utf-8",
    )
    package_path.write_text(
        "version: 'package'\nsystem_prompt: package prompt\nuser_prompt_template: '{text}'\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(loader, "USER_TEMPLATES_DIR", user_templates)
    monkeypatch.setattr(loader, "PACKAGE_TEMPLATES_DIR", package_templates)

    template = loader.load_prompt_template(AnnotationMode.TWO_PHASE, "en")

    assert template.system_prompt == "user prompt"
    assert template.source_path == str(user_path.resolve())


def test_prompt_template_builds_messages_with_examples() -> None:
    template = loader.get_mapping_prompt("en")
    messages = template.get_messages(
        '{"phrase": "scoliosis", "grounded_context": {}, "candidates": []}',
        include_examples=True,
    )

    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert any(message["role"] == "assistant" for message in messages)


def test_list_available_prompts_includes_benchmark_families() -> None:
    available = loader.list_available_prompts()

    assert {"direct_text", "tool_guided", "two_phase", "postprocess"}.issubset(
        available
    )
    assert "en" in available["direct_text"]
    assert "en" in available["tool_guided"]
    assert "en" in available["two_phase"]
    assert "en" in available["postprocess"]
    assert "agentic_judge" not in available


def test_loader_supports_direct_and_postprocess_prompt_families() -> None:
    direct = loader.get_prompt(AnnotationMode.DIRECT, "de")
    validation = loader.load_prompt_template(PostProcessingStep.VALIDATION, "en")

    assert direct.mode == "direct_text"
    assert direct.language == "de"
    assert validation.mode == "postprocess"
    assert validation.language == "en"
    assert "validate" in validation.system_prompt.lower()


def test_tool_guided_prompts_render_tool_query_default_from_config() -> None:
    template = loader.get_prompt(AnnotationMode.TOOL_TEXT, "en")
    messages = template.get_messages("Patient has seizures.", include_examples=False)

    assert f"num_results={DEFAULT_TOOL_QUERY_RESULTS}" in messages[0]["content"]
    assert f"num_results={DEFAULT_TOOL_QUERY_RESULTS}" in messages[-1]["content"]


def test_tool_guided_prompt_sources_use_placeholder_not_hardcoded_default() -> None:
    templates_dir = Path(loader.PACKAGE_TEMPLATES_DIR) / "tool_guided"

    for template_path in templates_dir.glob("en*.yaml"):
        content = template_path.read_text(encoding="utf-8")
        assert "{tool_query_results}" in content
