from __future__ import annotations

from pathlib import Path

import pytest

from phentrieve.llm import pipeline as pipeline_module
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

    assert template.language == "fr"
    assert template.version == "v4.1.0"
    assert "You map clinical phenotype phrases to HPO terms." in template.system_prompt
    assert template.source_path.endswith("two_phase/en_mapping.yaml")


def test_get_batch_mapping_prompt_uses_shared_english_template_with_requested_language() -> (
    None
):
    template = loader.get_batch_mapping_prompt("de")

    assert template.language == "de"
    assert template.version == "v4.1.0"
    assert template.source_path.endswith("two_phase/en_mapping_batch.yaml")


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


def test_mapping_prompt_prefix_stays_stable_before_variable_context() -> None:
    template = loader.get_mapping_prompt("en")

    first_prompt = template.render_user_prompt('{"phrase":"alpha"}')
    second_prompt = template.render_user_prompt('{"phrase":"beta"}')

    expected_prefix = (
        "Map the following JSON payload to the best HPO candidate.\n"
        "Return JSON only.\n\n"
        "Payload:\n"
    )

    assert first_prompt.startswith(expected_prefix)
    assert second_prompt.startswith(expected_prefix)
    assert first_prompt.index('{"phrase":"alpha"}') > len(expected_prefix) - 1
    assert second_prompt.index('{"phrase":"beta"}') > len(expected_prefix) - 1
    assert first_prompt[: len(expected_prefix)] == second_prompt[: len(expected_prefix)]


def test_mapping_prompt_renders_requested_language_in_system_prompt() -> None:
    template = loader.get_mapping_prompt("de")

    rendered = template.render_system_prompt(language=template.language)

    assert "de" in rendered
    assert "source language" in rendered
    assert "retrieval_score" in rendered


def test_batch_mapping_prompt_prefix_stays_stable_before_variable_context() -> None:
    template = loader.get_batch_mapping_prompt("en")

    first_prompt = template.render_user_prompt('{"items":[{"phrase":"alpha"}]}')
    second_prompt = template.render_user_prompt('{"items":[{"phrase":"beta"}]}')

    expected_prefix = (
        "Map the following JSON payload to the best HPO candidate.\n"
        "Return JSON only.\n\n"
        "Payload:\n"
    )

    assert first_prompt.startswith(expected_prefix)
    assert second_prompt.startswith(expected_prefix)
    assert (
        first_prompt.index('{"items":[{"phrase":"alpha"}]}') > len(expected_prefix) - 1
    )
    assert (
        second_prompt.index('{"items":[{"phrase":"beta"}]}') > len(expected_prefix) - 1
    )
    assert first_prompt[: len(expected_prefix)] == second_prompt[: len(expected_prefix)]


def test_batch_mapping_prompt_requires_item_id_based_selections() -> None:
    template = loader.get_batch_mapping_prompt("en")

    assert "- item_id" in template.system_prompt
    assert "original item_id" in template.system_prompt
    assert '{"mappings": [{"item_id": "item_1", "hpo_id": "HP:XXXXXXX"}]}' in (
        template.system_prompt
    )
    assert '"item_id": "item_1"' in template.few_shot_examples[0]["input"]
    assert '"item_id": "item_1"' in template.few_shot_examples[0]["output"]


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


def test_prompt_template_does_not_rewrite_placeholder_text_inside_payload() -> None:
    template = loader.PromptTemplate(
        system_prompt="System {tool_query_results}",
        user_prompt_template="Payload:\n{text}",
    )

    rendered = template.render_user_prompt(
        "Patient literal {tool_query_results} and {text} tokens."
    )

    assert (
        rendered == "Payload:\nPatient literal {tool_query_results} and {text} tokens."
    )


def test_grounded_phase1_prompt_uses_chunk_index_without_repeating_full_text() -> None:
    template = loader.get_prompt(AnnotationMode.TWO_PHASE, "en")

    rendered = pipeline_module._render_phase1_user_prompt(
        extraction_prompt=template,
        text="FULL NOTE SENTINEL",
        grounded_chunks=[
            {"chunk_id": 1, "text": "recurrent seizures"},
            {"chunk_id": 2, "text": "no skeletal anomalies"},
        ],
    )

    assert "Chunk index:" in rendered
    assert "chunk_id=1" in rendered
    assert "FULL NOTE SENTINEL" not in rendered


def test_legacy_phase1_prompt_keeps_full_text_when_no_grounding() -> None:
    template = loader.get_prompt(AnnotationMode.TWO_PHASE, "en")

    rendered = pipeline_module._render_phase1_user_prompt(
        extraction_prompt=template,
        text="FULL NOTE SENTINEL",
        grounded_chunks=[],
    )

    assert "FULL NOTE SENTINEL" in rendered
    assert "Chunk index:" in rendered
    assert "[]" in rendered
