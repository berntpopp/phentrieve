from __future__ import annotations

from pathlib import Path

import yaml

from phentrieve.llm.prompts.identity import build_prompt_bundle_identity


def _write_bundle(
    prompt_dir: Path, *, extraction_text: str = "Extract phenotypes."
) -> None:
    mode_dir = prompt_dir / "two_phase"
    mode_dir.mkdir(parents=True, exist_ok=True)
    prompts = {
        "en.yaml": {
            "version": "v3",
            "system_prompt": extraction_text,
            "user_prompt_template": "Clinical note: {text}",
            "few_shot_examples": [{"input": "short stature", "output": "HP:0004322"}],
        },
        "en_mapping.yaml": {
            "version": "v2",
            "system_prompt": "Map a phrase.",
            "user_prompt_template": "Phrase: {text}",
            "few_shot_examples": [],
        },
        "en_mapping_batch.yaml": {
            "version": "v4",
            "system_prompt": "Map several phrases.",
            "user_prompt_template": "Payload: {text}",
            "few_shot_examples": [],
        },
    }
    for filename, content in prompts.items():
        (mode_dir / filename).write_text(
            yaml.safe_dump(content, sort_keys=False), encoding="utf-8"
        )


def test_yaml_whitespace_and_key_order_do_not_change_component_hash(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    before = build_prompt_bundle_identity("two_phase", "en", tmp_path)

    extraction = tmp_path / "two_phase" / "en.yaml"
    extraction.write_text(
        "few_shot_examples:\n  - output: HP:0004322\n    input: short stature\n"
        "user_prompt_template: 'Clinical note: {text}'\n"
        "system_prompt: Extract phenotypes.\nversion: v3\n",
        encoding="utf-8",
    )

    after = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert before.components[0].sha256 == after.components[0].sha256
    assert before.sha256 == after.sha256


def test_prompt_text_change_changes_component_and_bundle_hash(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    before = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    _write_bundle(tmp_path, extraction_text="Extract phenotypes precisely.")

    after = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert before.components[0].sha256 != after.components[0].sha256
    assert before.sha256 != after.sha256


def test_two_phase_bundle_components_have_stable_order(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    identity = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert tuple(component.name for component in identity.components) == (
        "extraction",
        "mapping",
        "batch_mapping",
    )


def test_behavior_version_changes_bundle_hash(monkeypatch, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    before = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    monkeypatch.setattr("phentrieve.llm.prompts.identity.PROMPT_BEHAVIOR_VERSION", "2")
    after = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert before.sha256 != after.sha256


def test_unrelated_prompt_file_does_not_change_selected_bundle_hash(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    before = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    (tmp_path / "two_phase" / "de.yaml").write_text(
        "version: v99\nsystem_prompt: Unrelated\n", encoding="utf-8"
    )
    after = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert before.sha256 == after.sha256


def test_non_english_bundle_uses_localized_extraction_and_runtime_english_mapping(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    mode_dir = tmp_path / "two_phase"
    (mode_dir / "de.yaml").write_text(
        "version: de-extraction\nsystem_prompt: Deutsche Extraktion\n",
        encoding="utf-8",
    )
    (mode_dir / "de_mapping.yaml").write_text(
        "version: ignored-de-mapping\nsystem_prompt: Deutsche Zuordnung\n",
        encoding="utf-8",
    )
    (mode_dir / "de_mapping_batch.yaml").write_text(
        "version: ignored-de-batch\nsystem_prompt: Deutsche Sammelzuordnung\n",
        encoding="utf-8",
    )

    german = build_prompt_bundle_identity("two_phase", "de", tmp_path)
    english = build_prompt_bundle_identity("two_phase", "en", tmp_path)

    assert german.components[0].version == "de-extraction"
    assert german.components[0].sha256 != english.components[0].sha256
    assert tuple(component.version for component in german.components[1:]) == (
        "v2",
        "v4",
    )

    (mode_dir / "de_mapping.yaml").write_text(
        "version: changed-again\nsystem_prompt: Andere deutsche Zuordnung\n",
        encoding="utf-8",
    )
    after_localized_mapping_change = build_prompt_bundle_identity(
        "two_phase", "de", tmp_path
    )
    assert german.components[1:] == after_localized_mapping_change.components[1:]


def test_ignored_few_shot_metadata_does_not_change_hash(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    before = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    extraction = tmp_path / "two_phase" / "en.yaml"
    content = yaml.safe_load(extraction.read_text(encoding="utf-8"))
    content["few_shot_examples"][0]["note"] = "not emitted"
    extraction.write_text(yaml.safe_dump(content), encoding="utf-8")

    after = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert before.components[0].sha256 == after.components[0].sha256


def test_missing_and_empty_few_shot_fields_emit_same_identity(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    extraction = tmp_path / "two_phase" / "en.yaml"
    content = yaml.safe_load(extraction.read_text(encoding="utf-8"))
    content["few_shot_examples"] = [{}]
    extraction.write_text(yaml.safe_dump(content), encoding="utf-8")
    missing = build_prompt_bundle_identity("two_phase", "en", tmp_path)

    content["few_shot_examples"] = [{"input": "", "output": ""}]
    extraction.write_text(yaml.safe_dump(content), encoding="utf-8")
    explicit_empty = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert missing.components[0].sha256 == explicit_empty.components[0].sha256


def test_emitted_few_shot_content_changes_component_hash(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    before = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    extraction = tmp_path / "two_phase" / "en.yaml"
    content = yaml.safe_load(extraction.read_text(encoding="utf-8"))
    content["few_shot_examples"][0]["output"] = "HP:9999999"
    extraction.write_text(yaml.safe_dump(content), encoding="utf-8")

    after = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert before.components[0].sha256 != after.components[0].sha256


def test_component_version_changes_component_and_bundle_hash(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    before = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    extraction = tmp_path / "two_phase" / "en.yaml"
    content = yaml.safe_load(extraction.read_text(encoding="utf-8"))
    content["version"] = "v4"
    extraction.write_text(yaml.safe_dump(content), encoding="utf-8")

    after = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert before.components[0].sha256 != after.components[0].sha256
    assert before.sha256 != after.sha256


def test_equivalent_newline_styles_do_not_change_hash(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    extraction = tmp_path / "two_phase" / "en.yaml"
    content = yaml.safe_load(extraction.read_text(encoding="utf-8"))
    content["system_prompt"] = "First\r\nSecond\rThird"
    extraction.write_text(yaml.safe_dump(content), encoding="utf-8")
    mixed = build_prompt_bundle_identity("two_phase", "en", tmp_path)

    content["system_prompt"] = "First\nSecond\nThird"
    extraction.write_text(yaml.safe_dump(content), encoding="utf-8")
    lf = build_prompt_bundle_identity("two_phase", "en", tmp_path)
    assert mixed.components[0].sha256 == lf.components[0].sha256


def test_extraction_identity_keeps_language_placeholder_literal_like_runtime(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    extraction = tmp_path / "two_phase" / "en.yaml"
    content = yaml.safe_load(extraction.read_text(encoding="utf-8"))
    content["system_prompt"] = "Extract in {language}"
    content["user_prompt_template"] = (
        "Document: {text}; chunks: {chunk_index}; language: {language}"
    )
    extraction.write_text(yaml.safe_dump(content), encoding="utf-8")
    runtime_literal = build_prompt_bundle_identity("two_phase", "en", tmp_path)

    content["system_prompt"] = "Extract in en"
    content["user_prompt_template"] = (
        "Document: {text}; chunks: {chunk_index}; language: en"
    )
    extraction.write_text(yaml.safe_dump(content), encoding="utf-8")
    explicitly_substituted = build_prompt_bundle_identity("two_phase", "en", tmp_path)

    assert (
        runtime_literal.components[0].sha256
        != explicitly_substituted.components[0].sha256
    )
