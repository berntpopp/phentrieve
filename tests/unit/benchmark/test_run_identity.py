from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from phentrieve.benchmark import data_loader
from phentrieve.benchmark.run_identity import (
    DatasetIdentity,
    RetrievalAssetIdentity,
    behavioral_base_url_sha256,
    build_dataset_identity,
    build_run_fingerprints,
    load_retrieval_asset_identity,
    validate_evaluation_hpo_version,
)
from phentrieve.data_processing.bundle_manifest import (
    compute_directory_checksum,
    compute_file_checksum,
)
from phentrieve.llm.prompts.identity import (
    PromptBundleIdentity,
    PromptComponentIdentity,
)


def _write_dataset(path, documents) -> None:
    path.write_text(json.dumps({"documents": documents}), encoding="utf-8")


def _document(identifier: str, text: str, hpo_ids: list[str]) -> dict[str, object]:
    return {
        "id": identifier,
        "text": text,
        "gold_hpo_terms": [
            {"id": hpo_id, "assertion": "PRESENT"} for hpo_id in hpo_ids
        ],
    }


def _write_installed_bundle(tmp_path: Path) -> Path:
    database = tmp_path / "hpo_data.db"
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    database.write_bytes(b"ontology")
    (indexes / "vectors.bin").write_bytes(b"vectors")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "hpo_version": "v2026-06-23",
                "model": {
                    "name": "BAAI/bge-m3",
                    "slug": "bge-m3",
                    "dimension": 1024,
                    "multi_vector": True,
                    "revision": "model-commit",
                    "trust_remote_code": True,
                    "code_revision": "code-commit",
                },
                "checksums": {
                    "hpo_data.db": compute_file_checksum(database),
                    "indexes/": compute_directory_checksum(indexes),
                },
            }
        ),
        encoding="utf-8",
    )
    return indexes


def test_json_formatting_only_changes_source_hash(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    payload = {"documents": [_document("case-1", "Short stature", ["HP:0004322"])]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = build_dataset_identity(path, dataset="all")

    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    after = build_dataset_identity(path, dataset="all")

    assert after.source_sha256 != before.source_sha256
    assert after.input_sha256 == before.input_sha256
    assert after.gold_sha256 == before.gold_sha256
    assert after.document_ids_sha256 == before.document_ids_sha256


def test_dataset_identity_versions_and_hashes_effective_projection(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(path, [_document("case-1", "Short stature", ["HP:0004322"])])

    present = build_dataset_identity(
        path,
        dataset="all",
        projection={"present": "PRESENT", "negated": None},
    )
    assertion_aware = build_dataset_identity(
        path,
        dataset="all",
        projection={"present": "PRESENT", "negated": "ABSENT"},
    )

    assert present.schema_version == "phentrieve-dataset-identity/v2"
    assert present.projection_sha256 != assertion_aware.projection_sha256
    _, prompt, asset, model = _fingerprint_identities()
    assert (
        build_run_fingerprints(present, prompt, model, asset).scoring_sha256
        != build_run_fingerprints(assertion_aware, prompt, model, asset).scoring_sha256
    )


def test_text_change_changes_only_input_hash(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(path, [_document("case-1", "Short stature", ["HP:0004322"])])
    before = build_dataset_identity(path, dataset="all")
    _write_dataset(path, [_document("case-1", "Tall stature", ["HP:0004322"])])
    changed_text = build_dataset_identity(path, dataset="all")

    assert changed_text.input_sha256 != before.input_sha256
    assert changed_text.gold_sha256 == before.gold_sha256


def test_gold_change_changes_only_gold_hash(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(path, [_document("case-1", "Short stature", ["HP:0004322"])])
    before = build_dataset_identity(path, dataset="all")
    _write_dataset(path, [_document("case-1", "Short stature", ["HP:0000098"])])
    changed_gold = build_dataset_identity(path, dataset="all")

    assert changed_gold.input_sha256 == before.input_sha256
    assert changed_gold.gold_sha256 != before.gold_sha256


def test_document_order_does_not_change_canonical_hashes(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    documents = [
        _document("case-b", "Beta", ["HP:2", "HP:1"]),
        _document("case-a", "Alpha", ["HP:3"]),
    ]
    _write_dataset(path, documents)
    before = build_dataset_identity(path, dataset="all")
    _write_dataset(path, list(reversed(documents)))
    after = build_dataset_identity(path, dataset="all")

    assert after.input_sha256 == before.input_sha256
    assert after.gold_sha256 == before.gold_sha256
    assert after.document_ids_sha256 == before.document_ids_sha256
    assert after.execution_order_sha256 != before.execution_order_sha256


def test_gold_annotation_duplicates_and_order_do_not_change_gold_hash(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(path, [_document("case-a", "Alpha", ["HP:2", "HP:1", "HP:2"])])
    before = build_dataset_identity(path, dataset="all")
    _write_dataset(path, [_document("case-a", "Alpha", ["HP:1", "HP:2"])])
    after = build_dataset_identity(path, dataset="all")

    assert after.gold_sha256 == before.gold_sha256


def test_requested_document_order_changes_execution_identity(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(
        path,
        [_document("case-a", "Alpha", ["HP:1"]), _document("case-b", "Beta", ["HP:2"])],
    )
    first = build_dataset_identity(
        path, dataset="all", document_ids=["case-a", "case-b"]
    )
    second = build_dataset_identity(
        path, dataset="all", document_ids=["case-b", "case-a"]
    )
    assert first.input_sha256 == second.input_sha256
    assert first.execution_order_sha256 != second.execution_order_sha256
    _, prompt, asset, model = _fingerprint_identities()
    assert (
        build_run_fingerprints(first, prompt, model, asset).execution_sha256
        != build_run_fingerprints(second, prompt, model, asset).execution_sha256
    )


def test_non_present_assertion_change_changes_scoring_identity(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    document = _document("case-a", "Alpha", ["HP:1"])
    document["gold_hpo_terms"] = [{"id": "HP:1", "assertion": "NEGATED"}]
    _write_dataset(path, [document])
    before = build_dataset_identity(path, dataset="all")
    document["gold_hpo_terms"] = [{"id": "HP:1", "assertion": "FAMILY_HISTORY"}]
    _write_dataset(path, [document])
    after = build_dataset_identity(path, dataset="all")
    assert before.gold_sha256 == after.gold_sha256
    assert before.assertion_gold_sha256 != after.assertion_gold_sha256
    _, prompt, asset, model = _fingerprint_identities()
    assert (
        build_run_fingerprints(before, prompt, model, asset).scoring_sha256
        != build_run_fingerprints(after, prompt, model, asset).scoring_sha256
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "PRESENT"),
        ("  ", "PRESENT"),
        (" affirmed ", "PRESENT"),
        ("ABSENT", "ABSENT"),
        (" negated ", "ABSENT"),
        ("Family_History", "FAMILY_HISTORY"),
    ],
)
def test_gold_assertions_use_one_canonical_normalizer(raw, expected) -> None:
    normalize = data_loader.normalize_benchmark_assertion

    assert normalize(raw, reject_unknown=True) == expected


def test_unknown_explicit_gold_assertion_is_rejected() -> None:
    normalize = data_loader.normalize_benchmark_assertion

    with pytest.raises(ValueError, match="Unknown benchmark assertion"):
        normalize("historical", reject_unknown=True)


def test_json_gold_terms_are_normalized_for_dicts_and_tuples() -> None:
    assert data_loader.parse_gold_terms(
        [
            {"id": "HP:1", "assertion": " absent "},
            ["HP:2", "NEGATED"],
            "HP:3",
        ]
    ) == [("HP:1", "ABSENT"), ("HP:2", "ABSENT"), ("HP:3", "PRESENT")]


def test_endpoint_behavior_hash_preserves_path_and_query_without_credentials() -> None:
    base = behavioral_base_url_sha256("https://user:one@example.test/api?v=1")
    assert base == behavioral_base_url_sha256("https://other:two@example.test/api?v=1")
    assert base != behavioral_base_url_sha256(
        "https://user:one@example.test/api/v2?v=1"
    )
    assert base != behavioral_base_url_sha256("https://user:one@example.test/api?v=2")


def test_endpoint_behavior_hash_redacts_sensitive_query_values() -> None:
    first = behavioral_base_url_sha256(
        "https://example.test/v1?api_key=short&api-version=2026-01-01"
    )
    second = behavioral_base_url_sha256(
        "https://example.test/v1?api_key=different&api-version=2026-01-01"
    )
    changed_behavior = behavioral_base_url_sha256(
        "https://example.test/v1?api_key=short&api-version=2027-01-01"
    )

    assert first == second
    assert first != changed_behavior


def test_sanitized_endpoint_does_not_persist_query_credentials() -> None:
    from phentrieve.benchmark.run_identity import sanitize_behavioral_base_url

    sanitized = sanitize_behavioral_base_url(
        "https://user:password@example.test/v1?token=secret&api-version=1"
    )

    assert sanitized == "https://example.test/v1?api-version=1&token=REDACTED"


def test_dataset_identity_does_not_import_chromadb(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(path, [_document("case-a", "Alpha", ["HP:1"])])
    repository_root = Path(__file__).resolve().parents[3]
    code = f"""
import builtins
from pathlib import Path

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'chromadb' or name.startswith('chromadb.'):
        raise AssertionError('dataset identity imported chromadb')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import

import phentrieve.benchmark.run_identity as run_identity
assert Path(run_identity.__file__).resolve().is_relative_to(Path({str(repository_root)!r}))
build_dataset_identity = run_identity.build_dataset_identity
identity = build_dataset_identity(Path({str(path)!r}), dataset='all')
assert len(identity.input_sha256) == 64
"""
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repository_root), existing_pythonpath) if part
    )

    result = subprocess.run(  # noqa: S603 - fixed interpreter and test-only script
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_selected_document_ids_filter_and_identify_dataset(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(
        path,
        [
            _document("case-a", "Same text", ["HP:1"]),
            _document("case-b", "Same text", ["HP:1"]),
        ],
    )

    first = build_dataset_identity(path, dataset="all", document_ids=["case-a"])
    second = build_dataset_identity(path, dataset="all", document_ids=["case-b"])

    assert first.input_sha256 != second.input_sha256
    assert first.gold_sha256 != second.gold_sha256
    assert first.document_ids_sha256 != second.document_ids_sha256
    assert first.projection == "positive_hpo_present_v1"
    assert first.excluded_document_ids == ("case-b",)
    with pytest.raises(ValueError, match="Unknown requested document IDs: missing"):
        build_dataset_identity(path, dataset="all", document_ids=["missing"])


def test_duplicate_loaded_and_requested_document_ids_are_rejected(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(
        path,
        [
            _document("duplicate", "First", ["HP:1"]),
            _document("duplicate", "Second", ["HP:2"]),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate loaded document IDs: duplicate"):
        build_dataset_identity(path, dataset="all")

    _write_dataset(path, [_document("case-a", "First", ["HP:1"])])
    with pytest.raises(ValueError, match="Duplicate requested document IDs: case-a"):
        build_dataset_identity(path, dataset="all", document_ids=["case-a", "case-a"])


def test_directory_source_hash_is_scoped_to_selected_dataset(tmp_path) -> None:
    csc_dir = tmp_path / "CSC" / "annotations"
    gsc_dir = tmp_path / "GSC" / "annotations"
    csc_dir.mkdir(parents=True)
    gsc_dir.mkdir(parents=True)
    csc_path = csc_dir / "CSC_1.json"
    gsc_path = gsc_dir / "GSC_1.json"
    csc_payload = {
        "doc_id": "CSC_1",
        "full_text": "Short stature.",
        "annotations": [{"hpo_id": "HP:0004322", "assertion_status": "affirmed"}],
    }
    csc_path.write_text(json.dumps(csc_payload), encoding="utf-8")
    gsc_path.write_text(
        json.dumps({"doc_id": "GSC_1", "full_text": "Other.", "annotations": []}),
        encoding="utf-8",
    )
    before = build_dataset_identity(tmp_path, dataset="CSC")

    gsc_path.write_text('{\n  "unrelated": true\n}', encoding="utf-8")
    unrelated_changed = build_dataset_identity(tmp_path, dataset="CSC")
    csc_path.write_text(json.dumps(csc_payload, indent=4), encoding="utf-8")
    selected_changed = build_dataset_identity(tmp_path, dataset="CSC")

    assert unrelated_changed.source_sha256 == before.source_sha256
    assert selected_changed.source_sha256 != before.source_sha256
    assert selected_changed.input_sha256 == before.input_sha256
    assert selected_changed.gold_sha256 == before.gold_sha256


def test_asset_identity_reads_installed_bundle_manifest(tmp_path) -> None:
    _write_installed_bundle(tmp_path)

    identity = load_retrieval_asset_identity(tmp_path)

    assert identity.asset_type == "multi_vector"
    assert identity.embedding_model == "BAAI/bge-m3"
    assert identity.hpo_version == "v2026-06-23"
    assert len(identity.manifest_sha256) == 64
    assert len(identity.content_sha256) == 64
    assert identity.model_revision == "model-commit"
    assert identity.trust_remote_code is True
    assert identity.code_revision == "code-commit"


def test_asset_identity_rejects_incomplete_bundle_inventory(tmp_path) -> None:
    _write_installed_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"].pop("indexes/")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="required checksum entries"):
        load_retrieval_asset_identity(tmp_path)


def test_asset_identity_detects_tampered_bundle_content(tmp_path) -> None:
    _write_installed_bundle(tmp_path)
    (tmp_path / "indexes" / "vectors.bin").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Checksum verification failed.*indexes/"):
        load_retrieval_asset_identity(tmp_path)


def test_asset_identity_rejects_checksum_path_traversal(tmp_path) -> None:
    _write_installed_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["../outside.txt"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe checksum path"):
        load_retrieval_asset_identity(tmp_path)


@pytest.mark.parametrize(
    "hpo_value", [None, "", "   ", pytest.param("missing", id="missing")]
)
def test_asset_identity_requires_hpo_version_provenance(tmp_path, hpo_value) -> None:
    manifest = {
        "model": {
            "name": "BAAI/bge-m3",
            "slug": "bge-m3",
            "dimension": 1024,
            "multi_vector": False,
        }
    }
    if hpo_value != "missing":
        manifest["hpo_version"] = hpo_value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="no valid HPO version provenance"):
        load_retrieval_asset_identity(tmp_path)


def test_evaluation_hpo_must_match_asset_hpo() -> None:
    asset = RetrievalAssetIdentity(
        asset_type="single_vector",
        embedding_model="BAAI/bge-m3",
        hpo_version="v2026-06-23",
        manifest_sha256="a" * 64,
    )
    with pytest.raises(
        ValueError,
        match=(
            "Evaluation HPO version 'v2025-03-03' does not match retrieval asset "
            "HPO version 'v2026-06-23'"
        ),
    ):
        validate_evaluation_hpo_version("v2025-03-03", asset)


def test_evaluation_hpo_version_must_be_non_empty() -> None:
    asset = RetrievalAssetIdentity(
        asset_type="single_vector",
        embedding_model="BAAI/bge-m3",
        hpo_version="v2026-06-23",
        manifest_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="Evaluation HPO version must be non-empty"):
        validate_evaluation_hpo_version("   ", asset)


def _fingerprint_identities():
    dataset = DatasetIdentity(
        source_sha256="0" * 64,
        input_sha256="1" * 64,
        gold_sha256="2" * 64,
        document_ids_sha256="3" * 64,
        projection="positive_hpo_present_v1",
        excluded_document_ids=("excluded",),
    )
    prompt = PromptBundleIdentity(
        schema_version="phentrieve-prompt-bundle/v1",
        mode="two_phase",
        language="en",
        prompt_behavior_version="1",
        components=(PromptComponentIdentity("extraction", "v3", "4" * 64),),
        sha256="5" * 64,
    )
    asset = RetrievalAssetIdentity(
        asset_type="multi_vector",
        embedding_model="BAAI/bge-m3",
        hpo_version="v2026-06-23",
        manifest_sha256="6" * 64,
    )
    model = {
        "provider": "openai",
        "model": "gpt-5-mini",
        "temperature": 0.0,
        "seed": 42,
        "sampling": {"top_p": 0.9, "stop": ["END", "STOP"]},
    }
    return dataset, prompt, asset, model


def test_gold_only_change_changes_scoring_not_execution() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    before = build_run_fingerprints(dataset, prompt, model, asset)
    after = build_run_fingerprints(
        replace(dataset, gold_sha256="a" * 64), prompt, model, asset
    )

    assert after.execution_sha256 == before.execution_sha256
    assert after.scoring_sha256 != before.scoring_sha256


@pytest.mark.parametrize(
    "changed_prompt",
    [
        {"sha256": "a" * 64},
        {"prompt_behavior_version": "2"},
        {"components": (PromptComponentIdentity("extraction", "v4", "b" * 64),)},
    ],
)
def test_complete_prompt_identity_changes_execution(changed_prompt) -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    before = build_run_fingerprints(dataset, prompt, model, asset)
    after = build_run_fingerprints(
        dataset, replace(prompt, **changed_prompt), model, asset
    )

    assert after.execution_sha256 != before.execution_sha256


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("provider", "anthropic"),
        ("model", "claude-sonnet"),
        ("temperature", 0.7),
        ("seed", 7),
        ("frequency_penalty", 0.2),
    ],
)
def test_any_model_or_sampling_parameter_changes_execution(key, value) -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    before = build_run_fingerprints(dataset, prompt, model, asset)
    after = build_run_fingerprints(dataset, prompt, {**model, key: value}, asset)

    assert after.execution_sha256 != before.execution_sha256


@pytest.mark.parametrize(
    "replacement",
    [
        {"input_sha256": "a" * 64},
        {"document_ids_sha256": "b" * 64},
    ],
)
def test_input_and_evaluated_document_ids_change_execution(replacement) -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    before = build_run_fingerprints(dataset, prompt, model, asset)
    after = build_run_fingerprints(
        dataset=replace(dataset, **replacement), prompt=prompt, model=model, asset=asset
    )

    assert after.execution_sha256 != before.execution_sha256


def test_asset_manifest_identity_changes_execution() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    before = build_run_fingerprints(dataset, prompt, model, asset)
    after = build_run_fingerprints(
        dataset, prompt, model, replace(asset, manifest_sha256="a" * 64)
    )

    assert after.execution_sha256 != before.execution_sha256


@pytest.mark.parametrize(
    "replacement",
    [
        {"document_ids_sha256": "a" * 64},
        {"projection": "all_annotations_v2"},
    ],
)
def test_evaluated_document_ids_and_projection_change_scoring(replacement) -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    before = build_run_fingerprints(dataset, prompt, model, asset)
    after = build_run_fingerprints(
        replace(dataset, **replacement), prompt, model, asset
    )

    assert after.scoring_sha256 != before.scoring_sha256


def test_model_mapping_key_order_does_not_change_fingerprints() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    reversed_model = dict(reversed(model.items()))

    assert build_run_fingerprints(
        dataset, prompt, model, asset
    ) == build_run_fingerprints(dataset, prompt, reversed_model, asset)


def test_producer_provenance_is_outside_run_fingerprints() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    first_manifest = {
        "fingerprints": build_run_fingerprints(dataset, prompt, model, asset),
        "provenance": {"package_version": "1.0", "git_commit": "abc"},
    }
    second_manifest = {
        "fingerprints": build_run_fingerprints(dataset, prompt, model, asset),
        "provenance": {"package_version": "2.0", "git_commit": "def"},
    }

    assert first_manifest["provenance"] != second_manifest["provenance"]
    assert first_manifest["fingerprints"] == second_manifest["fingerprints"]


def test_producer_source_changes_execution_and_scoring_fingerprints() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    before = build_run_fingerprints(
        dataset,
        prompt,
        model,
        asset,
        producer_source_sha256="a" * 64,
    )
    after = build_run_fingerprints(
        dataset,
        prompt,
        model,
        asset,
        producer_source_sha256="b" * 64,
    )

    assert before.execution_sha256 != after.execution_sha256
    assert before.scoring_sha256 != after.scoring_sha256


def test_ontology_scoring_configuration_changes_only_scoring_fingerprint() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    before = build_run_fingerprints(
        dataset,
        prompt,
        model,
        asset,
        scoring={"ontology_enabled": True, "semantic_floor": 0.3},
    )
    after = build_run_fingerprints(
        dataset,
        prompt,
        model,
        asset,
        scoring={"ontology_enabled": True, "semantic_floor": 0.9},
    )

    assert before.execution_sha256 == after.execution_sha256
    assert before.scoring_sha256 != after.scoring_sha256


def test_non_json_model_values_are_rejected_precisely() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()

    with pytest.raises(
        TypeError,
        match=r"Model configuration value at \$\.invalid must be JSON-compatible; got set",
    ):
        build_run_fingerprints(dataset, prompt, {**model, "invalid": {1, 2}}, asset)


def test_non_string_model_keys_are_rejected_before_they_can_collide() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()

    with pytest.raises(
        TypeError, match=r"Model configuration key at \$ must be a string"
    ):
        build_run_fingerprints(dataset, prompt, {**model, 1: "numeric"}, asset)


def test_tuple_model_values_are_rejected() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()

    with pytest.raises(
        TypeError,
        match=r"Model configuration value at \$\.stop must be JSON-compatible; got tuple",
    ):
        build_run_fingerprints(dataset, prompt, {**model, "stop": ("END",)}, asset)


def test_nested_invalid_model_values_report_json_path() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()

    with pytest.raises(
        TypeError,
        match=r"Model configuration value at \$\.sampling\.items\[1\] must be JSON-compatible; got bytes",
    ):
        build_run_fingerprints(
            dataset,
            prompt,
            {**model, "sampling": {"items": ["valid", b"invalid"]}},
            asset,
        )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_model_numbers_are_rejected(non_finite) -> None:
    dataset, prompt, asset, model = _fingerprint_identities()

    with pytest.raises(
        ValueError,
        match=r"Model configuration number at \$\.sampling\.value must be finite",
    ):
        build_run_fingerprints(
            dataset,
            prompt,
            {**model, "sampling": {"value": non_finite}},
            asset,
        )


def test_nested_json_model_values_are_accepted() -> None:
    dataset, prompt, asset, model = _fingerprint_identities()
    nested_model = {
        **model,
        "options": {
            "enabled": True,
            "label": None,
            "count": 2,
            "weights": [0.25, 0.75],
            "nested": {"names": ["first", "second"]},
        },
    }

    fingerprints = build_run_fingerprints(dataset, prompt, nested_model, asset)

    assert len(fingerprints.execution_sha256) == 64
