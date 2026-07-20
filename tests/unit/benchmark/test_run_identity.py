from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from phentrieve.benchmark.run_identity import (
    RetrievalAssetIdentity,
    build_dataset_identity,
    load_retrieval_asset_identity,
    validate_evaluation_hpo_version,
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


def test_gold_annotation_duplicates_and_order_do_not_change_gold_hash(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    _write_dataset(path, [_document("case-a", "Alpha", ["HP:2", "HP:1", "HP:2"])])
    before = build_dataset_identity(path, dataset="all")
    _write_dataset(path, [_document("case-a", "Alpha", ["HP:1", "HP:2"])])
    after = build_dataset_identity(path, dataset="all")

    assert after.gold_sha256 == before.gold_sha256


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
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "hpo_version": "v2026-06-23",
                "model": {
                    "name": "BAAI/bge-m3",
                    "slug": "bge-m3",
                    "dimension": 1024,
                    "multi_vector": True,
                },
            }
        ),
        encoding="utf-8",
    )

    identity = load_retrieval_asset_identity(tmp_path)

    assert identity.asset_type == "multi_vector"
    assert identity.embedding_model == "BAAI/bge-m3"
    assert identity.hpo_version == "v2026-06-23"
    assert len(identity.manifest_sha256) == 64


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
