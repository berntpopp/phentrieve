from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from phentrieve.benchmark.result_store import (
    create_run_layout,
    discover_artifacts,
    safe_slug,
    sha256_file,
    sha256_path,
    utc_run_id,
    write_json,
    write_jsonl,
    write_manifest,
)


def test_safe_slug_and_utc_run_id_are_filesystem_safe() -> None:
    assert safe_slug("FremyCompany/BioLORD-2023-M") == (
        "fremycompany_biolord_2023_m"
    )
    assert utc_run_id(datetime(2026, 7, 10, 12, 34, 56, 123456, tzinfo=UTC)) == (
        "20260710T123456123456Z"
    )


def test_create_run_layout_uses_unique_collision_suffix(tmp_path) -> None:
    first = create_run_layout(
        tmp_path,
        "retrieval",
        "200cases_o3_v1",
        "Org/Model",
        run_id="20260710T120000000000Z",
    )
    second = create_run_layout(
        tmp_path,
        "retrieval",
        "200cases_o3_v1",
        "Org/Model",
        run_id="20260710T120000000000Z",
    )

    assert first.run_dir.name == "20260710T120000000000Z"
    assert second.run_dir.name == "20260710T120000000000Z-2"
    assert first.run_dir.parent == (
        tmp_path / "retrieval" / "200cases_o3_v1" / "org_model"
    )
    assert first.legacy_dir.is_dir()


def test_create_run_layout_requires_overwrite_for_explicit_existing_run(tmp_path) -> None:
    layout = create_run_layout(
        tmp_path, "extraction", "GSC", "model", run_id="named-run"
    )

    with pytest.raises(FileExistsError):
        create_run_layout(
            tmp_path,
            "extraction",
            "GSC",
            "model",
            run_id=layout.run_id,
            exact_run_id=True,
        )

    reused = create_run_layout(
        tmp_path,
        "extraction",
        "GSC",
        "model",
        run_id=layout.run_id,
        exact_run_id=True,
        overwrite=True,
    )
    assert reused.run_dir == layout.run_dir


def test_json_jsonl_checksum_and_manifest_round_trip(tmp_path) -> None:
    source = tmp_path / "cases.json"
    source.write_text('[{"text": "seizures"}]', encoding="utf-8")
    layout = create_run_layout(
        tmp_path / "results", "retrieval", "set", "model", run_id="run"
    )
    write_json(layout.summary_path, {"mrr_dense": 1.0})
    write_jsonl(layout.terms_path, [{"hpo_id": "HP:0001250", "rank": 1}])
    write_jsonl(layout.cases_path, [{"case_id": 0, "status": "complete"}])
    manifest = write_manifest(
        layout,
        {
            "status": "complete",
            "dataset": {"path": str(source), "sha256": sha256_file(source)},
        },
    )

    assert json.loads(layout.summary_path.read_text(encoding="utf-8")) == {
        "mrr_dense": 1.0
    }
    assert json.loads(layout.terms_path.read_text(encoding="utf-8")) == {
        "hpo_id": "HP:0001250",
        "rank": 1,
    }
    assert manifest["artifacts"]["summary"]["path"] == "summary.json"
    assert manifest["artifacts"]["term_results"]["path"] == "terms.jsonl"
    assert json.loads(layout.manifest_path.read_text(encoding="utf-8"))[
        "status"
    ] == "complete"


def test_sha256_path_hashes_directory_names_and_contents_deterministically(
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "annotations").mkdir(parents=True)
    (dataset / "annotations" / "b.json").write_text("b", encoding="utf-8")
    (dataset / "annotations" / "a.json").write_text("a", encoding="utf-8")

    first = sha256_path(dataset)
    second = sha256_path(dataset)
    (dataset / "annotations" / "a.json").write_text("changed", encoding="utf-8")

    assert first == second
    assert sha256_path(dataset) != first


def test_discover_artifacts_prefers_canonical_manifests_and_falls_back_to_legacy(
    tmp_path,
) -> None:
    canonical_root = tmp_path / "canonical"
    layout = create_run_layout(
        canonical_root, "retrieval", "set", "model", run_id="run"
    )
    write_json(layout.summary_path, {"model": "canonical"})
    write_manifest(layout, {"status": "complete"})
    write_json(canonical_root / "old_summary.json", {"model": "legacy"})

    assert discover_artifacts(canonical_root, "summary") == [layout.summary_path]

    legacy_root = tmp_path / "legacy"
    write_json(legacy_root / "model_summary.json", {"model": "legacy"})
    assert discover_artifacts(legacy_root, "summary") == [
        legacy_root / "model_summary.json"
    ]
