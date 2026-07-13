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
    assert safe_slug("FremyCompany/BioLORD-2023-M") == ("fremycompany_biolord_2023_m")
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


def test_create_run_layout_requires_overwrite_for_explicit_existing_run(
    tmp_path,
) -> None:
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
    assert (
        json.loads(layout.manifest_path.read_text(encoding="utf-8"))["status"]
        == "complete"
    )


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


def test_create_run_layout_accepts_llm_benchmark_type(tmp_path) -> None:
    layout = create_run_layout(
        tmp_path, "llm", "GeneReviews", "Org/Model", run_id="run"
    )

    assert layout.benchmark_type == "llm"
    assert layout.run_dir == (tmp_path / "llm" / "genereviews" / "org_model" / "run")


def test_write_manifest_registers_extra_artifacts_only_when_present(tmp_path) -> None:
    layout = create_run_layout(tmp_path, "llm", "GeneReviews", "model", run_id="run")
    predictions_dir = layout.run_dir / "predictions" / "two_phase"
    predictions_dir.mkdir(parents=True)
    (predictions_dir / "case_1.json").write_text("{}", encoding="utf-8")
    metrics_path = layout.run_dir / "metrics" / "benchmark_two_phase.json"
    missing_traces_dir = layout.run_dir / "traces" / "two_phase"

    manifest = write_manifest(
        layout,
        {"status": "complete"},
        extra_artifacts={
            "llm_predictions": (predictions_dir, "inode/directory"),
            "llm_traces": (missing_traces_dir, "inode/directory"),
            "metrics": (metrics_path, "application/json"),
        },
    )

    assert manifest["artifacts"]["llm_predictions"]["path"] == ("predictions/two_phase")
    assert "llm_traces" not in manifest["artifacts"]
    assert "metrics" not in manifest["artifacts"]


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


def test_discover_artifacts_finds_llm_runs_alongside_other_benchmark_types(
    tmp_path,
) -> None:
    retrieval_layout = create_run_layout(
        tmp_path, "retrieval", "set", "model", run_id="run"
    )
    write_json(retrieval_layout.summary_path, {"model": "retrieval-model"})
    write_manifest(retrieval_layout, {"status": "complete"})

    extraction_layout = create_run_layout(
        tmp_path, "extraction", "GSC", "model", run_id="run"
    )
    write_json(extraction_layout.summary_path, {"model": "extraction-model"})
    write_manifest(extraction_layout, {"status": "complete"})

    llm_layout = create_run_layout(
        tmp_path, "llm", "GeneReviews", "model", run_id="run"
    )
    write_json(llm_layout.summary_path, {"model": "llm-model"})
    write_manifest(llm_layout, {"status": "complete"})

    discovered = discover_artifacts(tmp_path, "summary")

    assert set(discovered) == {
        retrieval_layout.summary_path,
        extraction_layout.summary_path,
        llm_layout.summary_path,
    }


def test_overwrite_clears_previous_run_artifacts(tmp_path) -> None:
    """An overwritten run must not inherit artifacts it never produced.

    ``write_manifest`` decides what to advertise purely from ``Path.exists()``,
    so a stale file left behind by the previous run would be registered as an
    artifact of the new one.
    """
    first = create_run_layout(
        tmp_path, "extraction", "GSC", "model", run_id="exp", exact_run_id=True
    )
    write_jsonl(first.terms_path, [{"hpo_id": "HP:0001250", "run": 1}])
    write_jsonl(first.chunks_path, [{"chunk_id": 0, "run": 1}])
    write_json(first.legacy_dir / "model_summary.json", {"run": 1})
    (first.run_dir / "predictions").mkdir()
    (first.run_dir / "predictions" / "doc_b.json").write_text("{}", encoding="utf-8")
    write_manifest(first, {"status": "complete"})

    # Second run reuses the id but produces no chunk diagnostics this time.
    second = create_run_layout(
        tmp_path, "extraction", "GSC", "model", run_id="exp", overwrite=True
    )
    write_jsonl(second.terms_path, [{"hpo_id": "HP:0004322", "run": 2}])
    manifest = write_manifest(second, {"status": "complete"})

    assert second.run_dir == first.run_dir
    assert not second.chunks_path.exists()
    assert not (second.run_dir / "predictions").exists()
    assert not (second.legacy_dir / "model_summary.json").exists()
    assert "chunk_diagnostics" not in manifest["artifacts"]
    assert "legacy_compatibility" not in manifest["artifacts"]
    assert json.loads(second.terms_path.read_text(encoding="utf-8"))["run"] == 2


def test_overwrite_preserves_the_resume_checkpoint(tmp_path) -> None:
    """The LLM benchmark reuses a run directory precisely to resume from it."""
    first = create_run_layout(
        tmp_path, "llm", "CSC", "model", run_id="exp", exact_run_id=True
    )
    write_json(first.checkpoint_path, {"status": "running", "completed": ["doc-a"]})
    write_jsonl(first.terms_path, [{"hpo_id": "HP:0001250"}])

    second = create_run_layout(
        tmp_path, "llm", "CSC", "model", run_id="exp", overwrite=True
    )

    assert second.checkpoint_path.exists()
    assert json.loads(second.checkpoint_path.read_text(encoding="utf-8"))[
        "completed"
    ] == ["doc-a"]
    assert not second.terms_path.exists()


def test_discover_artifacts_can_filter_by_benchmark_type(tmp_path) -> None:
    """Every benchmark type writes a ``summary`` role into a shared root."""
    for benchmark_type in ("retrieval", "extraction", "llm"):
        layout = create_run_layout(
            tmp_path, benchmark_type, "set", "model", run_id="run"
        )
        write_json(layout.summary_path, {"benchmark_type": benchmark_type})
        write_manifest(layout, {"status": "complete"})

    assert len(discover_artifacts(tmp_path, "summary")) == 3
    retrieval_only = discover_artifacts(tmp_path, "summary", benchmark_type="retrieval")
    assert len(retrieval_only) == 1
    assert json.loads(retrieval_only[0].read_text(encoding="utf-8")) == {
        "benchmark_type": "retrieval"
    }
