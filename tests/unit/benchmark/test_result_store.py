from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from phentrieve.benchmark.result_store import (
    ArtifactEntry,
    active_checkpoint_path,
    create_run_layout,
    discover_artifacts,
    publish_manifest_v2,
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


def test_create_run_layout_can_validate_without_materializing(tmp_path) -> None:
    layout = create_run_layout(
        tmp_path, "llm", "GeneReviews", "Org/Model", run_id="run", materialize=False
    )

    assert not layout.run_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_create_run_layout_rejects_symlinked_run_directory(tmp_path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    run_dir = tmp_path / "results" / "llm" / "genereviews" / "model" / "run"
    run_dir.parent.mkdir(parents=True)
    try:
        run_dir.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="link or junction"):
        create_run_layout(
            tmp_path / "results",
            "llm",
            "GeneReviews",
            "model",
            run_id="run",
            overwrite=True,
        )

    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_create_run_layout_rejects_junctioned_run_directory(tmp_path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    run_dir = tmp_path / "results" / "llm" / "genereviews" / "model" / "run"
    run_dir.parent.mkdir(parents=True)
    cmd_executable = os.environ.get("COMSPEC")
    if not cmd_executable:
        pytest.skip("COMSPEC is unavailable")
    created = subprocess.run(  # noqa: S603 - resolved Windows command interpreter
        [cmd_executable, "/c", "mklink", "/J", str(run_dir), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr}")
    try:
        with pytest.raises(ValueError, match="link or junction"):
            create_run_layout(
                tmp_path / "results",
                "llm",
                "GeneReviews",
                "model",
                run_id="run",
                overwrite=True,
            )
        assert marker.read_text(encoding="utf-8") == "keep"
    finally:
        run_dir.rmdir()


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


def test_publish_manifest_v2_layers_hash_inventory_onto_run_layout(tmp_path) -> None:
    layout = create_run_layout(tmp_path, "llm", "CSC", "model", run_id="run")
    write_json(
        layout.summary_path, {"status": "completed", "execution_fingerprint": "d" * 64}
    )
    prediction = layout.run_dir / "predictions" / "two_phase" / "case_1.json"
    prediction.parent.mkdir(parents=True)
    prediction.write_text('{"hpo_id":"HP:0001250"}', encoding="utf-8")
    identities = {
        "dataset_identity": {"input_sha256": "a" * 64},
        "prompt_identity": {"sha256": "b" * 64},
        "execution_fingerprint": "d" * 64,
        "scoring_fingerprint": "e" * 64,
    }

    manifest = publish_manifest_v2(
        layout,
        identities,
        [
            ArtifactEntry(layout.summary_path, "summary", "application/json"),
            ArtifactEntry(prediction, "prediction", "application/json"),
        ],
    )

    assert manifest["schema_version"] == 2
    assert manifest["execution_fingerprint"] == "d" * 64
    published_summary = layout.run_dir / manifest["artifacts"]["summary"]["path"]
    assert published_summary.is_file()
    assert manifest["artifacts"]["summary"]["path"].startswith(".generations/")
    assert manifest["artifacts"]["summary"]["sha256"] == sha256_file(published_summary)
    assert manifest["artifacts"]["prediction:predictions/two_phase/case_1.json"][
        "sha256"
    ] == sha256_file(prediction)
    assert discover_artifacts(tmp_path, "summary", benchmark_type="llm") == [
        published_summary
    ]


def test_publish_manifest_v2_keeps_previous_generation_immutable(tmp_path) -> None:
    layout = create_run_layout(tmp_path, "llm", "CSC", "model", run_id="run")
    write_json(layout.summary_path, {"generation": 1})
    first = publish_manifest_v2(
        layout,
        {},
        [ArtifactEntry(layout.summary_path, "summary", "application/json")],
    )
    first_path = layout.run_dir / first["artifacts"]["summary"]["path"]

    write_json(layout.summary_path, {"generation": 2})
    second = publish_manifest_v2(
        layout,
        {},
        [ArtifactEntry(layout.summary_path, "summary", "application/json")],
    )
    second_path = layout.run_dir / second["artifacts"]["summary"]["path"]

    assert first_path != second_path
    assert json.loads(first_path.read_text(encoding="utf-8")) == {"generation": 1}
    assert json.loads(second_path.read_text(encoding="utf-8")) == {"generation": 2}
    assert discover_artifacts(tmp_path, "summary", benchmark_type="llm") == [
        second_path
    ]

    write_json(layout.summary_path, {"generation": 3})
    third = publish_manifest_v2(
        layout,
        {},
        [ArtifactEntry(layout.summary_path, "summary", "application/json")],
    )
    third_path = layout.run_dir / third["artifacts"]["summary"]["path"]
    generation_dirs = [
        path for path in (layout.run_dir / ".generations").iterdir() if path.is_dir()
    ]

    assert not first_path.exists()
    assert second_path.exists()
    assert third_path.exists()
    assert len(generation_dirs) == 2


def test_publish_manifest_v2_supports_relative_results_root(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    layout = create_run_layout(Path("results"), "llm", "CSC", "model", run_id="run")
    prediction = layout.run_dir / "predictions" / "two_phase" / "case.json"
    trace = layout.run_dir / "traces" / "two_phase" / "case.json"
    write_json(prediction, {"kind": "prediction"})
    write_json(trace, {"kind": "trace"})

    manifest = publish_manifest_v2(
        layout,
        {},
        [
            ArtifactEntry(prediction, "prediction", "application/json"),
            ArtifactEntry(trace, "trace", "application/json"),
        ],
    )

    assert manifest["artifacts"]["llm_predictions"]["path"].endswith(
        "predictions/two_phase"
    )
    assert manifest["artifacts"]["llm_traces"]["path"].endswith("traces/two_phase")


def test_active_checkpoint_prefers_committed_generation(tmp_path) -> None:
    layout = create_run_layout(tmp_path, "llm", "CSC", "model", run_id="run")
    write_json(layout.checkpoint_path, {"generation": 1})
    manifest = publish_manifest_v2(
        layout,
        {},
        [ArtifactEntry(layout.checkpoint_path, "checkpoint", "application/json")],
    )
    committed = layout.run_dir / manifest["artifacts"]["checkpoint"]["path"]
    write_json(layout.checkpoint_path, {"generation": 2})

    assert active_checkpoint_path(layout) == committed
    assert json.loads(active_checkpoint_path(layout).read_text(encoding="utf-8")) == {
        "generation": 1
    }


def test_failed_generation_publish_preserves_current_manifest(tmp_path) -> None:
    layout = create_run_layout(tmp_path, "llm", "CSC", "model", run_id="run")
    write_json(layout.summary_path, {"generation": 1})
    publish_manifest_v2(
        layout,
        {},
        [ArtifactEntry(layout.summary_path, "summary", "application/json")],
    )
    before = layout.manifest_path.read_bytes()

    with pytest.raises(ValueError, match="not a file"):
        publish_manifest_v2(
            layout,
            {},
            [
                ArtifactEntry(
                    layout.run_dir / "missing.json", "summary", "application/json"
                )
            ],
        )

    assert layout.manifest_path.read_bytes() == before


def test_publish_manifest_v2_retains_v1_singleton_role_aliases(tmp_path) -> None:
    layout = create_run_layout(tmp_path, "llm", "CSC", "model", run_id="run")
    files = {
        "checkpoint": layout.checkpoint_path,
        "term_results": layout.terms_path,
        "case_results": layout.cases_path,
        "chunk_diagnostics": layout.chunks_path,
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    manifest = publish_manifest_v2(
        layout,
        {},
        [ArtifactEntry(path, role, "application/json") for role, path in files.items()],
    )
    for role, path in files.items():
        assert manifest["artifacts"][role]["path"].startswith(".generations/")
        assert manifest["artifacts"][role]["path"].endswith(
            path.relative_to(layout.run_dir).as_posix()
        )
        assert manifest["artifacts"][role]["sha256"] == sha256_file(
            layout.run_dir / manifest["artifacts"][role]["path"]
        )
        assert manifest["artifacts"][
            f"{role}:{path.relative_to(layout.run_dir).as_posix()}"
        ]["sha256"] == sha256_file(path)


def test_publish_manifest_v2_rejects_inventory_outside_run(tmp_path) -> None:
    layout = create_run_layout(tmp_path / "runs", "llm", "CSC", "model", run_id="run")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the run directory"):
        publish_manifest_v2(
            layout, {}, [ArtifactEntry(outside, "prediction", "application/json")]
        )


@pytest.mark.parametrize(
    "reserved",
    [
        "schema_version",
        "run_id",
        "benchmark_type",
        "dataset_name",
        "model",
        "artifacts",
    ],
)
def test_publish_manifest_v2_rejects_reserved_top_level_metadata(
    tmp_path, reserved
) -> None:
    layout = create_run_layout(tmp_path, "llm", "CSC", "model", run_id="run")
    with pytest.raises(ValueError, match="reserved manifest keys"):
        publish_manifest_v2(layout, {reserved: "override"}, [])


def test_publish_manifest_v2_rejects_duplicate_singleton_roles(tmp_path) -> None:
    layout = create_run_layout(tmp_path, "llm", "CSC", "model", run_id="run")
    first = layout.run_dir / "first.json"
    second = layout.run_dir / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate singleton artifact role"):
        publish_manifest_v2(
            layout,
            {},
            [
                ArtifactEntry(first, "summary", "application/json"),
                ArtifactEntry(second, "summary", "application/json"),
            ],
        )


def test_discover_artifacts_ignores_malformed_and_tampered_manifests(tmp_path) -> None:
    malformed = tmp_path / "malformed" / "manifest.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("[]", encoding="utf-8")

    layout = create_run_layout(tmp_path, "llm", "CSC", "model", run_id="run")
    write_json(layout.summary_path, {"status": "complete"})
    manifest = publish_manifest_v2(
        layout,
        {},
        [ArtifactEntry(layout.summary_path, "summary", "application/json")],
    )
    published = layout.run_dir / manifest["artifacts"]["summary"]["path"]
    published.write_text('{"status":"tampered"}\n', encoding="utf-8")

    assert discover_artifacts(tmp_path, "summary", benchmark_type="llm") == []


@pytest.mark.parametrize(
    "malformed",
    [
        "{",
        "null",
        "42",
        json.dumps({"schema_version": 2, "artifacts": None}),
        json.dumps({"schema_version": 2, "artifacts": []}),
        json.dumps(
            {
                "schema_version": 2,
                "artifacts": {"summary": "not-an-object"},
            }
        ),
        json.dumps(
            {
                "schema_version": 2,
                "artifacts": {
                    "summary": {"role": "summary", "path": "../outside.json"}
                },
            }
        ),
    ],
)
def test_discovery_skips_each_malformed_manifest_but_keeps_valid_sibling(
    tmp_path, malformed
) -> None:
    invalid_manifest = tmp_path / "invalid" / "manifest.json"
    invalid_manifest.parent.mkdir()
    invalid_manifest.write_text(malformed, encoding="utf-8")
    valid = create_run_layout(tmp_path, "llm", "GeneReviews", "model", run_id="valid")
    write_json(valid.summary_path, {"valid": True})
    write_manifest(valid, {"status": "complete"})

    assert discover_artifacts(tmp_path, "summary", benchmark_type="llm") == [
        valid.summary_path.resolve()
    ]


def test_discover_artifacts_rejects_absolute_and_parent_paths(tmp_path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    for index, artifact_path in enumerate((str(outside.resolve()), "../outside.json")):
        run = tmp_path / f"run-{index}"
        run.mkdir()
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "benchmark_type": "llm",
                    "artifacts": {
                        "summary": {
                            "role": "summary",
                            "path": artifact_path,
                            "sha256": sha256_file(outside),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    assert discover_artifacts(tmp_path, "summary", benchmark_type="llm") == []
