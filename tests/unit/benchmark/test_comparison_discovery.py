from __future__ import annotations

from phentrieve.benchmark.result_store import (
    create_run_layout,
    write_json,
    write_manifest,
)
from phentrieve.evaluation.comparison_orchestrator import (
    compare_benchmark_summaries,
    load_benchmark_summaries,
    orchestrate_benchmark_comparison,
)
from phentrieve.evaluation.result_analyzer import load_summary_files


def test_comparison_loaders_discover_nested_canonical_summaries(tmp_path) -> None:
    for model, mrr in (("model-a", 0.8), ("model-b", 0.9)):
        layout = create_run_layout(
            tmp_path, "retrieval", "200cases_o3_v1", model, run_id="run"
        )
        write_json(
            layout.summary_path,
            {
                "model": model,
                "mrr_dense": mrr,
                "dataset_name": "200cases_o3_v1",
                "run_id": layout.run_id,
            },
        )
        write_manifest(layout, {"status": "complete"})
    write_json(tmp_path / "stale_summary.json", {"model": "stale"})

    comparison = load_benchmark_summaries(str(tmp_path))
    analysis = load_summary_files(str(tmp_path))

    assert {item["model"] for item in comparison} == {"model-a", "model-b"}
    assert {item["model"] for item in analysis} == {"model-a", "model-b"}
    comparison_frame = compare_benchmark_summaries(comparison)
    assert set(comparison_frame["Dataset"]) == {"200cases_o3_v1"}
    assert set(comparison_frame["Run ID"]) == {"run"}


def test_comparison_loaders_still_accept_legacy_flat_summaries(tmp_path) -> None:
    write_json(tmp_path / "model_summary.json", {"model": "legacy"})

    assert load_benchmark_summaries(str(tmp_path))[0]["model"] == "legacy"
    assert load_summary_files(str(tmp_path))[0]["model"] == "legacy"


def test_comparison_default_searches_the_results_root(tmp_path) -> None:
    layout = create_run_layout(
        tmp_path, "retrieval", "set", "model", run_id="run"
    )
    write_json(layout.summary_path, {"model": "model", "mrr_dense": 0.8})
    write_manifest(layout, {"status": "complete"})
    comparison = orchestrate_benchmark_comparison(results_dir_override=str(tmp_path))

    assert comparison is not None
    assert comparison.iloc[0]["Model"] == "model"
