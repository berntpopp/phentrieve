from __future__ import annotations

import json

from phentrieve.evaluation.runner import run_evaluation


class _FakeRetriever:
    def query(self, text: str, n_results: int) -> dict:
        assert text == "seizures"
        assert n_results == 3
        return {
            "ids": [["row-1", "row-2"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001250", "label": "Seizure"},
                    {
                        "hpo_id": "HP:0004322",
                        "label": "Abnormality of movement",
                    },
                ]
            ],
            "documents": [["Seizure", "Abnormality of movement"]],
            "distances": [[0.1, 0.3]],
        }


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_evaluation_writes_ranked_canonical_and_legacy_artifacts(
    tmp_path, monkeypatch
) -> None:
    test_file = tmp_path / "200cases_o3_v1.json"
    test_file.write_text(
        json.dumps(
            [
                {
                    "description": "case-one",
                    "text": "seizures",
                    "expected_hpo_ids": ["HP:0001250"],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.runner.load_embedding_model", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.runner.DenseRetriever.from_model_name",
        lambda **kwargs: _FakeRetriever(),
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.runner.load_hpo_graph_data", lambda: None
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.runner.calculate_test_case_max_ont_sim",
        lambda expected, retrieved, formula: 1.0,
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.runner.calculate_bootstrap_ci_for_metrics",
        lambda results, k_values: {},
    )

    result = run_evaluation(
        model_name="Org/Model",
        test_file=str(test_file),
        k_values=(1,),
        save_results=True,
        results_dir=tmp_path / "results",
    )

    assert result is not None
    run_dir = result["run_dir"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    terms = _read_jsonl(run_dir / "terms.jsonl")
    cases = _read_jsonl(run_dir / "cases.jsonl")

    assert manifest["benchmark_type"] == "retrieval"
    assert manifest["dataset"]["sha256"]
    assert [
        (term["hpo_id"], term["rank"], term["score"], term["is_gold"])
        for term in terms
    ] == [
        ("HP:0001250", 1, 0.9, True),
        ("HP:0004322", 2, 0.7, False),
    ]
    assert cases[0]["expected_hpo_ids"] == ["HP:0001250"]
    assert cases[0]["status"] == "complete"
    assert (run_dir / "legacy" / "model_summary.json").is_file()
    assert (run_dir / "legacy" / "model_detailed.csv").is_file()
