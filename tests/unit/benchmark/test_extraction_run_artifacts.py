from __future__ import annotations

import json

from phentrieve.benchmark.extraction_benchmark import (
    ExtractionBenchmark,
    ExtractionConfig,
)
from phentrieve.benchmark.result_store import create_run_layout


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_extraction_run_preserves_terms_cases_and_raw_chunk_candidates(
    tmp_path, monkeypatch
) -> None:
    benchmark = ExtractionBenchmark(
        "Org/Model",
        config=ExtractionConfig(
            model_name="Org/Model",
            dataset="GSC",
            bootstrap_ci=False,
            scoring_mode="present-only",
            chunk_retrieval_threshold=0.5,
            min_confidence_for_aggregated=0.5,
        ),
    )
    monkeypatch.setattr(
        "phentrieve.benchmark.extraction_benchmark.load_benchmark_data",
        lambda test_path, dataset: {
            "metadata": {
                "dataset_name": "rag_hpo_GSC",
                "source": "rag_hpo_paper",
                "total_documents": 1,
                "total_annotations": 1,
            },
            "documents": [
                {
                    "id": "GSC_1",
                    "text": "Patient has seizures.",
                    "gold_hpo_terms": [
                        {
                            "id": "HP:0001250",
                            "label": "Seizure",
                            "assertion": "PRESENT",
                        }
                    ],
                    "source_dataset": "GSC",
                }
            ],
        },
    )
    monkeypatch.setattr(
        benchmark.extractor,
        "extract_with_details",
        lambda text: (
            [("HP:0001250", "PRESENT")],
            {
                "processed_chunks": [
                    {
                        "chunk_idx": 0,
                        "text": text,
                        "start_char": 0,
                        "end_char": len(text),
                        "assertion_status": "affirmed",
                    }
                ],
                "raw_query_results": [
                    {
                        "metadatas": [
                            [
                                {"hpo_id": "HP:0001250", "label": "Seizure"},
                                {
                                    "hpo_id": "HP:0004322",
                                    "label": "Abnormality of movement",
                                },
                            ]
                        ],
                        "similarities": [[0.9, 0.42]],
                    }
                ],
                "chunk_results": [
                    {
                        "chunk_idx": 0,
                        "chunk_text": text,
                        "matches": [
                            {
                                "id": "HP:0001250",
                                "name": "Seizure",
                                "score": 0.9,
                                "assertion_status": "affirmed",
                            }
                        ],
                    }
                ],
                "aggregated_results": [
                    {
                        "id": "HP:0001250",
                        "name": "Seizure",
                        "score": 0.9,
                        "chunks": [{"chunk_idx": 0, "score": 0.9}],
                    }
                ],
            },
        ),
    )
    source = tmp_path / "source"
    source.mkdir()
    layout = create_run_layout(
        tmp_path / "results", "extraction", "GSC", "Org/Model", run_id="run"
    )

    benchmark.run_benchmark(
        source,
        layout.legacy_dir,
        run_layout=layout,
    )

    terms = _read_jsonl(layout.terms_path)
    cases = _read_jsonl(layout.cases_path)
    chunks = _read_jsonl(layout.chunks_path)
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))

    assert {term["hpo_id"]: term["outcome"] for term in terms} == {
        "HP:0001250": "tp",
        "HP:0004322": "filtered",
    }
    assert cases[0]["doc_id"] == "GSC_1"
    assert cases[0]["metrics"] == {"tp": 1, "fp": 0, "fn": 0}
    assert chunks[0]["candidates"][1] == {
        "rank": 2,
        "hpo_id": "HP:0004322",
        "label": "Abnormality of movement",
        "score": 0.42,
        "passes_threshold": False,
        "used_in_aggregation": False,
    }
    assert manifest["artifacts"]["chunk_diagnostics"]["path"] == (
        "diagnostics/chunks.jsonl"
    )
    assert (layout.legacy_dir / "extraction_results.json").is_file()
