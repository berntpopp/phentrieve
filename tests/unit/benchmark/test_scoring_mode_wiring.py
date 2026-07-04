import json
from pathlib import Path

from phentrieve.benchmark.extraction_benchmark import (
    ExtractionBenchmark,
    ExtractionConfig,
)


def _fixture(tmp_path: Path) -> Path:
    payload = {
        "documents": [
            {
                "id": "d1",
                "text": "no fever",
                "gold_hpo_terms": [{"id": "HP:0001945", "assertion": "ABSENT"}],
            }
        ]
    }
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(payload))
    return p


def test_present_only_drops_absent_from_metrics(tmp_path, monkeypatch):
    # Extractor predicts the same term as ABSENT -> strict TP, present-only drops both sides.
    cfg = ExtractionConfig(scoring_mode="present-only", bootstrap_ci=False)
    bench = ExtractionBenchmark("BAAI/bge-m3", config=cfg)
    monkeypatch.setattr(
        bench.extractor, "extract", lambda text: [("HP:0001945", "ABSENT")]
    )
    metrics = bench.run_benchmark(_fixture(tmp_path), tmp_path / "out")
    # present-only removed the only (absent) pair from both sides -> no tp/fp/fn -> zeros
    assert metrics.micro["f1"] == 0.0


def test_strict_scores_the_absent_tuple(tmp_path, monkeypatch):
    cfg = ExtractionConfig(scoring_mode="strict", bootstrap_ci=False)
    bench = ExtractionBenchmark("BAAI/bge-m3", config=cfg)
    monkeypatch.setattr(
        bench.extractor, "extract", lambda text: [("HP:0001945", "ABSENT")]
    )
    metrics = bench.run_benchmark(_fixture(tmp_path), tmp_path / "out2")
    assert metrics.micro["f1"] == 1.0
