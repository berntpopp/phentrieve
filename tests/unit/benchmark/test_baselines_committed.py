import json
from pathlib import Path

from phentrieve.benchmark.extraction_cli import _regressions

BASE = Path("tests/data/benchmarks/baselines")


def test_baseline_files_exist_and_have_metrics():
    for name in ("tiny_strict_summary.json", "tiny_present_only_summary.json"):
        data = json.loads((BASE / name).read_text())
        assert "micro_f1" in data


def test_baseline_does_not_regress_itself():
    data = json.loads((BASE / "tiny_present_only_summary.json").read_text())
    assert _regressions(data, data, tolerance=0.0) == []
