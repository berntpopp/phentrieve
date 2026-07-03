import json
import math
from pathlib import Path

from typer.testing import CliRunner

from phentrieve.benchmark.extraction_cli import _regressions, app

runner = CliRunner()


def _write_summary(path: Path, **fields: object) -> Path:
    base = {"micro_f1": 0.80, "micro_precision": 0.80, "micro_recall": 0.80}
    base.update(fields)
    path.write_text(json.dumps(base))
    return path


def test_no_regression_when_candidate_equal_or_better():
    base = {"micro_f1": 0.80, "micro_precision": 0.80, "micro_recall": 0.80}
    cand = {"micro_f1": 0.80, "micro_precision": 0.82, "micro_recall": 0.80}
    assert _regressions(base, cand, tolerance=0.0) == []


def test_regression_detected_on_f1_drop():
    base = {"micro_f1": 0.80, "micro_precision": 0.80, "micro_recall": 0.80}
    cand = {"micro_f1": 0.78, "micro_precision": 0.80, "micro_recall": 0.80}
    out = _regressions(base, cand, tolerance=0.0)
    assert any("micro_f1" in r for r in out)


def test_tolerance_absorbs_small_noise():
    base = {"micro_f1": 0.80, "micro_precision": 0.80, "micro_recall": 0.80}
    cand = {"micro_f1": 0.795, "micro_precision": 0.80, "micro_recall": 0.80}
    assert _regressions(base, cand, tolerance=0.01) == []


def test_regressions_fail_closed_on_nan_candidate():
    # A NaN candidate must NOT slip through the gate (NaN comparisons are False).
    base = {"micro_f1": 0.80, "micro_precision": 0.80, "micro_recall": 0.80}
    cand = {"micro_f1": math.nan, "micro_precision": 0.80, "micro_recall": 0.80}
    out = _regressions(base, cand, tolerance=0.0)
    assert any("micro_f1" in r for r in out)


def test_cli_no_regression_exits_zero(tmp_path):
    b = _write_summary(tmp_path / "b.json", scoring_mode="present-only")
    c = _write_summary(tmp_path / "c.json", scoring_mode="present-only")
    result = runner.invoke(
        app, ["assert-no-regression", "--baseline", str(b), "--candidate", str(c)]
    )
    assert result.exit_code == 0
    assert "No regression" in result.stdout


def test_cli_regression_exits_one(tmp_path):
    b = _write_summary(tmp_path / "b.json", scoring_mode="strict")
    c = _write_summary(tmp_path / "c.json", scoring_mode="strict", micro_f1=0.70)
    result = runner.invoke(
        app, ["assert-no-regression", "--baseline", str(b), "--candidate", str(c)]
    )
    assert result.exit_code == 1
    assert "REGRESSION" in result.stdout


def test_cli_scoring_mode_mismatch_exits_two(tmp_path):
    # Gating a present-only candidate against a strict baseline is apples-to-oranges.
    b = _write_summary(tmp_path / "b.json", scoring_mode="strict")
    c = _write_summary(tmp_path / "c.json", scoring_mode="present-only")
    result = runner.invoke(
        app, ["assert-no-regression", "--baseline", str(b), "--candidate", str(c)]
    )
    assert result.exit_code == 2
    assert "MODE MISMATCH" in result.stdout
