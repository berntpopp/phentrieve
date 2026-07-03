from phentrieve.benchmark.extraction_cli import _regressions


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
