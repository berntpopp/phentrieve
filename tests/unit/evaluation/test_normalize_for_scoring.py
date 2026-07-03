import pytest

from phentrieve.evaluation._extraction_types import ExtractionResult
from phentrieve.evaluation.extraction_metrics import normalize_for_scoring


def _r():
    return [
        ExtractionResult(
            doc_id="d1",
            predicted=[
                ("HP:1", "PRESENT"),
                ("HP:2", "ABSENT"),
                ("HP:3", "FAMILY_HISTORY"),
            ],
            gold=[("HP:1", "PRESENT"), ("HP:2", "ABSENT")],
        )
    ]


def test_strict_mode_is_identity():
    results = _r()
    assert normalize_for_scoring(results, "strict") is results


def test_present_only_drops_non_present_from_pred_and_gold():
    out = normalize_for_scoring(_r(), "present-only")
    assert out[0].predicted == [("HP:1", "PRESENT")]
    assert out[0].gold == [("HP:1", "PRESENT")]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        normalize_for_scoring(_r(), "bogus")
