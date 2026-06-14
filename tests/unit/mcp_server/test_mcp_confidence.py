"""D6 -- search flags low-confidence / no-high-confidence-match results."""

import pytest

from api.mcp.confidence import annotate_search_confidence, confidence_band

pytestmark = pytest.mark.unit


def test_band_thresholds():
    assert confidence_band(0.85) == "high"
    assert confidence_band(0.5) == "moderate"
    assert confidence_band(0.2) == "low"


def test_gibberish_top_hit_flagged():
    payload = {
        "results": [
            {"hpo_id": "HP:1", "similarity": 0.59},
            {"hpo_id": "HP:2", "similarity": 0.48},
        ]
    }
    out = annotate_search_confidence(payload)
    assert out["no_high_confidence_match"] is True
    assert out["results"][0]["confidence_band"] == "moderate"
    assert out["results"][1]["confidence_band"] == "moderate"


def test_strong_top_hit_not_flagged():
    payload = {"results": [{"hpo_id": "HP:1", "similarity": 0.82}]}
    out = annotate_search_confidence(payload)
    assert out["no_high_confidence_match"] is False
    assert out["results"][0]["confidence_band"] == "high"


def test_empty_results_not_flagged():
    out = annotate_search_confidence({"results": []})
    assert out["no_high_confidence_match"] is False
