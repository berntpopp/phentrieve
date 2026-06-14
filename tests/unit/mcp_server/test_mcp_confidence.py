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


def test_empty_results_flagged():
    """B1: a threshold-emptied result set is the strongest no-high-confidence
    signal, not the weakest. ``no_high_confidence_match`` must be True."""
    out = annotate_search_confidence({"results": []})
    assert out["no_high_confidence_match"] is True


def test_missing_results_key_flagged():
    """B1: a payload with no results key at all is also no-high-confidence."""
    out = annotate_search_confidence({})
    assert out["no_high_confidence_match"] is True


def test_flag_is_band_based_not_top_score_based():
    """B1: the flag keys off whether any surviving result is in the ``high``
    band, so it stays correct even if the top hit is trimmed by a later token
    budget. A moderate-only set is flagged; any high-band hit clears it."""
    moderate_only = annotate_search_confidence(
        {"results": [{"hpo_id": "HP:1", "similarity": 0.6}]}
    )
    assert moderate_only["no_high_confidence_match"] is True

    has_high = annotate_search_confidence(
        {
            "results": [
                {"hpo_id": "HP:1", "similarity": 0.82},
                {"hpo_id": "HP:2", "similarity": 0.45},
            ]
        }
    )
    assert has_high["no_high_confidence_match"] is False
