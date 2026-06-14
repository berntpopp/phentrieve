"""Unit tests for api.mcp.shaping (response modes and budgets)."""

from __future__ import annotations

import pytest

from api.mcp.shaping import (
    BUDGETS,
    DEFAULT_MODE,
    MODES,
    apply_response_mode,
    enforce_budget,
    resolve_mode,
)


def test_modes_and_default():
    assert MODES == ("minimal", "compact", "standard", "full")
    assert DEFAULT_MODE == "compact"
    assert (
        BUDGETS["minimal"] < BUDGETS["compact"] < BUDGETS["standard"] < BUDGETS["full"]
    )


def test_resolve_mode_defaults_and_validates():
    assert resolve_mode(None) == "compact"
    assert resolve_mode("full") == "full"
    with pytest.raises(ValueError):
        resolve_mode("verbose")


def test_minimal_strips_detail_fields():
    payload = {
        "results": [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.9,
                "definition": "x",
                "synonyms": ["a"],
            }
        ]
    }
    item = apply_response_mode(payload, "minimal")["results"][0]
    assert item == {"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.9}


def test_compact_drops_empty_and_detail_fields():
    payload = {
        "results": [
            {
                "hpo_id": "HP:1",
                "label": "x",
                "similarity": 0.5,
                "synonyms": [],
                "definition": "d",
                "component_scores": {},
            }
        ]
    }
    item = apply_response_mode(payload, "compact")["results"][0]
    assert "synonyms" not in item
    assert "definition" not in item  # detail field dropped at compact
    assert "component_scores" not in item
    assert item["label"] == "x"


def test_standard_keeps_definition():
    payload = {"results": [{"hpo_id": "HP:1", "definition": "d", "synonyms": ["a"]}]}
    item = apply_response_mode(payload, "standard")["results"][0]
    assert item["definition"] == "d"
    assert item["synonyms"] == ["a"]


def test_full_keeps_everything_unchanged():
    payload = {"results": [{"hpo_id": "HP:1", "definition": None, "synonyms": []}]}
    assert apply_response_mode(payload, "full") == payload


def test_meta_passes_through_unshaped():
    payload = {"results": [], "_meta": {"definition": "keep-me"}}
    out = apply_response_mode(payload, "minimal")
    assert out["_meta"] == {"definition": "keep-me"}


def test_phenopacket_json_blob_gated_to_standard_and_full():
    """R1: the serialized phenopacket_json string is a top-level detail field --
    dropped at minimal/compact so the canonical `phenopacket` object is the one
    form, kept at standard/full for consumers that want the serialized blob."""
    raw = {
        "phenopacket": {
            "id": "CASE-1",
            "phenotypicFeatures": [{"type": {"id": "HP:0001250"}}],
        },
        "phenopacket_json": '{"id": "CASE-1"}',
    }

    compact = apply_response_mode(raw, "compact")
    assert compact.get("phenopacket", {}).get("id") == "CASE-1"
    assert "phenopacket_json" not in compact

    minimal = apply_response_mode(raw, "minimal")
    assert "phenopacket_json" not in minimal

    standard = apply_response_mode(raw, "standard")
    assert "phenopacket_json" in standard
    assert standard.get("phenopacket", {}).get("id") == "CASE-1"

    full = apply_response_mode(raw, "full")
    assert "phenopacket_json" in full


def test_enforce_budget_truncates_and_reports():
    payload = {
        "results": [{"hpo_id": f"HP:{i:07d}", "label": "x" * 300} for i in range(300)]
    }
    out, trunc = enforce_budget(payload, "minimal", list_field="results")
    assert trunc is not None
    assert trunc["field"] == "results"
    assert trunc["total"] == 300
    assert 0 <= trunc["returned"] < 300
    assert len(out["results"]) == trunc["returned"]


def test_enforce_budget_noop_when_small():
    payload = {"results": [{"hpo_id": "HP:1"}]}
    out, trunc = enforce_budget(payload, "compact", list_field="results")
    assert trunc is None
    assert out == payload


def test_enforce_budget_full_never_truncates():
    payload = {
        "results": [{"hpo_id": f"HP:{i:07d}", "label": "x" * 300} for i in range(300)]
    }
    out, trunc = enforce_budget(payload, "full", list_field="results")
    assert trunc is None
    assert len(out["results"]) == 300
