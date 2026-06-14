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


def test_text_attributions_empty_array_kept_at_compact():
    """D1: a semantic-only match has no literal span, so text_attributions is an
    empty array. It must stay present as [] at compact (one contract: always an
    array; empty == semantic match) rather than being dropped like an empty
    detail field, which made the key inconsistently '[] / missing / populated'."""
    payload = {
        "aggregated_hpo_terms": [
            {"hpo_id": "HP:0001250", "score": 0.83, "text_attributions": []}
        ]
    }
    term = apply_response_mode(payload, "compact")["aggregated_hpo_terms"][0]
    assert "text_attributions" in term
    assert term["text_attributions"] == []


def test_meta_passes_through_unshaped():
    payload = {"results": [], "_meta": {"definition": "keep-me"}}
    out = apply_response_mode(payload, "minimal")
    assert out["_meta"] == {"definition": "keep-me"}


def test_phenopacket_json_blob_gated_to_standard_and_full():
    """R1: the serialized phenopacket_json string is a top-level detail field --
    dropped at minimal/compact so the canonical `phenopacket` object is the one
    form, kept at standard/full for consumers that want the serialized blob. The
    `phenopacket` object itself is opaque and survives WHOLE at every mode (it is
    a complete GA4GH document, never field-projected to {})."""
    packet = {
        "id": "CASE-1",
        "phenotypicFeatures": [{"type": {"id": "HP:0001250"}}],
    }
    raw = {"phenopacket": packet, "phenopacket_json": '{"id": "CASE-1"}'}
    opaque = ("phenopacket",)

    minimal = apply_response_mode(raw, "minimal", opaque_keys=opaque)
    assert minimal["phenopacket"] == packet  # not gutted to {}
    assert "phenopacket_json" not in minimal

    compact = apply_response_mode(raw, "compact", opaque_keys=opaque)
    assert compact["phenopacket"] == packet
    assert "phenopacket_json" not in compact

    standard = apply_response_mode(raw, "standard", opaque_keys=opaque)
    assert standard["phenopacket"] == packet
    assert "phenopacket_json" in standard

    full = apply_response_mode(raw, "full", opaque_keys=opaque)
    assert full["phenopacket"] == packet
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
