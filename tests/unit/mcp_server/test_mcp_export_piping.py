"""Export must consume extractor output directly (M2), preserve retrieval
confidence (H3), and raise a typed actionable error instead of a raw KeyError
(M3)."""

import json

import pytest

from api.mcp.envelope import McpToolError
from api.mcp.service_adapters import export_phenopacket_service

pytestmark = pytest.mark.unit


def test_export_accepts_raw_extractor_keys():
    """M2: aggregated_hpo_terms shape {id, name, assertion_status} pipes directly."""
    out = export_phenopacket_service(
        case_id="CASE-1",
        case_label=None,
        input_text=None,
        subject=None,
        phenotypes=[
            {"id": "HP:0001250", "name": "Seizure", "assertion_status": "affirmed"}
        ],
        include_annotation_sidecar=False,
    )
    packet = json.loads(out["phenopacket_json"])
    assert packet["id"] == "CASE-1"


def test_export_preserves_retrieval_confidence():
    """H3: a score on the input becomes the phenopacket evidence confidence."""
    out = export_phenopacket_service(
        case_id="CASE-2",
        case_label=None,
        input_text=None,
        subject=None,
        phenotypes=[
            {
                "id": "HP:0001250",
                "name": "Seizure",
                "assertion_status": "affirmed",
                "score": 0.91,
            }
        ],
        include_annotation_sidecar=False,
    )
    blob = out["phenopacket_json"]
    assert "0.0000" not in blob
    assert "0.9100" in blob


def test_export_negated_assertion_round_trips():
    out = export_phenopacket_service(
        case_id="CASE-3",
        case_label=None,
        input_text=None,
        subject=None,
        phenotypes=[
            {"id": "HP:0002315", "name": "Headache", "assertion_status": "negated"}
        ],
        include_annotation_sidecar=False,
    )
    packet = json.loads(out["phenopacket_json"])
    feats = packet.get("phenotypicFeatures", [])
    assert any(f.get("excluded") for f in feats)


def test_export_missing_id_raises_validation_failed():
    """M3: missing id yields a typed actionable error, not a raw KeyError repr."""
    with pytest.raises(McpToolError) as ei:
        export_phenopacket_service(
            case_id="CASE-4",
            case_label=None,
            input_text=None,
            subject=None,
            phenotypes=[{"name": "Seizure", "assertion_status": "affirmed"}],
            include_annotation_sidecar=False,
        )
    assert ei.value.error_code == "validation_failed"
    msg = str(ei.value)
    assert "hpo_id" in msg
    assert "id" in msg  # names the keys actually received / the mapping hint
