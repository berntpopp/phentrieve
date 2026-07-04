"""B2 Task 9: family_history_findings + experiencer/excluded on the MCP surface.

Covers projection (family list projected, experiencer/excluded survive), the
EXTRACT_SCHEMA envelope, the capabilities descriptor documenting the new fields
(so ``capabilities_version`` rolls), and independent budgeting of the family
list. The capabilities hash-stability invariants are re-asserted rather than
pinned to a hard-coded hash (F6).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from api.mcp.capabilities import build_capabilities, capabilities_version
from api.mcp.projection import (
    project_aggregated_terms_for_mcp,
    project_extract_payload,
)
from api.mcp.schemas import EXTRACT_SCHEMA

pytestmark = pytest.mark.unit


PROBAND_NEGATED = {
    "id": "HP:0001945",
    "name": "Fever",
    "score": 0.7,
    "status": "negated",
    "experiencer": "proband",
    "excluded": True,
    "evidence_count": 1,
    "source_chunk_ids": [1],
}
FAMILY_TERM = {
    "id": "HP:0002076",
    "name": "Migraine",
    "score": 0.8,
    "status": "present",
    "experiencer": "family_history",
    "excluded": False,
    "evidence_count": 1,
    "source_chunk_ids": [1],
}


def test_projection_preserves_experiencer_and_excluded():
    out = project_aggregated_terms_for_mcp([PROBAND_NEGATED])[0]
    assert out["experiencer"] == "proband"
    assert out["excluded"] is True
    assert out["assertion"] == "negated"


def test_project_extract_payload_projects_family_list():
    payload = {
        "aggregated_hpo_terms": [PROBAND_NEGATED],
        "family_history_findings": [FAMILY_TERM],
        "processed_chunks": [],
        "meta": {},
    }
    out = project_extract_payload(payload)

    assert "family_history_findings" in out
    fam = out["family_history_findings"][0]
    assert fam["hpo_id"] == "HP:0002076"
    assert fam["experiencer"] == "family_history"
    assert fam["excluded"] is False

    agg = out["aggregated_hpo_terms"][0]
    assert agg["experiencer"] == "proband"
    assert agg["excluded"] is True


def test_extract_schema_exposes_family_history_findings():
    props = EXTRACT_SCHEMA["properties"]
    assert "family_history_findings" in props
    # No regression: the proband list stays.
    assert "aggregated_hpo_terms" in props


def test_capabilities_descriptor_documents_new_output_fields():
    caps = build_capabilities()
    serialized = json.dumps(caps, default=str, sort_keys=True)
    assert "family_history_findings" in serialized
    assert "experiencer" in serialized
    assert "excluded" in serialized


def test_capabilities_version_hash_stability_preserved():
    base = build_capabilities()
    detailed = build_capabilities(details=["sample_calls", "argument_aliases"])
    # capabilities_version is a stable warm-cache key across the details expansion.
    assert base["capabilities_version"] == capabilities_version()
    assert detailed["capabilities_version"] == capabilities_version()
    # The detailed descriptor keeps its own distinct content hash.
    assert detailed["descriptor_hash"] != base["descriptor_hash"]


def _call_extract(monkeypatch, service_payload: dict, args: dict) -> dict:
    monkeypatch.setattr(
        "api.mcp.tools.retrieval.extract_hpo_terms_service",
        lambda **_kwargs: service_payload,
    )
    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    result = asyncio.run(mcp.call_tool("phentrieve_extract_hpo_terms", args))
    return result.structured_content


def test_extract_tool_surfaces_family_findings(monkeypatch):
    payload = {
        "meta": {"extraction_backend": "standard"},
        "processed_chunks": [],
        "aggregated_hpo_terms": [PROBAND_NEGATED],
        "family_history_findings": [FAMILY_TERM],
    }
    data = _call_extract(
        monkeypatch,
        payload,
        {"text": "x", "research_use_acknowledged": True, "response_mode": "compact"},
    )

    assert data["success"] is True
    assert "family_history_findings" in data
    fam = data["family_history_findings"][0]
    assert fam["hpo_id"] == "HP:0002076"
    assert fam["experiencer"] == "family_history"
    assert fam["excluded"] is False


def test_family_list_budgeted_independently(monkeypatch):
    """A large family list is truncated + disclosed, never silently exempted."""
    big_family = [
        {
            "id": f"HP:{i:07d}",
            "name": f"Some longer phenotype label number {i}",
            "score": 0.5,
            "status": "present",
            "experiencer": "family_history",
            "excluded": False,
            "evidence_count": 1,
            "source_chunk_ids": [1],
        }
        for i in range(200)
    ]
    payload = {
        "meta": {"extraction_backend": "standard"},
        "processed_chunks": [],
        # aggregated omitted so the family list is the sole budget pressure.
        "family_history_findings": big_family,
    }
    data = _call_extract(
        monkeypatch,
        payload,
        {"text": "x", "research_use_acknowledged": True, "response_mode": "minimal"},
    )

    assert len(data["family_history_findings"]) < 200
    trunc = data["_meta"]["family_history_findings_truncated"]
    assert trunc["field"] == "family_history_findings"
    assert trunc["total"] == 200
