"""Unit tests for api.mcp.schemas (permissive output schemas)."""

from __future__ import annotations

from api.mcp.schemas import (
    CAPABILITIES_SCHEMA,
    COMPARE_SCHEMA,
    EXTRACT_SCHEMA,
    PHENOPACKET_SCHEMA,
    SEARCH_SCHEMA,
    envelope_schema,
)


def test_envelope_schema_is_permissive_with_common_keys():
    s = envelope_schema(results={"type": "array"})
    assert s["type"] == "object"
    assert s["additionalProperties"] is True
    for key in (
        "success",
        "_meta",
        "error_code",
        "message",
        "retryable",
        "recovery_action",
    ):
        assert key in s["properties"]
    assert "results" in s["properties"]


def test_domain_schemas_expose_domain_keys():
    assert "results" in SEARCH_SCHEMA["properties"]
    assert "aggregated_hpo_terms" in EXTRACT_SCHEMA["properties"]
    assert "similarity_score" in COMPARE_SCHEMA["properties"]
    assert "phenopacket_json" in PHENOPACKET_SCHEMA["properties"]
    assert "capabilities_version" in CAPABILITIES_SCHEMA["properties"]
