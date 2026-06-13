"""Functional tests calling tools through the assembled server.

Covers lightweight tools that need no embedding model / vector index:
get_capabilities, diagnostics, chunk_text, and compare (ontology data),
plus the argument-validation middleware and response_mode behaviour.
"""

from __future__ import annotations

import asyncio

from api.mcp.facade import create_phentrieve_mcp


def _call(tool: str, args: dict) -> dict:
    mcp = create_phentrieve_mcp()
    result = asyncio.run(mcp.call_tool(tool, args))
    return result.structured_content


def _meta_invariants(data: dict) -> None:
    meta = data["_meta"]
    assert meta["tool"]
    assert meta["request_id"]
    assert meta["unsafe_for_clinical_use"] is True
    assert meta["capabilities_version"].startswith("sha256:")


def test_get_capabilities_success_envelope():
    data = _call("phentrieve_get_capabilities", {})
    assert data["success"] is True
    assert data["server"] == "phentrieve"
    assert data["capabilities_version"].startswith("sha256:")
    _meta_invariants(data)


def test_diagnostics_reports_subsystems():
    data = _call("phentrieve_diagnostics", {})
    assert data["success"] is True
    assert "ontology_data" in data["subsystems"]
    assert "recent_errors" in data
    _meta_invariants(data)


def test_chunk_text_returns_chunks_and_next_commands():
    data = _call(
        "phentrieve_chunk_text",
        {"text": "First finding. Second finding. Third finding.", "strategy": "simple"},
    )
    assert data["success"] is True
    assert data["chunk_count"] >= 1
    assert data["chunks"][0]["chunk_id"] == 1
    assert data["_meta"]["next_commands"]
    _meta_invariants(data)


def test_arg_validation_middleware_unknown_arg():
    data = _call("phentrieve_get_capabilities", {"nope": 1})
    assert data["success"] is False
    assert data["error_code"] == "validation_failed"
    assert data["field"] == "nope"


def test_arg_alias_normalization_disclosed():
    # 'sections' is an alias for 'details' on get_capabilities
    data = _call("phentrieve_get_capabilities", {"sections": ["sample_calls"]})
    assert data["success"] is True
    assert "sample_calls" in data
    assert data["_meta"]["argument_aliases_applied"] == [["sections", "details"]]


def test_compare_real_similarity():
    data = _call(
        "phentrieve_compare_hpo_terms",
        {"term1_id": "HP:0001250", "term2_id": "HP:0002133"},
    )
    assert data["success"] is True
    assert 0.0 <= data["similarity_score"] <= 1.0
    assert data["_meta"]["next_commands"]


def test_compare_not_found_envelope():
    data = _call(
        "phentrieve_compare_hpo_terms",
        {"term1_id": "HP:0000001", "term2_id": "HP:9999999"},
    )
    assert data["success"] is False
    assert data["error_code"] == "not_found"
    assert data["retryable"] is False
    assert data["recovery_action"] == "reformulate_input"


def test_compare_standard_mode_includes_citation():
    data = _call(
        "phentrieve_compare_hpo_terms",
        {
            "term1_id": "HP:0001250",
            "term2_id": "HP:0002133",
            "response_mode": "standard",
        },
    )
    assert "recommended_citation" in data["_meta"]
    assert data["_meta"]["response_mode"] == "standard"


def test_invalid_response_mode_is_rejected():
    # Literal type -> middleware validation_failed before the body runs.
    data = _call("phentrieve_chunk_text", {"text": "x", "response_mode": "verbose"})
    assert data["success"] is False
    assert data["error_code"] == "validation_failed"
    assert data["field"] == "response_mode"


def test_bad_hpo_id_pattern_rejected():
    data = _call(
        "phentrieve_compare_hpo_terms",
        {"term1_id": "not-an-id", "term2_id": "HP:0001250"},
    )
    assert data["success"] is False
    assert data["error_code"] == "validation_failed"
