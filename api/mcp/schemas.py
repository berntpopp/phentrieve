"""JSON output schemas for the typed MCP tools (MCP structured output).

The schemas are deliberately **permissive** (``additionalProperties: true``,
nothing ``required``) because ``response_mode`` projects fields out and the same
tool body returns either a success payload or an error envelope -- both must
validate against the one declared schema.

Patterned after ../hgnc-link/hgnc_link/mcp/schemas.py.
"""

from __future__ import annotations

from typing import Any

_META = {"type": "object", "additionalProperties": True}
_STR = {"type": "string"}
_INT = {"type": "integer"}
_NUM = {"type": "number"}
_BOOL = {"type": "boolean"}
_ARR = {"type": "array"}
_OBJ = {"type": "object", "additionalProperties": True}


def envelope_schema(**properties: Any) -> dict[str, Any]:
    """A permissive object schema carrying the common envelope keys + extras."""
    props: dict[str, Any] = {
        "success": _BOOL,
        "_meta": _META,
        "error_code": _STR,
        "message": _STR,
        "retryable": _BOOL,
        "recovery_action": _STR,
        "details": _OBJ,
        "field": _STR,
        "allowed_values": _ARR,
        "hint": _STR,
        **properties,
    }
    return {"type": "object", "additionalProperties": True, "properties": props}


SEARCH_SCHEMA = envelope_schema(results=_ARR)
EXTRACT_SCHEMA = envelope_schema(
    meta=_OBJ,
    processed_chunks=_ARR,
    aggregated_hpo_terms=_ARR,
)
COMPARE_SCHEMA = envelope_schema(
    term1_id=_STR,
    term2_id=_STR,
    formula_used=_STR,
    similarity_score=_NUM,
    lca_details=_OBJ,
)
PHENOPACKET_SCHEMA = envelope_schema(
    phenopacket_json=_STR,
    annotation_sidecar=_OBJ,
)
CHUNK_SCHEMA = envelope_schema(chunks=_ARR, chunk_count=_INT)
CAPABILITIES_SCHEMA = envelope_schema(
    server=_STR,
    version=_STR,
    transport=_STR,
    tools=_OBJ,
    response_modes=_OBJ,
    error_codes=_ARR,
    capabilities_version=_STR,
    descriptor_chars=_INT,
)
DIAGNOSTICS_SCHEMA = envelope_schema(
    status=_STR,
    subsystems=_OBJ,
    recent_errors=_ARR,
    minimum_workflow=_ARR,
    capabilities_version=_STR,
)
