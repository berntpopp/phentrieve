"""MCP-only normalization of the shared extraction schema.

The shared ``full_text_service`` emits a schema that also powers the REST API
and the Vue curation frontend, so it carries four overlapping score fields and
two parallel chunk-index schemes. For the MCP consumer (which feeds output into
a phenopacket, often unsupervised) that redundancy is a footgun and a token
cost. This module projects the shared schema down to a single canonical shape
*only at the MCP boundary*; the shared service output is left untouched.

Canonical aggregated term: ``hpo_id``, ``label``, ``score`` (max evidence
similarity), ``assertion``, ``rank``, ``evidence_count``, ``chunk_ids`` (1-based)
and ``top_evidence_chunk_id`` (1-based), plus ``text_attributions`` /
``definition`` / ``synonyms`` when present.

Canonical chunk: ``chunk_id``, ``text``, ``status``, ``start_char``/``end_char``
when present, and ``hpo_matches`` always present (each match normalized to
``hpo_id``/``label``/``score``/``assertion``). Empty-match chunks are dropped
unless ``include_unmatched`` is set.

Addresses evaluation defects M4 (redundant schema / dual indexing), T1 (empty
chunks), L5 (omitted hpo_matches key), and L7 (id collision -- terms remain keyed
by the (hpo_id, assertion) pair so present + negated do not silently merge).
"""

from __future__ import annotations

from typing import Any

# Score copies collapsed into the single canonical ``score``.
_DROP_SCORE_FIELDS = ("avg_score", "confidence", "max_score_from_evidence")
# Old index/identity fields removed after normalization.
_DROP_AFTER_NORMALIZE = (
    "chunks",
    "top_evidence_chunk_idx",
    "source_chunk_ids",
    "id",
    "name",
    "status",
    "assertion_status",
    "count",
)


def project_aggregated_terms_for_mcp(
    terms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse aggregated terms to one score field and one chunk-index scheme."""
    projected: list[dict[str, Any]] = []
    for term in terms:
        out = dict(term)
        out["hpo_id"] = term.get("hpo_id") or term.get("id")
        out["label"] = term.get("label") or term.get("name")
        out["assertion"] = (
            term.get("assertion") or term.get("status") or term.get("assertion_status")
        )
        out["score"] = term.get("score", term.get("max_score_from_evidence", 0.0))
        if "evidence_count" not in out and "count" in term:
            out["evidence_count"] = term["count"]

        # Single 1-based chunk-index scheme.
        if term.get("source_chunk_ids") is not None:
            out["chunk_ids"] = term["source_chunk_ids"]
        elif term.get("chunks") is not None:
            out["chunk_ids"] = [c + 1 for c in term["chunks"]]
        if out.get("top_evidence_chunk_id") is None:
            idx = term.get("top_evidence_chunk_idx")
            if idx is not None:
                out["top_evidence_chunk_id"] = idx + 1

        for field in (*_DROP_SCORE_FIELDS, *_DROP_AFTER_NORMALIZE):
            out.pop(field, None)

        # Uniform schema across records: every term carries text_attributions,
        # even when empty (defect D13). Drop null padding (e.g. start_char/end_char
        # left None when positions were not requested) so the MCP payload is not
        # bloated with default-valued keys (defect D7).
        out.setdefault("text_attributions", [])
        out = {k: v for k, v in out.items() if v is not None}
        projected.append(out)
    return projected


def _project_match(match: dict[str, Any]) -> dict[str, Any]:
    out = dict(match)
    out["hpo_id"] = match.get("hpo_id") or match.get("id")
    out["label"] = match.get("label") or match.get("name")
    out["assertion"] = match.get("assertion") or match.get("assertion_status")
    if "score" in match:
        out["score"] = match["score"]
    for old in ("id", "name", "assertion_status"):
        out.pop(old, None)
    return out


def project_processed_chunks_for_mcp(
    chunks: list[dict[str, Any]], *, include_unmatched: bool = False
) -> list[dict[str, Any]]:
    """Drop empty-match chunks (unless opted in) and guarantee hpo_matches."""
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        matches = chunk.get("hpo_matches") or []
        if not matches and not include_unmatched:
            continue
        nc = dict(chunk)
        nc["hpo_matches"] = [_project_match(m) for m in matches]
        out.append(nc)
    return out


def project_extract_payload(
    payload: dict[str, Any], *, include_unmatched_chunks: bool = False
) -> dict[str, Any]:
    """Project both list fields of an extract payload in place-safe fashion."""
    out = dict(payload)
    if isinstance(payload.get("aggregated_hpo_terms"), list):
        out["aggregated_hpo_terms"] = project_aggregated_terms_for_mcp(
            payload["aggregated_hpo_terms"]
        )
    if isinstance(payload.get("processed_chunks"), list):
        out["processed_chunks"] = project_processed_chunks_for_mcp(
            payload["processed_chunks"], include_unmatched=include_unmatched_chunks
        )
    return out
