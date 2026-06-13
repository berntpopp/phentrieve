"""Builders for ``_meta.next_commands`` entries: ``{tool, arguments}`` steps.

Every successful list/record response and every error envelope carries a small
set of ready-to-call next steps so an LLM client can chain the canonical
workflow without re-reading the capabilities surface.
"""

from __future__ import annotations

from typing import Any


def cmd(tool: str, **arguments: Any) -> dict[str, Any]:
    """One ready-to-call next step."""
    return {"tool": tool, "arguments": arguments}


def after_search(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """After search: compare the top hits, or widen if nothing matched."""
    ids = [r.get("hpo_id") for r in results if r.get("hpo_id")]
    if len(ids) >= 2:
        return [cmd("phentrieve_compare_hpo_terms", term1_id=ids[0], term2_id=ids[1])]
    if not ids:
        return [cmd("phentrieve_get_capabilities", details=["languages", "models"])]
    return [cmd("phentrieve_extract_hpo_terms", text="<surrounding clinical text>")]


def after_extract(aggregated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """After extract: hand the aggregated terms to the phenopacket exporter."""
    phenotypes = [
        {
            "hpo_id": t.get("hpo_id"),
            "label": t.get("label") or t.get("name"),
            "assertion": t.get("status", "affirmed"),
        }
        for t in aggregated[:25]
        if t.get("hpo_id")
    ]
    if not phenotypes:
        return [cmd("phentrieve_get_capabilities", details=["models"])]
    return [
        cmd("phentrieve_export_phenopacket", case_id="<case-id>", phenotypes=phenotypes)
    ]


def after_compare(term1_id: str, term2_id: str) -> list[dict[str, Any]]:
    """After compare: search for a related phenotype phrase."""
    return [cmd("phentrieve_search_hpo_terms", text="<related phenotype phrase>")]


def after_chunk(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """After chunk: search HPO terms for the first chunk's text."""
    if not chunks:
        return [cmd("phentrieve_get_capabilities")]
    return [cmd("phentrieve_search_hpo_terms", text=chunks[0].get("text", ""))]


def default_error_next_commands(tool_name: str) -> list[dict[str, Any]]:
    """A sensible recovery step for any error lacking an explicit fallback."""
    return [
        cmd("phentrieve_get_capabilities"),
        cmd("phentrieve_diagnostics"),
    ]
