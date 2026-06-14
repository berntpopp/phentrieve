"""Ontology similarity tool: compare two HPO terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio

from api.mcp.annotations import READ_ONLY_OPEN_WORLD
from api.mcp.envelope import McpErrorContext, run_mcp_tool
from api.mcp.next_commands import after_compare
from api.mcp.resources import recommended_citation
from api.mcp.schemas import COMPARE_SCHEMA
from api.mcp.service_adapters import compare_hpo_terms_service
from api.mcp.shaping import apply_response_mode, resolve_mode
from api.mcp.tools._common import HpoIdArg, ResponseMode, SimilarityFormulaArg

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register_similarity_tools(mcp: FastMCP) -> None:
    """Register the compare_hpo_terms tool."""

    @mcp.tool(
        name="phentrieve_compare_hpo_terms",
        title="Compare HPO Terms",
        annotations=READ_ONLY_OPEN_WORLD,
        output_schema=COMPARE_SCHEMA,
        description=(
            "Compute ontology semantic similarity between two HPO ids for research "
            "similarity analysis. A missing id returns a not_found error envelope. "
            "Research use only; not for clinical use. Signature: "
            "phentrieve_compare_hpo_terms(term1_id, term2_id, formula=, response_mode=)."
        ),
    )
    async def compare_hpo_terms(
        term1_id: HpoIdArg,
        term2_id: HpoIdArg,
        formula: SimilarityFormulaArg = "hybrid",
        response_mode: ResponseMode = "compact",
    ) -> dict[str, Any]:
        mode = resolve_mode(response_mode)

        async def call() -> dict[str, Any]:
            raw = await anyio.to_thread.run_sync(
                lambda: compare_hpo_terms_service(
                    term1_id=term1_id, term2_id=term2_id, formula=formula
                )
            )
            shaped = apply_response_mode(raw, mode)
            meta: dict[str, Any] = {
                "next_commands": after_compare(term1_id, term2_id, formula)
            }
            if mode in ("standard", "full"):
                meta["recommended_citation"] = recommended_citation()
            shaped["_meta"] = meta
            return shaped

        return await run_mcp_tool(
            "phentrieve_compare_hpo_terms",
            call,
            response_mode=mode,
            context=McpErrorContext(
                "phentrieve_compare_hpo_terms",
                {"term1_id": term1_id, "term2_id": term2_id},
            ),
        )
