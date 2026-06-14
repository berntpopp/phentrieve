"""GA4GH Phenopacket export tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import anyio
from pydantic import Field

from api.mcp.annotations import READ_ONLY_OPEN_WORLD
from api.mcp.envelope import McpErrorContext, run_mcp_tool
from api.mcp.schemas import PHENOPACKET_SCHEMA
from api.mcp.service_adapters import export_phenopacket_service
from api.mcp.shaping import apply_response_mode, resolve_mode
from api.mcp.tools._common import ResponseMode

if TYPE_CHECKING:
    from fastmcp import FastMCP

_Phenotypes = Annotated[
    list[dict[str, Any]],
    Field(
        min_length=1,
        description="Annotations to serialize: a list of "
        "{hpo_id, label, assertion} objects (assertion: affirmed|negated). "
        "The raw extractor shape {id, name, assertion_status} is also accepted, "
        "and a per-item score is preserved as the evidence confidence, so you "
        "can hand it the aggregated_hpo_terms from an extract call unchanged.",
    ),
]

_Subject = Annotated[
    dict[str, Any] | None,
    Field(
        description="Optional subject metadata: {id, sex, date_of_birth}. "
        "sex is UNKNOWN_SEX|FEMALE|MALE|OTHER_SEX.",
    ),
]


def register_phenopacket_tools(mcp: FastMCP) -> None:
    """Register the export_phenopacket tool."""

    @mcp.tool(
        name="phentrieve_export_phenopacket",
        title="Export GA4GH Phenopacket",
        annotations=READ_ONLY_OPEN_WORLD,
        output_schema=PHENOPACKET_SCHEMA,
        description=(
            "Serialize an HPO annotation set into a GA4GH Phenopacket v2 JSON "
            "bundle (optionally with an annotation sidecar). Research use only; "
            "not for clinical record generation. Signature: "
            "phentrieve_export_phenopacket(case_id, phenotypes, case_label=, "
            "input_text=, subject=, include_annotation_sidecar=, response_mode=)."
        ),
    )
    async def export_phenopacket(
        case_id: Annotated[str, Field(min_length=1, description="Case identifier.")],
        phenotypes: _Phenotypes,
        case_label: str | None = None,
        input_text: str | None = None,
        subject: _Subject = None,
        include_annotation_sidecar: bool = True,
        response_mode: ResponseMode = "compact",
    ) -> dict[str, Any]:
        mode = resolve_mode(response_mode)

        async def call() -> dict[str, Any]:
            raw = await anyio.to_thread.run_sync(
                lambda: export_phenopacket_service(
                    case_id=case_id,
                    case_label=case_label,
                    input_text=input_text,
                    subject=subject,
                    phenotypes=phenotypes,
                    include_annotation_sidecar=include_annotation_sidecar,
                )
            )
            shaped = apply_response_mode(raw, mode)
            shaped["_meta"] = {"next_commands": []}
            return shaped

        return await run_mcp_tool(
            "phentrieve_export_phenopacket",
            call,
            response_mode=mode,
            context=McpErrorContext(
                "phentrieve_export_phenopacket", {"case_id": case_id}
            ),
        )
