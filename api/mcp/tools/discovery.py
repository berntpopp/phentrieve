"""Discovery tools: capabilities and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import anyio
from pydantic import Field

from api.mcp.annotations import READ_ONLY_CLOSED_WORLD
from api.mcp.capabilities import build_capabilities
from api.mcp.envelope import McpErrorContext, run_mcp_tool
from api.mcp.schemas import CAPABILITIES_SCHEMA, DIAGNOSTICS_SCHEMA
from api.mcp.service_adapters import diagnostics_service

if TYPE_CHECKING:
    from fastmcp import FastMCP

_Details = Annotated[
    list[str] | None,
    Field(
        description="Optional capability sections to expand, e.g. "
        "['sample_calls', 'argument_aliases'].",
    ),
]


def register_discovery_tools(mcp: FastMCP) -> None:
    """Register the get_capabilities and diagnostics tools."""

    @mcp.tool(
        name="phentrieve_get_capabilities",
        title="Get Phentrieve Capabilities",
        annotations=READ_ONLY_CLOSED_WORLD,
        output_schema=CAPABILITIES_SCHEMA,
        description=(
            "Return the server capability surface: tools, response modes, limits, "
            "error codes, citation contract, and a stable capabilities_version "
            "(compare it to _meta.capabilities_version to skip re-fetching). Pass "
            "details=['sample_calls'] to expand. Signature: "
            "phentrieve_get_capabilities(details=)."
        ),
    )
    async def get_capabilities(details: _Details = None) -> dict[str, Any]:
        async def call() -> dict[str, Any]:
            return build_capabilities(details=details)

        return await run_mcp_tool(
            "phentrieve_get_capabilities",
            call,
            context=McpErrorContext("phentrieve_get_capabilities"),
        )

    @mcp.tool(
        name="phentrieve_diagnostics",
        title="Phentrieve Diagnostics",
        annotations=READ_ONLY_CLOSED_WORLD,
        output_schema=DIAGNOSTICS_SCHEMA,
        description=(
            "Report subsystem health (ontology data, embedding model, LLM backend, "
            "vector index) and recent sanitized errors for troubleshooting. "
            "Signature: phentrieve_diagnostics()."
        ),
    )
    async def diagnostics() -> dict[str, Any]:
        async def call() -> dict[str, Any]:
            return await anyio.to_thread.run_sync(diagnostics_service)

        return await run_mcp_tool(
            "phentrieve_diagnostics",
            call,
            context=McpErrorContext("phentrieve_diagnostics"),
        )
