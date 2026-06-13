"""Shared MCP tool annotations (read-only research server).

Patterned after ../hgnc-link/hgnc_link/mcp/annotations.py.
"""

from __future__ import annotations

from mcp.types import ToolAnnotations

# Tools that read from models / ontology data and may touch external resources
# (embedding models, LLM backend): read-only, idempotent, open-world.
READ_ONLY_OPEN_WORLD = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)

# Introspection tools (capabilities, diagnostics) that only describe this server:
# read-only, idempotent, closed-world.
READ_ONLY_CLOSED_WORLD = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
