"""MCP tool registration entry points."""

from __future__ import annotations

from api.mcp.tools.discovery import register_discovery_tools
from api.mcp.tools.phenopacket import register_phenopacket_tools
from api.mcp.tools.retrieval import register_retrieval_tools
from api.mcp.tools.similarity import register_similarity_tools

__all__ = [
    "register_discovery_tools",
    "register_phenopacket_tools",
    "register_retrieval_tools",
    "register_similarity_tools",
]
