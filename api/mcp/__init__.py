"""MCP server module for Phentrieve (Streamable HTTP, FastMCP v3).

Enables AI assistants to query HPO terms, extract phenotypes from research text,
compare terms, and export GA4GH Phenopackets via the Model Context Protocol.
"""

from api.mcp.facade import create_phentrieve_mcp
from api.mcp.server import mount_phentrieve_mcp_facade

__all__ = ["create_phentrieve_mcp", "mount_phentrieve_mcp_facade"]
