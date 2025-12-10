"""MCP server module for Phentrieve.

This module provides Model Context Protocol (MCP) server functionality,
enabling AI assistants like Claude to directly query HPO terms.
"""

from api.mcp.server import MCP_ALLOWED_OPERATIONS, create_mcp_server

__all__ = ["create_mcp_server", "MCP_ALLOWED_OPERATIONS"]
