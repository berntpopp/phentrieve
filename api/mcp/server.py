"""MCP Server implementation for Phentrieve.

This module creates an MCP server from the existing FastAPI application
using fastapi-mcp. It follows the Single Responsibility Principle by
keeping MCP concerns separate from the main API.

Key design decisions:
- Uses include_operations (allowlist) instead of exclude (more explicit)
- Does NOT modify api/main.py (separation of concerns)
- Reuses existing FastAPI schemas (DRY principle)
- Factory pattern for testability

Usage:
    # As CLI entry point (stdio transport for Claude Desktop)
    phentrieve-mcp

    # Or programmatically
    from api.mcp.server import create_mcp_server
    mcp = create_mcp_server(app)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Explicit allowlist of operations to expose as MCP tools
# This is safer than exclude - you know exactly what's exposed
# These must match the operation_id values in the router decorators
MCP_ALLOWED_OPERATIONS: list[str] = [
    "query_hpo_terms",  # GET /api/v1/query/
    "process_clinical_text",  # POST /api/v1/text/process
    "calculate_term_similarity",  # GET /api/v1/similarity/{id}/{id}
]


def create_mcp_server(app: FastAPI) -> Any:  # Returns FastApiMCP when installed
    """Create MCP server from FastAPI application.

    Args:
        app: FastAPI application instance with routes that have operation_id set.

    Returns:
        Configured FastApiMCP instance ready for mounting or stdio transport.

    Raises:
        ImportError: If fastapi-mcp is not installed.

    Note:
        This function does NOT call mcp.mount() - the caller decides
        whether to mount (HTTP mode) or run stdio (CLI mode).
    """
    # Import here to make fastapi-mcp an optional dependency
    from fastapi_mcp import FastApiMCP

    from api.mcp.config import settings

    mcp = FastApiMCP(
        app,
        name=settings.name,
        description=settings.description,
        # Allowlist pattern - explicit is better than implicit
        include_operations=MCP_ALLOWED_OPERATIONS,
    )

    logger.info(
        "MCP server '%s' created with %d tools: %s",
        settings.name,
        len(MCP_ALLOWED_OPERATIONS),
        ", ".join(MCP_ALLOWED_OPERATIONS),
    )

    return mcp


def main() -> None:
    """Entry point for MCP server (stdio transport).

    This is the entry point for the `phentrieve-mcp` command.
    It runs the MCP server in stdio mode for Claude Desktop integration.

    The server communicates via stdin/stdout using the MCP protocol.
    Do not print anything to stdout in this mode.
    """
    import asyncio

    # Configure logging to stderr only (stdout is for MCP protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler()],  # Goes to stderr by default
    )

    # Import app here to avoid circular imports
    from api.main import app

    mcp = create_mcp_server(app)

    logger.info("Starting Phentrieve MCP server (stdio transport)...")

    # Run in stdio mode (for Claude Desktop)
    asyncio.run(mcp.run_async())


if __name__ == "__main__":
    main()
