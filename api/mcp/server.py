"""MCP Server factory for Phentrieve.

This module creates an MCP server from the existing FastAPI application
using fastapi-mcp. It follows the Single Responsibility Principle by
keeping MCP concerns separate from the main API.

Key design decisions:
- Uses include_operations (allowlist) instead of exclude (more explicit)
- Does NOT import api/main.py to avoid cyclic imports (see cli.py for entry point)
- Reuses existing FastAPI schemas (DRY principle)
- Factory pattern for testability

Architecture (avoiding cyclic imports):
    - api/main.py imports api/mcp/server.py for HTTP mounting
    - api/mcp/cli.py imports api/main.py for CLI entry point
    - These are separate files, breaking the import cycle

Usage:
    # Programmatically (for HTTP mounting in api/main.py)
    from api.mcp.server import create_mcp_server
    mcp = create_mcp_server(app)
    mcp.mount()  # Mount at /mcp

    # CLI entry point is in cli.py (phentrieve-mcp command)
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

        The CLI entry point (main()) is in cli.py to avoid cyclic imports.
        This module does NOT import api.main, breaking the import cycle:
          - api/main.py -> api/mcp/server.py (for HTTP mounting)
          - api/mcp/cli.py -> api/main.py (for CLI entry point)
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
