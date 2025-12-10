"""HTTP MCP server entry point.

This module provides an HTTP transport for the MCP server, useful for
web-based MCP clients. The MCP endpoint is mounted at /mcp.

Usage:
    phentrieve mcp serve --http --port 8734
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Run MCP server with HTTP transport.

    Starts the FastAPI server with MCP mounted at /mcp endpoint.
    """
    import uvicorn

    from api.main import app
    from api.mcp.config import settings
    from api.mcp.server import create_mcp_server

    mcp = create_mcp_server(app)

    # Mount MCP at /mcp endpoint
    mcp.mount()

    logger.info(
        "Starting Phentrieve MCP server (HTTP) at http://%s:%d/mcp",
        settings.host,
        settings.port,
    )

    # Run the combined server
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
