"""Standalone HTTP MCP server entry point (FastMCP v3, Streamable HTTP).

Serves the Phentrieve MCP facade on its own port with the MCP endpoint at /mcp.

Usage:
    phentrieve mcp serve --port 8734
"""

from __future__ import annotations

import logging

logger = logging.getLogger("phentrieve.mcp")


def main() -> None:
    """Run the Phentrieve MCP server standalone over Streamable HTTP."""
    import uvicorn

    from api.mcp.config import settings
    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    app = mcp.http_app(path="/mcp")

    logger.info(
        "Starting Phentrieve MCP server (HTTP) at http://%s:%d/mcp",
        settings.host,
        settings.port,
    )
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
