"""CLI entry point for MCP server (stdio transport).

This module is intentionally separate from server.py to avoid cyclic imports.
The import chain is:
  - api/main.py -> api/mcp/server.py (for HTTP mounting)
  - api/mcp/cli.py -> api/main.py (for CLI entry point)

By keeping these in separate files, there's no import cycle.

Usage:
    # As CLI entry point (stdio transport for Claude Desktop)
    phentrieve-mcp

    # Or via Typer CLI
    phentrieve mcp serve
"""

from __future__ import annotations

import asyncio
import logging


def main() -> None:
    """Entry point for MCP server (stdio transport).

    This is the entry point for the `phentrieve-mcp` command.
    It runs the MCP server in stdio mode for Claude Desktop integration.

    The server communicates via stdin/stdout using the MCP protocol.
    Do not print anything to stdout in this mode.
    """
    # Configure logging to stderr only (stdout is for MCP protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler()],  # Goes to stderr by default
    )

    logger = logging.getLogger(__name__)

    # Import app here - this is the only place that imports api.main
    # from within the api/mcp/ module, avoiding cyclic imports
    from api.main import app
    from api.mcp.server import create_mcp_server

    mcp = create_mcp_server(app)

    logger.info("Starting Phentrieve MCP server (stdio transport)...")

    # Run in stdio mode (for Claude Desktop)
    asyncio.run(mcp.run_async())


if __name__ == "__main__":
    main()
