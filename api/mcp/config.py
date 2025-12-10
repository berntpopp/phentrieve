"""MCP server configuration using Pydantic settings.

This module provides configuration for the MCP server with sensible defaults
and environment variable overrides. All settings use the PHENTRIEVE_MCP_ prefix.

MCP Transport Modes:
    1. stdio (default): For Claude Desktop integration via stdin/stdout
       - Command: phentrieve mcp serve
       - No URL needed - uses command execution

    2. HTTP (same-domain): Mounts /mcp endpoint on existing API
       - Enable via: ENABLE_MCP_HTTP=true
       - URL: https://your-domain.com/mcp (same port as API)
       - Best for production Docker deployments

    3. HTTP (standalone): Separate HTTP server for MCP only
       - Command: phentrieve mcp serve --http --port 8735
       - URL: http://localhost:8735/mcp
"""

import os

from pydantic import Field
from pydantic_settings import BaseSettings


class MCPSettings(BaseSettings):
    """MCP server configuration.

    All settings can be overridden via environment variables
    with the prefix PHENTRIEVE_MCP_.

    Example:
        PHENTRIEVE_MCP_NAME=my-server phentrieve mcp serve
    """

    name: str = Field(
        default="phentrieve",
        description="MCP server name shown to clients",
    )
    description: str = Field(
        default="Extract HPO (Human Phenotype Ontology) terms from clinical text",
        description="MCP server description shown to clients",
    )

    # HTTP transport settings (for --http mode or same-domain mounting)
    host: str = Field(
        default="127.0.0.1",
        description="Host to bind for standalone HTTP transport",
    )
    port: int = Field(
        default=8734,
        description="Port to bind for standalone HTTP transport",
    )

    # Same-domain mounting: enable via ENABLE_MCP_HTTP=true
    # When enabled, /mcp endpoint is mounted on the main API at startup
    enable_http: bool = Field(
        default=False,
        description="Mount MCP at /mcp on the main API (for Docker/production)",
    )

    model_config = {"env_prefix": "PHENTRIEVE_MCP_"}


def is_mcp_http_enabled() -> bool:
    """Check if MCP HTTP mounting is enabled.

    Checks both PHENTRIEVE_MCP_ENABLE_HTTP and ENABLE_MCP_HTTP env vars
    for flexibility in Docker configuration.

    Returns:
        True if MCP should be mounted at /mcp on the main API.
    """
    # Check both env var formats for convenience
    return settings.enable_http or os.getenv("ENABLE_MCP_HTTP", "").lower() in (
        "true",
        "1",
        "yes",
    )


# Singleton instance - import this to use settings
settings = MCPSettings()
