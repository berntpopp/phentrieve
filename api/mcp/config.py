"""MCP server configuration using Pydantic settings.

This module provides configuration for the MCP server with sensible defaults
and environment variable overrides. All settings use the PHENTRIEVE_MCP_ prefix.
"""

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

    # HTTP transport settings (for --http mode)
    host: str = Field(
        default="127.0.0.1",
        description="Host to bind for HTTP transport",
    )
    port: int = Field(
        default=8734,
        description="Port to bind for HTTP transport",
    )

    model_config = {"env_prefix": "PHENTRIEVE_MCP_"}


# Singleton instance - import this to use settings
settings = MCPSettings()
