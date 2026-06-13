"""MCP server CLI commands.

This module provides CLI commands for managing the MCP server.
Commands are registered conditionally based on whether fastapi-mcp is installed.
"""

from typing import Annotated

import typer

app = typer.Typer(
    name="mcp",
    help="Model Context Protocol (MCP) server commands.",
    no_args_is_help=True,
)


def _check_mcp_installed() -> bool:
    """Check if MCP dependencies are installed."""
    try:
        import fastmcp  # noqa: F401

        return True
    except ImportError:
        return False


@app.command("serve")
def serve_mcp(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port for the Streamable HTTP MCP server.",
        ),
    ] = 8734,
) -> None:
    """Start the Phentrieve MCP server over Streamable HTTP (endpoint at /mcp).

    Phentrieve serves MCP only over Streamable HTTP (stdio has been removed).

    Examples:
        # Standalone HTTP MCP server
        phentrieve mcp serve --port 8734
    """
    if not _check_mcp_installed():
        typer.echo(
            "Error: MCP support requires the 'mcp' extra.\n"
            "Install with: pip install phentrieve[mcp]",
            err=True,
        )
        raise typer.Exit(1)

    import os

    os.environ["PHENTRIEVE_MCP_PORT"] = str(port)

    from api.mcp.http_server import main as http_main

    typer.echo(f"Starting Phentrieve MCP server (HTTP) at http://127.0.0.1:{port}/mcp")
    http_main()


@app.command("info")
def mcp_info() -> None:
    """Display MCP server configuration and Claude Desktop setup."""
    if not _check_mcp_installed():
        typer.echo(
            "Error: MCP support requires the 'mcp' extra.\n"
            "Install with: pip install phentrieve[mcp]",
            err=True,
        )
        raise typer.Exit(1)

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from api.mcp.config import settings
    from api.mcp.resources import get_capabilities_resource

    console = Console()

    # Tool table
    table = Table(title="Available MCP Tools", show_header=True)
    table.add_column("Tool Name", style="cyan", no_wrap=True)
    table.add_column("Endpoint")
    table.add_column("Description")

    tool_info = {
        "phentrieve_search_hpo_terms": (
            "/mcp",
            "Map a short research phenotype phrase to HPO candidates",
        ),
        "phentrieve_extract_hpo_terms": (
            "/mcp",
            "Quick deterministic research screening without LLM calls",
        ),
        "phentrieve_extract_hpo_terms_llm": (
            "/mcp",
            "LLM-assisted full-text research annotation",
        ),
        "phentrieve_compare_hpo_terms": (
            "/mcp",
            "Compare two HPO IDs for research similarity analysis",
        ),
        "phentrieve_export_phenopacket": (
            "/mcp",
            "Serialize annotations to a GA4GH Phenopacket v2 bundle",
        ),
        "phentrieve_chunk_text": (
            "/mcp",
            "Chunk text without retrieval, for client-driven loops",
        ),
        "phentrieve_get_capabilities": (
            "/mcp",
            "Report tools, response modes, limits, and the citation contract",
        ),
        "phentrieve_diagnostics": (
            "/mcp",
            "Subsystem health and recent errors",
        ),
    }

    for tool_name in get_capabilities_resource()["tools"]:
        endpoint, desc = tool_info.get(tool_name, ("/mcp", ""))
        table.add_row(tool_name, endpoint, desc)

    console.print(table)

    # Configuration
    console.print("\n[bold]Server Configuration:[/bold]")
    console.print(f"  Name: {settings.name}")
    console.print(f"  HTTP Host: {settings.host}")
    console.print(f"  HTTP Port: {settings.port}")
    console.print("  HTTP Transport: Streamable HTTP")
    console.print(f"  MCP URL: http://{settings.host}:{settings.port}/mcp")

    # Claude Desktop config
    config_json = """{
  "mcpServers": {
    "phentrieve": {
      "type": "http",
      "url": "http://127.0.0.1:8734/mcp"
    }
  }
}"""

    console.print("\n")
    console.print(
        Panel(
            config_json,
            title="Claude Desktop Configuration",
            subtitle="Add to ~/.config/claude/config.json",
        )
    )
