"""Main CLI entry point for Phentrieve.

This module defines the main Typer application and imports all subcommands
from their respective modules.
"""

import importlib.metadata
from pathlib import Path
from typing import Annotated, Optional

import typer

# Import all command groups
from phentrieve.cli import (
    benchmark_commands,
    data_commands,
    index_commands,
    mcp_commands,
    query_commands,
    similarity_commands,
    text_commands,
)

# Read version from pyproject.toml
__version__ = importlib.metadata.version("phentrieve")

# Create the main Typer app
app = typer.Typer(
    name="phentrieve",
    help="Phentrieve - AI-powered HPO term mapping using Retrieval-Augmented Generation (RAG)",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _get_hpo_info() -> dict[str, Optional[str | int]]:
    """Load HPO metadata from database for version display.

    Returns:
        Dictionary with version and term_count, or None values if unavailable.
    """
    result: dict[str, Optional[str | int]] = {"version": None, "term_count": None}

    try:
        from phentrieve.config import DEFAULT_HPO_DB_FILENAME
        from phentrieve.data_processing.hpo_database import HPODatabase
        from phentrieve.utils import get_default_data_dir

        # Try default data directory first
        db_path = get_default_data_dir() / DEFAULT_HPO_DB_FILENAME
        if not db_path.exists():
            # Fallback to relative path
            db_path = Path.cwd() / "data" / DEFAULT_HPO_DB_FILENAME

        if not db_path.exists():
            return result

        with HPODatabase(db_path) as db:
            result["version"] = db.get_metadata("hpo_version")
            result["term_count"] = db.get_term_count()
    except Exception:  # noqa: S110 - intentional silent fail for version display
        pass

    return result


def version_callback(value: bool):
    """Display version information and exit."""
    if value:
        typer.echo(f"Phentrieve CLI version: {__version__}")

        # Show HPO data info if available
        hpo_info = _get_hpo_info()
        if hpo_info["version"]:
            term_info = (
                f" ({hpo_info['term_count']:,} terms)" if hpo_info["term_count"] else ""
            )
            typer.echo(f"HPO Data: {hpo_info['version']}{term_info}")
        else:
            typer.echo("HPO Data: not loaded (run 'phentrieve data prepare')")

        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show the application version and exit.",
        ),
    ] = False,
):
    """Main callback for Phentrieve CLI - handles global options like --version."""
    pass


# Register command groups
app.add_typer(data_commands.app, name="data", help="Manage HPO data.")
app.add_typer(index_commands.app, name="index", help="Manage vector indexes.")
app.add_typer(text_commands.app, name="text", help="Process and analyze clinical text.")
app.add_typer(
    benchmark_commands.app, name="benchmark", help="Run and manage benchmarks."
)
app.add_typer(
    similarity_commands.app,
    name="similarity",
    help="Calculate HPO term similarities and related metrics.",
)
app.add_typer(
    mcp_commands.app,
    name="mcp",
    help="Model Context Protocol (MCP) server commands.",
)

# Main command for query
app.command(name="query")(query_commands.query_hpo)
