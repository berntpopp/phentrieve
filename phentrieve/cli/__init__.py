"""Main CLI entry point for Phentrieve.

This module defines the main Typer application and imports all subcommands
from their respective modules.
"""

import importlib.metadata
from typing import Annotated

import typer

# Import all command groups
from phentrieve.cli import (
    benchmark_commands,
    data_commands,
    index_commands,
    query_commands,
    similarity_commands,
    text_commands,
)

# Read version from pyproject.toml
__version__ = importlib.metadata.version("phentrieve")

# Create the main Typer app
app = typer.Typer(name="phentrieve", help="Phenotype Retrieval CLI Tool")


def version_callback(value: bool):
    """Display version information and exit."""
    if value:
        typer.echo(f"Phentrieve CLI version: {__version__}")
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
    """Phentrieve CLI main callback."""
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

# Main command for query
app.command(name="query")(query_commands.query_hpo)
