# Main CLI entry point for Phentrieve
import typer
import importlib.metadata
from typing_extensions import Annotated

from .utils import setup_logging_cli

# Read version from pyproject.toml
__version__ = importlib.metadata.version("phentrieve")

app = typer.Typer(name="phentrieve", help="Phenotype Retrieval CLI Tool")


# Placeholder subcommand groups
data_app = typer.Typer(name="data", help="Manage HPO data.")
app.add_typer(data_app)


index_app = typer.Typer(name="index", help="Manage vector indexes.")
app.add_typer(index_app)


benchmark_app = typer.Typer(name="benchmark", help="Run and manage benchmarks.")
app.add_typer(benchmark_app)


# Placeholder for top-level query command (will be added properly in its own issue)
@app.command("query")
def query_hpo_placeholder(
    text: Annotated[str, typer.Argument()] = "Placeholder query",
    debug: Annotated[bool, typer.Option("--debug")] = False,
):
    """Placeholder for the query command."""
    setup_logging_cli(debug=debug)
    typer.echo(f"Placeholder: Querying for '{text}'")


# Version callback
def version_callback(value: bool):
    if value:
        typer.echo(f"Phentrieve CLI version: {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show the application version and exit.",
        ),
    ] = False,
):
    """Phentrieve CLI main callback."""
    # This callback runs before any command
    pass


# Entry point for direct script execution during development
if __name__ == "__main__":
    app()
