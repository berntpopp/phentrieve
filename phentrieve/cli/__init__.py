"""Main CLI entry point for Phentrieve.

This module defines the main Typer application and imports lightweight
subcommands eagerly while lazily loading heavy command groups.
"""

import importlib
import importlib.metadata
from pathlib import Path
from typing import Annotated

import click
import typer
from typer.core import TyperGroup
from typer.main import get_command

# Import all command groups
from phentrieve.cli import (
    benchmark_commands,
    config_commands,
    index_commands,
    mcp_commands,
    query_commands,
)
from phentrieve.cli._profile import apply_profile_callback

# Read version from pyproject.toml
__version__ = importlib.metadata.version("phentrieve")


class _LazyTyperProxy(click.Group):
    """Proxy command group that loads a Typer app on first real use."""

    def __init__(self, *, import_path: str, help_text: str):
        name = import_path.rsplit(":", 1)[0].rsplit(".", 1)[-1].replace("_commands", "")
        super().__init__(name=name, help=help_text)
        self._import_path = import_path
        self._loaded_command: click.Command | None = None

    def _load(self) -> click.Command:
        if self._loaded_command is None:
            module_path, attr_name = self._import_path.split(":", 1)
            module = importlib.import_module(module_path)
            target = getattr(module, attr_name)
            self._loaded_command = get_command(target)
            self.params = list(self._loaded_command.params)
            self.callback = self._loaded_command.callback
            self.help = self._loaded_command.help
            self.short_help = self._loaded_command.short_help
            self.epilog = self._loaded_command.epilog
        return self._loaded_command

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        loaded = self._load()
        if isinstance(loaded, click.Group):
            return loaded.get_command(ctx, cmd_name)
        return None

    def list_commands(self, ctx: click.Context) -> list[str]:
        loaded = self._load()
        if isinstance(loaded, click.Group):
            return loaded.list_commands(ctx)
        return []

    def invoke(self, ctx: click.Context) -> object:
        loaded = self._load()
        return loaded.invoke(ctx)


class _LazyRootGroup(TyperGroup):
    """Root Typer group that exposes selected lazy subcommands."""

    _lazy_commands: list[tuple[str, _LazyTyperProxy]] = [
        (
            "data",
            _LazyTyperProxy(
                import_path="phentrieve.cli.data_commands:app",
                help_text="Manage HPO data.",
            ),
        ),
        (
            "text",
            _LazyTyperProxy(
                import_path="phentrieve.cli.text_commands:app",
                help_text="Process and analyze research phenotype text.",
            ),
        ),
        (
            "similarity",
            _LazyTyperProxy(
                import_path="phentrieve.cli.similarity_commands:app",
                help_text="Calculate HPO term similarities and related metrics.",
            ),
        ),
    ]

    def list_commands(self, ctx: click.Context) -> list[str]:
        names = super().list_commands(ctx)
        for name, _command in self._lazy_commands:
            if name not in names:
                names.append(name)
        return names

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        for name, lazy_command in self._lazy_commands:
            if name == cmd_name:
                return lazy_command
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        return None


# Create the main Typer app
app = typer.Typer(
    name="phentrieve",
    cls=_LazyRootGroup,
    help="Phentrieve - AI-powered HPO term mapping using Retrieval-Augmented Generation (RAG)",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _get_hpo_info() -> dict[str, str | int | None]:
    """Load HPO metadata from database for version display.

    Returns:
        Dictionary with version and term_count, or None values if unavailable.
    """
    result: dict[str, str | int | None] = {"version": None, "term_count": None}

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
    ctx: typer.Context,
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
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            "-P",
            envvar="PHENTRIEVE_PROFILE",
            callback=apply_profile_callback,
            is_eager=True,
            help=(
                "Apply a named profile globally. A subcommand-level "
                "--profile wins on conflict."
            ),
        ),
    ] = None,
    show_resolved_config: Annotated[
        bool,
        typer.Option(
            "--show-resolved-config",
            help=(
                "Print the resolved option values (with source labels) to "
                "stderr before running the command. Useful for diagnosing "
                "why a profile or YAML default did not take effect."
            ),
        ),
    ] = False,
):
    """Main callback for Phentrieve CLI - handles global options like --version."""
    ctx.ensure_object(dict)
    if isinstance(ctx.obj, dict):
        ctx.obj["show_resolved_config"] = show_resolved_config


# Register command groups
app.add_typer(typer.Typer(), name="data", help="Manage HPO data.")
app.add_typer(index_commands.app, name="index", help="Manage vector indexes.")
app.add_typer(
    typer.Typer(), name="text", help="Process and analyze research phenotype text."
)
app.add_typer(
    benchmark_commands.app, name="benchmark", help="Run and manage benchmarks."
)
app.add_typer(
    typer.Typer(),
    name="similarity",
    help="Calculate HPO term similarities and related metrics.",
)
app.add_typer(
    mcp_commands.app,
    name="mcp",
    help="Model Context Protocol (MCP) server commands.",
)
app.add_typer(
    config_commands.config_app,
    name="config",
    help="Inspect and validate phentrieve.yaml configuration profiles.",
)

# Main command for query
app.command(name="query")(query_commands.query_hpo)
