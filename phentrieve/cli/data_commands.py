"""Data-related commands for Phentrieve CLI.

This module contains commands for managing HPO data.
"""

from typing import Annotated, Optional

import typer

# Create the Typer app for this command group
app = typer.Typer()


@app.command("prepare")
def prepare_hpo_data(
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Force update even if files exist")
    ] = False,
    data_dir: Annotated[
        Optional[str],
        typer.Option("--data-dir", help="Custom directory for HPO data storage"),
    ] = None,
    include_obsolete: Annotated[
        bool,
        typer.Option(
            "--include-obsolete",
            help="Include obsolete HPO terms (for analysis). Default: filter out.",
        ),
    ] = False,
):
    """Prepare HPO data for indexing.

    Downloads the HPO ontology data, extracts terms, and precomputes
    graph properties needed for similarity calculations.

    By default, obsolete HPO terms are filtered out to prevent retrieval
    errors (Issue #133). Use --include-obsolete for analysis purposes.
    """
    from phentrieve.data_processing.hpo_parser import orchestrate_hpo_preparation
    from phentrieve.utils import setup_logging_cli

    setup_logging_cli(debug=debug)

    typer.echo("Starting HPO data preparation...")
    if include_obsolete:
        typer.echo("Note: Including obsolete terms (--include-obsolete flag set)")

    success = orchestrate_hpo_preparation(
        debug=debug,
        force_update=force,
        data_dir_override=data_dir,
        include_obsolete=include_obsolete,
    )

    if success:
        typer.secho(
            "HPO data preparation completed successfully!", fg=typer.colors.GREEN
        )
    else:
        typer.secho("HPO data preparation failed.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
