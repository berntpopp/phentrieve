"""Index-related commands for Phentrieve CLI.

This module contains commands for managing vector indexes.
"""

from typing import Annotated, Optional

import typer

# Create the Typer app for this command group
app = typer.Typer()


@app.command("build")
def build_index(
    model_name: Annotated[
        Optional[str],
        typer.Option("--model-name", help="Model name to use for embeddings"),
    ] = None,
    all_models: Annotated[
        bool, typer.Option("--all-models", help="Run for all benchmark models")
    ] = False,
    recreate: Annotated[
        bool, typer.Option("--recreate", help="Recreate index even if it exists")
    ] = False,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Batch size for indexing")
    ] = 100,
    trust_remote_code: Annotated[
        bool,
        typer.Option(
            "--trust-remote-code", help="Trust remote code when loading models"
        ),
    ] = False,
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Force CPU usage even if GPU is available")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
    multi_vector: Annotated[
        bool,
        typer.Option(
            "--multi-vector",
            help="Build multi-vector index (separate vectors for label, synonyms, definition)",
        ),
    ] = False,
):
    """Build vector index for HPO terms.

    This command creates a ChromaDB vector index from HPO term data
    using the specified embedding model. The index can be later used
    for semantic search of HPO terms.

    Use --multi-vector to create an index with separate embeddings for each
    component (label, synonyms, definition) enabling fine-grained retrieval
    with configurable aggregation strategies.
    """
    from phentrieve.indexing.chromadb_orchestrator import orchestrate_index_building
    from phentrieve.utils import setup_logging_cli

    setup_logging_cli(debug=debug)

    # Determine device based on CPU flag
    device_override = "cpu" if cpu else None

    index_type_str = "multi-vector" if multi_vector else "single-vector"
    typer.echo(f"Starting {index_type_str} index building process")

    success = orchestrate_index_building(
        model_name_arg=model_name,
        all_models=all_models,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        device_override=device_override,
        recreate=recreate,
        debug=debug,
        multi_vector=multi_vector,
    )

    if success:
        typer.secho("Index building completed successfully!", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "Index building failed for one or more models.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
