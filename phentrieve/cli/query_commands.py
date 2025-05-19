"""Query-related commands for Phentrieve CLI.

This module contains commands for querying HPO terms.
"""

import traceback
from typing import Optional
from typing_extensions import Annotated

import typer


def query_hpo(
    text: Annotated[
        Optional[str], typer.Argument(help="Clinical text to query for HPO terms")
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            "-i",
            help="Enable interactive mode for continuous querying",
        ),
    ] = False,
    model_name: Annotated[
        Optional[str],
        typer.Option("--model-name", "-m", help="Model name to use for embeddings"),
    ] = "FremyCompany/BioLORD-2023-M",
    num_results: Annotated[
        int,
        typer.Option(
            "--num-results", "-n", help="Number of results to display for each query"
        ),
    ] = 10,
    similarity_threshold: Annotated[
        float,
        typer.Option(
            "--similarity-threshold",
            "-t",
            help="Minimum similarity threshold for results",
        ),
    ] = 0.3,
    sentence_mode: Annotated[
        bool,
        typer.Option(
            "--sentence-mode",
            "-s",
            help="Process text sentence by sentence (helps with longer texts)",
        ),
    ] = False,
    trust_remote_code: Annotated[
        bool,
        typer.Option(
            "--trust-remote-code", help="Trust remote code when loading models"
        ),
    ] = False,
    enable_reranker: Annotated[
        bool, typer.Option("--enable-reranker", help="Enable cross-encoder re-ranking")
    ] = False,
    reranker_model: Annotated[
        Optional[str],
        typer.Option(
            "--reranker-model", "--rm", help="Cross-encoder model for re-ranking"
        ),
    ] = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    monolingual_reranker_model: Annotated[
        Optional[str],
        typer.Option(
            "--monolingual-reranker-model",
            help="Language-specific cross-encoder model for monolingual re-ranking",
        ),
    ] = "ml6team/cross-encoder-mmarco-german-distilbert-base",
    reranker_mode: Annotated[
        str,
        typer.Option(
            "--reranker-mode",
            "--mode",
            help="Mode for re-ranking (cross-lingual or monolingual)",
        ),
    ] = "cross-lingual",
    translation_dir: Annotated[
        Optional[str],
        typer.Option(
            "--translation-dir",
            "--td",
            help="Directory with HPO translations in target language",
        ),
    ] = None,
    rerank_count: Annotated[
        int,
        typer.Option("--rerank-count", "--rc", help="Number of candidates to re-rank"),
    ] = 10,
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Force CPU usage even if GPU is available")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
):
    """Query HPO terms with natural language clinical descriptions.

    This command allows querying the HPO term index with clinical text descriptions
    to find matching HPO terms. It supports various embedding models and optional
    cross-encoder re-ranking for improved results.
    """
    from phentrieve.retrieval.query_orchestrator import orchestrate_query
    from phentrieve.config import DEFAULT_MODEL, DEFAULT_TRANSLATIONS_SUBDIR
    from phentrieve.utils import setup_logging_cli, resolve_data_path

    # Set up logging
    setup_logging_cli(debug=debug)

    # Use default model if not specified
    if model_name is None:
        model_name = DEFAULT_MODEL
        typer.echo(f"Using default model: {model_name}")

    # Use default translation dir if not specified
    if translation_dir is None:
        translation_dir = DEFAULT_TRANSLATIONS_SUBDIR

    # Resolve translation directory path
    translation_dir_path = resolve_data_path(translation_dir)

    # Determine device based on CPU flag
    device_override = "cpu" if cpu else None

    # Custom output function that uses typer.echo
    def typer_echo(message):
        typer.echo(message)

    # Check for interactive mode
    if interactive:
        # Display welcome message for interactive mode
        typer.echo("\n===== Phentrieve HPO RAG Query Tool =====")
        typer.echo("Enter clinical descriptions to find matching HPO terms.")
        typer.echo("Type 'exit', 'quit', or 'q' to exit the program.\n")

        # Initialize the retriever and cross-encoder once for all queries
        success = orchestrate_query(
            interactive_setup=True,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            enable_reranker=enable_reranker,
            reranker_model=reranker_model,
            monolingual_reranker_model=monolingual_reranker_model,
            reranker_mode=reranker_mode,
            translation_dir=translation_dir_path,
            device_override=device_override,
            debug=debug,
            output_func=typer_echo,
        )

        if not success:
            typer.secho(
                "Failed to initialize retriever or models. Exiting.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        # Interactive query loop
        while True:
            try:
                user_input = typer.prompt("\nEnter text (or 'q' to quit)")
                if user_input.lower() in ["exit", "quit", "q"]:
                    typer.echo("Exiting.")
                    break

                if not user_input.strip():
                    continue

                # Process the query
                results = orchestrate_query(
                    query_text=user_input,
                    interactive_mode=True,
                    num_results=num_results,
                    similarity_threshold=similarity_threshold,
                    sentence_mode=sentence_mode,
                    rerank_count=rerank_count,
                    debug=debug,
                    output_func=typer_echo,
                )

            except KeyboardInterrupt:
                typer.echo("\nExiting.")
                break
            except Exception as e:
                typer.secho(f"Error: {str(e)}", fg=typer.colors.RED)
                if debug:
                    traceback.print_exc()

    # Single query mode
    else:
        if text is None:
            typer.secho(
                "Error: Text argument is required when not in interactive mode.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        typer.echo(f"Querying for HPO terms with text: '{text}'")

        # Call the orchestrator for a single query
        results = orchestrate_query(
            query_text=text,
            model_name=model_name,
            num_results=num_results,
            similarity_threshold=similarity_threshold,
            sentence_mode=sentence_mode,
            trust_remote_code=trust_remote_code,
            enable_reranker=enable_reranker,
            reranker_model=reranker_model,
            monolingual_reranker_model=monolingual_reranker_model,
            reranker_mode=reranker_mode,
            translation_dir=translation_dir_path,
            rerank_count=rerank_count,
            device_override=device_override,
            debug=debug,
            output_func=typer_echo,
        )

        # Display summary
        if results:
            typer.secho(
                "\nQuery processing completed successfully!", fg=typer.colors.GREEN
            )
        else:
            typer.secho(
                "\nNo results found or an error occurred during query processing.",
                fg=typer.colors.RED,
            )
