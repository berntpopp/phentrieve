"""Query-related commands for Phentrieve CLI.

This module contains commands for querying HPO terms.
"""

import traceback
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import typer

from phentrieve.retrieval.output_formatters import (
    format_results_as_text,
    format_results_as_json,
    format_results_as_jsonl,
)


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
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output-file",
            "-O",
            help="Path to save query results. If not specified, results are printed to the console.",
            show_default=False,
            writable=True,
            resolve_path=True,
            dir_okay=False,
            file_okay=True,
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            "-F",
            help="Format for the output (text, json, json_lines). Default is 'text'.",
            case_sensitive=False,
        ),
    ] = "text",
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Force CPU usage even if GPU is available")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
    detect_query_assertion: Annotated[
        bool,
        typer.Option(
            "--detect-query-assertion",
            help="Enable assertion detection on the input query text.",
        ),
    ] = False,
    query_assertion_language: Annotated[
        Optional[str],
        typer.Option(
            "--query-assertion-language",
            help="Language for query assertion detection (e.g., 'en', 'de'). Defaults to auto-detect or 'en'.",
        ),
    ] = None,
    query_assertion_preference: Annotated[
        str,
        typer.Option(
            "--query-assertion-preference",
            help="Assertion detection strategy for the query (dependency, keyword, any_negative).",
        ),
    ] = "dependency",
):
    """Query HPO terms with natural language clinical descriptions.

    This command allows querying the HPO term index with clinical text descriptions
    to find matching HPO terms. It supports various embedding models and optional
    cross-encoder re-ranking for improved results.

    Results can be printed to the console or saved to a file in various formats:
    - text: Human-readable text output (default)
    - json: Structured JSON output
    - json_lines: JSON Lines format (one JSON object per line)
    """
    from phentrieve.retrieval.query_orchestrator import orchestrate_query
    from phentrieve.config import DEFAULT_MODEL, DEFAULT_TRANSLATIONS_SUBDIR
    from phentrieve.utils import setup_logging_cli, resolve_data_path
    from phentrieve.retrieval.output_formatters import (
        format_results_as_text,
        format_results_as_json,
        format_results_as_jsonl,
    )

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
            detect_query_assertion=detect_query_assertion,
            query_assertion_language=query_assertion_language,
            query_assertion_preference=query_assertion_preference,
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

                # For JSON output, we need to be careful not to mix debug output with the JSON
                # so we'll use a no-op output function
                output_func_to_use = typer_echo
                if output_format.lower() in ["json", "json_lines"]:
                    output_func_to_use = (
                        lambda x: None
                    )  # No-op function to suppress output during query

                # Process the query
                query_results = orchestrate_query(
                    query_text=user_input,
                    interactive_mode=True,
                    sentence_mode=sentence_mode,
                    num_results=num_results,
                    similarity_threshold=similarity_threshold,
                    debug=debug,
                    output_func=output_func_to_use,
                    detect_query_assertion=detect_query_assertion,
                    query_assertion_language=query_assertion_language,
                    query_assertion_preference=query_assertion_preference,
                )

                # Format the results based on the selected output format
                if query_results:
                    formatted_output = ""
                    if output_format.lower() == "text":
                        formatted_output = format_results_as_text(
                            query_results, sentence_mode=sentence_mode
                        )
                    elif output_format.lower() == "json":
                        formatted_output = format_results_as_json(
                            query_results, sentence_mode=sentence_mode
                        )
                    elif output_format.lower() == "json_lines":
                        formatted_output = format_results_as_jsonl(query_results)

                    # Print to console
                    typer.echo(formatted_output)

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

        # Validate output format
        SUPPORTED_OUTPUT_FORMATS = ["text", "json", "json_lines"]
        if output_format.lower() not in SUPPORTED_OUTPUT_FORMATS:
            typer.secho(
                f"Error: Unsupported output format '{output_format}'. "
                f"Supported formats: {', '.join(SUPPORTED_OUTPUT_FORMATS)}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        typer.echo(f"Querying for HPO terms with text: '{text}'")

        # Call the orchestrator for a single query
        all_query_results = orchestrate_query(
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
            detect_query_assertion=detect_query_assertion,
            query_assertion_language=query_assertion_language,
            query_assertion_preference=query_assertion_preference,
        )

        # Check if we have results
        if not all_query_results:
            message = "No results found or an error occurred during query processing."
            if output_file is None:
                typer.secho(message, fg=typer.colors.RED)
            return

        # Format the results based on the selected output format
        formatted_output = ""
        if output_format.lower() == "text":
            formatted_output = format_results_as_text(
                all_query_results, sentence_mode=sentence_mode
            )
        elif output_format.lower() == "json":
            formatted_output = format_results_as_json(
                all_query_results, sentence_mode=sentence_mode
            )
        elif output_format.lower() == "json_lines":
            formatted_output = format_results_as_jsonl(all_query_results)

        # Output the results (to file or console)
        if output_file:
            try:
                # Ensure parent directory exists
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # Write the formatted output to the file
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(formatted_output)

                typer.secho(f"Results saved to {output_file}", fg=typer.colors.GREEN)
            except Exception as e:
                typer.secho(f"Error writing to file: {str(e)}", fg=typer.colors.RED)
                raise typer.Exit(code=1)
        else:
            # Print to console
            typer.echo(formatted_output)

        # Display summary
        if all_query_results:
            typer.secho(
                "\nQuery processing completed successfully!", fg=typer.colors.GREEN
            )
