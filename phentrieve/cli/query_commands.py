"""Query-related commands for Phentrieve CLI.

This module contains commands for querying HPO terms.
"""

import traceback
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from phentrieve.retrieval.output_formatters import (
    format_results_as_json,
    format_results_as_jsonl,
    format_results_as_text,
)


def enrich_query_results_with_details(
    structured_query_results: list[dict[str, Any]],
    data_dir_override: str | None = None,
) -> list[dict[str, Any]]:
    """
    Enrich structured query results with HPO term details (definitions, synonyms).

    Takes structured results from orchestrate_query() and enriches each
    result set's HPO terms with details from the database.

    Args:
        structured_query_results: List of result sets from orchestrate_query()
        data_dir_override: Optional data directory override

    Returns:
        Enriched structured results with definition and synonyms added to each HPO term
    """
    from phentrieve.retrieval.details_enrichment import enrich_results_with_details

    # Enrich each result set's HPO terms
    for result_set in structured_query_results:
        if "results" in result_set and result_set["results"]:
            # Enrich returns new list - replace the old one
            result_set["results"] = enrich_results_with_details(
                result_set["results"], data_dir_override
            )

    return structured_query_results


def _format_interactive_results(
    query_results: list[dict[str, Any]],
    output_format: str,
    sentence_mode: bool = False,
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
) -> str:
    """Format query results for display in various output formats.

    This is a general formatting function used in both interactive and
    non-interactive contexts. Supports text, JSON, JSON Lines, and
    Phenopacket v2 JSON output formats.

    Args:
        query_results: Query results to format
        output_format: Output format (text, json, json_lines, phenopacket_v2_json)
        sentence_mode: Whether sentence mode was used
        embedding_model: Name of embedding model used for retrieval
        reranker_model: Name of reranker model used (if enabled)

    Returns:
        Formatted output string
    """
    if not query_results:
        return ""

    if output_format.lower() == "text":
        return format_results_as_text(query_results, sentence_mode=sentence_mode)
    elif output_format.lower() == "json":
        return format_results_as_json(query_results, sentence_mode=sentence_mode)
    elif output_format.lower() == "json_lines":
        return format_results_as_jsonl(query_results)
    elif output_format.lower() == "phenopacket_v2_json":
        from phentrieve.phenopackets.utils import format_as_phenopacket_v2

        # For query command, we use aggregated results with rankings
        results_to_format = query_results if sentence_mode else [query_results[0]]

        if results_to_format:
            matches = results_to_format[0]["results"]

            aggregated_results = []
            for i, match in enumerate(matches):
                aggregated_results.append(
                    {
                        "id": match["hpo_id"],
                        "name": match.get("label") or match.get("name"),
                        "confidence": match["similarity"],
                        "rank": i + 1,
                    }
                )

            return format_as_phenopacket_v2(
                aggregated_results=aggregated_results,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
            )
        return format_as_phenopacket_v2()
    else:
        return format_results_as_text(query_results, sentence_mode=sentence_mode)


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
            help="Format for the output (text, json, json_lines, phenopacket_v2_json). Default is 'text'.",
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
    include_details: Annotated[
        bool,
        typer.Option(
            "--include-details",
            "-d",
            help="Include HPO term definitions and synonyms in output",
        ),
    ] = False,
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
    from phentrieve.config import DEFAULT_MODEL, DEFAULT_RERANKER_MODEL
    from phentrieve.retrieval.query_orchestrator import orchestrate_query
    from phentrieve.utils import setup_logging_cli

    # Set up logging
    setup_logging_cli(debug=debug)

    # Use default model if not specified
    if model_name is None:
        model_name = DEFAULT_MODEL
        typer.echo(f"Using default model: {model_name}")

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
            reranker_model=reranker_model or DEFAULT_RERANKER_MODEL,
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
        # Track output format for toggling in interactive mode
        interactive_output_format = output_format

        typer.echo("\nCommands:")
        typer.echo("  Type your query and press Enter to search")
        typer.echo("  Type '!t' to toggle between list and phenopacket output")
        typer.echo("  Type 'q' to quit")

        while True:
            try:
                user_input = typer.prompt("\nEnter text (or 'q' to quit)")

                # Handle toggle command
                if user_input.lower() in ["!t", "toggle"]:
                    if interactive_output_format.lower() == "text":
                        interactive_output_format = "phenopacket_v2_json"
                        typer.secho(
                            "Switched to phenopacket output format",
                            fg=typer.colors.CYAN,
                        )
                    elif interactive_output_format.lower() == "phenopacket_v2_json":
                        interactive_output_format = "text"
                        typer.secho(
                            "Switched to list output format", fg=typer.colors.CYAN
                        )
                    else:
                        # If format is json or json_lines, switch to text
                        old_format = interactive_output_format
                        interactive_output_format = "text"
                        typer.secho(
                            f"Switched from {old_format} to list output format",
                            fg=typer.colors.CYAN,
                        )
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    typer.echo("Exiting.")
                    break

                if not user_input.strip():
                    continue

                # For JSON/phenopacket output, we need to be careful not to mix debug output
                # so we'll use a no-op output function
                output_func_to_use = typer_echo
                if interactive_output_format.lower() in [
                    "json",
                    "json_lines",
                    "phenopacket_v2_json",
                ]:

                    def output_func_to_use(x):
                        return None  # No-op function to suppress output during query

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

                # Enrich with details if requested
                if (
                    include_details
                    and query_results
                    and isinstance(query_results, list)
                ):
                    query_results = enrich_query_results_with_details(query_results)

                # Format and display the results
                if query_results and isinstance(query_results, list):
                    formatted_output = _format_interactive_results(
                        query_results,
                        interactive_output_format,
                        sentence_mode,
                        embedding_model=model_name,
                        reranker_model=reranker_model if enable_reranker else None,
                    )
                    # Print to console
                    typer.echo(formatted_output)
                    typer.secho(
                        f"\nOutput format: {interactive_output_format} - Type '!t' to toggle",
                        fg=typer.colors.CYAN,
                        dim=True,
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

        # Validate output format
        SUPPORTED_OUTPUT_FORMATS = ["text", "json", "json_lines", "phenopacket_v2_json"]
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
            reranker_model=reranker_model or DEFAULT_RERANKER_MODEL,
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

        # Enrich with details if requested
        if include_details and isinstance(all_query_results, list):
            all_query_results = enrich_query_results_with_details(all_query_results)

        # Format the results based on the selected output format
        formatted_output = ""
        if all_query_results and isinstance(all_query_results, list):
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
            elif output_format.lower() == "phenopacket_v2_json":
                from phentrieve.phenopackets.utils import format_as_phenopacket_v2

                # Take the first result set if not in sentence mode
                results_to_format = (
                    all_query_results if sentence_mode else [all_query_results[0]]
                )

                # For query command, we use aggregated results with rankings
                # since these are direct HPO lookups without text chunks
                if results_to_format:
                    matches = results_to_format[0]["results"]

                    aggregated_results = []
                    for i, match in enumerate(matches):
                        aggregated_results.append(
                            {
                                "id": match["hpo_id"],
                                "name": match.get("label") or match.get("name"),
                                "confidence": match["similarity"],
                                "rank": i + 1,
                            }
                        )

                    # Pass as aggregated_results (first positional parameter)
                    # since query results are not chunk-based
                    formatted_output = format_as_phenopacket_v2(
                        aggregated_results=aggregated_results,
                        embedding_model=model_name,
                        reranker_model=reranker_model if enable_reranker else None,
                    )
                else:
                    formatted_output = format_as_phenopacket_v2()

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
