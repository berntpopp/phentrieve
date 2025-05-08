# Main CLI entry point for Phentrieve
import typer
import importlib.metadata
from typing_extensions import Annotated

from .utils import setup_logging_cli

# Read version from pyproject.toml
__version__ = importlib.metadata.version("phentrieve")

app = typer.Typer(name="phentrieve", help="Phenotype Retrieval CLI Tool")


# Subcommand groups
data_app = typer.Typer(name="data", help="Manage HPO data.")
app.add_typer(data_app)


@data_app.command("prepare")
def prepare_hpo_data(
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Force update even if files exist")
    ] = False,
    data_dir: Annotated[
        str, typer.Option("--data-dir", help="Custom directory for HPO data storage")
    ] = None,
):
    """Prepare HPO data for indexing.

    Downloads the HPO ontology data, extracts terms, and precomputes
    graph properties needed for similarity calculations.
    """
    from phentrieve.data_processing.hpo_parser import orchestrate_hpo_preparation
    from phentrieve.utils import setup_logging_cli

    setup_logging_cli(debug=debug)

    typer.echo("Starting HPO data preparation...")
    success = orchestrate_hpo_preparation(
        debug=debug, force_update=force, data_dir_override=data_dir
    )

    if success:
        typer.secho(
            "HPO data preparation completed successfully!", fg=typer.colors.GREEN
        )
    else:
        typer.secho("HPO data preparation failed.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


index_app = typer.Typer(name="index", help="Manage vector indexes.")
app.add_typer(index_app)


@index_app.command("build")
def build_index(
    model_name: Annotated[
        str, typer.Option("--model-name", help="Model name to use for embeddings")
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
):
    """Build vector index for HPO terms.

    This command creates a ChromaDB vector index from HPO term data
    using the specified embedding model. The index can be later used
    for semantic search of HPO terms.
    """
    from phentrieve.indexing.chromadb_orchestrator import orchestrate_index_building
    from phentrieve.utils import setup_logging_cli

    setup_logging_cli(debug=debug)

    # Determine device based on CPU flag
    device_override = "cpu" if cpu else None

    typer.echo(f"Starting index building process...")

    success = orchestrate_index_building(
        model_name_arg=model_name,
        all_models=all_models,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        device_override=device_override,
        recreate=recreate,
        debug=debug,
    )

    if success:
        typer.secho("Index building completed successfully!", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "Index building failed for one or more models.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)


benchmark_app = typer.Typer(name="benchmark", help="Run and manage benchmarks.")
app.add_typer(benchmark_app)


@benchmark_app.command("run")
def run_benchmarks(
    test_file: Annotated[
        str, typer.Option("--test-file", help="Test file with benchmark cases")
    ] = None,
    model_name: Annotated[
        str, typer.Option("--model-name", help="Model name to benchmark")
    ] = None,
    model_list: Annotated[
        str, typer.Option("--model-list", help="Comma-separated list of models")
    ] = None,
    all_models: Annotated[
        bool, typer.Option("--all-models", help="Run for all benchmark models")
    ] = False,
    similarity_threshold: Annotated[
        float, typer.Option("--similarity-threshold", help="Minimum similarity score")
    ] = 0.1,
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Force CPU usage even if GPU is available")
    ] = False,
    detailed: Annotated[
        bool, typer.Option("--detailed", help="Show detailed per-test-case results")
    ] = False,
    output: Annotated[
        str, typer.Option("--output", help="Output CSV file for detailed results")
    ] = None,
    create_sample: Annotated[
        bool, typer.Option("--create-sample", help="Create a sample test dataset")
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
        str, typer.Option("--reranker-model", help="Cross-encoder model for re-ranking")
    ] = None,
    monolingual_reranker_model: Annotated[
        str,
        typer.Option(
            "--monolingual-reranker-model",
            help="German cross-encoder model for monolingual re-ranking",
        ),
    ] = None,
    rerank_mode: Annotated[
        str,
        typer.Option(
            "--rerank-mode",
            help="Re-ranking mode: cross-lingual or monolingual",
        ),
    ] = None,
    translation_dir: Annotated[
        str,
        typer.Option("--translation-dir", help="Directory with HPO term translations"),
    ] = None,
    rerank_count: Annotated[
        int, typer.Option("--rerank-count", help="Number of candidates to rerank")
    ] = 10,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
):
    """Run benchmarks for retrieval models.

    This command evaluates HPO retrieval performance using test cases and produces
    detailed metrics including MRR, Hit Rate, and semantic similarity. It supports
    benchmarking single models or multiple models for comparison.
    """
    from phentrieve.evaluation.benchmark_orchestrator import orchestrate_benchmark
    from phentrieve.utils import setup_logging_cli

    setup_logging_cli(debug=debug)

    typer.echo("Starting benchmark evaluation...")

    results = orchestrate_benchmark(
        test_file=test_file,
        model_name=model_name,
        model_list=model_list,
        all_models=all_models,
        similarity_threshold=similarity_threshold,
        cpu=cpu,
        detailed=detailed,
        output=output,
        debug=debug,
        create_sample=create_sample,
        trust_remote_code=trust_remote_code,
        enable_reranker=enable_reranker,
        reranker_model=reranker_model,
        monolingual_reranker_model=monolingual_reranker_model,
        rerank_mode=rerank_mode,
        translation_dir=translation_dir,
        rerank_count=rerank_count,
    )

    if results:
        if isinstance(results, list):
            typer.secho(
                f"Benchmark completed successfully for {len(results)} models!",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho("Benchmark completed successfully!", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "Benchmark evaluation failed or no results were collected.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


@benchmark_app.command("compare")
def compare_benchmarks(
    summaries_dir: Annotated[
        str, typer.Option("--summaries-dir", help="Directory with summary files")
    ] = None,
    output_csv: Annotated[
        str, typer.Option("--output-csv", help="Path to save comparison CSV")
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
):
    """Compare results from multiple benchmark runs.

    This command analyzes benchmark results from multiple model runs and generates
    a comparison table showing various metrics such as MRR, Hit Rate, and semantic similarity.
    """
    from phentrieve.evaluation.comparison_orchestrator import (
        orchestrate_benchmark_comparison,
    )
    from phentrieve.utils import setup_logging_cli
    import pandas as pd

    setup_logging_cli(debug=debug)

    typer.echo("Comparing benchmark results...")

    comparison_df = orchestrate_benchmark_comparison(
        summaries_dir=summaries_dir,
        output_csv=output_csv,
        visualize=False,
        debug=debug,
    )

    if comparison_df is not None and not comparison_df.empty:
        # Display results
        pd.options.display.float_format = "{:.4f}".format
        typer.echo("\n===== Benchmark Comparison =====")
        typer.echo(f"Models compared: {len(comparison_df)}")
        typer.echo(comparison_df.to_string())

        typer.secho("Comparison completed successfully!", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "Comparison failed or no benchmark results found.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


@benchmark_app.command("visualize")
def visualize_benchmarks(
    summaries_dir: Annotated[
        str, typer.Option("--summaries-dir", help="Directory with summary files")
    ] = None,
    metrics: Annotated[
        str,
        typer.Option(
            "--metrics",
            help="Comma-separated list of metrics to visualize (default: all)",
        ),
    ] = "all",
    output_dir: Annotated[
        str, typer.Option("--output-dir", help="Directory to save visualizations")
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
):
    """Generate visualizations from benchmark results.

    This command creates charts and graphs comparing model performance across
    different metrics like MRR, Hit Rate, and semantic similarity. You can specify
    which metrics to visualize or generate charts for all available metrics.
    """
    from phentrieve.evaluation.comparison_orchestrator import (
        orchestrate_benchmark_comparison,
    )
    from phentrieve.utils import setup_logging_cli
    import pandas as pd

    setup_logging_cli(debug=debug)

    typer.echo("Generating visualizations from benchmark results...")

    # Run comparison with visualization enabled
    comparison_df = orchestrate_benchmark_comparison(
        summaries_dir=summaries_dir,
        output_csv=None,
        visualize=True,
        output_dir=output_dir,
        metrics=metrics,
        debug=debug,
    )

    if comparison_df is not None and not comparison_df.empty:
        typer.echo(f"\nGenerated visualizations for {len(comparison_df)} models")
        typer.secho("Visualizations generated successfully!", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "Visualization failed or no benchmark results found.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


# Main query command
@app.command("query")
def query_hpo(
    text: Annotated[
        str, typer.Argument(help="Clinical text to query for HPO terms")
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
        str, typer.Option("--model-name", "-m", help="Model name to use for embeddings")
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
        str,
        typer.Option(
            "--reranker-model", "--rm", help="Cross-encoder model for re-ranking"
        ),
    ] = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    monolingual_reranker_model: Annotated[
        str,
        typer.Option(
            "--monolingual-reranker-model",
            help="German cross-encoder model for monolingual re-ranking",
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
        str,
        typer.Option(
            "--translation-dir", "--td", help="Directory with German HPO translations"
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
    from phentrieve.utils import setup_logging_cli

    # Set up logging
    setup_logging_cli(debug=debug)

    # Use default model if not specified
    if model_name is None:
        model_name = DEFAULT_MODEL
        typer.echo(f"Using default model: {model_name}")

    # Use default translation dir if not specified
    if translation_dir is None:
        translation_dir = DEFAULT_TRANSLATIONS_SUBDIR

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
            translation_dir=translation_dir,
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
                    import traceback

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
            translation_dir=translation_dir,
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
            "-v",
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
