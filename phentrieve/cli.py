# Main CLI entry point for Phentrieve
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import typer
import importlib.metadata
from typing_extensions import Annotated
import json
import yaml


# Read version from pyproject.toml
__version__ = importlib.metadata.version("phentrieve")

app = typer.Typer(name="phentrieve", help="Phenotype Retrieval CLI Tool")


# Subcommand groups
data_app = typer.Typer(name="data", help="Manage HPO data.")
app.add_typer(data_app)

# Text processing subcommand group
text_app = typer.Typer(name="text", help="Process and analyze clinical text.")
app.add_typer(text_app)


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

    typer.echo("Starting index building process")

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
            help="Language-specific cross-encoder model for monolingual re-ranking",
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
    similarity_formula: Annotated[
        str,
        typer.Option(
            "--similarity-formula",
            help="Formula to use for similarity calculation (hybrid or simple_resnik_like)",
        ),
    ] = "hybrid",
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
    from phentrieve.utils import setup_logging_cli, resolve_data_path

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
        similarity_formula=similarity_formula,
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
    from phentrieve.utils import setup_logging_cli, resolve_data_path
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
    from phentrieve.utils import setup_logging_cli, resolve_data_path
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
            help="Language-specific cross-encoder model for monolingual re-ranking",
        ),
    ] = "ml6team/cross-encoder-mmarco-german-distilbert-base",  # Current default is German-specific, replace with appropriate language model
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


# Helper functions for CLI commands
def load_text_from_input(text_arg: Optional[str], file_arg: Optional[Path]) -> str:
    """Load text from command line argument, file, or stdin.

    Args:
        text_arg: Text provided as a command line argument
        file_arg: Path to a file to read text from

    Returns:
        The loaded text content

    Raises:
        typer.Exit: If no text is provided or if the file does not exist
    """
    raw_text = None

    if text_arg is not None:
        raw_text = text_arg
    elif file_arg is not None:
        if not file_arg.exists():
            typer.secho(
                f"Error: Input file {file_arg} does not exist.", fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        with open(file_arg, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        # Read from stdin if available
        if not sys.stdin.isatty():
            raw_text = sys.stdin.read()
        else:
            typer.secho(
                "Error: No text provided. Please provide text as an argument, "
                "via --input-file, or through stdin.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    if not raw_text or not raw_text.strip():
        typer.secho("Error: Empty text provided.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    return raw_text


def resolve_chunking_pipeline_config(
    config_file_arg: Optional[Path], strategy_arg: str
) -> List[Dict]:
    """Resolve chunking pipeline configuration from file or strategy.

    Args:
        config_file_arg: Path to a YAML or JSON configuration file
        strategy_arg: Name of a predefined chunking strategy

    Returns:
        List of chunking pipeline configuration dictionaries

    Raises:
        typer.Exit: If the configuration file does not exist or has an invalid format
    """
    from phentrieve.config import DEFAULT_CHUNK_PIPELINE_CONFIG

    chunking_pipeline_config = None

    # 1. First priority: Config file if provided
    if config_file_arg is not None:
        if not config_file_arg.exists():
            typer.secho(
                f"Error: Configuration file {config_file_arg} does not exist.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        suffix = config_file_arg.suffix.lower()
        with open(config_file_arg, "r", encoding="utf-8") as f:
            if suffix == ".json":
                config_data = json.load(f)
            elif suffix in (".yaml", ".yml"):
                config_data = yaml.safe_load(f)
            else:
                typer.secho(
                    f"Error: Unsupported config file format: {suffix}. Use .json, .yaml, or .yml",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

        chunking_pipeline_config = config_data.get("chunking_pipeline", None)

    # 2. Second priority: Strategy parameter
    if chunking_pipeline_config is None:
        if strategy_arg == "simple":
            chunking_pipeline_config = [{"type": "paragraph"}, {"type": "sentence"}]
        elif strategy_arg == "semantic":
            chunking_pipeline_config = [
                {"type": "paragraph"},
                {
                    "type": "semantic",
                    "config": {
                        "similarity_threshold": 0.4,
                        "min_chunk_sentences": 1,
                        "max_chunk_sentences": 3,
                    },
                },
            ]
        elif strategy_arg == "detailed":
            chunking_pipeline_config = [
                {"type": "paragraph"},
                {
                    "type": "semantic",
                    "config": {
                        "similarity_threshold": 0.4,
                        "min_chunk_sentences": 1,
                        "max_chunk_sentences": 3,
                    },
                },
                {"type": "fine_grained_punctuation"},
            ]
        else:
            typer.secho(
                f"Warning: Unknown strategy '{strategy_arg}'. Using default configuration.",
                fg=typer.colors.YELLOW,
            )

    # 3. Final fallback: Default configuration
    if chunking_pipeline_config is None:
        chunking_pipeline_config = DEFAULT_CHUNK_PIPELINE_CONFIG

    return chunking_pipeline_config


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


# Text processing commands
@text_app.command("chunk")
def chunk_text_command(
    text: Annotated[
        Optional[str],
        typer.Argument(
            help="Text to chunk (optional, will read from stdin if not provided)"
        ),
    ] = None,
    input_file: Annotated[
        Optional[Path],
        typer.Option(
            "--input-file", "-i", help="File to read text from instead of command line"
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language of the text (en, de, etc.)"),
    ] = "en",
    chunking_pipeline_config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config-file",
            "-c",
            help="Path to YAML or JSON file with chunking pipeline configuration",
        ),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Predefined chunking strategy (simple, semantic, detailed)",
        ),
    ] = "simple",
    semantic_chunker_model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name for semantic chunker (if using semantic strategy)",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format", "-o", help="Output format for chunks (lines, json_lines)"
        ),
    ] = "lines",
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging"),
    ] = False,
):
    """Chunk text using configurable chunking strategies.

    This command processes text through a chunking pipeline, which can include
    paragraph splitting, sentence segmentation, semantic chunking, and fine-grained
    punctuation-based splitting. The output is the resulting text chunks.

    Example usage:
    - Simple paragraph+sentence chunking: phentrieve text chunk "My text here"
    - Semantic chunking: phentrieve text chunk -s semantic -m "FremyCompany/BioLORD-2023-M" -i clinical_note.txt
    """
    from sentence_transformers import SentenceTransformer

    from phentrieve.config import (
        DEFAULT_CHUNK_PIPELINE_CONFIG,
        DEFAULT_LANGUAGE,
        DEFAULT_MODEL,
    )
    from phentrieve.text_processing.pipeline import TextProcessingPipeline
    from phentrieve.utils import detect_language, setup_logging_cli

    setup_logging_cli(debug=debug)

    # Load the raw text using helper function
    raw_text = load_text_from_input(text, input_file)

    # Detect or set the language
    if not language:
        # Try to auto-detect the language
        try:
            language = detect_language(raw_text, default_lang=DEFAULT_LANGUAGE)
            typer.echo(f"Auto-detected language: {language}")
        except ImportError:
            language = DEFAULT_LANGUAGE
            typer.echo(f"Using default language: {language}")

    # Get chunking pipeline configuration using helper function
    chunking_pipeline_config = resolve_chunking_pipeline_config(
        chunking_pipeline_config_file, strategy
    )

    # Determine if we need a semantic model
    needs_semantic_model = any(
        chunk_config.get("type") == "semantic"
        for chunk_config in chunking_pipeline_config
    )

    # Load the SBERT model if needed
    sbert_model = None
    if needs_semantic_model:
        model_name = semantic_chunker_model or DEFAULT_MODEL
        typer.echo(f"Loading sentence transformer model: {model_name}...")
        try:
            sbert_model = SentenceTransformer(model_name)
        except Exception as e:
            typer.secho(
                f"Error loading model '{model_name}': {str(e)}", fg=typer.colors.RED
            )
            raise typer.Exit(code=1)

    # Empty assertion config to disable assertion detection for this command
    assertion_config = {"disable": True}

    # Create the pipeline
    try:
        pipeline = TextProcessingPipeline(
            language=language,
            chunking_pipeline_config=chunking_pipeline_config,
            assertion_config=assertion_config,
            sbert_model_for_semantic_chunking=sbert_model,
        )
    except Exception as e:
        typer.secho(f"Error creating pipeline: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Process the text
    try:
        processed_chunks = pipeline.process(raw_text)
    except Exception as e:
        typer.secho(f"Error processing text: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Output the chunks in the requested format
    if output_format == "lines":
        for i, chunk_data in enumerate(processed_chunks):
            typer.echo(f"[{i+1}] {chunk_data['text']}")
    elif output_format == "json_lines":
        for chunk_data in processed_chunks:
            # Ensure Enum values are serialized properly
            chunk_json = {
                "text": chunk_data["text"],
                "source_indices": chunk_data["source_indices"],
            }
            typer.echo(json.dumps(chunk_json))
    else:
        typer.secho(
            f"Error: Unknown output format '{output_format}'. "
            f"Supported formats: lines, json_lines",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Summary
    typer.secho(
        f"\nText chunking completed. {len(processed_chunks)} chunks generated.",
        fg=typer.colors.GREEN,
    )


@text_app.command("process")
def process_text_for_hpo_command(
    text: Annotated[
        Optional[str],
        typer.Argument(
            help="Text to process (optional, will read from stdin if not provided)"
        ),
    ] = None,
    input_file: Annotated[
        Optional[Path],
        typer.Option(
            "--input-file", "-i", help="File to read text from instead of command line"
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language of the text (en, de, etc.)"),
    ] = "en",
    chunking_pipeline_config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config-file",
            "-c",
            help="Path to YAML or JSON file with chunking pipeline configuration",
        ),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Predefined chunking strategy (simple, semantic, detailed)",
        ),
    ] = "semantic",  # Changed default to semantic for better chunks
    semantic_chunker_model: Annotated[
        Optional[str],
        typer.Option(
            "--semantic-model",
            "--s-model",
            help="Model name for semantic chunker (if using semantic strategy)",
        ),
    ] = None,
    retrieval_model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model name for HPO term retrieval"),
    ] = None,
    similarity_threshold: Annotated[
        float,
        typer.Option(
            "--similarity-threshold",
            "--threshold",
            help="Minimum similarity score for HPO term matches",
        ),
    ] = 0.3,
    num_results: Annotated[
        int,
        typer.Option(
            "--num-results",
            "-n",
            help="Maximum number of HPO terms to return per query",
        ),
    ] = 10,
    no_assertion_detection: Annotated[
        bool,
        typer.Option(
            "--no-assertion",
            help="Disable assertion detection (treat all chunks as affirmed)",
        ),
    ] = False,
    assertion_preference: Annotated[
        str,
        typer.Option(
            "--assertion-preference",
            help="Assertion detection strategy preference (dependency, keyword, any_negative)",
        ),
    ] = "dependency",
    enable_reranker: Annotated[
        bool,
        typer.Option(
            "--enable-reranker",
            "--rerank",
            help="Enable cross-encoder reranking of results",
        ),
    ] = False,
    reranker_model: Annotated[
        Optional[str],
        typer.Option(
            "--reranker-model",
            help="Cross-encoder model for reranking (if reranking enabled)",
        ),
    ] = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    monolingual_reranker_model: Annotated[
        Optional[str],
        typer.Option(
            "--monolingual-reranker-model",
            help="Language-specific cross-encoder model for monolingual reranking",
        ),
    ] = "ml6team/cross-encoder-mmarco-german-distilbert-base",
    reranker_mode: Annotated[
        str,
        typer.Option(
            "--reranker-mode",
            help="Mode for re-ranking: 'cross-lingual' or 'monolingual'",
        ),
    ] = "cross-lingual",
    translation_dir: Annotated[
        Optional[str],
        typer.Option(
            "--translation-dir",
            help="Directory with HPO translations in target language (required for monolingual mode)",
        ),
    ] = None,
    rerank_count: Annotated[
        int,
        typer.Option(
            "--rerank-count",
            help="Number of candidates to consider for re-ranking",
        ),
    ] = 50,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            "-o",
            help="Output format for results (json_lines, rich_json_summary, csv_hpo_list)",
        ),
    ] = "rich_json_summary",
    min_confidence: Annotated[
        float,
        typer.Option(
            "--min-confidence",
            "--min-conf",
            help="Minimum confidence threshold for HPO terms in the results",
        ),
    ] = 0.0,
    top_term_per_chunk: Annotated[
        bool,
        typer.Option(
            "--top-term-per-chunk",
            "--top-only",
            help="Only include the highest-scored HPO term for each chunk",
        ),
    ] = False,
    cpu: Annotated[
        bool,
        typer.Option("--cpu", help="Force CPU usage even if GPU is available"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging"),
    ] = False,
):
    """Process clinical text to extract HPO terms.

    This command processes clinical texts through a chunking pipeline and assertion
    detection, then extracts HPO terms from each chunk. Results are aggregated to provide
    a comprehensive set of phenotype terms from the entire document.

    Example usage:
    - Basic processing: phentrieve text process "Patient presents with hearing loss."
    - From file with semantic chunking: phentrieve text process -i clinical_note.txt -s semantic -m "FremyCompany/BioLORD-2023-M"
    """
    import json
    import csv
    import sys
    from collections import defaultdict
    from io import StringIO
    from sentence_transformers import SentenceTransformer

    from phentrieve.config import (
        DEFAULT_CHUNK_PIPELINE_CONFIG,
        DEFAULT_MODEL,
        DEFAULT_LANGUAGE,
        DEFAULT_ASSERTION_CONFIG,
        MIN_SIMILARITY_THRESHOLD,
        DEFAULT_TRANSLATIONS_SUBDIR,
    )
    from phentrieve.text_processing.pipeline import TextProcessingPipeline
    from phentrieve.text_processing.assertion_detection import AssertionStatus
    from phentrieve.retrieval.dense_retriever import DenseRetriever
    from phentrieve.retrieval import reranker
    from phentrieve.utils import (
        setup_logging_cli,
        resolve_data_path,
        load_translation_text,
        detect_language,
    )

    setup_logging_cli(debug=debug)

    # Determine device
    device = "cpu" if cpu else None

    # Load the raw text using helper function
    raw_text = load_text_from_input(text, input_file)

    # Detect or set the language
    if not language:
        # Try to auto-detect the language
        try:
            language = detect_language(raw_text, default_lang=DEFAULT_LANGUAGE)
            typer.echo(f"Auto-detected language: {language}")
        except ImportError:
            language = DEFAULT_LANGUAGE
            typer.echo(f"Using default language: {language}")

    # Load chunking pipeline configuration using helper function
    chunking_pipeline_config = resolve_chunking_pipeline_config(
        chunking_pipeline_config_file, strategy
    )

    # 1. First priority: Config file if provided
    if chunking_pipeline_config_file is not None:
        if not chunking_pipeline_config_file.exists():
            typer.secho(
                f"Error: Configuration file {chunking_pipeline_config_file} does not exist.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        suffix = chunking_pipeline_config_file.suffix.lower()
        with open(chunking_pipeline_config_file, "r", encoding="utf-8") as f:
            if suffix == ".json":
                config_data = json.load(f)
            elif suffix in (".yaml", ".yml"):
                config_data = yaml.safe_load(f)
            else:
                typer.secho(
                    f"Error: Unsupported config file format: {suffix}. Use .json, .yaml, or .yml",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

        chunking_pipeline_config = config_data.get("chunking_pipeline", None)

    # 2. Second priority: Strategy parameter
    if chunking_pipeline_config is None:
        if strategy == "simple":
            chunking_pipeline_config = [{"type": "paragraph"}, {"type": "sentence"}]
        elif strategy == "semantic":
            chunking_pipeline_config = [
                {"type": "paragraph"},
                {
                    "type": "semantic",
                    "config": {
                        "similarity_threshold": 0.4,
                        "min_chunk_sentences": 1,
                        "max_chunk_sentences": 3,
                    },
                },
            ]
        elif strategy == "detailed":
            chunking_pipeline_config = [
                {"type": "paragraph"},
                {
                    "type": "semantic",
                    "config": {
                        "similarity_threshold": 0.4,
                        "min_chunk_sentences": 1,
                        "max_chunk_sentences": 3,
                    },
                },
                {"type": "fine_grained_punctuation"},
            ]
        else:
            typer.secho(
                f"Warning: Unknown strategy '{strategy}'. Using default configuration.",
                fg=typer.colors.YELLOW,
            )

    # 3. Final fallback: Default configuration
    if chunking_pipeline_config is None:
        chunking_pipeline_config = DEFAULT_CHUNK_PIPELINE_CONFIG

    # Determine if we need a semantic model
    needs_semantic_model = any(
        chunk_config.get("type") == "semantic"
        for chunk_config in chunking_pipeline_config
    )

    # Configure assertion detection
    assertion_config = dict(DEFAULT_ASSERTION_CONFIG)
    if no_assertion_detection:
        assertion_config["disable"] = True
    else:
        assertion_config["preference"] = assertion_preference

    # Resolve translation directory path if in monolingual mode
    if reranker_mode == "monolingual" and translation_dir is None:
        translation_dir = resolve_data_path(DEFAULT_TRANSLATIONS_SUBDIR)
        typer.echo(f"Using default translation directory: {translation_dir}")

    # Load models
    sbert_model = None
    retrieval_sbert_model = None
    cross_encoder = None

    # Decide which models to load
    retrieval_model_name = retrieval_model or DEFAULT_MODEL
    semantic_model_name = semantic_chunker_model or retrieval_model_name

    # Check if we can use the same model for both tasks
    can_share_model = (
        needs_semantic_model and semantic_model_name == retrieval_model_name
    )

    try:
        # For retrieval
        typer.echo(f"Loading retrieval model: {retrieval_model_name}...")
        retrieval_sbert_model = SentenceTransformer(retrieval_model_name, device=device)

        # For semantic chunking (if needed)
        if needs_semantic_model and not can_share_model:
            typer.echo(f"Loading semantic chunking model: {semantic_model_name}...")
            sbert_model = SentenceTransformer(semantic_model_name, device=device)
        elif needs_semantic_model:
            # Reuse the retrieval model for semantic chunking
            sbert_model = retrieval_sbert_model
    except Exception as e:
        typer.secho(f"Error loading model: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Initialize the DenseRetriever
        typer.echo("Loading HPO term index...")
        retriever = DenseRetriever.from_model_name(
            model=retrieval_sbert_model,  # Pass pre-loaded model
            model_name=retrieval_model_name,
            min_similarity=similarity_threshold,
        )
    except Exception as e:
        typer.secho(f"Error loading HPO index: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Load cross-encoder model if reranking is enabled
    if enable_reranker:
        try:
            # Select the appropriate model based on the reranker mode
            ce_model_name = reranker_model
            if reranker_mode == "monolingual":
                # For monolingual mode, use the language-specific model
                ce_model_name = monolingual_reranker_model

                # Check if translation directory exists for monolingual mode
                if not os.path.exists(translation_dir):
                    warning_msg = (
                        f"Translation directory not found: {translation_dir}. "
                        "Monolingual re-ranking will not work properly."
                    )
                    typer.secho(warning_msg, fg=typer.colors.YELLOW)
                    typer.echo("Falling back to cross-lingual mode.")
                    reranker_mode = "cross-lingual"
                    ce_model_name = reranker_model

            # Load the selected cross-encoder model
            typer.echo(f"Loading cross-encoder model: {ce_model_name}...")
            cross_encoder = reranker.load_cross_encoder(ce_model_name, device)

            if cross_encoder:
                typer.echo(
                    f"Cross-encoder re-ranking enabled in {reranker_mode} mode with model: {ce_model_name}"
                )
            else:
                warning_msg = f"Failed to load cross-encoder model {ce_model_name}, re-ranking will be disabled"
                typer.secho(warning_msg, fg=typer.colors.YELLOW)
                enable_reranker = False
        except Exception as e:
            typer.secho(
                f"Error loading cross-encoder model: {str(e)}", fg=typer.colors.YELLOW
            )
            typer.echo("Re-ranking will be disabled.")
            enable_reranker = False

    # Create the pipeline
    try:
        typer.echo("Initializing text processing pipeline...")
        pipeline = TextProcessingPipeline(
            language=language,
            chunking_pipeline_config=chunking_pipeline_config,
            assertion_config=assertion_config,
            sbert_model_for_semantic_chunking=sbert_model,
        )
    except Exception as e:
        typer.secho(f"Error creating pipeline: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Process the text
    try:
        typer.echo("Processing text through chunking and assertion pipeline...")
        processed_chunks = pipeline.process(raw_text)
        typer.echo(f"Generated {len(processed_chunks)} chunks.")
    except Exception as e:
        typer.secho(f"Error processing text: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Extract HPO terms from each chunk
    typer.echo("Extracting HPO terms from processed chunks...")

    chunk_results = []
    # Track aggregate results across all chunks
    all_hpo_terms = defaultdict(list)  # HPO_ID -> list of evidence

    for i, chunk_data in enumerate(processed_chunks):
        chunk_text = chunk_data["text"]
        assertion_status = chunk_data["status"]
        assertion_details = chunk_data["assertion_details"]

        # Log the assertion status for informational purposes
        # Note: Processing all chunks regardless of assertion status
        typer.echo(f"Chunk {i+1}: Status is {assertion_status.value} - processing")

        # Retrieve HPO terms for this chunk
        try:
            # Get matching HPO terms with the chunk text as query
            results = retriever.query(
                text=chunk_text,
                n_results=max(num_results, rerank_count if enable_reranker else 0),
                include_similarities=True,
            )

            # Filter results based on similarity threshold
            filtered_results = retriever.filter_results(
                results,
                min_similarity=similarity_threshold,
                max_results=max(num_results, rerank_count if enable_reranker else 0),
            )

            # Format the results into a list of dictionaries
            hpo_matches = []
            if filtered_results.get("ids") and filtered_results["ids"][0]:
                for i, doc_id in enumerate(filtered_results["ids"][0]):
                    if i < len(filtered_results["metadatas"][0]):
                        metadata = filtered_results["metadatas"][0][i]
                        similarity = (
                            filtered_results["similarities"][0][i]
                            if filtered_results.get("similarities")
                            else None
                        )

                        # Extract the name from metadata, usually contains "name" field
                        term_name = ""
                        if metadata:
                            if "name" in metadata and metadata["name"]:
                                term_name = metadata["name"]
                            elif "label" in metadata and metadata["label"]:
                                term_name = metadata["label"]
                            # Try to extract from nested properties if available
                            elif "properties" in metadata and isinstance(
                                metadata["properties"], dict
                            ):
                                props = metadata["properties"]
                                if "name" in props and props["name"]:
                                    term_name = props["name"]
                                elif "label" in props and props["label"]:
                                    term_name = props["label"]

                        hpo_match = {
                            "id": doc_id,
                            "name": term_name,
                            "score": similarity,
                            "rank": i,
                            "metadata": metadata,
                        }
                        hpo_matches.append(hpo_match)

                # Apply re-ranking if enabled
                if enable_reranker and cross_encoder and hpo_matches:
                    try:
                        # Prepare candidates for re-ranking
                        candidates_for_reranking = []

                        # Limit to the number of candidates specified by rerank_count
                        candidates_to_rerank = hpo_matches[
                            : min(len(hpo_matches), rerank_count)
                        ]

                        for candidate in candidates_to_rerank:
                            # Get the appropriate comparison text based on reranker mode
                            if reranker_mode == "monolingual":
                                # For monolingual mode, load the translation in the target language
                                try:
                                    comparison_text = load_translation_text(
                                        hpo_id=candidate["id"],
                                        language=language,
                                        translation_dir=translation_dir,
                                    )
                                except Exception as e:
                                    # If translation fails, fall back to English text
                                    comparison_text = candidate["name"]
                                    if debug:
                                        typer.echo(
                                            f"  Translation for {candidate['id']} failed: {str(e)}. Using English text."
                                        )
                            else:  # cross-lingual mode
                                # Use the English label directly
                                comparison_text = candidate["name"]

                            # Format candidate for re-ranking
                            rerank_candidate = {
                                "hpo_id": candidate["id"],
                                "english_doc": candidate["name"],
                                "metadata": candidate["metadata"],
                                "bi_encoder_score": candidate["score"],
                                "rank": candidate["rank"],
                                "comparison_text": comparison_text,
                            }
                            candidates_for_reranking.append(rerank_candidate)

                        # Perform re-ranking
                        if candidates_for_reranking:
                            reranked_candidates = reranker.rerank_with_cross_encoder(
                                query=chunk_text,
                                candidates=candidates_for_reranking,
                                cross_encoder_model=cross_encoder,
                            )

                            # Update hpo_matches with reranked results
                            if reranked_candidates:
                                # Create a mapping of hpo_id to reranked score
                                reranked_scores = {}
                                for i, candidate in enumerate(reranked_candidates):
                                    reranked_scores[candidate["hpo_id"]] = {
                                        "reranker_score": candidate[
                                            "cross_encoder_score"
                                        ],
                                        "new_rank": i,
                                    }

                                # Update original matches with reranker scores
                                for match in hpo_matches:
                                    if match["id"] in reranked_scores:
                                        match["reranker_score"] = reranked_scores[
                                            match["id"]
                                        ]["reranker_score"]
                                        match["reranked_rank"] = reranked_scores[
                                            match["id"]
                                        ]["new_rank"]

                                # If we're using reranking for final ordering, resort matches by reranker score
                                hpo_matches = sorted(
                                    hpo_matches,
                                    key=lambda x: x.get(
                                        "reranker_score", -float("inf")
                                    ),
                                    reverse=True,
                                )[
                                    :num_results
                                ]  # Limit to the requested number of results

                                if debug:
                                    typer.echo(
                                        f"  Re-ranked {len(reranked_candidates)} candidates"
                                    )
                    except Exception as e:
                        typer.secho(
                            f"  Error during re-ranking: {str(e)}",
                            fg=typer.colors.YELLOW,
                        )
                        if debug:
                            import traceback

                            traceback.print_exc()

            if not hpo_matches:
                typer.echo(f"Chunk {i+1}: No HPO terms found")
            else:
                typer.echo(f"Chunk {i+1}: Found {len(hpo_matches)} HPO terms")

            # Store chunk results
            chunk_result = {
                "chunk_id": i + 1,
                "text": chunk_text,
                "status": assertion_status.value,
                "hpo_terms": [],
            }

            # If top_term_per_chunk is enabled, keep only the highest-scoring match
            if top_term_per_chunk and hpo_matches:
                # Sort by score descending
                sorted_matches = sorted(
                    hpo_matches, key=lambda x: x["score"], reverse=True
                )
                # Keep only the top-scoring match
                hpo_matches = [sorted_matches[0]]
                typer.echo(
                    f"Chunk {i+1}: Taking only top term (score: {hpo_matches[0]['score']:.4f})"
                )

            for match in hpo_matches:
                # Basic match information
                hpo_term_info = {
                    "hpo_id": match["id"],
                    "name": match["name"],
                    "score": match["score"],
                    "reranker_score": match.get("reranker_score"),
                }

                # Add to chunk result
                chunk_result["hpo_terms"].append(hpo_term_info)

                # Add to aggregate results
                evidence = {
                    "chunk_id": i + 1,
                    "chunk_text": chunk_text,
                    "status": assertion_status.value,
                    "score": match["score"],
                    "reranker_score": match.get("reranker_score"),
                    "name": match["name"],  # Include the HPO term name in the evidence
                }
                all_hpo_terms[match["id"]].append(evidence)

            chunk_results.append(chunk_result)

        except Exception as e:
            typer.secho(
                f"Error retrieving HPO terms for chunk {i+1}: {str(e)}",
                fg=typer.colors.RED,
            )
            if debug:
                import traceback

                traceback.print_exc()

    # Aggregate results
    typer.echo("Aggregating HPO terms from all chunks...")
    aggregated_results = []

    for hpo_id, evidence_list in all_hpo_terms.items():
        # Get basic HPO information from the first evidence match
        if not evidence_list or not evidence_list[0].get("chunk_id"):
            continue  # Skip if we don't have any evidence for this HPO term

        # Since we already have the name in the evidence, we don't need to query again
        # Just extract the name from the first piece of evidence
        first_evidence = evidence_list[0]

        # Determine the best score
        max_score = max(evidence["score"] for evidence in evidence_list)
        max_reranker_score = max(
            (
                evidence.get("reranker_score", -float("inf"))
                for evidence in evidence_list
            ),
            default=None,
        )

        # Count the types of evidence
        affirmed_count = sum(
            1 for e in evidence_list if e["status"] == AssertionStatus.AFFIRMED.value
        )
        negated_count = sum(
            1 for e in evidence_list if e["status"] == AssertionStatus.NEGATED.value
        )
        normal_count = sum(
            1 for e in evidence_list if e["status"] == AssertionStatus.NORMAL.value
        )
        uncertain_count = sum(
            1 for e in evidence_list if e["status"] == AssertionStatus.UNCERTAIN.value
        )

        # Determine overall status based on evidence
        if affirmed_count > 0 or uncertain_count > 0:
            overall_status = AssertionStatus.AFFIRMED.value
        elif negated_count > 0:
            overall_status = AssertionStatus.NEGATED.value
        elif normal_count > 0:
            overall_status = AssertionStatus.NORMAL.value
        else:
            overall_status = AssertionStatus.UNCERTAIN.value

        # Calculate a confidence score based on all evidence
        evidence_count = len(evidence_list)
        confidence_score = max_score * (
            1 + 0.1 * min(9, evidence_count - 1)
        )  # Boost score based on evidence count

        # Look through the evidence to find the name of the HPO term
        # We need to extract this from the hpo_match data that was saved
        term_name = ""
        for evidence in evidence_list:
            if "name" in evidence:
                term_name = evidence["name"]
                break

        aggregated_results.append(
            {
                "hpo_id": hpo_id,
                "name": term_name,  # Use the name we found in the evidence
                "score": max_score,
                "reranker_score": max_reranker_score,
                "confidence": confidence_score,
                "evidence_count": evidence_count,
                "status": overall_status,
                "affirmed_count": affirmed_count,
                "negated_count": negated_count,
                "normal_count": normal_count,
                "uncertain_count": uncertain_count,
                "evidence": evidence_list,
            }
        )

    # Sort by confidence score (descending)
    aggregated_results.sort(key=lambda x: x["confidence"], reverse=True)

    # Apply min_confidence filtering if specified
    if min_confidence > 0.0:
        filtered_count = len(aggregated_results)
        aggregated_results = [
            result
            for result in aggregated_results
            if result["confidence"] >= min_confidence
        ]
        filtered_count -= len(aggregated_results)
        if filtered_count > 0:
            typer.echo(
                f"Filtered out {filtered_count} results below min_confidence threshold of {min_confidence}"
            )

    # Output the results in the requested format
    if output_format == "json_lines":
        # Output each chunk and its matches as a JSON object per line
        for chunk_result in chunk_results:
            typer.echo(json.dumps(chunk_result))

        # Output aggregated results as a final JSON object
        typer.echo(json.dumps({"aggregated_hpo_terms": aggregated_results}))

    elif output_format == "rich_json_summary":
        # Create a nicely formatted JSON summary
        summary = {
            "document": {
                "language": language,
                "total_chunks": len(processed_chunks),
                "total_hpo_terms": len(aggregated_results),
                "hpo_terms": [
                    {
                        "hpo_id": result["hpo_id"],
                        "name": result["name"],
                        "confidence": result["confidence"],
                        "status": result["status"],
                        "evidence_count": len(result["evidence"]),
                        "top_evidence": (
                            result["evidence"][0]["chunk_text"]
                            if result["evidence"]
                            else ""
                        ),
                    }
                    for result in aggregated_results
                ],
            }
        }
        # Format the JSON nicely
        formatted_json = json.dumps(summary, indent=2, ensure_ascii=False)
        typer.echo(formatted_json)

    elif output_format == "csv_hpo_list":
        # Create a CSV with HPO terms and basic info
        output = StringIO()
        fieldnames = ["hpo_id", "name", "confidence", "status", "evidence_count"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for r in aggregated_results:
            writer.writerow(
                {
                    "hpo_id": r["hpo_id"],
                    "name": r["name"],
                    "confidence": r["confidence"],
                    "status": r["status"],
                    "evidence_count": r["evidence_count"],
                }
            )

        typer.echo(output.getvalue())

    else:
        typer.secho(
            f"Error: Unknown output format '{output_format}'. "
            f"Supported formats: json_lines, rich_json_summary, csv_hpo_list",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Summary
    typer.secho(
        f"\nText processing completed. "
        f"Found {len(aggregated_results)} HPO terms across {len(processed_chunks)} text chunks.",
        fg=typer.colors.GREEN,
    )


# Entry point for direct script execution during development
if __name__ == "__main__":
    app()
