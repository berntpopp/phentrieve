"""Benchmark-related commands for Phentrieve CLI.

This module contains commands for running and analyzing benchmarks.
"""

from typing import Annotated, Optional

import typer

# Create the Typer app for this command group
app = typer.Typer()


@app.command("run")
def run_benchmarks(
    test_file: Annotated[
        Optional[str],
        typer.Option("--test-file", help="Test file with benchmark cases"),
    ] = None,
    model_name: Annotated[
        Optional[str], typer.Option("--model-name", help="Model name to benchmark")
    ] = None,
    model_list: Annotated[
        Optional[str],
        typer.Option("--model-list", help="Comma-separated list of models"),
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
        Optional[str],
        typer.Option("--output", help="Output CSV file for detailed results"),
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
        Optional[str],
        typer.Option("--reranker-model", help="Cross-encoder model for re-ranking"),
    ] = None,
    rerank_count: Annotated[
        int, typer.Option("--rerank-count", help="Number of candidates to rerank")
    ] = 10,
    similarity_formula: Annotated[
        str,
        typer.Option(
            "--similarity-formula",
            help="Formula for similarity calculation (hybrid or simple_resnik_like)",
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
    from phentrieve.utils import setup_logging_cli

    setup_logging_cli(debug=debug)

    typer.echo("Starting benchmark evaluation...")

    results = orchestrate_benchmark(
        test_file=test_file or "",
        model_name=model_name or "",
        model_list=model_list or "",
        all_models=all_models,
        similarity_threshold=similarity_threshold,
        cpu=cpu,
        detailed=detailed,
        output=output or "",
        debug=debug,
        create_sample=create_sample,
        trust_remote_code=trust_remote_code,
        enable_reranker=enable_reranker,
        reranker_model=reranker_model,
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


@app.command("compare")
def compare_benchmarks(
    summaries_dir: Annotated[
        Optional[str],
        typer.Option("--summaries-dir", help="Directory with summary files"),
    ] = None,
    output_csv: Annotated[
        Optional[str], typer.Option("--output-csv", help="Path to save comparison CSV")
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
):
    """Compare results from multiple benchmark runs.

    This command analyzes benchmark results from multiple model runs and generates
    a comparison table showing various metrics such as MRR, Hit Rate, and
    semantic similarity.
    """
    import pandas as pd

    from phentrieve.evaluation.comparison_orchestrator import (
        orchestrate_benchmark_comparison,
    )
    from phentrieve.utils import setup_logging_cli

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


@app.command("visualize")
def visualize_benchmarks(
    summaries_dir: Annotated[
        Optional[str],
        typer.Option("--summaries-dir", help="Directory with summary files"),
    ] = None,
    metrics: Annotated[
        str,
        typer.Option(
            "--metrics",
            help="Comma-separated list of metrics to visualize (default: all)",
        ),
    ] = "all",
    output_dir: Annotated[
        Optional[str],
        typer.Option("--output-dir", help="Directory to save visualizations"),
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
