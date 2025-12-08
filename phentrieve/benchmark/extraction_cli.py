"""CLI for HPO extraction benchmarking."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from phentrieve.benchmark.extraction_benchmark import (
    ExtractionBenchmark,
    ExtractionConfig,
)
from phentrieve.benchmark.extraction_reporter import ExtractionReporter

app = typer.Typer(help="HPO Extraction Benchmarking")
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def run(
    test_path: Path = typer.Argument(
        ..., help="Path to test dataset JSON file or PhenoBERT directory"
    ),
    model: str = typer.Option("BAAI/bge-m3", help="Embedding model to use"),
    language: str = typer.Option("en", help="Language code (en, de, es, fr, nl)"),
    output_dir: Path = typer.Option(
        Path("results/extraction"), help="Output directory for results"
    ),
    dataset: str = typer.Option(
        "all",
        help="PhenoBERT dataset: all, GSC_plus, ID_68, GeneReviews (only for directory input)",
    ),
    averaging: str = typer.Option(
        "micro", help="Averaging strategy: micro, macro, or weighted"
    ),
    include_assertions: bool = typer.Option(
        True, help="Include assertion detection in evaluation"
    ),
    relaxed_matching: bool = typer.Option(
        False, help="Enable hierarchical relaxed matching"
    ),
    bootstrap_ci: bool = typer.Option(
        True, help="Calculate bootstrap confidence intervals"
    ),
    bootstrap_samples: int = typer.Option(
        1000, help="Number of bootstrap samples for CI"
    ),
    chunk_threshold: float = typer.Option(
        0.3, help="Minimum similarity threshold for chunk retrieval"
    ),
    min_confidence: float = typer.Option(
        0.35, help="Minimum confidence for aggregated results"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose output"
    ),
):
    """Run extraction benchmark on test dataset."""
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    console.print("[bold cyan]Running extraction benchmark[/bold cyan]")
    console.print(f"Test path: {test_path}")
    console.print(f"Model: {model}")
    console.print(f"Language: {language}")
    if test_path.is_dir():
        console.print(f"Dataset: {dataset}")

    if not test_path.exists():
        console.print(f"[red]Error: Test path not found: {test_path}[/red]")
        raise typer.Exit(1)

    # Create config
    config = ExtractionConfig(
        model_name=model,
        language=language,
        averaging=averaging,
        include_assertions=include_assertions,
        relaxed_matching=relaxed_matching,
        bootstrap_ci=bootstrap_ci,
        bootstrap_samples=bootstrap_samples,
        chunk_retrieval_threshold=chunk_threshold,
        min_confidence_for_aggregated=min_confidence,
        dataset=dataset,
    )

    # Initialize benchmark
    benchmark = ExtractionBenchmark(model, config=config)

    # Run benchmark
    with console.status("Processing documents..."):
        try:
            metrics = benchmark.run_benchmark(test_path, output_dir)
        except Exception as e:
            console.print(f"[red]Error running benchmark: {e}[/red]")
            if verbose:
                import traceback

                console.print(traceback.format_exc())
            raise typer.Exit(1)

    # Display results
    _display_results(metrics)
    console.print(f"\n[green]Results saved to {output_dir}[/green]")


@app.command()
def compare(
    result1: Path = typer.Argument(..., help="First result file"),
    result2: Path = typer.Argument(..., help="Second result file"),
    output_file: Optional[Path] = typer.Option(None, help="Save comparison"),
):
    """Compare two extraction benchmark results."""
    console.print("[bold cyan]Comparing benchmark results[/bold cyan]")

    if not result1.exists():
        console.print(f"[red]Error: Result file not found: {result1}[/red]")
        raise typer.Exit(1)
    if not result2.exists():
        console.print(f"[red]Error: Result file not found: {result2}[/red]")
        raise typer.Exit(1)

    # Load results
    r1 = _load_results(result1)
    r2 = _load_results(result2)

    # Create comparison table
    table = Table(title="Extraction Benchmark Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Result 1", style="magenta")
    table.add_column("Result 2", style="magenta")
    table.add_column("Difference", style="yellow")

    # Add metrics
    for metric in ["precision", "recall", "f1"]:
        val1 = r1.get("corpus_metrics", {}).get("micro", {}).get(metric, 0)
        val2 = r2.get("corpus_metrics", {}).get("micro", {}).get(metric, 0)
        diff = val2 - val1

        table.add_row(
            f"Micro {metric.capitalize()}",
            f"{val1:.3f}",
            f"{val2:.3f}",
            f"{diff:+.3f}" if diff != 0 else "=",
        )

    console.print(table)

    # Statistical significance test
    ci1 = r1.get("corpus_metrics", {}).get("confidence_intervals", {})
    ci2 = r2.get("corpus_metrics", {}).get("confidence_intervals", {})
    if ci1 and ci2:
        _test_significance(ci1, ci2)

    # Save if requested
    if output_file:
        _save_comparison(r1, r2, output_file)
        console.print(f"[green]Comparison saved to {output_file}[/green]")


@app.command()
def report(
    results_dir: Path = typer.Argument(
        ..., help="Directory containing benchmark results"
    ),
    output_format: str = typer.Option(
        "markdown", help="Output format: markdown, html, or latex"
    ),
    output_file: Optional[Path] = typer.Option(None, help="Save report to file"),
):
    """Generate comprehensive benchmark report."""
    console.print("[bold cyan]Generating extraction benchmark report[/bold cyan]")

    if not results_dir.exists():
        console.print(f"[red]Error: Results directory not found: {results_dir}[/red]")
        raise typer.Exit(1)

    # Load all results in directory
    results = _load_all_results(results_dir)

    if not results:
        console.print("[yellow]No results found in directory[/yellow]")
        raise typer.Exit(1)

    # Generate report
    reporter = ExtractionReporter(output_format)
    report_content = reporter.generate_report(results)

    # Display or save
    if output_file:
        output_file.write_text(report_content)
        console.print(f"[green]Report saved to {output_file}[/green]")
    else:
        console.print(report_content)


def _display_results(metrics):
    """Display benchmark results in a table."""
    table = Table(title="Extraction Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Micro", style="magenta")
    table.add_column("Macro", style="green")
    table.add_column("Weighted", style="yellow")

    metric_names = ["precision", "recall", "f1"]
    for metric in metric_names:
        micro_val = metrics.micro.get(metric, 0)
        macro_val = metrics.macro.get(metric, 0)
        weighted_val = metrics.weighted.get(metric, 0)

        table.add_row(
            metric.capitalize(),
            f"{micro_val:.3f}",
            f"{macro_val:.3f}",
            f"{weighted_val:.3f}",
        )

    console.print(table)

    # Display confidence intervals if available
    if metrics.confidence_intervals:
        ci_table = Table(title="95% Confidence Intervals (Micro)")
        ci_table.add_column("Metric", style="cyan")
        ci_table.add_column("Lower", style="green")
        ci_table.add_column("Upper", style="green")

        for metric in metric_names:
            if metric in metrics.confidence_intervals:
                lower, upper = metrics.confidence_intervals[metric]
                ci_table.add_row(metric.capitalize(), f"{lower:.3f}", f"{upper:.3f}")

        console.print(ci_table)


def _load_results(result_file: Path) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(result_file) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _load_all_results(results_dir: Path) -> list:
    """Load all result files from directory."""
    results = []
    for file_path in results_dir.glob("**/extraction_results.json"):
        with open(file_path) as f:
            results.append(json.load(f))
    return results


def _test_significance(ci1: dict, ci2: dict):
    """Test statistical significance between results using CI overlap."""
    console.print("\n[bold]Statistical Significance (CI Overlap Test)[/bold]")

    for metric in ["precision", "recall", "f1"]:
        if metric not in ci1 or metric not in ci2:
            continue

        lower1, upper1 = ci1[metric]
        lower2, upper2 = ci2[metric]

        # Check if intervals overlap
        overlaps = not (upper1 < lower2 or upper2 < lower1)

        if overlaps:
            console.print(f"  {metric}: CIs overlap - difference NOT significant")
        else:
            console.print(
                f"  {metric}: CIs do NOT overlap - difference likely significant"
            )


def _save_comparison(r1: dict, r2: dict, output_file: Path):
    """Save comparison results to file."""
    comparison = {
        "result1": {
            "model": r1.get("metadata", {}).get("model", "unknown"),
            "metrics": r1.get("corpus_metrics", {}),
        },
        "result2": {
            "model": r2.get("metadata", {}).get("model", "unknown"),
            "metrics": r2.get("corpus_metrics", {}),
        },
        "differences": {},
    }

    # Calculate differences
    for metric in ["precision", "recall", "f1"]:
        val1 = r1.get("corpus_metrics", {}).get("micro", {}).get(metric, 0)
        val2 = r2.get("corpus_metrics", {}).get("micro", {}).get(metric, 0)
        comparison["differences"][metric] = val2 - val1

    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)
