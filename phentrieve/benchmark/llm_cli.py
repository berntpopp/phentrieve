"""CLI helpers for LLM full-text benchmarking."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from phentrieve.benchmark import llm_benchmark
from phentrieve.benchmark.data_loader import DEFAULT_PHENOBERT_DATASET
from phentrieve.benchmark.llm_benchmark import DEFAULT_LLM_BENCHMARK_MODE

app = typer.Typer(help="Benchmark LLM full-text extraction.")
console = Console()
DEFAULT_LLM_BENCHMARK_OUTPUT_DIR = Path("results") / "llm"


def _default_output_path() -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_LLM_BENCHMARK_OUTPUT_DIR / f"llm_benchmark_{timestamp}.json"


def run_llm_benchmark_cli(
    *,
    test_file: str,
    llm_model: str,
    llm_mode: str = DEFAULT_LLM_BENCHMARK_MODE,
    dataset: str = DEFAULT_PHENOBERT_DATASET,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run the lean LLM benchmark and persist the summary JSON."""
    test_file_path = Path(test_file)
    if not test_file_path.exists():
        raise ValueError(f"Benchmark test file not found: {test_file}")

    if test_file_path.is_file():
        try:
            json.loads(test_file_path.read_text(encoding="utf-8"))
        except JSONDecodeError as exc:
            raise ValueError(
                f"Benchmark test file must be valid JSON: {test_file}"
            ) from exc

    result = llm_benchmark.run_llm_benchmark(
        test_file=test_file,
        llm_model=llm_model,
        llm_mode=llm_mode,
        dataset=dataset,
    )
    resolved_output_path = Path(output_path) if output_path else _default_output_path()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(result)
    payload["output_path"] = str(resolved_output_path)
    resolved_output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


@app.callback(invoke_without_command=True)
def benchmark_llm(
    test_file: Annotated[
        str,
        typer.Option(
            "--test-file",
            help="Benchmark input path: PhenoBERT directory or JSON benchmark file.",
        ),
    ],
    llm_model: Annotated[
        str,
        typer.Option("--llm-model", help="Gemini model to benchmark."),
    ],
    llm_mode: Annotated[
        str,
        typer.Option("--llm-mode", help="LLM extraction mode."),
    ] = DEFAULT_LLM_BENCHMARK_MODE,
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help="Dataset subset for PhenoBERT directory input: all, GSC_plus, ID_68, GeneReviews.",
        ),
    ] = DEFAULT_PHENOBERT_DATASET,
    output_path: Annotated[
        str | None,
        typer.Option("--output-path", help="Path to save the benchmark summary JSON."),
    ] = None,
) -> None:
    """Run the LLM full-text benchmark."""
    try:
        result = run_llm_benchmark_cli(
            test_file=test_file,
            llm_model=llm_model,
            llm_mode=llm_mode,
            dataset=dataset,
            output_path=output_path,
        )
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    console.print("[bold cyan]LLM benchmark complete[/bold cyan]")
    console.print(f"Cases: {result['cases']}")
    console.print(f"Model: {result['llm_model']}")
    console.print(f"Mode: {result['llm_mode']}")
    console.print(f"Results saved to: {result['output_path']}")
