"""CLI helpers for LLM full-text benchmarking."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
from rich.console import Console

from phentrieve.benchmark import llm_benchmark
from phentrieve.benchmark.llm_benchmark import (
    DEFAULT_LLM_BENCHMARK_DATASET,
    DEFAULT_LLM_BENCHMARK_MODE,
)
from phentrieve.llm.config import DEFAULT_LLM_LANGUAGE
from phentrieve.utils import setup_logging_cli

app = typer.Typer(help="Benchmark LLM full-text extraction.")
console = Console()
logger = logging.getLogger(__name__)
DEFAULT_LLM_BENCHMARK_OUTPUT_DIR = Path("results") / "llm"


def _default_output_path() -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_LLM_BENCHMARK_OUTPUT_DIR / f"llm_benchmark_{timestamp}.json"


def _default_artifacts_dir(output_path: Path) -> Path:
    return output_path.parent / output_path.stem


def _artifact_filename_stem(record: dict[str, Any]) -> str:
    case_index = record.get("case_index")
    case_prefix = f"case_{case_index}" if case_index is not None else "case"
    doc_id = str(record.get("doc_id", "")).strip()
    sanitized_doc_id = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_id).strip("._")
    if not sanitized_doc_id:
        return case_prefix
    sanitized_doc_id = re.sub(r"_+", "_", sanitized_doc_id)
    return f"{case_prefix}_{sanitized_doc_id}"


def run_llm_benchmark_cli(
    *,
    test_file: str,
    llm_provider: str | None = None,
    llm_model: str,
    llm_base_url: str | None = None,
    llm_seed: int | None = None,
    llm_mode: str = DEFAULT_LLM_BENCHMARK_MODE,
    llm_internal_mode: str = "whole_document_grounded",
    dataset: str = DEFAULT_LLM_BENCHMARK_DATASET,
    doc_ids: list[str] | None = None,
    output_path: str | None = None,
    checkpoint_path: str | None = None,
    artifacts_dir: str | None = None,
    language: str = DEFAULT_LLM_LANGUAGE,
    prompt_templates_dir: str | None = None,
    input_cost_per_1m_tokens: float | None = None,
    output_cost_per_1m_tokens: float | None = None,
    cached_input_cost_per_1m_tokens: float | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run the LLM benchmark and persist summary plus comparison artifacts."""
    setup_logging_cli(debug=debug)
    logger.info(
        "Benchmark CLI input: test_file=%s model=%s mode=%s dataset=%s output_path=%s",
        test_file,
        llm_model,
        llm_mode,
        dataset,
        output_path or "default",
    )
    test_file_path = Path(test_file)
    if not test_file_path.exists():
        raise ValueError(f"Benchmark test file not found: {test_file}")

    if test_file_path.is_file():
        try:
            json.loads(test_file_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(
                f"Benchmark test file must be valid JSON: {test_file}"
            ) from exc

    resolved_output_path = Path(output_path) if output_path else _default_output_path()
    resolved_checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path else resolved_output_path
    )
    resolved_artifacts_dir = (
        Path(artifacts_dir)
        if artifacts_dir
        else _default_artifacts_dir(resolved_output_path)
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_artifacts_dir.mkdir(parents=True, exist_ok=True)

    existing_checkpoint = (
        _load_checkpoint_payload(
            path=resolved_checkpoint_path,
            current_run={
                "test_file": str(test_file_path),
                "dataset": dataset,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "llm_base_url": llm_base_url,
                "llm_seed": llm_seed,
                "llm_mode": llm_mode,
                "llm_internal_mode": llm_internal_mode,
                "language": language,
                "prompt_templates_dir": prompt_templates_dir,
                "requested_doc_ids": list(doc_ids) if doc_ids else None,
            },
            allow_completed=True,
        )
        if checkpoint_path is not None
        else None
    )

    def _persist_checkpoint(snapshot: dict[str, Any]) -> None:
        checkpoint_payload = dict(snapshot)
        checkpoint_payload["output_path"] = str(resolved_output_path)
        checkpoint_payload["checkpoint_path"] = str(resolved_checkpoint_path)
        checkpoint_payload["artifacts_dir"] = str(resolved_artifacts_dir)
        _write_json_atomic(resolved_checkpoint_path, checkpoint_payload)
        _write_benchmark_artifacts(
            artifacts_dir=resolved_artifacts_dir,
            benchmark_payload=checkpoint_payload,
        )

    result = llm_benchmark.run_llm_benchmark(
        test_file=test_file,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_seed=llm_seed,
        llm_mode=llm_mode,
        llm_internal_mode=llm_internal_mode,
        dataset=dataset,
        doc_ids=doc_ids,
        language=language,
        prompt_templates_dir=prompt_templates_dir,
        input_cost_per_1m_tokens=input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=output_cost_per_1m_tokens,
        checkpoint_state=existing_checkpoint,
        progress_callback=_persist_checkpoint,
    )

    payload = dict(result)
    payload["output_path"] = str(resolved_output_path)
    payload["checkpoint_path"] = str(resolved_checkpoint_path)
    payload["artifacts_dir"] = str(resolved_artifacts_dir)
    _write_json_atomic(resolved_output_path, payload)
    metrics_path, predictions_dir = _write_benchmark_artifacts(
        artifacts_dir=resolved_artifacts_dir,
        benchmark_payload=payload,
    )
    if metrics_path is not None:
        payload["metrics_path"] = str(metrics_path)
    if predictions_dir is not None:
        payload["predictions_dir"] = str(predictions_dir)
    _write_json_atomic(resolved_output_path, payload)
    _write_json_atomic(resolved_checkpoint_path, payload)
    logger.info("Saved LLM benchmark summary to %s", resolved_output_path)
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.{os.getpid()}.",
            suffix=f"{path.suffix}.tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(json.dumps(payload, indent=2))
            temp_path = Path(temp_file.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _load_checkpoint_payload(
    *,
    path: Path,
    current_run: dict[str, Any],
    allow_completed: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"Checkpoint file must be valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        return None
    if not _checkpoint_matches_run(payload=payload, current_run=current_run):
        raise ValueError(f"Checkpoint does not match current benchmark run: {path}")
    if payload.get("status") != "running" and not allow_completed:
        return None
    return payload


def _checkpoint_matches_run(
    *,
    payload: dict[str, Any],
    current_run: dict[str, Any],
) -> bool:
    return all(payload.get(key) == value for key, value in current_run.items())


def _write_benchmark_artifacts(
    *,
    artifacts_dir: Path,
    benchmark_payload: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    predictions_dir: Path | None = None
    traces_dir: Path | None = None
    prediction_records = benchmark_payload.get("prediction_records")
    if isinstance(prediction_records, list):
        predictions_dir = (
            artifacts_dir / "predictions" / str(benchmark_payload["llm_mode"])
        )
        predictions_dir.mkdir(parents=True, exist_ok=True)
        for record in prediction_records:
            artifact_stem = _artifact_filename_stem(record)
            (predictions_dir / f"{artifact_stem}.json").write_text(
                json.dumps(record, indent=2),
                encoding="utf-8",
            )
            if isinstance(record.get("trace"), dict):
                if traces_dir is None:
                    traces_dir = (
                        artifacts_dir / "traces" / str(benchmark_payload["llm_mode"])
                    )
                    traces_dir.mkdir(parents=True, exist_ok=True)
                (traces_dir / f"{artifact_stem}.json").write_text(
                    json.dumps(record["trace"], indent=2),
                    encoding="utf-8",
                )

    metrics_path: Path | None = None
    metrics = benchmark_payload.get("metrics")
    if isinstance(metrics, dict):
        metrics_dir = artifacts_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"benchmark_{benchmark_payload['llm_mode']}.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "model": benchmark_payload["llm_model"],
                        "mode": benchmark_payload["llm_mode"],
                        "language": benchmark_payload.get("language"),
                        "prompt_templates_dir": benchmark_payload.get(
                            "prompt_templates_dir"
                        ),
                        "dataset": benchmark_payload.get("dataset_metadata", {}),
                        "num_documents": benchmark_payload["cases"],
                    },
                    "assertion_aware_metrics": metrics.get("assertion_aware", {}),
                    "id_only_metrics": metrics.get("id_only", {}),
                    "token_usage": benchmark_payload.get("token_usage", {}),
                    "timing_breakdown": benchmark_payload.get("timing_breakdown", {}),
                    "estimated_cost": benchmark_payload.get("estimated_cost"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    logger.info("Saved benchmark artifacts to %s", artifacts_dir)
    return metrics_path, predictions_dir


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
        typer.Option("--llm-model", help="LLM model to benchmark."),
    ],
    llm_provider: Annotated[
        str | None,
        typer.Option("--llm-provider", help="Optional LLM provider to benchmark."),
    ] = None,
    llm_base_url: Annotated[
        str | None,
        typer.Option(
            "--llm-base-url",
            help="Optional LLM provider base URL for local or proxied deployments.",
        ),
    ] = None,
    llm_seed: Annotated[
        int | None,
        typer.Option(
            "--llm-seed",
            help="Optional provider seed for best-effort reproducibility.",
        ),
    ] = None,
    llm_mode: Annotated[
        Literal["two_phase"],
        typer.Option(
            "--llm-mode", help="LLM extraction mode. Only 'two_phase' is supported."
        ),
    ] = cast(Literal["two_phase"], DEFAULT_LLM_BENCHMARK_MODE),
    llm_internal_mode: Annotated[
        Literal["whole_document_legacy", "whole_document_grounded"],
        typer.Option(
            "--llm-internal-mode",
            help="Internal grounding mode for benchmarking.",
        ),
    ] = "whole_document_grounded",
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help="Dataset subset for PhenoBERT directory input: GSC_plus, ID_68, GeneReviews.",
        ),
    ] = DEFAULT_LLM_BENCHMARK_DATASET,
    doc_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--doc-id",
            help="Benchmark only the selected document id. Repeat for multiple documents.",
        ),
    ] = None,
    output_path: Annotated[
        str | None,
        typer.Option("--output-path", help="Path to save the benchmark summary JSON."),
    ] = None,
    checkpoint_path: Annotated[
        str | None,
        typer.Option(
            "--checkpoint-path",
            help="Path to save per-document benchmark checkpoint state.",
        ),
    ] = None,
    artifacts_dir: Annotated[
        str | None,
        typer.Option(
            "--artifacts-dir",
            help="Directory for benchmark metrics and per-document prediction artifacts.",
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", help="Prompt language for the benchmark run."),
    ] = DEFAULT_LLM_LANGUAGE,
    prompt_templates_dir: Annotated[
        str | None,
        typer.Option(
            "--prompt-templates-dir",
            help="Override the user prompt template directory for this benchmark run.",
        ),
    ] = None,
    input_cost_per_1m_tokens: Annotated[
        float | None,
        typer.Option(
            "--input-cost-per-1m-tokens",
            help="Optional input token price used for estimated benchmark cost reporting.",
        ),
    ] = None,
    output_cost_per_1m_tokens: Annotated[
        float | None,
        typer.Option(
            "--output-cost-per-1m-tokens",
            help="Optional output token price used for estimated benchmark cost reporting.",
        ),
    ] = None,
    cached_input_cost_per_1m_tokens: Annotated[
        float | None,
        typer.Option(
            "--cached-input-cost-per-1m-tokens",
            help="Optional cached-input token price used for estimated benchmark cost reporting.",
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging for the benchmark run."),
    ] = False,
) -> None:
    """Run the LLM full-text benchmark."""
    try:
        result = run_llm_benchmark_cli(
            test_file=test_file,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_seed=llm_seed,
            llm_mode=llm_mode,
            llm_internal_mode=llm_internal_mode,
            dataset=dataset,
            doc_ids=doc_ids,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            artifacts_dir=artifacts_dir,
            language=language,
            prompt_templates_dir=prompt_templates_dir,
            input_cost_per_1m_tokens=input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=output_cost_per_1m_tokens,
            cached_input_cost_per_1m_tokens=cached_input_cost_per_1m_tokens,
            debug=debug,
        )
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    logger.info(
        "Completed LLM benchmark: cases=%s model=%s mode=%s output=%s",
        result["cases"],
        result["llm_model"],
        result["llm_mode"],
        result["output_path"],
    )
    console.print("[bold cyan]LLM benchmark complete[/bold cyan]")
    console.print(f"Cases: {result['cases']}")
    console.print(f"Model: {result['llm_model']}")
    console.print(f"Mode: {result['llm_mode']}")
    console.print(f"Results saved to: {result['output_path']}")
    artifacts_dir_value = result.get("artifacts_dir")
    if artifacts_dir_value:
        console.print(f"Artifacts saved to: {artifacts_dir_value}")
