"""CLI helpers for LLM full-text benchmarking."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import httpx
import typer
from rich.console import Console

from phentrieve.benchmark import llm_benchmark
from phentrieve.benchmark.llm_benchmark import (
    DEFAULT_LLM_BENCHMARK_DATASET,
    DEFAULT_LLM_BENCHMARK_MODE,
)
from phentrieve.benchmark.result_store import (
    create_run_layout,
    sha256_path,
    write_json,
    write_jsonl,
    write_manifest,
)
from phentrieve.llm.config import DEFAULT_LLM_LANGUAGE, DEFAULT_OPENROUTER_BASE_URL
from phentrieve.llm.providers.resolver import resolve_llm_provider_request
from phentrieve.utils import setup_logging_cli

app = typer.Typer(help="Benchmark LLM full-text extraction.")
console = Console()
logger = logging.getLogger(__name__)
CHECKPOINT_DEFAULTS: dict[str, Any] = {
    "ontology_aware_metrics": False,
    "ontology_semantic_floor": 0.30,
    "ontology_similarity_formula": "hybrid",
}


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
    llm_timeout_seconds: int | None = None,
    llm_seed: int | None = None,
    llm_mode: str = DEFAULT_LLM_BENCHMARK_MODE,
    llm_internal_mode: str = "whole_document_grounded",
    dataset: str = DEFAULT_LLM_BENCHMARK_DATASET,
    doc_ids: list[str] | None = None,
    output_dir: str = "results",
    run_id: str | None = None,
    overwrite: bool = False,
    language: str = DEFAULT_LLM_LANGUAGE,
    prompt_templates_dir: str | None = None,
    pricing_config: str | None = None,
    input_cost_per_1m_tokens: float | None = None,
    output_cost_per_1m_tokens: float | None = None,
    cached_input_cost_per_1m_tokens: float | None = None,
    capture_phase1_debug: bool = False,
    ontology_aware_metrics: bool = False,
    ontology_semantic_floor: float = 0.30,
    ontology_similarity_formula: str = "hybrid",
    measure_energy: bool = False,
    per_document_energy: bool = False,
    electricity_cost_per_kwh: float | None = None,
    carbon_kg_per_kwh: float | None = None,
    currency: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run the LLM benchmark and persist canonical run-layout artifacts."""
    setup_logging_cli(debug=debug)
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

    dataset_name = dataset if test_file_path.is_dir() else test_file_path.stem
    run_layout = create_run_layout(
        Path(output_dir),
        "llm",
        dataset_name,
        llm_model,
        run_id=run_id,
        exact_run_id=run_id is not None,
        overwrite=overwrite,
    )
    checkpoint_path = run_layout.run_dir / "checkpoint.json"
    dataset_sha256 = sha256_path(test_file_path)
    logger.info(
        "Benchmark CLI input: test_file=%s model=%s mode=%s dataset=%s run_dir=%s",
        test_file,
        llm_model,
        llm_mode,
        dataset,
        run_layout.run_dir,
    )

    accounting_config = _load_accounting_config(
        pricing_config_path=pricing_config,
        llm_provider=llm_provider,
        llm_model=llm_model,
        input_cost_per_1m_tokens=input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=output_cost_per_1m_tokens,
        cached_input_cost_per_1m_tokens=cached_input_cost_per_1m_tokens,
        measure_energy=measure_energy,
        per_document_energy=per_document_energy,
        electricity_cost_per_kwh=electricity_cost_per_kwh,
        carbon_kg_per_kwh=carbon_kg_per_kwh,
        currency=currency,
    )

    # Match against the RESOLVED provider/model/base_url, not the raw CLI
    # inputs: run_llm_benchmark() persists whatever the provider resolver
    # settles on (e.g. the default provider when --llm-provider is omitted,
    # or the model-prefix-inferred provider for "ollama/llama3.1"-style
    # ids), so comparing raw inputs here would spuriously flag every such
    # run as a checkpoint mismatch.
    resolved_provider_request = resolve_llm_provider_request(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    existing_checkpoint = _load_checkpoint_payload(
        path=checkpoint_path,
        current_run={
            "test_file": str(test_file_path),
            "dataset": dataset,
            "llm_provider": resolved_provider_request.provider,
            "llm_model": resolved_provider_request.model,
            "llm_base_url": resolved_provider_request.base_url,
            "llm_timeout_seconds": llm_timeout_seconds,
            "llm_seed": llm_seed,
            "llm_mode": llm_mode,
            "llm_internal_mode": llm_internal_mode,
            "language": language,
            "capture_phase1_debug": capture_phase1_debug,
            "ontology_aware_metrics": ontology_aware_metrics,
            "ontology_semantic_floor": ontology_semantic_floor,
            "ontology_similarity_formula": ontology_similarity_formula,
            "prompt_templates_dir": prompt_templates_dir,
            "requested_doc_ids": list(doc_ids) if doc_ids else None,
        },
        allow_completed=True,
    )

    def _persist_checkpoint(snapshot: dict[str, Any]) -> None:
        checkpoint_payload = dict(snapshot)
        checkpoint_payload["run_id"] = run_layout.run_id
        checkpoint_payload["output_dir"] = str(output_dir)
        _write_json_atomic(checkpoint_path, checkpoint_payload)
        predictions_dir, traces_dir, metrics_path = _write_benchmark_artifacts(
            run_dir=run_layout.run_dir,
            benchmark_payload=checkpoint_payload,
        )
        write_manifest(
            run_layout,
            _manifest_metadata(
                payload=checkpoint_payload,
                dataset_sha256=dataset_sha256,
                test_file_path=test_file_path,
                status="partial",
            ),
            extra_artifacts=_extra_artifacts(
                predictions_dir=predictions_dir,
                traces_dir=traces_dir,
                metrics_path=metrics_path,
            ),
        )

    result = llm_benchmark.run_llm_benchmark(
        test_file=test_file,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_seed=llm_seed,
        llm_mode=llm_mode,
        llm_internal_mode=llm_internal_mode,
        dataset=dataset,
        doc_ids=doc_ids,
        language=language,
        prompt_templates_dir=prompt_templates_dir,
        input_cost_per_1m_tokens=input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=output_cost_per_1m_tokens,
        cached_input_cost_per_1m_tokens=cached_input_cost_per_1m_tokens,
        capture_phase1_debug=capture_phase1_debug,
        ontology_aware_metrics=ontology_aware_metrics,
        ontology_semantic_floor=ontology_semantic_floor,
        ontology_similarity_formula=ontology_similarity_formula,
        accounting_config=accounting_config,
        checkpoint_state=existing_checkpoint,
        progress_callback=_persist_checkpoint,
    )

    payload = dict(result)
    payload["run_id"] = run_layout.run_id
    payload["run_dir"] = str(run_layout.run_dir)
    payload["output_dir"] = str(output_dir)

    predictions_dir, traces_dir, metrics_path = _write_benchmark_artifacts(
        run_dir=run_layout.run_dir,
        benchmark_payload=payload,
    )
    if metrics_path is not None:
        payload["metrics_path"] = str(metrics_path)
    if predictions_dir is not None:
        payload["predictions_dir"] = str(predictions_dir)

    term_records = payload.get("term_records") or []
    case_records = payload.get("case_records") or []
    canonical_summary = {
        key: value
        for key, value in payload.items()
        if key not in {"results", "prediction_records", "term_records", "case_records"}
    }
    canonical_summary["run_id"] = run_layout.run_id
    canonical_summary["benchmark_type"] = "llm"
    canonical_summary["dataset_name"] = dataset_name
    write_json(run_layout.summary_path, canonical_summary)
    write_jsonl(run_layout.terms_path, term_records)
    write_jsonl(run_layout.cases_path, case_records)

    manifest_status = "failed" if result.get("status") == "failed" else "complete"
    write_manifest(
        run_layout,
        _manifest_metadata(
            payload=payload,
            dataset_sha256=dataset_sha256,
            test_file_path=test_file_path,
            status=manifest_status,
        ),
        extra_artifacts=_extra_artifacts(
            predictions_dir=predictions_dir,
            traces_dir=traces_dir,
            metrics_path=metrics_path,
        ),
    )
    _write_json_atomic(checkpoint_path, payload)
    logger.info("Saved LLM benchmark summary to %s", run_layout.summary_path)
    return payload


def _load_accounting_config(
    *,
    pricing_config_path: str | None,
    llm_provider: str | None,
    llm_model: str,
    input_cost_per_1m_tokens: float | None,
    output_cost_per_1m_tokens: float | None,
    cached_input_cost_per_1m_tokens: float | None,
    measure_energy: bool,
    per_document_energy: bool,
    electricity_cost_per_kwh: float | None,
    carbon_kg_per_kwh: float | None,
    currency: str | None,
) -> llm_benchmark.BenchmarkAccountingConfig:
    payload: dict[str, Any] = {}
    if pricing_config_path is not None:
        pricing_path = Path(pricing_config_path)
        try:
            payload = json.loads(pricing_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(
                f"Pricing config file not found: {pricing_config_path}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"Pricing config file must be valid JSON: {pricing_config_path}"
            ) from exc

    config = llm_benchmark.BenchmarkAccountingConfig.model_validate(payload or {})
    should_fetch_openrouter_pricing = (
        pricing_config_path is None
        and (llm_provider or "").strip().lower() == "openrouter"
        and input_cost_per_1m_tokens is None
        and output_cost_per_1m_tokens is None
        and cached_input_cost_per_1m_tokens is None
    )
    if should_fetch_openrouter_pricing:
        fetched_pricing = _fetch_openrouter_token_pricing(llm_model)
        if fetched_pricing is not None:
            config.token_pricing = fetched_pricing
            config.pricing_source = "openrouter_models_api"
    if input_cost_per_1m_tokens is not None:
        config.token_pricing.input_cost_per_1m_tokens = input_cost_per_1m_tokens
        config.pricing_source = "cli"
    if output_cost_per_1m_tokens is not None:
        config.token_pricing.output_cost_per_1m_tokens = output_cost_per_1m_tokens
        config.pricing_source = "cli"
    if cached_input_cost_per_1m_tokens is not None:
        config.token_pricing.cached_input_cost_per_1m_tokens = (
            cached_input_cost_per_1m_tokens
        )
        config.pricing_source = "cli"
    if measure_energy:
        config.energy_accounting.measure_energy = True
    if per_document_energy:
        config.energy_accounting.per_document_energy = True
    if electricity_cost_per_kwh is not None:
        config.energy_accounting.electricity_cost_per_kwh = electricity_cost_per_kwh
    if carbon_kg_per_kwh is not None:
        config.energy_accounting.carbon_kg_per_kwh = carbon_kg_per_kwh
    if currency is not None:
        config.energy_accounting.currency = currency
    return config


def _per_token_usd_to_per_1m_tokens(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(Decimal(str(value)) * Decimal(1_000_000))
    except (InvalidOperation, ValueError):
        return None


def _fetch_openrouter_token_pricing(
    llm_model: str,
) -> llm_benchmark.TokenPricingConfig | None:
    model_id = llm_model.strip()
    if not model_id:
        return None
    url = f"{DEFAULT_OPENROUTER_BASE_URL}/model/{model_id}"
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unable to fetch OpenRouter pricing for model=%s: %s",
            model_id,
            exc,
        )
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    pricing = data.get("pricing") if isinstance(data, dict) else None
    if not isinstance(pricing, dict):
        logger.warning(
            "OpenRouter pricing response for model=%s had no pricing", model_id
        )
        return None

    input_cost = _per_token_usd_to_per_1m_tokens(pricing.get("prompt"))
    output_cost = _per_token_usd_to_per_1m_tokens(pricing.get("completion"))
    cached_input_cost = _per_token_usd_to_per_1m_tokens(pricing.get("input_cache_read"))
    if input_cost is None or output_cost is None:
        logger.warning(
            "OpenRouter pricing response for model=%s lacked prompt/completion prices",
            model_id,
        )
        return None

    return llm_benchmark.TokenPricingConfig(
        input_cost_per_1m_tokens=input_cost,
        output_cost_per_1m_tokens=output_cost,
        cached_input_cost_per_1m_tokens=cached_input_cost,
    )


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
    return all(
        payload.get(key, CHECKPOINT_DEFAULTS.get(key)) == value
        for key, value in current_run.items()
    )


def _extra_artifacts(
    *,
    predictions_dir: Path | None,
    traces_dir: Path | None,
    metrics_path: Path | None,
) -> dict[str, tuple[Path, str]]:
    extra: dict[str, tuple[Path, str]] = {}
    if predictions_dir is not None:
        extra["llm_predictions"] = (predictions_dir, "inode/directory")
    if traces_dir is not None:
        extra["llm_traces"] = (traces_dir, "inode/directory")
    if metrics_path is not None:
        extra["metrics"] = (metrics_path, "application/json")
    return extra


def _manifest_metadata(
    *,
    payload: dict[str, Any],
    dataset_sha256: str,
    test_file_path: Path,
    status: str,
) -> dict[str, Any]:
    dataset_metadata = payload.get("dataset_metadata") or {}
    timing_breakdown = payload.get("timing_breakdown") or {}
    counts: dict[str, int] = {"documents": payload.get("cases", 0)}
    # term_records/case_records only exist on the final payload (written once at
    # completion, per judgment call #2); omit rather than report a misleading 0
    # while a checkpoint is still in progress.
    if "term_records" in payload:
        counts["terms"] = len(payload["term_records"] or [])
    if "case_records" in payload:
        counts["cases"] = len(payload["case_records"] or [])
    return {
        "status": status,
        "elapsed_seconds": timing_breakdown.get("wall_clock_seconds"),
        "dataset": {
            **dataset_metadata,
            "path": str(test_file_path.resolve()),
            "sha256": dataset_sha256,
        },
        "config": {
            "llm_provider": payload.get("llm_provider"),
            "llm_model": payload.get("llm_model"),
            "llm_mode": payload.get("llm_mode"),
            "llm_internal_mode": payload.get("llm_internal_mode"),
            "language": payload.get("language"),
            "ontology_aware_metrics": payload.get("ontology_aware_metrics"),
        },
        "counts": counts,
    }


def _write_benchmark_artifacts(
    *,
    run_dir: Path,
    benchmark_payload: dict[str, Any],
) -> tuple[Path | None, Path | None, Path | None]:
    predictions_dir: Path | None = None
    traces_dir: Path | None = None
    prediction_records = benchmark_payload.get("prediction_records")
    if isinstance(prediction_records, list):
        predictions_dir = run_dir / "predictions" / str(benchmark_payload["llm_mode"])
        predictions_dir.mkdir(parents=True, exist_ok=True)
        for record in prediction_records:
            artifact_stem = _artifact_filename_stem(record)
            (predictions_dir / f"{artifact_stem}.json").write_text(
                json.dumps(record, indent=2),
                encoding="utf-8",
            )
            if isinstance(record.get("trace"), dict):
                if traces_dir is None:
                    traces_dir = run_dir / "traces" / str(benchmark_payload["llm_mode"])
                    traces_dir.mkdir(parents=True, exist_ok=True)
                (traces_dir / f"{artifact_stem}.json").write_text(
                    json.dumps(record["trace"], indent=2),
                    encoding="utf-8",
                )

    metrics_path: Path | None = None
    metrics = benchmark_payload.get("metrics")
    if isinstance(metrics, dict):
        metrics_dir = run_dir / "metrics"
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
                    "estimated_token_cost": benchmark_payload.get(
                        "estimated_token_cost"
                    ),
                    "estimated_energy_cost": benchmark_payload.get(
                        "estimated_energy_cost"
                    ),
                    "estimated_cost": benchmark_payload.get("estimated_cost"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    logger.info("Saved benchmark artifacts to %s", run_dir)
    return predictions_dir, traces_dir, metrics_path


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
    llm_timeout_seconds: Annotated[
        int | None,
        typer.Option(
            "--llm-timeout-seconds",
            help="Optional provider request timeout in seconds.",
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
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", help="Root directory for unique benchmark runs."),
    ] = "results",
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Explicit run identifier; existing runs require --overwrite.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Allow reuse of an existing explicit run directory.",
        ),
    ] = False,
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
    pricing_config: Annotated[
        str | None,
        typer.Option(
            "--pricing-config",
            help="Optional JSON file with benchmark pricing and energy configuration.",
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
    capture_phase1_debug: Annotated[
        bool,
        typer.Option(
            "--capture-phase1-debug/--no-capture-phase1-debug",
            help="Capture phase 1 source text, prompt, and raw structured outputs in per-case traces.",
        ),
    ] = False,
    ontology_aware_metrics: Annotated[
        bool,
        typer.Option(
            "--ontology-aware-metrics/--no-ontology-aware-metrics",
            help="Calculate ontology-aware soft and partial benchmark metrics.",
        ),
    ] = False,
    ontology_semantic_floor: Annotated[
        float,
        typer.Option(
            "--ontology-semantic-floor",
            help="Minimum fallback semantic similarity for ontology-aware credit.",
        ),
    ] = 0.30,
    ontology_similarity_formula: Annotated[
        str,
        typer.Option(
            "--ontology-similarity-formula",
            help="Ontology fallback similarity formula: hybrid or simple_resnik_like.",
        ),
    ] = "hybrid",
    measure_energy: Annotated[
        bool,
        typer.Option(
            "--measure-energy/--no-measure-energy",
            help="Enable optional local benchmark energy accounting.",
        ),
    ] = False,
    per_document_energy: Annotated[
        bool,
        typer.Option(
            "--per-document-energy/--no-per-document-energy",
            help="Capture per-document energy estimates when energy accounting is enabled.",
        ),
    ] = False,
    electricity_cost_per_kwh: Annotated[
        float | None,
        typer.Option(
            "--electricity-cost-per-kwh",
            help="Optional electricity price used with local energy accounting.",
        ),
    ] = None,
    carbon_kg_per_kwh: Annotated[
        float | None,
        typer.Option(
            "--carbon-kg-per-kwh",
            help="Optional carbon intensity used with local energy accounting.",
        ),
    ] = None,
    currency: Annotated[
        str | None,
        typer.Option(
            "--currency",
            help="Optional currency label for user-supplied monetary estimates.",
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
            llm_timeout_seconds=llm_timeout_seconds,
            llm_seed=llm_seed,
            llm_mode=llm_mode,
            llm_internal_mode=llm_internal_mode,
            dataset=dataset,
            doc_ids=doc_ids,
            output_dir=output_dir,
            run_id=run_id,
            overwrite=overwrite,
            language=language,
            prompt_templates_dir=prompt_templates_dir,
            pricing_config=pricing_config,
            input_cost_per_1m_tokens=input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=output_cost_per_1m_tokens,
            cached_input_cost_per_1m_tokens=cached_input_cost_per_1m_tokens,
            capture_phase1_debug=capture_phase1_debug,
            ontology_aware_metrics=ontology_aware_metrics,
            ontology_semantic_floor=ontology_semantic_floor,
            ontology_similarity_formula=ontology_similarity_formula,
            measure_energy=measure_energy,
            per_document_energy=per_document_energy,
            electricity_cost_per_kwh=electricity_cost_per_kwh,
            carbon_kg_per_kwh=carbon_kg_per_kwh,
            currency=currency,
            debug=debug,
        )
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    logger.info(
        "Completed LLM benchmark: cases=%s model=%s mode=%s run_dir=%s",
        result["cases"],
        result["llm_model"],
        result["llm_mode"],
        result["run_dir"],
    )
    console.print("[bold cyan]LLM benchmark complete[/bold cyan]")
    console.print(f"Cases: {result['cases']}")
    console.print(f"Model: {result['llm_model']}")
    console.print(f"Mode: {result['llm_mode']}")
    console.print(f"\n[green]Results saved to {result['run_dir']}[/green]")
