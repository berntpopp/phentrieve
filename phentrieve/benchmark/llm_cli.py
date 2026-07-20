"""CLI helpers for LLM full-text benchmarking."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import asdict
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
    ArtifactEntry,
    create_run_layout,
    publish_manifest_v2,
    reset_run_artifacts,
    sha256_path,
    write_json,
    write_jsonl,
)
from phentrieve.benchmark.run_identity import (
    behavioral_base_url_sha256,
    build_dataset_identity,
    build_run_fingerprints,
    load_retrieval_asset_identity,
    sanitize_behavioral_base_url,
    validate_evaluation_hpo_version,
)
from phentrieve.llm.config import DEFAULT_LLM_LANGUAGE, DEFAULT_OPENROUTER_BASE_URL
from phentrieve.llm.prompts import loader as prompt_loader
from phentrieve.llm.prompts.identity import build_prompt_bundle_identity
from phentrieve.llm.providers.resolver import resolve_llm_provider_request
from phentrieve.utils import setup_logging_cli

app = typer.Typer(help="Benchmark LLM full-text extraction.")
console = Console()
logger = logging.getLogger(__name__)
CHECKPOINT_DEFAULTS: dict[str, Any] = {
    "capture_phase1_debug": False,
    "ontology_aware_metrics": False,
    "ontology_semantic_floor": 0.30,
    "ontology_similarity_formula": "hybrid",
}

# Identity keys with no meaningful historical default: a checkpoint written
# before the key existed carries no evidence either way, so it is resumed under
# the behaviour it was written with rather than being invalidated on upgrade.
# Checkpoints written from now on always carry these and are fully verified.
_UNVERIFIABLE_WHEN_ABSENT = frozenset({"prompt_templates_sha256"})


def _prompt_templates_sha256(prompt_templates_dir: str | None) -> dict[str, str | None]:
    """Hash the prompt templates this run will actually load.

    The prompt is an experimental variable, so results produced under one set of
    templates must not be merged with results produced under another. Recording
    only the templates *directory* in the checkpoint identity is not enough:
    editing a template in place between two runs of the same ``--run-id`` leaves
    the path unchanged, so the resume would silently mix old-prompt and
    new-prompt document outputs into one set of metrics.
    """
    user_dir = (
        Path(prompt_templates_dir)
        if prompt_templates_dir is not None
        else prompt_loader.USER_TEMPLATES_DIR
    )
    package_dir = prompt_loader.PACKAGE_TEMPLATES_DIR
    return {
        "package": sha256_path(package_dir) if package_dir.exists() else None,
        "user": sha256_path(user_dir) if user_dir.exists() else None,
    }


def _derive_run_status(case_records: list[dict[str, Any]]) -> str:
    """Derive run health from per-case outcomes.

    ``run_llm_benchmark`` reports ``completed`` for any run that finishes its
    document loop, including one in which every document failed, so the run
    status has to be recomputed from the cases themselves.
    """
    if not case_records:
        return "complete"
    failed = sum(1 for case in case_records if case.get("status") == "failed")
    if failed == 0:
        return "complete"
    return "failed" if failed == len(case_records) else "partial"


def _build_checkpoint_identity(
    *,
    test_file_path: Path,
    dataset_sha256: str,
    accounting_config: llm_benchmark.BenchmarkAccountingConfig,
    dataset: str,
    resolved_provider: str,
    resolved_model: str,
    resolved_base_url: str | None,
    llm_timeout_seconds: int | None,
    llm_seed: int | None,
    llm_mode: str,
    llm_internal_mode: str,
    language: str,
    capture_phase1_debug: bool,
    ontology_aware_metrics: bool,
    ontology_semantic_floor: float,
    ontology_similarity_formula: str,
    prompt_templates_dir: str | None,
    doc_ids: list[str] | None,
) -> dict[str, Any]:
    """Return every input that determines reusable checkpoint contents."""
    return {
        "test_file": str(test_file_path),
        "dataset_sha256": dataset_sha256,
        "accounting_config": accounting_config.model_dump(mode="json"),
        "dataset": dataset,
        "llm_provider": resolved_provider,
        "llm_model": resolved_model,
        "llm_base_url": resolved_base_url,
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
        "prompt_templates_sha256": _prompt_templates_sha256(prompt_templates_dir),
        "requested_doc_ids": list(doc_ids) if doc_ids else None,
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
    output_path: str | None = None,
    checkpoint_path: str | None = None,
    artifacts_dir: str | None = None,
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
    evaluation_hpo_version: str | None = None,
    _data_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the LLM benchmark and persist canonical run-layout artifacts."""
    if output_path or checkpoint_path or artifacts_dir:
        warnings.warn(
            "output_path, checkpoint_path, and artifacts_dir are deprecated; "
            "use output_dir, run_id, and overwrite instead",
            DeprecationWarning,
            stacklevel=2,
        )
    setup_logging_cli(debug=debug)
    legacy_checkpoint_path = checkpoint_path
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
        reset_existing=False,
    )
    canonical_checkpoint_path = run_layout.checkpoint_path
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
        seed=llm_seed,
    )
    effective_doc_ids = doc_ids or None
    dataset_identity = build_dataset_identity(
        test_file_path,
        dataset,
        effective_doc_ids,
        projection=llm_benchmark.DATASET_ASSERTION_PROJECTION.get(dataset),
    )
    prompt_identity = build_prompt_bundle_identity(
        llm_mode,
        language,
        Path(prompt_templates_dir) if prompt_templates_dir else None,
    )
    retrieval_identity = load_retrieval_asset_identity(_data_dir)
    resolved_evaluation_hpo = evaluation_hpo_version or retrieval_identity.hpo_version
    validate_evaluation_hpo_version(resolved_evaluation_hpo, retrieval_identity)
    model_identity = {
        "provider": resolved_provider_request.provider,
        "model": resolved_provider_request.model,
        "base_url": sanitize_behavioral_base_url(resolved_provider_request.base_url),
        "base_url_behavior_sha256": behavioral_base_url_sha256(
            resolved_provider_request.base_url
        ),
        "seed": resolved_provider_request.seed,
        "timeout_seconds": llm_timeout_seconds,
        "internal_mode": llm_internal_mode,
    }
    fingerprints = build_run_fingerprints(
        dataset_identity, prompt_identity, model_identity, retrieval_identity
    )
    identities = {
        "dataset_identity": asdict(dataset_identity),
        "prompt_identity": asdict(prompt_identity),
        "model_identity": model_identity,
        "evaluation_hpo_version": resolved_evaluation_hpo,
        "retrieval_asset_identity": asdict(retrieval_identity),
        "producer_identity": _build_producer_identity(),
        "execution_fingerprint": fingerprints.execution_sha256,
        "scoring_fingerprint": fingerprints.scoring_sha256,
    }
    checkpoint_configuration = _build_checkpoint_identity(
        test_file_path=test_file_path,
        dataset_sha256=dataset_sha256,
        accounting_config=accounting_config,
        dataset=dataset,
        resolved_provider=resolved_provider_request.provider,
        resolved_model=resolved_provider_request.model,
        resolved_base_url=sanitize_behavioral_base_url(
            resolved_provider_request.base_url
        ),
        llm_timeout_seconds=llm_timeout_seconds,
        llm_seed=llm_seed,
        llm_mode=llm_mode,
        llm_internal_mode=llm_internal_mode,
        language=language,
        capture_phase1_debug=capture_phase1_debug,
        ontology_aware_metrics=ontology_aware_metrics,
        ontology_semantic_floor=ontology_semantic_floor,
        ontology_similarity_formula=ontology_similarity_formula,
        prompt_templates_dir=prompt_templates_dir,
        doc_ids=effective_doc_ids,
    )
    checkpoint_identity = {**checkpoint_configuration, **identities}
    existing_checkpoint = _load_checkpoint_payload(
        path=canonical_checkpoint_path,
        current_run=checkpoint_configuration,
        execution_fingerprint=fingerprints.execution_sha256,
        scoring_fingerprint=fingerprints.scoring_sha256,
        allow_completed=True,
    )
    if overwrite:
        reset_run_artifacts(run_layout)

    def _persist_checkpoint(snapshot: dict[str, Any]) -> None:
        checkpoint_payload = cast(
            dict[str, Any], _sanitize_persisted_base_urls(snapshot)
        )
        checkpoint_payload.update(checkpoint_identity)
        checkpoint_payload["run_id"] = run_layout.run_id
        checkpoint_payload["output_dir"] = str(output_dir)
        _write_json_atomic(canonical_checkpoint_path, checkpoint_payload)
        predictions_dir, traces_dir, metrics_path = _write_benchmark_artifacts(
            run_dir=run_layout.run_dir,
            benchmark_payload=checkpoint_payload,
        )
        partial_metadata = _manifest_metadata(
            payload=checkpoint_payload,
            dataset_sha256=dataset_sha256,
            test_file_path=test_file_path,
            status="partial",
        )
        partial_inventory = [
            ArtifactEntry(canonical_checkpoint_path, "checkpoint", "application/json")
        ]
        if metrics_path is not None:
            partial_inventory.append(
                ArtifactEntry(metrics_path, "metrics", "application/json")
            )
        for role, directory in (("prediction", predictions_dir), ("trace", traces_dir)):
            if directory is not None:
                partial_inventory.extend(
                    ArtifactEntry(path, role, "application/json")
                    for path in sorted(directory.rglob("*.json"))
                )
        publish_manifest_v2(
            run_layout, {**partial_metadata, **identities}, partial_inventory
        )

    result = llm_benchmark.run_llm_benchmark(
        test_file=test_file,
        llm_provider=resolved_provider_request.provider,
        llm_model=resolved_provider_request.model,
        llm_base_url=resolved_provider_request.base_url,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_seed=llm_seed,
        llm_mode=llm_mode,
        llm_internal_mode=llm_internal_mode,
        dataset=dataset,
        doc_ids=effective_doc_ids,
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
        _resolved_provider_request=resolved_provider_request,
    )

    payload = cast(dict[str, Any], _sanitize_persisted_base_urls(result))
    payload.update(checkpoint_identity)
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

    manifest_status = (
        "failed"
        if result.get("status") == "failed"
        else _derive_run_status(case_records)
    )
    _write_json_atomic(canonical_checkpoint_path, payload)
    inventory = [
        ArtifactEntry(run_layout.summary_path, "summary", "application/json"),
        ArtifactEntry(canonical_checkpoint_path, "checkpoint", "application/json"),
        ArtifactEntry(run_layout.terms_path, "term_results", "application/x-ndjson"),
        ArtifactEntry(run_layout.cases_path, "case_results", "application/x-ndjson"),
    ]
    for role, path, media_type in (("metrics", metrics_path, "application/json"),):
        if path is not None and path.is_file():
            inventory.append(ArtifactEntry(path, role, media_type))
    for role, directory in (("prediction", predictions_dir), ("trace", traces_dir)):
        if directory is not None:
            inventory.extend(
                ArtifactEntry(path, role, "application/json")
                for path in sorted(directory.rglob("*.json"))
            )
    publish_manifest_v2(
        run_layout,
        {
            **_manifest_metadata(
                payload=payload,
                dataset_sha256=dataset_sha256,
                test_file_path=test_file_path,
                status=manifest_status,
            ),
            **identities,
        },
        inventory,
    )
    _write_legacy_artifacts(
        payload=payload,
        output_path=output_path,
        checkpoint_path=legacy_checkpoint_path,
        artifacts_dir=artifacts_dir,
    )
    logger.info("Saved LLM benchmark summary to %s", run_layout.summary_path)
    return payload


def _build_producer_identity() -> dict[str, str | None]:
    """Return source provenance without making Git availability a runtime requirement."""
    from phentrieve import __version__

    executable = shutil.which("git")
    if executable is None:
        return {
            "phentrieve_version": __version__,
            "commit": None,
            "provenance_status": "git_unavailable",
        }
    package_repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(  # noqa: S603 - executable resolved by shutil.which
            [
                executable,
                "-C",
                str(package_repository),
                "rev-parse",
                "--show-toplevel",
                "--is-inside-work-tree",
                "HEAD",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return {
            "phentrieve_version": __version__,
            "commit": None,
            "provenance_status": "git_unavailable",
        }
    output = completed.stdout.splitlines() if completed.returncode == 0 else []
    commit: str | None = None
    if len(output) == 3 and output[1].strip() == "true":
        top_level = Path(output[0]).resolve()
        candidate = output[2].strip()
        if package_repository == top_level and re.fullmatch(
            r"[0-9a-fA-F]{40}", candidate
        ):
            commit = candidate.lower()
    return {
        "phentrieve_version": __version__,
        "commit": commit,
        "provenance_status": "resolved" if commit else "git_error",
    }


def _sanitize_persisted_base_urls(value: Any, *, key: str | None = None) -> Any:
    """Recursively remove credentials and incidental URL components."""
    if isinstance(value, dict):
        return {
            child_key: _sanitize_persisted_base_urls(child, key=child_key)
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_persisted_base_urls(child) for child in value]
    if key is not None and key.endswith("base_url") and isinstance(value, str):
        return sanitize_behavioral_base_url(value)
    return value


def _write_legacy_artifacts(
    *,
    payload: dict[str, Any],
    output_path: str | None,
    checkpoint_path: str | None,
    artifacts_dir: str | None,
) -> None:
    """Preserve deprecated output locations during the migration period."""
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload["output_path"] = output_path
        _write_json_atomic(output, payload)
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload["checkpoint_path"] = checkpoint_path
        _write_json_atomic(checkpoint, payload)
    if artifacts_dir is None:
        return
    predictions, traces, metrics = _write_benchmark_artifacts(
        run_dir=Path(artifacts_dir), benchmark_payload=payload
    )
    payload["artifacts_dir"] = artifacts_dir
    if predictions is not None:
        payload["legacy_predictions_dir"] = str(predictions)
    if traces is not None:
        payload["legacy_traces_dir"] = str(traces)
    if metrics is not None:
        payload["legacy_metrics_path"] = str(metrics)


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
    current_run: dict[str, Any] | None = None,
    execution_fingerprint: str | None = None,
    scoring_fingerprint: str | None = None,
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
    if execution_fingerprint is not None:
        if payload.get("execution_fingerprint") != execution_fingerprint:
            raise ValueError(
                f"Checkpoint execution fingerprint mismatch: {path}. "
                "Use a new --run-id or remove the existing run deliberately."
            )
        if payload.get("scoring_fingerprint") != scoring_fingerprint:
            raise ValueError(
                f"Checkpoint scoring fingerprint mismatch: {path}. "
                "Use a new --run-id or remove the existing run deliberately."
            )
    if current_run is not None and not _checkpoint_matches_run(
        payload=payload, current_run=current_run
    ):
        raise ValueError(
            f"Checkpoint configuration mismatch: {path}. "
            "Use a new --run-id or remove the existing run deliberately."
        )
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
        if key in payload or key not in _UNVERIFIABLE_WHEN_ABSENT
    )


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
        case_records = payload["case_records"] or []
        counts["cases"] = len(case_records)
        counts["failed"] = sum(
            1 for case in case_records if case.get("status") == "failed"
        )
        counts["complete"] = counts["cases"] - counts["failed"]
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
            help=(
                "Resume an existing explicit run only when its checkpoint is "
                "compatible."
            ),
        ),
    ] = False,
    output_path: Annotated[
        str | None,
        typer.Option("--output-path", hidden=True),
    ] = None,
    checkpoint_path: Annotated[
        str | None,
        typer.Option("--checkpoint-path", hidden=True),
    ] = None,
    artifacts_dir: Annotated[
        str | None,
        typer.Option("--artifacts-dir", hidden=True),
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
    evaluation_hpo_version: Annotated[
        str | None,
        typer.Option(
            "--evaluation-hpo-version",
            help=(
                "Expected evaluation HPO version; must match the installed "
                "retrieval bundle."
            ),
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
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            artifacts_dir=artifacts_dir,
            language=language,
            prompt_templates_dir=prompt_templates_dir,
            evaluation_hpo_version=evaluation_hpo_version,
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
