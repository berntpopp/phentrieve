"""Lean LLM benchmark entrypoint for full-text extraction."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from phentrieve.benchmark import energy
from phentrieve.benchmark.data_loader import (
    CANONICAL_ASSERTION_MAP,
    load_benchmark_data,
    normalize_benchmark_assertion,
    parse_gold_terms,
)
from phentrieve.benchmark.run_identity import RetrievalAssetIdentity
from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    ExtractionResult,
    serialize_ontology_metrics,
)
from phentrieve.llm.config import DEFAULT_LLM_LANGUAGE, DEFAULT_PROVIDER_NAME
from phentrieve.llm.pipeline import (
    LLMPipelinePhaseError,
    TwoPhaseLLMPipeline,
    _render_phase1_user_prompt,
)
from phentrieve.llm.preprocessing import (
    build_extraction_groups,
    build_grounded_chunks_from_text_pipeline,
)
from phentrieve.llm.prompts import loader as prompt_loader
from phentrieve.llm.prompts.loader import get_prompt
from phentrieve.llm.provider import get_llm_provider
from phentrieve.llm.providers.base import ResolvedLLMProviderRequest
from phentrieve.llm.tools import ToolExecutor
from phentrieve.llm.types import AnnotationMode, GroundedChunk, LLMPipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_LLM_BENCHMARK_DATASET = "GeneReviews"
DEFAULT_LLM_BENCHMARK_MODE = "two_phase"
DEFAULT_METRIC_AVERAGING = "micro"
DEFAULT_ID_ONLY_ASSERTION = "PRESENT"
ASSERTION_PROJECTION_SCHEMA = "phentrieve-assertion-projection/v2"

DATASET_ASSERTION_PROJECTION: dict[str, dict[str, str | None]] = {
    "GeneReviews": {
        "PRESENT": "PRESENT",
        "ABSENT": None,
        "UNCERTAIN": "PRESENT",
        "FAMILY_HISTORY": None,
    },
    "CSC": {
        "PRESENT": "PRESENT",
        "ABSENT": None,
        "UNCERTAIN": None,
        "FAMILY_HISTORY": None,
    },
    "GSC": {
        "PRESENT": "PRESENT",
        "ABSENT": None,
        "UNCERTAIN": None,
        "FAMILY_HISTORY": None,
    },
}


def resolve_dataset_assertion_projection(
    dataset: str, source_dataset: str | None = None
) -> dict[str, str | None] | None:
    """Return the assertion projection used by runtime scoring."""
    effective_dataset = (
        source_dataset if dataset == "all" and source_dataset else dataset
    )
    return DATASET_ASSERTION_PROJECTION.get(effective_dataset)


def describe_dataset_assertion_projection(dataset: str) -> dict[str, object]:
    """Return a canonical identity payload for the runtime projection."""
    projection = resolve_dataset_assertion_projection(dataset)
    source_mappings = (
        {
            name: (
                dict(DATASET_ASSERTION_PROJECTION[name])
                if name in DATASET_ASSERTION_PROJECTION
                else None
            )
            for name in ("CSC", "GSC", "GeneReviews", "GSC_plus", "ID_68")
        }
        if dataset == "all"
        else None
    )
    return {
        "schema_version": ASSERTION_PROJECTION_SCHEMA,
        "mode": (
            "source_mapped"
            if source_mappings is not None
            else "mapped"
            if projection is not None
            else "normalized_passthrough"
        ),
        "mapping": dict(projection) if projection is not None else None,
        "source_mappings": source_mappings,
        "normalization": {
            "algorithm": "strip_casefold_lookup_default_v1",
            "mapping": dict(CANONICAL_ASSERTION_MAP),
            "unknown_prediction": DEFAULT_ID_ONLY_ASSERTION,
            "other_prediction": "drop_when_mapped",
            "unknown_gold": "reject",
        },
    }


def build_ontology_credit_config(
    *,
    semantic_floor: float,
    similarity_formula: str,
) -> Any:
    from phentrieve.evaluation.ontology_credit import (
        build_ontology_credit_config as build,
    )

    return build(
        semantic_floor=semantic_floor,
        similarity_formula=similarity_formula,
    )


def validate_hpo_graph_available() -> None:
    from phentrieve.evaluation.ontology_credit import (
        validate_hpo_graph_available as validate,
    )

    validate()


def _public_pipeline_error(exc: LLMPipelinePhaseError) -> tuple[str, str]:
    """Return stable persisted failure details without provider exception text."""
    phase = exc.phase if exc.phase in {"phase1", "phase2a", "phase2b"} else "pipeline"
    return f"{phase}_error", f"Benchmark phase {phase} failed"


class TokenPricingConfig(BaseModel):
    input_cost_per_1m_tokens: float | None = None
    output_cost_per_1m_tokens: float | None = None
    cached_input_cost_per_1m_tokens: float | None = None

    @field_validator("*")
    @classmethod
    def validate_non_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("pricing values must be non-negative")
        return value


class EnergyAccountingConfig(BaseModel):
    measure_energy: bool = False
    per_document_energy: bool = False
    electricity_cost_per_kwh: float | None = None
    carbon_kg_per_kwh: float | None = None
    currency: str | None = None
    country_iso_code: str | None = None
    region: str | None = None

    @field_validator("electricity_cost_per_kwh", "carbon_kg_per_kwh")
    @classmethod
    def validate_non_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("pricing values must be non-negative")
        return value


class BenchmarkAccountingConfig(BaseModel):
    token_pricing: TokenPricingConfig = Field(default_factory=TokenPricingConfig)
    energy_accounting: EnergyAccountingConfig = Field(
        default_factory=EnergyAccountingConfig
    )
    pricing_source: str | None = None


def _build_accounting_config(
    *,
    accounting_config: BenchmarkAccountingConfig | None,
    input_cost_per_1m_tokens: float | None,
    output_cost_per_1m_tokens: float | None,
    cached_input_cost_per_1m_tokens: float | None,
) -> BenchmarkAccountingConfig:
    if accounting_config is not None:
        return accounting_config
    return BenchmarkAccountingConfig(
        token_pricing=TokenPricingConfig(
            input_cost_per_1m_tokens=input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=output_cost_per_1m_tokens,
            cached_input_cost_per_1m_tokens=cached_input_cost_per_1m_tokens,
        )
    )


def _build_energy_cost(
    *,
    run_energy: dict[str, Any],
    config: EnergyAccountingConfig,
) -> dict[str, Any] | None:
    measurement_source = str(run_energy.get("measurement_source", "disabled"))
    payload: dict[str, Any] = {"measurement_source": measurement_source}
    if "reason" in run_energy:
        payload["reason"] = run_energy["reason"]

    energy_kwh = run_energy.get("energy_kwh")
    if energy_kwh is not None:
        payload["energy_kwh"] = round(float(energy_kwh), 6)

    carbon_kg = run_energy.get("carbon_kg")
    if carbon_kg is not None:
        payload["carbon_kg"] = round(float(carbon_kg), 6)

    if energy_kwh is not None and config.electricity_cost_per_kwh is not None:
        payload["electricity_cost"] = round(
            float(energy_kwh) * config.electricity_cost_per_kwh, 8
        )
        if config.currency:
            payload["currency"] = config.currency

    return payload


def _checkpoint_record_is_reusable(record: dict[str, Any]) -> bool:
    return str(record.get("status", "")).lower() != "failed"


def _build_grounded_chunks(
    *,
    text: str,
    language: str,
    chunking_pipeline_config: list[dict[str, Any]] | None,
    assertion_config: dict[str, Any] | None,
    retrieval_model_name: str,
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "status": chunk.status,
        }
        for chunk in build_grounded_chunks_from_text_pipeline(
            text=text,
            language=language,
            chunking_pipeline_config=chunking_pipeline_config,
            assertion_config=assertion_config,
            retrieval_model_name=retrieval_model_name,
        )
    ]


def _pipeline_run_supports_extraction_groups(pipeline: Any) -> bool:
    """Detect whether a pipeline.run implementation accepts extraction_groups."""
    try:
        signature = inspect.signature(pipeline.run)
    except (TypeError, ValueError):
        return False
    return "extraction_groups" in signature.parameters


def _build_provider_factory_kwargs(
    provider_factory: Any,
    **candidate_kwargs: Any,
) -> dict[str, Any]:
    try:
        signature = inspect.signature(provider_factory)
    except (TypeError, ValueError):
        return candidate_kwargs

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return candidate_kwargs

    return {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters
    }


def run_llm_benchmark(
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
    language: str = DEFAULT_LLM_LANGUAGE,
    prompt_templates_dir: str | None = None,
    input_cost_per_1m_tokens: float | None = None,
    output_cost_per_1m_tokens: float | None = None,
    cached_input_cost_per_1m_tokens: float | None = None,
    capture_phase1_debug: bool = False,
    ontology_aware_metrics: bool = False,
    ontology_semantic_floor: float = 0.30,
    ontology_similarity_formula: str = "hybrid",
    accounting_config: BenchmarkAccountingConfig | None = None,
    checkpoint_state: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    _resolved_provider_request: ResolvedLLMProviderRequest | None = None,
    _retrieval_asset_identity: RetrievalAssetIdentity | None = None,
    _retrieval_index_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the LLM benchmark directly against the configured provider."""
    if llm_mode != DEFAULT_LLM_BENCHMARK_MODE:
        raise ValueError(
            f"Unsupported llm_mode: {llm_mode!r}. Expected {DEFAULT_LLM_BENCHMARK_MODE!r}."
        )
    if llm_internal_mode not in {
        "whole_document_legacy",
        "whole_document_grounded",
    }:
        raise ValueError(
            "Unsupported llm_internal_mode: "
            f"{llm_internal_mode!r}. Expected 'whole_document_legacy' or "
            "'whole_document_grounded'."
        )

    ontology_config = None
    if ontology_aware_metrics:
        ontology_config = build_ontology_credit_config(
            semantic_floor=ontology_semantic_floor,
            similarity_formula=ontology_similarity_formula,
        )
        validate_hpo_graph_available()

    test_path = Path(test_file)
    logger.info(
        "Starting LLM benchmark: test_file=%s model=%s mode=%s internal_mode=%s dataset=%s",
        test_path,
        llm_model,
        llm_mode,
        llm_internal_mode,
        dataset,
    )
    test_data = load_benchmark_data(test_path, dataset=dataset)
    documents = test_data["documents"]
    if doc_ids:
        documents_by_id = {str(document["id"]): document for document in documents}
        ordered_doc_ids = [doc_id for doc_id in doc_ids if doc_id]
        documents = [
            documents_by_id[doc_id]
            for doc_id in ordered_doc_ids
            if doc_id in documents_by_id
        ]
        if not documents:
            raise ValueError(
                "No benchmark documents matched the requested doc_ids: "
                + ", ".join(doc_ids)
            )
    logger.info(
        "Loaded %d benchmark documents from %s",
        len(documents),
        test_data.get("metadata", {}).get("dataset_name", test_path.stem),
    )
    resolved_accounting_config = _build_accounting_config(
        accounting_config=accounting_config,
        input_cost_per_1m_tokens=input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=output_cost_per_1m_tokens,
        cached_input_cost_per_1m_tokens=cached_input_cost_per_1m_tokens,
    )
    resolved_request = _resolved_provider_request
    if resolved_request is not None and (
        resolved_request.provider != llm_provider
        or resolved_request.model != llm_model
        or resolved_request.base_url != llm_base_url
        or resolved_request.seed != llm_seed
    ):
        raise ValueError("Resolved provider request does not match benchmark arguments")
    provider_factory_kwargs = _build_provider_factory_kwargs(
        get_llm_provider,
        llm_model=resolved_request.model if resolved_request else llm_model,
        llm_provider=resolved_request.provider if resolved_request else llm_provider,
        llm_base_url=resolved_request.base_url if resolved_request else llm_base_url,
        api_key=resolved_request.api_key if resolved_request else None,
        timeout_seconds=llm_timeout_seconds,
        seed=resolved_request.seed if resolved_request else llm_seed,
    )
    provider = get_llm_provider(**provider_factory_kwargs)
    resolved_provider_name = getattr(
        provider, "provider_name", llm_provider or DEFAULT_PROVIDER_NAME
    )
    resolved_model_name = getattr(provider, "model_name", llm_model)
    resolved_base_url = getattr(provider, "base_url", None)
    if not isinstance(resolved_base_url, str) or not resolved_base_url:
        resolved_base_url = llm_base_url
    if resolved_request is not None and (
        resolved_provider_name != resolved_request.provider
        or resolved_model_name != resolved_request.model
        or resolved_base_url != resolved_request.base_url
    ):
        raise ValueError("Provider runtime identity mismatch with resolved request")
    logger.debug(
        "Initialized benchmark pipeline for model=%s mode=%s",
        llm_model,
        llm_mode,
    )

    restored_state = _restore_checkpoint_state(
        checkpoint_state=checkpoint_state,
        documents=documents,
    )
    assertion_results: list[ExtractionResult] = restored_state["assertion_results"]
    id_only_results: list[ExtractionResult] = restored_state["id_only_results"]
    results: list[dict[str, Any]] = restored_state["results"]
    prediction_records: list[dict[str, Any]] = restored_state["prediction_records"]
    total_prompt_tokens = restored_state["total_prompt_tokens"]
    total_completion_tokens = restored_state["total_completion_tokens"]
    total_api_calls = restored_state["total_api_calls"]
    prior_wall_clock_seconds = restored_state["wall_clock_seconds"]
    completed_case_indexes: set[int] = restored_state["completed_case_indexes"]

    if completed_case_indexes:
        logger.info(
            "Resuming LLM benchmark from checkpoint: completed_cases=%d remaining_cases=%d",
            len(completed_case_indexes),
            max(len(documents) - len(completed_case_indexes), 0),
        )

    benchmark_start_time = time.perf_counter()
    run_energy_tracker = energy.create_energy_tracker(
        resolved_accounting_config.energy_accounting
    )
    run_energy_tracker.start_run()
    evaluator = CorpusExtractionMetrics(averaging=DEFAULT_METRIC_AVERAGING)

    with _temporary_prompt_templates_dir(prompt_templates_dir):
        tool_executor = None
        if _retrieval_asset_identity is not None:
            tool_executor = ToolExecutor(
                model_name=_retrieval_asset_identity.embedding_model,
                model_revision=_retrieval_asset_identity.model_revision or None,
                trust_remote_code=_retrieval_asset_identity.trust_remote_code,
                code_revision=_retrieval_asset_identity.code_revision,
                index_dir=_retrieval_index_dir,
                multi_vector=_retrieval_asset_identity.asset_type == "multi_vector",
            )
        pipeline_kwargs: dict[str, Any] = {"provider": provider}
        if tool_executor is not None:
            pipeline_kwargs["tool_executor"] = tool_executor
        pipeline = TwoPhaseLLMPipeline(
            **_build_provider_factory_kwargs(TwoPhaseLLMPipeline, **pipeline_kwargs)
        )
        config = LLMPipelineConfig(
            provider=resolved_provider_name,
            model=resolved_model_name,
            base_url=resolved_base_url,
            mode=llm_mode,
            language=language,
            seed=llm_seed,
            capture_phase1_debug=capture_phase1_debug,
        )
        if len(completed_case_indexes) < len(documents) and hasattr(pipeline, "warmup"):
            logger.info("Benchmark warmup start: language=%s", language)
            warmup_start_time = time.perf_counter()
            pipeline.warmup(language=language)
            logger.info(
                "Benchmark warmup complete: language=%s elapsed=%.3fs",
                language,
                time.perf_counter() - warmup_start_time,
            )

        for index, document in enumerate(documents, start=1):
            if index in completed_case_indexes:
                continue

            doc_id = str(document["id"])
            logger.info(
                "Benchmark document start: %d/%d doc_id=%s",
                index,
                len(documents),
                doc_id,
            )
            gold_terms = parse_gold_terms(document["gold_hpo_terms"])
            gold_ids_only = [
                (hpo_id, DEFAULT_ID_ONLY_ASSERTION) for hpo_id, _ in gold_terms
            ]
            doc_energy_tracker = None
            if resolved_accounting_config.energy_accounting.per_document_energy:
                doc_energy_tracker = energy.create_energy_tracker(
                    resolved_accounting_config.energy_accounting
                )
                doc_energy_tracker.start_run()
            doc_start_time = time.perf_counter()
            grounded_chunks: list[dict[str, Any]] = []
            extraction_groups: list[dict[str, Any]] = []
            active_extraction_groups: list[dict[str, Any]] = []
            try:
                if llm_internal_mode == "whole_document_grounded":
                    try:
                        grounded_chunks = _build_grounded_chunks(
                            text=document["text"],
                            language=language,
                            chunking_pipeline_config=None,
                            assertion_config={"disable": True},
                            retrieval_model_name="FremyCompany/BioLORD-2023-M",
                        )
                        token_count_fn = getattr(provider, "count_tokens", None)
                        if callable(token_count_fn):
                            extraction_prompt = get_prompt(
                                AnnotationMode.TWO_PHASE, language
                            )
                            grounded_chunk_models = [
                                GroundedChunk(
                                    chunk_id=int(chunk["chunk_id"]),
                                    text=str(chunk.get("text", "")),
                                    start_char=chunk.get("start_char"),
                                    end_char=chunk.get("end_char"),
                                    status=str(chunk.get("status", "unknown")),
                                )
                                for chunk in grounded_chunks
                            ]
                            try:
                                user_prompt = _render_phase1_user_prompt(
                                    extraction_prompt=extraction_prompt,
                                    text=document["text"],
                                    grounded_chunks=grounded_chunks,
                                )
                                token_counts = provider.count_tokens(
                                    system_prompt=extraction_prompt.render_system_prompt(),
                                    user_prompt=user_prompt,
                                )
                                total_tokens = int(
                                    token_counts.get("total_tokens")
                                    or token_counts.get("prompt_tokens")
                                    or 0
                                )
                                if total_tokens > 30000:
                                    extraction_groups = [
                                        asdict(group)
                                        for group in build_extraction_groups(
                                            grounded_chunks=grounded_chunk_models,
                                            provider=provider,
                                            system_prompt=extraction_prompt.render_system_prompt(),
                                            max_prompt_tokens=30000,
                                        )
                                    ]
                                    active_extraction_groups = (
                                        extraction_groups
                                        if len(extraction_groups) > 1
                                        else []
                                    )
                            except (NotImplementedError, TypeError):
                                extraction_groups = []
                                active_extraction_groups = []
                    except Exception as exc:
                        raise LLMPipelinePhaseError(
                            "phase1", "Grounded preprocessing failed"
                        ) from exc
                run_kwargs: dict[str, Any] = {
                    "text": document["text"],
                    "grounded_chunks": grounded_chunks,
                    "config": config,
                }
                if (
                    active_extraction_groups
                    and _pipeline_run_supports_extraction_groups(pipeline)
                ):
                    run_kwargs["extraction_groups"] = active_extraction_groups
                pipeline_result = pipeline.run(**run_kwargs)
            except LLMPipelinePhaseError as exc:
                if doc_energy_tracker is not None:
                    doc_energy_tracker.stop_run()
                logger.exception(
                    "Benchmark document failed: %d/%d doc_id=%s phase=%s",
                    index,
                    len(documents),
                    doc_id,
                    exc.phase,
                )
                error_code, error_message = _public_pipeline_error(exc)
                result_record = {
                    "case_index": index,
                    "doc_id": document["id"],
                    "source_dataset": document.get("source_dataset"),
                    "status": "failed",
                    "error_phase": exc.phase,
                    "error_code": error_code,
                    "error_message": error_message,
                }
                results.append(result_record)
                prediction_records.append(
                    {
                        "case_index": index,
                        "doc_id": document["id"],
                        "status": "failed",
                        "error_phase": exc.phase,
                        "error_code": error_code,
                        "error_message": error_message,
                    }
                )
                assertion_results.append(
                    ExtractionResult(
                        doc_id=doc_id,
                        predicted=[],
                        gold=gold_terms,
                    )
                )
                id_only_results.append(
                    ExtractionResult(
                        doc_id=doc_id,
                        predicted=[],
                        gold=gold_ids_only,
                    )
                )
                continue
            doc_elapsed = time.perf_counter() - doc_start_time
            doc_estimated_energy_cost = None
            if doc_energy_tracker is not None:
                doc_estimated_energy_cost = _build_energy_cost(
                    run_energy=doc_energy_tracker.stop_run(),
                    config=resolved_accounting_config.energy_accounting,
                )
            token_usage = _normalize_token_usage(pipeline_result.meta)
            prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
            request_count = int(token_usage.get("api_calls", 0) or 0)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_api_calls += request_count
            estimated_cost = _estimate_cost(
                token_usage=token_usage,
                pricing=resolved_accounting_config.token_pricing,
            )

            predicted_terms = _serialize_predicted_terms(
                pipeline_result,
                dataset=dataset,
                source_dataset=document.get("source_dataset"),
            )
            predicted_with_assertions = sorted(
                _prediction_tuples_with_assertions(predicted_terms)
            )
            predicted_ids_only = sorted(_prediction_tuples_id_only(predicted_terms))

            assertion_results.append(
                ExtractionResult(
                    doc_id=doc_id,
                    predicted=predicted_with_assertions,
                    gold=gold_terms,
                )
            )
            id_only_results.append(
                ExtractionResult(
                    doc_id=doc_id,
                    predicted=predicted_ids_only,
                    gold=gold_ids_only,
                )
            )

            result_record = {
                "case_index": index,
                "doc_id": document["id"],
                "source_dataset": document.get("source_dataset"),
                "expected_hpo_ids": sorted({hpo_id for hpo_id, _ in gold_terms}),
                "predicted_hpo_ids": sorted(
                    {term["term_id"] for term in predicted_terms if term["term_id"]}
                ),
                "predicted_terms": predicted_terms,
                "timing_seconds": round(doc_elapsed, 6),
                "token_usage": token_usage,
                "partial_failure_counts": {
                    "phase1_completed_groups": int(
                        pipeline_result.meta.phase_counts.get(
                            "phase1_completed_groups", 0
                        )
                        or 0
                    ),
                    "phase1_failed_groups": int(
                        pipeline_result.meta.phase_counts.get("phase1_failed_groups", 0)
                        or 0
                    ),
                    "phase1_partial_failures": int(
                        pipeline_result.meta.phase_counts.get(
                            "phase1_partial_failures", 0
                        )
                        or 0
                    ),
                },
                "estimated_cost": estimated_cost,
            }
            prediction_record = _build_prediction_record(
                document=document,
                config=config,
                pipeline_result=pipeline_result,
                grounded_chunks=grounded_chunks,
                extraction_groups=active_extraction_groups,
                predicted_terms=predicted_terms,
                language=language,
                processing_time_seconds=doc_elapsed,
                prompt_templates_dir=prompt_templates_dir,
                estimated_cost=estimated_cost,
                estimated_energy_cost=doc_estimated_energy_cost,
                dataset=dataset,
            )
            results.append(result_record)
            prediction_records.append(prediction_record)

            phase_timings = dict(pipeline_result.meta.phase_timings)
            phase_counts = dict(pipeline_result.meta.phase_counts)
            logger.info(
                "Benchmark document complete: %d/%d doc_id=%s elapsed=%.3fs "
                "phase1=%.3fs phase2a=%.3fs phase2b_local=%.3fs phase2b_llm=%.3fs "
                "requests=%d extracted=%d actionable=%d candidate_sets=%d unresolved=%d "
                "local_matches=%d llm_mapped=%d local_fallbacks=%d",
                index,
                len(documents),
                doc_id,
                doc_elapsed,
                phase_timings.get("phase1_seconds", 0.0),
                phase_timings.get("phase2a_seconds", 0.0),
                phase_timings.get("phase2b_local_seconds", 0.0),
                phase_timings.get("phase2b_llm_seconds", 0.0),
                request_count,
                phase_counts.get("extracted_phrases", 0),
                phase_counts.get("actionable_phrases", 0),
                phase_counts.get("candidate_sets", 0),
                phase_counts.get("unresolved_phrases", 0),
                phase_counts.get("local_matches", 0),
                phase_counts.get("llm_mapped_phrases", 0),
                phase_counts.get("local_fallbacks", 0),
            )

            if progress_callback is not None:
                progress_callback(
                    _build_benchmark_payload(
                        test_path=test_path,
                        dataset=dataset,
                        documents=documents,
                        llm_provider=resolved_provider_name,
                        llm_model=resolved_model_name,
                        llm_base_url=resolved_base_url,
                        llm_timeout_seconds=llm_timeout_seconds,
                        llm_seed=llm_seed,
                        llm_mode=llm_mode,
                        llm_internal_mode=llm_internal_mode,
                        language=language,
                        prompt_templates_dir=prompt_templates_dir,
                        capture_phase1_debug=capture_phase1_debug,
                        ontology_aware_metrics=ontology_aware_metrics,
                        ontology_semantic_floor=ontology_semantic_floor,
                        ontology_similarity_formula=ontology_similarity_formula,
                        dataset_metadata=test_data.get("metadata", {}),
                        total_prompt_tokens=total_prompt_tokens,
                        total_completion_tokens=total_completion_tokens,
                        total_api_calls=total_api_calls,
                        wall_clock_seconds=prior_wall_clock_seconds
                        + (time.perf_counter() - benchmark_start_time),
                        estimated_cost=_estimate_cost(
                            token_usage=_sum_token_usage(results),
                            pricing=resolved_accounting_config.token_pricing,
                        ),
                        pricing_source=resolved_accounting_config.pricing_source,
                        requested_doc_ids=doc_ids,
                        results=results,
                        prediction_records=prediction_records,
                        metrics=None,
                        estimated_energy_cost=None,
                        status="running",
                    )
                )

    assertion_metrics = evaluator.calculate_all_metrics(assertion_results)
    id_only_metrics = evaluator.calculate_all_metrics(id_only_results)
    assertion_metrics_payload = _serialize_corpus_metrics(assertion_metrics)
    id_only_metrics_payload = _serialize_corpus_metrics(id_only_metrics)
    if ontology_config is not None:
        assertion_ontology_metrics = evaluator.calculate_ontology_aware_metrics(
            assertion_results,
            config=ontology_config,
        )
        id_only_ontology_metrics = evaluator.calculate_ontology_aware_metrics(
            id_only_results,
            config=ontology_config,
        )
        assertion_metrics_payload["ontology_metrics"] = serialize_ontology_metrics(
            assertion_ontology_metrics
        )
        id_only_metrics_payload["ontology_metrics"] = serialize_ontology_metrics(
            id_only_ontology_metrics
        )
    logger.info("Calculated benchmark metrics for %d documents", len(results))
    wall_clock_seconds = prior_wall_clock_seconds + (
        time.perf_counter() - benchmark_start_time
    )
    estimated_energy_cost = _build_energy_cost(
        run_energy=run_energy_tracker.stop_run(),
        config=resolved_accounting_config.energy_accounting,
    )

    term_records, case_records = _build_terms_and_cases(documents, results)

    final_payload = _build_benchmark_payload(
        test_path=test_path,
        dataset=dataset,
        documents=documents,
        llm_provider=resolved_provider_name,
        llm_model=resolved_model_name,
        llm_base_url=resolved_base_url,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_seed=llm_seed,
        llm_mode=llm_mode,
        llm_internal_mode=llm_internal_mode,
        language=language,
        prompt_templates_dir=prompt_templates_dir,
        capture_phase1_debug=capture_phase1_debug,
        ontology_aware_metrics=ontology_aware_metrics,
        ontology_semantic_floor=ontology_semantic_floor,
        ontology_similarity_formula=ontology_similarity_formula,
        dataset_metadata=test_data.get("metadata", {}),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_api_calls=total_api_calls,
        wall_clock_seconds=wall_clock_seconds,
        estimated_cost=_estimate_cost(
            token_usage=_sum_token_usage(results),
            pricing=resolved_accounting_config.token_pricing,
        ),
        estimated_energy_cost=estimated_energy_cost,
        pricing_source=resolved_accounting_config.pricing_source,
        requested_doc_ids=doc_ids,
        results=results,
        prediction_records=prediction_records,
        metrics={
            "assertion_aware": assertion_metrics_payload,
            "id_only": id_only_metrics_payload,
        },
        status="completed",
    )
    final_payload["term_records"] = term_records
    final_payload["case_records"] = case_records
    return final_payload


def _restore_checkpoint_state(
    *,
    checkpoint_state: dict[str, Any] | None,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(checkpoint_state, dict):
        return {
            "assertion_results": [],
            "id_only_results": [],
            "results": [],
            "prediction_records": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_api_calls": 0,
            "wall_clock_seconds": 0.0,
            "completed_case_indexes": set(),
        }

    raw_results = checkpoint_state.get("results")
    raw_prediction_records = checkpoint_state.get("prediction_records")
    if not isinstance(raw_results, list) or not isinstance(
        raw_prediction_records, list
    ):
        return {
            "assertion_results": [],
            "id_only_results": [],
            "results": [],
            "prediction_records": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_api_calls": 0,
            "wall_clock_seconds": 0.0,
            "completed_case_indexes": set(),
        }

    restored_results = sorted(
        (
            record
            for record in raw_results
            if isinstance(record, dict) and _checkpoint_record_is_reusable(record)
        ),
        key=lambda record: int(record.get("case_index", 0) or 0),
    )
    completed_case_indexes: set[int] = set()
    assertion_results: list[ExtractionResult] = []
    id_only_results: list[ExtractionResult] = []
    for record in restored_results:
        case_index = int(record.get("case_index", 0) or 0)
        if case_index < 1 or case_index > len(documents):
            raise ValueError("Checkpoint results do not match requested documents.")
        document = documents[case_index - 1]
        if str(record.get("doc_id")) != str(document["id"]):
            raise ValueError("Checkpoint results do not match requested documents.")
        completed_case_indexes.add(case_index)
        gold_terms = parse_gold_terms(document["gold_hpo_terms"])
        gold_ids_only = [
            (hpo_id, DEFAULT_ID_ONLY_ASSERTION) for hpo_id, _ in gold_terms
        ]
        predicted_terms = list(record.get("predicted_terms", []))
        assertion_results.append(
            ExtractionResult(
                doc_id=str(document["id"]),
                predicted=sorted(_prediction_tuples_with_assertions(predicted_terms)),
                gold=gold_terms,
            )
        )
        id_only_results.append(
            ExtractionResult(
                doc_id=str(document["id"]),
                predicted=sorted(_prediction_tuples_id_only(predicted_terms)),
                gold=gold_ids_only,
            )
        )

    restored_usage = _sum_token_usage(restored_results)
    return {
        "assertion_results": assertion_results,
        "id_only_results": id_only_results,
        "results": restored_results,
        "prediction_records": [
            record
            for record in raw_prediction_records
            if isinstance(record, dict) and _checkpoint_record_is_reusable(record)
        ],
        "total_prompt_tokens": int(restored_usage.get("prompt_tokens", 0) or 0),
        "total_completion_tokens": int(restored_usage.get("completion_tokens", 0) or 0),
        "total_api_calls": int(restored_usage.get("api_calls", 0) or 0),
        "wall_clock_seconds": float(
            checkpoint_state.get("timing_breakdown", {}).get("wall_clock_seconds", 0.0)
            or 0.0
        ),
        "completed_case_indexes": completed_case_indexes,
    }


def _normalize_token_usage(meta: Any) -> dict[str, int]:
    usage = dict(getattr(meta, "token_usage", {}) or {})
    if not usage:
        prompt_tokens = int(getattr(meta, "token_input", 0) or 0)
        completion_tokens = int(getattr(meta, "token_output", 0) or 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "api_calls": int(getattr(meta, "request_count", 0) or 0),
        }
    usage.setdefault("prompt_tokens", int(getattr(meta, "token_input", 0) or 0))
    usage.setdefault("completion_tokens", int(getattr(meta, "token_output", 0) or 0))
    usage.setdefault(
        "total_tokens",
        int(usage.get("prompt_tokens", 0) or 0)
        + int(usage.get("completion_tokens", 0) or 0),
    )
    usage.setdefault("api_calls", int(getattr(meta, "request_count", 0) or 0))
    return {key: int(value or 0) for key, value in usage.items()}


def _sum_token_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }
    for record in records:
        token_usage = record.get("token_usage", {})
        if not isinstance(token_usage, dict):
            continue
        for key, value in token_usage.items():
            totals[key] = int(totals.get(key, 0) or 0) + int(value or 0)
    return totals


def _build_terms_and_cases(
    documents: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build canonical terms.jsonl/cases.jsonl records from benchmark results.

    Works uniformly whether ``results`` came from a single run or a
    checkpoint-resumed run: ``_restore_checkpoint_state`` already merges
    prior completed records with newly processed ones into ``results``
    before the caller invokes this, so no resume-specific branching is
    needed here.
    """
    terms: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    ordered_records = sorted(
        results, key=lambda record: int(record.get("case_index", 0) or 0)
    )
    for record in ordered_records:
        case_index = int(record.get("case_index", 0) or 0)
        if case_index < 1 or case_index > len(documents):
            continue
        document = documents[case_index - 1]
        doc_id = str(record.get("doc_id", document.get("id", "")))
        gold_hpo_terms = document.get("gold_hpo_terms", [])
        gold_assertions = dict(parse_gold_terms(gold_hpo_terms))
        gold_labels = {
            str(term.get("id") or term.get("hpo_id") or ""): str(term.get("label", ""))
            for term in gold_hpo_terms
            if isinstance(term, dict)
        }
        gold_ids = set(gold_assertions)

        predicted_terms = list(record.get("predicted_terms", []))
        predicted_by_id: dict[str, dict[str, Any]] = {
            str(term["term_id"]): term
            for term in predicted_terms
            if term.get("term_id")
        }
        predicted_ids = set(predicted_by_id)

        for hpo_id in sorted(gold_ids | predicted_ids):
            is_gold = hpo_id in gold_ids
            is_predicted = hpo_id in predicted_ids
            if is_gold and is_predicted:
                outcome = "tp"
            elif is_predicted:
                outcome = "fp"
            else:
                outcome = "fn"
            predicted_term = predicted_by_id.get(hpo_id, {})
            terms.append(
                {
                    "doc_id": doc_id,
                    "hpo_id": hpo_id,
                    "label": gold_labels.get(hpo_id) or predicted_term.get("label", ""),
                    "is_gold": is_gold,
                    "is_predicted": is_predicted,
                    "gold_assertion": gold_assertions.get(hpo_id),
                    "predicted_assertion": predicted_term.get("assertion"),
                    "outcome": outcome,
                    "evidence": predicted_term.get("evidence"),
                    "category": predicted_term.get("category"),
                }
            )

        expected_hpo_ids = record.get("expected_hpo_ids")
        if not isinstance(expected_hpo_ids, list):
            expected_hpo_ids = sorted(gold_ids)
        predicted_hpo_ids = record.get("predicted_hpo_ids")
        if not isinstance(predicted_hpo_ids, list):
            predicted_hpo_ids = sorted(predicted_ids)
        expected_set = set(expected_hpo_ids)
        predicted_set = set(predicted_hpo_ids)
        cases.append(
            {
                "doc_id": doc_id,
                "expected_hpo_ids": expected_hpo_ids,
                "predicted_hpo_ids": predicted_hpo_ids,
                "metrics": {
                    "tp": len(predicted_set & expected_set),
                    "fp": len(predicted_set - expected_set),
                    "fn": len(expected_set - predicted_set),
                },
                "timing_seconds": record.get("timing_seconds"),
                "token_usage": record.get("token_usage"),
                "estimated_cost": record.get("estimated_cost"),
                "partial_failure_counts": record.get("partial_failure_counts"),
                "status": (
                    "failed"
                    if str(record.get("status", "")).lower() == "failed"
                    else "complete"
                ),
            }
        )
    return terms, cases


def _build_benchmark_payload(
    *,
    test_path: Path,
    dataset: str,
    documents: list[dict[str, Any]],
    llm_provider: str | None,
    llm_model: str,
    llm_base_url: str | None,
    llm_timeout_seconds: int | None,
    llm_seed: int | None,
    llm_mode: str,
    llm_internal_mode: str,
    language: str,
    prompt_templates_dir: str | None,
    capture_phase1_debug: bool,
    ontology_aware_metrics: bool,
    ontology_semantic_floor: float,
    ontology_similarity_formula: str,
    dataset_metadata: dict[str, Any],
    total_prompt_tokens: int,
    total_completion_tokens: int,
    total_api_calls: int,
    wall_clock_seconds: float,
    estimated_cost: dict[str, float] | None,
    estimated_energy_cost: dict[str, Any] | None,
    pricing_source: str | None,
    requested_doc_ids: list[str] | None,
    results: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    metrics: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "test_file": str(test_path),
        "dataset": dataset,
        "cases": len(documents),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "llm_timeout_seconds": llm_timeout_seconds,
        "llm_seed": llm_seed,
        "llm_mode": llm_mode,
        "llm_internal_mode": llm_internal_mode,
        "language": language,
        "prompt_templates_dir": prompt_templates_dir,
        "capture_phase1_debug": capture_phase1_debug,
        "ontology_aware_metrics": ontology_aware_metrics,
        "ontology_semantic_floor": ontology_semantic_floor,
        "ontology_similarity_formula": ontology_similarity_formula,
        "requested_doc_ids": list(requested_doc_ids) if requested_doc_ids else None,
        "dataset_metadata": dataset_metadata,
        "token_usage": {
            **_sum_token_usage(results),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "api_calls": total_api_calls,
        },
        "timing_breakdown": {
            "wall_clock_seconds": round(wall_clock_seconds, 6),
            "avg_seconds_per_case": round(wall_clock_seconds / len(documents), 6)
            if documents
            else 0.0,
        },
        "estimated_token_cost": estimated_cost,
        "estimated_cost": estimated_cost,
        "pricing_source": pricing_source,
        "estimated_energy_cost": estimated_energy_cost,
        "prediction_records": list(prediction_records),
        "results": list(results),
        # Corpus-level visibility into how much of the contract-v2 deliverable
        # (negated/absent/family findings) the present-only scoring drops.
        "assertion_distribution": _aggregate_assertion_distribution(
            list(prediction_records)
        ),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


def _serialize_corpus_metrics(metrics: Any) -> dict[str, Any]:
    return {
        "micro": metrics.micro,
        "macro": metrics.macro,
        "weighted": metrics.weighted,
        "confidence_intervals": metrics.confidence_intervals,
    }


@contextmanager
def _temporary_prompt_templates_dir(prompt_templates_dir: str | None):
    if prompt_templates_dir is None:
        yield
        return

    original_user_templates_dir = prompt_loader.USER_TEMPLATES_DIR
    prompt_loader.USER_TEMPLATES_DIR = Path(prompt_templates_dir)
    prompt_loader.load_prompt_template.cache_clear()
    try:
        yield
    finally:
        prompt_loader.USER_TEMPLATES_DIR = original_user_templates_dir
        prompt_loader.load_prompt_template.cache_clear()


def _estimate_cost(
    *,
    token_usage: dict[str, int],
    pricing: TokenPricingConfig,
) -> dict[str, float | int] | None:
    if (
        pricing.input_cost_per_1m_tokens is None
        or pricing.output_cost_per_1m_tokens is None
    ):
        return None

    prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
    thoughts_tokens = int(token_usage.get("thoughts_tokens", 0) or 0)
    cached_content_tokens = int(token_usage.get("cached_content_tokens", 0) or 0)

    billable_cached_tokens = min(prompt_tokens, max(cached_content_tokens, 0))
    billable_input_tokens = max(prompt_tokens - billable_cached_tokens, 0)
    billable_output_tokens = max(completion_tokens + thoughts_tokens, 0)

    input_cost = billable_input_tokens / 1_000_000 * pricing.input_cost_per_1m_tokens
    cached_input_cost = 0.0
    if pricing.cached_input_cost_per_1m_tokens is not None:
        cached_input_cost = (
            billable_cached_tokens / 1_000_000 * pricing.cached_input_cost_per_1m_tokens
        )
    output_cost = billable_output_tokens / 1_000_000 * pricing.output_cost_per_1m_tokens
    return {
        "input_cost": round(input_cost, 8),
        "cached_input_cost": round(cached_input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(input_cost + cached_input_cost + output_cost, 8),
        "billable_input_tokens": billable_input_tokens,
        "billable_cached_tokens": billable_cached_tokens,
        "billable_output_tokens": billable_output_tokens,
    }


def _build_prediction_record(
    *,
    document: dict[str, Any],
    config: LLMPipelineConfig,
    pipeline_result: Any,
    grounded_chunks: list[dict[str, Any]],
    extraction_groups: list[dict[str, Any]],
    predicted_terms: list[dict[str, str | None]],
    language: str,
    processing_time_seconds: float,
    prompt_templates_dir: str | None,
    estimated_cost: dict[str, float] | None,
    estimated_energy_cost: dict[str, Any] | None,
    dataset: str,
) -> dict[str, Any]:
    token_usage = _normalize_token_usage(pipeline_result.meta)
    phase_timings = dict(pipeline_result.meta.phase_timings)
    phase_counts = dict(pipeline_result.meta.phase_counts)
    phase_request_counts = dict(pipeline_result.meta.phase_request_counts)
    request_count = int(token_usage.get("api_calls", 0) or 0)
    final_annotations = [
        {
            "term_id": term.term_id,
            "label": term.label,
            "assertion": term.assertion,
            "evidence": term.evidence,
            "category": getattr(term, "category", None),
        }
        for term in pipeline_result.terms
    ]
    base_trace = dict(getattr(pipeline_result.meta, "trace", {}) or {})
    trace = {
        **base_trace,
        "final_annotations": final_annotations,
        "projected_predictions": list(predicted_terms),
        "projection": {
            "dataset": dataset,
            "assertion_projection": resolve_dataset_assertion_projection(
                dataset, document.get("source_dataset")
            ),
        },
    }

    return {
        "doc_id": str(document["id"]),
        "source_dataset": document.get("source_dataset"),
        "full_text": document.get("text", ""),
        "language": language,
        "source": "llm_annotation",
        "annotations": [
            {
                "hpo_id": term.term_id,
                "label": term.label,
                "assertion_status": term.assertion,
                "evidence_spans": (
                    [{"text_snippet": term.evidence}] if term.evidence else []
                ),
            }
            for term in pipeline_result.terms
        ],
        # Additive visibility into the contract-v2 deliverable: how many findings
        # were produced per RAW assertion and how many the dataset's present-only
        # projection drops from scoring. Does not change any metric.
        "assertion_distribution": _assertion_distribution(
            pipeline_result,
            dataset=dataset,
            source_dataset=document.get("source_dataset"),
        ),
        "metadata": {
            "llm_provider": pipeline_result.meta.llm_provider,
            "model": config.model,
            "mode": config.mode,
            "seed": config.seed,
            "prompt_version": pipeline_result.meta.prompt_version,
            "prompt_templates_dir": prompt_templates_dir,
            "token_usage": token_usage,
            "processing_time_seconds": round(processing_time_seconds, 6),
            "timing_breakdown": {
                "total_seconds": round(processing_time_seconds, 6),
                "llm_seconds": round(processing_time_seconds, 6),
                "tool_seconds": 0.0,
                **phase_timings,
            },
            "observability": _build_observability_counts(
                phase_counts=phase_counts,
                phase_request_counts=phase_request_counts,
                request_count=request_count,
                grounded_chunk_count=len(grounded_chunks),
                extraction_group_count=len(extraction_groups),
                trace=trace,
                token_count_source=pipeline_result.meta.token_count_source,
            ),
            "estimated_cost": estimated_cost,
            **(
                {"estimated_energy_cost": estimated_energy_cost}
                if estimated_energy_cost is not None
                else {}
            ),
        },
        "trace": trace,
    }


def _serialize_predicted_terms(
    pipeline_result: Any,
    *,
    dataset: str,
    source_dataset: str | None = None,
) -> list[dict[str, str | None]]:
    predicted_terms: list[dict[str, str | None]] = []
    for term in pipeline_result.terms:
        projected_assertion = _project_assertion_for_dataset(
            dataset=dataset,
            source_dataset=source_dataset,
            assertion=getattr(term, "assertion", None),
        )
        if projected_assertion is None:
            continue
        term_id = str(term.term_id).strip() if term.term_id is not None else ""
        predicted_terms.append(
            {
                "term_id": term_id,
                "label": str(term.label),
                "assertion": projected_assertion,
                "evidence": term.evidence,
                "category": getattr(term, "category", None),
            }
        )
    return predicted_terms


def _assertion_distribution(
    pipeline_result: Any, *, dataset: str, source_dataset: str | None = None
) -> dict[str, Any]:
    """Count predicted findings by RAW assertion (pre-projection).

    The LLM benchmark scores a dataset-specific present-only projection (see
    ``DATASET_ASSERTION_PROJECTION``), which drops ``negated``/``absent``/
    ``family_history``. That silently hides the negation/family/qualifier axes
    the extraction contract exists to produce. This report surfaces them without
    changing any score: ``proband_dropped_by_projection`` is exactly the count of
    contract findings the present-only metric cannot see.
    """
    by_assertion: dict[str, int] = {}
    scored = 0
    dropped = 0
    for term in pipeline_result.terms:
        raw = str(getattr(term, "assertion", "") or "").lower()
        by_assertion[raw] = by_assertion.get(raw, 0) + 1
        if (
            _project_assertion_for_dataset(
                dataset=dataset,
                source_dataset=source_dataset,
                assertion=getattr(term, "assertion", None),
            )
            is None
        ):
            dropped += 1
        else:
            scored += 1
    family = list(getattr(pipeline_result, "family_history_findings", []) or [])
    return {
        "proband_by_assertion": by_assertion,
        "proband_scored": scored,
        "proband_dropped_by_projection": dropped,
        "family_history_findings": len(family),
    }


def _aggregate_assertion_distribution(
    prediction_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sum per-document ``assertion_distribution`` blocks into a corpus total.

    ``documents_counted`` vs ``documents_total`` makes an undercount VISIBLE:
    records restored from a checkpoint written before this field existed carry no
    block and contribute nothing, so a resumed cross-version run would otherwise
    silently report a distribution for only the post-upgrade documents.
    """
    by_assertion: dict[str, int] = {}
    scored = 0
    dropped = 0
    family = 0
    counted = 0
    for record in prediction_records:
        dist = record.get("assertion_distribution")
        if not dist:
            continue
        counted += 1
        for key, value in (dist.get("proband_by_assertion") or {}).items():
            by_assertion[key] = by_assertion.get(key, 0) + int(value)
        scored += int(dist.get("proband_scored", 0) or 0)
        dropped += int(dist.get("proband_dropped_by_projection", 0) or 0)
        family += int(dist.get("family_history_findings", 0) or 0)
    return {
        "proband_by_assertion": by_assertion,
        "proband_scored": scored,
        "proband_dropped_by_projection": dropped,
        "family_history_findings": family,
        "documents_counted": counted,
        "documents_total": len(prediction_records),
    }


def _build_observability_counts(
    *,
    phase_counts: dict[str, int],
    phase_request_counts: dict[str, int],
    request_count: int,
    grounded_chunk_count: int,
    extraction_group_count: int,
    trace: dict[str, Any],
    token_count_source: str | None,
) -> dict[str, int | str]:
    """Keep prediction metadata observability shape stable across phase-1 modes."""
    phase1_trace = trace.get("phase1")
    phase1_groups = (
        list(phase1_trace.get("groups", [])) if isinstance(phase1_trace, dict) else []
    )
    phase1_extracted = (
        list(phase1_trace.get("extracted", []))
        if isinstance(phase1_trace, dict)
        else []
    )
    raw_phase1_mentions = sum(
        int(group.get("extracted_count", 0) or 0)
        for group in phase1_groups
        if isinstance(group, dict)
    )
    phase2b_llm_trace = trace.get("phase2b_llm")
    mapping_resolved = (
        list(phase2b_llm_trace.get("resolved", []))
        if isinstance(phase2b_llm_trace, dict)
        else []
    )
    unique_mapping_keys = {
        (
            str(item.get("phrase", "")),
            str(item.get("category", "")),
            str(item.get("selected_id", "")),
            str(item.get("term_id", "")),
            str(item.get("label", "")),
            str(item.get("assertion", "")),
            bool(item.get("local_fallback", False)),
            str(item.get("match_method", "")),
        )
        for item in mapping_resolved
        if isinstance(item, dict)
    }
    return {
        "request_count": request_count,
        **(
            {"token_count_source": token_count_source}
            if token_count_source is not None
            else {}
        ),
        **phase_counts,
        "phase2b_local_accept_count": int(
            phase_counts.get("phase2b_local_accept_count", 0) or 0
        ),
        "phase2b_deferred_count": int(
            phase_counts.get("phase2b_deferred_count", 0) or 0
        ),
        "phase2b_no_candidate_skip_count": int(
            phase_counts.get("phase2b_no_candidate_skip_count", 0) or 0
        ),
        "grounded_chunks": grounded_chunk_count,
        "extraction_groups": extraction_group_count,
        "failed_groups": int(phase_counts.get("phase1_failed_groups", 0) or 0),
        "deduplicated_phase1_mentions": max(
            raw_phase1_mentions - len(phase1_extracted),
            0,
        ),
        "deduplicated_unresolved_mappings": max(
            len(mapping_resolved) - len(unique_mapping_keys),
            0,
        ),
        "phase1_completed_groups": int(
            phase_counts.get("phase1_completed_groups", 0) or 0
        ),
        "phase1_failed_groups": int(phase_counts.get("phase1_failed_groups", 0) or 0),
        "phase1_partial_failures": int(
            phase_counts.get("phase1_partial_failures", 0) or 0
        ),
        "phase1_requests": int(phase_request_counts.get("phase1_requests", 0) or 0),
        "phase2b_llm_requests": int(
            phase_request_counts.get("phase2b_llm_requests", 0) or 0
        ),
    }


def _project_assertion_for_dataset(
    *,
    dataset: str,
    source_dataset: str | None = None,
    assertion: str | None,
) -> str | None:
    dataset_projection = resolve_dataset_assertion_projection(dataset, source_dataset)
    raw_assertion = assertion.strip().casefold() if assertion else ""
    if dataset_projection is not None and raw_assertion == "other":
        return None
    normalized_assertion = normalize_benchmark_assertion(
        assertion, reject_unknown=False
    )
    if dataset_projection is None:
        return normalized_assertion
    return dataset_projection.get(normalized_assertion, DEFAULT_ID_ONLY_ASSERTION)


def _prediction_tuples_with_assertions(
    predicted_terms: list[dict[str, str | None]],
) -> list[tuple[str, str]]:
    tuples: list[tuple[str, str]] = []
    for term in predicted_terms:
        term_id = term.get("term_id")
        assertion = term.get("assertion")
        if not term_id or not assertion:
            continue
        tuples.append((term_id, assertion))
    return tuples


def _prediction_tuples_id_only(
    predicted_terms: list[dict[str, str | None]],
) -> list[tuple[str, str]]:
    tuples: list[tuple[str, str]] = []
    for term in predicted_terms:
        term_id = term.get("term_id")
        if not term_id:
            continue
        tuples.append((term_id, DEFAULT_ID_ONLY_ASSERTION))
    return tuples
