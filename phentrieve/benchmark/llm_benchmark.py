"""Lean LLM benchmark entrypoint for full-text extraction."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from phentrieve.benchmark.data_loader import (
    LLM_ASSERTION_TO_BENCHMARK,
    load_benchmark_data,
    parse_gold_terms,
)
from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    ExtractionResult,
)
from phentrieve.llm.config import DEFAULT_LLM_LANGUAGE
from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.prompts import loader as prompt_loader
from phentrieve.llm.provider import get_llm_provider
from phentrieve.llm.types import LLMPipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_LLM_BENCHMARK_DATASET = "GeneReviews"
DEFAULT_LLM_BENCHMARK_MODE = "two_phase"
DEFAULT_METRIC_AVERAGING = "micro"
DEFAULT_ID_ONLY_ASSERTION = "PRESENT"


def run_llm_benchmark(
    *,
    test_file: str,
    llm_model: str,
    llm_mode: str = DEFAULT_LLM_BENCHMARK_MODE,
    dataset: str = DEFAULT_LLM_BENCHMARK_DATASET,
    doc_ids: list[str] | None = None,
    language: str = DEFAULT_LLM_LANGUAGE,
    prompt_templates_dir: str | None = None,
    input_cost_per_1m_tokens: float | None = None,
    output_cost_per_1m_tokens: float | None = None,
) -> dict[str, Any]:
    """Run the LLM benchmark directly against the configured provider."""
    if llm_mode != DEFAULT_LLM_BENCHMARK_MODE:
        raise ValueError(
            f"Unsupported llm_mode: {llm_mode!r}. Expected {DEFAULT_LLM_BENCHMARK_MODE!r}."
        )

    test_path = Path(test_file)
    logger.info(
        "Starting LLM benchmark: test_file=%s model=%s mode=%s dataset=%s",
        test_path,
        llm_model,
        llm_mode,
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
    provider = get_llm_provider(llm_model=llm_model)
    logger.debug(
        "Initialized benchmark pipeline for model=%s mode=%s",
        llm_model,
        llm_mode,
    )

    assertion_results: list[ExtractionResult] = []
    id_only_results: list[ExtractionResult] = []
    results: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    benchmark_start_time = time.perf_counter()

    with _temporary_prompt_templates_dir(prompt_templates_dir):
        pipeline = TwoPhaseLLMPipeline(provider=provider)
        config = LLMPipelineConfig(
            model=llm_model,
            mode=llm_mode,
            language=language,
        )

        for index, document in enumerate(documents, start=1):
            logger.debug(
                "Processing benchmark document %s (%d chars)",
                document["id"],
                len(document.get("text", "")),
            )
            gold_terms = parse_gold_terms(document["gold_hpo_terms"])
            gold_ids_only = [
                (hpo_id, DEFAULT_ID_ONLY_ASSERTION) for hpo_id, _ in gold_terms
            ]
            doc_start_time = time.perf_counter()
            pipeline_result = pipeline.run(text=document["text"], config=config)
            doc_elapsed = time.perf_counter() - doc_start_time
            prompt_tokens = int(pipeline_result.meta.token_input or 0)
            completion_tokens = int(pipeline_result.meta.token_output or 0)
            total_tokens = prompt_tokens + completion_tokens
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            estimated_cost = _estimate_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_cost_per_1m_tokens=input_cost_per_1m_tokens,
                output_cost_per_1m_tokens=output_cost_per_1m_tokens,
            )

            predicted_terms = _serialize_predicted_terms(pipeline_result)
            predicted_with_assertions = sorted(
                _prediction_tuples_with_assertions(predicted_terms)
            )
            predicted_ids_only = sorted(_prediction_tuples_id_only(predicted_terms))

            assertion_results.append(
                ExtractionResult(
                    doc_id=str(document["id"]),
                    predicted=predicted_with_assertions,
                    gold=gold_terms,
                )
            )
            id_only_results.append(
                ExtractionResult(
                    doc_id=str(document["id"]),
                    predicted=predicted_ids_only,
                    gold=gold_ids_only,
                )
            )

            results.append(
                {
                    "case_index": index,
                    "doc_id": document["id"],
                    "source_dataset": document.get("source_dataset"),
                    "expected_hpo_ids": sorted({hpo_id for hpo_id, _ in gold_terms}),
                    "predicted_hpo_ids": sorted(
                        {term["term_id"] for term in predicted_terms if term["term_id"]}
                    ),
                    "predicted_terms": predicted_terms,
                    "timing_seconds": round(doc_elapsed, 6),
                    "token_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                    "estimated_cost": estimated_cost,
                }
            )
            prediction_records.append(
                _build_prediction_record(
                    document=document,
                    config=config,
                    pipeline_result=pipeline_result,
                    language=language,
                    processing_time_seconds=doc_elapsed,
                    prompt_templates_dir=prompt_templates_dir,
                    estimated_cost=estimated_cost,
                )
            )

    evaluator = CorpusExtractionMetrics(averaging=DEFAULT_METRIC_AVERAGING)
    assertion_metrics = evaluator.calculate_all_metrics(assertion_results)
    id_only_metrics = evaluator.calculate_all_metrics(id_only_results)
    logger.info("Calculated benchmark metrics for %d documents", len(results))
    total_token_usage = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "api_calls": len(documents),
    }
    total_estimated_cost = _estimate_cost(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        input_cost_per_1m_tokens=input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=output_cost_per_1m_tokens,
    )
    wall_clock_seconds = time.perf_counter() - benchmark_start_time

    return {
        "test_file": str(test_path),
        "dataset": dataset,
        "cases": len(documents),
        "llm_model": llm_model,
        "llm_mode": llm_mode,
        "language": language,
        "prompt_templates_dir": prompt_templates_dir,
        "metrics": {
            "assertion_aware": _serialize_corpus_metrics(assertion_metrics),
            "id_only": _serialize_corpus_metrics(id_only_metrics),
        },
        "dataset_metadata": test_data.get("metadata", {}),
        "token_usage": total_token_usage,
        "timing_breakdown": {
            "wall_clock_seconds": round(wall_clock_seconds, 6),
            "avg_seconds_per_case": round(wall_clock_seconds / len(documents), 6)
            if documents
            else 0.0,
        },
        "estimated_cost": total_estimated_cost,
        "prediction_records": prediction_records,
        "results": results,
    }


def _normalize_assertion(assertion: str | None) -> str:
    if assertion is None:
        return DEFAULT_ID_ONLY_ASSERTION
    return LLM_ASSERTION_TO_BENCHMARK.get(
        assertion.strip().lower(), DEFAULT_ID_ONLY_ASSERTION
    )


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
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_1m_tokens: float | None,
    output_cost_per_1m_tokens: float | None,
) -> dict[str, float] | None:
    if input_cost_per_1m_tokens is None or output_cost_per_1m_tokens is None:
        return None

    input_cost = prompt_tokens / 1_000_000 * input_cost_per_1m_tokens
    output_cost = completion_tokens / 1_000_000 * output_cost_per_1m_tokens
    return {
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(input_cost + output_cost, 8),
    }


def _build_prediction_record(
    *,
    document: dict[str, Any],
    config: LLMPipelineConfig,
    pipeline_result: Any,
    language: str,
    processing_time_seconds: float,
    prompt_templates_dir: str | None,
    estimated_cost: dict[str, float] | None,
) -> dict[str, Any]:
    prompt_tokens = int(pipeline_result.meta.token_input or 0)
    completion_tokens = int(pipeline_result.meta.token_output or 0)
    total_tokens = prompt_tokens + completion_tokens

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
        "metadata": {
            "model": config.model,
            "mode": config.mode,
            "prompt_version": pipeline_result.meta.prompt_version,
            "prompt_templates_dir": prompt_templates_dir,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "api_calls": 1,
            },
            "processing_time_seconds": round(processing_time_seconds, 6),
            "timing_breakdown": {
                "total_seconds": round(processing_time_seconds, 6),
                "llm_seconds": round(processing_time_seconds, 6),
                "tool_seconds": 0.0,
            },
            "estimated_cost": estimated_cost,
        },
    }


def _serialize_predicted_terms(
    pipeline_result: Any,
) -> list[dict[str, str | None]]:
    predicted_terms: list[dict[str, str | None]] = []
    for term in pipeline_result.terms:
        term_id = str(term.term_id).strip() if term.term_id is not None else ""
        predicted_terms.append(
            {
                "term_id": term_id,
                "label": str(term.label),
                "assertion": _normalize_assertion(term.assertion),
                "evidence": term.evidence,
            }
        )
    return predicted_terms


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
