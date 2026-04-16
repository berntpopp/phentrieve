"""Lean LLM benchmark entrypoint for full-text extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from phentrieve.benchmark.data_loader import (
    DEFAULT_PHENOBERT_DATASET,
    LLM_ASSERTION_TO_BENCHMARK,
    load_benchmark_data,
    parse_gold_terms,
)
from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    ExtractionResult,
)
from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.provider import get_llm_provider
from phentrieve.llm.types import LLMPipelineConfig

DEFAULT_LLM_BENCHMARK_DATASET = DEFAULT_PHENOBERT_DATASET
DEFAULT_LLM_BENCHMARK_MODE = "two_phase"
DEFAULT_METRIC_AVERAGING = "micro"
DEFAULT_ID_ONLY_ASSERTION = "PRESENT"


def run_llm_benchmark(
    *,
    test_file: str,
    llm_model: str,
    llm_mode: str = DEFAULT_LLM_BENCHMARK_MODE,
    dataset: str = DEFAULT_LLM_BENCHMARK_DATASET,
) -> dict[str, Any]:
    """Run the LLM benchmark directly against the configured provider."""
    if llm_mode != DEFAULT_LLM_BENCHMARK_MODE:
        raise ValueError(
            f"Unsupported llm_mode: {llm_mode!r}. Expected {DEFAULT_LLM_BENCHMARK_MODE!r}."
        )

    test_path = Path(test_file)
    test_data = load_benchmark_data(test_path, dataset=dataset)
    documents = test_data["documents"]
    provider = get_llm_provider(llm_model=llm_model)
    pipeline = TwoPhaseLLMPipeline(provider=provider)
    config = LLMPipelineConfig(model=llm_model, mode=llm_mode)

    assertion_results: list[ExtractionResult] = []
    id_only_results: list[ExtractionResult] = []
    results: list[dict[str, Any]] = []

    for index, document in enumerate(documents, start=1):
        gold_terms = parse_gold_terms(document["gold_hpo_terms"])
        gold_ids_only = [
            (hpo_id, DEFAULT_ID_ONLY_ASSERTION) for hpo_id, _ in gold_terms
        ]
        pipeline_result = pipeline.run(text=document["text"], config=config)
        predicted_terms = [
            {
                "term_id": term.term_id,
                "label": term.label,
                "assertion": _normalize_assertion(term.assertion),
                "evidence": term.evidence,
            }
            for term in pipeline_result.terms
        ]
        predicted_with_assertions = sorted(
            {(term["term_id"], term["assertion"]) for term in predicted_terms}
        )
        predicted_ids_only = sorted(
            {(term["term_id"], DEFAULT_ID_ONLY_ASSERTION) for term in predicted_terms}
        )

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
                    {term["term_id"] for term in predicted_terms}
                ),
                "predicted_terms": predicted_terms,
            }
        )

    evaluator = CorpusExtractionMetrics(averaging=DEFAULT_METRIC_AVERAGING)
    assertion_metrics = evaluator.calculate_all_metrics(assertion_results)
    id_only_metrics = evaluator.calculate_all_metrics(id_only_results)

    return {
        "test_file": str(test_path),
        "dataset": dataset,
        "cases": len(documents),
        "llm_model": llm_model,
        "llm_mode": llm_mode,
        "metrics": {
            "assertion_aware": _serialize_corpus_metrics(assertion_metrics),
            "id_only": _serialize_corpus_metrics(id_only_metrics),
        },
        "dataset_metadata": test_data.get("metadata", {}),
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
