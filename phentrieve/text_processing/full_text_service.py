"""Shared full-text service and stable response adapter.

This module provides a thin service boundary that dispatches between standard
and LLM extraction backends while normalizing responses into a stable shape.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from phentrieve.config import (
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MODEL,
)
from phentrieve.llm.config import DEFAULT_LLM_MODE
from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.provider import get_llm_provider
from phentrieve.llm.types import LLMPipelineConfig

StableBackendResponse = dict[str, Any]
BackendCallable = Callable[..., StableBackendResponse | Mapping[str, Any] | None]
logger = logging.getLogger(__name__)


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _normalize_status(status: Any) -> str:
    if hasattr(status, "value"):
        return str(status.value)
    if status is None:
        return "unknown"
    return str(status)


def adapt_full_text_response(
    response: Mapping[str, Any] | None,
    *,
    extraction_backend: str,
) -> StableBackendResponse:
    """Normalize backend output into the stable full-text response shape."""
    meta: dict[str, Any] = {}
    processed_chunks: list[Any] = []
    aggregated_hpo_terms: list[Any] = []

    if response is not None:
        response_meta = response.get("meta")
        if isinstance(response_meta, Mapping):
            meta.update(response_meta)

        processed_chunks = _coerce_list(
            response.get("processed_chunks") or response.get("chunks")
        )
        aggregated_hpo_terms = _coerce_list(
            response.get("aggregated_hpo_terms") or response.get("aggregated_results")
        )

    meta["extraction_backend"] = extraction_backend
    return {
        "meta": meta,
        "processed_chunks": processed_chunks,
        "aggregated_hpo_terms": aggregated_hpo_terms,
    }


def _adapt_processed_chunks(
    processed_chunks: list[Mapping[str, Any]],
    chunk_results: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    chunk_result_by_idx = {
        chunk_result.get("chunk_idx"): chunk_result for chunk_result in chunk_results
    }

    adapted_chunks: list[dict[str, Any]] = []
    for idx, chunk in enumerate(processed_chunks):
        status = chunk.get("status")
        chunk_result = chunk_result_by_idx.get(idx)
        adapted_chunks.append(
            {
                "chunk_id": idx + 1,
                "text": chunk.get("text", ""),
                "status": _normalize_status(status),
                "assertion_details": chunk.get("assertion_details"),
                "hpo_matches": [
                    {
                        "id": match.get("id"),
                        "name": match.get("name"),
                        "score": match.get("score", 0.0),
                        "assertion_status": match.get("assertion_status"),
                    }
                    for match in _coerce_list(
                        chunk_result.get("matches") if chunk_result else None
                    )
                ],
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
            }
        )

    return adapted_chunks


def _adapt_aggregated_terms(
    aggregated_results: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    adapted_terms: list[dict[str, Any]] = []
    for term in aggregated_results:
        adapted_term = dict(term)
        source_chunk_ids = [chunk_idx + 1 for chunk_idx in term.get("chunks", [])]
        top_evidence_chunk_idx = term.get("top_evidence_chunk_idx")
        text_attributions: list[dict[str, Any]] = []
        for attribution in _coerce_list(term.get("text_attributions")):
            text_attributions.append(
                {
                    "chunk_id": attribution.get("chunk_idx", 0) + 1,
                    "start_char": attribution.get("start_char"),
                    "end_char": attribution.get("end_char"),
                    "matched_text_in_chunk": attribution.get("matched_text_in_chunk"),
                }
            )

        adapted_term.update(
            {
                "confidence": term.get("confidence", term.get("avg_score", 0.0)),
                "status": term.get("status", term.get("assertion_status", "unknown")),
                "evidence_count": term.get("evidence_count", term.get("count", 0)),
                "source_chunk_ids": source_chunk_ids,
                "max_score_from_evidence": term.get("score", 0.0),
                "top_evidence_chunk_id": (
                    top_evidence_chunk_idx + 1
                    if top_evidence_chunk_idx is not None
                    else None
                ),
                "text_attributions": text_attributions,
                "score": term.get("score", 0.0),
            }
        )
        adapted_terms.append(adapted_term)

    return adapted_terms


def adapt_standard_response(
    pipeline_result: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    extraction_result: tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    | Mapping[str, Any]
    | None,
) -> StableBackendResponse:
    """Convert pipeline and extraction outputs into the stable response shape."""
    if isinstance(pipeline_result, Mapping):
        processed_chunks = _coerce_list(
            pipeline_result.get("processed_chunks") or pipeline_result.get("chunks")
        )
    else:
        processed_chunks = _coerce_list(pipeline_result)

    if isinstance(extraction_result, Mapping):
        aggregated_results = _coerce_list(
            extraction_result.get("aggregated_hpo_terms")
            or extraction_result.get("aggregated_results")
        )
        chunk_results = _coerce_list(
            extraction_result.get("chunk_results")
            or extraction_result.get("detailed_chunk_results")
        )
    elif extraction_result is None:
        aggregated_results = []
        chunk_results = []
    else:
        aggregated_results = list(extraction_result[0])
        chunk_results = list(extraction_result[1])

    adapted_chunks = _adapt_processed_chunks(processed_chunks, chunk_results)
    adapted_terms = _adapt_aggregated_terms(aggregated_results)

    return adapt_full_text_response(
        {
            "meta": {
                "num_processed_chunks": len(adapted_chunks),
                "num_aggregated_hpo_terms": len(adapted_terms),
            },
            "processed_chunks": adapted_chunks,
            "aggregated_hpo_terms": adapted_terms,
        },
        extraction_backend="standard",
    )


def run_standard_backend(*, text: str, **kwargs: Any) -> StableBackendResponse:
    """Run the existing pipeline and extraction orchestrator for standard mode."""
    from phentrieve.embeddings import load_embedding_model
    from phentrieve.retrieval.dense_retriever import DenseRetriever
    from phentrieve.text_processing.hpo_extraction_orchestrator import (
        orchestrate_hpo_extraction,
    )
    from phentrieve.text_processing.pipeline import TextProcessingPipeline

    text_pipeline = kwargs.pop("text_pipeline", None)
    retriever = kwargs.pop("retriever", None)

    language = kwargs.pop("language", DEFAULT_LANGUAGE)
    chunking_pipeline_config = kwargs.pop("chunking_pipeline_config", None)
    assertion_config = kwargs.pop("assertion_config", None) or dict(
        DEFAULT_ASSERTION_CONFIG
    )
    include_positions = kwargs.pop("include_positions", False)
    retrieval_model_name = kwargs.pop("retrieval_model_name", DEFAULT_MODEL)
    sbert_model_for_semantic_chunking = kwargs.pop(
        "sbert_model_for_semantic_chunking", None
    )

    if text_pipeline is None:
        if chunking_pipeline_config is None:
            from phentrieve.config import get_default_chunk_pipeline_config

            chunking_pipeline_config = get_default_chunk_pipeline_config()
        if sbert_model_for_semantic_chunking is None:
            sbert_model_for_semantic_chunking = load_embedding_model(
                retrieval_model_name
            )

        text_pipeline = TextProcessingPipeline(
            language=language,
            chunking_pipeline_config=chunking_pipeline_config,
            assertion_config=assertion_config,
            sbert_model_for_semantic_chunking=sbert_model_for_semantic_chunking,
        )

    if retriever is None:
        if sbert_model_for_semantic_chunking is None:
            sbert_model_for_semantic_chunking = load_embedding_model(
                retrieval_model_name
            )
        retriever = DenseRetriever.from_model_name(
            model=sbert_model_for_semantic_chunking,
            model_name=retrieval_model_name,
        )
        if retriever is None:
            raise RuntimeError(
                f"Failed to initialize dense retriever for model '{retrieval_model_name}'"
            )

    processed_chunks = text_pipeline.process(text, include_positions=include_positions)
    if not processed_chunks:
        return adapt_standard_response([], ([], []))

    text_chunks = [chunk["text"] for chunk in processed_chunks]
    assertion_statuses: list[str | None] = [
        _normalize_status(chunk.get("status")) for chunk in processed_chunks
    ]

    aggregated_results, chunk_results = orchestrate_hpo_extraction(
        text_chunks=text_chunks,
        retriever=retriever,
        num_results_per_chunk=kwargs.pop("num_results_per_chunk", 10),
        chunk_retrieval_threshold=kwargs.pop(
            "chunk_retrieval_threshold", DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
        ),
        cross_encoder=kwargs.pop("cross_encoder", None),
        language=language,
        top_term_per_chunk=kwargs.pop("top_term_per_chunk", False),
        min_confidence_for_aggregated=kwargs.pop(
            "min_confidence_for_aggregated", DEFAULT_MIN_CONFIDENCE_AGGREGATED
        ),
        assertion_statuses=assertion_statuses,
        include_details=kwargs.pop("include_details", False),
    )

    return adapt_standard_response(
        processed_chunks, (aggregated_results, chunk_results)
    )


def run_llm_backend(*, text: str, **kwargs: Any) -> StableBackendResponse:
    """Run the shared LLM backend through the full-text service boundary."""
    llm_model = kwargs.get("llm_model") or os.getenv("PHENTRIEVE_LLM_MODEL")
    if not llm_model:
        raise RuntimeError(
            "No LLM model configured. Provide --llm-model or set PHENTRIEVE_LLM_MODEL."
        )

    llm_mode = (kwargs.get("llm_mode") or "two_phase").strip()
    if llm_mode != DEFAULT_LLM_MODE:
        raise ValueError(
            f"Unsupported LLM mode: {llm_mode!r}. Expected {DEFAULT_LLM_MODE!r}."
        )

    provider_factory = kwargs.get("provider_factory") or get_llm_provider
    pipeline_factory = kwargs.get("pipeline_factory") or TwoPhaseLLMPipeline

    logger.info("Running LLM backend: model=%s, mode=%s", llm_model, llm_mode)
    logger.debug(
        "LLM backend request context: language=%s, text_chars=%d",
        kwargs.get("language") or "en",
        len(text),
    )

    provider = provider_factory(llm_model=llm_model)
    pipeline = pipeline_factory(provider=provider)
    result = pipeline.run(
        text=text,
        config=LLMPipelineConfig(
            model=llm_model,
            mode=llm_mode,
            language=kwargs.get("language"),
        ),
    )

    result_payload = {
        "meta": {
            "extraction_backend": "llm",
            "llm_model": result.meta.llm_model,
            "llm_mode": result.meta.llm_mode,
            "prompt_version": result.meta.prompt_version,
            "num_aggregated_hpo_terms": len(result.terms),
            "token_input": result.meta.token_input,
            "token_output": result.meta.token_output,
        },
        "processed_chunks": [],
        "aggregated_hpo_terms": [
            {
                "id": term.term_id,
                "name": term.label,
                "evidence": term.evidence,
                "status": term.assertion,
            }
            for term in result.terms
        ],
    }

    logger.info(
        "LLM backend completed: model=%s, mode=%s, terms=%d",
        result.meta.llm_model,
        result.meta.llm_mode,
        len(result.terms),
    )
    return result_payload


class FullTextService:
    """Dispatch between full-text extraction backends and normalize responses."""

    def __init__(
        self,
        *,
        standard_backend: BackendCallable,
        llm_backend: BackendCallable,
    ) -> None:
        self._standard_backend = standard_backend
        self._llm_backend = llm_backend

    def process(
        self,
        *,
        text: str,
        extraction_backend: str,
        **kwargs: Any,
    ) -> StableBackendResponse:
        backend_name = extraction_backend.strip().lower()
        if backend_name not in {"standard", "llm"}:
            raise ValueError(
                f"Unsupported extraction backend: {extraction_backend!r}. "
                "Expected one of: standard, llm."
            )
        backend = self._llm_backend if backend_name == "llm" else self._standard_backend
        response = backend(text=text, **kwargs)
        return adapt_full_text_response(response, extraction_backend=backend_name)


_DEFAULT_FULL_TEXT_SERVICE = FullTextService(
    standard_backend=run_standard_backend,
    llm_backend=run_llm_backend,
)


def run_full_text_service(
    *, text: str, extraction_backend: str, **kwargs: Any
) -> StableBackendResponse:
    """Run the shared full-text service boundary used by CLI and API paths."""
    return _DEFAULT_FULL_TEXT_SERVICE.process(
        text=text,
        extraction_backend=extraction_backend,
        **kwargs,
    )
