"""Shared full-text service and stable response adapter.

This module provides a thin service boundary that dispatches between standard
and LLM extraction backends while normalizing responses into a stable shape.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from phentrieve.config import (
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MODEL,
)
from phentrieve.llm.config import (
    DEFAULT_LLM_MODE,
    DEFAULT_LLM_MODEL,
    DEFAULT_PROVIDER_NAME,
)
from phentrieve.llm.pipeline import TwoPhaseLLMPipeline, _render_phase1_user_prompt
from phentrieve.llm.preprocessing import (
    build_extraction_groups,
    build_grounded_chunks_from_text_pipeline,
)
from phentrieve.llm.prompts.loader import get_prompt
from phentrieve.llm.provider import get_llm_provider
from phentrieve.llm.types import AnnotationMode, LLMPipelineConfig

StableBackendResponse = dict[str, Any]
BackendCallable = Callable[..., StableBackendResponse | Mapping[str, Any] | None]
logger = logging.getLogger(__name__)
MAX_GROUNDED_PHASE1_INPUT_TOKENS = 30000


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


@dataclass(frozen=True)
class _PreprocessedGroundedDocument:
    grounded_chunks: list[dict[str, Any]]
    extraction_groups: list[dict[str, Any]]


def preprocess_grounded_document(
    *,
    text: str,
    language: str,
    provider: Any,
    extraction_prompt: Any,
    chunking_pipeline_config: list[dict[str, Any]] | None,
    assertion_config: dict[str, Any] | None,
    retrieval_model_name: str,
    include_positions: bool = True,
) -> _PreprocessedGroundedDocument:
    grounded_chunks = build_grounded_chunks_from_text_pipeline(
        text=text,
        language=language,
        chunking_pipeline_config=chunking_pipeline_config,
        assertion_config=assertion_config,
        retrieval_model_name=retrieval_model_name,
        include_positions=include_positions,
    )
    extraction_groups: list[dict[str, Any]] = []
    token_count_fn = getattr(provider, "count_tokens", None)
    if callable(token_count_fn):
        try:
            grounded_chunk_payload = [asdict(chunk) for chunk in grounded_chunks]
            user_prompt = _render_phase1_user_prompt(
                extraction_prompt=extraction_prompt,
                text=text,
                grounded_chunks=grounded_chunk_payload,
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
            if total_tokens > MAX_GROUNDED_PHASE1_INPUT_TOKENS:
                extraction_groups = [
                    asdict(group)
                    for group in build_extraction_groups(
                        grounded_chunks=grounded_chunks,
                        provider=provider,
                        system_prompt=extraction_prompt.render_system_prompt(),
                        max_prompt_tokens=MAX_GROUNDED_PHASE1_INPUT_TOKENS,
                    )
                ]
        except (NotImplementedError, TypeError):
            extraction_groups = []
    return _PreprocessedGroundedDocument(
        grounded_chunks=[asdict(chunk) for chunk in grounded_chunks],
        extraction_groups=extraction_groups,
    )


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


def _coerce_chunk_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _score_evidence_record(record: Mapping[str, Any]) -> float | None:
    return max(
        (
            score
            for score in (
                _coerce_float(record.get("score")),
                _coerce_float(record.get("confidence")),
                _coerce_float(record.get("retrieval_score")),
            )
            if score is not None
        ),
        default=None,
    )


def _infer_text_attribution_offsets(
    *,
    chunk_text: str,
    matched_text: str | None,
    start_char: Any,
    end_char: Any,
) -> tuple[int | None, int | None]:
    if isinstance(start_char, int) and isinstance(end_char, int):
        return start_char, end_char
    if not matched_text:
        return None, None

    start_idx = chunk_text.lower().find(matched_text.lower())
    if start_idx < 0:
        return None, None
    return start_idx, start_idx + len(matched_text)


def _adapt_llm_text_attributions(
    evidence_records: list[Mapping[str, Any]],
    *,
    chunk_text_by_id: Mapping[int, str],
) -> list[dict[str, Any]]:
    text_attributions: list[dict[str, Any]] = []
    seen: set[tuple[int, int | None, int | None, str | None]] = set()

    for record in evidence_records:
        chunk_ids = [
            _coerce_chunk_id(chunk_id) for chunk_id in record.get("chunk_ids", [])
        ]
        matched_text = record.get("evidence_text") or record.get("phrase")
        for chunk_id in chunk_ids:
            if chunk_id is None:
                continue
            start_char, end_char = _infer_text_attribution_offsets(
                chunk_text=chunk_text_by_id.get(chunk_id, ""),
                matched_text=matched_text,
                start_char=record.get("start_char"),
                end_char=record.get("end_char"),
            )
            if start_char is None or end_char is None:
                continue
            key = (
                chunk_id,
                start_char,
                end_char,
                matched_text,
            )
            if key in seen:
                continue
            seen.add(key)
            text_attributions.append(
                {
                    "chunk_id": chunk_id,
                    "start_char": start_char,
                    "end_char": end_char,
                    "matched_text_in_chunk": matched_text,
                }
            )

    return text_attributions


def _adapt_llm_aggregated_terms(
    terms: Sequence[Any],
    *,
    grounded_chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    adapted_terms: list[dict[str, Any]] = []
    chunk_text_by_id = {
        chunk_id: str(chunk.get("text", ""))
        for chunk in grounded_chunks
        if (chunk_id := _coerce_chunk_id(chunk.get("chunk_id"))) is not None
    }

    for term in terms:
        evidence_records = [
            record.model_dump() if hasattr(record, "model_dump") else dict(record)
            for record in getattr(term, "evidence_records", [])
        ]
        evidence_scores = [
            score
            for record in evidence_records
            for score in (
                _coerce_float(record.get("score")),
                _coerce_float(record.get("confidence")),
                _coerce_float(record.get("retrieval_score")),
            )
            if score is not None
        ]
        term_score = _coerce_float(getattr(term, "score", None))
        term_confidence = _coerce_float(getattr(term, "confidence", None))
        reranker_score = _coerce_float(getattr(term, "reranker_score", None))
        max_evidence_score = (
            max(
                [*evidence_scores, term_score]
                if term_score is not None
                else evidence_scores
            )
            if evidence_scores or term_score is not None
            else 0.0
        )
        source_chunk_ids = sorted(
            {
                chunk_id
                for record in evidence_records
                for chunk_id in (
                    _coerce_chunk_id(value) for value in record.get("chunk_ids", [])
                )
                if chunk_id is not None
            }
        )
        text_attributions = _adapt_llm_text_attributions(
            evidence_records,
            chunk_text_by_id=chunk_text_by_id,
        )
        top_evidence_chunk_id = None
        top_evidence_score = None
        for record in evidence_records:
            record_score = _score_evidence_record(record)
            if record_score is None:
                continue
            for chunk_id in (
                _coerce_chunk_id(value) for value in record.get("chunk_ids", [])
            ):
                if chunk_id is None:
                    continue
                if top_evidence_score is None or record_score > top_evidence_score:
                    top_evidence_score = record_score
                    top_evidence_chunk_id = chunk_id
                break

        adapted_terms.append(
            {
                "id": term.term_id,
                "name": term.label,
                "evidence": term.evidence,
                "status": term.assertion,
                "evidence_records": evidence_records,
                "confidence": (
                    term_confidence
                    if term_confidence is not None
                    else (term_score if term_score is not None else max_evidence_score)
                ),
                "evidence_count": len(evidence_records),
                "source_chunk_ids": source_chunk_ids,
                "max_score_from_evidence": max_evidence_score,
                "top_evidence_chunk_id": top_evidence_chunk_id
                if top_evidence_chunk_id is not None
                else (source_chunk_ids[0] if source_chunk_ids else None),
                "text_attributions": text_attributions,
                "score": term_score if term_score is not None else max_evidence_score,
            }
        )
        if reranker_score is not None:
            adapted_terms[-1]["reranker_score"] = reranker_score

    return adapted_terms


def _adapt_llm_processed_chunks(
    grounded_chunks: Sequence[Mapping[str, Any]],
    adapted_terms: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    matches_by_chunk_id: dict[int, list[dict[str, Any]]] = {
        chunk_id: []
        for chunk in grounded_chunks
        if (chunk_id := _coerce_chunk_id(chunk.get("chunk_id"))) is not None
    }

    for term in adapted_terms:
        for chunk_id in term.get("source_chunk_ids", []):
            if chunk_id not in matches_by_chunk_id:
                continue
            matches_by_chunk_id[chunk_id].append(
                {
                    "id": term.get("id"),
                    "name": term.get("name"),
                    "score": term.get("score", 0.0),
                    "assertion_status": term.get("status", "unknown"),
                }
            )

    adapted_chunks: list[dict[str, Any]] = []
    for chunk in grounded_chunks:
        chunk_id = _coerce_chunk_id(chunk.get("chunk_id"))
        if chunk_id is None:
            continue
        adapted_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk.get("text", ""),
                "status": _normalize_status(chunk.get("status")),
                "assertion_details": chunk.get("assertion_details"),
                "hpo_matches": matches_by_chunk_id.get(chunk_id, []),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
            }
        )

    return adapted_chunks


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
    llm_provider = kwargs.get("llm_provider") or os.getenv("PHENTRIEVE_LLM_PROVIDER")
    llm_model = (
        kwargs.get("llm_model")
        or os.getenv("PHENTRIEVE_LLM_MODEL")
        or DEFAULT_LLM_MODEL
    )
    llm_base_url = kwargs.get("llm_base_url") or os.getenv("PHENTRIEVE_LLM_BASE_URL")

    llm_mode = (kwargs.get("llm_mode") or "two_phase").strip()
    if llm_mode != DEFAULT_LLM_MODE:
        raise ValueError(
            f"Unsupported LLM mode: {llm_mode!r}. Expected {DEFAULT_LLM_MODE!r}."
        )
    llm_internal_mode = kwargs.get("llm_internal_mode", "whole_document_grounded")
    if llm_internal_mode not in {
        "whole_document_legacy",
        "whole_document_grounded",
    }:
        raise ValueError(
            "Unsupported llm_internal_mode: "
            f"{llm_internal_mode!r}. Expected 'whole_document_legacy' or "
            "'whole_document_grounded'."
        )

    provider_factory = kwargs.get("provider_factory") or get_llm_provider
    pipeline_factory = kwargs.get("pipeline_factory") or TwoPhaseLLMPipeline

    logger.info("Running LLM backend: model=%s, mode=%s", llm_model, llm_mode)
    logger.debug(
        "LLM backend request context: language=%s, text_chars=%d",
        kwargs.get("language") or "en",
        len(text),
    )

    provider_factory_kwargs = _build_provider_factory_kwargs(
        provider_factory,
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
    )
    provider = provider_factory(**provider_factory_kwargs)
    extraction_prompt = get_prompt(
        AnnotationMode.TWO_PHASE,
        kwargs.get("language") or DEFAULT_LANGUAGE,
    )
    grounded_chunks: list[dict[str, Any]] = []
    extraction_groups: list[dict[str, Any]] = []
    active_extraction_groups: list[dict[str, Any]] = []
    if llm_internal_mode == "whole_document_grounded":
        preprocessed = preprocess_grounded_document(
            text=text,
            language=kwargs.get("language") or DEFAULT_LANGUAGE,
            provider=provider,
            extraction_prompt=extraction_prompt,
            chunking_pipeline_config=kwargs.get("chunking_pipeline_config"),
            assertion_config={"disable": True},
            retrieval_model_name=kwargs.get("retrieval_model_name", DEFAULT_MODEL),
        )
        grounded_chunks = preprocessed.grounded_chunks
        extraction_groups = preprocessed.extraction_groups
        active_extraction_groups = (
            extraction_groups if len(extraction_groups) > 1 else []
        )
        logger.info(
            "LLM grounding summary: language=%s chunks=%d groups=%d mode=%s",
            kwargs.get("language") or DEFAULT_LANGUAGE,
            len(grounded_chunks),
            len(extraction_groups),
            llm_internal_mode,
        )
        try:
            if active_extraction_groups:
                grouped_prompt_tokens_total = sum(
                    int(group.get("estimated_prompt_tokens", 0) or 0)
                    for group in active_extraction_groups
                )
                grouped_prompt_tokens_max = max(
                    (
                        int(group.get("estimated_prompt_tokens", 0) or 0)
                        for group in active_extraction_groups
                    ),
                    default=0,
                )
                grouped_prompt_chars_total = sum(
                    len(str(group.get("text", "")))
                    for group in active_extraction_groups
                )
                logger.debug(
                    "LLM phase1 preflight: grouped_prompt_tokens_max=%s grouped_prompt_tokens_total=%s grouped_prompt_chars_total=%s text_chars=%d grounded_chunks=%d extraction_groups=%d",
                    grouped_prompt_tokens_max,
                    grouped_prompt_tokens_total,
                    grouped_prompt_chars_total,
                    len(text),
                    len(grounded_chunks),
                    len(active_extraction_groups),
                )
                if grouped_prompt_tokens_max > MAX_GROUNDED_PHASE1_INPUT_TOKENS:
                    logger.warning(
                        "LLM phase1 grouped prompt exceeds configured token budget"
                    )
            else:
                user_prompt = _render_phase1_user_prompt(
                    extraction_prompt=extraction_prompt,
                    text=text,
                    grounded_chunks=grounded_chunks,
                )
                if not hasattr(provider, "count_tokens"):
                    raise NotImplementedError
                token_counts = provider.count_tokens(
                    system_prompt=extraction_prompt.render_system_prompt(),
                    user_prompt=user_prompt,
                )
                logger.debug(
                    "LLM phase1 preflight: prompt_tokens=%s total_tokens=%s text_chars=%d grounded_chunks=%d extraction_groups=%d user_prompt_chars=%d",
                    token_counts.get("prompt_tokens"),
                    token_counts.get("total_tokens"),
                    len(text),
                    len(grounded_chunks),
                    len(active_extraction_groups),
                    len(user_prompt),
                )
                if (
                    isinstance(token_counts, dict)
                    and int(token_counts.get("total_tokens", 0) or 0)
                    > MAX_GROUNDED_PHASE1_INPUT_TOKENS
                ):
                    logger.warning("LLM phase1 prompt exceeds configured token budget")
        except (AttributeError, NotImplementedError):
            logger.debug("Provider does not support token preflight counting")

    pipeline = pipeline_factory(provider=provider)
    resolved_provider_name = getattr(provider, "provider_name", None)
    if not isinstance(resolved_provider_name, str) or not resolved_provider_name:
        resolved_provider_name = llm_provider or DEFAULT_PROVIDER_NAME

    resolved_model_name = getattr(provider, "model_name", None)
    if not isinstance(resolved_model_name, str) or not resolved_model_name:
        resolved_model_name = llm_model
    resolved_base_url = getattr(provider, "base_url", None)
    if not isinstance(resolved_base_url, str) or not resolved_base_url:
        resolved_base_url = llm_base_url

    pipeline_kwargs: dict[str, Any] = {
        "text": text,
        "grounded_chunks": grounded_chunks,
        "config": LLMPipelineConfig(
            provider=resolved_provider_name,
            model=resolved_model_name,
            base_url=resolved_base_url,
            mode=llm_mode,
            language=kwargs.get("language"),
        ),
    }
    if llm_internal_mode == "whole_document_grounded" and active_extraction_groups:
        pipeline_kwargs["extraction_groups"] = active_extraction_groups
    result = pipeline.run(**pipeline_kwargs)
    phase_counts = dict(result.meta.phase_counts)
    phase_request_counts = dict(result.meta.phase_request_counts)
    trace = dict(result.meta.trace)

    adapted_terms = _adapt_llm_aggregated_terms(
        result.terms,
        grounded_chunks=grounded_chunks,
    )
    adapted_chunks = _adapt_llm_processed_chunks(grounded_chunks, adapted_terms)

    result_payload = {
        "meta": {
            "extraction_backend": "llm",
            "llm_provider": result.meta.llm_provider,
            "llm_model": result.meta.llm_model,
            "llm_mode": result.meta.llm_mode,
            "llm_internal_mode": llm_internal_mode,
            "prompt_version": result.meta.prompt_version,
            "num_processed_chunks": len(adapted_chunks),
            "num_aggregated_hpo_terms": len(adapted_terms),
            "token_input": result.meta.token_input,
            "token_output": result.meta.token_output,
            "observability": {
                "request_count": int(result.meta.request_count or 0),
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
                "grounded_chunks": len(grounded_chunks),
                "extraction_groups": len(active_extraction_groups),
                "failed_groups": int(phase_counts.get("phase1_failed_groups", 0) or 0),
                "deduplicated_phase1_mentions": _count_deduplicated_phase1_mentions(
                    trace
                ),
                "deduplicated_unresolved_mappings": _count_deduplicated_mappings(trace),
                "phase1_completed_groups": int(
                    phase_counts.get("phase1_completed_groups", 0) or 0
                ),
                "phase1_failed_groups": int(
                    phase_counts.get("phase1_failed_groups", 0) or 0
                ),
                "phase1_partial_failures": int(
                    phase_counts.get("phase1_partial_failures", 0) or 0
                ),
                "phase1_requests": int(
                    phase_request_counts.get("phase1_requests", 0) or 0
                ),
                "phase2b_llm_requests": int(
                    phase_request_counts.get("phase2b_llm_requests", 0) or 0
                ),
            },
        },
        "processed_chunks": adapted_chunks,
        "aggregated_hpo_terms": adapted_terms,
    }

    logger.info(
        "LLM backend completed: model=%s, mode=%s, terms=%d",
        result.meta.llm_model,
        result.meta.llm_mode,
        len(result.terms),
    )
    return result_payload


def _count_deduplicated_phase1_mentions(trace: dict[str, Any]) -> int:
    phase1_trace = trace.get("phase1")
    if not isinstance(phase1_trace, dict):
        return 0
    phase1_groups = phase1_trace.get("groups", [])
    phase1_extracted = phase1_trace.get("extracted", [])
    if not isinstance(phase1_groups, list) or not isinstance(phase1_extracted, list):
        return 0
    raw_mentions = sum(
        int(group.get("extracted_count", 0) or 0)
        for group in phase1_groups
        if isinstance(group, dict)
    )
    return max(raw_mentions - len(phase1_extracted), 0)


def _count_deduplicated_mappings(trace: dict[str, Any]) -> int:
    phase2b_llm_trace = trace.get("phase2b_llm")
    if not isinstance(phase2b_llm_trace, dict):
        return 0
    resolved = phase2b_llm_trace.get("resolved", [])
    if not isinstance(resolved, list):
        return 0
    unique_keys = {
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
        for item in resolved
        if isinstance(item, dict)
    }
    return max(len(resolved) - len(unique_keys), 0)


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
