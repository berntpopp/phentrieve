from typing import Any

from fastapi.concurrency import run_in_threadpool

from api.schemas.text_processing_schemas import (
    AggregatedHPOTermAPI,
    HPOMatchInChunkAPI,
    ProcessedChunkAPI,
    TextAttributionSpanAPI,
    TextProcessingRequest,
    TextProcessingResponseAPI,
)
from api.services.text_processing_context import (
    get_chunking_config_for_api,
    prepare_standard_text_processing_context,
    validate_model_name,
)
from phentrieve.assertion_vocab import is_excluded
from phentrieve.config import (
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MODEL,
)
from phentrieve.llm.security_policy import resolve_public_llm_target
from phentrieve.retrieval.adaptive_rechunker import (
    adaptive_config_from_profile_block,
)
from phentrieve.text_processing.full_text_service import run_full_text_service


def _coerce_response_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _coerce_text_attribution_chunk_id(attr: dict[str, Any]) -> int:
    chunk_id = attr.get("chunk_id")
    if isinstance(chunk_id, int):
        return chunk_id

    chunk_idx = attr.get("chunk_idx")
    if isinstance(chunk_idx, int):
        return chunk_idx + 1

    return 1


def _build_aggregated_hpo_term(term: dict[str, Any]) -> AggregatedHPOTermAPI:
    """Build an API term, carrying experiencer and deriving excluded (B2, F2).

    ``excluded`` prefers a precomputed value from the shared service (the LLM
    backend sets it) and otherwise derives it from the aggregated status via
    ``is_excluded`` so the deterministic backend -- which has no precomputed
    excluded -- still surfaces the actionable ruled-out flag.
    """
    return AggregatedHPOTermAPI(
        hpo_id=str(term.get("hpo_id") or term.get("id") or ""),
        name=term.get("name", ""),
        confidence=term.get("confidence", 0.0),
        status=term.get("status", "unknown"),
        experiencer=term.get("experiencer"),
        excluded=bool(term.get("excluded", is_excluded(term.get("status", "")))),
        evidence_count=term.get("evidence_count", 0),
        source_chunk_ids=term.get("source_chunk_ids")
        or [chunk_idx + 1 for chunk_idx in term.get("chunks", [])],
        max_score_from_evidence=term.get("max_score_from_evidence", term.get("score")),
        top_evidence_chunk_id=term.get("top_evidence_chunk_id"),
        text_attributions=[
            TextAttributionSpanAPI(
                chunk_id=_coerce_text_attribution_chunk_id(attr),
                start_char=attr.get("start_char", 0),
                end_char=attr.get("end_char", 0),
                matched_text_in_chunk=attr.get("matched_text_in_chunk", ""),
            )
            for attr in _coerce_response_items(term.get("text_attributions"))
            if isinstance(attr, dict)
        ],
        definition=term.get("definition"),
        synonyms=term.get("synonyms"),
        score=term.get("score"),
    )


def validate_response_chunk_references(
    processed_chunks: list[ProcessedChunkAPI],
    aggregated_terms: list[AggregatedHPOTermAPI],
) -> None:
    """
    Validate chunk ID references in API response for internal consistency.

    This is called under __debug__ to catch response adaptation regressions
    during development/testing without production overhead.
    """
    total_chunks = len(processed_chunks)
    chunk_ids = {chunk.chunk_id for chunk in processed_chunks}

    expected_ids = set(range(1, total_chunks + 1))
    assert chunk_ids == expected_ids, (
        f"Chunk IDs not sequential 1-based. Expected {expected_ids}, got {chunk_ids}"
    )

    for term in aggregated_terms:
        invalid_source_ids = set(term.source_chunk_ids) - chunk_ids
        assert not invalid_source_ids, (
            f"HPO term {term.id} has invalid source_chunk_ids: "
            f"{invalid_source_ids} (valid range: 1-{total_chunks})"
        )

    for term in aggregated_terms:
        for attr in term.text_attributions:
            assert attr.chunk_id in chunk_ids, (
                f"HPO term {term.id} has text_attribution with invalid "
                f"chunk_id {attr.chunk_id} (valid range: 1-{total_chunks})"
            )

    for term in aggregated_terms:
        if term.top_evidence_chunk_id is not None:
            assert term.top_evidence_chunk_id in chunk_ids, (
                f"HPO term {term.id} has invalid top_evidence_chunk_id "
                f"{term.top_evidence_chunk_id} (valid range: 1-{total_chunks})"
            )


def adapt_shared_service_response_to_api(
    service_result: dict[str, Any],
    *,
    request: TextProcessingRequest,
    standard_context: dict[str, Any] | None = None,
) -> TextProcessingResponseAPI:
    """Convert shared-service output into the API response contract."""
    service_meta = service_result.get("meta")
    meta: dict[str, Any] = dict(service_meta) if isinstance(service_meta, dict) else {}

    processed_chunks: list[ProcessedChunkAPI] = []
    for idx, chunk in enumerate(
        _coerce_response_items(service_result.get("processed_chunks"))
    ):
        if not isinstance(chunk, dict):
            continue

        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, int):
            chunk_id = idx + 1

        processed_chunks.append(
            ProcessedChunkAPI(
                chunk_id=chunk_id,
                text=chunk.get("text", ""),
                status=str(chunk.get("status", "unknown")),
                assertion_details=chunk.get("assertion_details"),
                hpo_matches=[
                    HPOMatchInChunkAPI(
                        hpo_id=str(match.get("hpo_id") or match.get("id") or ""),
                        name=match.get("name", ""),
                        score=match.get("score", 0.0),
                    )
                    for match in _coerce_response_items(chunk.get("hpo_matches"))
                    if isinstance(match, dict)
                ],
                start_char=chunk.get("start_char"),
                end_char=chunk.get("end_char"),
            )
        )

    aggregated_terms = [
        _build_aggregated_hpo_term(term)
        for term in _coerce_response_items(service_result.get("aggregated_hpo_terms"))
        if isinstance(term, dict)
    ]
    family_history_findings = [
        _build_aggregated_hpo_term(term)
        for term in _coerce_response_items(
            service_result.get("family_history_findings")
        )
        if isinstance(term, dict)
    ]

    if standard_context is not None:
        meta.update(
            {
                "request_parameters": request.model_dump(exclude_none=True),
                "effective_language": standard_context["actual_language"],
                "effective_chunking_strategy_config": standard_context[
                    "chunking_pipeline_config"
                ],
                "effective_retrieval_model": standard_context["retrieval_model_name"],
                "num_processed_chunks": len(processed_chunks),
                "num_aggregated_hpo_terms": len(aggregated_terms),
            }
        )
    else:
        meta.setdefault("num_processed_chunks", len(processed_chunks))
        meta.setdefault("num_aggregated_hpo_terms", len(aggregated_terms))

    response = TextProcessingResponseAPI.model_validate(
        {
            "meta": meta,
            "processed_chunks": processed_chunks,
            "aggregated_hpo_terms": aggregated_terms,
            "family_history_findings": family_history_findings,
        }
    )

    if __debug__:
        validate_response_chunk_references(
            response.processed_chunks, response.aggregated_hpo_terms
        )

    return response


async def execute_standard_text_processing(
    context: dict[str, Any],
    request: TextProcessingRequest,
) -> TextProcessingResponseAPI:
    """Execute the shared full-text service for the standard backend."""
    service_kwargs: dict[str, Any] = {
        "text": request.text,
        "extraction_backend": request.extraction_backend,
        "language": context["actual_language"],
        "chunking_pipeline_config": context["chunking_pipeline_config"],
        "assertion_config": {
            **DEFAULT_ASSERTION_CONFIG,
            "disable": request.no_assertion_detection,
            "preference": request.assertion_preference,
            "language": context["actual_language"],
        },
        "retrieval_model_name": context["retrieval_model_name"],
        "text_pipeline": context["text_pipeline"],
        "retriever": context["retriever"],
        "sbert_model_for_semantic_chunking": context["text_pipeline"].sbert_model,
        "chunk_retrieval_threshold": (
            request.chunk_retrieval_threshold
            if request.chunk_retrieval_threshold is not None
            else DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
        ),
        "num_results_per_chunk": (
            request.num_results_per_chunk
            if request.num_results_per_chunk is not None
            else 10
        ),
        "min_confidence_for_aggregated": (
            request.aggregated_term_confidence
            if request.aggregated_term_confidence is not None
            else DEFAULT_MIN_CONFIDENCE_AGGREGATED
        ),
        "top_term_per_chunk": (
            request.top_term_per_chunk_for_aggregation
            if request.top_term_per_chunk_for_aggregation is not None
            else False
        ),
        "include_details": (
            request.include_details if request.include_details is not None else False
        ),
        "include_positions": request.include_chunk_positions,
    }
    if request.adaptive_rechunking is not None:
        service_kwargs["adaptive_rechunking"] = adaptive_config_from_profile_block(
            block=request.adaptive_rechunking,
            yaml_block=None,
            cli_overrides=None,
        )

    service_result = await run_in_threadpool(run_full_text_service, **service_kwargs)
    return adapt_shared_service_response_to_api(
        service_result,
        request=request,
        standard_context=context,
    )


async def execute_llm_text_processing(
    request: TextProcessingRequest,
) -> TextProcessingResponseAPI:
    """Execute the shared full-text service for the public LLM backend."""
    target = resolve_public_llm_target()
    actual_language = request.language or DEFAULT_LANGUAGE
    retrieval_model_name_to_load = validate_model_name(
        "retrieval_model_name", request.retrieval_model_name or DEFAULT_MODEL
    )
    service_kwargs: dict[str, Any] = {
        "text": request.text,
        "extraction_backend": request.extraction_backend,
        "language": actual_language,
        "llm_provider": target.provider,
        "llm_model": target.model,
        "llm_base_url": target.base_url,
        "llm_mode": request.llm_mode or "two_phase",
        "llm_internal_mode": request.llm_internal_mode or "whole_document_grounded",
        "chunking_pipeline_config": get_chunking_config_for_api(request),
        "assertion_config": {
            **DEFAULT_ASSERTION_CONFIG,
            "disable": request.no_assertion_detection,
            "preference": request.assertion_preference,
            "language": actual_language,
        },
        "retrieval_model_name": retrieval_model_name_to_load,
    }

    service_result = await run_in_threadpool(run_full_text_service, **service_kwargs)
    return adapt_shared_service_response_to_api(service_result, request=request)


async def process_text_via_shared_service(
    request: TextProcessingRequest,
) -> TextProcessingResponseAPI:
    """Process text through the shared full-text service for all API requests."""
    if request.extraction_backend == "standard":
        context = await prepare_standard_text_processing_context(request)
        return await execute_standard_text_processing(context, request)

    return await execute_llm_text_processing(request)
