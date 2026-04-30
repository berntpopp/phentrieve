import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

import api.config as api_config
from api.dependencies import (
    get_dense_retriever_dependency,
    get_sbert_model_dependency,
)
from api.llm_quota import (
    DailyQuotaStore,
    QuotaExceededError,
    QuotaStatus,
    hash_subject_key,
    quota_reset_at_iso,
    resolve_subject_ip,
)
from api.research_use import (
    RESEARCH_USE_LIMITATION,
    require_research_use_acknowledgement,
    research_ack_openapi_parameter,
)
from api.schemas.text_processing_schemas import (
    AggregatedHPOTermAPI,
    HPOMatchInChunkAPI,
    ProcessedChunkAPI,
    TextAttributionSpanAPI,
    TextProcessingRequest,
    TextProcessingResponseAPI,
)
from phentrieve.config import (
    BENCHMARK_MODELS,
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_MODEL,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)
from phentrieve.llm.security_policy import resolve_public_llm_target
from phentrieve.retrieval.adaptive_rechunker import (
    adaptive_config_from_profile_block,
)
from phentrieve.text_processing.full_text_service import run_full_text_service
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.utils import detect_language
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/text", tags=["Text Processing and HPO Extraction"])
ALLOWED_TEXT_PROCESSING_MODELS = {DEFAULT_MODEL, *BENCHMARK_MODELS}


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


def _validate_model_name(field_name: str, model_name: str | None) -> str:
    """Validate a caller-supplied model name against the server allowlist."""
    if model_name is None:
        return DEFAULT_MODEL
    if model_name not in ALLOWED_TEXT_PROCESSING_MODELS:
        allowed_models = ", ".join(sorted(ALLOWED_TEXT_PROCESSING_MODELS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported {field_name}: {model_name}. "
                f"Allowed values: {allowed_models}."
            ),
        )
    return model_name


def _get_trust_remote_code_for_model(model_name: str) -> bool:
    """Return the server-owned trust policy for an allowed text-processing model."""
    normalized_name = model_name.lower()
    return "biolord" in normalized_name or "jina" in normalized_name


def _is_production_environment() -> bool:
    return api_config.PHENTRIEVE_ENV.strip().lower() == "production"


def _get_trusted_proxy_cidrs() -> list[str]:
    return [
        cidr.strip()
        for cidr in api_config.PHENTRIEVE_TRUSTED_PROXY_CIDRS.split(",")
        if cidr.strip()
    ]


def _get_llm_quota_store() -> DailyQuotaStore:
    return DailyQuotaStore(
        db_path=Path(api_config.PHENTRIEVE_LLM_QUOTA_DB_PATH),
        daily_limit=api_config.PHENTRIEVE_LLM_DAILY_LIMIT,
    )


def check_llm_quota_or_raise(http_request: Request) -> QuotaStatus:
    client_host = http_request.client.host if http_request.client else None
    subject_ip = resolve_subject_ip(
        client_host=client_host,
        x_forwarded_for=http_request.headers.get("x-forwarded-for"),
        trusted_proxy_cidrs=_get_trusted_proxy_cidrs(),
    )
    if subject_ip is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Unable to resolve a trusted anonymous subject for LLM quota "
                "enforcement. Verify proxy forwarding headers and "
                "PHENTRIEVE_TRUSTED_PROXY_CIDRS."
            ),
        )

    usage_date_utc = datetime.now(UTC).date().isoformat()
    subject_key = hash_subject_key(subject_ip)
    try:
        quota_status = _get_llm_quota_store().get_status(
            subject_key=subject_key,
            usage_date_utc=usage_date_utc,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Unable to evaluate LLM quota state. Verify "
                "PHENTRIEVE_LLM_QUOTA_DB_PATH and filesystem permissions."
            ),
        ) from exc

    if quota_status.quota_remaining <= 0:
        raise QuotaExceededError(
            quota_used=quota_status.quota_used,
            quota_limit=quota_status.quota_limit,
            quota_remaining=quota_status.quota_remaining,
            usage_date_utc=quota_status.usage_date_utc,
        )

    return quota_status


def _record_llm_quota_success(quota_status: QuotaStatus) -> QuotaStatus:
    return _get_llm_quota_store().record_success(
        subject_key=quota_status.subject_key,
        usage_date_utc=quota_status.usage_date_utc,
    )


def _get_chunking_config_for_api(
    request: TextProcessingRequest,
) -> list[dict[str, Any]]:
    """
    Get chunking configuration based on request strategy and parameters.

    This is an API-specific wrapper around the shared config resolver.

    Args:
        request: Text processing request with strategy and parameters

    Returns:
        Chunking pipeline configuration list
    """
    from phentrieve.text_processing.config_resolver import resolve_chunking_config

    # Pydantic schema ensures chunking_strategy always has a default value
    strategy_name = request.chunking_strategy.lower()

    # Extract parameters with defaults from config
    ws = (
        request.window_size
        if request.window_size is not None
        else DEFAULT_WINDOW_SIZE_TOKENS
    )
    ss = (
        request.step_size if request.step_size is not None else DEFAULT_STEP_SIZE_TOKENS
    )
    th = (
        request.split_threshold
        if request.split_threshold is not None
        else DEFAULT_SPLITTING_THRESHOLD
    )
    msl = (
        request.min_segment_length
        if request.min_segment_length is not None
        else DEFAULT_MIN_SEGMENT_LENGTH_WORDS
    )

    logger.debug(
        "API: Building config for '%s': ws=%s, ss=%s, th=%s, msl=%s",
        _sanitize(strategy_name),
        _sanitize(ws),
        _sanitize(ss),
        _sanitize(th),
        _sanitize(msl),
    )

    # Use shared resolver
    return resolve_chunking_config(
        strategy_name=strategy_name,
        config_file=None,  # API doesn't support config files
        window_size=ws,
        step_size=ss,
        threshold=th,
        min_segment_length=msl,
    )


def _validate_response_chunk_references(
    processed_chunks: list[ProcessedChunkAPI],
    aggregated_terms: list[AggregatedHPOTermAPI],
) -> None:
    """
    Validate chunk ID references in API response for internal consistency.

    This function checks invariants that must hold for a valid response:
    1. Chunk IDs are sequential and 1-based
    2. All source_chunk_ids reference existing chunks
    3. All text_attribution chunk_ids reference existing chunks
    4. top_evidence_chunk_id references an existing chunk (if present)

    This is called under __debug__ (Python assertions enabled) to catch
    bugs during development/testing without production overhead.

    Args:
        processed_chunks: List of processed chunks with chunk_id
        aggregated_terms: List of aggregated HPO terms with chunk references

    Raises:
        AssertionError: If any invariant is violated
    """
    total_chunks = len(processed_chunks)
    chunk_ids = {chunk.chunk_id for chunk in processed_chunks}

    # Invariant 1: Chunk IDs are sequential 1-based
    expected_ids = set(range(1, total_chunks + 1))
    assert chunk_ids == expected_ids, (
        f"Chunk IDs not sequential 1-based. Expected {expected_ids}, got {chunk_ids}"
    )

    # Invariant 2: All source_chunk_ids reference existing chunks
    for term in aggregated_terms:
        invalid_source_ids = set(term.source_chunk_ids) - chunk_ids
        assert not invalid_source_ids, (
            f"HPO term {term.id} has invalid source_chunk_ids: "
            f"{invalid_source_ids} (valid range: 1-{total_chunks})"
        )

    # Invariant 3: All text_attribution chunk_ids reference existing chunks
    for term in aggregated_terms:
        for attr in term.text_attributions:
            assert attr.chunk_id in chunk_ids, (
                f"HPO term {term.id} has text_attribution with invalid "
                f"chunk_id {attr.chunk_id} (valid range: 1-{total_chunks})"
            )

    # Invariant 4: top_evidence_chunk_id references existing chunk (if present)
    for term in aggregated_terms:
        if term.top_evidence_chunk_id is not None:
            assert term.top_evidence_chunk_id in chunk_ids, (
                f"HPO term {term.id} has invalid top_evidence_chunk_id "
                f"{term.top_evidence_chunk_id} (valid range: 1-{total_chunks})"
            )


async def _prepare_standard_request_context(
    request: TextProcessingRequest,
) -> dict[str, Any]:
    """Prepare standard-backend dependencies and config using the legacy API path."""
    actual_language = request.language
    if not actual_language or actual_language.lower() == "auto":
        try:
            actual_language = await run_in_threadpool(
                detect_language, request.text, default_lang=DEFAULT_LANGUAGE
            )
            logger.info("API: Auto-detected language: %s", _sanitize(actual_language))
        except Exception as lang_e:  # noqa: BLE001
            logger.warning(
                "API: Language detection failed: %s. Defaulting to %s.",
                _sanitize(lang_e),
                DEFAULT_LANGUAGE,
            )
            actual_language = DEFAULT_LANGUAGE

    retrieval_model_name_to_load = _validate_model_name(
        "retrieval_model_name", request.retrieval_model_name
    )
    sbert_for_chunking_name_to_load = _validate_model_name(
        "semantic_model_name",
        request.semantic_model_name
        if request.semantic_model_name is not None
        else retrieval_model_name_to_load,
    )

    logger.info(
        "API: Effective retrieval model: %s",
        _sanitize(retrieval_model_name_to_load),
    )
    logger.info(
        "API: Effective semantic model for chunking: %s",
        _sanitize(sbert_for_chunking_name_to_load),
    )

    retrieval_sbert_model = await get_sbert_model_dependency(
        model_name_requested=retrieval_model_name_to_load,
        trust_remote_code=_get_trust_remote_code_for_model(
            retrieval_model_name_to_load
        ),
    )

    if sbert_for_chunking_name_to_load != retrieval_model_name_to_load:
        logger.info(
            "API: Using separate semantic model for chunking: %s",
            _sanitize(sbert_for_chunking_name_to_load),
        )
        sbert_for_chunking = await get_sbert_model_dependency(
            model_name_requested=sbert_for_chunking_name_to_load,
            trust_remote_code=_get_trust_remote_code_for_model(
                sbert_for_chunking_name_to_load
            ),
        )
    else:
        sbert_for_chunking = retrieval_sbert_model

    retriever = await get_dense_retriever_dependency(
        sbert_model_name_for_retriever=retrieval_model_name_to_load
    )

    chunking_pipeline_cfg = _get_chunking_config_for_api(request)

    assertion_cfg = dict(DEFAULT_ASSERTION_CONFIG)
    assertion_cfg["disable"] = request.no_assertion_detection
    assertion_cfg["preference"] = request.assertion_preference
    assertion_cfg["language"] = actual_language

    logger.info("API: Using assertion configuration: %s", assertion_cfg)

    text_pipeline = TextProcessingPipeline(
        language=actual_language,
        chunking_pipeline_config=chunking_pipeline_cfg,
        assertion_config=assertion_cfg,
        sbert_model_for_semantic_chunking=sbert_for_chunking,
    )

    return {
        "actual_language": actual_language,
        "retrieval_model_name": retrieval_model_name_to_load,
        "chunking_pipeline_config": chunking_pipeline_cfg,
        "retriever": retriever,
        "text_pipeline": text_pipeline,
    }


def _adapt_shared_service_response_to_api(
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

    aggregated_terms: list[AggregatedHPOTermAPI] = []
    for term in _coerce_response_items(service_result.get("aggregated_hpo_terms")):
        if not isinstance(term, dict):
            continue

        aggregated_terms.append(
            AggregatedHPOTermAPI(
                hpo_id=str(term.get("hpo_id") or term.get("id") or ""),
                name=term.get("name", ""),
                confidence=term.get("confidence", 0.0),
                status=term.get("status", "unknown"),
                evidence_count=term.get("evidence_count", 0),
                source_chunk_ids=term.get("source_chunk_ids")
                or [chunk_idx + 1 for chunk_idx in term.get("chunks", [])],
                max_score_from_evidence=term.get(
                    "max_score_from_evidence", term.get("score")
                ),
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
        )

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
        }
    )

    if __debug__:
        _validate_response_chunk_references(
            response.processed_chunks, response.aggregated_hpo_terms
        )

    return response


@router.post(
    "/process",
    response_model=TextProcessingResponseAPI,
    operation_id="process_clinical_text",
    summary="Process research phenotype text to extract HPO terms",
    description=(
        f"{RESEARCH_USE_LIMITATION} Process text with chunking, assertion "
        "detection, and HPO term extraction. When LLM extraction is selected "
        "in production, "
        "clients can opt into automatic fallback to the standard backend by "
        "sending `X-Phentrieve-Allow-Standard-Fallback: true`."
    ),
    openapi_extra={
        "parameters": [
            {
                "name": "X-Phentrieve-Allow-Standard-Fallback",
                "in": "header",
                "required": False,
                "schema": {
                    "type": "string",
                    "enum": ["true"],
                },
                "description": (
                    "Optional opt-in for LLM requests in production. When set "
                    "to `true`, a quota-exhausted LLM request falls back to "
                    "the standard extraction backend instead of returning "
                    "`429 Too Many Requests`."
                ),
            },
            research_ack_openapi_parameter(),
        ]
    },
)
async def process_text_extract_hpo(
    http_request: Request,
    request: TextProcessingRequest,
):
    """
    Process research phenotype text to extract Human Phenotype Ontology (HPO) terms.

    This endpoint replicates the functionality of the `phentrieve text process` CLI command,
    accepting raw research phenotype text input along with various processing configurations.
    It returns processed text chunks with assertion statuses and aggregated HPO terms.

    Heavy NLP operations are executed asynchronously to prevent blocking the API server.

    Includes adaptive timeout based on text length to prevent frontend disconnects.
    """
    require_research_use_acknowledgement(http_request)

    logger.info(
        "API: Received request to process text. Language: %s, Strategy: %s",
        _sanitize(request.language),
        _sanitize(request.chunking_strategy),
    )

    # Calculate adaptive timeout based on text length
    text_length = len(request.text)
    if text_length < 500:
        timeout_seconds = 30
    elif text_length < 2000:
        timeout_seconds = 60
    elif text_length < 5000:
        timeout_seconds = 120
    else:
        timeout_seconds = 180

    logger.info(
        "API: Processing %s chars with %ss timeout", text_length, timeout_seconds
    )

    quota_status: QuotaStatus | None = None
    forced_standard_fallback: dict[str, Any] | None = None
    allow_standard_fallback = request.allow_standard_fallback or (
        http_request.headers.get("x-phentrieve-allow-standard-fallback", "").lower()
        == "true"
    )
    if request.extraction_backend == "llm" and _is_production_environment():
        try:
            quota_status = check_llm_quota_or_raise(http_request)
        except QuotaExceededError as exc:
            if allow_standard_fallback:
                request = request.model_copy(
                    update={
                        "extraction_backend": "standard",
                        "llm_mode": None,
                        "llm_internal_mode": None,
                    }
                )
                forced_standard_fallback = {
                    "fallback_reason": "llm_quota_exhausted",
                    "llm_quota_limit": exc.quota_limit,
                    "llm_quota_reset_at": quota_reset_at_iso(exc.usage_date_utc),
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=exc.to_detail(),
                ) from exc

    try:
        # Wrap processing with timeout protection
        response = await asyncio.wait_for(
            _process_text_via_shared_service(request), timeout=timeout_seconds
        )
        if quota_status is not None:
            try:
                updated_quota_status = _record_llm_quota_success(quota_status)
            except QuotaExceededError as exc:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=exc.to_detail(),
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "Unable to persist LLM quota usage. Verify "
                        "PHENTRIEVE_LLM_QUOTA_DB_PATH and filesystem permissions."
                    ),
                ) from exc
            response.meta["quota_limit"] = updated_quota_status.quota_limit
            response.meta["quota_remaining"] = updated_quota_status.quota_remaining
            response.meta["quota_reset_at"] = quota_reset_at_iso(
                updated_quota_status.usage_date_utc
            )
        if forced_standard_fallback is not None:
            response.meta.update(forced_standard_fallback)
        return response
    except asyncio.exceptions.TimeoutError:
        logger.error(
            "API: Request timed out after %ss (text length: %s chars)",
            timeout_seconds,
            text_length,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"Text processing timed out after {timeout_seconds} seconds. "
                f"Text length: {text_length} characters. "
                f"Suggestions: (1) reduce text length, or "
                f"(2) use 'simple' chunking strategy."
            ),
        )


async def _process_text_via_shared_service(request: TextProcessingRequest):
    """Process text through the shared full-text service for all requests."""
    standard_context: dict[str, Any] | None = None
    service_kwargs: dict[str, Any] = {
        "text": request.text,
        "extraction_backend": request.extraction_backend,
    }

    if request.extraction_backend == "standard":
        standard_context = await _prepare_standard_request_context(request)
        service_kwargs.update(
            {
                "language": standard_context["actual_language"],
                "chunking_pipeline_config": standard_context[
                    "chunking_pipeline_config"
                ],
                "assertion_config": {
                    **DEFAULT_ASSERTION_CONFIG,
                    "disable": request.no_assertion_detection,
                    "preference": request.assertion_preference,
                    "language": standard_context["actual_language"],
                },
                "retrieval_model_name": standard_context["retrieval_model_name"],
                "text_pipeline": standard_context["text_pipeline"],
                "retriever": standard_context["retriever"],
                "sbert_model_for_semantic_chunking": standard_context[
                    "text_pipeline"
                ].sbert_model,
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
                    request.include_details
                    if request.include_details is not None
                    else False
                ),
                "include_positions": request.include_chunk_positions,
            }
        )
        if request.adaptive_rechunking is not None:
            service_kwargs["adaptive_rechunking"] = adaptive_config_from_profile_block(
                block=request.adaptive_rechunking,
                yaml_block=None,
                cli_overrides=None,
            )
    else:
        target = resolve_public_llm_target()
        actual_language = request.language or DEFAULT_LANGUAGE
        retrieval_model_name_to_load = _validate_model_name(
            "retrieval_model_name", request.retrieval_model_name or DEFAULT_MODEL
        )
        service_kwargs.update(
            {
                "language": actual_language,
                "llm_provider": target.provider,
                "llm_model": target.model,
                "llm_base_url": target.base_url,
                "llm_mode": request.llm_mode or "two_phase",
                "llm_internal_mode": (
                    request.llm_internal_mode or "whole_document_grounded"
                ),
                "chunking_pipeline_config": _get_chunking_config_for_api(request),
                "assertion_config": {
                    **DEFAULT_ASSERTION_CONFIG,
                    "disable": request.no_assertion_detection,
                    "preference": request.assertion_preference,
                    "language": actual_language,
                },
                "retrieval_model_name": retrieval_model_name_to_load,
            }
        )

    service_result = await run_in_threadpool(run_full_text_service, **service_kwargs)
    return _adapt_shared_service_response_to_api(
        service_result,
        request=request,
        standard_context=standard_context,
    )


async def _process_text_internal(request: TextProcessingRequest):
    """Compatibility wrapper for the shared-service-based text processing path."""
    return await _process_text_via_shared_service(request)
