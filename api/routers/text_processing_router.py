import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

import api.config as api_config
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
    TextProcessingRequest,
    TextProcessingResponseAPI,
)
from api.services.text_processing_context import (
    get_chunking_config_for_api as _get_chunking_config_for_api,
)
from api.services.text_processing_context import (
    get_trust_remote_code_for_model as _get_trust_remote_code_for_model,
)
from api.services.text_processing_context import (
    prepare_standard_text_processing_context as _prepare_standard_request_context,
)
from api.services.text_processing_context import (
    validate_model_name as _validate_model_name,
)
from api.services.text_processing_execution import (
    process_text_via_shared_service as _service_process_text_via_shared_service,
)
from api.services.text_processing_execution import (
    run_full_text_service,
)
from api.services.text_processing_execution import (
    validate_response_chunk_references as _validate_response_chunk_references,
)
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/text", tags=["Text Processing and HPO Extraction"])

__all__ = [
    "QuotaExceededError",
    "_get_chunking_config_for_api",
    "_get_trust_remote_code_for_model",
    "_prepare_standard_request_context",
    "_process_text_internal",
    "_process_text_via_shared_service",
    "_validate_model_name",
    "_validate_response_chunk_references",
    "run_full_text_service",
]


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
    """Compatibility wrapper for the shared-service-based text processing path."""
    return await _service_process_text_via_shared_service(request)


async def _process_text_internal(request: TextProcessingRequest):
    """Compatibility wrapper for the shared-service-based text processing path."""
    return await _process_text_via_shared_service(request)
