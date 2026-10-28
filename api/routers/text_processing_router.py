import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

import api.config as api_config
from api.llm_quota import (
    QuotaExceededError,
    QuotaStatus,
    QuotaStoreUnavailableError,
    anonymous_subject_key,
    hash_subject_key,
    quota_enforced,
    quota_reset_at_iso,
    read_llm_quota,
    release_llm_quota,
    reserve_llm_quota,
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
from api.services import text_processing_execution
from phentrieve.text_processing.full_text_service import LLMCancelToken
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/text", tags=["Text Processing and HPO Extraction"])


def _resolve_authenticated_subject(http_request: Request) -> tuple[str, bool] | None:
    """Return (subject_key, is_verified) for a logged-in user, else None.

    Only consulted when auth is enabled. A verified user is keyed on their id;
    an unverified user falls through to the anonymous (IP) tier so they get the
    lower limit until they verify their email.
    """
    if not api_config.PHENTRIEVE_AUTH_ENABLED:
        return None
    try:
        from api.auth.deps import get_optional_user
    except ImportError:
        return None
    user = get_optional_user(http_request)
    if user is None:
        return None
    return hash_subject_key(f"user:{user.id}"), user.is_verified


def _resolve_quota_subject(http_request: Request) -> tuple[str, int, bool, bool]:
    """Return (subject_key, daily_limit, authenticated, verified).

    Verified users get the authenticated tier; everyone else is keyed on a
    trusted client IP with the anonymous limit.
    """
    resolved = _resolve_authenticated_subject(http_request)
    if resolved is not None:
        subject_key, verified = resolved
        if verified:
            return (
                subject_key,
                api_config.PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT,
                True,
                True,
            )
        authenticated = True
    else:
        authenticated = False

    anonymous_key = anonymous_subject_key(http_request)
    if anonymous_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Unable to resolve a trusted anonymous subject for LLM quota "
                "enforcement. Verify proxy forwarding headers and "
                "PHENTRIEVE_TRUSTED_PROXY_CIDRS."
            ),
        )
    return (
        anonymous_key,
        api_config.PHENTRIEVE_LLM_DAILY_LIMIT,
        authenticated,
        False,
    )


def reserve_llm_quota_or_raise(http_request: Request) -> QuotaStatus:
    """Reserve one LLM call before running it so parallel requests cannot all pass."""
    subject_key, daily_limit, _authenticated, _verified = _resolve_quota_subject(
        http_request
    )
    try:
        return reserve_llm_quota(subject_key=subject_key, daily_limit=daily_limit)
    except QuotaStoreUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@router.get("/quota", summary="Get current LLM daily quota status")
def get_llm_quota_status(http_request: Request) -> dict[str, Any]:
    """Return the caller's current LLM quota for today without consuming it.

    Reflects the authenticated tier (10/day for verified users) or the
    anonymous IP tier. ``enforced`` indicates whether the quota is currently
    applied at all (see ``PHENTRIEVE_LLM_QUOTA_ENFORCE``).
    """
    subject_key, daily_limit, authenticated, verified = _resolve_quota_subject(
        http_request
    )
    try:
        quota_status = read_llm_quota(subject_key=subject_key, daily_limit=daily_limit)
    except QuotaStoreUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    detail = quota_status.to_detail()
    detail["quota_reset_at"] = quota_reset_at_iso(quota_status.usage_date_utc)
    detail["authenticated"] = authenticated
    detail["verified"] = verified
    detail["enforced"] = quota_enforced()
    return detail


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
                    "Optional opt-in for LLM requests. When set to `true`, an "
                    "LLM request that is quota-exhausted (instead of `429 Too "
                    "Many Requests`) or whose LLM backend fails falls back to "
                    "the standard extraction backend."
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
    if request.extraction_backend == "llm" and quota_enforced():
        try:
            quota_status = reserve_llm_quota_or_raise(http_request)
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

    # Set when the client disconnects (user cancelled) or we time out; the LLM
    # worker then skips its remaining paid provider calls.
    cancel_event = LLMCancelToken()
    disconnect_watcher = (
        asyncio.create_task(_cancel_on_disconnect(http_request, cancel_event))
        if request.extraction_backend == "llm"
        else None
    )
    try:
        # Wrap processing with timeout protection
        try:
            response = await asyncio.wait_for(
                text_processing_execution.process_text_via_shared_service(
                    request, cancel_event
                ),
                timeout=timeout_seconds,
            )
        except (HTTPException, asyncio.exceptions.TimeoutError):
            raise
        except Exception as exc:
            # Mirror the MCP adapter: an opted-in LLM request whose backend
            # fails reruns on the standard backend instead of returning a 500.
            # A cancelled request is never rerun: the client has gone away.
            if (
                request.extraction_backend != "llm"
                or not allow_standard_fallback
                or cancel_event.is_set()
            ):
                raise
            logger.warning(
                "API: LLM backend failed (%s); falling back to standard backend.",
                type(exc).__name__,
            )
            # Refund only if no paid LLM call started; after spend the
            # reservation is kept, as for a timeout or a cancel.
            if quota_status is not None and cancel_event.llm_calls_started == 0:
                release_llm_quota(quota_status)
            quota_status = None
            request = request.model_copy(
                update={
                    "extraction_backend": "standard",
                    "llm_mode": None,
                    "llm_internal_mode": None,
                }
            )
            forced_standard_fallback = {"fallback_reason": "llm_backend_error"}
            response = await asyncio.wait_for(
                text_processing_execution.process_text_via_shared_service(request),
                timeout=timeout_seconds,
            )
    except asyncio.exceptions.TimeoutError:
        # Keep the reservation: the in-flight paid LLM call still completes,
        # but no further calls start.
        cancel_event.set()
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
    except Exception:
        # Failed before producing a result: the caller got nothing, refund it.
        # A cancel after paid LLM calls started is charged, like a timeout, so
        # start-and-cancel loops cannot spend LLM budget for free.
        cancelled_after_llm_spend = (
            cancel_event.is_set() and cancel_event.llm_calls_started > 0
        )
        if quota_status is not None and not cancelled_after_llm_spend:
            release_llm_quota(quota_status)
        raise
    finally:
        if disconnect_watcher is not None:
            disconnect_watcher.cancel()

    if quota_status is not None:
        response.meta["quota_limit"] = quota_status.quota_limit
        response.meta["quota_remaining"] = quota_status.quota_remaining
        response.meta["quota_reset_at"] = quota_reset_at_iso(
            quota_status.usage_date_utc
        )
    if forced_standard_fallback is not None:
        response.meta.update(forced_standard_fallback)
    return response


async def _cancel_on_disconnect(
    http_request: Request, cancel_event: LLMCancelToken
) -> None:
    """Flag the run as cancelled once the client goes away (e.g. it hit Cancel)."""
    while not cancel_event.is_set():
        if await http_request.is_disconnected():
            logger.info("API: Client disconnected; cancelling remaining LLM calls")
            cancel_event.set()
            return
        await asyncio.sleep(0.5)
