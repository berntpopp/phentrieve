"""MCP envelope boundary: Family B success/_meta injection and structured errors.

Tool bodies return a plain dict and may raise :class:`McpToolError`;
:func:`run_mcp_tool` injects ``success`` and a complete ``_meta`` block on
success, and converts any exception into a structured error dict (returned,
never raised) so the LLM client sees a typed, recoverable failure rather than an
opaque masked message.

Per-call ``_meta`` carries: ``tool``, ``request_id``, ``elapsed_ms``,
``response_mode`` (when applicable), ``capabilities_version``,
``unsafe_for_clinical_use``, and ``next_commands``. ``recommended_citation`` is
added by tool bodies at standard/full verbosity.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from api.llm_quota import QuotaExceededError

logger = logging.getLogger("phentrieve.mcp")

ERROR_CODES: set[str] = {
    "invalid_input",
    "validation_failed",
    "not_found",
    "ambiguous_query",
    "llm_quota_exhausted",
    "llm_unavailable",
    "upstream_unavailable",
    "temporarily_unavailable",
    "internal_error",
}

_RETRYABLE = {
    "llm_quota_exhausted",
    "llm_unavailable",
    "upstream_unavailable",
    "temporarily_unavailable",
}
_RECOVERY = {
    "invalid_input": "reformulate_input",
    "validation_failed": "reformulate_input",
    "not_found": "reformulate_input",
    "ambiguous_query": "reformulate_input",
    "llm_quota_exhausted": "retry_backoff",
    "llm_unavailable": "switch_tool",
    "upstream_unavailable": "retry_backoff",
    "temporarily_unavailable": "retry_backoff",
    "internal_error": "retry_backoff",
}

_RECENT_ERRORS: deque[dict[str, Any]] = deque(maxlen=50)
_SAFE_INTERNAL_MESSAGE = "An internal error occurred. The request was not completed."


@dataclass
class McpErrorContext:
    """Per-call context so envelopes can name the failing tool and recovery."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class McpToolError(Exception):
    """Raised inside a tool body to emit a specific error code/message."""

    def __init__(
        self, error_code: str, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        if error_code not in ERROR_CODES:
            error_code = "internal_error"
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}


def recovery_action_for(error_code: str) -> str:
    return _RECOVERY.get(error_code, "retry_backoff")


def retryable_for(error_code: str) -> bool:
    return error_code in _RETRYABLE


def error_code_for_exception(exc: BaseException) -> str:
    """Map an exception to a stable error_code (used by run_mcp_tool)."""
    if isinstance(exc, McpToolError):
        return exc.error_code
    if isinstance(exc, QuotaExceededError):
        return "llm_quota_exhausted"
    if isinstance(exc, PydanticValidationError):
        return "validation_failed"
    if isinstance(exc, (ValueError, KeyError)):
        return "invalid_input"
    return "internal_error"


def _safe_message(exc: BaseException) -> str:
    return (str(exc) or exc.__class__.__name__)[:280]


def _classify(exc: BaseException) -> tuple[str, str, dict[str, Any]]:
    """Return ``(error_code, client_safe_message, extra_fields)``."""
    code = error_code_for_exception(exc)
    if isinstance(exc, McpToolError):
        return code, exc.message, dict(exc.details)
    if isinstance(exc, PydanticValidationError):
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "input"
        return code, f"Invalid input -- `{loc}`: {first['msg']}", {"field": loc}
    if code == "internal_error":
        return code, _SAFE_INTERNAL_MESSAGE, {}
    return code, _safe_message(exc), {}


def record_error(tool: str, error_code: str, raw_message: str) -> None:
    """Append a sanitized error record to the in-process ring (diagnostics)."""
    _RECENT_ERRORS.append(
        {"tool": tool, "error_code": error_code, "message": raw_message[:280]}
    )


def get_recent_errors() -> list[dict[str, Any]]:
    return list(_RECENT_ERRORS)


def _request_id() -> str:
    return uuid.uuid4().hex[:12]


def _base_meta(
    tool: str, request_id: str, elapsed_ms: float, response_mode: str | None
) -> dict[str, Any]:
    from api.mcp.capabilities import capabilities_version

    meta: dict[str, Any] = {
        "tool": tool,
        "request_id": request_id,
        "elapsed_ms": round(elapsed_ms, 2),
        "unsafe_for_clinical_use": True,
        "capabilities_version": capabilities_version(),
    }
    if response_mode is not None:
        meta["response_mode"] = response_mode
    return meta


async def run_mcp_tool(
    tool_name: str,
    call: Callable[[], Awaitable[dict[str, Any]]],
    *,
    response_mode: str | None = None,
    context: McpErrorContext | None = None,
) -> dict[str, Any]:
    """Execute a tool body, returning the result dict or a structured error dict."""
    request_id = _request_id()
    start = time.perf_counter()
    logger.info("mcp_tool_started", extra={"tool": tool_name, "request_id": request_id})
    try:
        result = await call()
        elapsed = (time.perf_counter() - start) * 1000.0
        if not isinstance(result, dict):
            raise McpToolError("internal_error", "Tool returned a non-dict result.")
        body_meta: dict[str, Any] = result.pop("_meta", None) or {}
        result.setdefault("success", True)
        result["_meta"] = {
            **_base_meta(tool_name, request_id, elapsed, response_mode),
            **body_meta,
        }
        logger.info(
            "mcp_tool_completed",
            extra={"tool": tool_name, "request_id": request_id, "elapsed_ms": elapsed},
        )
        return result
    except Exception as exc:  # broad catch is the error-boundary contract
        from api.mcp.next_commands import default_error_next_commands

        elapsed = (time.perf_counter() - start) * 1000.0
        code, message, extra = _classify(exc)
        record_error(tool_name, code, str(exc))
        logger.warning(
            "mcp_tool_failed",
            extra={
                "tool": tool_name,
                "request_id": request_id,
                "error_code": code,
                "elapsed_ms": elapsed,
            },
        )
        envelope: dict[str, Any] = {
            "success": False,
            "error_code": code,
            "message": message,
            "retryable": retryable_for(code),
            "recovery_action": recovery_action_for(code),
            "_meta": {
                **_base_meta(tool_name, request_id, elapsed, response_mode),
                "next_commands": default_error_next_commands(tool_name),
            },
        }
        if extra:
            envelope.update(extra)
        return envelope


def build_arg_error_envelope(
    *,
    tool_name: str,
    loc: str,
    error_type: str,
    valid_params: list[str],
    signature: str,
    suggestion: str | None,
    constraints: tuple[list[str], str] | None = None,
) -> dict[str, Any]:
    """Standard invalid-input envelope for an argument-binding failure."""
    from api.mcp.capabilities import capabilities_version

    base_meta = {
        "tool": tool_name,
        "request_id": _request_id(),
        "unsafe_for_clinical_use": True,
        "capabilities_version": capabilities_version(),
        "next_commands": [{"tool": "phentrieve_get_capabilities", "arguments": {}}],
    }
    if constraints is not None:
        allowed, human = constraints
        message = f"Invalid value for argument `{loc}` of {tool_name}: {human}."
        return {
            "success": False,
            "error_code": "validation_failed",
            "message": message[:280],
            "retryable": False,
            "recovery_action": "reformulate_input",
            "field": loc,
            "allowed_values": allowed,
            "hint": signature,
            "_meta": base_meta,
        }
    if error_type in ("missing", "missing_argument"):
        head = f"Missing required argument `{loc}` for {tool_name}."
    elif error_type == "unexpected_keyword_argument":
        head = f"Unknown argument `{loc}` for {tool_name}."
    else:
        head = f"Invalid value for argument `{loc}` of {tool_name}."
    dym = f" Did you mean `{suggestion}`?" if suggestion else ""
    message = f"{head}{dym} Valid argument names are listed in allowed_values."
    return {
        "success": False,
        "error_code": "validation_failed",
        "message": message[:280],
        "retryable": False,
        "recovery_action": "reformulate_input",
        "field": loc,
        "allowed_values": valid_params,
        "hint": signature,
        "_meta": base_meta,
    }
