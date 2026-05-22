from __future__ import annotations

from typing import Any

from phentrieve.llm.types import Phase1FailureClass, Phase1Mode


class LLMPipelinePhaseError(RuntimeError):
    def __init__(
        self,
        phase: str,
        message: str,
        *,
        usage: dict[str, int] | None = None,
        request_count: int = 0,
        elapsed_seconds: float | None = None,
        groups_trace: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.usage = dict(usage or {})
        self.request_count = int(request_count or 0)
        self.elapsed_seconds = elapsed_seconds
        self.groups_trace = list(groups_trace or [])
        self.phase1_trace: dict[str, Any] = {}
        self.phase_counts: dict[str, int] = {}
        self.phase_request_counts: dict[str, int] = {}
        self.initial_mode: Phase1Mode | None = None
        self.final_mode: Phase1Mode | None = None
        self.fallback_triggered = False
        self.failure_class: Phase1FailureClass = None


def phase1_failure_message(exc: Exception) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        parts.append(type(current).__name__.lower())
        message = str(current).strip().lower()
        if message:
            parts.append(message)
        current = current.__cause__
    return " ".join(parts)


def classify_phase1_failure(exc: Exception) -> Phase1FailureClass:
    if isinstance(exc, LLMPipelinePhaseError) and exc.failure_class is not None:
        return exc.failure_class
    message = phase1_failure_message(exc)
    if "refusal" in message:
        return "structured_refusal"
    if "timeout" in message or "deadline_exceeded" in message or "timed out" in message:
        return "provider_timeout"
    if (
        "json" in message
        or "unterminated" in message
        or "no structured response payload" in message
    ):
        return "structured_json_invalid"
    if (
        "validationerror" in message
        or "schema" in message
        or "field required" in message
        or "input should" in message
        or "literal" in message
        or "pydantic" in message
    ):
        return "structured_schema_validation_failed"
    if (
        "unauthorized" in message
        or "forbidden" in message
        or "authentication" in message
        or "api key" in message
        or "permission" in message
        or "billing" in message
        or "quota" in message
        or "401" in message
        or "403" in message
    ):
        return "provider_auth_error"
    if (
        "connection" in message
        or "network" in message
        or "transport" in message
        or "dns" in message
        or "ssl" in message
        or "remoteprotocol" in message
        or "readerror" in message
        or "connecterror" in message
    ):
        return "provider_transport_error"
    if (
        "unsupported" in message
        or "misconfig" in message
        or "configuration" in message
        or "invalid base_url" in message
        or "model not found" in message
        or "unknown provider" in message
    ):
        return "provider_config_error"
    return "provider_execution_error"


def next_phase1_mode(current_mode: Phase1Mode) -> Phase1Mode | None:
    if current_mode == "ungrouped":
        return "grouped_large"
    if current_mode == "grouped_large":
        return "grouped_small"
    return None


def group_source_chunk_ids(group_trace: dict[str, Any]) -> list[int]:
    chunk_ids = group_trace.get("source_chunk_ids")
    if not isinstance(chunk_ids, list):
        chunk_ids = group_trace.get("chunk_ids", [])
    return [int(chunk_id) for chunk_id in chunk_ids]
