"""Unit tests for api.mcp.envelope (Family B run_mcp_tool boundary)."""

from __future__ import annotations

import asyncio

from api.llm_quota import QuotaExceededError
from api.mcp.envelope import (
    ERROR_CODES,
    McpToolError,
    error_code_for_exception,
    get_recent_errors,
    recovery_action_for,
    retryable_for,
    run_mcp_tool,
)


def _run(coro):
    return asyncio.run(coro)


def test_error_codes_core_set():
    assert {
        "invalid_input",
        "validation_failed",
        "not_found",
        "llm_quota_exhausted",
        "internal_error",
    } <= ERROR_CODES


def test_recovery_action_and_retryable():
    assert recovery_action_for("not_found") == "reformulate_input"
    assert recovery_action_for("llm_quota_exhausted") == "retry_backoff"
    assert recovery_action_for("llm_unavailable") == "switch_tool"
    assert retryable_for("llm_quota_exhausted") is True
    assert retryable_for("not_found") is False


def test_error_code_for_exception():
    exc = QuotaExceededError(
        quota_used=5, quota_limit=5, quota_remaining=0, usage_date_utc="2026-06-13"
    )
    assert error_code_for_exception(exc) == "llm_quota_exhausted"
    assert error_code_for_exception(ValueError("bad")) == "invalid_input"
    assert error_code_for_exception(RuntimeError("x")) == "internal_error"


def test_run_mcp_tool_success_stamps_meta():
    async def call():
        return {"results": [{"hpo_id": "HP:1"}]}

    out = _run(
        run_mcp_tool("phentrieve_search_hpo_terms", call, response_mode="compact")
    )
    assert out["success"] is True
    meta = out["_meta"]
    assert meta["tool"] == "phentrieve_search_hpo_terms"
    assert isinstance(meta["request_id"], str) and meta["request_id"]
    assert isinstance(meta["elapsed_ms"], (int, float))
    assert meta["response_mode"] == "compact"
    assert meta["unsafe_for_clinical_use"] is True
    assert meta["capabilities_version"].startswith("sha256:")


def test_run_mcp_tool_merges_body_meta():
    async def call():
        return {
            "results": [],
            "_meta": {"next_commands": [{"tool": "x", "arguments": {}}]},
        }

    out = _run(run_mcp_tool("phentrieve_search_hpo_terms", call))
    assert out["_meta"]["next_commands"] == [{"tool": "x", "arguments": {}}]
    assert out["_meta"]["unsafe_for_clinical_use"] is True


def test_run_mcp_tool_known_error_envelope():
    async def call():
        raise McpToolError(
            "not_found", "No HPO term HP:9999999.", details={"field": "term2_id"}
        )

    out = _run(run_mcp_tool("phentrieve_compare_hpo_terms", call))
    assert out["success"] is False
    assert out["error_code"] == "not_found"
    assert out["retryable"] is False
    assert out["recovery_action"] == "reformulate_input"
    assert out["field"] == "term2_id"
    assert out["_meta"]["unsafe_for_clinical_use"] is True
    assert out["_meta"]["next_commands"]


def test_run_mcp_tool_internal_error_sanitized_and_recorded():
    async def call():
        raise RuntimeError("kaboom secret path /etc/passwd")

    out = _run(run_mcp_tool("phentrieve_search_hpo_terms", call))
    assert out["success"] is False
    assert out["error_code"] == "internal_error"
    assert "/etc/passwd" not in out["message"]
    recent = get_recent_errors()
    assert recent and recent[-1]["tool"] == "phentrieve_search_hpo_terms"
