"""MCP LLM quota policy on top of the shared api.llm_quota helpers (#434)."""

from __future__ import annotations

import pytest

import api.config as api_config
from api.llm_quota import QuotaExceededError
from api.mcp.service_adapters import extract_hpo_terms_llm_service

pytestmark = pytest.mark.unit


def _service(**kwargs):
    return {
        "meta": {"extraction_backend": kwargs["extraction_backend"]},
        "processed_chunks": [],
        "aggregated_hpo_terms": [],
    }


def _call(allow_standard_fallback: bool):
    return extract_hpo_terms_llm_service(
        text="Seizures.",
        language="en",
        include_details=False,
        include_chunk_positions=False,
        num_results_per_chunk=3,
        chunk_retrieval_threshold=0.5,
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        allow_standard_fallback=allow_standard_fallback,
        service=_service,
    )


@pytest.fixture
def production_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(api_config, "PHENTRIEVE_ENV", "production")
    monkeypatch.setattr(api_config, "PHENTRIEVE_LLM_DAILY_LIMIT", 1)
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_LLM_QUOTA_DB_PATH", str(tmp_path / "quota.db")
    )


def test_mcp_records_success_then_exhausts_in_production(production_quota):
    first = _call(allow_standard_fallback=False)
    assert first["meta"]["quota_limit"] == 1
    assert first["meta"]["quota_remaining"] == 0

    with pytest.raises(QuotaExceededError):
        _call(allow_standard_fallback=False)

    fallback = _call(allow_standard_fallback=True)
    assert fallback["meta"]["fallback_reason"] == "llm_quota_exhausted"
    assert fallback["meta"]["extraction_backend"] == "standard"


def test_mcp_quota_policy_ignores_rest_enforce_override(production_quota, monkeypatch):
    # MCP policy is unchanged by the refactor: production-only, and the REST
    # PHENTRIEVE_LLM_QUOTA_ENFORCE override does not apply to it.
    monkeypatch.setattr(api_config, "PHENTRIEVE_LLM_QUOTA_ENFORCE", "false")
    _call(allow_standard_fallback=False)
    with pytest.raises(QuotaExceededError):
        _call(allow_standard_fallback=False)


def test_mcp_does_not_enforce_quota_outside_production(tmp_path, monkeypatch):
    monkeypatch.setattr(api_config, "PHENTRIEVE_ENV", "development")
    monkeypatch.setattr(api_config, "PHENTRIEVE_LLM_QUOTA_ENFORCE", "true")
    monkeypatch.setattr(api_config, "PHENTRIEVE_LLM_DAILY_LIMIT", 1)
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_LLM_QUOTA_DB_PATH", str(tmp_path / "quota.db")
    )
    for _ in range(3):
        result = _call(allow_standard_fallback=False)
        assert "quota_limit" not in result["meta"]
