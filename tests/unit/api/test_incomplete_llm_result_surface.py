"""A partial LLM result is flagged at the top level of REST meta and MCP _meta.

When a phase-1 extraction group fails and no fallback recovers it, the request
still succeeds without that group's findings (#395). The pipeline counts such
groups in ``phase1_unrecovered_groups``; REST and MCP add ``incomplete: true``
plus a short ``incomplete_reason``. Complete results carry neither key.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from api.schemas.text_processing_schemas import TextProcessingRequest
from api.services.text_processing_execution import (
    adapt_shared_service_response_to_api,
)
from phentrieve.llm.pipeline_trace import incomplete_extraction_signal
from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype
from phentrieve.text_processing.full_text_service import (
    adapt_full_text_response,
    run_llm_backend,
)

pytestmark = pytest.mark.unit


def _api_response(phase_counts: dict[str, int]):
    result = LLMExtractionResult(
        terms=[
            LLMPhenotype(
                term_id="HP:0001250",
                label="Seizure",
                assertion="present",
                evidence="recurrent seizures",
            )
        ],
        meta=LLMMeta(llm_model="stub", llm_mode="two_phase", phase_counts=phase_counts),
    )
    provider = SimpleNamespace(
        provider_name="ollama", model_name="stub", base_url="http://localhost"
    )
    pipeline = SimpleNamespace(run=lambda **_kwargs: result)
    payload = run_llm_backend(
        text="Recurrent seizures. No ataxia.",
        llm_model="stub",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_legacy",
        provider_factory=lambda **_kw: provider,
        pipeline_factory=lambda **_kw: pipeline,
    )
    normalized = adapt_full_text_response(payload, extraction_backend="llm")
    return adapt_shared_service_response_to_api(
        normalized, request=TextProcessingRequest(text="x")
    )


def test_rest_meta_flags_unrecovered_phase1_group():
    response = _api_response(
        {"phase1_failed_groups": 1, "phase1_unrecovered_groups": 1}
    )

    assert response.meta["incomplete"] is True
    assert "1 chunk group(s)" in response.meta["incomplete_reason"]
    assert response.meta["observability"]["phase1_unrecovered_groups"] == 1


def test_rest_meta_has_no_flag_when_a_fallback_recovered_the_group():
    # A grouped_large window failed but grouped_small recovered it.
    response = _api_response(
        {"phase1_failed_groups": 1, "phase1_unrecovered_groups": 0}
    )

    assert "incomplete" not in response.meta
    assert "incomplete_reason" not in response.meta


def test_signal_ignores_missing_or_malformed_observability():
    assert incomplete_extraction_signal(None) == {}
    assert incomplete_extraction_signal({}) == {}
    assert incomplete_extraction_signal({"phase1_unrecovered_groups": 0}) == {}


def _call_llm_tool(monkeypatch, unrecovered: int, response_mode: str) -> dict:
    payload = {
        "meta": {
            "extraction_backend": "llm",
            "observability": {"phase1_unrecovered_groups": unrecovered},
        },
        "processed_chunks": [],
        "aggregated_hpo_terms": [],
    }
    monkeypatch.setattr(
        "api.mcp.tools.retrieval.extract_hpo_terms_llm_service",
        lambda **_kwargs: payload,
    )
    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    result = asyncio.run(
        mcp.call_tool(
            "phentrieve_extract_hpo_terms_llm",
            {
                "text": "x",
                "research_use_acknowledged": True,
                "response_mode": response_mode,
            },
        )
    )
    return result.structured_content


@pytest.mark.parametrize("response_mode", ["minimal", "compact", "full"])
def test_mcp_llm_tool_flags_unrecovered_phase1_group(monkeypatch, response_mode):
    data = _call_llm_tool(monkeypatch, unrecovered=2, response_mode=response_mode)

    assert data["_meta"]["incomplete"] is True
    assert "2 chunk group(s)" in data["_meta"]["incomplete_reason"]


def test_mcp_llm_tool_has_no_flag_for_complete_result(monkeypatch):
    data = _call_llm_tool(monkeypatch, unrecovered=0, response_mode="compact")

    assert "incomplete" not in data["_meta"]
