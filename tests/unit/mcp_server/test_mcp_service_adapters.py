"""Unit tests for api.mcp.service_adapters domain logic."""

from __future__ import annotations

import json

import pytest

from api.mcp.envelope import McpToolError
from api.mcp.service_adapters import (
    chunk_text_service,
    export_phenopacket_service,
    extract_hpo_terms_llm_service,
    extract_hpo_terms_service,
)


def test_chunk_text_service_simple_strategy():
    out = chunk_text_service(
        text="First sentence. Second sentence.", language="en", strategy="simple"
    )
    assert out["chunk_count"] >= 1
    assert out["chunks"][0]["chunk_id"] == 1
    assert "text" in out["chunks"][0]


def test_chunk_text_service_unknown_strategy_raises_invalid_input():
    with pytest.raises(McpToolError) as ei:
        chunk_text_service(text="x", language="en", strategy="does-not-exist")
    assert ei.value.error_code == "invalid_input"


def test_export_phenopacket_service_round_trips_case_id():
    out = export_phenopacket_service(
        case_id="CASE-1",
        case_label="demo",
        input_text=None,
        subject=None,
        phenotypes=[
            {"hpo_id": "HP:0001250", "label": "Seizure", "assertion": "affirmed"},
            {
                "hpo_id": "HP:0001263",
                "label": "Global developmental delay",
                "assertion": "negated",
            },
        ],
        include_annotation_sidecar=True,
    )
    assert "phenopacket_json" in out
    packet = json.loads(out["phenopacket_json"])
    assert packet["id"] == "CASE-1"


def test_extract_service_uses_injected_service():
    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

    out = extract_hpo_terms_service(
        text="hi",
        language="en",
        include_details=False,
        include_chunk_positions=False,
        num_results_per_chunk=5,
        chunk_retrieval_threshold=0.5,
        service=fake_service,
    )
    assert out["aggregated_hpo_terms"] == []
    assert captured["extraction_backend"] == "standard"
    assert captured["num_results_per_chunk"] == 5


def test_extract_llm_falls_back_on_backend_error_when_allowed(monkeypatch):
    import api.config as api_config

    monkeypatch.setattr(api_config, "PHENTRIEVE_ENV", "development", raising=False)

    def flaky_service(**kwargs):
        if kwargs.get("extraction_backend") == "llm":
            raise RuntimeError("backend down")
        return {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

    out = extract_hpo_terms_llm_service(
        text="hi",
        language="en",
        include_details=True,
        include_chunk_positions=True,
        num_results_per_chunk=5,
        chunk_retrieval_threshold=0.5,
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        allow_standard_fallback=True,
        service=flaky_service,
    )
    assert out["meta"]["fallback_reason"] == "llm_backend_error"


def test_extract_llm_reraises_backend_error_without_fallback(monkeypatch):
    import api.config as api_config

    monkeypatch.setattr(api_config, "PHENTRIEVE_ENV", "development", raising=False)

    def flaky_service(**kwargs):
        raise RuntimeError("backend down")

    with pytest.raises(RuntimeError):
        extract_hpo_terms_llm_service(
            text="hi",
            language="en",
            include_details=True,
            include_chunk_positions=True,
            num_results_per_chunk=5,
            chunk_retrieval_threshold=0.5,
            llm_mode="two_phase",
            llm_internal_mode="whole_document_grounded",
            allow_standard_fallback=False,
            service=flaky_service,
        )
