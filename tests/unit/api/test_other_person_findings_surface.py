"""other_person_findings and per-term temporality on the service, REST and MCP surfaces.

Drives the real adapter chain like ``test_family_findings_surface``:
run_llm_backend() -> adapt_full_text_response() ->
adapt_shared_service_response_to_api(), plus the MCP projection, schema and
extract tool.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from api.mcp.capabilities import build_capabilities
from api.mcp.projection import project_extract_payload
from api.mcp.schemas import EXTRACT_SCHEMA
from api.schemas.text_processing_schemas import TextProcessingRequest
from api.services.text_processing_execution import (
    adapt_shared_service_response_to_api,
)
from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype
from phentrieve.text_processing.full_text_service import (
    adapt_full_text_response,
    run_llm_backend,
)

pytestmark = pytest.mark.unit


def _extraction_result() -> LLMExtractionResult:
    """A historical proband finding and a finding of another person."""
    return LLMExtractionResult(
        terms=[
            LLMPhenotype(
                term_id="HP:0002099",
                label="Asthma",
                assertion="present",
                experiencer="proband",
                temporality="historical",
                evidence="asthma in childhood",
            ),
        ],
        other_person_findings=[
            LLMPhenotype(
                term_id="HP:0000821",
                label="Hypothyroidism",
                assertion="present",
                experiencer="other",
                temporality="current",
                evidence="her roommate has hypothyroidism",
            ),
        ],
        meta=LLMMeta(llm_model="stub", llm_mode="two_phase"),
    )


def _run_backend(result: LLMExtractionResult) -> dict:
    provider = SimpleNamespace(
        provider_name="ollama", model_name="stub", base_url="http://localhost"
    )
    pipeline = SimpleNamespace(run=lambda **_kwargs: result)
    return run_llm_backend(
        text="Asthma in childhood. Her roommate has hypothyroidism.",
        llm_model="stub",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_legacy",
        provider_factory=lambda **_kw: provider,
        pipeline_factory=lambda **_kw: pipeline,
    )


def test_run_llm_backend_emits_other_person_findings_and_temporality():
    payload = _run_backend(_extraction_result())

    assert [term["id"] for term in payload["aggregated_hpo_terms"]] == ["HP:0002099"]
    assert payload["aggregated_hpo_terms"][0]["temporality"] == "historical"
    other = payload["other_person_findings"]
    assert [term["id"] for term in other] == ["HP:0000821"]
    assert other[0]["experiencer"] == "other"
    assert other[0]["temporality"] == "current"


def test_full_chain_surfaces_other_person_findings_on_api():
    payload = _run_backend(_extraction_result())
    normalized = adapt_full_text_response(payload, extraction_backend="llm")

    response = adapt_shared_service_response_to_api(
        normalized, request=TextProcessingRequest(text="x")
    )

    assert response.aggregated_hpo_terms[0].temporality == "historical"
    assert [term.id for term in response.other_person_findings] == ["HP:0000821"]
    assert response.other_person_findings[0].experiencer == "other"
    assert response.family_history_findings == []


def test_api_defaults_without_other_person_findings_or_temporality():
    response = adapt_shared_service_response_to_api(
        {
            "meta": {"extraction_backend": "standard"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [
                {"id": "HP:0001250", "name": "Seizure", "status": "affirmed"}
            ],
        },
        request=TextProcessingRequest(text="x"),
    )

    assert response.other_person_findings == []
    assert response.aggregated_hpo_terms[0].temporality is None


def test_mcp_projects_other_person_findings():
    out = project_extract_payload(
        {
            "aggregated_hpo_terms": [],
            "other_person_findings": [
                {
                    "id": "HP:0000821",
                    "name": "Hypothyroidism",
                    "status": "present",
                    "experiencer": "other",
                    "temporality": "current",
                    "excluded": False,
                    "source_chunk_ids": [1],
                }
            ],
        }
    )

    term = out["other_person_findings"][0]
    assert term["hpo_id"] == "HP:0000821"
    assert term["experiencer"] == "other"
    assert term["temporality"] == "current"


def test_mcp_schema_and_capabilities_document_the_new_fields():
    assert "other_person_findings" in EXTRACT_SCHEMA["properties"]
    serialized = json.dumps(build_capabilities(), default=str, sort_keys=True)
    assert "other_person_findings" in serialized
    assert "temporality" in serialized


def test_mcp_extract_tool_surfaces_other_person_findings(monkeypatch):
    payload = {
        "meta": {"extraction_backend": "llm"},
        "processed_chunks": [],
        "aggregated_hpo_terms": [],
        "other_person_findings": [
            {
                "id": "HP:0000821",
                "name": "Hypothyroidism",
                "status": "present",
                "experiencer": "other",
                "temporality": "current",
                "excluded": False,
                "source_chunk_ids": [1],
            }
        ],
    }
    monkeypatch.setattr(
        "api.mcp.tools.retrieval.extract_hpo_terms_service",
        lambda **_kwargs: payload,
    )
    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    result = asyncio.run(
        mcp.call_tool(
            "phentrieve_extract_hpo_terms",
            {
                "text": "x",
                "research_use_acknowledged": True,
                "response_mode": "compact",
            },
        )
    )
    data = result.structured_content

    term = data["other_person_findings"][0]
    assert term["hpo_id"] == "HP:0000821"
    assert term["temporality"] == "current"
