"""B2 Task 9: family_history_findings + per-term experiencer + derived excluded.

Drives the REAL adapter chain -- run_llm_backend() -> adapt_full_text_response()
-> adapt_shared_service_response_to_api() -- rather than a bare model_validate,
so the field is exercised at every surface boundary it must survive (F4). Also
covers the deterministic-backend fallback where ``excluded`` is derived from the
pipeline ``status`` via ``is_excluded`` (F2).
"""

from types import SimpleNamespace

import pytest

from api.schemas.text_processing_schemas import (
    TextProcessingRequest,
    TextProcessingResponseAPI,
)
from api.services.text_processing_execution import (
    adapt_shared_service_response_to_api,
)
from phentrieve.llm.types import (
    LLMExtractionResult,
    LLMMeta,
    LLMPhenotype,
)
from phentrieve.text_processing.full_text_service import (
    adapt_full_text_response,
    run_llm_backend,
)

pytestmark = pytest.mark.unit


def _extraction_result() -> LLMExtractionResult:
    """A proband present term, a proband negated term, and a family finding."""
    return LLMExtractionResult(
        terms=[
            LLMPhenotype(
                term_id="HP:0001250",
                label="Seizure",
                assertion="present",
                experiencer="proband",
                evidence="recurrent seizures",
            ),
            LLMPhenotype(
                term_id="HP:0001945",
                label="Fever",
                assertion="negated",
                experiencer="proband",
                evidence="no fever",
            ),
        ],
        family_history_findings=[
            LLMPhenotype(
                term_id="HP:0002076",
                label="Migraine",
                assertion="present",
                experiencer="family_history",
                evidence="mother has migraine",
            ),
        ],
        meta=LLMMeta(llm_model="stub", llm_mode="two_phase"),
    )


def _run_backend(result: LLMExtractionResult) -> dict:
    """Run the real LLM backend with a stubbed provider + pipeline (no model)."""
    provider = SimpleNamespace(
        provider_name="ollama", model_name="stub", base_url="http://localhost"
    )
    pipeline = SimpleNamespace(run=lambda **_kwargs: result)
    return run_llm_backend(
        text="Patient had recurrent seizures but no fever. Mother has migraine.",
        llm_model="stub",
        llm_mode="two_phase",
        # legacy mode skips grounding preprocessing (no retrieval / data needed).
        llm_internal_mode="whole_document_legacy",
        provider_factory=lambda **_kw: provider,
        pipeline_factory=lambda **_kw: pipeline,
    )


def _find(terms: list[dict], hpo_id: str) -> dict:
    return next(term for term in terms if term.get("id") == hpo_id)


def test_run_llm_backend_emits_family_findings_and_derived_excluded():
    payload = _run_backend(_extraction_result())

    family = payload["family_history_findings"]
    assert len(family) == 1
    assert family[0]["id"] == "HP:0002076"
    assert family[0]["experiencer"] == "family_history"
    assert family[0]["excluded"] is False

    seizure = _find(payload["aggregated_hpo_terms"], "HP:0001250")
    fever = _find(payload["aggregated_hpo_terms"], "HP:0001945")
    assert seizure["experiencer"] == "proband"
    assert seizure["excluded"] is False
    assert fever["experiencer"] == "proband"
    # ``negated`` (pipeline vocab) canonicalizes to excluded (F2).
    assert fever["excluded"] is True


def test_adapt_full_text_response_preserves_family_findings():
    payload = _run_backend(_extraction_result())

    normalized = adapt_full_text_response(payload, extraction_backend="llm")

    assert "family_history_findings" in normalized
    assert [term["id"] for term in normalized["family_history_findings"]] == [
        "HP:0002076"
    ]
    # Existing keys are untouched.
    assert {term["id"] for term in normalized["aggregated_hpo_terms"]} == {
        "HP:0001250",
        "HP:0001945",
    }


def test_full_chain_surfaces_family_experiencer_and_excluded_on_api():
    payload = _run_backend(_extraction_result())
    normalized = adapt_full_text_response(payload, extraction_backend="llm")

    response = adapt_shared_service_response_to_api(
        normalized, request=TextProcessingRequest(text="x")
    )

    assert isinstance(response, TextProcessingResponseAPI)
    assert len(response.family_history_findings) == 1
    family_term = response.family_history_findings[0]
    assert family_term.id == "HP:0002076"
    assert family_term.experiencer == "family_history"
    assert family_term.excluded is False

    by_id = {term.id: term for term in response.aggregated_hpo_terms}
    assert by_id["HP:0001250"].experiencer == "proband"
    assert by_id["HP:0001250"].excluded is False
    assert by_id["HP:0001945"].experiencer == "proband"
    assert by_id["HP:0001945"].excluded is True
    # status is NOT rewritten -- only the derived excluded signal is added (F2).
    assert by_id["HP:0001945"].status == "negated"


def test_api_builder_derives_excluded_from_status_when_absent():
    """Deterministic backend lacks a precomputed ``excluded``; derive it (F2)."""
    service_result = {
        "meta": {"extraction_backend": "standard"},
        "processed_chunks": [],
        "aggregated_hpo_terms": [
            {
                "id": "HP:0001250",
                "name": "Seizure",
                "confidence": 0.9,
                "status": "affirmed",
                "evidence_count": 1,
                "source_chunk_ids": [],
            },
            {
                "id": "HP:0001945",
                "name": "Fever",
                "confidence": 0.8,
                "status": "negated",
                "evidence_count": 1,
                "source_chunk_ids": [],
            },
        ],
        "family_history_findings": [],
    }

    response = adapt_shared_service_response_to_api(
        service_result, request=TextProcessingRequest(text="x")
    )

    by_id = {term.id: term for term in response.aggregated_hpo_terms}
    assert by_id["HP:0001250"].excluded is False
    assert by_id["HP:0001945"].excluded is True
    assert response.family_history_findings == []
