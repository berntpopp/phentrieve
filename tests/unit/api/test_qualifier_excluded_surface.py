"""B3 Task 11: surface negated_qualifier-derived excluded findings (REST + MCP).

Two concerns:

  (a) round-trip -- a service-payload term dict carrying
      ``match_method="negated_qualifier_derived"``, ``qualifier_surface_text``,
      and ``negated_qualifier`` survives ``adapt_shared_service_response_to_api``
      / ``AggregatedHPOTermAPI`` (REST) and ``project_aggregated_terms_for_mcp``
      / ``project_extract_payload`` (MCP) with the fields populated.

  (b) F5 service-survival -- a GENERATED excluded ``LLMPhenotype`` whose
      inherited evidence records carry VALID chunk ids survives the REAL
      ``_adapt_llm_aggregated_terms``, while an otherwise-identical term whose
      evidence records reference an invalid chunk id is DROPPED at the
      ``if raw_source_chunk_ids and not source_chunk_ids: continue`` guard in
      ``full_text_service.py``. The present-vs-absent contrast proves the
      inherited valid chunk ids are what keep the generated term alive.
"""

from __future__ import annotations

import pytest

from api.mcp.projection import (
    project_aggregated_terms_for_mcp,
    project_extract_payload,
)
from api.schemas.text_processing_schemas import TextProcessingRequest
from api.services.text_processing_execution import (
    adapt_shared_service_response_to_api,
)
from phentrieve.llm.config import NEGATED_ASSERTION
from phentrieve.llm.types import LLMPhenotype, LLMPhenotypeEvidence
from phentrieve.text_processing.full_text_service import _adapt_llm_aggregated_terms

pytestmark = pytest.mark.unit


def _excluded_service_payload() -> dict:
    """Shared-service payload for a negated_qualifier-derived excluded finding."""
    return {
        "meta": {"extraction_backend": "llm"},
        "processed_chunks": [
            {
                "chunk_id": 1,
                "text": "recurrent seizures but no fever",
                "status": "affirmed",
                "hpo_matches": [],
            }
        ],
        "aggregated_hpo_terms": [
            {
                "id": "HP:0001945",
                "name": "Fever",
                "confidence": 0.91,
                "status": "negated",
                "experiencer": "proband",
                "excluded": True,
                "negated_qualifier": "fever",
                "qualifier_surface_text": "fever",
                "match_method": "negated_qualifier_derived",
                "evidence_count": 1,
                "source_chunk_ids": [1],
                "top_evidence_chunk_id": 1,
                "text_attributions": [],
            }
        ],
        "family_history_findings": [],
    }


def test_excluded_finding_round_trips_through_rest():
    """REST: AggregatedHPOTermAPI carries the three derived-exclusion fields."""
    response = adapt_shared_service_response_to_api(
        _excluded_service_payload(), request=TextProcessingRequest(text="x")
    )

    term = response.aggregated_hpo_terms[0]
    assert term.id == "HP:0001945"
    assert term.excluded is True
    # status is NOT rewritten -- only the derived signals ride alongside it.
    assert term.status == "negated"
    assert term.negated_qualifier == "fever"
    assert term.qualifier_surface_text == "fever"
    assert term.match_method == "negated_qualifier_derived"


def test_excluded_finding_survives_mcp_projection():
    """MCP: the projection carries qualifier_surface_text / match_method."""
    payload = _excluded_service_payload()

    projected = project_aggregated_terms_for_mcp(payload["aggregated_hpo_terms"])[0]
    assert projected["hpo_id"] == "HP:0001945"
    assert projected["excluded"] is True
    assert projected["qualifier_surface_text"] == "fever"
    assert projected["match_method"] == "negated_qualifier_derived"
    assert projected["negated_qualifier"] == "fever"

    out = project_extract_payload(payload)
    agg = out["aggregated_hpo_terms"][0]
    assert agg["qualifier_surface_text"] == "fever"
    assert agg["match_method"] == "negated_qualifier_derived"
    assert agg["negated_qualifier"] == "fever"


def _grounded_chunks() -> list[dict]:
    return [
        {"chunk_id": 1, "text": "recurrent seizures but no fever"},
        {"chunk_id": 2, "text": "mother has migraine"},
    ]


def _generated_excluded_term(*, term_id: str, chunk_ids: list[int]) -> LLMPhenotype:
    """A B3-generated excluded term inheriting a source finding's evidence.

    ``qualifier_surface_text`` / ``match_method`` mirror what
    ``_build_qualifier_exclusions`` emits; ``negated_qualifier`` is None on the
    generated term. ``chunk_ids`` are the INHERITED source-finding chunk refs.
    """
    return LLMPhenotype(
        term_id=term_id,
        label="Fever",
        evidence="fever",
        assertion=NEGATED_ASSERTION,
        experiencer="proband",
        negated_qualifier=None,
        qualifier_surface_text="fever",
        match_method="negated_qualifier_derived",
        confidence=0.9,
        score=0.9,
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="fever", evidence_text="fever", chunk_ids=chunk_ids
            )
        ],
    )


def test_generated_excluded_term_survives_when_inherited_chunk_refs_are_valid():
    """F5: valid inherited chunk ids keep the generated term alive; invalid drop.

    Both terms are byte-for-byte identical in shape apart from the chunk id
    their evidence record references. The only removal path in
    ``_adapt_llm_aggregated_terms`` is the ``raw_source_chunk_ids`` guard, so the
    present-vs-absent contrast isolates that guard: the valid-ref term survives,
    the invalid-ref term is dropped.
    """
    valid = _generated_excluded_term(term_id="HP:0001945", chunk_ids=[1])
    invalid = _generated_excluded_term(term_id="HP:0000252", chunk_ids=[999])

    adapted = _adapt_llm_aggregated_terms(
        [valid, invalid], grounded_chunks=_grounded_chunks()
    )

    by_id = {term["id"]: term for term in adapted}
    # Inherited VALID chunk id (1 is in grounded_chunks) keeps it alive.
    assert "HP:0001945" in by_id
    # Invalid ref (999 not in grounded_chunks) hits the drop guard.
    assert "HP:0000252" not in by_id

    survivor = by_id["HP:0001945"]
    assert survivor["excluded"] is True
    assert survivor["status"] == NEGATED_ASSERTION
    assert survivor["source_chunk_ids"] == [1]
    # The B3 provenance fields ride through the service adapter unchanged.
    assert survivor["match_method"] == "negated_qualifier_derived"
    assert survivor["qualifier_surface_text"] == "fever"
    # Generated term carries no negated_qualifier of its own (None -> omitted).
    assert "negated_qualifier" not in survivor
