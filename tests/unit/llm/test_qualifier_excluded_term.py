"""Qualifier-derived excluded finding tests (extraction contract v2, block B3).

These tests pin Task 10's requirement: after proband resolution, a resolved
proband finding that carries a truthy ``negated_qualifier`` (the ``Y`` of an
"X without Y" phrase) triggers a SECOND retrieval on ``Y``. When ``Y`` maps
above ``self.similarity_threshold`` the pipeline emits a GENERATED excluded
``LLMPhenotype`` for ``Y`` (``assertion == "negated"``,
``match_method == "negated_qualifier_derived"``,
``qualifier_surface_text == Y``, ``experiencer == "proband"``). Below the floor
(or when nothing is retrieved) NO term is generated and the source finding's
``negated_qualifier`` metadata string is retained unchanged.

Critical (F5): the generated term MUST inherit the SOURCE proband finding's
evidence records so it carries valid ``chunk_ids``. A generated excluded term
whose evidence references no valid chunk is silently dropped at the aggregation
service boundary (``full_text_service.py``); the "X without Y" span lives in the
same chunk as X, so inheriting X's evidence records is correct.

The retriever and the phase-2 LLM mapping provider are stubbed, so no model /
Gemini / network is required (mirrors the existing tests/unit/llm pipeline
stubs).
"""

from __future__ import annotations

from typing import Any

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import (
    LLMExtractionResult,
    LLMPipelineConfig,
)

TwoPhaseLLMPipeline = pipeline_module.TwoPhaseLLMPipeline


class FakeProvider(LLMProvider):
    """Structured-prompt provider stub (mirrors tests/unit/llm/test_pipeline)."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.structured_calls: list[dict[str, Any]] = []
        self.last_request_count = 0

    def complete(self, messages):  # pragma: no cover - never invoked here
        raise AssertionError("provider.complete must not be called")

    def run_structured_prompt(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
        max_output_tokens=None,
    ):
        self.structured_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
                "max_output_tokens": max_output_tokens,
            }
        )
        response = self.responses.pop(0)
        self.last_usage = response.get(
            "usage",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        self.last_request_count = int(response.get("request_count", 1))
        if "exception" in response:
            raise response["exception"]
        parsed = response_model.model_validate(response["parsed"])
        self.last_structured_payload = parsed.model_dump(mode="json")
        return parsed

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        return {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}


class PhraseAwareToolExecutor:
    """Retriever stub returning candidates keyed by the queried phrase.

    ``_retrieve_candidates`` aligns the returned batch results positionally to
    the expanded query list, so returning ``{"phrase": phrase, "candidates":
    ...}`` per input phrase lets the source phrase (X) and the qualifier phrase
    (Y) retrieve DIFFERENT candidate sets.
    """

    def __init__(self, candidates_by_phrase: dict[str, list[dict[str, Any]]]) -> None:
        self.candidates_by_phrase = candidates_by_phrase
        self.queries: list[dict[str, Any]] = []

    def query_batch_hpo_terms(self, *, phrases, language, n_results):
        self.queries.append(
            {"phrases": list(phrases), "language": language, "n_results": n_results}
        )
        return [
            {
                "phrase": phrase,
                "candidates": list(self.candidates_by_phrase.get(phrase, [])),
            }
            for phrase in phrases
        ]


def _grounded_phenotype(
    phrase: str,
    category: str,
    *,
    chunk_id: int,
    negated_qualifier: str | None = None,
    experiencer: str = "proband",
) -> dict[str, Any]:
    phenotype: dict[str, Any] = {
        "phrase": phrase,
        "category": category,
        "chunk_ids": [chunk_id],
        "evidence_text": phrase,
        "experiencer": experiencer,
    }
    if negated_qualifier is not None:
        phenotype["negated_qualifier"] = negated_qualifier
    return phenotype


def _config() -> LLMPipelineConfig:
    return LLMPipelineConfig(
        provider="fake",
        model="fake",
        mode="two_phase",
        language="en",
    )


# The source phrase X ("rash") exact-matches its candidate term_name, so it
# resolves LOCALLY -> no phase-2 mapping LLM call is required.
RASH_CANDIDATE = {"hpo_id": "HP:0000988", "term_name": "rash", "score": 0.95}
# The qualifier Y ("fever") maps ABOVE the default floor (0.35).
FEVER_CANDIDATE_ABOVE = {"hpo_id": "HP:0001945", "term_name": "Fever", "score": 0.9}
# The qualifier Y ("fever") maps BELOW the default floor (0.35).
FEVER_CANDIDATE_BELOW = {"hpo_id": "HP:0001945", "term_name": "Fever", "score": 0.2}


def _phase1_only_provider() -> FakeProvider:
    """Phase 1 emits one proband finding X="rash" with negated_qualifier="fever".

    X resolves locally, so no mapping response is needed.
    """
    return FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        _grounded_phenotype(
                            "rash",
                            "Abnormal",
                            chunk_id=1,
                            negated_qualifier="fever",
                        ),
                    ]
                },
                "request_count": 1,
            },
        ]
    )


def test_qualifier_above_floor_emits_generated_excluded_finding() -> None:
    """Case A: a resolved proband finding carrying negated_qualifier="fever"
    where "fever" maps above the floor yields a GENERATED excluded finding.

    The generated finding is present, present in ``result.terms`` alongside the
    source finding, carries ``assertion == "negated"``,
    ``match_method == "negated_qualifier_derived"``,
    ``qualifier_surface_text == "fever"``, ``experiencer == "proband"``, and --
    critically (F5) -- inherits the source finding's chunk_ids in its evidence
    records.
    """
    provider = _phase1_only_provider()
    tool_executor = PhraseAwareToolExecutor(
        {
            "rash": [RASH_CANDIDATE],
            "fever": [FEVER_CANDIDATE_ABOVE],
        }
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=1,
    )

    result = pipeline.run(
        text="Patient had a rash without fever.",
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had a rash without fever."}],
        config=_config(),
    )

    assert isinstance(result, LLMExtractionResult)

    # The source proband finding still resolves into result.terms.
    source = next((t for t in result.terms if t.term_id == "HP:0000988"), None)
    assert source is not None
    assert source.experiencer == "proband"

    # A generated excluded finding for the qualifier "fever" appears.
    generated = next((t for t in result.terms if t.term_id == "HP:0001945"), None)
    assert generated is not None
    assert generated.assertion == "negated"
    assert generated.experiencer == "proband"
    assert generated.match_method == "negated_qualifier_derived"
    assert generated.qualifier_surface_text == "fever"
    assert generated.label == "Fever"
    assert generated.confidence is not None
    assert generated.confidence >= pipeline.similarity_threshold
    assert generated.score == FEVER_CANDIDATE_ABOVE["score"]

    # F5: the generated term inherits the source finding's chunk_ids so it
    # survives the aggregation service boundary.
    generated_chunk_ids = [
        chunk_id
        for record in generated.evidence_records
        for chunk_id in record.chunk_ids
    ]
    assert 1 in generated_chunk_ids


def test_qualifier_below_floor_keeps_metadata_and_emits_no_term() -> None:
    """Case B: when "fever" maps BELOW the floor, no term is generated and the
    source finding's ``negated_qualifier`` metadata string is retained."""
    provider = _phase1_only_provider()
    tool_executor = PhraseAwareToolExecutor(
        {
            "rash": [RASH_CANDIDATE],
            "fever": [FEVER_CANDIDATE_BELOW],
        }
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=1,
    )

    result = pipeline.run(
        text="Patient had a rash without fever.",
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had a rash without fever."}],
        config=_config(),
    )

    # Only the source finding resolves; no generated excluded term.
    assert {t.term_id for t in result.terms} == {"HP:0000988"}

    source = next(t for t in result.terms if t.term_id == "HP:0000988")
    # The metadata string is retained on the source finding.
    assert source.negated_qualifier == "fever"


def test_qualifier_unmappable_keeps_metadata_and_emits_no_term() -> None:
    """Case B (variant): when the qualifier retrieves NO candidate at all, no
    term is generated and the source finding's metadata string is retained."""
    provider = _phase1_only_provider()
    tool_executor = PhraseAwareToolExecutor(
        {
            "rash": [RASH_CANDIDATE],
            # "fever" intentionally absent -> zero candidates retrieved.
        }
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=1,
    )

    result = pipeline.run(
        text="Patient had a rash without fever.",
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had a rash without fever."}],
        config=_config(),
    )

    assert {t.term_id for t in result.terms} == {"HP:0000988"}
    source = next(t for t in result.terms if t.term_id == "HP:0000988")
    assert source.negated_qualifier == "fever"
