"""Family-experiencer resolution tests (extraction contract v2, block B2).

These tests pin the requirement that phrases the model attributes to a relative
(``experiencer == "family_history"``) -- or that carry the legacy
``Family_History`` category -- are PARTITIONED out of the proband set BEFORE the
actionable filter and resolved through the SAME retrieval + mapping path as
proband phrases, producing a SEPARATE ``resolved_family`` list.

Two invariants matter:

1. Family phenotypes must NOT leak into ``result.terms`` (the proband set). A
   relative's mention mapped as a proband HPO term is the exact
   assertion/experiencer-boundary bug this effort targets.
2. The family LLM mapping pass must be COUNTED in the meta accounting
   (``request_count`` and ``phase_request_counts``); a family pass that silently
   escapes the accumulators underreports cost.

The retriever and the phase-2 LLM mapping provider are stubbed, so no model /
Gemini / network is required (mirrors the existing tests/unit/llm pipeline
stubs).

``resolved_family`` is exposed for Task 6 via the pipeline test hook
``TwoPhaseLLMPipeline._last_resolved_family`` (a list[LLMPhenotype] set at the
end of ``run()``). Task 7 will attach the same local list to the result object;
until then this hook keeps Task 6 independently testable.
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
    """Retriever stub that returns candidates keyed by the queried phrase.

    ``_retrieve_candidates`` aligns the returned batch results positionally to
    the expanded query list, so returning ``{"phrase": phrase, "candidates":
    ...}`` per input phrase lets a proband phrase and a family phrase retrieve
    DIFFERENT candidate sets (unlike a single canned batch shared across every
    call).
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
    experiencer: str | None = None,
) -> dict[str, Any]:
    phenotype: dict[str, Any] = {
        "phrase": phrase,
        "category": category,
        "chunk_ids": [chunk_id],
        "evidence_text": phrase,
    }
    if experiencer is not None:
        phenotype["experiencer"] = experiencer
    return phenotype


def _config() -> LLMPipelineConfig:
    return LLMPipelineConfig(
        provider="fake",
        model="fake",
        mode="two_phase",
        language="en",
    )


PROBAND_CANDIDATE = {
    "hpo_id": "HP:0001250",
    "term_name": "Seizure",
    "score": 0.9,
}
FAMILY_CANDIDATE = {
    "hpo_id": "HP:0001657",
    "term_name": "Prolonged QT interval",
    "score": 0.88,
}


def test_family_experiencer_phrase_resolved_off_the_proband_set() -> None:
    """A phrase the model tags ``experiencer="family_history"`` (even with a
    legacy ``Abnormal`` category that WOULD otherwise pass the actionable
    filter) is partitioned to the family path: it resolves into
    ``_last_resolved_family`` and is ABSENT from the proband ``result.terms``.
    The proband phrase still resolves into ``result.terms``. The family mapping
    LLM pass is counted in the meta accounting.
    """
    provider = FakeProvider(
        responses=[
            # Phase 1: one proband phrase + one family-history phrase. The
            # family phrase carries category "Abnormal" on purpose -- absent a
            # partition it would pass the actionable filter and leak into the
            # proband set.
            {
                "parsed": {
                    "phenotypes": [
                        _grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_id=1,
                            experiencer="proband",
                        ),
                        _grounded_phenotype(
                            "long QT syndrome",
                            "Abnormal",
                            chunk_id=2,
                            experiencer="family_history",
                        ),
                    ]
                },
                "request_count": 1,
            },
            # Proband mapping pass (batch size 1 -> single-item mapping).
            {
                "parsed": {"phrase": "recurrent seizures", "hpo_id": "HP:0001250"},
                "request_count": 1,
            },
            # Family mapping pass -- MUST be counted in the accounting.
            {
                "parsed": {"phrase": "long qt syndrome", "hpo_id": "HP:0001657"},
                "request_count": 1,
            },
        ]
    )
    tool_executor = PhraseAwareToolExecutor(
        {
            "recurrent seizures": [PROBAND_CANDIDATE],
            "long QT syndrome": [FAMILY_CANDIDATE],
        }
    )
    # mapping_batch_size=1 forces one single-item mapping prompt per unresolved
    # phrase, so the proband and family passes are directly comparable.
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=1,
    )

    result = pipeline.run(
        text="Patient had recurrent seizures. Mother has long QT syndrome.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Patient had recurrent seizures."},
            {"chunk_id": 2, "text": "Mother has long QT syndrome."},
        ],
        config=_config(),
    )

    assert isinstance(result, LLMExtractionResult)

    proband_ids = {term.term_id for term in result.terms}
    # No proband leakage: the family HPO id must NOT be in the proband set.
    assert "HP:0001657" not in proband_ids
    # The proband phrase still resolves into result.terms.
    assert "HP:0001250" in proband_ids

    # Family phrase resolved via the SAME retrieval + mapping path, into the
    # separate resolved_family list (exposed via the Task-6 test hook).
    resolved_family = pipeline._last_resolved_family
    family_ids = {term.term_id for term in resolved_family}
    assert family_ids == {"HP:0001657"}
    family_term = next(term for term in resolved_family if term.term_id == "HP:0001657")
    assert family_term.experiencer == "family_history"

    # ACCOUNTING: request_count covers phase 1 (1) + proband mapping (1) +
    # family mapping (1); phase_request_counts records BOTH mapping passes.
    assert result.meta.request_count == 3
    assert result.meta.phase_request_counts["phase2b_llm_requests"] == 1
    assert result.meta.phase_request_counts["family_phase2b_llm_requests"] == 1


def test_category_only_family_finding_is_partitioned_and_resolved() -> None:
    """A category-only ``Family_History`` finding (experiencer left at the schema
    default "proband") is caught by the partition's CATEGORY clause. It was
    previously DROPPED by the actionable filter (family_history is not
    actionable); B2 now resolves it into ``resolved_family`` while keeping it out
    of the proband ``result.terms``.
    """
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        _grounded_phenotype(
                            "recurrent seizures", "Abnormal", chunk_id=1
                        ),
                        # Category-only family finding; no experiencer set.
                        _grounded_phenotype(
                            "hearing loss", "Family_History", chunk_id=2
                        ),
                    ]
                },
                "request_count": 1,
            },
        ]
    )
    tool_executor = PhraseAwareToolExecutor(
        {
            # Exact term match -> both resolve LOCALLY (no mapping LLM call).
            "recurrent seizures": [
                {
                    "hpo_id": "HP:0001250",
                    "term_name": "recurrent seizures",
                    "score": 0.95,
                }
            ],
            "hearing loss": [
                {"hpo_id": "HP:0000365", "term_name": "hearing loss", "score": 0.95}
            ],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="Patient had recurrent seizures. The mother has hearing loss.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Patient had recurrent seizures."},
            {"chunk_id": 2, "text": "The mother has hearing loss."},
        ],
        config=_config(),
    )

    proband_ids = {term.term_id for term in result.terms}
    assert proband_ids == {"HP:0001250"}
    assert "HP:0000365" not in proband_ids

    family_ids = {term.term_id for term in pipeline._last_resolved_family}
    assert family_ids == {"HP:0000365"}
    family_term = next(
        term for term in pipeline._last_resolved_family if term.term_id == "HP:0000365"
    )
    assert family_term.experiencer == "family_history"
    assert family_term.category == "family_history"

    # The family retrieval happened through the real retrieval path: a SECOND
    # query batch scoped to the family phrase.
    assert tool_executor.queries[0]["phrases"] == ["recurrent seizures"]
    assert tool_executor.queries[1]["phrases"] == ["hearing loss"]


def test_resolve_items_helper_resolves_family_items_directly() -> None:
    """The shared resolver helper ``_resolve_items`` resolves family items the
    same way it resolves proband items, so Task 6 can drive the family path
    directly without the full run() wrapper."""
    provider = FakeProvider(
        responses=[
            {
                "parsed": {"phrase": "long qt syndrome", "hpo_id": "HP:0001657"},
                "request_count": 1,
            },
        ]
    )
    tool_executor = PhraseAwareToolExecutor({"long QT syndrome": [FAMILY_CANDIDATE]})
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=1,
    )

    family_items = [
        {
            "phrase": "long QT syndrome",
            "category": "abnormal",
            "negated_qualifier": None,
            "chunk_ids": [2],
            "evidence_text": "long QT syndrome",
            "start_char": None,
            "end_char": None,
            "experiencer": "family_history",
            "assertion": None,
        }
    ]

    outcome = pipeline._resolve_items(
        family_items,
        grounded_chunks=[{"chunk_id": 2, "text": "Mother has long QT syndrome."}],
        language="en",
    )

    resolved_ids = {term.term_id for term in outcome.resolved}
    assert resolved_ids == {"HP:0001657"}
    assert outcome.resolved[0].experiencer == "family_history"
    # The helper reports the mapping LLM request so run() can fold it into the
    # shared accounting.
    assert outcome.request_count == 1
    assert outcome.phase2b_llm_requests == 1


def test_proband_only_run_leaves_family_hook_empty() -> None:
    """No-regression guard: a proband-only run produces an EMPTY resolved_family
    hook and adds no family_* meta keys, so proband meta stays byte-identical."""
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        _grounded_phenotype(
                            "recurrent seizures", "Abnormal", chunk_id=1
                        )
                    ]
                },
                "request_count": 1,
            },
            {
                "parsed": {"phrase": "recurrent seizures", "hpo_id": "HP:0001250"},
                "request_count": 1,
            },
        ]
    )
    tool_executor = PhraseAwareToolExecutor({"recurrent seizures": [PROBAND_CANDIDATE]})
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=1,
    )

    result = pipeline.run(
        text="Patient had recurrent seizures.",
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
        config=_config(),
    )

    assert {term.term_id for term in result.terms} == {"HP:0001250"}
    assert pipeline._last_resolved_family == []
    assert not any(
        key.startswith("family_") for key in result.meta.phase_request_counts
    )
    assert not any(key.startswith("family_") for key in result.meta.phase_counts)
    assert "family" not in result.meta.trace
