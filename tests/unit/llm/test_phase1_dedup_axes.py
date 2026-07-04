"""Phase-1 dedup + fuzzy-merge must be experiencer/assertion-aware (B2).

B1 made ``experiencer`` and ``assertion`` orthogonal axes (polarity and
family-ness no longer live in ``category``). The Phase-1 dedup+merge runs
BEFORE the B2 family partition, so if it is axis-blind it can silently
collapse two genuinely DIFFERENT findings that share phrase/category/chunk/
evidence but differ only in experiencer (proband vs family_history) or in
assertion (present vs absent). These tests fence that behavior.
"""

from __future__ import annotations

from typing import Any

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.pipeline_phase1 import phase1_extraction_dedup_key
from phentrieve.llm.provider import LLMProvider, ToolExecutor
from phentrieve.llm.types import LLMPipelineConfig

TwoPhaseLLMPipeline = pipeline_module.TwoPhaseLLMPipeline


class _StructuredProvider(LLMProvider):
    """Minimal provider stub driving the ungrouped structured phase-1 path."""

    provider_name = "gemini"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.last_usage: dict[str, int] = {}
        self.last_request_count = 0

    def complete(self, messages):  # pragma: no cover - not used on this path
        raise RuntimeError("unused")

    def run_structured_prompt(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
        max_output_tokens=None,
    ):
        response = self.responses.pop(0)
        self.last_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        self.last_request_count = 1
        parsed = response_model.model_validate(response["parsed"])
        self.last_structured_payload = parsed.model_dump(mode="json")
        return parsed

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        chunk_count = max(user_prompt.count("chunk_id="), 1)
        return {
            "prompt_tokens": chunk_count * 10,
            "completion_tokens": 0,
            "total_tokens": chunk_count * 10,
        }


class _PhraseKeyedToolExecutor(ToolExecutor):
    """Retriever stub returning candidates keyed by the queried phrase."""

    def __init__(self, candidates_by_phrase: dict[str, list[dict[str, Any]]]) -> None:
        super().__init__()
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


def _phase1_item(
    *,
    phrase: str = "hearing loss",
    category: str = "abnormal",
    chunk_ids: list[int] | None = None,
    evidence_text: str = "hearing loss",
    start_char: int | None = None,
    end_char: int | None = None,
    experiencer: str | None = "proband",
    assertion: str | None = None,
) -> dict[str, Any]:
    """Build a phase-1 extraction dict as produced after structured parsing."""
    return {
        "phrase": phrase,
        "category": category,
        "chunk_ids": list(chunk_ids or [1]),
        "evidence_text": evidence_text,
        "start_char": start_char,
        "end_char": end_char,
        "experiencer": experiencer,
        "assertion": assertion,
    }


# ---------------------------------------------------------------------------
# (a) EXACT-DUP KEY: the dedup key must separate the assertion + experiencer axes
# ---------------------------------------------------------------------------


def test_dedup_key_separates_differing_assertion() -> None:
    present = _phase1_item(assertion="present")
    absent = _phase1_item(assertion="absent")
    assert phase1_extraction_dedup_key(present) != phase1_extraction_dedup_key(absent)


def test_dedup_key_separates_differing_experiencer() -> None:
    proband = _phase1_item(experiencer="proband")
    family = _phase1_item(experiencer="family_history")
    assert phase1_extraction_dedup_key(proband) != phase1_extraction_dedup_key(family)


def test_dedup_key_is_stable_for_identical_axes() -> None:
    first = _phase1_item(experiencer="proband", assertion="present")
    second = _phase1_item(experiencer="proband", assertion="present")
    assert phase1_extraction_dedup_key(first) == phase1_extraction_dedup_key(second)


def test_dedup_key_silent_assertion_is_uniform() -> None:
    # Backward-compat: two items with no explicit assertion contribute a uniform
    # "" for the assertion slot, so their relative equality is unchanged.
    first = _phase1_item(assertion=None)
    second = _phase1_item(assertion=None)
    assert phase1_extraction_dedup_key(first) == phase1_extraction_dedup_key(second)


# ---------------------------------------------------------------------------
# (b) FUZZY MERGE REFUSAL: _should_merge refuses when an axis differs
# ---------------------------------------------------------------------------


def test_should_merge_refuses_differing_experiencer() -> None:
    proband = _phase1_item(experiencer="proband")
    family = _phase1_item(experiencer="family_history")
    assert not TwoPhaseLLMPipeline._should_merge_phase1_extractions(proband, family)


def test_should_merge_refuses_differing_assertion() -> None:
    present = _phase1_item(assertion="present")
    absent = _phase1_item(assertion="absent")
    assert not TwoPhaseLLMPipeline._should_merge_phase1_extractions(present, absent)


def test_should_merge_allows_identical_axes() -> None:
    # Genuine fragments of the SAME finding (same axes) must still merge so the
    # common dedup/consolidation case is unchanged.
    existing = _phase1_item(
        evidence_text="hearing loss", experiencer="proband", assertion="present"
    )
    incoming = _phase1_item(
        evidence_text="bilateral hearing loss",
        experiencer="proband",
        assertion="present",
    )
    assert TwoPhaseLLMPipeline._should_merge_phase1_extractions(existing, incoming)


def test_should_merge_allows_silent_assertion_duplicates() -> None:
    # A genuine silent (None) duplicate pair -- same phrase/experiencer, both
    # model-silent on assertion -- still merges.
    existing = _phase1_item(assertion=None)
    incoming = _phase1_item(assertion=None)
    assert TwoPhaseLLMPipeline._should_merge_phase1_extractions(existing, incoming)


# ---------------------------------------------------------------------------
# (c) END-TO-END: co-located proband + family "hearing loss" both survive
# ---------------------------------------------------------------------------


def test_colocated_proband_and_family_mentions_are_not_collapsed() -> None:
    """A proband "hearing loss" and a family "hearing loss" co-located in ONE
    chunk (no spans, equal evidence, same phrase/category) differ ONLY in
    experiencer. The axis-blind fuzzy merge would drop one; the axis-aware
    merge keeps both so the proband finding lands in ``result.terms`` and the
    family finding in ``result.family_history_findings``.
    """
    provider = _StructuredProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "hearing loss",
                            "category": "Abnormal",
                            "chunk_ids": [1],
                            "evidence_text": "hearing loss",
                            "experiencer": "proband",
                        },
                        {
                            "phrase": "hearing loss",
                            "category": "Abnormal",
                            "chunk_ids": [1],
                            "evidence_text": "hearing loss",
                            "experiencer": "family_history",
                        },
                    ]
                }
            }
        ]
    )
    tool_executor = _PhraseKeyedToolExecutor(
        {
            "hearing loss": [
                {"hpo_id": "HP:0000365", "term_name": "hearing loss", "score": 0.95}
            ]
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="the patient has hearing loss; his mother also has hearing loss",
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "the patient has hearing loss; his mother also has hearing loss",
            }
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    proband_terms = {term.term_id: term for term in result.terms}
    family_terms = {term.term_id: term for term in result.family_history_findings}

    # The proband mention survives as a proband term.
    assert "HP:0000365" in proband_terms
    assert proband_terms["HP:0000365"].experiencer == "proband"

    # The family mention survives as a family-history finding (NOT dropped, NOT
    # re-tagged into result.terms).
    assert "HP:0000365" in family_terms
    assert family_terms["HP:0000365"].experiencer == "family_history"
    assert family_terms["HP:0000365"].category == "family_history"
