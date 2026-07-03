"""Axis threading tests (extraction contract v2, block B1).

These tests pin the requirement that the model's own RAW WIRE ``experiencer``
(proband | family_history | other) and ``assertion`` (present | absent |
uncertain) survive the phase-1 -> retrieve -> mapping-payload chain unchanged,
so a later task can consume the model's assertion as polarity.

The retriever and the LLM provider are stubbed, so no model / Gemini / network
is required.
"""

from __future__ import annotations

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.pipeline_phase1 import phase1_extraction_dedup_key
from phentrieve.llm.pipeline_phase2 import compact_mapping_item
from phentrieve.llm.provider import LLMProvider

TwoPhaseLLMPipeline = pipeline_module.TwoPhaseLLMPipeline


class _StubProvider(LLMProvider):
    """Minimal provider stub; the axis-threading tests never call the model."""

    def complete(self, messages):  # pragma: no cover - never invoked here
        raise AssertionError("provider.complete must not be called")

    def run_structured_prompt(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
        max_output_tokens=None,
    ):  # pragma: no cover - never invoked here
        raise AssertionError("provider.run_structured_prompt must not be called")


class _StubToolExecutor:
    """Retriever stub that returns no candidates.

    The axes are attached from the parsed item itself in the per-item rebuild
    loop of ``_retrieve_candidates``; they do not depend on retrieval output, so
    an empty batch keeps the test focused purely on axis threading.
    """

    def __init__(self):
        self.queries = []

    def query_batch_hpo_terms(self, *, phrases, language, n_results):
        self.queries.append(
            {"phrases": list(phrases), "language": language, "n_results": n_results}
        )
        return []


def _actionable_item(**overrides):
    item = {
        "phrase": "seizures",
        "category": "abnormal",
        "negated_qualifier": None,
        "chunk_ids": [1],
        "evidence_text": "recurrent seizures",
        "start_char": 0,
        "end_char": 10,
        "experiencer": "family_history",
        "assertion": "absent",
    }
    item.update(overrides)
    return item


def test_retrieve_candidates_preserves_experiencer_and_assertion():
    """Crux: the per-item rebuild in ``_retrieve_candidates`` must keep the
    model's raw experiencer + assertion on every emerging candidate dict."""
    pipeline = TwoPhaseLLMPipeline(
        provider=_StubProvider(),
        tool_executor=_StubToolExecutor(),
    )

    results = pipeline._retrieve_candidates(
        actionable=[_actionable_item()],
        grounded_chunks=[],
        language="en",
    )

    assert len(results) == 1
    rebuilt = results[0]
    assert rebuilt["experiencer"] == "family_history"
    assert rebuilt["assertion"] == "absent"


def test_compact_mapping_item_includes_experiencer_and_assertion():
    """Mapping payload must carry the axes so the phase-2 LLM (and Task 5's
    ``phenotype_from_candidate``) can read the model's assertion + experiencer."""
    item = {
        "phrase": "seizures",
        "category": "abnormal",
        "grounded_context": {},
        "candidates": [],
        "experiencer": "family_history",
        "assertion": "absent",
    }

    compact = compact_mapping_item(item)

    assert compact["experiencer"] == "family_history"
    assert compact["assertion"] == "absent"


def test_phase1_dedup_key_separates_experiencer():
    """A proband vs family mention of an otherwise-identical phrase must not
    collapse into one item before retrieval: differing experiencer -> different
    dedup key."""
    proband = _actionable_item(experiencer="proband")
    family = _actionable_item(experiencer="family_history")

    assert phase1_extraction_dedup_key(proband) != phase1_extraction_dedup_key(family)


def test_phase1_dedup_key_backward_compatible_without_experiencer():
    """Determinism guardrail: items with no experiencer axis (deterministic
    extractor path) still produce identical dedup keys, so existing behavior is
    unchanged."""
    a = _actionable_item()
    b = _actionable_item()
    a.pop("experiencer")
    b.pop("experiencer")

    assert phase1_extraction_dedup_key(a) == phase1_extraction_dedup_key(b)

    # None and absent must key identically to a blank string too.
    c = _actionable_item(experiencer=None)
    assert phase1_extraction_dedup_key(a) == phase1_extraction_dedup_key(c)
