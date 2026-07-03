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
from phentrieve.llm.pipeline_phase2 import (
    compact_mapping_item,
    phenotype_from_candidate,
)
from phentrieve.llm.prompts.loader import get_prompt
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import AnnotationMode, LLMExtractedPhenotype

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


class _CannedPhase1Provider(LLMProvider):
    """Provider stub whose structured prompt returns preset phase-1 phenotype
    objects, so the parse step in ``_extract_phase1_phenotypes`` can be driven
    without a model.

    Crucially it returns the constructed Pydantic phenotypes verbatim, so each
    phenotype's ``model_fields_set`` is preserved -- that set is what
    distinguishes a MODEL-EMITTED assertion from a SCHEMA-DEFAULTED one.
    """

    class _Response:
        def __init__(self, phenotypes):
            self.phenotypes = list(phenotypes)

    def __init__(self, phenotypes):
        super().__init__()
        self._phenotypes = list(phenotypes)
        self.last_usage = {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
        self.last_request_count = 1

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
        return self._Response(self._phenotypes)


def _parse_phase1(*phenotypes):
    """Drive the phase-1 parse step over the given wire phenotype objects and
    return the parsed item dicts (the first element of the method's tuple)."""
    pipeline = TwoPhaseLLMPipeline(
        provider=_CannedPhase1Provider(phenotypes),
        tool_executor=_StubToolExecutor(),
    )
    parsed, _usage, _requests, _debug = pipeline._extract_phase1_phenotypes(
        text="clinical note",
        grounded_chunks=[],
        extraction_prompt=get_prompt(AnnotationMode.TWO_PHASE, "en"),
    )
    return parsed


def test_parse_threads_explicit_model_assertion():
    """(a) The model EXPLICITLY emitted ``assertion="absent"`` -> it IS in
    ``model_fields_set`` -> the parsed item threads the raw wire value so the
    downstream model-wins resolution honors it."""
    explicit = LLMExtractedPhenotype(
        phrase="hearing loss", category="Abnormal", assertion="absent"
    )
    assert "assertion" in explicit.model_fields_set

    parsed = _parse_phase1(explicit)

    assert len(parsed) == 1
    assert parsed[0]["assertion"] == "absent"


def test_parse_treats_schema_default_assertion_as_unset():
    """(b) B1 regression crux: the model OMITTED assertion (a category-only
    ``Normal`` finding, as the extraction few-shots teach it to emit) -> the
    schema default ``"present"`` is NOT in ``model_fields_set`` -> the parsed
    item must thread ``None``, NOT the defaulted ``"present"``.

    Threading the defaulted ``"present"`` is the bug: it makes the downstream
    ``item.get("assertion") is not None`` model-wins guard flip a ruled-out
    finding to present."""
    silent = LLMExtractedPhenotype(phrase="normal vision", category="Normal")
    # The default is materialised on the attribute, but the field was never
    # explicitly set by the "model".
    assert silent.assertion == "present"
    assert "assertion" not in silent.model_fields_set

    parsed = _parse_phase1(silent)

    assert len(parsed) == 1
    assert parsed[0]["assertion"] is None


def test_model_silent_normal_finding_falls_back_to_negated_end_to_end():
    """End-to-end: a category-``Normal`` model-silent finding, parsed and then
    run through ``phenotype_from_candidate``, resolves to the pipeline polarity
    ``negated`` via the category fallback -- NOT ``present``. This is the
    behavior the B1 fix restores versus the silent present-flip regression."""
    silent = LLMExtractedPhenotype(phrase="normal vision", category="Normal")

    parsed = _parse_phase1(silent)
    phenotype = phenotype_from_candidate(
        item=parsed[0],
        candidate={"hpo_id": "HP:0000001", "term_name": "All", "score": 0.9},
    )

    assert phenotype.assertion == "negated"
    assert phenotype.category == "normal"
