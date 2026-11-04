"""Temporality axis (current | historical | unknown) from phase 1 to the terms."""

from __future__ import annotations

import pytest

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.pipeline_phase1 import phase1_extraction_dedup_key
from phentrieve.llm.prompts.loader import get_prompt
from phentrieve.llm.types import (
    LLMExtractedPhenotype,
    LLMGroundedExtractedPhenotype,
    LLMPipelineConfig,
)
from tests.unit.llm.test_pipeline import (
    FakeProvider,
    PhraseKeyedToolExecutor,
    grounded_phenotype,
)

TEXT = "The patient had migraine as a child. Now he has migraine again."


def _run(*phenotypes):
    provider = FakeProvider(responses=[{"parsed": {"phenotypes": list(phenotypes)}}])
    tool_executor = PhraseKeyedToolExecutor(
        {"migraine": [{"hpo_id": "HP:0002076", "term_name": "migraine", "score": 0.95}]}
    )
    pipeline = pipeline_module.TwoPhaseLLMPipeline(
        provider=provider, tool_executor=tool_executor
    )
    return pipeline.run(
        text=TEXT,
        grounded_chunks=[{"chunk_id": 1, "text": TEXT}],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )


def _migraine(temporality=None, evidence_text="migraine"):
    phenotype = grounded_phenotype("migraine", "Abnormal", evidence_text=evidence_text)
    if temporality is not None:
        phenotype["temporality"] = temporality
    return phenotype


def test_schema_accepts_the_three_temporality_values():
    for value in ("current", "historical", "unknown"):
        phenotype = LLMGroundedExtractedPhenotype(
            phrase="migraine", category="Abnormal", chunk_ids=[1], temporality=value
        )
        assert phenotype.temporality == value


@pytest.mark.parametrize(
    "schema", [LLMGroundedExtractedPhenotype, LLMExtractedPhenotype]
)
def test_unexpected_temporality_is_coerced_to_unknown(schema):
    # One stray value from a JSON-mode provider must not fail the whole group.
    extra = {"chunk_ids": [1]} if schema is LLMGroundedExtractedPhenotype else {}
    for value in ("past", None, 3):
        phenotype = schema(
            phrase="migraine", category="Abnormal", temporality=value, **extra
        )
        assert phenotype.temporality == "unknown"
    assert (
        schema(
            phrase="migraine", category="Abnormal", temporality=" Historical ", **extra
        ).temporality
        == "historical"
    )


def test_missing_and_unknown_temporality_have_the_same_phase1_key():
    item = {"phrase": "migraine", "chunk_ids": [1], "evidence_text": "migraine"}

    assert phase1_extraction_dedup_key(item) == phase1_extraction_dedup_key(
        {**item, "temporality": "unknown"}
    )


def test_prompt_defines_temporality():
    prompt = get_prompt("two_phase", "en").render_system_prompt()

    assert "temporality" in prompt
    assert "historical" in prompt
    assert "no longer present" in prompt


def test_prompt_keeps_non_phenotype_information_out_of_the_patient_categories():
    prompt = " ".join(get_prompt("two_phase", "en").render_system_prompt().split())

    assert (
        "Exception: non-phenotype information (onset, inheritance, age, measurement) "
        "always gets category Other"
    ) in prompt


def test_model_temporality_reaches_the_term():
    result = _run(_migraine("historical"))

    assert [t.temporality for t in result.terms] == ["historical"]


def test_missing_temporality_stays_unknown():
    result = _run(_migraine())

    assert [t.temporality for t in result.terms] == ["unknown"]


def test_historical_and_current_mentions_stay_separate_terms():
    result = _run(
        _migraine("historical", evidence_text="migraine as a child"),
        _migraine("current", evidence_text="migraine again"),
    )

    assert sorted(t.temporality for t in result.terms) == ["current", "historical"]


def test_unknown_mention_folds_into_the_term_with_a_known_temporality():
    # The UI does not show temporality, so an "unknown" twin of the same term
    # would look like an exact duplicate row.
    result = _run(
        _migraine("historical", evidence_text="migraine as a child"),
        _migraine(evidence_text="migraine again"),
    )

    assert [t.temporality for t in result.terms] == ["historical"]
    assert len(result.terms[0].evidence_records) == 2
