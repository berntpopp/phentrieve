"""Temporality axis (current | historical | unknown) from phase 1 to the terms."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.prompts.loader import get_prompt
from phentrieve.llm.types import LLMGroundedExtractedPhenotype, LLMPipelineConfig
from tests.unit.llm.test_pipeline import (
    FakeProvider,
    PhraseKeyedToolExecutor,
    grounded_phenotype,
)

TEXT = "The patient had asthma as a child. Now he has asthma again."


def _run(*phenotypes):
    provider = FakeProvider(responses=[{"parsed": {"phenotypes": list(phenotypes)}}])
    tool_executor = PhraseKeyedToolExecutor(
        {"asthma": [{"hpo_id": "HP:0002099", "term_name": "asthma", "score": 0.95}]}
    )
    pipeline = pipeline_module.TwoPhaseLLMPipeline(
        provider=provider, tool_executor=tool_executor
    )
    return pipeline.run(
        text=TEXT,
        grounded_chunks=[{"chunk_id": 1, "text": TEXT}],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )


def _asthma(temporality=None, evidence_text="asthma"):
    phenotype = grounded_phenotype("asthma", "Abnormal", evidence_text=evidence_text)
    if temporality is not None:
        phenotype["temporality"] = temporality
    return phenotype


def test_schema_accepts_the_three_temporality_values_only():
    for value in ("current", "historical", "unknown"):
        LLMGroundedExtractedPhenotype(
            phrase="asthma", category="Abnormal", chunk_ids=[1], temporality=value
        )
    with pytest.raises(ValidationError):
        LLMGroundedExtractedPhenotype(
            phrase="asthma", category="Abnormal", chunk_ids=[1], temporality="past"
        )


def test_prompt_defines_temporality():
    prompt = get_prompt("two_phase", "en").render_system_prompt()

    assert "temporality" in prompt
    assert "historical" in prompt
    assert "no longer present" in prompt


def test_model_temporality_reaches_the_term():
    result = _run(_asthma("historical"))

    assert [t.temporality for t in result.terms] == ["historical"]


def test_missing_temporality_stays_unknown():
    result = _run(_asthma())

    assert [t.temporality for t in result.terms] == ["unknown"]


def test_historical_and_current_mentions_stay_separate_terms():
    result = _run(
        _asthma("historical", evidence_text="asthma as a child"),
        _asthma("current", evidence_text="asthma again"),
    )

    assert sorted(t.temporality for t in result.terms) == ["current", "historical"]
