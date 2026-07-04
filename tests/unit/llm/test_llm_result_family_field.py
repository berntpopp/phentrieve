"""Tests for `LLMExtractionResult.family_history_findings` (extraction contract
v2, block B2).

Task 6 built a local ``resolved_family: list[LLMPhenotype]`` inside
``TwoPhaseLLMPipeline.run()`` and exposed it ONLY via a temporary instance
attribute (``pipeline._last_resolved_family``) as a test seam. Task 7 attaches
that same list to the result object as ``family_history_findings`` and removes
the instance-attribute seam (a reviewer flagged it as a concurrency race if the
pipeline instance is reused across concurrent ``run()`` calls).

These tests pin two things:

1. ``LLMExtractionResult`` accepts and defaults ``family_history_findings`` to
   an empty list when constructed with only ``terms`` + ``meta`` (schema-level
   contract, independent of the pipeline).
2. Using the same stubs as ``test_family_resolve.py``, a family-history phrase
   lands in ``result.family_history_findings`` and NOT in ``result.terms``.
"""

from __future__ import annotations

from typing import Any

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import (
    LLMExtractionResult,
    LLMMeta,
    LLMPipelineConfig,
)

TwoPhaseLLMPipeline = pipeline_module.TwoPhaseLLMPipeline


def test_llm_extraction_result_defaults_family_history_findings_to_empty_list() -> None:
    """Constructing with only ``terms`` + ``meta`` defaults the new field to
    an empty list, matching the existing ``default_factory=list`` style used
    by ``terms``."""
    result = LLMExtractionResult(
        terms=[], meta=LLMMeta(llm_model="fake", llm_mode="two_phase")
    )

    assert result.family_history_findings == []


class FakeProvider(LLMProvider):
    """Structured-prompt provider stub (mirrors tests/unit/llm/test_pipeline)."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__()
        self.responses = list(responses)

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
    """Retriever stub that returns candidates keyed by the queried phrase."""

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


def test_family_phrase_lands_in_result_family_history_findings_not_terms() -> None:
    """A phrase tagged ``experiencer="family_history"`` resolves into
    ``result.family_history_findings`` and is ABSENT from ``result.terms``."""
    provider = FakeProvider(
        responses=[
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
            {
                "parsed": {"phrase": "recurrent seizures", "hpo_id": "HP:0001250"},
                "request_count": 1,
            },
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

    proband_ids = {term.term_id for term in result.terms}
    family_ids = {term.term_id for term in result.family_history_findings}

    assert "HP:0001657" not in proband_ids
    assert "HP:0001250" in proband_ids
    assert family_ids == {"HP:0001657"}

    family_term = next(
        term for term in result.family_history_findings if term.term_id == "HP:0001657"
    )
    assert family_term.experiencer == "family_history"


def test_proband_only_run_leaves_family_history_findings_empty() -> None:
    """No-regression guard: a proband-only run produces an EMPTY
    ``family_history_findings`` list on the result."""
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
    assert result.family_history_findings == []
