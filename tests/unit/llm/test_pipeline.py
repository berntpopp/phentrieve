import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from phentrieve.llm.pipeline import LLMPipelinePhaseError, TwoPhaseLLMPipeline
from phentrieve.llm.prompts.loader import (
    PromptTemplate,
    get_mapping_prompt,
    get_prompt,
    load_prompt_template,
)
from phentrieve.llm.provider import LLMProvider, ToolExecutor
from phentrieve.llm.types import (
    AnnotationMode,
    LLMExtractionResult,
    LLMGroundedExtractedPhenotype,
    LLMPhenotype,
    LLMPhenotypeEvidence,
    LLMPipelineConfig,
    LLMResponse,
)


class FakeProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.calls = []
        self.structured_calls = []
        self.last_request_count = 0

    def complete(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0)
        usage = response.get("usage") or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        self.last_usage = usage
        self.last_request_count = response.get("request_count", 1)
        return LLMResponse(
            content=response["content"],
            model="gemini-2.5-flash",
            provider="gemini",
            finish_reason="stop",
            usage=usage,
        )

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
        if "exception" in response:
            raise response["exception"]
        self.last_usage = response.get("usage") or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        self.last_request_count = response.get("request_count", 1)
        payload = response.get("parsed", response)
        return response_model.model_validate(payload)


class FakeToolExecutor:
    def __init__(self, batch_results):
        self.batch_results = batch_results
        self.queries = []

    def query_batch_hpo_terms(self, *, phrases, language, n_results):
        self.queries.append(
            {
                "phrases": list(phrases),
                "language": language,
                "n_results": n_results,
            }
        )
        return list(self.batch_results)


class FailingStructuredProvider(LLMProvider):
    def complete(self, messages):
        raise RuntimeError("unused")

    def run_structured_prompt(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
        max_output_tokens=None,
    ):
        raise RuntimeError("boom")


FAILED_GENEREVIEWS_DOC = json.loads(
    Path(
        "tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK532447.json"
    ).read_text(encoding="utf-8")
)


def grounded_phenotype(
    phrase: str,
    category: str,
    *,
    chunk_ids: list[int] | None = None,
    evidence_text: str | None = None,
) -> dict[str, object]:
    return {
        "phrase": phrase,
        "category": category,
        "chunk_ids": list(chunk_ids or [1]),
        "evidence_text": evidence_text or phrase,
    }


def test_grounded_extracted_phenotype_requires_chunk_ids() -> None:
    with pytest.raises(ValidationError):
        LLMGroundedExtractedPhenotype(phrase="seizures", category="Abnormal")


def test_llm_phenotype_can_store_multiple_evidence_records() -> None:
    phenotype = LLMPhenotype(
        term_id="HP:0001250",
        label="Seizure",
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="recurrent seizures",
                evidence_text="recurrent seizures",
                chunk_ids=[2],
                start_char=14,
                end_char=32,
                match_method="local",
            )
        ],
    )

    assert phenotype.evidence_records[0].chunk_ids == [2]


def test_load_prompt_template_reads_yaml_template():
    template = load_prompt_template(AnnotationMode.TWO_PHASE, "en")

    assert isinstance(template, PromptTemplate)
    assert template.version
    assert "expert clinical geneticist" in template.system_prompt


def test_get_prompt_falls_back_to_english():
    template = get_prompt(AnnotationMode.TWO_PHASE, "fr")

    assert template.language == "en"
    assert template.source_path.endswith(
        "phentrieve/llm/prompts/templates/two_phase/en.yaml"
    )


def test_two_phase_pipeline_maps_phrase_via_retrieved_candidates():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [grounded_phenotype("recurrent seizures", "Abnormal")]
                },
            },
            {
                "parsed": {
                    "phrase": "recurrent seizures",
                    "hpo_id": "HP:0001250",
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "original_sentence": "Patient had recurrent seizures since infancy.",
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Seizure",
                        "score": 0.92,
                    }
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="Patient had recurrent seizures since infancy.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert isinstance(result, LLMExtractionResult)
    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0001250",
            label="Seizure",
            evidence="recurrent seizures",
            assertion="present",
            category="abnormal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="recurrent seizures",
                    evidence_text="recurrent seizures",
                    chunk_ids=[1],
                    match_method="llm",
                )
            ],
        )
    ]
    assert result.meta.llm_model == "gemini-2.5-flash"
    assert result.meta.llm_mode == "two_phase"
    assert result.meta.prompt_version
    assert len(provider.structured_calls) == 2
    assert tool_executor.queries == [
        {
            "phrases": ["recurrent seizures"],
            "language": "en",
            "n_results": 50,
        }
    ]
    assert len(provider.calls) == 0


def test_phase1_returns_chunk_ids_and_evidence_text():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "recurrent seizures",
                            "category": "Abnormal",
                            "chunk_ids": [1],
                            "evidence_text": "recurrent seizures",
                        }
                    ]
                }
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider, tool_executor=FakeToolExecutor([])
    )

    result = pipeline._extract_phase1_phenotypes(
        text="Patient had recurrent seizures.",
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
        extraction_prompt=get_prompt(AnnotationMode.TWO_PHASE, "en"),
    )

    assert result[0][0]["chunk_ids"] == [1]


def test_phase1_runs_once_per_extraction_group() -> None:
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_ids=[1, 2],
                        )
                    ]
                }
            },
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "ataxia",
                            "Abnormal",
                            chunk_ids=[3],
                        )
                    ]
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Recurrent seizures",
                        "score": 0.95,
                    }
                ],
            },
            {
                "phrase": "ataxia",
                "candidates": [
                    {
                        "hpo_id": "HP:0001251",
                        "term_name": "Ataxia",
                        "score": 0.95,
                    }
                ],
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    pipeline.run(
        text="Patient had recurrent seizures and ataxia.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Chunk one."},
            {"chunk_id": 2, "text": "Chunk two."},
            {"chunk_id": 3, "text": "Chunk three."},
        ],
        extraction_groups=[
            {
                "group_id": 1,
                "chunk_ids": [1, 2],
                "text": "chunk_id=1: WRONG.\nchunk_id=2: WRONG.",
            },
            {
                "group_id": 2,
                "chunk_ids": [3],
                "text": "chunk_id=3: WRONG.",
            },
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert len(provider.structured_calls) == 2
    assert provider.structured_calls[0]["user_prompt"].endswith(
        "chunk_id=1: WRONG.\nchunk_id=2: WRONG.\n"
    )
    assert provider.structured_calls[1]["user_prompt"].endswith("chunk_id=3: WRONG.\n")


def test_grouped_phase1_uses_budgeted_group_payload_text() -> None:
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_ids=[1, 2],
                        )
                    ]
                }
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=FakeToolExecutor([]),
    )

    pipeline.run(
        text="Patient had recurrent seizures.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Canonical chunk one."},
            {"chunk_id": 2, "text": "Canonical chunk two."},
        ],
        extraction_groups=[
            {
                "group_id": 1,
                "chunk_ids": [1, 2],
                "text": "chunk_id=1: Budgeted chunk one.\nchunk_id=2: Budgeted chunk two.",
                "estimated_prompt_tokens": 12,
            }
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert len(provider.structured_calls) == 1
    assert provider.structured_calls[0]["user_prompt"].endswith(
        "chunk_id=1: Budgeted chunk one.\nchunk_id=2: Budgeted chunk two.\n"
    )
    assert "Canonical chunk one." not in provider.structured_calls[0]["user_prompt"]
    assert "Canonical chunk two." not in provider.structured_calls[0]["user_prompt"]


def test_grouped_phase1_aggregates_mentions_before_retrieval() -> None:
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_ids=[1, 2],
                        )
                    ]
                }
            },
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "normal cognition",
                            "Normal",
                            chunk_ids=[3],
                        )
                    ]
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Recurrent seizures",
                        "score": 0.95,
                    }
                ],
            },
            {
                "phrase": "normal cognition",
                "candidates": [
                    {
                        "hpo_id": "HP:0011446",
                        "term_name": "Normal cognition",
                        "score": 0.95,
                    }
                ],
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=10,
    )

    result = pipeline.run(
        text="Patient had recurrent seizures. Cognition remained normal.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Chunk one."},
            {"chunk_id": 2, "text": "Chunk two."},
            {"chunk_id": 3, "text": "Chunk three."},
        ],
        extraction_groups=[
            {
                "group_id": 1,
                "chunk_ids": [1, 2],
                "text": "chunk_id=1: Chunk one.\nchunk_id=2: Chunk two.",
            },
            {
                "group_id": 2,
                "chunk_ids": [3],
                "text": "chunk_id=3: Chunk three.",
            },
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert tool_executor.queries == [
        {
            "phrases": ["recurrent seizures", "normal cognition"],
            "language": "en",
            "n_results": 50,
        }
    ]
    assert result.meta.phase_counts["extracted_phrases"] == 2
    assert result.meta.phase_counts["actionable_phrases"] == 2


def test_grouped_phase1_deduplicates_mentions_across_groups() -> None:
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_ids=[1],
                        )
                    ]
                }
            },
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_ids=[1],
                        )
                    ]
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Recurrent seizures",
                        "score": 0.95,
                    }
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
    )

    result = pipeline.run(
        text="Patient had recurrent seizures.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Chunk one."},
            {"chunk_id": 2, "text": "Chunk two."},
        ],
        extraction_groups=[
            {
                "group_id": 1,
                "chunk_ids": [1],
                "text": "chunk_id=1: Chunk one.",
            },
            {
                "group_id": 2,
                "chunk_ids": [1, 2],
                "text": "chunk_id=1: Chunk one.\nchunk_id=2: Chunk two.",
            },
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert tool_executor.queries == [
        {
            "phrases": ["recurrent seizures"],
            "language": "en",
            "n_results": 50,
        }
    ]
    assert result.meta.phase_counts["extracted_phrases"] == 1
    assert result.meta.phase_counts["actionable_phrases"] == 1
    assert result.meta.trace["phase1"]["extracted"] == [
        {
            "phrase": "recurrent seizures",
            "category": "abnormal",
            "chunk_ids": [1],
            "evidence_text": "recurrent seizures",
            "actionable": True,
        }
    ]


def test_grouped_phase1_deduplicates_overlapping_mentions_and_merges_grounding() -> (
    None
):
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "recurrent seizures",
                            "category": "Abnormal",
                            "chunk_ids": [1, 2],
                            "evidence_text": "Patient had recurrent seizures",
                            "start_char": 12,
                            "end_char": 30,
                        }
                    ]
                }
            },
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "recurrent seizures",
                            "category": "Abnormal",
                            "chunk_ids": [2, 3],
                            "evidence_text": "Patient had recurrent seizures since infancy",
                            "start_char": 12,
                            "end_char": 44,
                        }
                    ]
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Recurrent seizures",
                        "score": 0.95,
                    }
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="Patient had recurrent seizures since infancy.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Patient had recurrent"},
            {"chunk_id": 2, "text": "recurrent seizures"},
            {"chunk_id": 3, "text": "since infancy."},
        ],
        extraction_groups=[
            {
                "group_id": 1,
                "chunk_ids": [1, 2],
                "text": "chunk_id=1: Patient had recurrent\nchunk_id=2: recurrent seizures",
            },
            {
                "group_id": 2,
                "chunk_ids": [2, 3],
                "text": "chunk_id=2: recurrent seizures\nchunk_id=3: since infancy.",
            },
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert tool_executor.queries == [
        {
            "phrases": ["recurrent seizures"],
            "language": "en",
            "n_results": 50,
        }
    ]
    assert result.meta.phase_counts["extracted_phrases"] == 1
    assert result.meta.trace["phase1"]["extracted"] == [
        {
            "phrase": "recurrent seizures",
            "category": "abnormal",
            "chunk_ids": [1, 2, 3],
            "evidence_text": "Patient had recurrent seizures since infancy",
            "actionable": True,
        }
    ]
    assert result.terms[0].evidence_records == [
        LLMPhenotypeEvidence(
            phrase="recurrent seizures",
            evidence_text="Patient had recurrent seizures since infancy",
            chunk_ids=[1, 2, 3],
            start_char=12,
            end_char=44,
            match_method="local",
        )
    ]


def test_grouped_phase1_keeps_group_chunk_ids_in_trace() -> None:
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_ids=[1, 2],
                        )
                    ]
                },
                "request_count": 1,
            },
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "ataxia",
                            "Abnormal",
                            chunk_ids=[3],
                        )
                    ]
                },
                "request_count": 1,
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Recurrent seizures",
                        "score": 0.95,
                    }
                ],
            },
            {
                "phrase": "ataxia",
                "candidates": [
                    {
                        "hpo_id": "HP:0001251",
                        "term_name": "Ataxia",
                        "score": 0.95,
                    }
                ],
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="Patient had recurrent seizures and ataxia.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Chunk one."},
            {"chunk_id": 2, "text": "Chunk two."},
            {"chunk_id": 3, "text": "Chunk three."},
        ],
        extraction_groups=[
            {
                "group_id": 7,
                "chunk_ids": [1, 2],
                "text": "chunk_id=1: Chunk one.\nchunk_id=2: Chunk two.",
            },
            {
                "group_id": 8,
                "chunk_ids": [3],
                "text": "chunk_id=3: Chunk three.",
            },
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert [group["group_id"] for group in result.meta.trace["phase1"]["groups"]] == [
        7,
        8,
    ]
    assert [
        group["source_chunk_ids"] for group in result.meta.trace["phase1"]["groups"]
    ] == [[1, 2], [3]]
    assert result.meta.trace["phase1"]["groups"][0]["extracted"] == [
        {
            "phrase": "recurrent seizures",
            "category": "abnormal",
            "chunk_ids": [1, 2],
            "evidence_text": "recurrent seizures",
            "actionable": True,
        }
    ]
    assert result.meta.trace["phase1"]["groups"][1]["extracted"] == [
        {
            "phrase": "ataxia",
            "category": "abnormal",
            "chunk_ids": [3],
            "evidence_text": "ataxia",
            "actionable": True,
        }
    ]


def test_group_failure_is_recorded_without_zeroing_document(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="phentrieve.llm.pipeline")
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_ids=[1, 2],
                        )
                    ]
                },
                "request_count": 1,
            },
            {
                "exception": RuntimeError("backend exploded"),
                "request_count": 1,
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Recurrent seizures",
                        "score": 0.95,
                    }
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="Patient had recurrent seizures and later worsening gait instability.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Chunk one."},
            {"chunk_id": 2, "text": "Chunk two."},
            {"chunk_id": 3, "text": "Chunk three."},
        ],
        extraction_groups=[
            {
                "group_id": 1,
                "chunk_ids": [1, 2],
                "text": "chunk_id=1: Chunk one.\nchunk_id=2: Chunk two.",
            },
            {
                "group_id": 2,
                "chunk_ids": [3],
                "text": "chunk_id=3: Chunk three.",
            },
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert [term.term_id for term in result.terms] == ["HP:0001250"]
    assert result.meta.phase_counts["extracted_phrases"] == 1
    assert result.meta.phase_counts["phase1_partial_failures"] == 1
    assert result.meta.trace["phase1"]["extracted"] == [
        {
            "phrase": "recurrent seizures",
            "category": "abnormal",
            "chunk_ids": [1, 2],
            "evidence_text": "recurrent seizures",
            "actionable": True,
        }
    ]
    phase1_groups = result.meta.trace["phase1"]["groups"]
    assert len(phase1_groups) == 2
    assert phase1_groups[0]["group_id"] == 1
    assert phase1_groups[0]["chunk_ids"] == [1, 2]
    assert phase1_groups[0]["status"] == "completed"
    assert phase1_groups[0]["extracted_count"] == 1
    assert phase1_groups[0]["extracted"] == [
        {
            "phrase": "recurrent seizures",
            "category": "abnormal",
            "chunk_ids": [1, 2],
            "evidence_text": "recurrent seizures",
            "actionable": True,
        }
    ]
    assert phase1_groups[1]["group_id"] == 2
    assert phase1_groups[1]["chunk_ids"] == [3]
    assert phase1_groups[1]["status"] == "failed"
    assert phase1_groups[1]["error"] == "Structured extraction failed"
    assert phase1_groups[1]["error_type"] == "LLMPipelinePhaseError"
    assert phase1_groups[1]["extracted_count"] == 0
    assert phase1_groups[1]["extracted"] == []
    assert "Continuing after grouped phase 1 failure" in caplog.text


def test_grouped_phase1_zero_hit_success_does_not_raise() -> None:
    provider = FakeProvider(
        responses=[
            {"parsed": {"phenotypes": []}, "request_count": 1},
            {"parsed": {"phenotypes": []}, "request_count": 1},
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=FakeToolExecutor([]),
    )

    result = pipeline.run(
        text="Clinical note without phenotype mentions.",
        grounded_chunks=[
            {"chunk_id": 1, "text": "Chunk one."},
            {"chunk_id": 2, "text": "Chunk two."},
        ],
        extraction_groups=[
            {
                "group_id": 1,
                "chunk_ids": [1],
                "text": "chunk_id=1: Chunk one.",
            },
            {
                "group_id": 2,
                "chunk_ids": [2],
                "text": "chunk_id=2: Chunk two.",
            },
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == []
    assert result.meta.phase_counts["extracted_phrases"] == 0
    assert result.meta.phase_counts["phase1_completed_groups"] == 2
    assert result.meta.phase_counts["phase1_failed_groups"] == 0
    assert result.meta.phase_counts["phase1_partial_failures"] == 0
    assert [group["status"] for group in result.meta.trace["phase1"]["groups"]] == [
        "completed",
        "completed",
    ]
    assert [
        group["extracted_count"] for group in result.meta.trace["phase1"]["groups"]
    ] == [
        0,
        0,
    ]
    assert result.meta.trace["phase1"]["extracted"] == []


def test_retrieval_uses_grounded_context_instead_of_original_sentence():
    item = {
        "phrase": "frequent falls",
        "category": "abnormal",
        "chunk_ids": [2],
        "evidence_text": "frequent falls",
    }
    grounded_chunks = [
        {"chunk_id": 1, "text": "The child walks independently."},
        {"chunk_id": 2, "text": "The child has frequent falls while walking."},
    ]
    pipeline = TwoPhaseLLMPipeline(
        provider=FakeProvider([]), tool_executor=FakeToolExecutor([])
    )

    context = pipeline._build_grounded_context(
        item=item,
        grounded_chunks=grounded_chunks,
    )

    assert (
        context["primary_chunk_text"] == "The child has frequent falls while walking."
    )
    assert "original_sentence" not in context


def test_phase2_mapping_uses_structured_prompt():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phrase": "frequent falls",
                    "hpo_id": "HP:0002355",
                }
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider, tool_executor=FakeToolExecutor([])
    )

    pipeline._run_mapping_batch(
        batch=[
            {
                "phrase": "frequent falls",
                "category": "abnormal",
                "grounded_context": {
                    "primary_chunk_text": "The child has frequent falls while walking."
                },
                "candidates": [
                    {"hpo_id": "HP:0002355", "term_name": "Difficulty walking"}
                ],
            }
        ],
        mapping_prompt=get_mapping_prompt("en"),
    )

    assert provider.structured_calls


def test_two_phase_pipeline_uses_mapping_prompt_for_unresolved_phrase():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [grounded_phenotype("frequent falls", "Abnormal")]
                },
            },
            {
                "parsed": {
                    "phrase": "frequent falls",
                    "hpo_id": "HP:0002355",
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "frequent falls",
                "original_sentence": "The child has frequent falls while walking.",
                "candidates": [
                    {
                        "hpo_id": "HP:0002355",
                        "term_name": "Difficulty walking",
                        "score": 0.81,
                    },
                    {
                        "hpo_id": "HP:0002317",
                        "term_name": "Unsteady gait",
                        "score": 0.78,
                    },
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="The child has frequent falls while walking.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0002355",
            label="Difficulty walking",
            evidence="frequent falls",
            assertion="present",
            category="abnormal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="frequent falls",
                    evidence_text="frequent falls",
                    chunk_ids=[1],
                    match_method="llm",
                )
            ],
        )
    ]
    assert len(provider.structured_calls) == 2
    assert "primary_chunk_text" in provider.structured_calls[1]["user_prompt"]


def test_two_phase_pipeline_records_trace_for_extraction_and_mapping():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype("frequent falls", "Abnormal"),
                        grounded_phenotype("normal intelligence", "Normal"),
                    ]
                },
                "request_count": 1,
            },
            {
                "parsed": {
                    "mappings": [{"phrase": "frequent falls", "hpo_id": "HP:0002355"}]
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "frequent falls",
                "original_sentence": "The child has frequent falls while walking.",
                "candidates": [
                    {
                        "hpo_id": "HP:0002355",
                        "term_name": "Difficulty walking",
                        "score": 0.81,
                    },
                    {
                        "hpo_id": "HP:0002317",
                        "term_name": "Unsteady gait",
                        "score": 0.78,
                    },
                ],
            },
            {
                "phrase": "normal intelligence",
                "original_sentence": "The child has normal intelligence.",
                "candidates": [
                    {
                        "hpo_id": "HP:0001249",
                        "term_name": "Intellectual disability",
                        "score": 0.74,
                    }
                ],
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=10,
    )

    result = pipeline.run(
        text="The child has frequent falls while walking. The child has normal intelligence.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.meta.trace["phase1"]["extracted"] == [
        {
            "phrase": "frequent falls",
            "category": "abnormal",
            "chunk_ids": [1],
            "evidence_text": "frequent falls",
            "actionable": True,
        },
        {
            "phrase": "normal intelligence",
            "category": "normal",
            "chunk_ids": [1],
            "evidence_text": "normal intelligence",
            "actionable": True,
        },
    ]
    assert (
        result.meta.trace["phase2a"]["candidate_sets"][0]["phrase"] == "frequent falls"
    )
    assert (
        result.meta.trace["phase2a"]["candidate_sets"][1]["phrase"]
        == "normal intelligence"
    )
    assert result.meta.trace["phase2b_local"]["resolved"] == []
    assert result.meta.trace["phase2b_local"]["unresolved"] == [
        {"phrase": "frequent falls", "category": "abnormal"},
        {"phrase": "normal intelligence", "category": "normal"},
    ]
    assert result.meta.trace["phase2b_llm"]["resolved"] == [
        {
            "phrase": "frequent falls",
            "selected_id": "HP:0002355",
            "term_id": "HP:0002355",
            "label": "Difficulty walking",
            "assertion": "present",
            "category": "abnormal",
            "match_method": "llm",
            "local_fallback": False,
        },
        {
            "phrase": "normal intelligence",
            "selected_id": None,
            "term_id": None,
            "label": None,
            "assertion": "negated",
            "category": "normal",
            "match_method": "llm",
            "local_fallback": False,
        },
    ]


def test_two_phase_pipeline_preserves_evidence_records_for_local_matches() -> None:
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_ids=[4],
                            evidence_text="Patient had recurrent seizures.",
                        )
                    ]
                }
            }
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "evidence_text": "Patient had recurrent seizures.",
                "chunk_ids": [4],
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Recurrent seizures",
                        "score": 0.99,
                    }
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="Patient had recurrent seizures.",
        grounded_chunks=[
            {"chunk_id": 4, "text": "Patient had recurrent seizures."},
        ],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0001250",
            label="Recurrent seizures",
            evidence="recurrent seizures",
            assertion="present",
            category="abnormal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="recurrent seizures",
                    evidence_text="Patient had recurrent seizures.",
                    chunk_ids=[4],
                    match_method="local",
                )
            ],
        )
    ]


def test_two_phase_pipeline_batches_unresolved_phrase_mapping_calls():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype("frequent falls", "Abnormal"),
                        grounded_phenotype("sleep disturbances", "Abnormal"),
                    ]
                },
            },
            {
                "parsed": {
                    "mappings": [
                        {"phrase": "frequent falls", "hpo_id": "HP:0002355"},
                        {"phrase": "sleep disturbances", "hpo_id": "HP:0002360"},
                    ]
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "frequent falls",
                "original_sentence": "The child has frequent falls while walking.",
                "candidates": [
                    {
                        "hpo_id": "HP:0002355",
                        "term_name": "Difficulty walking",
                        "score": 0.81,
                    }
                ],
            },
            {
                "phrase": "sleep disturbances",
                "original_sentence": "Sleep disturbances were reported.",
                "candidates": [
                    {
                        "hpo_id": "HP:0002360",
                        "term_name": "Sleep abnormality",
                        "score": 0.85,
                    }
                ],
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=2,
    )

    result = pipeline.run(
        text="The child has frequent falls while walking. Sleep disturbances were reported.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0002355",
            label="Difficulty walking",
            evidence="frequent falls",
            assertion="present",
            category="abnormal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="frequent falls",
                    evidence_text="frequent falls",
                    chunk_ids=[1],
                    match_method="llm",
                )
            ],
        ),
        LLMPhenotype(
            term_id="HP:0002360",
            label="Sleep abnormality",
            evidence="sleep disturbances",
            assertion="present",
            category="abnormal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="sleep disturbances",
                    evidence_text="sleep disturbances",
                    chunk_ids=[1],
                    match_method="llm",
                )
            ],
        ),
    ]
    assert len(provider.structured_calls) == 2


def test_two_phase_pipeline_batch_mapping_accepts_normalized_returned_phrase_keys():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype("knock-knee (genu valgum)", "Abnormal"),
                        grounded_phenotype("Legg Perthes disease", "Abnormal"),
                    ]
                },
            },
            {
                "parsed": {
                    "mappings": [
                        {
                            "phrase": "knock knee (genu valgum)",
                            "hpo_id": "HP:0002857",
                        },
                        {
                            "phrase": "legg perthes disease",
                            "hpo_id": "HP:0005743",
                        },
                    ]
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "knock-knee (genu valgum)",
                "original_sentence": "The child has knock-knee (genu valgum).",
                "candidates": [
                    {
                        "hpo_id": "HP:0002857",
                        "term_name": "Genu valgum",
                        "score": 0.88,
                    }
                ],
            },
            {
                "phrase": "Legg Perthes disease",
                "original_sentence": "The child has Legg Perthes disease.",
                "candidates": [
                    {
                        "hpo_id": "HP:0005743",
                        "term_name": "Avascular necrosis of the capital femoral epiphysis",
                        "score": 0.99,
                    }
                ],
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=10,
    )

    result = pipeline.run(
        text="The child has knock-knee (genu valgum). The child has Legg Perthes disease.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0002857",
            label="Genu valgum",
            evidence="knock-knee (genu valgum)",
            assertion="present",
            category="abnormal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="knock-knee (genu valgum)",
                    evidence_text="knock-knee (genu valgum)",
                    chunk_ids=[1],
                    match_method="llm",
                )
            ],
        ),
        LLMPhenotype(
            term_id="HP:0005743",
            label="Avascular necrosis of the capital femoral epiphysis",
            evidence="Legg Perthes disease",
            assertion="present",
            category="abnormal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="Legg Perthes disease",
                    evidence_text="Legg Perthes disease",
                    chunk_ids=[1],
                    match_method="llm",
                )
            ],
        ),
    ]


def test_two_phase_pipeline_retrieves_all_categories_and_preserves_assertions():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype("recurrent seizures", "Abnormal"),
                        grounded_phenotype("nystagmus", "Suspected"),
                        grounded_phenotype("skeletal anomalies", "Normal"),
                        grounded_phenotype("hearing loss", "Family_History"),
                        grounded_phenotype("onset in infancy", "Other"),
                    ]
                },
            }
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "original_sentence": "Patient had recurrent seizures since infancy.",
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Recurrent seizures",
                        "score": 0.95,
                    }
                ],
            },
            {
                "phrase": "nystagmus",
                "original_sentence": "Nystagmus was suspected clinically.",
                "candidates": [
                    {
                        "hpo_id": "HP:0000639",
                        "term_name": "Nystagmus",
                        "score": 0.95,
                    }
                ],
            },
            {
                "phrase": "skeletal anomalies",
                "original_sentence": "No skeletal anomalies were noted.",
                "candidates": [
                    {
                        "hpo_id": "HP:0000924",
                        "term_name": "skeletal anomalies",
                        "score": 0.95,
                    }
                ],
            },
            {
                "phrase": "hearing loss",
                "original_sentence": "The mother has hearing loss.",
                "candidates": [
                    {
                        "hpo_id": "HP:0000365",
                        "term_name": "hearing loss",
                        "score": 0.95,
                    }
                ],
            },
            {
                "phrase": "onset in infancy",
                "original_sentence": "Symptoms began in infancy.",
                "candidates": [
                    {
                        "hpo_id": "HP:0003593",
                        "term_name": "onset in infancy",
                        "score": 0.95,
                    }
                ],
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text=(
            "Patient had recurrent seizures since infancy. "
            "Nystagmus was suspected clinically. "
            "No skeletal anomalies were noted. "
            "The mother has hearing loss. "
            "Symptoms began in infancy."
        ),
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert tool_executor.queries == [
        {
            "phrases": [
                "recurrent seizures",
                "nystagmus",
                "skeletal anomalies",
                "hearing loss",
            ],
            "language": "en",
            "n_results": 50,
        }
    ]
    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0001250",
            label="Recurrent seizures",
            evidence="recurrent seizures",
            assertion="present",
            category="abnormal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="recurrent seizures",
                    evidence_text="recurrent seizures",
                    chunk_ids=[1],
                    match_method="local",
                )
            ],
        ),
        LLMPhenotype(
            term_id="HP:0000639",
            label="Nystagmus",
            evidence="nystagmus",
            assertion="uncertain",
            category="suspected",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="nystagmus",
                    evidence_text="nystagmus",
                    chunk_ids=[1],
                    match_method="local",
                )
            ],
        ),
        LLMPhenotype(
            term_id="HP:0000924",
            label="skeletal anomalies",
            evidence="skeletal anomalies",
            assertion="negated",
            category="normal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="skeletal anomalies",
                    evidence_text="skeletal anomalies",
                    chunk_ids=[1],
                    match_method="local",
                )
            ],
        ),
        LLMPhenotype(
            term_id="HP:0000365",
            label="hearing loss",
            evidence="hearing loss",
            assertion="family_history",
            category="family_history",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="hearing loss",
                    evidence_text="hearing loss",
                    chunk_ids=[1],
                    match_method="local",
                )
            ],
        ),
    ]
    assert result.meta.phase_counts["extracted_phrases"] == 5
    assert result.meta.phase_counts["actionable_phrases"] == 4


def test_two_phase_pipeline_accumulates_usage_and_logs_phases(caplog):
    caplog.set_level(logging.DEBUG, logger="phentrieve.llm.pipeline")
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [grounded_phenotype("frequent falls", "Abnormal")]
                },
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
                "request_count": 2,
            },
            {
                "parsed": {
                    "phrase": "frequent falls",
                    "hpo_id": "HP:0002355",
                },
                "usage": {
                    "prompt_tokens": 13,
                    "completion_tokens": 17,
                    "total_tokens": 30,
                },
                "request_count": 3,
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "frequent falls",
                "original_sentence": "The child has frequent falls while walking.",
                "candidates": [
                    {
                        "hpo_id": "HP:0002355",
                        "term_name": "Difficulty walking",
                        "score": 0.81,
                    },
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="The child has frequent falls while walking.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.meta.token_input == 24
    assert result.meta.token_output == 24
    assert result.meta.request_count == 5
    assert result.meta.phase_timings["phase1_seconds"] >= 0.0
    assert result.meta.phase_timings["phase2a_seconds"] >= 0.0
    assert result.meta.phase_timings["phase2b_local_seconds"] >= 0.0
    assert result.meta.phase_timings["phase2b_llm_seconds"] >= 0.0
    assert result.meta.phase_counts == {
        "extracted_phrases": 1,
        "actionable_phrases": 1,
        "candidate_sets": 1,
        "unresolved_phrases": 1,
        "local_matches": 0,
        "llm_mapped_phrases": 1,
        "local_fallbacks": 0,
    }
    assert "phase1_completed_groups" not in result.meta.phase_counts
    assert "phase1_failed_groups" not in result.meta.phase_counts
    assert "phase1_partial_failures" not in result.meta.phase_counts
    assert any(
        "Phase 1: extracting phenotype phrases" in record.message
        for record in caplog.records
    )
    assert any(
        "Phase 2A: retrieving candidates" in record.message for record in caplog.records
    )
    assert any(
        "Phase 2B-local: resolving local matches" in record.message
        for record in caplog.records
    )
    assert any(
        "Phase 2B-llm: mapping unresolved phrases" in record.message
        for record in caplog.records
    )
    assert any(
        "Phase 2B-llm complete" in record.message and "mapped=1" in record.message
        for record in caplog.records
    )


def test_two_phase_pipeline_separates_llm_mappings_from_local_fallbacks(
    mocker,
) -> None:
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [grounded_phenotype("frequent falls", "Abnormal")]
                },
                "request_count": 1,
            },
            {
                "parsed": {
                    "phrase": "frequent falls",
                    "hpo_id": None,
                },
                "request_count": 1,
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "frequent falls",
                "candidates": [
                    {
                        "hpo_id": "HP:0002355",
                        "term_name": "Frequent falls",
                        "score": 0.8,
                    }
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)
    mocker.patch.object(
        pipeline,
        "_try_local_match",
        side_effect=[
            None,
            {
                "hpo_id": "HP:0002355",
                "term_name": "Frequent falls",
                "score": 0.8,
            },
        ],
    )

    result = pipeline.run(
        text="The child has frequent falls.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert [term.term_id for term in result.terms] == ["HP:0002355"]
    assert result.meta.phase_counts["unresolved_phrases"] == 1
    assert result.meta.phase_counts["local_matches"] == 0
    assert result.meta.phase_counts["llm_mapped_phrases"] == 0
    assert result.meta.phase_counts["local_fallbacks"] == 1
    assert result.meta.trace["phase2b_llm"]["resolved"] == [
        {
            "phrase": "frequent falls",
            "selected_id": None,
            "term_id": "HP:0002355",
            "label": "Frequent falls",
            "assertion": "present",
            "category": "abnormal",
            "match_method": "llm",
            "local_fallback": True,
        }
    ]


def test_two_phase_pipeline_logs_malformed_phase1_content(caplog):
    caplog.set_level(logging.WARNING, logger="phentrieve.llm.pipeline")
    provider = FakeProvider(
        responses=[
            {
                "parsed": {"phenotypes": []},
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider, tool_executor=FakeToolExecutor([])
    )

    result = pipeline.run(
        text="Patient had recurrent seizures.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == []
    assert not caplog.records


def test_two_phase_pipeline_logs_malformed_phase1_structure():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": {
                        "phrase": "recurrent seizures",
                        "category": "Abnormal",
                    }
                },
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider, tool_executor=FakeToolExecutor([])
    )

    with pytest.raises(LLMPipelinePhaseError) as exc_info:
        pipeline.run(
            text="Patient had recurrent seizures.",
            config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
        )

    assert exc_info.value.phase == "phase1"
    assert str(exc_info.value) == "Structured extraction failed"


def test_two_phase_pipeline_falls_back_to_local_match_after_invalid_mapping_selection(
    mocker,
):
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [grounded_phenotype("frequent falls", "Abnormal")]
                },
            },
            {
                "parsed": {
                    "phrase": "frequent falls",
                    "hpo_id": "HP:9999999",
                }
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "frequent falls",
                "original_sentence": "The child has frequent falls while walking.",
                "candidates": [
                    {
                        "hpo_id": "HP:0002355",
                        "term_name": "Difficulty walking",
                        "score": 0.81,
                    }
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)
    mocker.patch.object(
        pipeline,
        "_try_local_match",
        side_effect=[
            None,
            {
                "hpo_id": "HP:0002355",
                "term_name": "Difficulty walking",
                "score": 0.81,
            },
        ],
    )

    result = pipeline.run(
        text="The child has frequent falls while walking.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0002355",
            label="Difficulty walking",
            evidence="frequent falls",
            assertion="present",
            category="abnormal",
            evidence_records=[
                LLMPhenotypeEvidence(
                    phrase="frequent falls",
                    evidence_text="frequent falls",
                    chunk_ids=[1],
                    match_method="local",
                )
            ],
        )
    ]
    assert pipeline._try_local_match.call_count == 2


def test_two_phase_pipeline_handles_empty_batch_result_shape_without_crashing():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "progressive central nervous system dysfunction",
                            "Abnormal",
                        )
                    ]
                }
            }
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="Progressive central nervous system dysfunction was reported.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == []


def test_two_phase_pipeline_chunks_large_failed_genereviews_document_queries():
    phrases = [
        grounded_phenotype(f"failed-doc-phenotype-{index}", "Abnormal")
        for index in range(56)
    ]
    provider = FakeProvider(responses=[{"parsed": {"phenotypes": phrases}}])

    class ChunkingRetriever:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def query_batch(self, texts, n_results, include_similarities=True):
            self.calls.append(list(texts))
            return [
                {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                    "similarities": [[]],
                }
                for _ in texts
            ]

    retriever = ChunkingRetriever()
    tool_executor = ToolExecutor(retriever=retriever, retrieval_batch_size=25)
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text=FAILED_GENEREVIEWS_DOC["full_text"],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == []
    assert [len(call) for call in retriever.calls] == [25, 25, 6]


def test_deduplicate_terms_keeps_assertion_variants():
    terms = [
        LLMPhenotype(
            term_id="HP:0001250",
            label="Seizure",
            assertion="present",
            category="abnormal",
        ),
        LLMPhenotype(
            term_id="HP:0001250",
            label="Seizure",
            assertion="negated",
            category="normal",
        ),
    ]

    deduped = TwoPhaseLLMPipeline._deduplicate_terms(terms)

    assert len(deduped) == 2


def test_phase1_failure_is_recorded_in_trace_not_silenced(caplog):
    caplog.set_level(logging.ERROR, logger="phentrieve.llm.pipeline")
    provider = FailingStructuredProvider()
    pipeline = TwoPhaseLLMPipeline(
        provider=provider, tool_executor=FakeToolExecutor([])
    )

    with pytest.raises(RuntimeError):
        pipeline.run(
            text="Patient had recurrent seizures.",
            grounded_chunks=[
                {"chunk_id": 1, "text": "Patient had recurrent seizures."}
            ],
            config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
        )


def test_two_phase_pipeline_rejects_unsupported_mode():
    provider = FakeProvider(responses=[])
    pipeline = TwoPhaseLLMPipeline(
        provider=provider, tool_executor=FakeToolExecutor([])
    )

    with pytest.raises(ValueError, match="Unsupported LLM mode"):
        pipeline.run(
            text="Patient had recurrent seizures.",
            config=LLMPipelineConfig(model="gemini-2.5-flash", mode="tool_text"),
        )
