import json
import logging
from pathlib import Path

import pytest

from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.prompts.loader import (
    PromptTemplate,
    get_prompt,
    load_prompt_template,
)
from phentrieve.llm.provider import LLMProvider, ToolExecutor
from phentrieve.llm.types import (
    AnnotationMode,
    LLMExtractionResult,
    LLMPhenotype,
    LLMPipelineConfig,
    LLMResponse,
)


class FakeProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.calls = []
        self.structured_calls = []

    def complete(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0)
        usage = response.get("usage") or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        self.last_usage = usage
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
        self.last_usage = response.get("usage") or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
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


FAILED_GENEREVIEWS_DOC = json.loads(
    Path(
        "tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK532447.json"
    ).read_text(encoding="utf-8")
)


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
                    "phenotypes": [
                        {"phrase": "recurrent seizures", "category": "Abnormal"}
                    ]
                },
            },
            {"content": '{"hpo_id":"HP:0001250"}'},
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
        )
    ]
    assert result.meta.llm_model == "gemini-2.5-flash"
    assert result.meta.llm_mode == "two_phase"
    assert result.meta.prompt_version
    assert len(provider.structured_calls) == 1
    assert tool_executor.queries == [
        {
            "phrases": ["recurrent seizures"],
            "language": "en",
            "n_results": 200,
        }
    ]
    assert len(provider.calls) == 1


def test_two_phase_pipeline_uses_mapping_prompt_for_unresolved_phrase():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [{"phrase": "frequent falls", "category": "Abnormal"}]
                },
            },
            {"content": '{"hpo_id":"HP:0002355"}'},
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
        )
    ]
    assert len(provider.calls) == 1
    assert "original_sentence" in provider.calls[0][-1]["content"]


def test_two_phase_pipeline_accumulates_usage_and_logs_phases(caplog):
    caplog.set_level(logging.DEBUG, logger="phentrieve.llm.pipeline")
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [{"phrase": "frequent falls", "category": "Abnormal"}]
                },
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
            {
                "content": '{"hpo_id":"HP:0002355"}',
                "usage": {
                    "prompt_tokens": 13,
                    "completion_tokens": 17,
                    "total_tokens": 30,
                },
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


def test_two_phase_pipeline_logs_malformed_phase1_structure(caplog):
    caplog.set_level(logging.WARNING, logger="phentrieve.llm.pipeline")
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

    result = pipeline.run(
        text="Patient had recurrent seizures.",
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.terms == []
    assert any(
        "Phase 1 structured extraction failed" in record.message
        for record in caplog.records
    )


def test_two_phase_pipeline_falls_back_to_local_match_after_invalid_mapping_selection(
    mocker,
):
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [{"phrase": "frequent falls", "category": "Abnormal"}]
                },
            },
            {"content": '{"hpo_id":"HP:9999999"}'},
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
        )
    ]
    assert pipeline._try_local_match.call_count == 2


def test_two_phase_pipeline_handles_empty_batch_result_shape_without_crashing():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "progressive central nervous system dysfunction",
                            "category": "Abnormal",
                        }
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
        {"phrase": f"failed-doc-phenotype-{index}", "category": "Abnormal"}
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
