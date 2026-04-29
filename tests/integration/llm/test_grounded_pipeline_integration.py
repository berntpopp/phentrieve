import pytest

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import LLMExtractionResult, LLMPipelineConfig, LLMResponse

pytestmark = pytest.mark.integration

TwoPhaseLLMPipeline = pipeline_module.TwoPhaseLLMPipeline


class FakeProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.last_request_count = 0
        self.count_token_calls = []

    def complete(self, messages) -> LLMResponse:
        raise AssertionError("unused in grounded integration tests")

    def run_structured_prompt(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
        max_output_tokens=None,
    ):
        response = self.responses.pop(0)
        if "exception" in response:
            self.last_usage = response.get("usage") or {}
            self.last_request_count = response.get("request_count", 1)
            raise response["exception"]
        payload = response.get("parsed", response)
        self.last_usage = response.get("usage") or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        self.last_request_count = response.get("request_count", 1)
        return response_model.model_validate(payload)

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        self.count_token_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        chunk_count = max(user_prompt.count("chunk_id="), 1)
        prompt_tokens = chunk_count * 10
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        }


class FakeToolExecutor:
    def __init__(self, batch_results):
        self.batch_results = list(batch_results)

    def query_batch_hpo_terms(self, *, phrases, language, n_results):
        return list(self.batch_results)


@pytest.mark.integration
def test_grounded_llm_pipeline_grouped_path_preserves_english_provenance():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "recurrent seizures",
                            "category": "Abnormal",
                            "chunk_ids": [2],
                            "evidence_text": "seizures with loss of awareness.",
                        }
                    ]
                }
            },
            {
                "parsed": {
                    "phrase": "recurrent seizures",
                    "hpo_id": "HP:0001250",
                }
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=FakeToolExecutor(
            [
                {
                    "phrase": "recurrent seizures",
                    "candidates": [
                        {
                            "hpo_id": "HP:0001250",
                            "term_name": "Recurrent seizures",
                            "score": 0.97,
                        }
                    ],
                }
            ]
        ),
    )

    result: LLMExtractionResult = pipeline.run(
        text=("Patient has recurrent\nseizures with loss of awareness.\nNo headaches."),
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "Patient has recurrent",
                "start_char": 0,
                "end_char": 21,
            },
            {
                "chunk_id": 2,
                "text": "seizures with loss of awareness.",
                "start_char": 22,
                "end_char": 54,
            },
            {
                "chunk_id": 3,
                "text": "No headaches.",
                "start_char": 55,
                "end_char": 68,
            },
        ],
        extraction_groups=[
            {
                "group_id": 11,
                "chunk_ids": [1, 2],
                "text": (
                    "chunk_id=1: Patient has recurrent\n"
                    "chunk_id=2: seizures with loss of awareness."
                ),
            },
            {
                "group_id": 12,
                "chunk_ids": [2, 3],
                "text": (
                    "chunk_id=2: seizures with loss of awareness.\n"
                    "chunk_id=3: No headaches."
                ),
            },
        ],
        config=LLMPipelineConfig(
            model="gemini-2.5-flash",
            mode="two_phase",
            language="en",
        ),
    )

    phase1_groups = result.meta.trace["phase1"]["groups"]
    assert [group["source_chunk_ids"] for group in phase1_groups] == [[1, 2], [2, 3]]
    assert phase1_groups[0]["extracted"][0]["chunk_ids"] == [2]
    assert result.meta.trace["phase1"]["extracted"][0]["chunk_ids"] == [2]
    grounded_context = result.meta.trace["phase2a"]["candidate_sets"][0][
        "grounded_context"
    ]
    assert grounded_context["chunk_ids"] == [2]
    assert grounded_context["primary_chunk_text"] == "seizures with loss of awareness."
    assert grounded_context["neighbor_chunk_texts"] == [
        "Patient has recurrent",
        "No headaches.",
    ]


@pytest.mark.integration
def test_grounded_llm_pipeline_grouped_path_preserves_german_provenance():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "wiederkehrende Anfaelle",
                            "category": "Abnormal",
                            "chunk_ids": [2],
                            "evidence_text": "wiederkehrende Anfaelle und Schwindel.",
                        }
                    ]
                }
            },
            {
                "parsed": {
                    "phrase": "wiederkehrende Anfaelle",
                    "hpo_id": "HP:0001250",
                }
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=FakeToolExecutor(
            [
                {
                    "phrase": "wiederkehrende Anfaelle",
                    "candidates": [
                        {
                            "hpo_id": "HP:0001250",
                            "term_name": "wiederkehrende Anfaelle",
                            "score": 0.96,
                        }
                    ],
                }
            ]
        ),
    )

    result: LLMExtractionResult = pipeline.run(
        text=(
            "Die Patientin berichtet ueber\n"
            "wiederkehrende Anfaelle und Schwindel.\n"
            "Keine Kopfschmerzen."
        ),
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "Die Patientin berichtet ueber",
                "start_char": 0,
                "end_char": 29,
            },
            {
                "chunk_id": 2,
                "text": "wiederkehrende Anfaelle und Schwindel.",
                "start_char": 30,
                "end_char": 68,
            },
            {
                "chunk_id": 3,
                "text": "Keine Kopfschmerzen.",
                "start_char": 69,
                "end_char": 89,
            },
        ],
        extraction_groups=[
            {
                "group_id": 21,
                "chunk_ids": [1, 2],
                "text": (
                    "chunk_id=1: Die Patientin berichtet ueber\n"
                    "chunk_id=2: wiederkehrende Anfaelle und Schwindel."
                ),
            },
            {
                "group_id": 22,
                "chunk_ids": [2, 3],
                "text": (
                    "chunk_id=2: wiederkehrende Anfaelle und Schwindel.\n"
                    "chunk_id=3: Keine Kopfschmerzen."
                ),
            },
        ],
        config=LLMPipelineConfig(
            model="gemini-2.5-flash",
            mode="two_phase",
            language="de",
        ),
    )

    phase1_groups = result.meta.trace["phase1"]["groups"]
    assert [group["source_chunk_ids"] for group in phase1_groups] == [[1, 2], [2, 3]]
    assert phase1_groups[0]["extracted"][0]["chunk_ids"] == [2]
    assert result.meta.trace["phase1"]["extracted"][0]["chunk_ids"] == [2]
    grounded_context = result.meta.trace["phase2a"]["candidate_sets"][0][
        "grounded_context"
    ]
    assert grounded_context["chunk_ids"] == [2]
    assert grounded_context["primary_chunk_text"] == (
        "wiederkehrende Anfaelle und Schwindel."
    )
    assert grounded_context["neighbor_chunk_texts"] == [
        "Die Patientin berichtet ueber",
        "Keine Kopfschmerzen.",
    ]


@pytest.mark.integration
def test_grounded_llm_pipeline_chains_fallback_to_grouped_small_and_preserves_provenance(
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "DEFAULT_PHASE1_LARGE_GROUP_MAX_PROMPT_TOKENS",
        21,
    )
    monkeypatch.setattr(
        pipeline_module,
        "DEFAULT_PHASE1_SMALL_GROUP_MAX_CHUNKS",
        1,
    )
    provider = FakeProvider(
        responses=[
            {
                "exception": RuntimeError("invalid json payload"),
                "request_count": 1,
            },
            {
                "exception": RuntimeError("invalid json payload"),
                "request_count": 1,
            },
            {
                "exception": RuntimeError("invalid json payload"),
                "request_count": 1,
            },
            {"parsed": {"phenotypes": []}, "request_count": 1},
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "recurrent seizures",
                            "category": "Abnormal",
                            "chunk_ids": [2],
                            "evidence_text": "seizures with loss of awareness.",
                        }
                    ]
                },
                "request_count": 1,
            },
            {"parsed": {"phenotypes": []}, "request_count": 1},
            {
                "parsed": {
                    "phrase": "recurrent seizures",
                    "hpo_id": "HP:0001250",
                },
                "request_count": 1,
            },
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=FakeToolExecutor(
            [
                {
                    "phrase": "recurrent seizures",
                    "candidates": [
                        {
                            "hpo_id": "HP:0001250",
                            "term_name": "Recurrent seizures",
                            "score": 0.97,
                        }
                    ],
                }
            ]
        ),
    )

    result: LLMExtractionResult = pipeline.run(
        text=("Patient has recurrent\nseizures with loss of awareness.\nNo headaches."),
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "Patient has recurrent",
                "start_char": 0,
                "end_char": 21,
            },
            {
                "chunk_id": 2,
                "text": "seizures with loss of awareness.",
                "start_char": 22,
                "end_char": 54,
            },
            {
                "chunk_id": 3,
                "text": "No headaches.",
                "start_char": 55,
                "end_char": 68,
            },
        ],
        extraction_groups=[],
        config=LLMPipelineConfig(
            model="gemini-2.5-flash",
            mode="two_phase",
            language="en",
        ),
    )
    assert result.meta.trace["phase1"]["initial_mode"] == "ungrouped"
    assert result.meta.trace["phase1"]["final_mode"] == "grouped_small"
    assert result.meta.trace["phase1"]["fallback_triggered"] is True
    assert [attempt["mode"] for attempt in result.meta.trace["phase1"]["attempts"]] == [
        "ungrouped",
        "grouped_large",
        "grouped_small",
    ]
    phase1_groups = result.meta.trace["phase1"]["groups"]
    assert [group["source_chunk_ids"] for group in phase1_groups] == [[1], [2], [3]]
    assert result.meta.trace["phase1"]["attempts"][1]["groups"][0][
        "source_chunk_ids"
    ] == [
        1,
        2,
    ]
    assert result.meta.trace["phase1"]["attempts"][2]["groups"][1]["extracted"][0][
        "chunk_ids"
    ] == [2]
    assert result.meta.trace["phase1"]["extracted"][0]["chunk_ids"] == [2]
    grounded_context = result.meta.trace["phase2a"]["candidate_sets"][0][
        "grounded_context"
    ]
    assert grounded_context["chunk_ids"] == [2]
    assert grounded_context["primary_chunk_text"] == "seizures with loss of awareness."
    assert grounded_context["neighbor_chunk_texts"] == [
        "Patient has recurrent",
        "No headaches.",
    ]
