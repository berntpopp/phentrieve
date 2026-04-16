import pytest

from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import LLMExtractionResult, LLMPipelineConfig, LLMResponse

pytestmark = pytest.mark.integration


class FakeProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.last_request_count = 0

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
        payload = response.get("parsed", response)
        self.last_usage = response.get("usage") or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        self.last_request_count = response.get("request_count", 1)
        return response_model.model_validate(payload)


class FakeToolExecutor:
    def __init__(self, batch_results):
        self.batch_results = list(batch_results)

    def query_batch_hpo_terms(self, *, phrases, language, n_results):
        return list(self.batch_results)


@pytest.mark.integration
def test_grounded_llm_pipeline_preserves_english_chunk_provenance():
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
                            "term_name": "Paroxysmal event",
                            "score": 0.97,
                        }
                    ],
                }
            ]
        ),
    )

    result: LLMExtractionResult = pipeline.run(
        text="Patient had recurrent seizures.",
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "Patient had recurrent seizures.",
                "start_char": 0,
                "end_char": 31,
            }
        ],
        config=LLMPipelineConfig(
            model="gemini-2.5-flash",
            mode="two_phase",
            language="en",
        ),
    )

    assert result.meta.trace["phase1"]["extracted"][0]["chunk_ids"] == [1]
    assert (
        result.meta.trace["phase2a"]["candidate_sets"][0]["grounded_context"][
            "primary_chunk_text"
        ]
        == "Patient had recurrent seizures."
    )


@pytest.mark.integration
def test_grounded_llm_pipeline_preserves_german_chunk_provenance():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "wiederkehrende Anfaelle",
                            "category": "Abnormal",
                            "chunk_ids": [1],
                            "evidence_text": "wiederkehrende Anfaelle",
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
                            "term_name": "Paroxysmal event",
                            "score": 0.96,
                        }
                    ],
                }
            ]
        ),
    )

    result: LLMExtractionResult = pipeline.run(
        text="Patientin hat wiederkehrende Anfaelle.",
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "Patientin hat wiederkehrende Anfaelle.",
                "start_char": 0,
                "end_char": 39,
            }
        ],
        config=LLMPipelineConfig(
            model="gemini-2.5-flash",
            mode="two_phase",
            language="de",
        ),
    )

    assert any(item["chunk_ids"] for item in result.meta.trace["phase1"]["extracted"])
    assert (
        result.meta.trace["phase2a"]["candidate_sets"][0]["grounded_context"][
            "primary_chunk_text"
        ]
        == "Patientin hat wiederkehrende Anfaelle."
    )
