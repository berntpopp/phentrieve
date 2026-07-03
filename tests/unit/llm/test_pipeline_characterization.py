import pytest

import phentrieve.llm.pipeline as pipeline_module
from phentrieve.llm.prompts.loader import get_mapping_prompt, get_prompt
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import LLMPipelineConfig, LLMResponse


class FakeProvider(LLMProvider):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.structured_calls: list[dict[str, object]] = []
        self.last_request_count = 0

    def complete(self, messages):  # pragma: no cover - not used by these tests
        return LLMResponse(
            content="{}",
            model="fake",
            provider="fake",
            finish_reason="stop",
            usage={},
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


class FakeToolExecutor:
    def __init__(self, batch_results: list[dict[str, object]]) -> None:
        self.batch_results = list(batch_results)
        self.queries: list[dict[str, object]] = []

    def query_batch_hpo_terms(self, *, phrases, language, n_results):
        self.queries.append(
            {
                "phrases": list(phrases),
                "language": language,
                "n_results": n_results,
            }
        )
        return list(self.batch_results)


def test_phase1_extraction_shape_preserves_grounding_fields() -> None:
    provider = FakeProvider(
        [
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "recurrent seizures",
                            "category": "Abnormal",
                            "chunk_ids": [2],
                            "evidence_text": "recurrent seizures since infancy",
                            "start_char": 8,
                            "end_char": 26,
                        }
                    ]
                },
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "total_tokens": 27,
                },
                "request_count": 1,
            }
        ]
    )
    pipeline = pipeline_module.TwoPhaseLLMPipeline(provider=provider)

    extracted, usage, request_count, debug = pipeline._extract_phase1_phenotypes(
        text="Patient has recurrent seizures since infancy.",
        grounded_chunks=[
            {
                "chunk_id": 2,
                "text": "Patient has recurrent seizures since infancy.",
            }
        ],
        extraction_prompt=get_prompt("two_phase", "en"),
        capture_debug=True,
    )

    assert extracted == [
        {
            "phrase": "recurrent seizures",
            "category": "abnormal",
            "negated_qualifier": None,
            "chunk_ids": [2],
            "evidence_text": "recurrent seizures since infancy",
            "start_char": 8,
            "end_char": 26,
            "experiencer": "proband",
            "assertion": "present",
        }
    ]
    assert usage == {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27}
    assert request_count == 1
    assert debug is not None
    assert debug["parsed_extracted"] == extracted


def test_phase2_mapping_batch_shape_uses_single_item_prompt() -> None:
    provider = FakeProvider(
        [{"parsed": {"phrase": "recurrent seizures", "hpo_id": "HP:0001250"}}]
    )
    pipeline = pipeline_module.TwoPhaseLLMPipeline(provider=provider)

    response, usage = pipeline._run_mapping_batch(
        batch=[
            {
                "phrase": "recurrent seizures",
                "category": "abnormal",
                "grounded_context": {
                    "primary_chunk_text": "Patient has recurrent seizures.",
                    "neighbor_chunk_texts": [],
                },
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Seizure",
                        "score": 0.91,
                    }
                ],
            }
        ],
        mapping_prompt=get_mapping_prompt("en"),
    )

    assert response.hpo_id == "HP:0001250"
    assert usage["total_tokens"] == 15
    assert provider.structured_calls[0]["response_model"].__name__ == (
        "LLMMappingSelection"
    )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError("refusal from structured output"), "structured_refusal"),
        (RuntimeError("deadline_exceeded timeout"), "provider_timeout"),
        (RuntimeError("unterminated JSON object"), "structured_json_invalid"),
        (
            RuntimeError("field required by schema"),
            "structured_schema_validation_failed",
        ),
        (RuntimeError("401 unauthorized api key"), "provider_auth_error"),
        (RuntimeError("network transport readerror"), "provider_transport_error"),
        (RuntimeError("unknown provider configuration"), "provider_config_error"),
        (RuntimeError("boom"), "provider_execution_error"),
    ],
)
def test_phase1_failure_classification_is_stable(exc: Exception, expected: str) -> None:
    assert pipeline_module._classify_phase1_failure(exc) == expected


def test_mapping_trace_provenance_preserves_source_metadata() -> None:
    annotated = (
        pipeline_module.TwoPhaseLLMPipeline._annotate_mapping_trace_with_provenance(
            {"phrase": "falls", "term_id": "HP:0002527"},
            {
                "phrase": "falls",
                "evidence_text": "Frequent falls were noted.",
                "chunk_ids": [3, 4],
                "start_char": 12,
                "end_char": 26,
            },
        )
    )

    assert annotated == {
        "phrase": "falls",
        "term_id": "HP:0002527",
        "evidence_text": "Frequent falls were noted.",
        "chunk_ids": [3, 4],
        "start_char": 12,
        "end_char": 26,
    }


def test_phase1_json_failure_falls_back_to_grouped_large_and_records_trace() -> None:
    provider = FakeProvider(
        [
            {"exception": RuntimeError("unterminated JSON object")},
            {"parsed": {"phenotypes": []}},
        ]
    )
    pipeline = pipeline_module.TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=FakeToolExecutor([]),
    )

    result = pipeline.run(
        text="Patient has recurrent seizures.",
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "Patient has recurrent seizures.",
                "status": "matched",
            }
        ],
        config=LLMPipelineConfig(
            provider="fake",
            model="fake",
            mode="two_phase",
            language="en",
        ),
    )

    phase1_trace = result.meta.trace["phase1"]
    assert phase1_trace["initial_mode"] == "ungrouped"
    assert phase1_trace["final_mode"] == "grouped_large"
    assert phase1_trace["fallback_triggered"] is True
    assert phase1_trace["failure_class"] == "structured_json_invalid"
    assert [attempt["status"] for attempt in phase1_trace["attempts"]] == [
        "failed",
        "completed",
    ]
    assert result.meta.phase_counts["phase1_fallbacks"] == 1
