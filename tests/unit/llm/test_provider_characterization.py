from __future__ import annotations

from typing import Any

import httpx
import pytest

from phentrieve.llm.provider import (
    GeminiStructuredOutputProvider,
    OllamaStructuredOutputProvider,
    ToolExecutor,
    build_response_json_schema,
    resolve_llm_provider_request,
)
from phentrieve.llm.types import LLMExtractedPhenotypes


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def query(self, query: str, n_results: int) -> dict[str, Any]:
        self.calls.append((query, n_results))
        return {
            "metadatas": [[{"hpo_id": "HP:0001250", "label": "Seizure"}]],
            "similarities": [[0.95]],
        }


class FakeOllamaResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "message": {"content": '{"phenotypes":[]}'},
            "prompt_eval_count": 7,
            "eval_count": 3,
            "done_reason": "stop",
        }


def test_resolve_llm_provider_request_uses_model_prefix_and_preserves_options() -> None:
    request = resolve_llm_provider_request(
        llm_provider=None,
        llm_model="openai/gpt-5.4-mini",
        llm_base_url="https://llm.example.test/v1",
        api_key="test-key",
        seed=42,
    )

    assert request.provider == "openai"
    assert request.model == "gpt-5.4-mini"
    assert request.base_url == "https://llm.example.test/v1"
    assert request.api_key == "test-key"
    assert request.seed == 42


def test_ollama_structured_prompt_request_shape_and_usage(monkeypatch) -> None:
    requests: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *, timeout: int) -> None:
            self.timeout = timeout

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, json: dict[str, Any]) -> FakeOllamaResponse:
            requests.append({"url": url, "json": json, "timeout": self.timeout})
            return FakeOllamaResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    provider = OllamaStructuredOutputProvider(
        model_name="qwen3.5:35b",
        base_url="http://localhost:11434/",
        seed=13,
        timeout_seconds=123,
    )

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
        max_output_tokens=2048,
    )

    payload = requests[0]["json"]
    assert result.phenotypes == []
    assert requests[0]["url"] == "http://localhost:11434/api/chat"
    assert requests[0]["timeout"] == 123
    assert payload["model"] == "qwen3.5:35b"
    assert payload["messages"][0] == {"role": "system", "content": "system"}
    assert "Return JSON only" in payload["messages"][1]["content"]
    assert payload["format"]["type"] == "object"
    assert payload["options"] == {
        "temperature": 0,
        "num_predict": 2048,
        "seed": 13,
    }
    assert provider.last_usage == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
    assert provider.last_finish_reason == "stop"


def test_tool_executor_caps_query_results_and_normalizes_retriever_output() -> None:
    retriever = FakeRetriever()
    executor = ToolExecutor(retriever=retriever, max_num_results=2, multi_vector=False)

    result = executor.execute(
        "query_hpo_terms",
        {"query": "seizures", "num_results": 10},
    )

    assert retriever.calls == [("seizures", 2)]
    assert result == [{"hpo_id": "HP:0001250", "term_name": "Seizure", "score": 0.95}]


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ConnectError("network unavailable"), True),
        (type("StatusError", (Exception,), {"status_code": 429})("rate limited"), True),
        (type("StatusError", (Exception,), {"status_code": 400})("bad request"), False),
    ],
)
def test_gemini_provider_retryable_error_classification(
    exc: Exception,
    expected: bool,
) -> None:
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    assert provider._is_retryable_provider_error(exc) is expected


def test_structured_error_classification_does_not_retry_refusals() -> None:
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    assert provider._is_retryable_structured_error(ValueError("invalid json")) is True
    assert provider._is_retryable_structured_error(RuntimeError("refusal")) is False


def test_response_json_schema_is_compacted_for_provider_requests() -> None:
    schema = build_response_json_schema(LLMExtractedPhenotypes)

    phenotype_items = schema["properties"]["phenotypes"]["items"]
    assert schema["type"] == "object"
    assert "title" not in schema
    assert "maxItems" not in schema["properties"]["phenotypes"]
    assert "maxLength" not in phenotype_items["properties"]["phrase"]
    assert phenotype_items["properties"]["category"]["enum"] == [
        "Abnormal",
        "Normal",
        "Suspected",
        "Family_History",
        "Other",
    ]
    assert phenotype_items["additionalProperties"] is False
