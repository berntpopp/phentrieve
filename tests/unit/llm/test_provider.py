from __future__ import annotations

import inspect
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest

from phentrieve.llm import config as llm_config
from phentrieve.llm import provider as provider_module
from phentrieve.llm.provider import (
    GeminiStructuredOutputProvider,
    LLMProvider,
    ToolExecutor,
    get_llm_provider,
)
from phentrieve.llm.types import (
    LLMExtractedPhenotype,
    LLMExtractedPhenotypes,
    LLMMeta,
    LLMPipelineConfig,
)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool]] = []
        self.multi_vector_calls: list[tuple[str, int, str]] = []

    def query(self, query: str, n_results: int, include_similarities: bool = False):
        self.calls.append((query, n_results, include_similarities))
        return {
            "metadatas": [[{"hpo_id": "HP:0001250", "label": "Seizure"}]],
            "similarities": [[0.95]],
        }

    def query_batch(
        self, texts: list[str], n_results: int, include_similarities: bool = True
    ):
        self.calls.append(("__batch__", len(texts), include_similarities))
        return [
            {
                "metadatas": [[{"hpo_id": "HP:0001250", "label": "Seizure"}]],
                "similarities": [[0.95]],
            }
            for _ in texts
        ]

    def query_multi_vector(
        self,
        text: str,
        n_results: int = 10,
        aggregation_strategy: str = "label_synonyms_max",
        component_weights: dict[str, float] | None = None,
        custom_formula: str | None = None,
    ):
        del component_weights, custom_formula
        self.multi_vector_calls.append((text, n_results, aggregation_strategy))
        return [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.95,
            }
        ]


class FakeTextProcessor:
    def process(self, text: str) -> list[dict[str, Any]]:
        return [
            {
                "hpo_id": "HP:0001250",
                "term_name": "Seizure",
                "assertion": "present",
                "score": 0.95,
                "evidence_text": text,
            }
        ]


def test_llm_pipeline_config_includes_provider() -> None:
    config = LLMPipelineConfig(
        provider="ollama",
        model="qwen3.5:35b",
        mode="two_phase",
        language="en",
    )

    assert config.provider == "ollama"
    assert config.model == "qwen3.5:35b"


def test_llm_meta_includes_provider_identity() -> None:
    meta = LLMMeta(
        llm_provider="ollama",
        llm_model="qwen3.5:35b",
        llm_mode="two_phase",
    )

    assert meta.llm_provider == "ollama"


def test_provider_config_exposes_ollama_defaults() -> None:
    assert llm_config.DEFAULT_PROVIDER_NAME == "gemini"
    assert llm_config.DEFAULT_OLLAMA_BASE_URL == "http://localhost:11434"
    assert llm_config.DEFAULT_OLLAMA_TIMEOUT_SECONDS == 300


def test_get_llm_provider_defaults_to_gemini(monkeypatch) -> None:
    monkeypatch.setenv("PHENTRIEVE_GEMINI_API_KEY", "test-key")
    provider = get_llm_provider(llm_model="gemini-2.5-flash")

    assert provider.model_name == "gemini-2.5-flash"


def test_get_llm_provider_rejects_non_gemini_models() -> None:
    with pytest.raises(ValueError, match="Gemini-only"):
        get_llm_provider(llm_model="openai/gpt-4o")


def test_llm_provider_complete_signature_is_message_only() -> None:
    signature = inspect.signature(LLMProvider.complete)

    assert list(signature.parameters) == ["self", "messages"]


def test_tool_executor_caps_query_results_and_formats_matches() -> None:
    retriever = FakeRetriever()
    executor = ToolExecutor(retriever=retriever, max_num_results=3, multi_vector=False)

    result = executor.execute(
        "query_hpo_terms",
        {"query": "seizures", "num_results": 10},
    )

    assert retriever.calls == [("seizures", 3, False)]
    assert result == [{"hpo_id": "HP:0001250", "term_name": "Seizure", "score": 0.95}]


def test_tool_executor_process_clinical_text_uses_injected_processor() -> None:
    executor = ToolExecutor(text_processor=FakeTextProcessor())

    result = executor.execute(
        "process_clinical_text",
        {"text": "Patient has recurrent seizures.", "language": "en"},
    )

    assert result == [
        {
            "hpo_id": "HP:0001250",
            "term_name": "Seizure",
            "assertion": "present",
            "score": 0.95,
            "evidence_text": "Patient has recurrent seizures.",
        }
    ]


def test_tool_executor_chunks_large_batch_queries() -> None:
    retriever = FakeRetriever()
    executor = ToolExecutor(
        retriever=retriever,
        retrieval_batch_size=2,
        multi_vector=False,
    )

    result = executor.query_batch_hpo_terms(
        phrases=["one", "two", "three", "four", "five"],
        language="en",
        n_results=5,
    )

    assert retriever.calls == [
        ("__batch__", 2, True),
        ("__batch__", 2, True),
        ("__batch__", 1, True),
    ]
    assert len(result) == 5


def test_tool_executor_uses_multi_vector_queries_when_enabled() -> None:
    retriever = FakeRetriever()
    executor = ToolExecutor(retriever=retriever, multi_vector=True)

    result = executor.query_batch_hpo_terms(
        phrases=["seizures", "ataxia"],
        language="en",
        n_results=5,
    )

    assert retriever.multi_vector_calls == [
        ("seizures", 5, "label_synonyms_max"),
        ("ataxia", 5, "label_synonyms_max"),
    ]
    assert result == [
        {
            "phrase": "seizures",
            "candidates": [
                {"hpo_id": "HP:0001250", "term_name": "Seizure", "score": 0.95}
            ],
        },
        {
            "phrase": "ataxia",
            "candidates": [
                {"hpo_id": "HP:0001250", "term_name": "Seizure", "score": 0.95}
            ],
        },
    ]


def test_process_clinical_text_defaults_come_from_config(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_process_clinical_text(
        self,
        *,
        text: str,
        language: str = "auto",
        num_results_per_chunk: int = 0,
        chunk_retrieval_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        captured.update(
            {
                "text": text,
                "language": language,
                "num_results_per_chunk": num_results_per_chunk,
                "chunk_retrieval_threshold": chunk_retrieval_threshold,
            }
        )
        return []

    monkeypatch.setattr(
        ToolExecutor,
        "_process_clinical_text",
        fake_process_clinical_text,
        raising=True,
    )

    executor = ToolExecutor()
    executor.execute("process_clinical_text", {"text": "patient has fever"})

    assert captured["num_results_per_chunk"] == (
        llm_config.DEFAULT_PROCESS_CLINICAL_TEXT_NUM_RESULTS_PER_CHUNK
    )
    assert captured["chunk_retrieval_threshold"] == (
        llm_config.DEFAULT_PROCESS_CLINICAL_TEXT_CHUNK_RETRIEVAL_THRESHOLD
    )


def test_tool_executor_rejects_unknown_tool() -> None:
    executor = ToolExecutor()

    with pytest.raises(ValueError, match="Unknown tool"):
        executor.execute("not_a_tool", {})


def test_llm_provider_tracks_usage_per_thread() -> None:
    barrier = Barrier(2)

    class ThreadAwareProvider(LLMProvider):
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
            del system_prompt, max_output_tokens
            marker = int(user_prompt)
            self.last_usage = {
                "prompt_tokens": marker,
                "completion_tokens": marker + 1,
                "total_tokens": marker + 2,
            }
            self.last_request_count = marker
            barrier.wait()
            return response_model.model_validate({"phenotypes": []})

    provider = ThreadAwareProvider()

    def run_prompt(marker: int) -> tuple[dict[str, int], int]:
        provider.run_structured_prompt(
            system_prompt="system",
            user_prompt=str(marker),
            response_model=LLMExtractedPhenotypes,
        )
        return provider.last_usage, provider.last_request_count

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(run_prompt, 1)
        second_future = executor.submit(run_prompt, 2)

    assert first_future.result() == (
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        1,
    )
    assert second_future.result() == (
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 4},
        2,
    )


def test_llm_extra_matches_supported_runtime_dependencies() -> None:
    with open("pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    llm_extra = pyproject["project"]["optional-dependencies"]["llm"]

    assert any(dep.startswith("google-genai") for dep in llm_extra)
    assert not any(dep.startswith("openai") for dep in llm_extra)


def test_packaged_prompt_families_exclude_agentic_judge() -> None:
    with open("pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    prompt_globs = pyproject["tool"]["setuptools"]["package-data"][
        "phentrieve.llm.prompts"
    ]

    assert not any("agentic_judge" in pattern for pattern in prompt_globs)


class _FakeUsageMetadata:
    def __init__(
        self,
        prompt: int,
        completion: int,
        total: int,
        *,
        thoughts: int = 0,
        cached: int = 0,
    ) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = completion
        self.total_token_count = total
        self.thoughts_token_count = thoughts
        self.cached_content_token_count = cached


class _FakeResponse:
    def __init__(
        self,
        *,
        text: str | None = None,
        parsed: Any | None = None,
        finish_reason: str | None = "STOP",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        thoughts_tokens: int = 0,
        cached_content_tokens: int = 0,
    ) -> None:
        self.text = text
        self.parsed = parsed
        self.finish_reason = finish_reason
        self.usage_metadata = _FakeUsageMetadata(
            prompt_tokens,
            completion_tokens,
            total_tokens,
            thoughts=thoughts_tokens,
            cached=cached_content_tokens,
        )


def _install_fake_google_genai(monkeypatch, responses: list[_FakeResponse]):
    class _FakeServerError(Exception):
        def __init__(self, status_code: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code

    class _FakeClientError(Exception):
        def __init__(self, status_code: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeHttpOptions:
        def __init__(self, *, timeout: int) -> None:
            self.timeout = timeout

    class _FakeCountTokensConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeModels:
        def __init__(self, queued_responses: list[_FakeResponse | Exception]) -> None:
            self._responses = list(queued_responses)
            self.calls: list[dict[str, Any]] = []
            self.count_token_calls: list[dict[str, Any]] = []

        def generate_content(self, *, model, contents, config):
            self.calls.append(
                {
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
            )
            next_item = self._responses.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return next_item

        def count_tokens(self, *, model, contents, config=None):
            self.count_token_calls.append(
                {
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
            )
            return SimpleNamespace(total_tokens=1234)

    fake_models = _FakeModels(responses)

    class _FakeClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.models = fake_models

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    types_module = ModuleType("google.genai.types")
    errors_module = ModuleType("google.genai.errors")
    types_module.GenerateContentConfig = _FakeGenerateContentConfig
    types_module.CountTokensConfig = _FakeCountTokensConfig
    types_module.HttpOptions = _FakeHttpOptions
    genai_module.Client = _FakeClient
    genai_module.types = types_module
    genai_module.errors = errors_module
    errors_module.ServerError = _FakeServerError
    errors_module.ClientError = _FakeClientError
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    monkeypatch.setitem(sys.modules, "google.genai.errors", errors_module)
    fake_models.server_error = _FakeServerError
    fake_models.client_error = _FakeClientError
    return fake_models


def test_structured_prompt_uses_json_schema_and_manual_validation(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            _FakeResponse(
                text='{"phenotypes":[{"phrase":"recurrent seizures","category":"Abnormal"}]}',
                parsed=None,
                prompt_tokens=12,
                completion_tokens=8,
                total_tokens=20,
            )
        ],
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
        max_output_tokens=8192,
    )

    config_kwargs = fake_models.calls[0]["config"].kwargs
    assert result.phenotypes[0].phrase == "recurrent seizures"
    assert config_kwargs["response_mime_type"] == "application/json"
    assert "response_json_schema" in config_kwargs
    assert "response_schema" not in config_kwargs
    assert "title" not in config_kwargs["response_json_schema"]
    assert "description" not in config_kwargs["response_json_schema"]
    assert config_kwargs["response_json_schema"]["properties"]["phenotypes"]["items"][
        "properties"
    ]["category"]["enum"] == [
        "Abnormal",
        "Normal",
        "Suspected",
        "Family_History",
        "Other",
    ]
    assert (
        "maxItems"
        not in config_kwargs["response_json_schema"]["properties"]["phenotypes"]
    )
    assert (
        "maxLength"
        not in config_kwargs["response_json_schema"]["properties"]["phenotypes"][
            "items"
        ]["properties"]["phrase"]
    )
    assert (
        config_kwargs["response_json_schema"]["properties"]["phenotypes"]["description"]
        == "Distinct phenotype phrases extracted from the clinical text."
    )
    assert config_kwargs["response_json_schema"]["properties"]["phenotypes"]["items"][
        "properties"
    ]["phrase"]["description"].startswith("A concise phenotype phrase")
    assert config_kwargs["max_output_tokens"] == 8192
    assert provider.last_usage["total_tokens"] == 20


def test_structured_prompt_forwards_seed_and_extended_usage(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            _FakeResponse(
                parsed=LLMExtractedPhenotypes(
                    phenotypes=[
                        LLMExtractedPhenotype(
                            phrase="recurrent seizures",
                            category="Abnormal",
                        )
                    ]
                ),
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=23,
                thoughts_tokens=5,
                cached_content_tokens=3,
            )
        ],
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
        seed=42,
    )

    provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    config_kwargs = fake_models.calls[0]["config"].kwargs
    assert config_kwargs["seed"] == 42
    assert provider.last_usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 23,
        "thoughts_tokens": 5,
        "cached_content_tokens": 3,
    }


def test_count_tokens_uses_sdk_count_tokens_api(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(monkeypatch, [])
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    counts = provider.count_tokens(
        system_prompt="system prompt",
        user_prompt="user prompt",
    )

    assert counts == {
        "prompt_tokens": 1234,
        "completion_tokens": 0,
        "total_tokens": 1234,
    }
    assert len(fake_models.count_token_calls) == 1
    assert fake_models.count_token_calls[0]["model"] == "gemini-2.5-flash"
    assert (
        fake_models.count_token_calls[0]["contents"] == "system prompt\n\nuser prompt"
    )
    assert fake_models.count_token_calls[0]["config"] is None


def test_structured_prompt_retries_after_invalid_json(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            _FakeResponse(
                text='{"phenotypes":[{"phrase":"recurrent seizures"',
                parsed=None,
                finish_reason="MAX_TOKENS",
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
            ),
            _FakeResponse(
                text='{"phenotypes":[{"phrase":"recurrent seizures","category":"Abnormal"}]}',
                parsed=None,
                finish_reason="STOP",
                prompt_tokens=11,
                completion_tokens=9,
                total_tokens=20,
            ),
        ],
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
        structured_retries=1,
    )

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    assert result.phenotypes[0].category == "Abnormal"
    assert len(fake_models.calls) == 2
    assert fake_models.calls[0]["config"].kwargs["max_output_tokens"] == 8192
    assert fake_models.calls[1]["config"].kwargs["max_output_tokens"] == 16384
    assert provider.last_finish_reason == "STOP"
    assert provider.last_usage["total_tokens"] == 40


def test_structured_prompt_retries_after_extra_data_json_error(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            _FakeResponse(
                text='{"phenotypes":[]} trailing',
                parsed=None,
                finish_reason="STOP",
                prompt_tokens=9,
                completion_tokens=5,
                total_tokens=14,
            ),
            _FakeResponse(
                text='{"phenotypes":[{"phrase":"recurrent seizures","category":"Abnormal"}]}',
                parsed=None,
                finish_reason="STOP",
                prompt_tokens=11,
                completion_tokens=9,
                total_tokens=20,
            ),
        ],
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
        structured_retries=1,
    )

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    assert result.phenotypes[0].category == "Abnormal"
    assert len(fake_models.calls) == 2


def test_structured_prompt_uses_single_relaxed_schema_without_size_constraints(
    monkeypatch,
) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            _FakeResponse(
                text='{"phenotypes":[{"phrase":"recurrent seizures","category":"Abnormal"}]}',
                parsed=None,
                finish_reason="STOP",
                prompt_tokens=11,
                completion_tokens=9,
                total_tokens=20,
            )
        ],
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    schema = fake_models.calls[0]["config"].kwargs["response_json_schema"]
    assert result.phenotypes[0].category == "Abnormal"
    assert len(fake_models.calls) == 1
    assert "maxItems" not in schema["properties"]["phenotypes"]
    assert (
        "maxLength"
        not in schema["properties"]["phenotypes"]["items"]["properties"]["phrase"]
    )
    assert schema["properties"]["phenotypes"]["items"]["properties"]["phrase"][
        "description"
    ]


def test_structured_prompt_accepts_long_phrase_when_served_schema_is_relaxed(
    monkeypatch,
) -> None:
    long_phrase = "phenotype detail " * 76
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            _FakeResponse(
                text=(
                    '{"phenotypes":[{"phrase":"'
                    + long_phrase
                    + '","category":"Other"}]}'
                ),
                parsed=None,
                finish_reason="STOP",
                prompt_tokens=11,
                completion_tokens=9,
                total_tokens=20,
            )
        ],
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    assert result.phenotypes[0].phrase == long_phrase
    assert len(fake_models.calls) == 1


def test_structured_prompt_does_not_retry_non_retryable_schema_client_error(
    monkeypatch,
) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [_FakeResponse()],
    )
    fake_models._responses[0] = fake_models.client_error(
        400, "too many states for serving"
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    with pytest.raises(Exception, match="too many states for serving"):
        provider.run_structured_prompt(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractedPhenotypes,
        )

    assert len(fake_models.calls) == 1


def test_complete_retries_after_transient_server_error(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            _FakeResponse(),
            _FakeResponse(
                text="ok",
                finish_reason="STOP",
                prompt_tokens=7,
                completion_tokens=3,
                total_tokens=10,
            ),
        ],
    )
    fake_models._responses[0] = fake_models.server_error(503, "temporarily unavailable")
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        provider_module,
        "time",
        SimpleNamespace(sleep=sleep_calls.append),
        raising=False,
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    result = provider.complete([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert len(fake_models.calls) == 2
    assert sleep_calls == [pytest.approx(sleep_calls[0])]
    assert sleep_calls[0] > 0.0


def test_structured_prompt_retries_after_transient_server_error(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            _FakeResponse(),
            _FakeResponse(
                text='{"phenotypes":[{"phrase":"recurrent seizures","category":"Abnormal"}]}',
                parsed=None,
                finish_reason="STOP",
                prompt_tokens=11,
                completion_tokens=9,
                total_tokens=20,
            ),
        ],
    )
    fake_models._responses[0] = fake_models.server_error(503, "temporarily unavailable")
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        provider_module,
        "time",
        SimpleNamespace(sleep=sleep_calls.append),
        raising=False,
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    assert result.phenotypes[0].phrase == "recurrent seizures"
    assert len(fake_models.calls) == 2
    assert sleep_calls == [pytest.approx(sleep_calls[0])]
    assert sleep_calls[0] > 0.0


def test_complete_does_not_retry_non_retryable_client_error(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [_FakeResponse()],
    )
    fake_models._responses[0] = fake_models.client_error(400, "bad request")
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        provider_module,
        "time",
        SimpleNamespace(sleep=sleep_calls.append),
        raising=False,
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    with pytest.raises(Exception, match="bad request"):
        provider.complete([{"role": "user", "content": "hello"}])

    assert len(fake_models.calls) == 1
    assert sleep_calls == []


def test_complete_retries_after_rate_limit_client_error(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            _FakeResponse(),
            _FakeResponse(
                text="ok",
                finish_reason="STOP",
                prompt_tokens=6,
                completion_tokens=2,
                total_tokens=8,
            ),
        ],
    )
    fake_models._responses[0] = fake_models.client_error(429, "rate limited")
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        provider_module,
        "time",
        SimpleNamespace(sleep=sleep_calls.append),
        raising=False,
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    result = provider.complete([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert len(fake_models.calls) == 2
    assert sleep_calls[0] > 0.0


def test_complete_retries_after_transport_error(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            httpx.ConnectError("network is unreachable"),
            _FakeResponse(
                text="ok",
                finish_reason="STOP",
                prompt_tokens=6,
                completion_tokens=2,
                total_tokens=8,
            ),
        ],
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        provider_module,
        "time",
        SimpleNamespace(sleep=sleep_calls.append),
        raising=False,
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    result = provider.complete([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert len(fake_models.calls) == 2
    assert sleep_calls[0] > 0.0


def test_structured_prompt_retries_after_transport_error(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [
            httpx.ConnectError("network is unreachable"),
            _FakeResponse(
                text='{"phenotypes":[{"phrase":"recurrent seizures","category":"Abnormal"}]}',
                parsed=None,
                finish_reason="STOP",
                prompt_tokens=11,
                completion_tokens=9,
                total_tokens=20,
            ),
        ],
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        provider_module,
        "time",
        SimpleNamespace(sleep=sleep_calls.append),
        raising=False,
    )
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    assert result.phenotypes[0].phrase == "recurrent seizures"
    assert len(fake_models.calls) == 2
    assert sleep_calls[0] > 0.0


def test_complete_clears_stale_metadata_on_failure(monkeypatch) -> None:
    fake_models = _install_fake_google_genai(
        monkeypatch,
        [_FakeResponse()],
    )
    fake_models._responses[0] = fake_models.client_error(400, "bad request")
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
    )
    provider.last_usage = {"total_tokens": 99}
    provider.last_finish_reason = "STOP"

    with pytest.raises(Exception, match="bad request"):
        provider.complete([{"role": "user", "content": "hello"}])

    assert len(fake_models.calls) == 1
    assert provider.last_usage == {}
    assert provider.last_finish_reason is None


def test_transient_retry_delay_is_capped_after_jitter(monkeypatch) -> None:
    class _FixedRng:
        @staticmethod
        def uniform(_start: float, end: float) -> float:
            return end

    monkeypatch.setattr(provider_module, "_retry_rng", _FixedRng())
    provider = GeminiStructuredOutputProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
        retry_initial_backoff_seconds=8.0,
        retry_backoff_multiplier=2.0,
        retry_max_backoff_seconds=8.0,
        retry_jitter_seconds=0.25,
    )

    delay = provider._next_transient_retry_delay(attempt=3)

    assert delay == 8.0
