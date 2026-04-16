from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from random import SystemRandom
from typing import Any

from pydantic import BaseModel, ValidationError

from phentrieve.config import DEFAULT_MODEL
from phentrieve.llm.config import (
    DEFAULT_LLM_MULTI_VECTOR_AGGREGATION_STRATEGY,
    DEFAULT_LLM_RETRIEVAL_BATCH_SIZE,
    DEFAULT_PROCESS_CLINICAL_TEXT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_PROCESS_CLINICAL_TEXT_NUM_RESULTS_PER_CHUNK,
    DEFAULT_PROVIDER_MAX_TOKENS,
    DEFAULT_PROVIDER_NAME,
    DEFAULT_PROVIDER_RETRY_BACKOFF_MULTIPLIER,
    DEFAULT_PROVIDER_RETRY_INITIAL_BACKOFF_SECONDS,
    DEFAULT_PROVIDER_RETRY_JITTER_SECONDS,
    DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS,
    DEFAULT_PROVIDER_RETRYABLE_STATUS_CODES,
    DEFAULT_PROVIDER_STRUCTURED_RETRIES,
    DEFAULT_PROVIDER_STRUCTURED_RETRY_TOKEN_MULTIPLIER,
    DEFAULT_PROVIDER_TEMPERATURE,
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    DEFAULT_PROVIDER_TRANSIENT_RETRIES,
    DEFAULT_TOOL_QUERY_RESULTS,
)
from phentrieve.llm.types import LLMResponse

logger = logging.getLogger(__name__)
_retry_rng = SystemRandom()


class LLMProvider(ABC):
    model_name: str = ""
    temperature: float = DEFAULT_PROVIDER_TEMPERATURE
    last_usage: dict[str, int]
    last_finish_reason: str | None
    last_request_count: int

    def __init__(self) -> None:
        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0

    @abstractmethod
    def complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def run_structured_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        max_output_tokens: int | None = None,
    ) -> BaseModel:
        raise NotImplementedError


class GeminiStructuredOutputProvider(LLMProvider):
    """Gemini provider supporting both free-text and structured JSON prompts."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        temperature: float = DEFAULT_PROVIDER_TEMPERATURE,
        max_tokens: int = DEFAULT_PROVIDER_MAX_TOKENS,
        timeout_seconds: int = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        transient_retries: int = DEFAULT_PROVIDER_TRANSIENT_RETRIES,
        retryable_status_codes: tuple[
            int, ...
        ] = DEFAULT_PROVIDER_RETRYABLE_STATUS_CODES,
        retry_initial_backoff_seconds: float = (
            DEFAULT_PROVIDER_RETRY_INITIAL_BACKOFF_SECONDS
        ),
        retry_backoff_multiplier: float = DEFAULT_PROVIDER_RETRY_BACKOFF_MULTIPLIER,
        retry_max_backoff_seconds: float = DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS,
        retry_jitter_seconds: float = DEFAULT_PROVIDER_RETRY_JITTER_SECONDS,
        structured_retries: int = DEFAULT_PROVIDER_STRUCTURED_RETRIES,
        structured_retry_token_multiplier: int = (
            DEFAULT_PROVIDER_STRUCTURED_RETRY_TOKEN_MULTIPLIER
        ),
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.transient_retries = transient_retries
        self.retryable_status_codes = retryable_status_codes
        self.retry_initial_backoff_seconds = retry_initial_backoff_seconds
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.retry_max_backoff_seconds = retry_max_backoff_seconds
        self.retry_jitter_seconds = retry_jitter_seconds
        self.structured_retries = structured_retries
        self.structured_retry_token_multiplier = structured_retry_token_multiplier
        self.provider_name = DEFAULT_PROVIDER_NAME
        self._api_key = (
            api_key
            or os.getenv("PHENTRIEVE_GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "Gemini API key not configured. Set PHENTRIEVE_GEMINI_API_KEY, "
                "GEMINI_API_KEY, or GOOGLE_API_KEY."
            )

    def complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires the optional llm dependencies. "
                "Install them with `uv sync --extra llm`."
            ) from exc

        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0
        system_prompt, user_prompt = self._render_messages(messages)

        response, request_count = self._generate_with_transient_retry(
            genai_module=genai,
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                http_options=types.HttpOptions(timeout=self.timeout_seconds * 1000),
            ),
        )
        usage = self._extract_usage(response)
        self.last_usage = usage
        self.last_finish_reason = self._extract_finish_reason(response)
        self.last_request_count = request_count
        response_text = getattr(response, "text", None)
        return LLMResponse(
            content=response_text if isinstance(response_text, str) else None,
            model=self.model_name,
            provider=self.provider_name,
            finish_reason=self.last_finish_reason,
            usage=usage,
            temperature=self.temperature,
        )

    def run_structured_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        max_output_tokens: int | None = None,
    ) -> BaseModel:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires the optional llm dependencies. "
                "Install them with `uv sync --extra llm`."
            ) from exc

        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0
        last_exception: Exception | None = None
        aggregate_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        response_schema = self._build_response_json_schema(response_model)
        output_tokens = max_output_tokens or self.max_tokens

        for attempt in range(1, self.structured_retries + 2):
            response, request_count = self._generate_with_transient_retry(
                genai_module=genai,
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                    temperature=self.temperature,
                    max_output_tokens=output_tokens,
                    http_options=types.HttpOptions(timeout=self.timeout_seconds * 1000),
                ),
                structured=True,
            )

            self.last_request_count += request_count
            response_usage = self._extract_usage(response)
            for key, value in response_usage.items():
                aggregate_usage[key] = int(aggregate_usage.get(key, 0) or 0) + int(
                    value or 0
                )
            self.last_usage = dict(aggregate_usage)
            self.last_finish_reason = self._extract_finish_reason(response)
            try:
                return self._validate_structured_response(
                    response=response,
                    response_model=response_model,
                )
            except (ValidationError, ValueError, RuntimeError) as exc:
                last_exception = exc
                if (
                    attempt > self.structured_retries
                    or not self._is_retryable_structured_error(exc)
                ):
                    raise
                logger.warning(
                    "Gemini structured response validation failed on attempt %d/%d "
                    "(finish_reason=%s): %s",
                    attempt,
                    self.structured_retries + 1,
                    self.last_finish_reason,
                    exc,
                )
                output_tokens = self._next_retry_output_tokens(output_tokens)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Gemini returned no structured response payload.")

    @staticmethod
    def _render_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
        system_parts: list[str] = []
        user_parts: list[str] = []
        assistant_parts: list[str] = []

        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                assistant_parts.append(content)
            else:
                user_parts.append(content)

        transcript_parts: list[str] = []
        if assistant_parts:
            for assistant_content in assistant_parts:
                transcript_parts.append(f"Assistant example:\n{assistant_content}")
        if user_parts:
            transcript_parts.append("\n\n".join(user_parts))

        return "\n\n".join(system_parts), "\n\n".join(transcript_parts)

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "completion_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
        }

    @staticmethod
    def _validate_structured_response(
        *,
        response: Any,
        response_model: type[BaseModel],
    ) -> BaseModel:
        parsed: Any = getattr(response, "parsed", None)
        if isinstance(parsed, response_model):
            return parsed
        if parsed is not None:
            return response_model.model_validate(parsed)

        response_text = getattr(response, "text", None)
        if isinstance(response_text, str) and response_text.strip():
            return response_model.model_validate_json(response_text)

        raise RuntimeError("Gemini returned no structured response payload.")

    @staticmethod
    def _extract_finish_reason(response: Any) -> str | None:
        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason is not None:
            return str(finish_reason)

        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, list) and candidates:
            candidate_finish_reason = getattr(candidates[0], "finish_reason", None)
            if candidate_finish_reason is not None:
                return str(candidate_finish_reason)
        return None

    @staticmethod
    def _is_retryable_structured_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "invalid json" in message
            or "json" in message
            and "eof" in message
            or "unterminated" in message
            or "expecting value" in message
            or "extra data" in message
            or "expecting property name enclosed in double quotes" in message
            or "no structured response payload" in message
        )

    @classmethod
    def _build_response_json_schema(
        cls,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        full_schema = response_model.model_json_schema()
        return cls._compact_json_schema(
            schema=full_schema,
            definitions=full_schema.get("$defs", {}),
        )

    @classmethod
    def _compact_json_schema(
        cls,
        *,
        schema: dict[str, Any],
        definitions: dict[str, Any],
    ) -> dict[str, Any]:
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            definition_name = ref.split("/")[-1]
            definition = definitions.get(definition_name)
            if isinstance(definition, dict):
                return cls._compact_json_schema(
                    schema=definition,
                    definitions=definitions,
                )

        compact: dict[str, Any] = {}
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            compact["type"] = schema_type

        if "enum" in schema and isinstance(schema["enum"], list):
            compact["enum"] = list(schema["enum"])

        description = schema.get("description")
        if isinstance(description, str) and description.strip():
            compact["description"] = description

        properties = schema.get("properties")
        if isinstance(properties, dict):
            compact["properties"] = {
                key: cls._compact_json_schema(
                    schema=value,
                    definitions=definitions,
                )
                for key, value in properties.items()
                if isinstance(value, dict)
            }

        items = schema.get("items")
        if isinstance(items, dict):
            compact["items"] = cls._compact_json_schema(
                schema=items,
                definitions=definitions,
            )

        required = schema.get("required")
        if isinstance(required, list):
            compact["required"] = list(required)

        additional_properties = schema.get("additionalProperties")
        if isinstance(additional_properties, bool):
            compact["additionalProperties"] = additional_properties
        elif isinstance(additional_properties, dict):
            compact["additionalProperties"] = cls._compact_json_schema(
                schema=additional_properties,
                definitions=definitions,
            )

        return compact

    def _next_retry_output_tokens(self, current_output_tokens: int) -> int:
        if self.structured_retry_token_multiplier <= 1:
            return current_output_tokens
        return current_output_tokens * self.structured_retry_token_multiplier

    def _generate_with_transient_retry(
        self,
        *,
        genai_module: Any,
        model: str,
        contents: str,
        config: Any,
        structured: bool = False,
    ) -> tuple[Any, int]:
        last_exception: Exception | None = None
        request_count = 0
        for attempt in range(1, self.transient_retries + 2):
            try:
                with genai_module.Client(api_key=self._api_key) as client:
                    request_count += 1
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    return response, request_count
            except Exception as exc:
                last_exception = exc
                if (
                    attempt > self.transient_retries
                    or not self._is_retryable_provider_error(exc)
                ):
                    raise
                delay_seconds = self._next_transient_retry_delay(attempt)
                logger.warning(
                    "Gemini request failed with transient error on attempt %d/%d "
                    "(structured=%s model=%s status=%s); retrying in %.2fs: %s",
                    attempt,
                    self.transient_retries + 1,
                    structured,
                    self.model_name,
                    getattr(exc, "status_code", None),
                    delay_seconds,
                    exc,
                )
                time.sleep(delay_seconds)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Gemini request failed without returning a response.")

    def _is_retryable_provider_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code in self.retryable_status_codes:
            return True

        message = str(exc).lower()
        return (
            "unavailable" in message
            or "temporarily overloaded" in message
            or "try again later" in message
            or "deadline_exceeded" in message
            or "internal" in message
        )

    def _next_transient_retry_delay(self, attempt: int) -> float:
        exponential_delay = self.retry_initial_backoff_seconds * (
            self.retry_backoff_multiplier ** (attempt - 1)
        )
        bounded_delay = min(exponential_delay, self.retry_max_backoff_seconds)
        jitter = _retry_rng.uniform(0.0, self.retry_jitter_seconds)
        return min(bounded_delay + jitter, self.retry_max_backoff_seconds)


class ToolExecutor:
    def __init__(
        self,
        *,
        retriever: Any | None = None,
        text_processor: Any | None = None,
        max_num_results: int = DEFAULT_TOOL_QUERY_RESULTS,
        retrieval_batch_size: int = DEFAULT_LLM_RETRIEVAL_BATCH_SIZE,
        multi_vector: bool = True,
        multi_vector_aggregation_strategy: str = (
            DEFAULT_LLM_MULTI_VECTOR_AGGREGATION_STRATEGY
        ),
    ) -> None:
        self._retriever = retriever
        self._text_processor = text_processor
        self.max_num_results = max_num_results
        self.retrieval_batch_size = retrieval_batch_size
        self.multi_vector = multi_vector
        self.multi_vector_aggregation_strategy = multi_vector_aggregation_strategy
        self._cached_embedding_model: Any | None = None
        self._cached_retriever: Any | None = None
        self._cached_pipelines: dict[str, Any] = {}

    def warmup(self, *, language: str) -> None:
        self._get_phentrieve_components(language)

    def query_hpo_terms(
        self,
        *,
        query: str,
        num_results: int = DEFAULT_TOOL_QUERY_RESULTS,
    ) -> list[dict[str, Any]]:
        _embedding_model, _pipeline, retriever = self._get_phentrieve_components("en")
        if retriever is None:
            return []

        capped_results = min(int(num_results), self.max_num_results)
        if self.multi_vector and hasattr(retriever, "query_multi_vector"):
            multi_vector_results = retriever.query_multi_vector(
                query,
                n_results=capped_results,
                aggregation_strategy=self.multi_vector_aggregation_strategy,
            )
            return [
                {
                    "hpo_id": result.get("hpo_id", ""),
                    "term_name": result.get("label", ""),
                    "score": float(result.get("similarity", 0.0)),
                }
                for result in multi_vector_results
            ]

        raw = retriever.query(query, n_results=capped_results)
        metadatas = raw.get("metadatas", [[]])[0]
        similarities = (
            raw.get("similarities", [[]])[0] if raw.get("similarities") else []
        )

        results: list[dict[str, Any]] = []
        for index, metadata in enumerate(metadatas):
            similarity = similarities[index] if index < len(similarities) else 0.0
            results.append(
                {
                    "hpo_id": metadata.get("hpo_id", ""),
                    "term_name": metadata.get("label", ""),
                    "score": float(similarity),
                }
            )
        return results

    def query_batch_hpo_terms(
        self,
        *,
        phrases: list[str],
        language: str,
        n_results: int,
    ) -> list[dict[str, Any]]:
        _embedding_model, _pipeline, retriever = self._get_phentrieve_components(
            language
        )
        if retriever is None:
            return []
        if self.multi_vector and hasattr(retriever, "query_multi_vector"):
            results: list[dict[str, Any]] = []
            capped_results = min(int(n_results), self.max_num_results)
            for phrase in phrases:
                multi_vector_results = retriever.query_multi_vector(
                    phrase,
                    n_results=capped_results,
                    aggregation_strategy=self.multi_vector_aggregation_strategy,
                )
                results.append(
                    {
                        "phrase": phrase,
                        "candidates": [
                            {
                                "hpo_id": result.get("hpo_id", ""),
                                "term_name": result.get("label", ""),
                                "score": float(result.get("similarity", 0.0)),
                            }
                            for result in multi_vector_results
                        ],
                    }
                )
            return results
        if hasattr(retriever, "query_batch"):
            batch_results: list[dict[str, Any]] = []
            for start in range(0, len(phrases), self.retrieval_batch_size):
                batch_phrases = phrases[start : start + self.retrieval_batch_size]
                batch_results.extend(
                    list(retriever.query_batch(batch_phrases, n_results=n_results))
                )
            return batch_results
        return [
            {
                "metadatas": [
                    [
                        {
                            "hpo_id": result.get("hpo_id", ""),
                            "label": result.get("term_name", ""),
                        }
                        for result in self.query_hpo_terms(
                            query=phrase,
                            num_results=n_results,
                        )
                    ]
                ],
                "similarities": [
                    [
                        float(result.get("score", 0.0))
                        for result in self.query_hpo_terms(
                            query=phrase,
                            num_results=n_results,
                        )
                    ]
                ],
            }
            for phrase in phrases
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "query_hpo_terms":
            return self.query_hpo_terms(
                query=str(arguments.get("query", "")),
                num_results=int(
                    arguments.get("num_results", DEFAULT_TOOL_QUERY_RESULTS)
                ),
            )
        if tool_name == "process_clinical_text":
            return self._process_clinical_text(
                text=str(arguments.get("text", "")),
                language=str(arguments.get("language", "auto")),
                num_results_per_chunk=int(
                    arguments.get(
                        "num_results_per_chunk",
                        DEFAULT_PROCESS_CLINICAL_TEXT_NUM_RESULTS_PER_CHUNK,
                    )
                ),
                chunk_retrieval_threshold=float(
                    arguments.get(
                        "chunk_retrieval_threshold",
                        DEFAULT_PROCESS_CLINICAL_TEXT_CHUNK_RETRIEVAL_THRESHOLD,
                    )
                ),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def _process_clinical_text(
        self,
        *,
        text: str,
        language: str = "auto",
        num_results_per_chunk: int = (
            DEFAULT_PROCESS_CLINICAL_TEXT_NUM_RESULTS_PER_CHUNK
        ),
        chunk_retrieval_threshold: float = (
            DEFAULT_PROCESS_CLINICAL_TEXT_CHUNK_RETRIEVAL_THRESHOLD
        ),
    ) -> list[dict[str, Any]]:
        if self._text_processor is not None:
            processed = self._text_processor.process(text)
            return [
                {
                    "hpo_id": item.get("hpo_id", ""),
                    "term_name": item.get("term_name", ""),
                    "assertion": item.get("assertion", "affirmed"),
                    "score": item.get("score", 0.0),
                    "evidence_text": item.get("evidence_text"),
                }
                for item in processed
            ]

        try:
            from phentrieve.text_processing.hpo_extraction_orchestrator import (
                orchestrate_hpo_extraction,
            )
        except Exception as exc:  # pragma: no cover - dependency guard
            logger.warning("[TOOL] process_clinical_text unavailable: %s", exc)
            return []

        _embedding_model, pipeline, retriever = self._get_phentrieve_components(
            language
        )
        if retriever is None:
            return []

        chunks = pipeline.process(text)
        if not chunks:
            return []

        chunk_texts = [chunk["text"] for chunk in chunks]
        assertion_statuses: list[str | None] = []
        for chunk in chunks:
            status = chunk.get("status")
            if status is None:
                assertion_statuses.append("affirmed")
            elif isinstance(status, str):
                assertion_statuses.append(status)
            else:
                assertion_statuses.append(status.value)

        aggregated_results, _chunk_results = orchestrate_hpo_extraction(
            text_chunks=chunk_texts,
            retriever=retriever,
            assertion_statuses=assertion_statuses,
            language=language if language != "auto" else "en",
            num_results_per_chunk=num_results_per_chunk,
            chunk_retrieval_threshold=chunk_retrieval_threshold,
        )

        output: list[dict[str, Any]] = []
        for result in aggregated_results:
            assertion = result.get("assertion_status") or "affirmed"
            evidence_parts = [
                attr.get("chunk_text", "")
                for attr in result.get("text_attributions", [])
                if attr.get("chunk_text", "")
            ]
            output.append(
                {
                    "hpo_id": result.get("id", ""),
                    "term_name": result.get("name", ""),
                    "assertion": assertion,
                    "score": result.get("score", 0.0),
                    "evidence_text": "; ".join(evidence_parts)
                    if evidence_parts
                    else "",
                }
            )
        return output

    def _get_phentrieve_components(self, language: str) -> tuple[Any, Any, Any]:
        from phentrieve.config import get_sliding_window_punct_conj_cleaned_config
        from phentrieve.embeddings import load_embedding_model
        from phentrieve.retrieval.dense_retriever import DenseRetriever
        from phentrieve.text_processing.pipeline import TextProcessingPipeline

        language_key = language if language != "auto" else "en"

        if self._retriever is not None:
            self._cached_retriever = self._retriever

        if self._cached_embedding_model is None:
            self._cached_embedding_model = load_embedding_model(DEFAULT_MODEL)

        if language_key not in self._cached_pipelines:
            self._cached_pipelines[language_key] = TextProcessingPipeline(
                language=language_key,
                chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
                assertion_config={"disable": False},
                sbert_model_for_semantic_chunking=self._cached_embedding_model,
            )

        if self._cached_retriever is None:
            self._cached_retriever = DenseRetriever.from_model_name(
                model=self._cached_embedding_model,
                model_name=DEFAULT_MODEL,
                multi_vector=self.multi_vector,
            )

        return (
            self._cached_embedding_model,
            self._cached_pipelines[language_key],
            self._cached_retriever,
        )


def get_llm_provider(
    *,
    llm_model: str,
    api_key: str | None = None,
) -> LLMProvider:
    resolved_provider = os.getenv("PHENTRIEVE_LLM_PROVIDER", DEFAULT_PROVIDER_NAME)
    if resolved_provider.strip().lower() != DEFAULT_PROVIDER_NAME:
        raise ValueError(
            f"Gemini-only provider factory does not support provider "
            f"{resolved_provider!r}."
        )

    if "/" in llm_model and not llm_model.startswith("gemini/"):
        raise ValueError(
            f"Gemini-only provider factory does not support model {llm_model!r}."
        )

    return GeminiStructuredOutputProvider(model_name=llm_model, api_key=api_key)
