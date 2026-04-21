from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from random import SystemRandom
from threading import local
from typing import Any

import httpx
from pydantic import BaseModel

from phentrieve.config import DEFAULT_MODEL
from phentrieve.llm.config import (
    DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_MULTI_VECTOR_AGGREGATION_STRATEGY,
    DEFAULT_LLM_RETRIEVAL_BATCH_SIZE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
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
    SUPPORTED_PROVIDER_NAMES,
)
from phentrieve.llm.types import LLMResponse

logger = logging.getLogger(__name__)
_retry_rng = SystemRandom()


class LLMProvider(ABC):
    provider_name: str = ""
    model_name: str = ""
    temperature: float = DEFAULT_PROVIDER_TEMPERATURE

    def __init__(self) -> None:
        self._thread_state = local()
        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0
        self.last_structured_payload = None
        self.structured_retries = DEFAULT_PROVIDER_STRUCTURED_RETRIES
        self.structured_retry_token_multiplier = (
            DEFAULT_PROVIDER_STRUCTURED_RETRY_TOKEN_MULTIPLIER
        )

    @property
    def last_usage(self) -> dict[str, int]:
        return dict(getattr(self._thread_state, "last_usage", {}) or {})

    @last_usage.setter
    def last_usage(self, value: dict[str, int]) -> None:
        self._thread_state.last_usage = dict(value or {})

    @property
    def last_finish_reason(self) -> str | None:
        return getattr(self._thread_state, "last_finish_reason", None)

    @last_finish_reason.setter
    def last_finish_reason(self, value: str | None) -> None:
        self._thread_state.last_finish_reason = value

    @property
    def last_request_count(self) -> int:
        return int(getattr(self._thread_state, "last_request_count", 0) or 0)

    @last_request_count.setter
    def last_request_count(self, value: int) -> None:
        self._thread_state.last_request_count = int(value or 0)

    @property
    def last_structured_payload(self) -> dict[str, Any] | None:
        payload = getattr(self._thread_state, "last_structured_payload", None)
        return dict(payload) if isinstance(payload, dict) else None

    @last_structured_payload.setter
    def last_structured_payload(self, value: dict[str, Any] | None) -> None:
        self._thread_state.last_structured_payload = (
            dict(value) if isinstance(value, dict) else None
        )

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

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        raise NotImplementedError

    def _run_structured_with_recovery(
        self,
        *,
        invoke: Callable[[int], tuple[Any, int]],
        parse: Callable[[Any], BaseModel],
        initial_output_tokens: int,
        max_output_tokens: int,
        structured_retries: int,
        structured_retry_token_multiplier: int,
    ) -> BaseModel:
        current_output_tokens = initial_output_tokens
        aggregate_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        last_exception: Exception | None = None
        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0
        self.last_structured_payload = None

        for attempt in range(1, structured_retries + 2):
            response, request_count = invoke(current_output_tokens)
            self._record_structured_attempt(
                response=response,
                request_count=request_count,
                aggregate_usage=aggregate_usage,
            )
            try:
                parsed = parse(response)
                self.last_structured_payload = parsed.model_dump(mode="json")
                return parsed
            except Exception as exc:
                last_exception = exc
                if (
                    attempt > structured_retries
                    or not self._is_retryable_structured_error(exc)
                ):
                    raise
                logger.warning(
                    "%s structured response validation failed on attempt %d/%d "
                    "(finish_reason=%s): %s",
                    self.__class__.__name__.removesuffix("StructuredOutputProvider"),
                    attempt,
                    structured_retries + 1,
                    self.last_finish_reason,
                    exc,
                )
                current_output_tokens = self._next_retry_output_tokens(
                    current_output_tokens,
                    max_output_tokens=max_output_tokens,
                    retry_token_multiplier=structured_retry_token_multiplier,
                )

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Structured response request failed without a payload.")

    def _record_structured_attempt(
        self,
        *,
        response: Any,
        request_count: int,
        aggregate_usage: dict[str, int],
    ) -> None:
        self.last_request_count += int(request_count or 0)
        response_usage = self._extract_structured_usage(response)
        for key, value in response_usage.items():
            aggregate_usage[key] = int(aggregate_usage.get(key, 0) or 0) + int(
                value or 0
            )
        self.last_usage = dict(aggregate_usage)
        self.last_finish_reason = self._extract_structured_finish_reason(response)

    def _extract_structured_usage(self, response: Any) -> dict[str, int]:
        extractor = getattr(self, "_extract_usage", None)
        if callable(extractor):
            usage = extractor(response)
            if isinstance(usage, dict):
                return usage
        return {}

    def _extract_structured_finish_reason(self, response: Any) -> str | None:
        extractor = getattr(self, "_extract_finish_reason", None)
        if not callable(extractor):
            return None
        finish_reason = extractor(response)
        return str(finish_reason) if finish_reason is not None else None

    def _is_retryable_structured_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        if any(
            token in message
            for token in ("refusal", "billing", "unsupported", "non-json")
        ):
            return False
        return (
            "invalid json" in message
            or "json_invalid" in message
            or "unterminated" in message
            or "eof while parsing" in message
            or "expecting value" in message
            or "extra data" in message
            or "no structured response payload" in message
        )

    def _next_retry_output_tokens(
        self,
        current_output_tokens: int,
        *,
        max_output_tokens: int,
        retry_token_multiplier: int,
    ) -> int:
        if retry_token_multiplier <= 1:
            return current_output_tokens
        return min(
            current_output_tokens * retry_token_multiplier,
            max_output_tokens,
        )


@dataclass(frozen=True)
class ResolvedLLMProviderRequest:
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    seed: int | None = None


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


def build_response_json_schema(response_model: type[BaseModel]) -> dict[str, Any]:
    full_schema = response_model.model_json_schema()
    return compact_json_schema(
        schema=full_schema,
        definitions=full_schema.get("$defs", {}),
    )


def compact_json_schema(
    *,
    schema: dict[str, Any],
    definitions: dict[str, Any],
) -> dict[str, Any]:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        definition_name = ref.split("/")[-1]
        definition = definitions.get(definition_name)
        if isinstance(definition, dict):
            return compact_json_schema(
                schema=definition,
                definitions=definitions,
            )

    compact: dict[str, Any] = {}
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        compact["type"] = schema_type
    elif isinstance(schema_type, list) and schema_type:
        compact["type"] = list(schema_type)

    if "enum" in schema and isinstance(schema["enum"], list):
        compact["enum"] = list(schema["enum"])

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        compact["anyOf"] = [
            compact_json_schema(
                schema=item,
                definitions=definitions,
            )
            for item in any_of
            if isinstance(item, dict)
        ]

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        compact["oneOf"] = [
            compact_json_schema(
                schema=item,
                definitions=definitions,
            )
            for item in one_of
            if isinstance(item, dict)
        ]

    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        compact["description"] = description

    properties = schema.get("properties")
    if isinstance(properties, dict):
        compact["properties"] = {
            key: compact_json_schema(
                schema=value,
                definitions=definitions,
            )
            for key, value in properties.items()
            if isinstance(value, dict)
        }

    items = schema.get("items")
    if isinstance(items, dict):
        compact["items"] = compact_json_schema(
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
        compact["additionalProperties"] = compact_json_schema(
            schema=additional_properties,
            definitions=definitions,
        )
    elif compact.get("type") == "object" or "properties" in compact:
        compact["additionalProperties"] = False

    return compact


class GeminiStructuredOutputProvider(LLMProvider):
    """Gemini provider supporting both free-text and structured JSON prompts."""

    token_count_source = "exact"  # noqa: S105

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        seed: int | None = None,
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
        self.seed = seed
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
        system_prompt, user_prompt = _render_messages(messages)

        response, request_count = self._generate_with_transient_retry(
            genai_module=genai,
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                seed=self.seed,
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

        response_schema = build_response_json_schema(response_model)
        initial_output_tokens = max_output_tokens or self.max_tokens
        retry_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else max(self.max_tokens, DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS)
        )
        return self._run_structured_with_recovery(
            invoke=lambda output_tokens: self._generate_with_transient_retry(
                genai_module=genai,
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                    temperature=self.temperature,
                    max_output_tokens=output_tokens,
                    seed=self.seed,
                    http_options=types.HttpOptions(timeout=self.timeout_seconds * 1000),
                ),
                structured=True,
            ),
            parse=lambda response: self._validate_structured_response(
                response=response,
                response_model=response_model,
            ),
            initial_output_tokens=initial_output_tokens,
            max_output_tokens=retry_output_tokens,
            structured_retries=self.structured_retries,
            structured_retry_token_multiplier=self.structured_retry_token_multiplier,
        )

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires the optional llm dependencies. "
                "Install them with `uv sync --extra llm`."
            ) from exc

        contents = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        with genai.Client(api_key=self._api_key) as client:
            response = client.models.count_tokens(
                model=self.model_name,
                contents=contents,
            )
        total_tokens = int(getattr(response, "total_tokens", 0) or 0)
        return {
            "prompt_tokens": total_tokens,
            "completion_tokens": 0,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "completion_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
            "thoughts_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
            "cached_content_tokens": int(
                getattr(usage, "cached_content_token_count", 0) or 0
            ),
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

    def _next_retry_output_tokens(
        self,
        current_output_tokens: int,
        *,
        max_output_tokens: int,
        retry_token_multiplier: int,
    ) -> int:
        if retry_token_multiplier <= 1:
            return current_output_tokens
        return min(
            current_output_tokens * retry_token_multiplier,
            max_output_tokens,
        )

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
        if isinstance(exc, httpx.TransportError):
            return True

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


class OllamaStructuredOutputProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        seed: int | None = None,
        temperature: float = DEFAULT_PROVIDER_TEMPERATURE,
        timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        transient_retries: int = DEFAULT_PROVIDER_TRANSIENT_RETRIES,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = DEFAULT_PROVIDER_MAX_TOKENS
        self.timeout_seconds = timeout_seconds
        self.transient_retries = transient_retries
        self._counting_mode = "estimated"

    @property
    def token_count_source(self) -> str:
        return self._counting_mode

    def complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0
        options: dict[str, Any] = {
            "temperature": self.temperature,
        }
        payload: dict[str, Any] = {
            "model": self.model_name,
            "stream": False,
            "messages": messages,
            "options": options,
        }
        if self.seed is not None:
            options["seed"] = self.seed

        response, request_count = self._post_with_transient_retry(payload=payload)

        body = response.json()
        usage = self._extract_ollama_usage(body)
        self.last_usage = usage
        self.last_finish_reason = body.get("done_reason")
        self.last_request_count = request_count
        return LLMResponse(
            content=str(body.get("message", {}).get("content", "") or ""),
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
        response_schema = build_response_json_schema(response_model)
        initial_output_tokens = max_output_tokens or self.max_tokens
        retry_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else max(self.max_tokens, DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS)
        )
        return self._run_structured_with_recovery(
            invoke=lambda output_tokens: self._post_structured_with_transient_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=response_schema,
                max_output_tokens=output_tokens,
            ),
            parse=lambda response: self._parse_ollama_structured_response(
                response=response,
                response_model=response_model,
            ),
            initial_output_tokens=initial_output_tokens,
            max_output_tokens=retry_output_tokens,
            structured_retries=self.structured_retries,
            structured_retry_token_multiplier=self.structured_retry_token_multiplier,
        )

    @staticmethod
    def _extract_structured_usage(response: Any) -> dict[str, int]:
        body = response.json()
        return OllamaStructuredOutputProvider._extract_ollama_usage(body)

    @staticmethod
    def _extract_structured_finish_reason(response: Any) -> str | None:
        body = response.json()
        done_reason = body.get("done_reason")
        return str(done_reason) if done_reason is not None else None

    @staticmethod
    def _augment_prompt_with_schema(
        *,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> str:
        schema_text = json.dumps(
            response_schema,
            separators=(",", ":"),
            ensure_ascii=True,
            sort_keys=True,
        )
        return (
            f"{user_prompt}\n\n"
            "Return JSON only. Follow this JSON schema exactly:\n"
            f"{schema_text}"
        )

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        estimated_total = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        return {
            "prompt_tokens": estimated_total,
            "completion_tokens": 0,
            "total_tokens": estimated_total,
        }

    @staticmethod
    def _extract_ollama_usage(body: dict[str, Any]) -> dict[str, int]:
        prompt_tokens = int(body.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(body.get("eval_count", 0) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _post_with_transient_retry(
        self,
        *,
        payload: dict[str, Any],
    ) -> tuple[httpx.Response, int]:
        last_exception: Exception | None = None
        request_count = 0
        for attempt in range(1, self.transient_retries + 2):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    request_count += 1
                    response = client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    return response, request_count
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exception = exc
                if attempt > self.transient_retries or not self._is_retryable_error(
                    exc
                ):
                    raise
                delay_seconds = self._next_retry_delay(attempt)
                logger.warning(
                    "Ollama request failed with transient error on attempt %d/%d "
                    "(model=%s status=%s); retrying in %.2fs: %s",
                    attempt,
                    self.transient_retries + 1,
                    self.model_name,
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None,
                    delay_seconds,
                    exc,
                )
                time.sleep(delay_seconds)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Ollama request failed without returning a response.")

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, httpx.TimeoutException):
            return True
        return (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code in DEFAULT_PROVIDER_RETRYABLE_STATUS_CODES
        )

    def _next_retry_delay(self, attempt: int) -> float:
        exponential_delay = DEFAULT_PROVIDER_RETRY_INITIAL_BACKOFF_SECONDS * (
            DEFAULT_PROVIDER_RETRY_BACKOFF_MULTIPLIER ** (attempt - 1)
        )
        bounded_delay = min(
            exponential_delay, DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS
        )
        jitter = _retry_rng.uniform(0.0, DEFAULT_PROVIDER_RETRY_JITTER_SECONDS)
        return min(bounded_delay + jitter, DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS)

    def _should_disable_thinking(self) -> bool:
        normalized_name = self.model_name.strip().lower()
        base_name = normalized_name.split(":", 1)[0]
        return base_name == "gemma4"

    def _post_structured_with_transient_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
    ) -> tuple[httpx.Response, int]:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": max_output_tokens,
        }
        payload: dict[str, Any] = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": self._augment_prompt_with_schema(
                        user_prompt=user_prompt,
                        response_schema=response_schema,
                    ),
                },
            ],
            "format": response_schema,
            "options": options,
        }
        if self._should_disable_thinking():
            payload["think"] = False
        if self.seed is not None:
            options["seed"] = self.seed

        return self._post_with_transient_retry(payload=payload)

    @staticmethod
    def _parse_ollama_structured_response(
        *,
        response: Any,
        response_model: type[BaseModel],
    ) -> BaseModel:
        body = response.json()
        content = str(body.get("message", {}).get("content", "") or "")
        if not content.strip():
            raise RuntimeError("Ollama returned no structured response payload.")
        stripped_content = content.strip()
        if stripped_content.startswith("```"):
            fence_lines = stripped_content.splitlines()
            if len(fence_lines) >= 3 and fence_lines[-1].strip() == "```":
                stripped_content = "\n".join(fence_lines[1:-1]).strip()
        first_non_whitespace = stripped_content.lstrip()[:1]
        if first_non_whitespace not in {"{", "["}:
            raise RuntimeError("Ollama returned non-JSON structured response payload.")
        return response_model.model_validate_json(stripped_content)


class AnthropicStructuredOutputProvider(LLMProvider):
    provider_name = "anthropic"
    token_count_source = "estimated"  # noqa: S105
    _DEFAULT_MODEL_MAX_OUTPUT_TOKENS = 64000
    _OPUS_MODEL_MAX_OUTPUT_TOKENS = 128000

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        seed: int | None = None,
        temperature: float = DEFAULT_PROVIDER_TEMPERATURE,
        max_tokens: int = DEFAULT_PROVIDER_MAX_TOKENS,
        timeout_seconds: int = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        transient_retries: int = DEFAULT_PROVIDER_TRANSIENT_RETRIES,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") if isinstance(base_url, str) else None
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.transient_retries = transient_retries
        self._api_key = (
            api_key
            or os.getenv("PHENTRIEVE_ANTHROPIC_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("CLAUDE_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "Anthropic API key not configured. Set PHENTRIEVE_ANTHROPIC_API_KEY, "
                "ANTHROPIC_API_KEY, or CLAUDE_API_KEY."
            )

    def complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0
        system_prompt, user_prompt = _render_messages(messages)
        response, request_count = self._create_message_with_transient_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        usage = self._extract_usage(response)
        self.last_usage = usage
        self.last_finish_reason = self._extract_finish_reason(response)
        self.last_request_count = request_count
        return LLMResponse(
            content=self._extract_text_content(response),
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
        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0
        response_schema = build_response_json_schema(response_model)
        initial_output_tokens = max_output_tokens or self.max_tokens
        retry_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else max(self.max_tokens, DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS)
        )
        return self._run_structured_with_recovery(
            invoke=lambda output_tokens: self._create_message_with_transient_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=output_tokens,
                output_schema=response_schema,
            ),
            parse=lambda response: self._parse_anthropic_structured_response(
                response=response,
                response_model=response_model,
            ),
            initial_output_tokens=initial_output_tokens,
            max_output_tokens=retry_output_tokens,
            structured_retries=self.structured_retries,
            structured_retry_token_multiplier=self.structured_retry_token_multiplier,
        )

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic support requires the optional llm dependencies. "
                "Install them with `uv sync --extra llm`."
            ) from exc

        client = self._create_client(anthropic_module=anthropic)
        response = client.messages.count_tokens(
            model=self.model_name,
            system=system_prompt or None,
            messages=[{"role": "user", "content": user_prompt}],
        )
        prompt_tokens = int(getattr(response, "input_tokens", 0) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        }

    def _supports_sampling_parameters(self) -> bool:
        # Claude Opus 4.7 and later reject explicit sampling parameters entirely.
        return not self.model_name.startswith("claude-opus-4-7")

    def _create_message_with_transient_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> tuple[Any, int]:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic support requires the optional llm dependencies. "
                "Install them with `uv sync --extra llm`."
            ) from exc

        request_count = 0
        last_exception: Exception | None = None
        for attempt in range(1, self.transient_retries + 2):
            try:
                client = self._create_client(anthropic_module=anthropic)
                request_count += 1
                create_kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "max_tokens": self._resolve_max_output_tokens(
                        requested_max_tokens=max_output_tokens or self.max_tokens
                    ),
                    "system": system_prompt or None,
                    "messages": [{"role": "user", "content": user_prompt}],
                }
                if self._supports_sampling_parameters():
                    create_kwargs["temperature"] = self.temperature
                if output_schema is not None:
                    create_kwargs["output_config"] = {
                        "format": {
                            "type": "json_schema",
                            "schema": output_schema,
                        }
                    }
                response = client.messages.create(**create_kwargs)
                return response, request_count
            except Exception as exc:
                last_exception = exc
                if (
                    attempt > self.transient_retries
                    or not self._is_retryable_provider_error(exc)
                ):
                    raise
                delay_seconds = self._next_retry_delay(attempt)
                logger.warning(
                    "Anthropic request failed with transient error on attempt %d/%d "
                    "(model=%s status=%s); retrying in %.2fs: %s",
                    attempt,
                    self.transient_retries + 1,
                    self.model_name,
                    getattr(exc, "status_code", None),
                    delay_seconds,
                    exc,
                )
                time.sleep(delay_seconds)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Anthropic request failed without returning a response.")

    def _create_client(self, *, anthropic_module: Any) -> Any:
        client_kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": self.timeout_seconds,
            "max_retries": 0,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        return anthropic_module.Anthropic(**client_kwargs)

    @staticmethod
    def _extract_text_content(response: Any) -> str:
        content_blocks = getattr(response, "content", None)
        if not isinstance(content_blocks, list):
            return ""
        text_parts = [
            str(getattr(block, "text", "") or "")
            for block in content_blocks
            if getattr(block, "type", None) == "text"
        ]
        return "".join(text_parts)

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _extract_finish_reason(response: Any) -> str | None:
        stop_reason = getattr(response, "stop_reason", None)
        return str(stop_reason) if stop_reason is not None else None

    @staticmethod
    def _is_retryable_provider_error(exc: Exception) -> bool:
        message = str(exc).lower()
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if status_code == 429 and (
                "billing_not_active" in message
                or "insufficient_quota" in message
                or "check your billing details" in message
            ):
                return False
            return status_code in DEFAULT_PROVIDER_RETRYABLE_STATUS_CODES
        return "timeout" in message or "connection" in message

    def _next_retry_delay(self, attempt: int) -> float:
        exponential_delay = DEFAULT_PROVIDER_RETRY_INITIAL_BACKOFF_SECONDS * (
            DEFAULT_PROVIDER_RETRY_BACKOFF_MULTIPLIER ** (attempt - 1)
        )
        bounded_delay = min(
            exponential_delay, DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS
        )
        jitter = _retry_rng.uniform(0.0, DEFAULT_PROVIDER_RETRY_JITTER_SECONDS)
        return min(bounded_delay + jitter, DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS)

    def _resolve_max_output_tokens(self, *, requested_max_tokens: int) -> int:
        model_name = self.model_name.lower()
        if "opus" in model_name:
            return min(requested_max_tokens, self._OPUS_MODEL_MAX_OUTPUT_TOKENS)
        return min(requested_max_tokens, self._DEFAULT_MODEL_MAX_OUTPUT_TOKENS)

    def _parse_anthropic_structured_response(
        self,
        *,
        response: Any,
        response_model: type[BaseModel],
    ) -> BaseModel:
        refusal = self._extract_finish_reason(response)
        if refusal is not None and "refusal" in refusal.lower():
            raise RuntimeError(f"Anthropic structured output refusal: {refusal}")
        content = self._extract_text_content(response)
        if not content.strip():
            raise RuntimeError("Anthropic returned no structured response payload.")
        return response_model.model_validate_json(content)


class OpenAIStructuredOutputProvider(LLMProvider):
    provider_name = "openai"
    token_count_source = "estimated"  # noqa: S105
    _SUPPORTED_STRUCTURED_MODELS = (
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    )

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        seed: int | None = None,
        temperature: float = DEFAULT_PROVIDER_TEMPERATURE,
        max_tokens: int = DEFAULT_PROVIDER_MAX_TOKENS,
        timeout_seconds: int = DEFAULT_OPENAI_TIMEOUT_SECONDS,
        transient_retries: int = DEFAULT_PROVIDER_TRANSIENT_RETRIES,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") if isinstance(base_url, str) else None
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.transient_retries = transient_retries
        self._api_key = (
            api_key
            or os.getenv("PHENTRIEVE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("CHATGPT_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "OpenAI API key not configured. Set PHENTRIEVE_OPENAI_API_KEY, "
                "OPENAI_API_KEY, or CHATGPT_API_KEY."
            )
        self._ensure_supported_model()

    def complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0
        response, request_count = self._create_response_with_transient_retry(
            messages=messages
        )
        self.last_usage = self._extract_usage(response)
        self.last_finish_reason = self._extract_finish_reason(response)
        self.last_request_count = request_count
        return LLMResponse(
            content=self._extract_text_content(response),
            model=self.model_name,
            provider=self.provider_name,
            finish_reason=self.last_finish_reason,
            usage=self.last_usage,
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
        self.last_usage = {}
        self.last_finish_reason = None
        self.last_request_count = 0
        response_schema = self._build_openai_response_schema(response_model)
        initial_output_tokens = max_output_tokens or self.max_tokens
        retry_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else max(self.max_tokens, DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS)
        )
        return self._run_structured_with_recovery(
            invoke=lambda output_tokens: self._create_response_with_transient_retry(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format={
                    "type": "json_schema",
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_schema,
                },
                max_output_tokens=output_tokens,
            ),
            parse=lambda response: self._parse_openai_structured_response(
                response=response,
                response_model=response_model,
            ),
            initial_output_tokens=initial_output_tokens,
            max_output_tokens=retry_output_tokens,
            structured_retries=self.structured_retries,
            structured_retry_token_multiplier=self.structured_retry_token_multiplier,
        )

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        estimated_total = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        return {
            "prompt_tokens": estimated_total,
            "completion_tokens": 0,
            "total_tokens": estimated_total,
        }

    def _create_response_with_transient_retry(
        self,
        *,
        messages: list[dict[str, Any]],
        text_format: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[Any, int]:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support requires the optional llm dependencies. "
                "Install them with `uv sync --extra llm`."
            ) from exc

        request_count = 0
        last_exception: Exception | None = None
        for attempt in range(1, self.transient_retries + 2):
            try:
                client = self._create_client(openai_module=openai)
                request_count += 1
                create_kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "input": messages,
                    "max_output_tokens": max_output_tokens or self.max_tokens,
                    "temperature": self.temperature,
                }
                if self.seed is not None:
                    create_kwargs["seed"] = self.seed
                if text_format is not None:
                    create_kwargs["text"] = {"format": text_format}
                response = client.responses.create(**create_kwargs)
                return response, request_count
            except Exception as exc:
                last_exception = exc
                if (
                    attempt > self.transient_retries
                    or not self._is_retryable_provider_error(exc)
                ):
                    raise
                delay_seconds = self._next_retry_delay(attempt)
                logger.warning(
                    "OpenAI request failed with transient error on attempt %d/%d "
                    "(model=%s status=%s); retrying in %.2fs: %s",
                    attempt,
                    self.transient_retries + 1,
                    self.model_name,
                    getattr(exc, "status_code", None),
                    delay_seconds,
                    exc,
                )
                time.sleep(delay_seconds)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("OpenAI request failed without returning a response.")

    def _create_client(self, *, openai_module: Any) -> Any:
        client_kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": self.timeout_seconds,
            "max_retries": 0,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        return openai_module.OpenAI(**client_kwargs)

    def _ensure_supported_model(self) -> None:
        model_name = self.model_name.lower()
        if any(
            model_name == supported
            or (
                model_name.startswith(f"{supported}-")
                and model_name[len(supported) + 1 : len(supported) + 5].isdigit()
            )
            for supported in self._SUPPORTED_STRUCTURED_MODELS
        ):
            return
        raise ValueError(
            f"OpenAI model {self.model_name!r} is not supported because native "
            "structured outputs are required."
        )

    def _build_openai_response_schema(
        self,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        return self._sanitize_openai_schema(build_response_json_schema(response_model))

    def _sanitize_openai_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "anyOf" and sanitized.get("type") == "object":
                continue
            if key == "properties" and isinstance(value, dict):
                sanitized["properties"] = {
                    prop_name: self._sanitize_openai_schema(prop_schema)
                    for prop_name, prop_schema in value.items()
                    if isinstance(prop_schema, dict)
                }
                continue
            if key == "items" and isinstance(value, dict):
                sanitized["items"] = self._sanitize_openai_schema(value)
                continue
            if key in {
                "additionalProperties",
                "required",
                "type",
                "enum",
                "description",
            }:
                sanitized[key] = value

        if sanitized.get("type") == "object":
            properties = sanitized.get("properties", {})
            if isinstance(properties, dict):
                sanitized["required"] = list(properties.keys())
            sanitized["additionalProperties"] = False
        if "anyOf" in schema and "type" not in sanitized:
            sanitized["anyOf"] = [
                self._sanitize_openai_schema(item)
                for item in schema["anyOf"]
                if isinstance(item, dict)
            ]
        if "oneOf" in schema and "type" not in sanitized:
            sanitized["anyOf"] = [
                self._sanitize_openai_schema(item)
                for item in schema["oneOf"]
                if isinstance(item, dict)
            ]
        return sanitized

    @staticmethod
    def _extract_text_content(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return ""

        text_parts: list[str] = []
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text_parts.append(str(getattr(content, "text", "") or ""))
        return "".join(text_parts)

    @staticmethod
    def _extract_refusal(response: Any) -> str | None:
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return None
        for item in output:
            if getattr(item, "type", None) == "refusal":
                refusal = getattr(item, "refusal", None)
                if isinstance(refusal, str) and refusal.strip():
                    return refusal
        return None

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _extract_finish_reason(response: Any) -> str | None:
        incomplete_details = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete_details, "reason", None)
        if reason is not None:
            return str(reason)
        status = getattr(response, "status", None)
        return str(status) if status is not None else None

    @staticmethod
    def _is_retryable_provider_error(exc: Exception) -> bool:
        message = str(exc).lower()
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if status_code == 429 and (
                "billing_not_active" in message
                or "insufficient_quota" in message
                or "check your billing details" in message
            ):
                return False
            return status_code in DEFAULT_PROVIDER_RETRYABLE_STATUS_CODES
        return "timeout" in message or "connection" in message

    def _next_retry_delay(self, attempt: int) -> float:
        exponential_delay = DEFAULT_PROVIDER_RETRY_INITIAL_BACKOFF_SECONDS * (
            DEFAULT_PROVIDER_RETRY_BACKOFF_MULTIPLIER ** (attempt - 1)
        )
        bounded_delay = min(
            exponential_delay, DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS
        )
        jitter = _retry_rng.uniform(0.0, DEFAULT_PROVIDER_RETRY_JITTER_SECONDS)
        return min(bounded_delay + jitter, DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS)

    def _parse_openai_structured_response(
        self,
        *,
        response: Any,
        response_model: type[BaseModel],
    ) -> BaseModel:
        refusal = self._extract_refusal(response)
        if refusal is not None:
            raise RuntimeError(f"OpenAI structured output refusal: {refusal}")
        content = self._extract_text_content(response)
        if not content.strip():
            raise RuntimeError("OpenAI returned no structured response payload.")
        return response_model.model_validate_json(content)


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
                                "matched_text": result.get("matched_text"),
                                "matched_component": result.get("matched_component"),
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
    llm_provider: str | None = None,
    llm_base_url: str | None = None,
    api_key: str | None = None,
    seed: int | None = None,
    timeout_seconds: int | None = None,
) -> LLMProvider:
    request = resolve_llm_provider_request(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        api_key=api_key,
        seed=seed,
    )

    if request.provider == "gemini":
        return GeminiStructuredOutputProvider(
            model_name=request.model,
            api_key=request.api_key,
            seed=request.seed,
            timeout_seconds=(
                DEFAULT_PROVIDER_TIMEOUT_SECONDS
                if timeout_seconds is None
                else timeout_seconds
            ),
        )
    if request.provider == "ollama":
        return OllamaStructuredOutputProvider(
            model_name=request.model,
            base_url=request.base_url or DEFAULT_OLLAMA_BASE_URL,
            seed=request.seed,
            timeout_seconds=(
                DEFAULT_OLLAMA_TIMEOUT_SECONDS
                if timeout_seconds is None
                else timeout_seconds
            ),
        )
    if request.provider == "anthropic":
        return AnthropicStructuredOutputProvider(
            model_name=request.model,
            api_key=request.api_key,
            base_url=request.base_url,
            seed=request.seed,
            timeout_seconds=(
                DEFAULT_PROVIDER_TIMEOUT_SECONDS
                if timeout_seconds is None
                else timeout_seconds
            ),
        )
    if request.provider == "openai":
        return OpenAIStructuredOutputProvider(
            model_name=request.model,
            api_key=request.api_key,
            base_url=request.base_url,
            seed=request.seed,
            timeout_seconds=(
                DEFAULT_OPENAI_TIMEOUT_SECONDS
                if timeout_seconds is None
                else timeout_seconds
            ),
        )

    raise ValueError(f"Provider {request.provider!r} is not implemented in phase one.")


def resolve_llm_provider_request(
    *,
    llm_provider: str | None,
    llm_model: str,
    llm_base_url: str | None = None,
    api_key: str | None = None,
    seed: int | None = None,
) -> ResolvedLLMProviderRequest:
    inferred_provider: str | None = None
    model_name = llm_model

    if "/" in llm_model:
        prefix, remainder = llm_model.split("/", 1)
        normalized_prefix = prefix.strip().lower()
        if normalized_prefix not in SUPPORTED_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown provider prefix {prefix!r} in model {llm_model!r}."
            )
        inferred_provider = normalized_prefix
        model_name = remainder

    explicit_provider = llm_provider.strip().lower() if llm_provider else None
    env_provider = os.getenv("PHENTRIEVE_LLM_PROVIDER")
    resolved_provider = (
        explicit_provider
        or inferred_provider
        or (env_provider.strip().lower() if env_provider else DEFAULT_PROVIDER_NAME)
    )
    if resolved_provider not in SUPPORTED_PROVIDER_NAMES:
        supported = ", ".join(SUPPORTED_PROVIDER_NAMES)
        raise ValueError(
            f"Unknown provider {resolved_provider!r}. Supported providers: {supported}."
        )

    if (
        inferred_provider
        and explicit_provider
        and explicit_provider != inferred_provider
    ):
        raise ValueError(
            f"Explicit llm_provider={explicit_provider!r} does not match model prefix "
            f"{inferred_provider!r}."
        )

    resolved_base_url = (
        llm_base_url
        or os.getenv("PHENTRIEVE_LLM_BASE_URL")
        or (DEFAULT_OLLAMA_BASE_URL if resolved_provider == "ollama" else None)
    )

    return ResolvedLLMProviderRequest(
        provider=resolved_provider,
        model=model_name,
        base_url=resolved_base_url,
        api_key=api_key,
        seed=seed,
    )
