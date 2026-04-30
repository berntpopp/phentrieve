from __future__ import annotations

import logging
import os
import time
from random import SystemRandom
from typing import Any

from pydantic import BaseModel

from phentrieve.llm.config import (
    DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
    DEFAULT_PROVIDER_MAX_TOKENS,
    DEFAULT_PROVIDER_RETRY_BACKOFF_MULTIPLIER,
    DEFAULT_PROVIDER_RETRY_INITIAL_BACKOFF_SECONDS,
    DEFAULT_PROVIDER_RETRY_JITTER_SECONDS,
    DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS,
    DEFAULT_PROVIDER_RETRYABLE_STATUS_CODES,
    DEFAULT_PROVIDER_TEMPERATURE,
    DEFAULT_PROVIDER_TRANSIENT_RETRIES,
)
from phentrieve.llm.providers.base import LLMProvider, build_response_json_schema
from phentrieve.llm.types import LLMResponse

logger = logging.getLogger(__name__)
_retry_rng = SystemRandom()


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
