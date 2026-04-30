from __future__ import annotations

import logging
import os
import time
from random import SystemRandom
from typing import Any

import httpx
from pydantic import BaseModel

from phentrieve.llm.config import (
    DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS,
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
)
from phentrieve.llm.providers.base import (
    LLMProvider,
    _render_messages,
    build_response_json_schema,
)
from phentrieve.llm.types import LLMResponse

logger = logging.getLogger(__name__)
_retry_rng = SystemRandom()


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
