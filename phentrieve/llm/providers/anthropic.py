from __future__ import annotations

import importlib
import logging
import os
import time
from random import SystemRandom
from typing import Any

from pydantic import BaseModel

from phentrieve.llm.config import (
    DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_PROVIDER_MAX_TOKENS,
    DEFAULT_PROVIDER_RETRY_BACKOFF_MULTIPLIER,
    DEFAULT_PROVIDER_RETRY_INITIAL_BACKOFF_SECONDS,
    DEFAULT_PROVIDER_RETRY_JITTER_SECONDS,
    DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECONDS,
    DEFAULT_PROVIDER_RETRYABLE_STATUS_CODES,
    DEFAULT_PROVIDER_TEMPERATURE,
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    DEFAULT_PROVIDER_TRANSIENT_RETRIES,
)
from phentrieve.llm.providers.base import (
    LLMProvider,
    _render_messages,
    build_response_json_schema,
    canonicalize_llm_base_url,
)
from phentrieve.llm.types import LLMResponse

logger = logging.getLogger(__name__)
_retry_rng = SystemRandom()


def _load_anthropic_module() -> Any:
    try:
        return importlib.import_module("anthropic")
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic support requires the optional llm dependencies. "
            "Install them with `uv sync --extra llm`."
        ) from exc


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
        self.base_url = canonicalize_llm_base_url(base_url)
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
        anthropic = _load_anthropic_module()
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
        anthropic = _load_anthropic_module()
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
