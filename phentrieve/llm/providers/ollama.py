from __future__ import annotations

import json
import logging
import time
from random import SystemRandom
from typing import Any

import httpx
from pydantic import BaseModel

from phentrieve.llm.config import (
    DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
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
        self.base_url = base_url
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
