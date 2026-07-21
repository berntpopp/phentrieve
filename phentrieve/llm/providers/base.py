from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from threading import local
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel

from phentrieve.llm.config import (
    DEFAULT_PROVIDER_STRUCTURED_RETRIES,
    DEFAULT_PROVIDER_STRUCTURED_RETRY_TOKEN_MULTIPLIER,
    DEFAULT_PROVIDER_TEMPERATURE,
)
from phentrieve.llm.types import LLMResponse

logger = logging.getLogger(__name__)


def canonicalize_llm_base_url(value: str | None) -> str | None:
    """Return one validated provider base URL representation."""
    if value is None or not value.strip():
        return None
    raw = value.strip()
    parsed = urlsplit(raw)
    if parsed.fragment:
        raise ValueError("LLM base URL must not contain a fragment")
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("LLM base URL must be an absolute HTTP(S) URL")
    path = parsed.path.rstrip("/")
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username is not None or parsed.password is not None:
        userinfo = parsed.username or ""
        if parsed.password is not None:
            userinfo = f"{userinfo}:{parsed.password}"
        netloc = f"{userinfo}@{netloc}"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


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
