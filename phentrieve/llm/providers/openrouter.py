from __future__ import annotations

import os

from phentrieve.llm.config import (
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_PROVIDER_MAX_TOKENS,
    DEFAULT_PROVIDER_TEMPERATURE,
    DEFAULT_PROVIDER_TRANSIENT_RETRIES,
)
from phentrieve.llm.providers.openai import OpenAIStructuredOutputProvider


class OpenRouterStructuredOutputProvider(OpenAIStructuredOutputProvider):
    """OpenRouter provider using its OpenAI-compatible Responses endpoint."""

    provider_name = "openrouter"

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
        resolved_api_key = (
            api_key
            or os.getenv("PHENTRIEVE_OPENROUTER_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("PHENTRIEVE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("CHATGPT_API_KEY")
        )
        if not resolved_api_key:
            raise RuntimeError(
                "OpenRouter API key not configured. Set "
                "PHENTRIEVE_OPENROUTER_API_KEY, OPENROUTER_API_KEY, "
                "PHENTRIEVE_OPENAI_API_KEY, OPENAI_API_KEY, or CHATGPT_API_KEY."
            )

        super().__init__(
            model_name=model_name,
            api_key=resolved_api_key,
            base_url=base_url or DEFAULT_OPENROUTER_BASE_URL,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            transient_retries=transient_retries,
        )

    def _ensure_supported_model(self) -> None:
        return
