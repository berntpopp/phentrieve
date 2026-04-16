from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class LLMProvider(ABC):
    @abstractmethod
    def run_structured_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        raise NotImplementedError


class GeminiStructuredOutputProvider(LLMProvider):
    """Gemini structured-output provider bound to a concrete model name."""

    def __init__(self, *, model_name: str, api_key: str | None = None) -> None:
        self._model_name = model_name
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

    def run_structured_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires the optional llm dependencies. "
                "Install them with `uv sync --extra llm`."
            ) from exc

        with genai.Client(api_key=self._api_key) as client:
            response = client.models.generate_content(
                model=self._model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=response_model,
                    temperature=0,
                ),
            )

        parsed: Any = getattr(response, "parsed", None)
        if isinstance(parsed, response_model):
            return parsed
        if parsed is not None:
            return response_model.model_validate(parsed)

        response_text = getattr(response, "text", None)
        if isinstance(response_text, str) and response_text.strip():
            return response_model.model_validate_json(response_text)

        raise RuntimeError("Gemini returned no structured response payload.")


def get_llm_provider(
    *,
    llm_model: str,
    provider_name: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Build the configured provider for the given model."""
    provider_value = provider_name or os.getenv("PHENTRIEVE_LLM_PROVIDER") or "gemini"
    resolved_provider = provider_value.strip().lower()
    if resolved_provider != "gemini":
        raise ValueError(
            f"Unsupported LLM provider: {resolved_provider!r}. Expected 'gemini'."
        )

    return GeminiStructuredOutputProvider(model_name=llm_model, api_key=api_key)
