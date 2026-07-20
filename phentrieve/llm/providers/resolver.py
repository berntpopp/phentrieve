from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

from phentrieve.llm.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_PROVIDER_NAME,
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    SUPPORTED_PROVIDER_NAMES,
)
from phentrieve.llm.providers.anthropic import AnthropicStructuredOutputProvider
from phentrieve.llm.providers.base import LLMProvider, ResolvedLLMProviderRequest
from phentrieve.llm.providers.gemini import GeminiStructuredOutputProvider
from phentrieve.llm.providers.ollama import OllamaStructuredOutputProvider
from phentrieve.llm.providers.openai import OpenAIStructuredOutputProvider
from phentrieve.llm.providers.openrouter import OpenRouterStructuredOutputProvider


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


# Public REST/API and MCP surfaces must not call this resolver directly with
# client-supplied provider/model/base URL values. Route public requests through
# phentrieve.llm.security_policy first so public callers cannot select providers,
# models, or base URLs.
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
    if request.provider == "openrouter":
        return OpenRouterStructuredOutputProvider(
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
    explicit_provider = llm_provider.strip().lower() if llm_provider else None

    if "/" in llm_model and explicit_provider != "openrouter":
        prefix, remainder = llm_model.split("/", 1)
        normalized_prefix = prefix.strip().lower()
        if normalized_prefix not in SUPPORTED_PROVIDER_NAMES:
            raise ValueError(
                f"Unknown provider prefix {prefix!r} in model {llm_model!r}."
            )
        inferred_provider = normalized_prefix
        model_name = remainder

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

    raw_base_url = (
        llm_base_url
        or os.getenv("PHENTRIEVE_LLM_BASE_URL")
        or (DEFAULT_OLLAMA_BASE_URL if resolved_provider == "ollama" else None)
        or (DEFAULT_OPENROUTER_BASE_URL if resolved_provider == "openrouter" else None)
    )
    resolved_base_url = canonicalize_llm_base_url(raw_base_url)
    if resolved_provider == "gemini" and resolved_base_url is not None:
        raise ValueError("Provider 'gemini' does not support a base URL")

    return ResolvedLLMProviderRequest(
        provider=resolved_provider,
        model=model_name,
        base_url=resolved_base_url,
        api_key=api_key,
        seed=seed,
    )
