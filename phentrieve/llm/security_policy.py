from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phentrieve.llm.config import DEFAULT_LLM_MODEL, DEFAULT_PROVIDER_NAME

PUBLIC_LLM_MODES = ("two_phase",)


class PublicLLMPolicyError(ValueError):
    """Raised when public callers attempt unsupported LLM configuration."""


@dataclass(frozen=True, slots=True)
class PublicLLMTarget:
    provider: str
    model: str
    display_name: str
    base_url: str | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    def capability_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "display_name": self.display_name,
        }


DEFAULT_PUBLIC_LLM_TARGET = PublicLLMTarget(
    provider=DEFAULT_PROVIDER_NAME,
    model=DEFAULT_LLM_MODEL,
    display_name="Gemini 3.1 Flash Lite",
)
PUBLIC_LLM_TARGETS: tuple[PublicLLMTarget, ...] = (DEFAULT_PUBLIC_LLM_TARGET,)


def _has_client_selection(
    *,
    requested_provider: str | None,
    requested_model: str | None,
    requested_base_url: str | None,
) -> bool:
    return any(
        value is not None and str(value).strip()
        for value in (requested_provider, requested_model, requested_base_url)
    )


def resolve_public_llm_target(
    *,
    requested_provider: str | None = None,
    requested_model: str | None = None,
    requested_base_url: str | None = None,
) -> PublicLLMTarget:
    if _has_client_selection(
        requested_provider=requested_provider,
        requested_model=requested_model,
        requested_base_url=requested_base_url,
    ):
        raise PublicLLMPolicyError(
            "Public LLM provider/model/base URL selection is not supported. "
            "Omit llm_provider, llm_model, and llm_base_url; the server uses its "
            "configured public LLM target."
        )
    return DEFAULT_PUBLIC_LLM_TARGET


def get_public_llm_capabilities() -> dict[str, Any]:
    return {
        "default_llm_provider": DEFAULT_PUBLIC_LLM_TARGET.provider,
        "default_llm_model": DEFAULT_PUBLIC_LLM_TARGET.model,
        "configured_llm_models": [target.model for target in PUBLIC_LLM_TARGETS],
        "allowed_llm_targets": [
            target.capability_dict() for target in PUBLIC_LLM_TARGETS
        ],
        "llm_modes": list(PUBLIC_LLM_MODES),
        "research_use_only": True,
    }
