from __future__ import annotations

import pytest

from phentrieve.llm.security_policy import (
    PUBLIC_LLM_MODES,
    PublicLLMPolicyError,
    get_public_llm_capabilities,
    resolve_public_llm_target,
)


def test_public_llm_target_defaults_to_single_server_owned_target() -> None:
    target = resolve_public_llm_target()

    assert target.provider == "gemini"
    assert target.model == "gemini-3.1-flash-lite"
    assert target.base_url is None


@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [
        ("gemini", None, None),
        (None, "gemini-3.1-flash-lite", None),
        ("gemini", "gemini-3.1-flash-lite", None),
        ("openai", "gpt-5.4-mini", None),
        ("anthropic", "claude-sonnet-4-6", None),
        ("ollama", "qwen3:32b", None),
        (None, None, "https://token@example.test/v1"),
    ],
)
def test_public_llm_target_rejects_any_client_selection(
    provider: str | None,
    model: str | None,
    base_url: str | None,
) -> None:
    with pytest.raises(PublicLLMPolicyError) as exc_info:
        resolve_public_llm_target(
            requested_provider=provider,
            requested_model=model,
            requested_base_url=base_url,
        )

    message = str(exc_info.value)
    assert "Public LLM provider/model/base URL selection is not supported" in message
    assert "https://token@example.test" not in message


def test_public_llm_capabilities_are_read_only_and_sanitized() -> None:
    capabilities = get_public_llm_capabilities()

    assert capabilities == {
        "default_llm_provider": "gemini",
        "default_llm_model": "gemini-3.1-flash-lite",
        "configured_llm_models": ["gemini-3.1-flash-lite"],
        "allowed_llm_targets": [
            {
                "provider": "gemini",
                "model": "gemini-3.1-flash-lite",
                "display_name": "Gemini 3.1 Flash Lite",
            }
        ],
        "llm_modes": ["two_phase"],
        "research_use_only": True,
    }
    assert "base_url" not in str(capabilities)
    assert "api_key" not in str(capabilities).lower()


def test_public_llm_modes_stay_two_phase_only() -> None:
    assert PUBLIC_LLM_MODES == ("two_phase",)
