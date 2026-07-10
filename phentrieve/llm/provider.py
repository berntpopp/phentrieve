from __future__ import annotations

import sys
import time
from random import SystemRandom
from types import ModuleType

import httpx

from phentrieve.llm.providers.anthropic import AnthropicStructuredOutputProvider
from phentrieve.llm.providers.base import (
    LLMProvider,
    ResolvedLLMProviderRequest,
    build_response_json_schema,
    compact_json_schema,
)
from phentrieve.llm.providers.gemini import GeminiStructuredOutputProvider
from phentrieve.llm.providers.ollama import OllamaStructuredOutputProvider
from phentrieve.llm.providers.openai import OpenAIStructuredOutputProvider
from phentrieve.llm.providers.openrouter import OpenRouterStructuredOutputProvider
from phentrieve.llm.providers.resolver import (
    get_llm_provider,
    resolve_llm_provider_request,
)
from phentrieve.llm.tools import ToolExecutor

_retry_rng = SystemRandom()

_COMPAT_MODULES = (
    "phentrieve.llm.providers.gemini",
    "phentrieve.llm.providers.ollama",
    "phentrieve.llm.providers.anthropic",
    "phentrieve.llm.providers.openai",
    "phentrieve.llm.providers.openrouter",
)
_COMPAT_NAMES = {"_retry_rng", "time", "httpx"}
_COMPAT_VALUES = {
    "_retry_rng": _retry_rng,
    "time": time,
    "httpx": httpx,
}


def _propagate_compat_value(name: str, value: object) -> None:
    if name not in _COMPAT_NAMES:
        return
    for module_name in _COMPAT_MODULES:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, name):
            setattr(module, name, value)


class _ProviderFacadeModule(ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        _propagate_compat_value(name, value)


for _name, _value in _COMPAT_VALUES.items():
    _propagate_compat_value(_name, _value)

sys.modules[__name__].__class__ = _ProviderFacadeModule

__all__ = [
    "AnthropicStructuredOutputProvider",
    "GeminiStructuredOutputProvider",
    "LLMProvider",
    "OllamaStructuredOutputProvider",
    "OpenAIStructuredOutputProvider",
    "OpenRouterStructuredOutputProvider",
    "ResolvedLLMProviderRequest",
    "ToolExecutor",
    "build_response_json_schema",
    "compact_json_schema",
    "get_llm_provider",
    "resolve_llm_provider_request",
]
