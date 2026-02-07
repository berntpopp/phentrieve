"""
LLM Annotation System for Phentrieve.

This module provides a provider-agnostic LLM annotation system for mapping
clinical text to HPO terms. It supports multiple annotation modes:

- Direct Text: LLM outputs HPO IDs directly from its training knowledge
- Tool-Guided (Term Search): LLM extracts phrases, queries Phentrieve, selects best matches
- Tool-Guided (Text Process): Phentrieve processes text, LLM validates and selects

Providers are unified through LiteLLM, supporting GitHub Models, Gemini, Anthropic,
OpenAI, Ollama, and 100+ other models.

Example usage:
    from phentrieve.llm import LLMAnnotationPipeline, AnnotationMode

    pipeline = LLMAnnotationPipeline(model="github/gpt-4o")
    result = pipeline.run(
        text="Patient has seizures and no cardiac abnormalities",
        mode=AnnotationMode.TOOL_TEXT,
    )
    for annotation in result.annotations:
        print(f"{annotation.hpo_id}: {annotation.term_name} ({annotation.assertion})")
"""

from phentrieve.llm.pipeline import LLMAnnotationPipeline, create_pipeline
from phentrieve.llm.pricing import estimate_cost
from phentrieve.llm.provider import LLMProvider, LLMProviderError, get_available_models
from phentrieve.llm.types import (
    AnnotationMode,
    AnnotationResult,
    AssertionStatus,
    HPOAnnotation,
    LLMResponse,
    PostProcessingStats,
    PostProcessingStep,
    TimingEvent,
    TokenUsage,
    ToolCall,
)

__all__ = [
    "AnnotationMode",
    "AnnotationResult",
    "AssertionStatus",
    "HPOAnnotation",
    "LLMAnnotationPipeline",
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "PostProcessingStats",
    "PostProcessingStep",
    "TimingEvent",
    "TokenUsage",
    "ToolCall",
    "create_pipeline",
    "estimate_cost",
    "get_available_models",
]
