"""
Prompt template management for the LLM annotation system.

This module handles loading and rendering prompt templates from:
1. User overrides (~/.phentrieve/prompts/)
2. Package defaults (phentrieve/llm/prompts/templates/)
"""

from phentrieve.llm.prompts.loader import (
    PromptTemplate,
    get_prompt,
    list_available_prompts,
    load_prompt_template,
)

__all__ = [
    "PromptTemplate",
    "get_prompt",
    "list_available_prompts",
    "load_prompt_template",
]
