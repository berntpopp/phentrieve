"""
Annotation strategies for the LLM annotation system.

This module provides different strategies for extracting HPO annotations
from clinical text:

- DirectTextStrategy: LLM outputs HPO IDs directly from training knowledge
- ToolGuidedStrategy: LLM uses Phentrieve tools (term search or text processing)
"""

from phentrieve.llm.annotation.base import AnnotationStrategy
from phentrieve.llm.annotation.direct_text import DirectTextStrategy
from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy

__all__ = [
    "AnnotationStrategy",
    "DirectTextStrategy",
    "ToolGuidedStrategy",
]
