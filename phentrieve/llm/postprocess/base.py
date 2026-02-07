"""
Base class for post-processors.

Post-processors refine and validate annotations after the primary
annotation step. Each processor takes annotations and returns
modified annotations along with token usage statistics.
"""

from abc import ABC, abstractmethod

from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import (
    HPOAnnotation,
    PostProcessingStats,
    PostProcessingStep,
    TokenUsage,
)


class PostProcessor(ABC):
    """
    Abstract base class for annotation post-processors.

    Post-processors take a list of annotations and the original text,
    then return a refined list of annotations along with token usage. They may:

    - Remove false positives
    - Correct assertion status
    - Upgrade to more specific terms
    - Adjust confidence scores

    Attributes:
        step: The post-processing step this processor implements.
        provider: The LLM provider instance to use.
    """

    step: PostProcessingStep

    def __init__(self, provider: LLMProvider) -> None:
        """
        Initialize the post-processor.

        Args:
            provider: The LLM provider to use for processing.
        """
        self.provider = provider

    @abstractmethod
    def process(
        self,
        annotations: list[HPOAnnotation],
        original_text: str,
        language: str = "en",
    ) -> tuple[list[HPOAnnotation], TokenUsage, PostProcessingStats]:
        """
        Process and refine annotations.

        Args:
            annotations: The annotations to process.
            original_text: The original clinical text.
            language: Language code for prompt selection.

        Returns:
            Tuple of (refined annotations, token usage, post-processing stats).
        """
        pass

    @property
    def name(self) -> str:
        """Get the human-readable name of this processor."""
        return self.__class__.__name__
