"""
Base class for annotation strategies.

This module defines the abstract base class that all annotation strategies
must implement. Strategies encapsulate different approaches for extracting
HPO annotations from clinical text using LLMs.
"""

from abc import ABC, abstractmethod

from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import AnnotationMode, AnnotationResult


class AnnotationStrategy(ABC):
    """
    Abstract base class for annotation strategies.

    Each strategy implements a different approach for extracting HPO
    annotations from clinical text. Strategies are responsible for:

    1. Preparing prompts appropriate for the mode
    2. Interacting with the LLM provider
    3. Parsing responses into structured annotations
    4. Validating HPO IDs against the database (optional)

    Attributes:
        mode: The annotation mode this strategy implements.
        provider: The LLM provider instance to use.
    """

    mode: AnnotationMode

    def __init__(self, provider: LLMProvider) -> None:
        """
        Initialize the annotation strategy.

        Args:
            provider: The LLM provider to use for completions.
        """
        self.provider = provider

    @abstractmethod
    def annotate(
        self,
        text: str,
        language: str = "en",
        validate_hpo_ids: bool = True,
    ) -> AnnotationResult:
        """
        Extract HPO annotations from clinical text.

        This is the main entry point for annotation. Implementations should:
        1. Load appropriate prompt templates
        2. Send request to LLM
        3. Parse and validate results
        4. Return structured AnnotationResult

        Args:
            text: The clinical text to annotate.
            language: Language code (e.g., "en", "de").
            validate_hpo_ids: Whether to validate HPO IDs against the database.

        Returns:
            AnnotationResult containing extracted annotations and metadata.
        """
        pass

    @property
    def name(self) -> str:
        """Get the human-readable name of this strategy."""
        return self.__class__.__name__
