"""
Unified LLM annotation pipeline.

This module provides the main entry point for LLM-based annotation,
handling mode selection, post-processing, and result formatting.
"""

import logging
import time
from typing import Any

from phentrieve.llm.annotation import (
    AnnotationStrategy,
    DirectTextStrategy,
    ToolGuidedStrategy,
)
from phentrieve.llm.postprocess import (
    AssertionReviewPostProcessor,
    PostProcessor,
    RefinementPostProcessor,
    ValidationPostProcessor,
)
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import (
    AnnotationMode,
    AnnotationResult,
    PostProcessingStep,
)

logger = logging.getLogger(__name__)


class LLMAnnotationPipeline:
    """
    Unified pipeline for LLM-based HPO annotation.

    This pipeline provides a single interface for all annotation modes
    and post-processing options. It handles:

    1. Mode selection (direct, tool_term, tool_text)
    2. Primary annotation using the selected strategy
    3. Optional post-processing (validation, refinement, assertion review)
    4. Result formatting and provenance tracking

    Example usage:
        pipeline = LLMAnnotationPipeline(model="github/gpt-4o")
        result = pipeline.run(
            text="Patient has seizures and no cardiac abnormalities",
            mode=AnnotationMode.TOOL_TEXT,
            postprocess=[PostProcessingStep.VALIDATION],
        )
        for annotation in result.annotations:
            print(f"{annotation.hpo_id}: {annotation.term_name}")

    Attributes:
        provider: The LLM provider instance.
        validate_hpo_ids: Whether to validate HPO IDs against the database.
    """

    def __init__(
        self,
        model: str = "github/gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        validate_hpo_ids: bool = True,
    ) -> None:
        """
        Initialize the annotation pipeline.

        Args:
            model: LiteLLM model string (e.g., "github/gpt-4o").
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            validate_hpo_ids: Whether to validate HPO IDs against the database.
        """
        self.provider = LLMProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.validate_hpo_ids = validate_hpo_ids

        # Cache strategies
        self._strategies: dict[AnnotationMode, AnnotationStrategy] = {}
        self._postprocessors: dict[PostProcessingStep, PostProcessor] = {}

    def run(
        self,
        text: str,
        mode: AnnotationMode = AnnotationMode.TOOL_TEXT,
        language: str = "auto",
        postprocess: list[PostProcessingStep] | None = None,
    ) -> AnnotationResult:
        """
        Run the annotation pipeline on clinical text.

        Args:
            text: The clinical text to annotate.
            mode: The annotation mode to use.
            language: Language code or "auto" for detection.
            postprocess: Optional list of post-processing steps.

        Returns:
            AnnotationResult with extracted and processed annotations.
        """
        start_time = time.time()

        # Detect language if auto
        if language == "auto":
            language = self._detect_language(text)

        # Get or create strategy for mode
        strategy = self._get_strategy(mode)

        # Run primary annotation
        result = strategy.annotate(
            text=text,
            language=language,
            validate_hpo_ids=self.validate_hpo_ids,
        )

        # Run post-processing if requested
        if postprocess:
            result = self._run_postprocessing(result, postprocess)

        # Update processing time
        result.processing_time_seconds = time.time() - start_time

        return result

    def run_batch(
        self,
        texts: list[str],
        mode: AnnotationMode = AnnotationMode.TOOL_TEXT,
        language: str = "auto",
        postprocess: list[PostProcessingStep] | None = None,
    ) -> list[AnnotationResult]:
        """
        Run the annotation pipeline on multiple texts.

        Args:
            texts: List of clinical texts to annotate.
            mode: The annotation mode to use.
            language: Language code or "auto" for detection.
            postprocess: Optional list of post-processing steps.

        Returns:
            List of AnnotationResult objects.
        """
        results = []
        for i, text in enumerate(texts):
            logger.info("Processing text %d/%d", i + 1, len(texts))
            result = self.run(
                text=text,
                mode=mode,
                language=language,
                postprocess=postprocess,
            )
            results.append(result)

        return results

    def _get_strategy(self, mode: AnnotationMode) -> AnnotationStrategy:
        """Get or create an annotation strategy for the mode."""
        if mode not in self._strategies:
            if mode == AnnotationMode.DIRECT:
                self._strategies[mode] = DirectTextStrategy(self.provider)
            elif mode in (AnnotationMode.TOOL_TERM, AnnotationMode.TOOL_TEXT):
                self._strategies[mode] = ToolGuidedStrategy(
                    self.provider,
                    mode=mode,
                )
            else:
                raise ValueError(f"Unknown annotation mode: {mode}")

        return self._strategies[mode]

    def _get_postprocessor(self, step: PostProcessingStep) -> PostProcessor:
        """Get or create a post-processor for the step."""
        if step not in self._postprocessors:
            if step == PostProcessingStep.VALIDATION:
                self._postprocessors[step] = ValidationPostProcessor(self.provider)
            elif step == PostProcessingStep.REFINEMENT:
                self._postprocessors[step] = RefinementPostProcessor(self.provider)
            elif step == PostProcessingStep.ASSERTION_REVIEW:
                self._postprocessors[step] = AssertionReviewPostProcessor(self.provider)
            else:
                raise ValueError(f"Unknown post-processing step: {step}")

        return self._postprocessors[step]

    def _run_postprocessing(
        self,
        result: AnnotationResult,
        steps: list[PostProcessingStep],
    ) -> AnnotationResult:
        """Run post-processing steps on the result."""
        annotations = result.annotations
        applied_steps: list[PostProcessingStep] = []

        for step in steps:
            try:
                processor = self._get_postprocessor(step)
                annotations = processor.process(
                    annotations=annotations,
                    original_text=result.input_text,
                    language=result.language,
                )
                applied_steps.append(step)
                logger.info(
                    "Post-processing step '%s' complete: %d annotations",
                    step.value,
                    len(annotations),
                )
            except Exception as e:
                logger.error("Post-processing step '%s' failed: %s", step.value, e)

        # Update result with processed annotations
        result.annotations = annotations
        result.post_processing_steps = applied_steps

        return result

    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            from phentrieve.text_processing.language_detection import detect_language

            result: str = detect_language(text)
            return result
        except ImportError:
            logger.debug("Language detection not available, defaulting to 'en'")
            return "en"
        except Exception as e:
            logger.warning("Language detection failed: %s, defaulting to 'en'", e)
            return "en"


def create_pipeline(
    model: str = "github/gpt-4o",
    temperature: float = 0.0,
    **kwargs: Any,
) -> LLMAnnotationPipeline:
    """
    Convenience function to create an annotation pipeline.

    Args:
        model: LiteLLM model string.
        temperature: Sampling temperature.
        **kwargs: Additional arguments passed to LLMAnnotationPipeline.

    Returns:
        Configured LLMAnnotationPipeline instance.
    """
    return LLMAnnotationPipeline(
        model=model,
        temperature=temperature,
        **kwargs,
    )
