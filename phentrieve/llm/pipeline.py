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
    CombinedPostProcessor,
    PostProcessor,
    RefinementPostProcessor,
    ValidationPostProcessor,
)
from phentrieve.llm.provider import LLMProvider, ToolExecutor
from phentrieve.llm.types import (
    AnnotationMode,
    AnnotationResult,
    HPOAnnotation,
    PostProcessingStats,
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
        # Lazy-initialized ToolExecutor for hallucination filtering in DIRECT mode
        self._tool_executor: ToolExecutor | None = None

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
            t0 = time.time()
            language = self._detect_language(text)
            logger.debug(
                "[PIPELINE] Language detection: '%s' in %.2fs",
                language,
                time.time() - t0,
            )

        # Get or create strategy for mode
        t0 = time.time()
        strategy = self._get_strategy(mode)
        logger.debug("[PIPELINE] Strategy ready in %.2fs", time.time() - t0)

        # Run primary annotation
        t0 = time.time()
        result = strategy.annotate(
            text=text,
            language=language,
            validate_hpo_ids=self.validate_hpo_ids,
        )
        logger.debug(
            "[PIPELINE] Annotation completed in %.2fs (%d annotations)",
            time.time() - t0,
            len(result.annotations),
        )

        # For DIRECT mode, run Phentrieve pipeline to get candidate IDs
        # and filter out hallucinated HPO terms that the LLM invented
        if mode == AnnotationMode.DIRECT and result.annotations:
            t0 = time.time()
            result.annotations = self._filter_direct_against_retrieval(
                result.annotations, text, language
            )
            logger.debug(
                "[PIPELINE] Direct-mode hallucination filtering in %.2fs (%d annotations remain)",
                time.time() - t0,
                len(result.annotations),
            )

        # Run post-processing if requested
        if postprocess:
            t0 = time.time()
            result = self._run_postprocessing(result, postprocess)
            logger.debug(
                "[PIPELINE] Post-processing completed in %.2fs (%d annotations remain)",
                time.time() - t0,
                len(result.annotations),
            )

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
            logger.debug("[PIPELINE] Creating new strategy for mode: %s", mode.value)
            if mode == AnnotationMode.DIRECT:
                self._strategies[mode] = DirectTextStrategy(self.provider)
            elif mode in (AnnotationMode.TOOL_TERM, AnnotationMode.TOOL_TEXT):
                self._strategies[mode] = ToolGuidedStrategy(
                    self.provider,
                    mode=mode,
                )
            else:
                raise ValueError(f"Unknown annotation mode: {mode}")
        else:
            logger.debug("[PIPELINE] Reusing cached strategy for mode: %s", mode.value)

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
            elif step == PostProcessingStep.COMBINED:
                self._postprocessors[step] = CombinedPostProcessor(self.provider)
            else:
                raise ValueError(f"Unknown post-processing step: {step}")

        return self._postprocessors[step]

    def _run_postprocessing(
        self,
        result: AnnotationResult,
        steps: list[PostProcessingStep],
    ) -> AnnotationResult:
        """Run post-processing steps on the result.

        When all three individual steps (VALIDATION, ASSERTION_REVIEW, REFINEMENT)
        are requested, automatically routes to the COMBINED post-processor to
        reduce API calls from 3 to 1.
        """
        # Auto-route to combined when all 3 individual steps are requested
        individual_steps = {
            PostProcessingStep.VALIDATION,
            PostProcessingStep.ASSERTION_REVIEW,
            PostProcessingStep.REFINEMENT,
        }
        if individual_steps.issubset(set(steps)):
            logger.info(
                "[POSTPROCESS] All 3 individual steps requested — routing to combined post-processor"
            )
            steps = [PostProcessingStep.COMBINED]

        annotations = result.annotations
        applied_steps: list[PostProcessingStep] = []
        all_stats: list[PostProcessingStats] = []

        for step in steps:
            try:
                processor = self._get_postprocessor(step)
                annotations, step_token_usage, step_stats = processor.process(
                    annotations=annotations,
                    original_text=result.input_text,
                    language=result.language,
                )
                applied_steps.append(step)
                all_stats.append(step_stats)

                # Accumulate token usage from postprocessor
                result.token_usage.merge(step_token_usage)

                logger.info(
                    "[POSTPROCESS] Completed '%s' - %d annotations remain",
                    step.value,
                    len(annotations),
                )
            except Exception as e:
                logger.error(
                    "[POSTPROCESS] Step '%s' failed: %s - continuing with previous annotations",
                    step.value,
                    e,
                )

        # Update result with processed annotations
        result.annotations = annotations
        result.post_processing_steps = applied_steps
        result.post_processing_stats = all_stats

        return result

    def _get_tool_executor(self) -> ToolExecutor:
        """Get or create a cached ToolExecutor for retrieval-based filtering."""
        if self._tool_executor is None:
            self._tool_executor = ToolExecutor()
        return self._tool_executor

    def _filter_direct_against_retrieval(
        self,
        annotations: list[HPOAnnotation],
        text: str,
        language: str,
    ) -> list[HPOAnnotation]:
        """
        Filter DIRECT mode annotations against Phentrieve retrieval results.

        Runs the Phentrieve pipeline on the text to get candidate HPO IDs,
        then removes any LLM annotations not found in the candidates.
        """
        try:
            executor = self._get_tool_executor()
            candidates = executor.execute(
                "process_clinical_text",
                {"text": text, "language": language},
            )

            # Extract candidate HPO IDs
            candidate_ids: set[str] = set()
            if isinstance(candidates, list):
                for item in candidates:
                    if isinstance(item, dict):
                        hpo_id = item.get("hpo_id")
                        if hpo_id:
                            candidate_ids.add(hpo_id)

            if not candidate_ids:
                logger.debug(
                    "[FILTER] No candidates from Phentrieve retrieval — "
                    "skipping direct-mode hallucination filter"
                )
                return annotations

            logger.debug(
                "[FILTER] %d Phentrieve candidates for direct-mode filtering",
                len(candidate_ids),
            )

            filtered: list[HPOAnnotation] = []
            removed = 0
            for annotation in annotations:
                if annotation.hpo_id in candidate_ids:
                    filtered.append(annotation)
                else:
                    removed += 1
                    logger.warning(
                        "[FILTER] Removing %s (%s) — not in Phentrieve retrieval results "
                        "(likely hallucinated by LLM in direct mode)",
                        annotation.hpo_id,
                        annotation.term_name,
                    )

            if removed > 0:
                logger.info(
                    "[FILTER] Direct mode: kept %d of %d annotations, "
                    "removed %d not found in retrieval",
                    len(filtered),
                    len(annotations),
                    removed,
                )

            return filtered

        except Exception as e:
            logger.warning(
                "[FILTER] Direct-mode hallucination filtering failed (%s) — "
                "returning unfiltered annotations",
                e,
            )
            return annotations

    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            from phentrieve.text_processing.language_detection import detect_language

            result: str = detect_language(text)
            logger.debug("[PIPELINE] Detected language: %s", result)
            return result
        except ImportError:
            logger.debug(
                "[PIPELINE] Language detection module not available - using 'en' as default"
            )
            return "en"
        except Exception as e:
            logger.warning(
                "[PIPELINE] Language detection failed (%s) - using 'en' as default", e
            )
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
