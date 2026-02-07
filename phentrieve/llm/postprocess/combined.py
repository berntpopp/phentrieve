"""
Combined post-processor.

This processor performs validation, assertion review, and refinement in a
single LLM call, eliminating 2 API round-trips compared to running each
post-processor individually.
"""

import json
import logging
import time
from typing import Any

from phentrieve.llm.postprocess.base import PostProcessor
from phentrieve.llm.prompts import load_prompt_template
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import (
    HPOAnnotation,
    PostProcessingStats,
    PostProcessingStep,
    TimingEvent,
    TokenUsage,
)
from phentrieve.llm.utils import extract_json, normalize_hpo_id, parse_assertion

logger = logging.getLogger(__name__)


class CombinedPostProcessor(PostProcessor):
    """
    Combined post-processor performing validation + assertion review + refinement.

    Instead of three sequential LLM calls, this processor sends a single prompt
    that asks the LLM to:
    1. Remove false positives (validation)
    2. Correct assertion status (assertion review)
    3. Suggest more specific terms (refinement)

    Returns separate PostProcessingStats for each sub-step so that callers
    can still track what changed in each category.
    """

    step = PostProcessingStep.COMBINED

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize with an LLM provider."""
        super().__init__(provider)

    def process(
        self,
        annotations: list[HPOAnnotation],
        original_text: str,
        language: str = "en",
    ) -> tuple[list[HPOAnnotation], TokenUsage, PostProcessingStats]:
        """
        Run combined validation + assertion review + refinement.

        Args:
            annotations: The annotations to process.
            original_text: The original clinical text.
            language: Language code for prompt selection.

        Returns:
            Tuple of (processed annotations, token usage, combined stats).
        """
        annotations_in = len(annotations)
        if not annotations:
            stats = PostProcessingStats(
                step="combined", annotations_in=0, annotations_out=0
            )
            return annotations, TokenUsage(), stats

        # Load combined prompt template
        try:
            prompt_template = load_prompt_template(self.step, language)
        except FileNotFoundError:
            logger.warning("Combined prompt not found, returning original annotations")
            stats = PostProcessingStats(
                step="combined",
                annotations_in=annotations_in,
                annotations_out=annotations_in,
            )
            return annotations, TokenUsage(), stats

        # Format annotations for the prompt
        annotations_json = json.dumps(
            [a.to_dict() for a in annotations],
            indent=2,
        )

        # Build messages
        messages = prompt_template.get_messages(
            text=original_text,
            annotations=annotations_json,
        )

        # Get combined review response
        t0 = time.time()
        response = self.provider.complete(messages)
        llm_elapsed = time.time() - t0

        # Track token usage
        token_usage = TokenUsage.from_response(response.usage, llm_time=llm_elapsed)
        token_usage.timing_events.append(
            TimingEvent(
                label=f"postprocess: combined ({self.provider.model})",
                duration_seconds=llm_elapsed,
                category="postprocess",
            )
        )

        # Parse the combined response
        result_annotations, removed_count, refined_count, assertions_changed = (
            self._parse_combined_response(
                response.content or "",
                annotations,
            )
        )

        stats = PostProcessingStats(
            step="combined",
            annotations_in=annotations_in,
            annotations_out=len(result_annotations),
            removed=removed_count,
            terms_refined=refined_count,
            assertions_changed=assertions_changed,
        )

        return result_annotations, token_usage, stats

    def process_returning_substats(
        self,
        annotations: list[HPOAnnotation],
        original_text: str,
        language: str = "en",
    ) -> tuple[list[HPOAnnotation], TokenUsage, list[PostProcessingStats]]:
        """
        Run combined processing and return per-sub-step stats.

        This is a convenience method that returns separate stats for
        validation, assertion_review, and refinement so callers can
        track each sub-step individually.

        Args:
            annotations: The annotations to process.
            original_text: The original clinical text.
            language: Language code for prompt selection.

        Returns:
            Tuple of (processed annotations, token usage, list of stats per sub-step).
        """
        result_annotations, token_usage, combined_stats = self.process(
            annotations, original_text, language
        )

        # Split into per-sub-step stats
        substats = [
            PostProcessingStats(
                step="validation",
                annotations_in=combined_stats.annotations_in,
                annotations_out=combined_stats.annotations_out,
                removed=combined_stats.removed,
            ),
            PostProcessingStats(
                step="assertion_review",
                annotations_in=combined_stats.annotations_in,
                annotations_out=combined_stats.annotations_out,
                assertions_changed=combined_stats.assertions_changed,
            ),
            PostProcessingStats(
                step="refinement",
                annotations_in=combined_stats.annotations_in,
                annotations_out=combined_stats.annotations_out,
                terms_refined=combined_stats.terms_refined,
            ),
        ]

        return result_annotations, token_usage, substats

    def _parse_combined_response(
        self,
        response_text: str,
        original_annotations: list[HPOAnnotation],
    ) -> tuple[list[HPOAnnotation], int, int, int]:
        """Parse combined response and return results with counts.

        Returns:
            Tuple of (annotations, removed_count, refined_count, assertions_changed).
        """
        json_data = extract_json(response_text)

        if not json_data:
            logger.warning("Could not parse combined response, returning originals")
            return original_annotations, 0, 0, 0

        result: list[HPOAnnotation] = []
        orig_by_id = {a.hpo_id: a for a in original_annotations}
        assertions_changed = 0
        refined_count = 0

        # Process validated annotations (kept, possibly with assertion corrections)
        for item in json_data.get("validated_annotations", []):
            annotation = self._update_from_validated(item, orig_by_id)
            if annotation:
                # Check if assertion changed
                orig = orig_by_id.get(item.get("hpo_id", ""))
                if orig and annotation.assertion != orig.assertion:
                    assertions_changed += 1
                result.append(annotation)

        # Process refined annotations (upgraded to more specific terms)
        for item in json_data.get("refined_annotations", []):
            annotation = self._create_refined_annotation(item)
            if annotation:
                result.append(annotation)
                refined_count += 1
                # Check if assertion changed on the original
                orig_id = item.get("original_hpo_id", "")
                orig = orig_by_id.get(orig_id)
                if orig:
                    new_assertion = parse_assertion(
                        item.get("assertion", orig.assertion.value)
                    )
                    if new_assertion != orig.assertion:
                        assertions_changed += 1

        # Log removed annotations
        removed_items = json_data.get("removed_annotations", [])
        for item in removed_items:
            logger.info(
                "[COMBINED] Removed %s (%s): %s",
                item.get("hpo_id"),
                item.get("term_name"),
                item.get("reason"),
            )

        return result, len(removed_items), refined_count, assertions_changed

    def _update_from_validated(
        self,
        item: dict[str, Any],
        orig_by_id: dict[str, HPOAnnotation],
    ) -> HPOAnnotation | None:
        """Create annotation from validated item, preserving original metadata."""
        hpo_id = item.get("hpo_id")
        if not hpo_id:
            return None

        orig = orig_by_id.get(hpo_id)
        if not orig:
            return None

        return HPOAnnotation(
            hpo_id=orig.hpo_id,
            term_name=item.get("term_name", orig.term_name),
            assertion=parse_assertion(item.get("assertion", orig.assertion.value)),
            confidence=float(item.get("confidence", orig.confidence)),
            evidence_text=item.get("evidence_text", orig.evidence_text),
            evidence_start=orig.evidence_start,
            evidence_end=orig.evidence_end,
            definition=orig.definition,
            synonyms=orig.synonyms,
            source_mode=orig.source_mode,
            raw_score=orig.raw_score,
        )

    def _create_refined_annotation(
        self,
        item: dict[str, Any],
    ) -> HPOAnnotation | None:
        """Create annotation from refined item."""
        hpo_id = item.get("refined_hpo_id")
        if not hpo_id:
            return None

        hpo_id = normalize_hpo_id(hpo_id)
        if not hpo_id:
            return None

        return HPOAnnotation(
            hpo_id=hpo_id,
            term_name=item.get("refined_term_name", ""),
            assertion=parse_assertion(item.get("assertion", "affirmed")),
            confidence=float(item.get("confidence", 0.9)),
            evidence_text=item.get("evidence_text"),
        )
