"""
Validation post-processor.

This processor has the LLM re-check each annotation against the original
text to remove false positives and correct errors.
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
from phentrieve.llm.utils import extract_json, parse_assertion

logger = logging.getLogger(__name__)


class ValidationPostProcessor(PostProcessor):
    """
    Validation post-processor for checking annotations against text.

    This processor asks the LLM to review each annotation and verify:
    - Whether the annotation is actually supported by the text
    - Whether the assertion status is correct
    - Whether the evidence text is accurate

    Annotations identified as false positives are removed, and corrected
    annotations replace their originals.
    """

    step = PostProcessingStep.VALIDATION

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
        Validate annotations against the original text.

        Args:
            annotations: The annotations to validate.
            original_text: The original clinical text.
            language: Language code for prompt selection.

        Returns:
            Tuple of (validated annotations, token usage, stats).
        """
        annotations_in = len(annotations)
        if not annotations:
            stats = PostProcessingStats(
                step="validation", annotations_in=0, annotations_out=0
            )
            return annotations, TokenUsage(), stats

        # Load validation prompt template
        try:
            prompt_template = load_prompt_template(self.step, language)
        except FileNotFoundError:
            logger.warning(
                "Validation prompt not found, returning original annotations"
            )
            stats = PostProcessingStats(
                step="validation",
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

        # Get validation response
        t0 = time.time()
        response = self.provider.complete(messages)
        llm_elapsed = time.time() - t0

        # Track token usage
        token_usage = TokenUsage.from_response(response.usage, llm_time=llm_elapsed)
        token_usage.timing_events.append(
            TimingEvent(
                label=f"postprocess: validation ({self.provider.model})",
                duration_seconds=llm_elapsed,
                category="postprocess",
            )
        )

        # Parse the validation result
        validated, removed_count, added_count = self._parse_validation_response(
            response.content or "",
            annotations,
        )

        stats = PostProcessingStats(
            step="validation",
            annotations_in=annotations_in,
            annotations_out=len(validated),
            removed=removed_count,
            added=added_count,
        )

        return validated, token_usage, stats

    def _parse_validation_response(
        self,
        response_text: str,
        original_annotations: list[HPOAnnotation],
    ) -> tuple[list[HPOAnnotation], int, int]:
        """Parse validation response and return refined annotations with counts.

        Returns:
            Tuple of (validated annotations, removed count, added/corrected count).
        """
        json_data = extract_json(response_text)

        if not json_data:
            logger.warning("Could not parse validation response, returning originals")
            return original_annotations, 0, 0

        validated: list[HPOAnnotation] = []

        # Add validated annotations (kept as-is or with minor adjustments)
        for item in json_data.get("validated_annotations", []):
            annotation = self._find_and_update_annotation(item, original_annotations)
            if annotation:
                validated.append(annotation)

        # Add corrected annotations
        corrected_count = 0
        for item in json_data.get("corrected_annotations", []):
            annotation = self._create_corrected_annotation(item)
            if annotation:
                validated.append(annotation)
                corrected_count += 1

        # Log removed annotations
        removed_items = json_data.get("removed_annotations", [])
        for item in removed_items:
            logger.info(
                "Removed annotation %s (%s): %s",
                item.get("hpo_id"),
                item.get("term_name"),
                item.get("reason"),
            )

        return validated, len(removed_items), corrected_count

    def _find_and_update_annotation(
        self,
        item: dict[str, Any],
        original_annotations: list[HPOAnnotation],
    ) -> HPOAnnotation | None:
        """Find original annotation and update with validation info."""
        hpo_id = item.get("hpo_id")
        if not hpo_id:
            return None

        # Find original
        for orig in original_annotations:
            if orig.hpo_id == hpo_id:
                # Return updated annotation
                return HPOAnnotation(
                    hpo_id=orig.hpo_id,
                    term_name=item.get("term_name", orig.term_name),
                    assertion=parse_assertion(
                        item.get("assertion", orig.assertion.value)
                    ),
                    confidence=float(item.get("confidence", orig.confidence)),
                    evidence_text=item.get("evidence_text", orig.evidence_text),
                    evidence_start=orig.evidence_start,
                    evidence_end=orig.evidence_end,
                    definition=orig.definition,
                    synonyms=orig.synonyms,
                    source_mode=orig.source_mode,
                    raw_score=orig.raw_score,
                )

        return None

    def _create_corrected_annotation(
        self,
        item: dict[str, Any],
    ) -> HPOAnnotation | None:
        """Create annotation from corrected item."""
        hpo_id = item.get("hpo_id")
        if not hpo_id:
            return None

        return HPOAnnotation(
            hpo_id=hpo_id,
            term_name=item.get("term_name", ""),
            assertion=parse_assertion(item.get("corrected_assertion", "affirmed")),
            confidence=float(item.get("confidence", 0.9)),
            evidence_text=item.get("evidence_text"),
        )
