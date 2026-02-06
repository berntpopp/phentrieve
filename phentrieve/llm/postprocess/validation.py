"""
Validation post-processor.

This processor has the LLM re-check each annotation against the original
text to remove false positives and correct errors.
"""

import json
import logging
import re
from typing import Any

from phentrieve.llm.postprocess.base import PostProcessor
from phentrieve.llm.prompts import load_prompt_template
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import (
    AssertionStatus,
    HPOAnnotation,
    PostProcessingStep,
    TokenUsage,
)

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
    ) -> tuple[list[HPOAnnotation], TokenUsage]:
        """
        Validate annotations against the original text.

        Args:
            annotations: The annotations to validate.
            original_text: The original clinical text.
            language: Language code for prompt selection.

        Returns:
            Tuple of (validated annotations, token usage).
        """
        if not annotations:
            return annotations, TokenUsage()

        # Load validation prompt template
        try:
            prompt_template = load_prompt_template(self.step, language)
        except FileNotFoundError:
            logger.warning(
                "Validation prompt not found, returning original annotations"
            )
            return annotations, TokenUsage()

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
        response = self.provider.complete(messages)

        # Track token usage
        token_usage = TokenUsage.from_response(response.usage)

        # Parse the validation result
        validated = self._parse_validation_response(
            response.content or "",
            annotations,
        )

        return validated, token_usage

    def _parse_validation_response(
        self,
        response_text: str,
        original_annotations: list[HPOAnnotation],
    ) -> list[HPOAnnotation]:
        """Parse validation response and return refined annotations."""
        json_data = self._extract_json(response_text)

        if not json_data:
            logger.warning("Could not parse validation response, returning originals")
            return original_annotations

        validated: list[HPOAnnotation] = []

        # Add validated annotations (kept as-is or with minor adjustments)
        for item in json_data.get("validated_annotations", []):
            annotation = self._find_and_update_annotation(item, original_annotations)
            if annotation:
                validated.append(annotation)

        # Add corrected annotations
        for item in json_data.get("corrected_annotations", []):
            annotation = self._create_corrected_annotation(item)
            if annotation:
                validated.append(annotation)

        # Log removed annotations
        for item in json_data.get("removed_annotations", []):
            logger.info(
                "Removed annotation %s (%s): %s",
                item.get("hpo_id"),
                item.get("term_name"),
                item.get("reason"),
            )

        return validated

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from response text."""
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None

        try:
            parsed: dict[str, Any] = json.loads(json_str)
            return parsed
        except json.JSONDecodeError:
            return None

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
                    assertion=self._parse_assertion(
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
            assertion=self._parse_assertion(
                item.get("corrected_assertion", "affirmed")
            ),
            confidence=float(item.get("confidence", 0.9)),
            evidence_text=item.get("evidence_text"),
        )

    def _parse_assertion(self, assertion_str: str) -> AssertionStatus:
        """Parse assertion string to enum."""
        if isinstance(assertion_str, AssertionStatus):
            return assertion_str

        assertion_str = str(assertion_str).lower().strip()

        if assertion_str in ("negated", "negative", "absent", "excluded"):
            return AssertionStatus.NEGATED
        elif assertion_str in ("uncertain", "possible", "suspected"):
            return AssertionStatus.UNCERTAIN
        else:
            return AssertionStatus.AFFIRMED
