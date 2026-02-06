"""
Assertion review post-processor.

This processor validates the assertion status (affirmed/negated) of
each annotation, specifically focusing on negation detection accuracy.
"""

import json
import logging
import re
from typing import Any

from phentrieve.llm.postprocess.base import PostProcessor
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import (
    AssertionStatus,
    HPOAnnotation,
    PostProcessingStep,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# System prompt for assertion review
ASSERTION_REVIEW_PROMPT = """You are an expert clinical geneticist reviewing HPO annotations for correct assertion status.

Your task is to verify the assertion status (affirmed/negated/uncertain) for each annotation.

Pay special attention to:
1. Negation phrases: "no", "without", "denies", "absent", "negative for", "ruled out"
2. Uncertainty phrases: "possible", "suspected", "probable", "may have", "might"
3. Affirmation phrases: "has", "presents with", "shows", "demonstrates", "exhibits"

Context matters:
- "History of seizures" = affirmed (past history counts)
- "Family history of seizures" = NOT the patient, should not be annotated
- "No evidence of seizures" = negated
- "Possible seizures" = uncertain

OUTPUT FORMAT (JSON):
```json
{
  "reviewed_annotations": [
    {
      "hpo_id": "HP:XXXXXXX",
      "term_name": "Term name",
      "original_assertion": "affirmed|negated|uncertain",
      "correct_assertion": "affirmed|negated|uncertain",
      "confidence": 0.95,
      "evidence_text": "exact quote showing assertion context",
      "review_note": "Explanation if assertion was changed"
    }
  ]
}
```"""


class AssertionReviewPostProcessor(PostProcessor):
    """
    Assertion review post-processor for validating negation detection.

    This processor specifically focuses on verifying that the assertion
    status (affirmed/negated/uncertain) is correct for each annotation.
    It's particularly useful for catching negation detection errors.
    """

    step = PostProcessingStep.ASSERTION_REVIEW

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
        Review assertion status for all annotations.

        Args:
            annotations: The annotations to review.
            original_text: The original clinical text.
            language: Language code (used for context but prompts are in English).

        Returns:
            Tuple of (annotations with corrected assertion status, token usage).
        """
        if not annotations:
            return annotations, TokenUsage()

        # Format annotations for the prompt
        annotations_json = json.dumps(
            [a.to_dict() for a in annotations],
            indent=2,
        )

        user_prompt = f"""Please review the assertion status for each annotation.

ORIGINAL TEXT:
---
{original_text}
---

ANNOTATIONS TO REVIEW:
{annotations_json}

For each annotation, verify if the assertion status (affirmed/negated/uncertain) is correct based on the text context."""

        messages = [
            {"role": "system", "content": ASSERTION_REVIEW_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Get review response
        response = self.provider.complete(messages)

        # Track token usage
        token_usage = TokenUsage.from_response(response.usage)

        # Parse and apply corrections
        reviewed = self._parse_review_response(
            response.content or "",
            annotations,
        )

        return reviewed, token_usage

    def _parse_review_response(
        self,
        response_text: str,
        original_annotations: list[HPOAnnotation],
    ) -> list[HPOAnnotation]:
        """Parse review response and apply corrections."""
        json_data = self._extract_json(response_text)

        if not json_data or "reviewed_annotations" not in json_data:
            logger.warning(
                "Could not parse assertion review response, returning originals"
            )
            return original_annotations

        # Build lookup from original annotations
        orig_by_id = {a.hpo_id: a for a in original_annotations}

        result: list[HPOAnnotation] = []
        reviewed_ids: set[str] = set()

        for item in json_data.get("reviewed_annotations", []):
            hpo_id = item.get("hpo_id")
            if not hpo_id or hpo_id not in orig_by_id:
                continue

            orig = orig_by_id[hpo_id]
            reviewed_ids.add(hpo_id)

            correct_assertion = item.get("correct_assertion", "")

            # Check if assertion changed
            new_assertion = self._parse_assertion(correct_assertion)

            if new_assertion != orig.assertion:
                logger.info(
                    "Corrected assertion for %s: %s -> %s (%s)",
                    hpo_id,
                    orig.assertion.value,
                    new_assertion.value,
                    item.get("review_note", ""),
                )

            # Create updated annotation
            result.append(
                HPOAnnotation(
                    hpo_id=orig.hpo_id,
                    term_name=item.get("term_name", orig.term_name),
                    assertion=new_assertion,
                    confidence=float(item.get("confidence", orig.confidence)),
                    evidence_text=item.get("evidence_text", orig.evidence_text),
                    evidence_start=orig.evidence_start,
                    evidence_end=orig.evidence_end,
                    definition=orig.definition,
                    synonyms=orig.synonyms,
                    source_mode=orig.source_mode,
                    raw_score=orig.raw_score,
                )
            )

        # Add any annotations that weren't reviewed
        for orig in original_annotations:
            if orig.hpo_id not in reviewed_ids:
                result.append(orig)

        return result

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
