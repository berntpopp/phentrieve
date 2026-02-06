"""
Refinement post-processor.

This processor checks if more specific HPO terms could be used for
the annotations, upgrading general terms to more specific ones when
supported by the text.
"""

import json
import logging
import re
from typing import Any

from phentrieve.llm.postprocess.base import PostProcessor
from phentrieve.llm.prompts import load_prompt_template
from phentrieve.llm.provider import PHENTRIEVE_TOOLS, LLMProvider
from phentrieve.llm.types import (
    AssertionStatus,
    HPOAnnotation,
    PostProcessingStep,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class RefinementPostProcessor(PostProcessor):
    """
    Refinement post-processor for upgrading to more specific terms.

    The HPO is hierarchical, with general terms having more specific
    descendants. This processor checks if the clinical text supports
    using a more specific term than the one currently annotated.

    Example:
    - Original: HP:0001250 (Seizure)
    - Refined: HP:0002123 (Recurrent seizure) - if "recurrent" is in text
    """

    step = PostProcessingStep.REFINEMENT

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
        Refine annotations to use more specific HPO terms.

        Args:
            annotations: The annotations to refine.
            original_text: The original clinical text.
            language: Language code for prompt selection.

        Returns:
            Tuple of (refined annotations, token usage).
        """
        if not annotations:
            return annotations, TokenUsage()

        # Load refinement prompt template
        try:
            prompt_template = load_prompt_template(self.step, language)
        except FileNotFoundError:
            logger.warning(
                "Refinement prompt not found, returning original annotations"
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

        # Get refinement response (with tool access for searching)
        if self.provider.supports_tools():
            response = self.provider.complete(
                messages=messages,
                tools=PHENTRIEVE_TOOLS,
                tool_choice="auto",
            )
        else:
            response = self.provider.complete(messages)

        # Track token usage
        token_usage = TokenUsage.from_response(response.usage)

        # Parse the refinement result
        refined = self._parse_refinement_response(
            response.content or "",
            annotations,
        )

        return refined, token_usage

    def _parse_refinement_response(
        self,
        response_text: str,
        original_annotations: list[HPOAnnotation],
    ) -> list[HPOAnnotation]:
        """Parse refinement response and return updated annotations."""
        json_data = self._extract_json(response_text)

        if not json_data:
            logger.warning("Could not parse refinement response, returning originals")
            return original_annotations

        result: list[HPOAnnotation] = []
        processed_ids: set[str] = set()

        # Add kept annotations (already optimal)
        for item in json_data.get("kept_annotations", []):
            annotation = self._find_original_annotation(
                item.get("hpo_id"),
                original_annotations,
            )
            if annotation:
                result.append(annotation)
                processed_ids.add(annotation.hpo_id)

        # Add refined annotations
        for item in json_data.get("refined_annotations", []):
            annotation = self._create_refined_annotation(item)
            if annotation:
                result.append(annotation)
                # Mark original as processed
                orig_id = item.get("original_hpo_id")
                if orig_id:
                    processed_ids.add(orig_id)
                    logger.info(
                        "Refined %s (%s) -> %s (%s): %s",
                        orig_id,
                        item.get("original_term_name"),
                        annotation.hpo_id,
                        annotation.term_name,
                        item.get("refinement_reason"),
                    )

        # Add any unprocessed original annotations
        for orig in original_annotations:
            if orig.hpo_id not in processed_ids:
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

    def _find_original_annotation(
        self,
        hpo_id: str | None,
        original_annotations: list[HPOAnnotation],
    ) -> HPOAnnotation | None:
        """Find original annotation by HPO ID."""
        if not hpo_id:
            return None

        for orig in original_annotations:
            if orig.hpo_id == hpo_id:
                return orig

        return None

    def _create_refined_annotation(
        self,
        item: dict[str, Any],
    ) -> HPOAnnotation | None:
        """Create annotation from refined item."""
        hpo_id = item.get("refined_hpo_id")
        if not hpo_id:
            return None

        # Normalize HPO ID
        hpo_id = self._normalize_hpo_id(hpo_id)
        if not hpo_id:
            return None

        return HPOAnnotation(
            hpo_id=hpo_id,
            term_name=item.get("refined_term_name", ""),
            assertion=self._parse_assertion(item.get("assertion", "affirmed")),
            confidence=float(item.get("confidence", 0.9)),
            evidence_text=item.get("evidence_text"),
        )

    def _normalize_hpo_id(self, hpo_id: str) -> str | None:
        """Normalize HPO ID to standard format."""
        hpo_id = hpo_id.strip().upper()

        if re.match(r"^HP:\d{7}$", hpo_id):
            return hpo_id

        if re.match(r"^\d{7}$", hpo_id):
            return f"HP:{hpo_id}"

        match = re.match(r"^HP[:\s_-]?(\d+)$", hpo_id, re.IGNORECASE)
        if match:
            number = match.group(1).zfill(7)
            return f"HP:{number}"

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
