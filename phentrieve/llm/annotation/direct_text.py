"""
Direct text annotation strategy.

This strategy has the LLM output HPO IDs directly from its training knowledge,
without using any tools. This is the fastest approach but may hallucinate
non-existent HPO terms.
"""

import json
import logging
import re
import time
from typing import Any

from phentrieve.llm.annotation.base import AnnotationStrategy
from phentrieve.llm.prompts import get_prompt
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import (
    AnnotationMode,
    AnnotationResult,
    AssertionStatus,
    HPOAnnotation,
    TimingEvent,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class DirectTextStrategy(AnnotationStrategy):
    """
    Direct text annotation strategy.

    The LLM directly outputs HPO IDs from its training knowledge in a single
    API call. This is the fastest approach but relies entirely on the model's
    knowledge of HPO, which may be incomplete or outdated.

    Pros:
    - Single API call (fast, low cost)
    - Simple implementation
    - Works with any LLM

    Cons:
    - May hallucinate non-existent HPO IDs
    - Limited to model's training data cutoff
    - No retrieval augmentation for accuracy
    """

    mode = AnnotationMode.DIRECT

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize with an LLM provider."""
        super().__init__(provider)

    def annotate(
        self,
        text: str,
        language: str = "en",
        validate_hpo_ids: bool = True,
    ) -> AnnotationResult:
        """
        Extract HPO annotations using direct LLM output.

        Args:
            text: The clinical text to annotate.
            language: Language code for prompt selection.
            validate_hpo_ids: Whether to validate HPO IDs against database.

        Returns:
            AnnotationResult with extracted annotations.
        """
        start_time = time.time()

        # Load prompt template
        prompt_template = get_prompt(self.mode, language)

        # Build messages
        messages = prompt_template.get_messages(text)

        # Get completion from LLM
        llm_t0 = time.time()
        response = self.provider.complete(messages)
        llm_elapsed = time.time() - llm_t0

        # Track token usage
        token_usage = TokenUsage.from_response(response.usage, llm_time=llm_elapsed)
        token_usage.timing_events.append(
            TimingEvent(
                label=f"LLM call ({self.provider.model})",
                duration_seconds=llm_elapsed,
                category="llm",
            )
        )

        # Parse the response
        annotations = self._parse_response(response.content or "")

        # Validate HPO IDs if requested
        if validate_hpo_ids:
            annotations = self._validate_annotations(annotations)

        processing_time = time.time() - start_time

        return AnnotationResult(
            annotations=annotations,
            input_text=text,
            language=language,
            mode=self.mode,
            model=self.provider.model,
            prompt_version=prompt_template.version,
            temperature=self.provider.temperature,
            raw_llm_response=response.content,
            processing_time_seconds=processing_time,
            token_usage=token_usage,
        )

    def _parse_response(self, response_text: str) -> list[HPOAnnotation]:
        """
        Parse JSON annotations from LLM response.

        Args:
            response_text: Raw response from LLM.

        Returns:
            List of parsed HPOAnnotation objects.
        """
        annotations: list[HPOAnnotation] = []

        # Try to extract JSON from the response
        json_data = self._extract_json(response_text)

        if not json_data or "annotations" not in json_data:
            logger.warning(
                "Could not parse annotations from response: %s", response_text[:200]
            )
            return annotations

        for item in json_data.get("annotations", []):
            try:
                annotation = self._parse_annotation_item(item)
                if annotation:
                    annotations.append(annotation)
            except Exception as e:
                logger.warning("Failed to parse annotation item: %s - %s", item, e)

        return annotations

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract JSON object from text, handling markdown code blocks."""
        # Try to find JSON in code block
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None

        try:
            parsed: dict[str, Any] = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error: %s", e)
            return None

    def _parse_annotation_item(self, item: dict[str, Any]) -> HPOAnnotation | None:
        """Parse a single annotation item from JSON."""
        hpo_id = item.get("hpo_id", "")
        if not hpo_id:
            return None

        # Normalize HPO ID format
        hpo_id = self._normalize_hpo_id(hpo_id)
        if not hpo_id:
            return None

        # Parse assertion status
        assertion_str = item.get("assertion", "affirmed").lower()
        assertion = self._parse_assertion(assertion_str)

        return HPOAnnotation(
            hpo_id=hpo_id,
            term_name=item.get("term_name", ""),
            assertion=assertion,
            confidence=float(item.get("confidence", 1.0)),
            evidence_text=item.get("evidence_text"),
            source_mode=self.mode,
        )

    def _normalize_hpo_id(self, hpo_id: str) -> str | None:
        """Normalize HPO ID to standard format (HP:XXXXXXX)."""
        # Handle various formats
        hpo_id = hpo_id.strip().upper()

        # Already in correct format
        if re.match(r"^HP:\d{7}$", hpo_id):
            return hpo_id

        # Missing HP: prefix
        if re.match(r"^\d{7}$", hpo_id):
            return f"HP:{hpo_id}"

        # Has HP prefix but different separator
        match = re.match(r"^HP[:\s_-]?(\d+)$", hpo_id, re.IGNORECASE)
        if match:
            number = match.group(1).zfill(7)
            return f"HP:{number}"

        logger.warning("Invalid HPO ID format: %s", hpo_id)
        return None

    def _parse_assertion(self, assertion_str: str) -> AssertionStatus:
        """Parse assertion string to enum."""
        assertion_str = assertion_str.lower().strip()

        if assertion_str in (
            "negated",
            "negative",
            "absent",
            "excluded",
            "no",
            "denied",
        ):
            return AssertionStatus.NEGATED
        elif assertion_str in ("uncertain", "possible", "suspected", "probable"):
            return AssertionStatus.UNCERTAIN
        else:
            return AssertionStatus.AFFIRMED

    def _validate_annotations(
        self,
        annotations: list[HPOAnnotation],
    ) -> list[HPOAnnotation]:
        """
        Validate HPO IDs against the database.

        Invalid IDs are logged and removed from the results.
        """
        try:
            from pathlib import Path

            from phentrieve.config import DEFAULT_HPO_DB_FILENAME
            from phentrieve.data_processing.hpo_database import HPODatabase
            from phentrieve.utils import get_default_data_dir

            # Search multiple locations for the HPO database
            candidates = [
                get_default_data_dir() / DEFAULT_HPO_DB_FILENAME,  # User config dir
                Path.cwd() / "data" / DEFAULT_HPO_DB_FILENAME,  # Project ./data
                Path(__file__).resolve().parents[3]
                / "data"
                / DEFAULT_HPO_DB_FILENAME,  # Package root
            ]

            db_path = None
            for candidate in candidates:
                if candidate.exists():
                    db_path = candidate
                    break

            if db_path is None:
                logger.debug(
                    "[VALIDATE] HPO database not found - skipping ID validation. "
                    "Run 'phentrieve data prepare' to download HPO data."
                )
                return annotations

            db = HPODatabase(db_path)
            hpo_ids = [a.hpo_id for a in annotations]
            valid_terms = db.get_terms_by_ids(hpo_ids)
            valid_ids = set(valid_terms.keys())

            validated = []
            invalid_count = 0
            for annotation in annotations:
                if annotation.hpo_id in valid_ids:
                    validated.append(annotation)
                else:
                    invalid_count += 1
                    logger.warning(
                        "[VALIDATE] Removing %s (%s) - not found in HPO database",
                        annotation.hpo_id,
                        annotation.term_name,
                    )

            if invalid_count > 0:
                logger.info(
                    "[VALIDATE] Result: %d of %d annotations have valid HPO IDs",
                    len(validated),
                    len(annotations),
                )
            else:
                logger.debug(
                    "[VALIDATE] All %d HPO IDs verified against database",
                    len(annotations),
                )

            return validated

        except Exception as e:
            logger.warning(
                "[VALIDATE] Database validation failed (%s) - annotations not validated",
                e,
            )
            return annotations
