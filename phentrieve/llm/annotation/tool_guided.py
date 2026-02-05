"""
Tool-guided annotation strategy.

This strategy uses Phentrieve tools to assist the LLM in finding and
validating HPO terms. It supports two sub-modes:

- Term Search: LLM extracts phrases, queries Phentrieve, selects best matches
- Text Process: Phentrieve processes text through full pipeline, LLM validates
"""

import json
import logging
import re
import time
from typing import Any

from phentrieve.llm.annotation.base import AnnotationStrategy
from phentrieve.llm.prompts import get_prompt
from phentrieve.llm.provider import LLMProvider, ToolExecutor
from phentrieve.llm.types import (
    AnnotationMode,
    AnnotationResult,
    AssertionStatus,
    HPOAnnotation,
    ToolCall,
)

logger = logging.getLogger(__name__)


class ToolGuidedStrategy(AnnotationStrategy):
    """
    Tool-guided annotation strategy using Phentrieve tools.

    This strategy provides the LLM with tools to query HPO terms or process
    clinical text through Phentrieve's pipeline. The LLM decides how to use
    the tools and validates/selects from the results.

    Sub-modes:
    - TOOL_TERM: LLM extracts clinical phrases and queries for matching HPO terms
    - TOOL_TEXT: Phentrieve processes text, LLM reviews and selects candidates

    Attributes:
        mode: Either TOOL_TERM or TOOL_TEXT annotation mode.
        provider: The LLM provider instance.
        tool_executor: Executor for Phentrieve tools.
        max_iterations: Maximum tool call iterations before stopping.
    """

    def __init__(
        self,
        provider: LLMProvider,
        mode: AnnotationMode = AnnotationMode.TOOL_TEXT,
        max_iterations: int | None = None,
    ) -> None:
        """
        Initialize the tool-guided strategy.

        Args:
            provider: The LLM provider to use.
            mode: Either TOOL_TERM or TOOL_TEXT.
            max_iterations: Maximum number of tool call iterations.
                           Defaults to 2 for TOOL_TEXT (one call + response),
                           5 for TOOL_TERM (may need multiple queries).
        """
        if mode not in (AnnotationMode.TOOL_TERM, AnnotationMode.TOOL_TEXT):
            raise ValueError(f"Invalid mode for ToolGuidedStrategy: {mode}")

        super().__init__(provider)
        self.mode = mode
        self.tool_executor = ToolExecutor()

        # TOOL_TEXT only needs 1-2 iterations (call tool once, get response)
        # TOOL_TERM may need more iterations for multiple term queries
        if max_iterations is None:
            self.max_iterations = 2 if mode == AnnotationMode.TOOL_TEXT else 5
        else:
            self.max_iterations = max_iterations

    def annotate(
        self,
        text: str,
        language: str = "en",
        validate_hpo_ids: bool = True,
    ) -> AnnotationResult:
        """
        Extract HPO annotations using tool-guided approach.

        Args:
            text: The clinical text to annotate.
            language: Language code for prompt selection.
            validate_hpo_ids: Whether to validate HPO IDs against database.

        Returns:
            AnnotationResult with extracted annotations and tool call log.
        """
        start_time = time.time()

        # Load prompt template based on mode
        prompt_template = get_prompt(self.mode, language)

        # Build initial messages
        messages = prompt_template.get_messages(text)

        # Check if provider supports tools
        if self.provider.supports_tools():
            # Use native tool calling
            response, tool_calls = self.provider.complete_with_tools(
                messages=messages,
                tool_executor=self.tool_executor,
                max_iterations=self.max_iterations,
            )
        else:
            # Fallback to prompt-based tool emulation
            response, tool_calls = self._emulate_tool_calling(messages, text, language)

        # Parse the final response
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
            tool_calls=tool_calls,
            raw_llm_response=response.content,
            processing_time_seconds=processing_time,
        )

    def _emulate_tool_calling(
        self,
        messages: list[dict[str, str]],
        text: str,
        language: str,
    ) -> tuple[Any, list[ToolCall]]:
        """
        Emulate tool calling for models without native support.

        This method adds tool results directly to the prompt when the model
        doesn't support native function calling.
        """
        tool_calls: list[ToolCall] = []

        if self.mode == AnnotationMode.TOOL_TEXT:
            # For text processing mode, run the pipeline and include results
            result = self.tool_executor.execute(
                "process_clinical_text",
                {"text": text, "language": language},
            )

            tool_call = ToolCall(
                name="process_clinical_text",
                arguments={"text": text, "language": language},
                result=result,
            )
            tool_calls.append(tool_call)

            # Add tool results to the user message
            tool_results_text = f"\n\nPHENTRIEVE ANALYSIS RESULTS:\n```json\n{json.dumps(result, indent=2)}\n```\n\nPlease review these candidate annotations and provide your final validated annotations."

            # Modify the last user message to include results
            messages[-1]["content"] += tool_results_text

        # Get final response
        response = self.provider.complete(messages)

        return response, tool_calls

    def _parse_response(self, response_text: str) -> list[HPOAnnotation]:
        """Parse JSON annotations from LLM response."""
        annotations: list[HPOAnnotation] = []

        json_data = self._extract_json(response_text)

        if not json_data or "annotations" not in json_data:
            logger.warning("Could not parse annotations from response")
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

        hpo_id = self._normalize_hpo_id(hpo_id)
        if not hpo_id:
            return None

        assertion_str = item.get("assertion", "affirmed").lower()
        assertion = self._parse_assertion(assertion_str)

        return HPOAnnotation(
            hpo_id=hpo_id,
            term_name=item.get("term_name", ""),
            assertion=assertion,
            confidence=float(item.get("confidence", 1.0)),
            evidence_text=item.get("evidence_text"),
            source_mode=self.mode,
            raw_score=item.get("score"),
        )

    def _normalize_hpo_id(self, hpo_id: str) -> str | None:
        """Normalize HPO ID to standard format (HP:XXXXXXX)."""
        hpo_id = hpo_id.strip().upper()

        if re.match(r"^HP:\d{7}$", hpo_id):
            return hpo_id

        if re.match(r"^\d{7}$", hpo_id):
            return f"HP:{hpo_id}"

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
        """Validate HPO IDs against the database."""
        try:
            from phentrieve.config import get_config_value
            from phentrieve.data_processing.hpo_database import HPODatabase

            db_path = get_config_value("data", None, "hpo_database_path")
            if not db_path:
                logger.debug("No HPO database path configured, skipping validation")
                return annotations

            db = HPODatabase(db_path)
            hpo_ids = [a.hpo_id for a in annotations]
            valid_terms = db.get_terms_by_ids(hpo_ids)
            valid_ids = set(valid_terms.keys())

            validated = []
            for annotation in annotations:
                if annotation.hpo_id in valid_ids:
                    validated.append(annotation)
                else:
                    logger.warning(
                        "Invalid or unknown HPO ID removed: %s (%s)",
                        annotation.hpo_id,
                        annotation.term_name,
                    )

            return validated

        except Exception as e:
            logger.warning("HPO validation failed, returning unvalidated: %s", e)
            return annotations
