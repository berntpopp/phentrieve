"""
Shared utility functions for the LLM annotation system.

This module consolidates helper functions that were previously duplicated
across annotation strategies and post-processors:
- JSON extraction from LLM responses
- HPO ID normalization
- Assertion status parsing
"""

import json
import logging
import re
from typing import Any

from phentrieve.llm.types import AssertionStatus

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM response text.

    Handles markdown code blocks (```json ... ```) and raw JSON objects.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON dict, or None if extraction/parsing fails.
    """
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


def normalize_hpo_id(hpo_id: str) -> str | None:
    """Normalize an HPO ID to the standard HP:XXXXXXX format.

    Accepts various formats:
    - ``HP:0001250`` (already correct)
    - ``0001250`` (missing prefix)
    - ``HP 1250``, ``HP_1250``, ``HP-1250`` (alternative separators, short numbers)

    Args:
        hpo_id: Raw HPO ID string.

    Returns:
        Normalized ID like ``HP:0001250``, or None if the format is invalid.
    """
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


def parse_assertion(assertion_str: str | AssertionStatus) -> AssertionStatus:
    """Parse an assertion string into an AssertionStatus enum.

    Accepts the canonical enum values plus common synonyms used by LLMs:
    - Negated: ``negated``, ``negative``, ``absent``, ``excluded``, ``no``, ``denied``
    - Uncertain: ``uncertain``, ``possible``, ``suspected``, ``probable``
    - Everything else maps to AFFIRMED.

    If *assertion_str* is already an ``AssertionStatus``, it is returned as-is.

    Args:
        assertion_str: Raw assertion string or AssertionStatus enum.

    Returns:
        The corresponding AssertionStatus enum value.
    """
    if isinstance(assertion_str, AssertionStatus):
        return assertion_str

    assertion_str = str(assertion_str).lower().strip()

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
