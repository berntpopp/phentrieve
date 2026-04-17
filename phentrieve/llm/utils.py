from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
HPO_ID_PATTERN = re.compile(r"\b(HP[:\s_-]?\d{1,7})\b", re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()

    for code_block_match in JSON_BLOCK_PATTERN.finditer(text):
        payload = code_block_match.group(1).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    for brace_match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[brace_match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    logger.debug("Failed to parse JSON payload from LLM response.")
    return None


def extract_hpo_id(text: str) -> str | None:
    match = HPO_ID_PATTERN.search(text)
    if not match:
        return None
    return normalize_hpo_id(match.group(1))


def token_sort_similarity(left: str, right: str) -> float:
    left_tokens = sorted(left.split())
    right_tokens = sorted(right.split())
    return (
        SequenceMatcher(
            None,
            " ".join(left_tokens),
            " ".join(right_tokens),
        ).ratio()
        * 100.0
    )


def normalize_hpo_id(hpo_id: str) -> str | None:
    """Normalize loose HPO identifiers into canonical HP:XXXXXXX form."""
    cleaned = hpo_id.strip().upper()

    if re.fullmatch(r"HP:\d{7}", cleaned):
        return cleaned
    if re.fullmatch(r"\d{7}", cleaned):
        return f"HP:{cleaned}"

    match = re.fullmatch(r"HP[:\s_-]?(\d+)", cleaned, re.IGNORECASE)
    if match:
        return f"HP:{match.group(1).zfill(7)}"

    logger.warning("Invalid HPO ID format: %s", hpo_id)
    return None


def parse_assertion(assertion_str: str | Any) -> Any:
    """Parse an assertion string into the canonical AssertionStatus enum."""
    from phentrieve.llm.types import AssertionStatus

    if isinstance(assertion_str, AssertionStatus):
        return assertion_str

    normalized = str(assertion_str).lower().strip()
    if normalized in {"negated", "negative", "absent", "excluded", "no", "denied"}:
        return AssertionStatus.NEGATED
    if normalized in {"uncertain", "possible", "suspected", "probable"}:
        return AssertionStatus.UNCERTAIN
    return AssertionStatus.PRESENT
