"""Shared extraction evaluation types."""

from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """Single document extraction result."""

    doc_id: str
    predicted: list[tuple[str, str]]  # (hpo_id, assertion)
    gold: list[tuple[str, str]]  # (hpo_id, assertion)
