"""Canonical EXPORT assertion vocabulary (affirmed / negated / normal / uncertain).

Used only at the export boundaries (MCP/REST/Vue) so an LLM ``assertion="absent"``
can never silently export as an affirmed (present) feature. The LLM pipeline keeps
its own present/negated/uncertain vocabulary -- do not use this inside the pipeline.
"""

from __future__ import annotations

AFFIRMED = "affirmed"
NEGATED = "negated"
NORMAL = "normal"
UNCERTAIN = "uncertain"

_NEGATED = {"negated", "negative", "absent", "excluded", "no", "denied"}
_UNCERTAIN = {"uncertain", "possible", "suspected", "probable"}
_NORMAL = {"normal"}


def canonicalize_assertion(raw: str | None) -> str:
    normalized = str(raw).strip().lower() if raw is not None else ""
    if normalized in _NEGATED:
        return NEGATED
    if normalized in _UNCERTAIN:
        return UNCERTAIN
    if normalized in _NORMAL:
        return NORMAL
    return AFFIRMED


def is_excluded(raw: str | None) -> bool:
    """True iff ``raw`` denotes a ruled-out finding (``excluded: true``)."""
    return canonicalize_assertion(raw) == NEGATED
