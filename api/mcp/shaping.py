"""Response-mode shaping and token budgets for the Phentrieve MCP server.

Tools accept ``response_mode`` (minimal | compact | standard | full, default
compact). :func:`apply_response_mode` projects verbose fields out of list/dict
payloads; :func:`enforce_budget` truncates an over-budget list field and reports
what was dropped (never silently). Char budgets follow the maintainer's *-link
house defaults.
"""

from __future__ import annotations

import json
from typing import Any, Literal

ResponseMode = Literal["minimal", "compact", "standard", "full"]
MODES: tuple[ResponseMode, ...] = ("minimal", "compact", "standard", "full")
DEFAULT_MODE: ResponseMode = "compact"

BUDGETS: dict[str, int] = {
    "minimal": 4000,
    "compact": 12000,
    "standard": 24000,
    "full": 48000,
}

# Verbose detail fields: dropped at minimal; dropped-if-present at compact;
# kept at standard/full.
_DETAIL_FIELDS = (
    "definition",
    "synonyms",
    "component_scores",
    "comments",
    "text_attributions",
    "assertion_details",
)
# Identity/score fields kept even at minimal verbosity.
_MINIMAL_KEEP = (
    "hpo_id",
    "label",
    "name",
    "similarity",
    "confidence",
    "score",
    "assertion",
    "status",
    "term1_id",
    "term2_id",
    "formula_used",
    "similarity_score",
    "chunk_id",
)

# Keys whose empty value must still be serialized (schema-stability contract).
# hpo_matches: [] must never collapse to a missing key (defect L5).
_ALWAYS_KEEP_EMPTY = ("hpo_matches",)


def resolve_mode(requested: str | None) -> ResponseMode:
    """Validate/normalize a requested response_mode, defaulting to compact."""
    if requested is None:
        return DEFAULT_MODE
    if requested not in MODES:
        raise ValueError(f"response_mode must be one of {MODES}")
    return requested


def _shape_item(
    item: dict[str, Any],
    mode: ResponseMode,
    keep_detail_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    if mode == "full":
        return item
    if mode == "minimal":
        keep = set(_MINIMAL_KEEP) | set(keep_detail_fields)
        return {k: v for k, v in item.items() if k in keep and v is not None}
    out: dict[str, Any] = {}
    for key, value in item.items():
        if key in _ALWAYS_KEEP_EMPTY and (value == [] or value is None):
            out[key] = []
            continue
        if value is None or value == [] or value == {}:
            continue
        if (
            mode == "compact"
            and key in _DETAIL_FIELDS
            and key not in keep_detail_fields
        ):
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            out[key] = [_shape_item(v, mode, keep_detail_fields) for v in value]
        elif isinstance(value, dict):
            out[key] = _shape_item(value, mode, keep_detail_fields)
        else:
            out[key] = value
    return out


def apply_response_mode(
    payload: dict[str, Any],
    mode: ResponseMode,
    *,
    keep_detail_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a shaped copy of ``payload`` for the given mode.

    Lists of dict items are shaped per item; nested dicts are shaped recursively;
    scalar and ``_meta`` keys pass through. ``full`` returns the payload unchanged.
    ``keep_detail_fields`` names detail fields (e.g. definition/synonyms) that must
    survive compact/minimal because the caller explicitly requested them
    (honors include_details=True at compact verbosity, defect M5).
    """
    if mode == "full":
        return payload
    shaped: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "_meta":
            shaped[key] = value
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            shaped[key] = [_shape_item(i, mode, keep_detail_fields) for i in value]
        elif isinstance(value, dict):
            shaped[key] = _shape_item(value, mode, keep_detail_fields)
        elif value is None and mode in ("minimal", "compact"):
            continue
        else:
            shaped[key] = value
    return shaped


def _chars(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, default=str))


def enforce_budget(
    payload: dict[str, Any], mode: ResponseMode, *, list_field: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Truncate ``payload[list_field]`` until under the mode's char budget.

    Returns ``(payload, truncation_info_or_None)`` where ``truncation_info`` is
    ``{"field", "returned", "total"}``. ``full`` mode is never truncated.
    """
    if mode == "full":
        return payload, None
    budget = BUDGETS[mode]
    items = payload.get(list_field)
    if _chars(payload) <= budget or not isinstance(items, list):
        return payload, None
    total = len(items)
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _chars({**payload, list_field: items[:mid]}) <= budget:
            lo = mid
        else:
            hi = mid - 1
    payload = {**payload, list_field: items[:lo]}
    return payload, {"field": list_field, "returned": lo, "total": total}
