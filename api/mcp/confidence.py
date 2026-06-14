"""Low-confidence signalling for retrieval results (assessment defect D6).

Embedding search returns nearest neighbours even for gibberish, so a naive
consumer can mistake a 0.55 hit for a real match. We annotate each result with a
``confidence_band`` and set a top-level ``no_high_confidence_match`` flag when the
best hit is below the high floor, so the client can abstain or widen its query
instead of silently trusting noise. The similarity_threshold is unchanged: this
signals, it does not re-filter.
"""

from __future__ import annotations

from typing import Any

# Cosine-similarity floors for the embedding retriever's score scale.
HIGH_FLOOR = 0.7
MODERATE_FLOOR = 0.4


def confidence_band(score: float) -> str:
    """Map a similarity score to high | moderate | low."""
    if score >= HIGH_FLOOR:
        return "high"
    if score >= MODERATE_FLOOR:
        return "moderate"
    return "low"


def annotate_search_confidence(
    payload: dict[str, Any], *, score_key: str = "similarity"
) -> dict[str, Any]:
    """Annotate each result with confidence_band and flag a low-confidence top hit.

    Returns a shallow copy; the flag is computed over the full result set (before
    any token-budget truncation) so it reflects the query's true best match.
    """
    results = payload.get("results") or []
    top = max((r.get(score_key, 0.0) for r in results), default=0.0)
    for result in results:
        result["confidence_band"] = confidence_band(result.get(score_key, 0.0))
    out = dict(payload)
    out["no_high_confidence_match"] = bool(results) and top < HIGH_FLOOR
    return out
