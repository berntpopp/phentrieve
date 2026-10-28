"""
Shared HPO term-to-term similarity computation.

The REST similarity router and the MCP ``compare_hpo_terms`` tool both call
``compute_term_similarity``; they only translate its errors and shape its
result for their transport (issue #415).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.evaluation.metrics import (
    HPODataUnavailableError,
    SimilarityFormula,
    _resolve_hpo_db_path,
    calculate_semantic_similarity,
    find_lowest_common_ancestor,
    load_hpo_graph_data,
)
from phentrieve.utils import normalize_id
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)

__all__ = [
    "HPODataUnavailableError",
    "HPOTermNotFoundError",
    "TermSimilarityResult",
    "compute_term_similarity",
    "load_hpo_labels",
]


class HPOTermNotFoundError(LookupError):
    """Raised when one or both HPO terms are absent from the ontology graph."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            f"HPO term(s) not found in ontology data: {', '.join(missing)}"
        )


@dataclass(frozen=True)
class TermSimilarityResult:
    """Similarity between two HPO terms, with optional LCA details."""

    term1_id: str
    term2_id: str
    term1_label: str | None
    term2_label: str | None
    formula: SimilarityFormula
    similarity_score: float
    term1_depth: int
    term2_depth: int
    max_depth: int
    # Lowest common ancestor; only populated when requested and one exists.
    lca_id: str | None = None
    lca_label: str | None = None
    lca_depth: int | None = None


def load_hpo_labels(
    term_ids: Iterable[str | None], db_path: str | None = None
) -> dict[str, str]:
    """Best-effort id -> label lookup from the same database as the graph data.

    Labels are a non-essential enrichment: returns ``{}`` when the database is
    missing or unreadable. Nothing is cached, so labels appear as soon as the
    data has been prepared.
    """
    ids = sorted({term_id for term_id in term_ids if term_id})
    if not ids:
        return {}
    resolved_path = _resolve_hpo_db_path(db_path)
    if not os.path.exists(resolved_path):
        return {}
    try:
        with HPODatabase(resolved_path) as db:
            terms = db.get_terms_by_ids(ids)
    except Exception as exc:  # noqa: BLE001 - labels are optional enrichment
        logger.warning("Could not load HPO labels: %s", _sanitize(exc))
        return {}
    return {
        term_id: term["label"] for term_id, term in terms.items() if term.get("label")
    }


def compute_term_similarity(
    term1_id: str,
    term2_id: str,
    formula: SimilarityFormula = SimilarityFormula.HYBRID,
    *,
    with_lca: bool = False,
) -> TermSimilarityResult:
    """Compute the ontology similarity of two HPO terms.

    Args:
        term1_id: First HPO term ID (normalized via ``normalize_id``).
        term2_id: Second HPO term ID (normalized via ``normalize_id``).
        formula: Similarity formula to apply.
        with_lca: Also resolve the lowest common ancestor (id, label, depth).

    Raises:
        HPODataUnavailableError: If the HPO graph data is not available.
        HPOTermNotFoundError: If either term is not in the ontology graph.
    """
    t1 = normalize_id(term1_id)
    t2 = normalize_id(term2_id)

    ancestors, depths = load_hpo_graph_data()
    if not ancestors or not depths:
        raise HPODataUnavailableError("HPO graph data is not available")

    missing = [term for term in (t1, t2) if term not in depths]
    if missing:
        raise HPOTermNotFoundError(list(dict.fromkeys(missing)))

    score = calculate_semantic_similarity(t1, t2, formula=formula)

    lca_id: str | None = None
    lca_depth: int | None = None
    if with_lca:
        found_id, found_depth = find_lowest_common_ancestor(
            t1, t2, ancestors_dict=ancestors
        )
        if found_id is not None and found_depth != -1:
            lca_id, lca_depth = found_id, int(found_depth)

    labels = load_hpo_labels([t1, t2, lca_id])
    return TermSimilarityResult(
        term1_id=t1,
        term2_id=t2,
        term1_label=labels.get(t1),
        term2_label=labels.get(t2),
        formula=formula,
        similarity_score=score,
        term1_depth=depths[t1],
        term2_depth=depths[t2],
        max_depth=max(depths.values()),
        lca_id=lca_id,
        lca_label=labels.get(lca_id) if lca_id else None,
        lca_depth=lca_depth,
    )
