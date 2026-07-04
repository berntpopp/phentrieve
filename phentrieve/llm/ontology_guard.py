"""Ontology guard: reject resolved terms that are not phenotypic abnormalities.

Retrieval can drift into non-phenotype HPO branches -- Clinical modifier
(HP:0012823), Clinical course (HP:0031797), Mode of inheritance (HP:0000005),
Frequency, Blood group, Past medical history. For example a bare ``normal``
qualifier retrieves "Mild" (HP:0012825), and a "visuospatial" fragment can
retrieve "Spatial pattern" (HP:0012836). Patient phenotype annotations live
under Phenotypic abnormality (HP:0000118), so a resolved term KNOWN to sit
outside that subtree is not a valid standalone phenotype and is dropped.

Fail open: a term whose ancestry is unknown (or when the HPO graph cannot be
loaded) is kept, so incomplete data never silently drops a real finding.
"""

from __future__ import annotations

import logging

from phentrieve.config import DEFAULT_HPO_DB_FILENAME, PHENOTYPE_ROOT
from phentrieve.utils import get_default_data_dir, resolve_data_path

logger = logging.getLogger(__name__)

# Path-keyed cache of SUCCESSFUL, non-empty ancestry loads only. A failed or
# empty load is deliberately NOT memoized so the guard recovers on a later call
# once ``hpo_data.db`` becomes available (e.g. an API process that starts before
# the data bundle is present) instead of latching the fail-open ``{}`` for the
# whole process lifetime. Keying by resolved DB path also re-loads when the data
# root changes in-process rather than serving a stale map.
_ANCESTORS_CACHE: dict[str, dict[str, frozenset[str]]] = {}


def _load_ancestors_map(db_path) -> dict[str, frozenset[str]]:
    from phentrieve.data_processing.hpo_database import HPODatabase

    with HPODatabase(db_path) as db:
        ancestors, _ = db.load_graph_data()
    return {term: frozenset(anc) for term, anc in ancestors.items()}


def _ancestors_map() -> dict[str, frozenset[str]]:
    """Return ``{term_id: frozenset(ancestor_ids)}`` from the HPO graph.

    Returns an empty map (guard disabled, fail open) when the database is
    unavailable so the pipeline never hard-depends on graph data at runtime; the
    empty result is not cached, so a later call retries.
    """
    try:
        data_dir = resolve_data_path(None, "data_dir", get_default_data_dir)
        db_path = data_dir / DEFAULT_HPO_DB_FILENAME
        cache_key = str(db_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Ontology guard could not resolve HPO db path (%s); disabled (fail open).",
            exc,
        )
        return {}

    cached = _ANCESTORS_CACHE.get(cache_key)
    if cached:
        return cached

    try:
        if not db_path.exists():
            logger.warning(
                "HPO database not found: %s; ontology guard disabled "
                "(fail open, will retry).",
                db_path,
            )
            return {}
        loaded = _load_ancestors_map(db_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Ontology guard could not load HPO graph (%s); disabled "
            "(fail open, will retry).",
            exc,
        )
        return {}

    # Only memoize a successful, non-empty load.
    if loaded:
        _ANCESTORS_CACHE[cache_key] = loaded
    return loaded


def is_non_phenotypic_abnormality(term_id: str) -> bool:
    """True iff ``term_id`` is KNOWN to sit outside the Phenotypic abnormality
    (HP:0000118) subtree (a clinical modifier / course / inheritance / etc.).

    Unknown terms and the phenotype root itself return ``False`` (kept).
    """
    term_id = (term_id or "").strip()
    if not term_id or term_id == PHENOTYPE_ROOT:
        return False
    ancestors = _ancestors_map().get(term_id)
    if ancestors is None:
        return False  # unknown ancestry -> keep (fail open)
    return PHENOTYPE_ROOT not in ancestors
