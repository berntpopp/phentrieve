"""
HPO term details enrichment utilities.

Provides pure functions for adding definitions and synonyms to query results.
Following functional programming principles: no side effects, returns new objects.
"""

import logging
from functools import lru_cache

from phentrieve.config import DEFAULT_HPO_DB_FILENAME
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.utils import get_default_data_dir, resolve_data_path

logger = logging.getLogger(__name__)


class DatabaseNotFoundError(FileNotFoundError):
    """Raised when HPO database file doesn't exist."""

    pass


@lru_cache(maxsize=1)
def get_shared_database(db_path_str: str) -> HPODatabase:
    """
    Get shared HPODatabase instance for reuse across requests.

    Thread-safe for concurrent reads (SQLite Row factory).
    Connection is cached and reused to avoid overhead.

    Args:
        db_path_str: String path to database (strings are hashable for LRU cache)

    Returns:
        Shared HPODatabase instance

    Note:
        Following existing codebase pattern (@lru_cache for singletons).
        Call .cache_clear() in tests to reset connection.

    Example:
        >>> db = get_shared_database("/path/to/hpo_data.db")
        >>> terms = db.get_terms_by_ids(["HP:0001250"])
        >>> # Next call reuses same connection
        >>> db2 = get_shared_database("/path/to/hpo_data.db")
        >>> assert db is db2  # Same instance
    """
    return HPODatabase(db_path_str)


def enrich_results_with_details(
    results: list[dict],
    data_dir_override: str | None = None,
) -> list[dict]:
    """
    Enrich HPO query results with definitions and synonyms.

    Returns NEW dictionaries with added detail fields. Input is NOT modified.
    Following pure function principles: no side effects.

    Args:
        results: List of result dicts containing at minimum:
                 - hpo_id: str
                 - label: str
                 Additional fields (similarity, scores) are preserved.
        data_dir_override: Optional data directory path override

    Returns:
        NEW list with enriched result dictionaries containing:
        - All original fields (preserved via spread operator)
        - definition: str | None (None if missing in database)
        - synonyms: list[str] | None (None if missing in database)

    Raises:
        sqlite3.Error: On database connection/query failures (fail fast)

    Performance:
        - Single database query for all terms (batch lookup)
        - O(n) where n = len(results)
        - Typical: < 50ms for 10-50 terms

    Error Handling:
        - Missing database: Handled gracefully, returns results with None details
        - Missing terms: Handled gracefully, adds None for those terms
        - Database errors: Propagate (fail fast, don't mask problems)

    Example:
        >>> results = [
        ...     {"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95}
        ... ]
        >>> enriched = enrich_results_with_details(results)
        >>> enriched[0]["definition"]  # New field added
        "A seizure is an intermittent abnormality..."
        >>> enriched[0]["similarity"]  # Original field preserved
        0.95
        >>> results[0].get("definition")  # Original NOT modified
        None

    Note:
        Uses get_shared_database() for connection reuse (performance).
        Follows DRY principle: reuses HPODatabase.get_terms_by_ids().
    """
    if not results:
        logger.debug("Empty results list, returning empty list")
        return []

    # Resolve database path
    data_dir = resolve_data_path(data_dir_override, "data_dir", get_default_data_dir)
    db_path = data_dir / DEFAULT_HPO_DB_FILENAME

    # Check database exists (expected error at startup - handle gracefully)
    if not db_path.exists():
        logger.warning(
            f"Database not found: {db_path}. "
            "Returning results without details. "
            "Run 'phentrieve data prepare' to generate database."
        )
        # Return NEW dicts with None details (preserve original)
        return [{**result, "definition": None, "synonyms": None} for result in results]

    # Fetch term details (let database errors propagate - unexpected failures)
    hpo_ids = [result["hpo_id"] for result in results]

    # Use shared connection (cached, thread-safe)
    db = get_shared_database(str(db_path))

    # No close() - connection is shared and reused
    # Let any database errors propagate (fail fast)
    terms_map = db.get_terms_by_ids(hpo_ids)

    # Create NEW enriched dictionaries (pure function - no mutation)
    enriched_results = []
    for result in results:
        hpo_id = result["hpo_id"]
        term_data = terms_map.get(hpo_id, {})

        # Extract details (convert empty strings to None for API clarity)
        definition = term_data.get("definition", "")
        synonyms = term_data.get("synonyms", [])

        # Create NEW dict with all original fields + details
        enriched = {
            **result,  # Spread all original fields
            "definition": definition if definition else None,
            "synonyms": synonyms if synonyms else None,
        }
        enriched_results.append(enriched)

    logger.debug(
        f"Enriched {len(enriched_results)} results with details "
        f"({len(terms_map)} found in database)"
    )

    return enriched_results
