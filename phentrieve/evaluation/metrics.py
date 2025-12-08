"""
Evaluation metrics for HPO retrieval assessment.

This module provides functions for evaluating the performance of HPO term retrieval
using various metrics including:
- Mean Reciprocal Rank (MRR)
- Hit Rate at K (HR@K)
- Ontology-based semantic similarity
"""

import logging
import math
import os
from enum import Enum
from functools import lru_cache
from typing import Any, Optional, Union

from phentrieve.config import DEFAULT_HPO_DB_FILENAME
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.utils import (
    calculate_similarity,
    get_default_data_dir,
)
from phentrieve.utils import (
    sanitize_log_value as _sanitize,
)


class SimilarityFormula(Enum):
    """
    Available semantic similarity formulas for ontology-based similarity calculations.

    SIMPLE_RESNIK_LIKE: A simplified version of Resnik similarity that
                       focuses purely on the depth of the lowest common ancestor.

    HYBRID: A hybrid formula that combines the depth of the lowest common ancestor
           with factors like the specificity of the terms and their relation to the LCA.
    """

    SIMPLE_RESNIK_LIKE = "simple_resnik_like"
    HYBRID = "hybrid"

    @classmethod
    def from_string(cls, formula_str: str) -> "SimilarityFormula":
        """Convert a string representation to SimilarityFormula enum value."""
        formula_str = formula_str.lower()
        if formula_str in ("simple", "simple_resnik", "simple_resnik_like"):
            return cls.SIMPLE_RESNIK_LIKE
        elif formula_str in ("hybrid", "complex"):
            return cls.HYBRID
        else:
            logging.warning(
                "Unknown similarity formula '%s', defaulting to HYBRID",
                _sanitize(formula_str),
            )
            return cls.HYBRID


def _resolve_hpo_db_path(db_path: str | None = None) -> str:
    """
    Resolve HPO database path to absolute path.

    This ensures consistent cache keys by normalizing None and relative paths
    to the same absolute path.

    Args:
        db_path: Path to HPO database file, or None for default

    Returns:
        Absolute path to HPO database file
    """
    if db_path is None:
        # Try direct data directory first
        if os.path.exists("data") and os.path.exists(f"data/{DEFAULT_HPO_DB_FILENAME}"):
            db_path = f"data/{DEFAULT_HPO_DB_FILENAME}"
        else:
            data_dir = get_default_data_dir()
            db_path = str(data_dir / DEFAULT_HPO_DB_FILENAME)

    # Convert to absolute path for consistent cache keys
    return os.path.abspath(db_path)


@lru_cache(maxsize=1)
def _load_hpo_graph_data_impl(
    db_path: str,
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """
    Internal cached implementation of HPO graph data loading.

    This function is cached with @lru_cache for thread-safe memoization.
    The cache key is based on the normalized absolute db_path.

    Args:
        db_path: Absolute path to HPO SQLite database file

    Returns:
        Tuple of (ancestors_dict, depths_dict)
        - ancestors_dict: {term_id: set of ancestor IDs}
        - depths_dict: {term_id: depth from root}

    Note:
        Returns empty dictionaries if database not found or loading fails.

        Thread-safe: This function uses @lru_cache's built-in locking mechanism.
        Multiple concurrent calls will wait for the first load to complete.
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logging.error("HPO database not found: %s", _sanitize(db_path))
            logging.error(
                "Please run 'phentrieve data prepare' to generate the database."
            )
            return {}, {}

        # Load graph data from database
        logging.info("Loading HPO graph data from database: %s...", _sanitize(db_path))
        db = HPODatabase(db_path)
        ancestors, depths = db.load_graph_data()
        db.close()

        logging.info("Loaded ancestor sets for %s HPO terms", len(ancestors))
        logging.info("Loaded depth values for %s HPO terms", len(depths))

        # Log sample data for debugging
        if ancestors:
            sample_terms = list(ancestors.keys())[:3]
            for term in sample_terms:
                term_ancestors = ancestors.get(term, set())
                logging.debug(
                    "Sample term %s has %s ancestors",
                    _sanitize(term),
                    len(term_ancestors),
                )

        # Log sample depth data
        if depths:
            sample_terms = list(depths.keys())[:5]
            logging.debug(
                "Sample depths: %s",
                [(_sanitize(t), depths[t]) for t in sample_terms],
            )

            # Statistics on depth distribution
            depth_values = list(depths.values())
            if depth_values:
                min_depth = min(depth_values)
                max_depth = max(depth_values)
                avg_depth = sum(depth_values) / len(depth_values)
                logging.debug(
                    "Depth statistics: min=%s, max=%s, avg=%.2f",
                    min_depth,
                    max_depth,
                    avg_depth,
                )

        return ancestors, depths

    except Exception as e:
        logging.error("Error loading HPO graph data: %s", _sanitize(e), exc_info=True)
        return {}, {}


def load_hpo_graph_data(
    db_path: str | None = None,
    ancestors_path: str | None = None,  # Deprecated, kept for compatibility
    depths_path: str | None = None,  # Deprecated, kept for compatibility
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """
    Load precomputed HPO graph data from SQLite database.

    This is a wrapper function that normalizes the db_path argument before
    calling the cached implementation. This ensures consistent cache behavior
    regardless of whether None or an explicit path is provided.

    Args:
        db_path: Path to the HPO SQLite database file (preferred)
        ancestors_path: (Deprecated) Path to ancestors pickle file - for backward compatibility
        depths_path: (Deprecated) Path to depths pickle file - for backward compatibility

    Returns:
        Tuple of (ancestors_dict, depths_dict)
        - ancestors_dict: {term_id: set of ancestor IDs}
        - depths_dict: {term_id: depth from root}

    Note:
        Returns empty dictionaries if database not found or loading fails.
        Results are cached via @lru_cache. To clear cache (e.g., after data updates),
        call: load_hpo_graph_data.cache_clear()

        Thread-safe: The underlying cached implementation uses @lru_cache which
        provides built-in thread-safety via locks.

    Examples:
        >>> ancestors, depths = load_hpo_graph_data()
        >>> len(ancestors)  # e.g., 19534
        >>> # Clear cache if needed
        >>> load_hpo_graph_data.cache_clear()
    """
    # Normalize path to ensure consistent cache keys
    normalized_path = _resolve_hpo_db_path(db_path)

    # Deprecated parameters are ignored (kept for API compatibility)
    if ancestors_path is not None or depths_path is not None:
        logging.warning(
            "ancestors_path and depths_path parameters are deprecated. "
            "HPO data is now loaded from SQLite database only."
        )

    # Call cached implementation with normalized path
    return _load_hpo_graph_data_impl(normalized_path)


# Expose lru_cache methods on public API wrapper
# This is the standard Python pattern for wrapper functions around @lru_cache decorated functions.
# The type: ignore comments are necessary because function objects don't have these attributes
# in their type stubs, but lru_cache adds them at runtime. This allows callers to access
# cache management methods (e.g., load_hpo_graph_data.cache_clear()) without needing to
# import the internal implementation function.
load_hpo_graph_data.cache_clear = _load_hpo_graph_data_impl.cache_clear  # type: ignore[attr-defined]
load_hpo_graph_data.cache_info = _load_hpo_graph_data_impl.cache_info  # type: ignore[attr-defined]


def find_lowest_common_ancestor(
    term1: str, term2: str, ancestors_dict: Optional[dict[str, set[str]]] = None
) -> tuple[Optional[str], int | float]:
    """
    Find the lowest common ancestor (LCA) of two HPO terms.

    Args:
        term1: First HPO term ID
        term2: Second HPO term ID
        ancestors_dict: Dict mapping terms to their ancestor sets

    Returns:
        LCA term ID and its depth, or (None, -1) if no common ancestor
    """
    if ancestors_dict is None:
        ancestors_dict, _ = load_hpo_graph_data()

    # Get ancestors for both terms
    ancestors1 = ancestors_dict.get(term1, set())
    ancestors2 = ancestors_dict.get(term2, set())

    # Find common ancestors
    common_ancestors = ancestors1.intersection(ancestors2)

    if not common_ancestors:
        return None, -1

    # Get depths for common ancestors
    _, depths_dict = load_hpo_graph_data()

    # Find the deepest common ancestor (LCA)
    lca = None
    max_depth: int | float = -1
    for ancestor in common_ancestors:
        depth = depths_dict.get(ancestor, float("inf"))
        if depth != float("inf") and depth > max_depth:
            max_depth = depth
            lca = ancestor

    return lca, max_depth


def calculate_resnik_similarity(term1: str, term2: str) -> float:
    """
    Calculate Resnik similarity between two HPO terms.
    Based on information content of the LCA.

    For simplicity, we use depth as a proxy for information content.

    Args:
        term1: First HPO term ID
        term2: Second HPO term ID

    Returns:
        Similarity score (0-1)
    """
    # Get the LCA and its depth
    lca, lca_depth = find_lowest_common_ancestor(term1, term2)

    if lca is None:
        logging.debug(
            "No LCA found between %s and %s", _sanitize(term1), _sanitize(term2)
        )
        return 0.0

    # Get depth of the two terms
    _, depths_dict = load_hpo_graph_data()

    depth1 = depths_dict.get(term1, float("inf"))
    depth2 = depths_dict.get(term2, float("inf"))

    if depth1 == float("inf") or depth2 == float("inf"):
        logging.debug(
            "Missing depth for %s=%s or %s=%s",
            _sanitize(term1),
            depth1,
            _sanitize(term2),
            depth2,
        )
        return 0.0

    # Calculate Resnik-like similarity based on LCA depth
    # Calculate path length from each term to LCA
    path_length1 = depth1 - lca_depth
    path_length2 = depth2 - lca_depth
    total_path_length = path_length1 + path_length2

    # Check if path lengths are valid
    if total_path_length < 0:
        logging.warning(
            "Invalid path lengths: term1=%s, term2=%s, LCA=%s",
            _sanitize(depth1),
            _sanitize(depth2),
            _sanitize(lca_depth),
        )
        total_path_length = 0

    # Calculate Resnik similarity based on path length and depth
    if total_path_length == 0:  # Identical or very closely related terms
        similarity = 1.0
    else:
        # Two factors contribute to similarity:
        # 1. How deep the LCA is (deeper = more specific = higher similarity)
        # 2. How far the terms are from their LCA (farther = lower similarity)

        # Scale LCA depth by maximum possible depth in the ontology
        _, depths = load_hpo_graph_data()
        max_possible_depth = (
            max(depths.values()) if depths else 20
        )  # Default if no data

        # LCA depth factor (higher depth = more similarity)
        depth_factor = lca_depth / max_possible_depth

        # Distance factor (shorter total path = more similarity)
        # Normalize by the sum of depths to get a relative distance
        max_possible_distance = depth1 + depth2 if (depth1 + depth2) > 0 else 1
        distance_factor = 1 - (total_path_length / max_possible_distance)

        # Combine factors with more weight on depth
        similarity = (0.7 * depth_factor) + (0.3 * distance_factor)

    logging.debug(
        "Similarity between %s and %s: %.4f (LCA=%s, LCA depth=%s, depths=%s,%s)",
        _sanitize(term1),
        _sanitize(term2),
        similarity,
        _sanitize(lca),
        lca_depth,
        depth1,
        depth2,
    )

    return max(0.0, min(1.0, similarity))  # Ensure result is 0-1


def calculate_simple_resnik_similarity(term1: str, term2: str) -> float:
    """
    Calculate a simplified Resnik-like similarity between two HPO terms.
    This formula focuses purely on the depth of the lowest common ancestor (LCA).

    Args:
        term1: First HPO term ID
        term2: Second HPO term ID

    Returns:
        Similarity score (0-1), where 1 is exact match and 0 is no similarity
    """
    # Get the LCA and its depth
    lca, lca_depth = find_lowest_common_ancestor(term1, term2)

    if lca is None:
        logging.debug(
            "No LCA found between %s and %s", _sanitize(term1), _sanitize(term2)
        )
        return 0.0

    # Get depth of the ontology (maximum possible depth)
    _, depths_dict = load_hpo_graph_data()
    max_possible_depth = max(depths_dict.values()) if depths_dict else 20

    # Calculate similarity solely based on LCA depth
    # Deeper LCA = higher similarity
    similarity = lca_depth / max_possible_depth

    logging.debug(
        "Simple similarity between %s and %s: %.4f (LCA=%s, LCA depth=%s)",
        _sanitize(term1),
        _sanitize(term2),
        similarity,
        _sanitize(lca),
        lca_depth,
    )

    return max(0.0, min(1.0, similarity))  # Ensure result is 0-1


def calculate_semantic_similarity(
    expected_term: str,
    retrieved_term: str,
    formula: SimilarityFormula = SimilarityFormula.HYBRID,
) -> float:
    """
    Calculate semantic similarity between expected term and retrieved term.

    Args:
        expected_term: Expected HPO term ID
        retrieved_term: Retrieved HPO term ID
        formula: Which similarity formula to use (default: HYBRID)

    Returns:
        Similarity score (0-1), where 1 is exact match and 0 is no similarity
    """
    # Load HPO graph data if not already loaded
    ancestors_dict, depths_dict = load_hpo_graph_data()

    # Check if we have valid data for both terms
    if expected_term not in ancestors_dict or retrieved_term not in ancestors_dict:
        logging.debug(
            "Missing term data: %s or %s not in ancestors",
            _sanitize(expected_term),
            _sanitize(retrieved_term),
        )
        # If terms are identical, return 1.0 even if missing from ancestors dict
        if expected_term == retrieved_term:
            return 1.0
        return 0.0

    # If terms are identical, return 1.0
    if expected_term == retrieved_term:
        logging.debug(
            "Exact match between %s and %s",
            _sanitize(expected_term),
            _sanitize(retrieved_term),
        )
        return 1.0

    # Choose formula based on parameter
    if formula == SimilarityFormula.SIMPLE_RESNIK_LIKE:
        # Use simple depth-based similarity
        return calculate_simple_resnik_similarity(expected_term, retrieved_term)
    else:  # Default to HYBRID formula
        # Use standard Resnik similarity with additional factors
        return calculate_resnik_similarity(expected_term, retrieved_term)


def calculate_max_similarity(
    expected_terms: list[str],
    retrieved_terms: Union[list[str], list[dict[str, Any]]],
    top_k: Optional[int] = None,
    formula: SimilarityFormula = SimilarityFormula.HYBRID,
) -> list[float]:
    """
    Calculate maximum similarity between each expected term and any retrieved term.

    Args:
        expected_terms: List of expected HPO term IDs
        retrieved_terms: List of retrieved HPO term IDs (or list of dicts with 'hpo_id' key)
        top_k: Only consider top K retrieved terms, if specified
        formula: Which similarity formula to use (default: HYBRID)

    Returns:
        List of max similarities, one for each expected term
    """
    # Ensure we have the graph data loaded
    load_hpo_graph_data()

    # Extract term IDs if retrieved_terms contains dictionaries
    retrieved_ids: list[str | Any]
    if retrieved_terms and isinstance(retrieved_terms[0], dict):
        retrieved_ids = [
            item["hpo_id"] for item in retrieved_terms[:top_k] if isinstance(item, dict)
        ]
    else:
        # Cast to list[str | Any] to match the declared type
        sliced_terms = retrieved_terms[:top_k] if top_k else retrieved_terms
        retrieved_ids = list(sliced_terms)

    max_similarities = []

    for expected_term in expected_terms:
        if not retrieved_ids:
            max_similarities.append(0.0)
            continue

        # Calculate similarity with each retrieved term and find max
        max_sim = 0.0
        for i, retrieved_item in enumerate(retrieved_ids):
            # Skip if we've reached top_k
            if top_k is not None and i >= top_k:
                break

            # Get term ID from retrieved item (can be string or dict)
            if isinstance(retrieved_item, dict):
                retrieved_term = retrieved_item.get("hpo_id", "")
            else:
                retrieved_term = retrieved_item

            if not retrieved_term:
                continue

            # Calculate similarity between expected and retrieved term using specified formula
            sim = calculate_semantic_similarity(expected_term, retrieved_term, formula)
            max_sim = max(max_sim, sim)

        max_similarities.append(max_sim)

    return max_similarities


def average_max_similarity(
    expected_terms: list[str],
    retrieved_terms: Union[list[str], list[dict[str, Any]]],
    top_k: Optional[int] = None,
    formula: SimilarityFormula = SimilarityFormula.HYBRID,
) -> float:
    """
    Calculate average maximum similarity between expected terms and top retrieved terms.

    This metric gives a score showing how similar, on average, the retrieved terms
    are to the expected terms, based on ontology structure.

    Args:
        expected_terms: List of expected HPO term IDs
        retrieved_terms: List of retrieved HPO term IDs (or list of dicts with 'hpo_id' key)
        top_k: Only consider top K retrieved terms, if specified
        formula: Which similarity formula to use (default: HYBRID)

    Returns:
        Average maximum similarity score (0-1)
    """
    # Get max similarities for each expected term
    max_similarities = calculate_max_similarity(
        expected_terms, retrieved_terms, top_k, formula
    )

    return sum(max_similarities) / len(max_similarities)


def calculate_test_case_max_ont_sim(
    expected_ids: list[str],
    retrieved_ids: list[str],
    formula: SimilarityFormula = SimilarityFormula.HYBRID,
) -> float:
    """
    Calculates the single highest semantic similarity between any expected ID
    and any retrieved ID for a single test case. Returns 1.0 if any exact match exists.

    Args:
        expected_ids: List of ground truth HPO IDs for the test case.
        retrieved_ids: List of retrieved HPO IDs (typically top K).

    Returns:
        The maximum similarity score (float between 0.0 and 1.0) found.
    """
    if not expected_ids or not retrieved_ids:
        return 0.0

    overall_max_sim = 0.0

    # Check for exact match first across all pairs
    # Using sets for efficient intersection check
    expected_set = set(expected_ids)
    retrieved_set = set(retrieved_ids)
    if not expected_set.isdisjoint(retrieved_set):
        # If any exact match exists, the max possible similarity is 1.0
        logging.debug(
            "Exact match found for test case (%s), MaxOntSim is 1.0",
            _sanitize(expected_set.intersection(retrieved_set)),
        )
        return 1.0

    # If no exact match, calculate all pairwise semantic similarities
    # This requires the graph data to be loaded, assume it's handled/cached by calculate_semantic_similarity
    try:
        load_hpo_graph_data()  # Ensure data is loaded/cached
        for exp_id in expected_ids:
            for ret_id in retrieved_ids:
                # calculate_semantic_similarity handles the 1.0 check internally too, but we did it above for clarity.
                similarity = calculate_semantic_similarity(exp_id, ret_id, formula)
                if similarity > overall_max_sim:
                    overall_max_sim = similarity
                    # Optimization: if we reach 1.0, we can stop
                    if overall_max_sim >= 1.0:
                        break  # Exit inner loop
            if overall_max_sim >= 1.0:
                break  # Exit outer loop
    except Exception as e:
        logging.error(
            "Error during pairwise similarity calculation for MaxOntSim: %s",
            _sanitize(e),
            exc_info=True,
        )
        # Depending on desired behavior, could return 0.0 or raise
        return 0.0  # Default to 0 on error during calculation

    logging.debug("Calculated MaxOntSim for test case: %.4f", overall_max_sim)
    # Clamp result just in case, though calculate_semantic_similarity should handle it
    return max(0.0, min(1.0, overall_max_sim))


def mean_reciprocal_rank(results: dict[str, Any], expected_ids: list[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        results: Results from dense retriever query
        expected_ids: List of expected HPO IDs

    Returns:
        float: MRR value (0 if no matches)
    """
    if (
        not results
        or not results.get("ids")
        or not results["ids"][0]
        or not expected_ids
    ):
        return 0.0

    # Get all retrieved HPO IDs with their ranks and similarity scores
    retrieved_ids = []
    for i, (hpo_id, metadata, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        # Add HPO ID with its actual rank (1-based) and similarity score
        similarity = calculate_similarity(distance)
        retrieved_ids.append((metadata["hpo_id"], i + 1, similarity))

    # Sort by similarity score (descending)
    retrieved_ids.sort(key=lambda x: x[2], reverse=True)

    # Re-rank based on similarity
    ranked_ids = [
        (hpo_id, i + 1, sim) for i, (hpo_id, _, sim) in enumerate(retrieved_ids)
    ]

    # Find the first match
    for hpo_id, rank, _ in ranked_ids:
        if hpo_id in expected_ids:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(
    results: dict[str, Any], expected_ids: list[str], k: int = 5
) -> float:
    """
    Calculate Hit Rate at K.

    Args:
        results: Results from dense retriever query
        expected_ids: List of expected HPO IDs
        k: Number of top results to consider

    Returns:
        float: 1.0 if any expected ID is in top K, 0.0 otherwise
    """
    if (
        not results
        or not results.get("ids")
        or not results["ids"][0]
        or not expected_ids
    ):
        return 0.0

    # Get all retrieved HPO IDs with similarity scores
    retrieved_ids = []
    for _i, (hpo_id, metadata, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        similarity = calculate_similarity(distance)
        retrieved_ids.append((metadata["hpo_id"], similarity))

    # Sort by similarity (descending)
    retrieved_ids.sort(key=lambda x: x[1], reverse=True)

    # Take top k
    top_k_ids = retrieved_ids[:k]

    # Check if any expected ID is in top K
    for hpo_id, _ in top_k_ids:
        if hpo_id in expected_ids:
            return 1.0

    return 0.0


def ndcg_at_k(
    results: dict[str, Any],
    expected_ids: list[str],
    k: int = 10,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        results: Retrieval results from dense retriever
        expected_ids: List of relevant HPO IDs
        k: Number of results to consider
        relevance_scores: Optional dict mapping HPO ID to relevance score (default: binary)

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if not results or not results.get("metadatas") or not expected_ids:
        return 0.0

    # Extract retrieved HPO IDs with similarity scores
    retrieved_ids = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        hpo_id = metadata.get("hpo_id", "")
        similarity = calculate_similarity(distance)
        retrieved_ids.append((hpo_id, similarity))

    # Sort by similarity (descending) to get ranking
    retrieved_ids.sort(key=lambda x: x[1], reverse=True)

    # Calculate DCG
    dcg = 0.0
    for i, (hpo_id, _) in enumerate(retrieved_ids[:k]):
        if hpo_id in expected_ids:
            # Binary relevance (1 if relevant, 0 otherwise)
            # Or use graded relevance if provided
            rel = relevance_scores.get(hpo_id, 1.0) if relevance_scores else 1.0
            # Discount factor: 1 / log2(rank + 1)
            dcg += rel / math.log2(i + 2)  # +2 because rank is 1-indexed

    # Calculate ideal DCG (all relevant docs at top ranks)
    ideal_rels = sorted(
        [
            relevance_scores.get(hpo_id, 1.0) if relevance_scores else 1.0
            for hpo_id in expected_ids
        ],
        reverse=True,
    )[:k]

    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def recall_at_k(
    results: dict[str, Any],
    expected_ids: list[str],
    k: int = 10,
) -> float:
    """
    Calculate Recall at K.

    Args:
        results: Retrieval results from dense retriever
        expected_ids: List of relevant HPO IDs
        k: Number of results to consider

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not results or not results.get("metadatas") or not expected_ids:
        return 0.0

    # Extract retrieved HPO IDs with similarity scores
    retrieved_ids = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        hpo_id = metadata.get("hpo_id", "")
        similarity = calculate_similarity(distance)
        retrieved_ids.append((hpo_id, similarity))

    # Sort by similarity (descending)
    retrieved_ids.sort(key=lambda x: x[1], reverse=True)

    # Take top k
    top_k_ids = [hpo_id for hpo_id, _ in retrieved_ids[:k]]

    # Count relevant items found
    relevant_found = len(set(top_k_ids).intersection(set(expected_ids)))

    return relevant_found / len(expected_ids)


def precision_at_k(
    results: dict[str, Any],
    expected_ids: list[str],
    k: int = 10,
) -> float:
    """
    Calculate Precision at K.

    Args:
        results: Retrieval results from dense retriever
        expected_ids: List of relevant HPO IDs
        k: Number of results to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if not results or not results.get("metadatas"):
        return 0.0

    # Extract retrieved HPO IDs with similarity scores
    retrieved_ids = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        hpo_id = metadata.get("hpo_id", "")
        similarity = calculate_similarity(distance)
        retrieved_ids.append((hpo_id, similarity))

    # Sort by similarity (descending)
    retrieved_ids.sort(key=lambda x: x[1], reverse=True)

    # Take top k
    top_k_ids = retrieved_ids[:k]

    if not top_k_ids:
        return 0.0

    # Count relevant items in top K
    relevant_in_top_k = sum(1 for hpo_id, _ in top_k_ids if hpo_id in expected_ids)

    return relevant_in_top_k / len(top_k_ids)


def average_precision_at_k(
    results: dict[str, Any],
    expected_ids: list[str],
    k: int = 10,
) -> float:
    """
    Calculate Average Precision at K for a single query.

    Args:
        results: Retrieval results from dense retriever
        expected_ids: List of relevant HPO IDs
        k: Number of results to consider

    Returns:
        AP@K score (0.0 to 1.0)
    """
    if not results or not results.get("metadatas") or not expected_ids:
        return 0.0

    # Extract retrieved HPO IDs with similarity scores
    retrieved_ids = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        hpo_id = metadata.get("hpo_id", "")
        similarity = calculate_similarity(distance)
        retrieved_ids.append((hpo_id, similarity))

    # Sort by similarity (descending)
    retrieved_ids.sort(key=lambda x: x[1], reverse=True)

    expected_set = set(expected_ids)
    relevant_count = 0
    precision_sum = 0.0

    for i, (hpo_id, _) in enumerate(retrieved_ids[:k]):
        if hpo_id in expected_set:
            relevant_count += 1
            # Precision at this rank
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    if relevant_count == 0:
        return 0.0

    # Normalize by number of relevant docs (up to k)
    num_relevant = min(len(expected_ids), k)
    return precision_sum / num_relevant
