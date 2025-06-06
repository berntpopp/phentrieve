"""
Evaluation metrics for HPO retrieval assessment.

This module provides functions for evaluating the performance of HPO term retrieval
using various metrics including:
- Mean Reciprocal Rank (MRR)
- Hit Rate at K (HR@K)
- Ontology-based semantic similarity
"""

import logging
import os
import pickle
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from phentrieve.config import DEFAULT_ANCESTORS_FILENAME, DEFAULT_DEPTHS_FILENAME
from phentrieve.utils import get_default_data_dir
from phentrieve.utils import calculate_similarity


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
                f"Unknown similarity formula '{formula_str}', defaulting to HYBRID"
            )
            return cls.HYBRID


# Global caches for HPO graph data
_hpo_ancestors = None
_hpo_term_depths = None


def load_hpo_graph_data(
    ancestors_path: str = None, depths_path: str = None
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """
    Load precomputed HPO graph data from pickle files.

    Args:
        ancestors_path: Path to the ancestors pickle file
        depths_path: Path to the term depths pickle file

    Returns:
        Tuple of (ancestors_dict, depths_dict)
    """
    global _hpo_ancestors, _hpo_term_depths

    # Return cached data if already loaded
    if _hpo_ancestors is not None and _hpo_term_depths is not None:
        logging.debug("Using cached HPO graph data")
        return _hpo_ancestors, _hpo_term_depths

    # Resolve paths if not provided
    if ancestors_path is None:
        # Try direct data directory first
        if os.path.exists("data") and os.path.exists(
            f"data/{DEFAULT_ANCESTORS_FILENAME}"
        ):
            ancestors_path = f"data/{DEFAULT_ANCESTORS_FILENAME}"
        else:
            data_dir = get_default_data_dir()
            ancestors_path = data_dir / DEFAULT_ANCESTORS_FILENAME

    if depths_path is None:
        # Try direct data directory first
        if os.path.exists("data") and os.path.exists(f"data/{DEFAULT_DEPTHS_FILENAME}"):
            depths_path = f"data/{DEFAULT_DEPTHS_FILENAME}"
        else:
            data_dir = get_default_data_dir()
            depths_path = data_dir / DEFAULT_DEPTHS_FILENAME

    try:
        # Check if files exist
        if not os.path.exists(ancestors_path):
            logging.error(f"Ancestors file not found: {ancestors_path}")
            return {}, {}
        if not os.path.exists(depths_path):
            logging.error(f"Depths file not found: {depths_path}")
            return {}, {}

        # Load ancestor sets
        with open(ancestors_path, "rb") as f:
            _hpo_ancestors = pickle.load(f)
        logging.info(f"Loaded ancestor sets for {len(_hpo_ancestors)} HPO terms")

        # Log sample data for debugging
        if _hpo_ancestors:
            sample_terms = list(_hpo_ancestors.keys())[:3]
            for term in sample_terms:
                ancestors = _hpo_ancestors.get(term, set())
                logging.debug(f"Sample term {term} has {len(ancestors)} ancestors")

        # Load term depths
        with open(depths_path, "rb") as f:
            _hpo_term_depths = pickle.load(f)
        logging.info(f"Loaded depth values for {len(_hpo_term_depths)} HPO terms")

        # Log sample depth data
        if _hpo_term_depths:
            sample_terms = list(_hpo_term_depths.keys())[:5]
            logging.debug(
                f"Sample depths: {[(t, _hpo_term_depths[t]) for t in sample_terms]}"
            )

            # Statistics on depth distribution
            depths = list(_hpo_term_depths.values())
            if depths:
                min_depth = min(depths)
                max_depth = max(depths)
                avg_depth = sum(depths) / len(depths)
                logging.debug(
                    f"Depth statistics: min={min_depth}, max={max_depth}, avg={avg_depth:.2f}"
                )

        return _hpo_ancestors, _hpo_term_depths

    except Exception as e:
        logging.error(f"Error loading HPO graph data: {e}")
        return {}, {}


def find_lowest_common_ancestor(
    term1: str, term2: str, ancestors_dict: Optional[Dict[str, Set[str]]] = None
) -> Tuple[Optional[str], int]:
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
    max_depth = -1
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
        logging.debug(f"No LCA found between {term1} and {term2}")
        return 0.0

    # Get depth of the two terms
    _, depths_dict = load_hpo_graph_data()

    depth1 = depths_dict.get(term1, float("inf"))
    depth2 = depths_dict.get(term2, float("inf"))

    if depth1 == float("inf") or depth2 == float("inf"):
        logging.debug(f"Missing depth for {term1}={depth1} or {term2}={depth2}")
        return 0.0

    # Calculate Resnik-like similarity based on LCA depth
    # Calculate path length from each term to LCA
    path_length1 = depth1 - lca_depth
    path_length2 = depth2 - lca_depth
    total_path_length = path_length1 + path_length2

    # Check if path lengths are valid
    if total_path_length < 0:
        logging.warning(
            f"Invalid path lengths: term1={depth1}, term2={depth2}, LCA={lca_depth}"
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
        f"Similarity between {term1} and {term2}: {similarity:.4f} "
        f"(LCA={lca}, LCA depth={lca_depth}, depths={depth1},{depth2})"
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
        logging.debug(f"No LCA found between {term1} and {term2}")
        return 0.0

    # Get depth of the ontology (maximum possible depth)
    _, depths_dict = load_hpo_graph_data()
    max_possible_depth = max(depths_dict.values()) if depths_dict else 20

    # Calculate similarity solely based on LCA depth
    # Deeper LCA = higher similarity
    similarity = lca_depth / max_possible_depth

    logging.debug(
        f"Simple similarity between {term1} and {term2}: {similarity:.4f} "
        f"(LCA={lca}, LCA depth={lca_depth})"
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
            f"Missing term data: {expected_term} or {retrieved_term} not in ancestors"
        )
        # If terms are identical, return 1.0 even if missing from ancestors dict
        if expected_term == retrieved_term:
            return 1.0
        return 0.0

    # If terms are identical, return 1.0
    if expected_term == retrieved_term:
        logging.debug(f"Exact match between {expected_term} and {retrieved_term}")
        return 1.0

    # Choose formula based on parameter
    if formula == SimilarityFormula.SIMPLE_RESNIK_LIKE:
        # Use simple depth-based similarity
        return calculate_simple_resnik_similarity(expected_term, retrieved_term)
    else:  # Default to HYBRID formula
        # Use standard Resnik similarity with additional factors
        return calculate_resnik_similarity(expected_term, retrieved_term)


def calculate_max_similarity(
    expected_terms: List[str],
    retrieved_terms: Union[List[str], List[Dict[str, Any]]],
    top_k: Optional[int] = None,
    formula: SimilarityFormula = SimilarityFormula.HYBRID,
) -> List[float]:
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
    if retrieved_terms and isinstance(retrieved_terms[0], dict):
        retrieved_ids = [item["hpo_id"] for item in retrieved_terms[:top_k]]
    else:
        retrieved_ids = retrieved_terms[:top_k] if top_k else retrieved_terms

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
    expected_terms: List[str],
    retrieved_terms: Union[List[str], List[Dict[str, Any]]],
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
    expected_ids: List[str],
    retrieved_ids: List[str],
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
            f"Exact match found for test case ({expected_set.intersection(retrieved_set)}), MaxOntSim is 1.0"
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
            f"Error during pairwise similarity calculation for MaxOntSim: {e}",
            exc_info=True,
        )
        # Depending on desired behavior, could return 0.0 or raise
        return 0.0  # Default to 0 on error during calculation

    logging.debug(f"Calculated MaxOntSim for test case: {overall_max_sim:.4f}")
    # Clamp result just in case, though calculate_semantic_similarity should handle it
    return max(0.0, min(1.0, overall_max_sim))


def mean_reciprocal_rank(results: Dict[str, Any], expected_ids: List[str]) -> float:
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
    results: Dict[str, Any], expected_ids: List[str], k: int = 5
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
    for i, (hpo_id, metadata, distance) in enumerate(
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
