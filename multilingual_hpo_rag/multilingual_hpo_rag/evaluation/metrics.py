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
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from multilingual_hpo_rag.config import HPO_ANCESTORS_FILE, HPO_DEPTHS_FILE, HPO_ROOT_ID
from multilingual_hpo_rag.retrieval.dense_retriever import calculate_similarity


# Global caches for HPO graph data
_hpo_ancestors = None
_hpo_term_depths = None


def load_hpo_graph_data(
    ancestors_path: str = HPO_ANCESTORS_FILE, depths_path: str = HPO_DEPTHS_FILE
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
        return _hpo_ancestors, _hpo_term_depths

    try:
        # Load ancestor sets
        with open(ancestors_path, "rb") as f:
            _hpo_ancestors = pickle.load(f)
        logging.info(f"Loaded ancestor sets for {len(_hpo_ancestors)} HPO terms")

        # Load term depths
        with open(depths_path, "rb") as f:
            _hpo_term_depths = pickle.load(f)
        logging.info(f"Loaded depth values for {len(_hpo_term_depths)} HPO terms")

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
        return 0.0

    # Get depth of the two terms
    _, depths_dict = load_hpo_graph_data()

    depth1 = depths_dict.get(term1, float("inf"))
    depth2 = depths_dict.get(term2, float("inf"))

    if depth1 == float("inf") or depth2 == float("inf"):
        return 0.0

    # Calculate Resnik-like similarity based on LCA depth
    # Normalize by max depth of the two terms to get a score between 0 and 1
    max_depth = max(depth1, depth2)

    if max_depth == 0:
        return 0.0

    # Normalize by max possible depth to get a value between 0 and 1
    return lca_depth / max_depth


def calculate_semantic_similarity(expected_term: str, retrieved_term: str) -> float:
    """
    Calculate semantic similarity between expected term and retrieved term.

    Args:
        expected_term: Expected HPO term ID
        retrieved_term: Retrieved HPO term ID

    Returns:
        Similarity score (0-1), where 1 is exact match and 0 is no similarity
    """
    # Load HPO graph data if not already loaded
    load_hpo_graph_data()

    # If terms are identical, return 1.0
    if expected_term == retrieved_term:
        return 1.0

    # Calculate Resnik similarity
    return calculate_resnik_similarity(expected_term, retrieved_term)


def calculate_max_similarity(
    expected_terms: List[str],
    retrieved_terms: Union[List[str], List[Dict[str, Any]]],
    top_k: Optional[int] = None,
) -> List[float]:
    """
    Calculate maximum similarity between each expected term and any retrieved term.

    Args:
        expected_terms: List of expected HPO term IDs
        retrieved_terms: List of retrieved HPO term IDs (or list of dicts with 'hpo_id' key)
        top_k: Only consider top K retrieved terms, if specified

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

        # Calculate similarity to each retrieved term and take the max
        similarities = [
            calculate_semantic_similarity(expected_term, retrieved_term)
            for retrieved_term in retrieved_ids
        ]

        max_similarities.append(max(similarities) if similarities else 0.0)

    return max_similarities


def average_max_similarity(
    expected_terms: List[str],
    retrieved_terms: Union[List[str], List[Dict[str, Any]]],
    top_k: Optional[int] = None,
) -> float:
    """
    Calculate average maximum similarity between expected terms and top retrieved terms.

    This metric gives a score showing how similar, on average, the retrieved terms
    are to the expected terms, based on ontology structure.

    Args:
        expected_terms: List of expected HPO term IDs
        retrieved_terms: List of retrieved HPO term IDs (or list of dicts with 'hpo_id' key)
        top_k: Only consider top K retrieved terms, if specified

    Returns:
        Average maximum similarity score (0-1)
    """
    if not expected_terms:
        return 0.0

    max_similarities = calculate_max_similarity(expected_terms, retrieved_terms, top_k)

    return sum(max_similarities) / len(max_similarities)


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
