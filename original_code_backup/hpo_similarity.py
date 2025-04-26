#!/usr/bin/env python3
"""
HPO Ontology Similarity Metrics

This module provides functions for calculating semantic similarity between HPO terms
using the ontology structure. It relies on precomputed HPO graph properties.
"""

import os
import pickle
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths for precomputed data
DEFAULT_ANCESTORS_PATH = os.path.join("data", "hpo_ancestors.pkl")
DEFAULT_DEPTHS_PATH = os.path.join("data", "hpo_term_depths.pkl")

# Global caches
_hpo_ancestors = None
_hpo_term_depths = None


def load_hpo_graph_data(
    ancestors_path=DEFAULT_ANCESTORS_PATH, depths_path=DEFAULT_DEPTHS_PATH
):
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
        logger.info(f"Loaded ancestor sets for {len(_hpo_ancestors)} HPO terms")

        # Load term depths
        with open(depths_path, "rb") as f:
            _hpo_term_depths = pickle.load(f)
        logger.info(f"Loaded depth values for {len(_hpo_term_depths)} HPO terms")

        return _hpo_ancestors, _hpo_term_depths

    except Exception as e:
        logger.error(f"Error loading HPO graph data: {e}")
        return {}, {}


def find_lowest_common_ancestor(term1, term2, ancestors_dict=None):
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


def calculate_resnik_similarity(term1, term2):
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


def calculate_semantic_similarity(expected_term, retrieved_term):
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


def calculate_max_similarity(expected_terms, retrieved_terms, top_k=None):
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


def average_max_similarity(expected_terms, retrieved_terms, top_k=None):
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


# Function to test the module
def test_similarity():
    """
    Test the similarity calculations with some example HPO terms.
    """
    # Load graph data
    ancestors_dict, depths_dict = load_hpo_graph_data()

    # Example HPO terms
    terms = [
        "HP:0000118",  # Phenotypic abnormality (root)
        "HP:0000707",  # Abnormality of the nervous system
        "HP:0001250",  # Seizure
        "HP:0000924",  # Abnormality of the skeletal system
    ]

    # Print depth information
    print("Term depths:")
    for term in terms:
        depth = depths_dict.get(term, "Unknown")
        print(f"  {term}: {depth}")

    # Calculate pairwise similarities
    print("\nPairwise similarities:")
    for i, term1 in enumerate(terms):
        for term2 in terms[i:]:
            sim = calculate_semantic_similarity(term1, term2)
            print(f"  {term1} - {term2}: {sim:.4f}")

    # Test average_max_similarity
    expected = [
        "HP:0001250",
        "HP:0000924",
    ]  # Seizure, Abnormality of the skeletal system
    retrieved = [
        {"hpo_id": "HP:0001250"},  # Seizure (exact match)
        {
            "hpo_id": "HP:0000707"
        },  # Abnormality of the nervous system (parent of Seizure)
        {"hpo_id": "HP:0000118"},  # Phenotypic abnormality (root)
    ]

    avg_sim = average_max_similarity(expected, retrieved)
    print(f"\nAverage Max Similarity: {avg_sim:.4f}")

    # Test with top-k
    for k in [1, 2, 3]:
        avg_sim_k = average_max_similarity(expected, retrieved, top_k=k)
        print(f"Average Max Similarity @{k}: {avg_sim_k:.4f}")


if __name__ == "__main__":
    test_similarity()
