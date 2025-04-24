#!/usr/bin/env python3
"""
Precompute HPO Graph Properties

This script parses the Human Phenotype Ontology (HPO) structure from hp.json
to build a graph representation and precompute important properties:
- Ancestor sets for each term (for fast subsumption checking)
- Term depths (for similarity calculation)

These precomputed properties are essential for efficient semantic
similarity calculations during benchmarking.
"""

import os
import json
import pickle
import argparse
import logging
import sys
from collections import defaultdict, deque

# Constants
PHENOTYPE_ROOT = "HP:0000118"  # Phenotypic abnormality - root of the phenotype tree
DEFAULT_HPO_JSON_PATH = os.path.join("data", "hp.json")
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_ANCESTORS_FILE = "hpo_ancestors.pkl"
DEFAULT_DEPTHS_FILE = "hpo_term_depths.pkl"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def normalize_id(hpo_id):
    """Normalize HPO ID format to handle variations (including URIs)."""
    if not hpo_id:
        return None

    # Handle URI format: http://purl.obolibrary.org/obo/HP_0000123
    if hpo_id.startswith("http://purl.obolibrary.org/obo/HP_"):
        id_part = hpo_id.split("/")[-1].replace("HP_", "")
        return f"HP:{id_part}"

    # Standard HP:0000123 format
    elif hpo_id.startswith("HP:"):
        return hpo_id

    # HP_0000123 format
    elif hpo_id.startswith("HP_"):
        return "HP:" + hpo_id[3:]

    # Handle numeric strings
    elif hpo_id.isdigit() or (hpo_id.startswith("0") and hpo_id.isdigit()):
        # Format as 7-digit ID with HP: prefix
        return f"HP:{int(hpo_id):07d}"

    return hpo_id


def load_and_build_graphs(hpo_json_path):
    """
    Load the HPO JSON file and build directed graphs.

    Returns:
        Tuple containing:
        - child_to_parents: Dict mapping child term to list of parent terms
        - parent_to_children: Dict mapping parent term to list of child terms
        - all_hpo_term_ids: Set of all HPO term IDs
    """
    logger.info(f"Loading HPO ontology from {hpo_json_path}")
    try:
        with open(hpo_json_path, "r", encoding="utf-8") as f:
            hpo_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load HPO JSON file: {e}")
        return None, None, None

    # Extract nodes and edges
    all_hpo_term_ids = set()
    child_to_parents = defaultdict(list)
    parent_to_children = defaultdict(list)

    # Process nodes first to get all terms
    logger.info("Processing HPO nodes")
    nodes_processed = 0
    for node in hpo_data.get("graphs", [{}])[0].get("nodes", []):
        term_id = node.get("id")
        if term_id:
            try:
                normalized_id = normalize_id(term_id)
                if normalized_id and normalized_id.startswith("HP:"):
                    all_hpo_term_ids.add(normalized_id)
                    nodes_processed += 1
            except Exception as e:
                logger.warning(f"Error normalizing ID {term_id}: {e}")

    logger.info(f"Processed {nodes_processed} HPO nodes")

    # Process edges to build parent-child relationships
    logger.info("Processing HPO edges (is_a relationships)")
    edges_processed = 0
    for edge in hpo_data.get("graphs", [{}])[0].get("edges", []):
        pred = edge.get("pred")
        if pred == "is_a":
            try:
                sub = normalize_id(edge.get("sub"))
                obj = normalize_id(edge.get("obj"))

                if sub and obj and sub.startswith("HP:") and obj.startswith("HP:"):
                    # sub is_a obj => obj is parent, sub is child
                    child_to_parents[sub].append(obj)
                    parent_to_children[obj].append(sub)
                    edges_processed += 1
            except Exception as e:
                logger.warning(f"Error processing edge: {e}")

    logger.info(f"Processed {edges_processed} is_a relationships")

    # Add root if missing
    if PHENOTYPE_ROOT not in all_hpo_term_ids:
        logger.warning(
            f"Root node {PHENOTYPE_ROOT} not found in HPO data, adding it manually"
        )
        all_hpo_term_ids.add(PHENOTYPE_ROOT)

    logger.info(
        f"Built HPO graph with {len(all_hpo_term_ids)} terms and {sum(len(parents) for parents in child_to_parents.values())} is_a relationships"
    )

    return child_to_parents, parent_to_children, all_hpo_term_ids


def calculate_depths(parent_to_children, all_term_ids, root_id=PHENOTYPE_ROOT):
    """
    Calculate shortest path depths for all HPO terms using BFS from the root.

    Args:
        parent_to_children: Dict mapping parent term to list of child terms
        all_term_ids: Set of all HPO term IDs
        root_id: Root node to start BFS from

    Returns:
        Dict mapping HPO term ID to its depth from root
    """
    logger.info(f"Calculating term depths from root {root_id}")

    depths = {}
    for term_id in all_term_ids:
        depths[term_id] = float("inf")  # Initialize all depths to infinity

    # BFS from root
    queue = deque([(root_id, 0)])  # (term_id, depth)
    visited = {root_id}

    while queue:
        current_id, current_depth = queue.popleft()
        depths[current_id] = current_depth

        for child_id in parent_to_children.get(current_id, []):
            if child_id not in visited:
                visited.add(child_id)
                queue.append((child_id, current_depth + 1))

    # Count unreachable nodes
    unreachable_count = sum(1 for depth in depths.values() if depth == float("inf"))
    if unreachable_count > 0:
        logger.warning(f"{unreachable_count} terms are unreachable from root {root_id}")

    logger.info(f"Calculated depths for {len(depths) - unreachable_count} HPO terms")
    return depths


def calculate_all_ancestors(child_to_parents, all_term_ids):
    """
    Calculate all ancestors (including self) for each HPO term.

    Args:
        child_to_parents: Dict mapping child term to list of parent terms
        all_term_ids: Set of all HPO term IDs

    Returns:
        Dict mapping HPO term ID to set of all ancestor terms (including self)
    """
    logger.info("Calculating ancestor sets for all HPO terms")

    ancestors_map = {}

    for term_id in all_term_ids:
        # Include self in ancestors
        ancestors = {term_id}

        # Start BFS from parents
        queue = deque(child_to_parents.get(term_id, []))
        while queue:
            parent_id = queue.popleft()
            if parent_id not in ancestors:
                ancestors.add(parent_id)
                # Add parent's parents to queue
                queue.extend(
                    [
                        p
                        for p in child_to_parents.get(parent_id, [])
                        if p not in ancestors
                    ]
                )

        ancestors_map[term_id] = ancestors

    avg_ancestor_count = (
        sum(len(ancestors) for ancestors in ancestors_map.values()) / len(ancestors_map)
        if ancestors_map
        else 0
    )
    logger.info(
        f"Calculated ancestor sets for {len(ancestors_map)} HPO terms (avg {avg_ancestor_count:.2f} ancestors per term)"
    )

    return ancestors_map


def run_precomputation(hpo_json_path, output_dir, ancestors_file, depths_file):
    """
    Run the full precomputation process.

    Args:
        hpo_json_path: Path to hp.json
        output_dir: Directory to save output files
        ancestors_file: Filename for ancestors pickle
        depths_file: Filename for depths pickle

    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build HPO graph
    child_to_parents, parent_to_children, all_term_ids = load_and_build_graphs(
        hpo_json_path
    )
    if not all((child_to_parents, parent_to_children, all_term_ids)):
        logger.error("Failed to build HPO graph")
        return False

    # Calculate depths
    depths_map = calculate_depths(parent_to_children, all_term_ids)

    # Calculate ancestor sets
    ancestors_map = calculate_all_ancestors(child_to_parents, all_term_ids)

    # Save to pickle files
    ancestors_path = os.path.join(output_dir, ancestors_file)
    depths_path = os.path.join(output_dir, depths_file)

    try:
        logger.info(f"Saving ancestor sets to {ancestors_path}")
        with open(ancestors_path, "wb") as f:
            pickle.dump(ancestors_map, f)

        logger.info(f"Saving term depths to {depths_path}")
        with open(depths_path, "wb") as f:
            pickle.dump(depths_map, f)

        logger.info("Precomputation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to save precomputed data: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute HPO graph properties for similarity calculation"
    )

    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_HPO_JSON_PATH,
        help=f"Path to hp.json file (default: {DEFAULT_HPO_JSON_PATH})",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output files (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--ancestors-file",
        default=DEFAULT_ANCESTORS_FILE,
        help=f"Filename for ancestors pickle (default: {DEFAULT_ANCESTORS_FILE})",
    )

    parser.add_argument(
        "--depths-file",
        default=DEFAULT_DEPTHS_FILE,
        help=f"Filename for depths pickle (default: {DEFAULT_DEPTHS_FILE})",
    )

    args = parser.parse_args()

    success = run_precomputation(
        args.input, args.output_dir, args.ancestors_file, args.depths_file
    )

    sys.exit(0 if success else 1)
