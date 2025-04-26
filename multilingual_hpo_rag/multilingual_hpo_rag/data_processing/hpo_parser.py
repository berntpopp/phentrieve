"""
HPO data processing and preparation module.

This module provides functions for downloading, parsing, and processing
Human Phenotype Ontology (HPO) data including:
- Downloading the HPO JSON file
- Extracting individual HPO terms
- Building the HPO graph structure
- Precomputing graph properties (ancestor sets, term depths)
"""

import json
import logging
import os
import pickle
import re
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import requests
from tqdm import tqdm

from multilingual_hpo_rag.config import (
    HPO_FILE_PATH,
    HPO_TERMS_DIR,
    HPO_ANCESTORS_FILE,
    HPO_DEPTHS_FILE,
    DATA_DIR,
    PHENOTYPE_ROOT,
)


# HPO download settings
HPO_JSON_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-03-03/hp.json"

# Root ID for phenotypic abnormalities is defined in config.py

# HPO branches to exclude
EXCLUDED_ROOTS = {
    "HP:0000005",  # Mode of inheritance
    "HP:0012823",  # Clinical modifier
    "HP:0031797",  # Clinical course
    "HP:0040279",  # Frequency
    "HP:0025354",  # Evidence
}


def download_hpo_json() -> bool:
    """
    Download the HPO JSON file if it doesn't exist.

    Returns:
        bool: True if download successful or file already exists, False otherwise
    """
    if not os.path.exists(HPO_FILE_PATH):
        logging.info(f"Downloading {HPO_JSON_URL}...")
        os.makedirs(os.path.dirname(HPO_FILE_PATH), exist_ok=True)
        try:
            response = requests.get(HPO_JSON_URL, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            # Show download progress
            downloaded = 0
            with open(HPO_FILE_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    f.write(chunk)
                    # Simple progress indicator
                    sys.stdout.write(
                        f"\rDownloaded {downloaded / 1024 / 1024:.1f} MB of {total_size / 1024 / 1024:.1f} MB"
                    )
                    sys.stdout.flush()
            logging.info(f"\nSaved HPO data to {HPO_FILE_PATH}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading HPO file: {e}")
            return False
    else:
        logging.info(f"{HPO_FILE_PATH} already exists.")
        return True


def normalize_id(hpo_id: str) -> str:
    """
    Normalize an HPO ID from various formats to standard HP:XXXXXXX format.

    Args:
        hpo_id: HPO identifier in various formats

    Returns:
        Normalized HPO ID string
    """
    if not hpo_id:
        return ""

    # Handle URLs in format http://purl.obolibrary.org/obo/HP_0000118
    if hpo_id.startswith("http"):
        parts = hpo_id.split("/")
        if len(parts) > 0:
            hpo_id = parts[-1].replace("HP_", "HP:")

    # Replace underscore with colon if needed
    hpo_id = hpo_id.replace("_", ":")

    # Standard HP:0000123 format
    if hpo_id.startswith("HP:"):
        return hpo_id

    # Handle numeric strings
    if hpo_id.isdigit() or (hpo_id.startswith("0") and hpo_id.isdigit()):
        # Format as 7-digit ID with HP: prefix
        return f"HP:{int(hpo_id):07d}"

    return hpo_id


def extract_hpo_terms() -> bool:
    """
    Extract individual HPO terms from the main JSON file into separate files.

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    # Check if the source file exists
    if not os.path.exists(HPO_FILE_PATH):
        logging.error(f"Error: {HPO_FILE_PATH} not found.")
        return False

    # Create/recreate the output directory
    if os.path.exists(HPO_TERMS_DIR):
        logging.info(f"Removing existing directory: {HPO_TERMS_DIR}")
        shutil.rmtree(HPO_TERMS_DIR)

    os.makedirs(HPO_TERMS_DIR, exist_ok=True)

    # Read the main HPO JSON file
    logging.info(f"Parsing {HPO_FILE_PATH}...")
    try:
        with open(HPO_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {HPO_FILE_PATH}.")
        return False
    except Exception as e:
        logging.error(f"Error reading {HPO_FILE_PATH}: {e}")
        return False

    # Extract nodes and their relationships
    nodes = {}
    edges = {}

    logging.info("Building HPO hierarchy structure...")

    # Process all graphs in the file
    for graph in data.get("graphs", []):
        # Process nodes first
        for node in graph.get("nodes", []):
            # Normalize the ID
            node_id = normalize_id(node.get("id", ""))

            # Only process HP terms
            if node_id.startswith("HP:"):
                nodes[node_id] = node

        # Now process the relationships (edges)
        for edge in graph.get("edges", []):
            # We're interested in 'is_a' relationships to build the hierarchy
            if (
                edge.get("pred") == "is_a"
                or edge.get("pred") == "http://purl.obolibrary.org/obo/BFO_0000050"
            ):
                subj = normalize_id(edge.get("sub", edge.get("subj", "")))
                obj = normalize_id(edge.get("obj", ""))

                # Only process edges between HP terms
                if subj.startswith("HP:") and obj.startswith("HP:"):
                    # Store the relationship: obj is parent of subj
                    if obj not in edges:
                        edges[obj] = []
                    if subj not in edges[obj]:
                        edges[obj].append(subj)  # Parent -> Children mapping

    # Log some stats
    logging.info(
        f"Found {len(nodes)} nodes and {sum(len(children) for children in edges.values())} edges"
    )

    # Verify the root node exists and has children
    if PHENOTYPE_ROOT not in nodes:
        logging.error(f"Root node {PHENOTYPE_ROOT} not found in the data!")
        return False

    if PHENOTYPE_ROOT not in edges or not edges[PHENOTYPE_ROOT]:
        logging.error(f"Root node {PHENOTYPE_ROOT} has no children!")
        return False

    logging.info(
        f"Root node {PHENOTYPE_ROOT} has {len(edges.get(PHENOTYPE_ROOT, []))} direct children"
    )

    # Collect all phenotype terms starting from the root
    phenotype_terms_ids = set()
    phenotype_terms_ids.add(PHENOTYPE_ROOT)  # Add the root itself

    # Breadth-first search to find all descendants of the phenotype root
    logging.info("Identifying all phenotype terms...")
    to_visit = [PHENOTYPE_ROOT]
    visited = set(to_visit)

    while to_visit:
        current_id = to_visit.pop(0)

        # Get all children of the current node
        children = edges.get(current_id, [])

        for child_id in children:
            if child_id not in visited and child_id not in EXCLUDED_ROOTS:
                phenotype_terms_ids.add(child_id)
                to_visit.append(child_id)
                visited.add(child_id)

    # Remove excluded roots and their descendants
    for excluded_root in EXCLUDED_ROOTS:
        if excluded_root in phenotype_terms_ids:
            phenotype_terms_ids.remove(excluded_root)

        # Find and remove descendants of excluded roots
        if excluded_root in edges:
            excluded_descendants = set()
            to_visit = [excluded_root]
            visited_exclusions = set([excluded_root])

            while to_visit:
                current_id = to_visit.pop(0)
                excluded_descendants.add(current_id)

                for child_id in edges.get(current_id, []):
                    if child_id not in visited_exclusions:
                        to_visit.append(child_id)
                        visited_exclusions.add(child_id)

            # Remove the excluded descendants from our phenotype terms
            phenotype_terms_ids -= excluded_descendants

    logging.info(
        f"Found {len(phenotype_terms_ids)} phenotypic abnormality terms in HPO"
    )

    # Save each phenotype term as a separate JSON file
    logging.info("Saving individual HPO term files...")
    term_count = 0

    with tqdm(total=len(phenotype_terms_ids), desc="Extracting HPO terms") as pbar:
        for term_id in phenotype_terms_ids:
            if term_id in nodes:
                node = nodes[term_id]
                # Create filename from term ID
                clean_id = term_id.replace(":", "_")

                # Create a file for each term
                output_path = os.path.join(HPO_TERMS_DIR, f"{clean_id}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(node, f, indent=2)
                term_count += 1
                pbar.update(1)

    logging.info(f"Successfully extracted {term_count} HPO terms to {HPO_TERMS_DIR}")
    return True


def load_and_build_graphs(
    hpo_json_path: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Set[str]]:
    """
    Load the HPO JSON file and build directed graphs.

    Args:
        hpo_json_path: Path to the HPO JSON file

    Returns:
        Tuple containing:
        - child_to_parents: Dict mapping child term to list of parent terms
        - parent_to_children: Dict mapping parent term to list of child terms
        - all_hpo_term_ids: Set of all HPO term IDs
    """
    logging.info(f"Loading HPO ontology from {hpo_json_path}")
    try:
        with open(hpo_json_path, "r", encoding="utf-8") as f:
            hpo_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load HPO JSON file: {e}")
        return {}, {}, set()

    # Extract nodes and edges
    all_hpo_term_ids = set()
    child_to_parents = defaultdict(list)
    parent_to_children = defaultdict(list)

    # Process nodes first to get all terms
    logging.info("Processing HPO nodes")
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
                logging.warning(f"Error normalizing ID {term_id}: {e}")

    logging.info(f"Processed {nodes_processed} HPO nodes")

    # Process edges to build parent-child relationships
    logging.info("Processing HPO edges (is_a relationships)")
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
                logging.warning(f"Error processing edge: {e}")

    logging.info(f"Processed {edges_processed} is_a relationships")

    # Add root if missing
    if PHENOTYPE_ROOT not in all_hpo_term_ids:
        logging.warning(f"HPO root {PHENOTYPE_ROOT} not found in nodes, adding manually")
        all_hpo_term_ids.add(PHENOTYPE_ROOT)

    # Check for orphaned nodes (no parents except root)
    orphaned = 0
    for term_id in all_hpo_term_ids:
        if term_id != PHENOTYPE_ROOT and term_id not in child_to_parents:
            orphaned += 1
            logging.debug(f"Term {term_id} has no parents, connecting to root")
            child_to_parents[term_id].append(PHENOTYPE_ROOT)
            parent_to_children[PHENOTYPE_ROOT].append(term_id)

    if orphaned:
        logging.warning(f"Connected {orphaned} orphaned terms to root")

    return child_to_parents, parent_to_children, all_hpo_term_ids


def calculate_depths(
    parent_to_children: Dict[str, List[str]],
    all_term_ids: Set[str],
    root_id: str = PHENOTYPE_ROOT,
) -> Dict[str, int]:
    """
    Calculate shortest path depths for all HPO terms using BFS from the root.

    Args:
        parent_to_children: Dict mapping parent term to list of child terms
        all_term_ids: Set of all HPO term IDs
        root_id: Root node to start BFS from

    Returns:
        Dict mapping HPO term ID to its depth from root
    """
    logging.info(f"Calculating term depths from root {root_id}")

    # Initialize all depths to -1 (unreachable)
    depths = {term_id: -1 for term_id in all_term_ids}

    # BFS to calculate shortest path from root
    queue = deque([(root_id, 0)])  # (term_id, depth)
    depths[root_id] = 0

    while queue:
        term_id, depth = queue.popleft()

        for child_id in parent_to_children.get(term_id, []):
            # If we haven't visited this child yet
            if depths[child_id] == -1:
                depths[child_id] = depth + 1
                queue.append((child_id, depth + 1))

    # Count reachable terms
    reachable_count = sum(1 for d in depths.values() if d >= 0)
    unreachable_count = sum(1 for d in depths.values() if d < 0)

    logging.info(f"Calculated depths for {reachable_count} HPO terms")
    if unreachable_count:
        logging.warning(
            f"Warning: {unreachable_count} terms are unreachable from root {root_id}"
        )

    max_depth = max(depths.values()) if depths else 0
    logging.info(f"Maximum depth in HPO tree: {max_depth}")

    return depths


def calculate_all_ancestors(
    child_to_parents: Dict[str, List[str]], all_term_ids: Set[str]
) -> Dict[str, Set[str]]:
    """
    Calculate all ancestors (including self) for each HPO term.

    Args:
        child_to_parents: Dict mapping child term to list of parent terms
        all_term_ids: Set of all HPO term IDs

    Returns:
        Dict mapping HPO term ID to set of all ancestor terms (including self)
    """
    logging.info("Calculating ancestor sets for all HPO terms")

    ancestors_map = {}

    for term_id in all_term_ids:
        # Include self in ancestors
        ancestors = {term_id}

        # Start BFS from parents
        queue = deque(child_to_parents.get(term_id, []))
        while queue:
            # Exclude terms under certain branches (e.g., mode of inheritance)
            if PHENOTYPE_ROOT in ancestors and any(root in ancestors for root in EXCLUDED_ROOTS):
                logging.debug(f"{node_id} excluded as it's under excluded root")
                continue

            if PHENOTYPE_ROOT in ancestors:
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
    logging.info(
        f"Calculated ancestor sets for {len(ancestors_map)} HPO terms (avg {avg_ancestor_count:.2f} ancestors per term)"
    )

    return ancestors_map


def prepare_hpo_ontology_data() -> bool:
    """
    Prepare all necessary HPO ontology data for RAG system.

    This function coordinates the full HPO data preparation process:
    1. Download HPO JSON file
    2. Extract individual HPO terms
    3. Build HPO graph structure
    4. Precompute graph properties

    Returns:
        bool: True if the preparation process was successful, False otherwise
    """
    # Make sure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Download HPO JSON
    if not download_hpo_json():
        logging.error("Failed to download HPO JSON file.")
        return False

    # 2. Extract individual HPO terms
    if not extract_hpo_terms():
        logging.error("Failed to extract HPO terms.")
        return False

    # 3 & 4. Build graph and precompute properties
    logging.info("Building HPO graph structure...")
    child_to_parents, parent_to_children, all_term_ids = load_and_build_graphs(
        HPO_FILE_PATH
    )

    if not all((child_to_parents, parent_to_children, all_term_ids)):
        logging.error("Failed to build HPO graph structure.")
        return False

    # Calculate depths
    logging.info("Calculating term depths...")
    depths_map = calculate_depths(parent_to_children, all_term_ids)

    # Calculate ancestor sets
    logging.info("Calculating term ancestors...")
    ancestors_map = calculate_all_ancestors(child_to_parents, all_term_ids)

    # Save to pickle files
    try:
        logging.info(f"Saving ancestor sets to {HPO_ANCESTORS_FILE}")
        with open(HPO_ANCESTORS_FILE, "wb") as f:
            pickle.dump(ancestors_map, f)

        logging.info(f"Saving term depths to {HPO_DEPTHS_FILE}")
        with open(HPO_DEPTHS_FILE, "wb") as f:
            pickle.dump(depths_map, f)

        logging.info("HPO ontology data preparation completed successfully")
        return True

    except Exception as e:
        logging.error(f"Failed to save precomputed HPO data: {e}")
        return False
