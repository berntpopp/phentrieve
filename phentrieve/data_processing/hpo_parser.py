"""
HPO data processing and preparation module.

This module provides functions for downloading, parsing, and processing
Human Phenotype Ontology (HPO) data including:
- Downloading the HPO JSON file
- Extracting individual HPO terms
- Building the HPO graph structure
- Precomputing graph properties (ancestor sets, term depths)
"""

import os
import json
import functools
import logging
import pickle
import shutil
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
import networkx as nx
import requests
from tqdm import tqdm

from phentrieve.config import (
    DEFAULT_HPO_FILENAME,
    DEFAULT_HPO_TERMS_SUBDIR,
    DEFAULT_ANCESTORS_FILENAME,
    DEFAULT_DEPTHS_FILENAME,
    PHENOTYPE_ROOT,
)

from phentrieve.utils import (
    normalize_id,
    resolve_data_path,
    get_default_data_dir,
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


def download_hpo_json(hpo_file_path: Path) -> bool:
    """
    Download the latest version of the HPO JSON file.

    Args:
        hpo_file_path: Full path to save the HPO JSON file

    Returns:
        bool: True if the file was downloaded successfully, False otherwise
    """
    os.makedirs(os.path.dirname(hpo_file_path), exist_ok=True)

    try:
        response = requests.get(HPO_JSON_URL, stream=True)
        response.raise_for_status()  # Raise an error for bad responses

        with open(hpo_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    except Exception as e:
        print(f"Error downloading HPO JSON file: {str(e)}")
        return False


def load_hpo_json(hpo_file_path: Path) -> Optional[dict]:
    """
    Load the HPO JSON file.

    Args:
        hpo_file_path: Full path to the HPO JSON file

    Returns:
        dict: The loaded HPO JSON data or None if it fails
    """
    try:
        if not os.path.exists(hpo_file_path):
            print(f"HPO JSON file not found at {hpo_file_path}")
            if download_hpo_json(hpo_file_path):
                print("HPO JSON file downloaded successfully")
            else:
                print("Failed to download HPO JSON file")
                return None

        with open(hpo_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data

    except Exception as e:
        print(f"Error loading HPO JSON file: {str(e)}")
        return None


def _parse_hpo_json_to_nodes_and_edges(
    hpo_data: Dict,
) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
    """Parses raw HPO JSON data into term data and parent->child relationships."""
    nodes = {}
    edges = defaultdict(list)  # parent_id -> list_of_child_ids

    logging.debug("Parsing nodes and edges from HPO JSON...")
    graph = hpo_data.get("graphs", [{}])[0]  # Assume single graph structure

    # Process nodes first
    for node in graph.get("nodes", []):
        node_id_norm = normalize_id(node.get("id", ""))
        if node_id_norm.startswith("HP:"):
            # Store node data using normalized ID as key
            nodes[node_id_norm] = node  # Store raw node data initially

    # Process edges for parent->child mapping (is_a)
    for edge in graph.get("edges", []):
        pred = edge.get("pred")
        if pred == "is_a" or pred == "http://purl.obolibrary.org/obo/BFO_0000050":
            subj_norm = normalize_id(edge.get("sub", edge.get("subj", "")))
            obj_norm = normalize_id(edge.get("obj", ""))

            if subj_norm.startswith("HP:") and obj_norm.startswith("HP:"):
                # obj is parent of subj
                if obj_norm in nodes and subj_norm in nodes:  # Ensure both terms exist
                    edges[obj_norm].append(subj_norm)

    logging.debug(
        f"Parsed {len(nodes)} nodes and {sum(len(v) for v in edges.values())} parent->child edges."
    )
    return nodes, edges  # Return raw nodes dict and parent->child edges


def _identify_phenotypic_term_ids(
    nodes: Dict[str, Dict],
    edges: Dict[str, List[str]],
    is_a_relationships: Dict[str, List[str]],
) -> Set[str]:
    """Identifies phenotypic term IDs by BFS from PHENOTYPE_ROOT, respecting exclusions."""
    phenotype_terms_ids = set()
    root_id = PHENOTYPE_ROOT
    excluded_roots = EXCLUDED_ROOTS

    if root_id not in nodes:
        logging.error(f"Phenotype root {root_id} not found in parsed nodes!")
        return phenotype_terms_ids

    # BFS from phenotype root
    queue = deque([root_id])
    visited = {root_id}
    phenotype_terms_ids.add(root_id)

    while queue:
        current_id = queue.popleft()
        children = edges.get(current_id, [])  # Use parent->child edges map

        for child_id in children:
            # Process child only if it exists in nodes and hasn't been visited
            if child_id in nodes and child_id not in visited:
                visited.add(child_id)
                # Check if this child is an excluded root
                if child_id in excluded_roots:
                    continue  # Skip this branch

                phenotype_terms_ids.add(child_id)
                queue.append(child_id)  # Add to queue only if not excluded

    # Secondary check: Remove terms that are descendants of excluded roots
    terms_to_remove = set()
    for term_id in phenotype_terms_ids:
        # Quick BFS upwards for this term to check ancestry
        ancestors_queue = deque(is_a_relationships.get(term_id, []))
        term_visited = set(ancestors_queue)
        while ancestors_queue:
            parent = ancestors_queue.popleft()
            if parent in excluded_roots:
                terms_to_remove.add(term_id)
                break  # Found an excluded ancestor
            grandparents = is_a_relationships.get(parent, [])
            for gp in grandparents:
                if gp not in term_visited:
                    term_visited.add(gp)
                    ancestors_queue.append(gp)

    final_phenotype_ids = phenotype_terms_ids - terms_to_remove
    logging.info(f"Identified {len(final_phenotype_ids)} phenotypic terms via BFS.")

    return final_phenotype_ids


def extract_hpo_terms(graph: dict) -> Dict[str, dict]:
    """
    Extract all HPO terms from the graph.

    Args:
        graph: The HPO JSON graph data

    Returns:
        Dictionary mapping HPO IDs to term data
    """
    terms = {}

    # Extract all nodes
    for node in graph.get("graphs", [{}])[0].get("nodes", []):
        # Skip nodes without an ID
        if "id" not in node:
            continue

        original_id = node["id"]

        # Format for HP URIs: http://purl.obolibrary.org/obo/HP_0000001
        # Convert to the format HP:0000001
        if "obo/HP_" in original_id:
            # Extract the numeric part and format as HP:number
            hp_number = original_id.split("HP_")[1]
            term_id = f"HP:{hp_number}"
        elif original_id.startswith("HP:"):
            # Already in the right format
            term_id = original_id
        else:
            # Skip non-HP terms
            continue

        # Double-check that we have a valid HP ID
        if not term_id.startswith("HP:"):
            continue

        # Extract other properties
        term_data = {
            "id": term_id,
            "label": node.get("lbl", ""),
            "definition": "",
            "synonyms": [],
            "is_a": [],
            "xrefs": [],
        }

        # Add definition
        if "meta" in node and "definition" in node["meta"]:
            term_data["definition"] = (
                node["meta"]["definition"]["val"]
                if "val" in node["meta"]["definition"]
                else ""
            )

        # Add synonyms
        if "meta" in node and "synonyms" in node["meta"]:
            term_data["synonyms"] = [
                s["val"] for s in node["meta"]["synonyms"] if "val" in s
            ]

        # Add xrefs
        if "meta" in node and "xrefs" in node["meta"]:
            term_data["xrefs"] = [x["val"] for x in node["meta"]["xrefs"] if "val" in x]

        # Store the term
        terms[term_id] = term_data

    return terms


def build_ontology_graph(graph: dict, terms: Dict[str, dict]) -> Dict[str, List[str]]:
    """
    Extract parent-child relationships to build the ontology graph.

    Args:
        graph: The HPO JSON graph data
        terms: Dictionary of HPO terms

    Returns:
        Dictionary mapping term IDs to lists of direct parent IDs
    """
    # Initialize the is_a relationships dictionary
    is_a_relationships = {term_id: [] for term_id in terms}

    # Process edges
    for edge in graph.get("graphs", [{}])[0].get("edges", []):
        pred = edge.get("pred", "")
        subj = edge.get("sub", "")
        obj = edge.get("obj", "")

        # Only process is_a relationships
        if pred == "is_a" and subj.startswith("HP:") and obj.startswith("HP:"):
            if subj in is_a_relationships:
                is_a_relationships[subj].append(obj)

    return is_a_relationships


def save_terms_as_json_files(
    nodes: Dict[str, Dict], term_ids_to_save: Set[str], terms_dir: Path
) -> None:
    """
    Saves specified HPO terms from the nodes map as individual JSON files.

    Args:
        nodes: Dictionary of all HPO terms/nodes
        term_ids_to_save: Set of term IDs to save as individual files
        terms_dir: Directory path where term JSON files will be saved
    """
    os.makedirs(terms_dir, exist_ok=True)
    logging.info(f"Saving {len(term_ids_to_save)} HPO terms to individual JSON files")

    for term_id in tqdm(term_ids_to_save, desc="Saving terms to JSON files"):
        if term_id in nodes:
            node_data = nodes[term_id]

            # Format the file ID: HP:0000123 -> HP_0000123.json
            file_id = term_id.replace(":", "_")
            file_path = terms_dir / f"{file_id}.json"

            # Save the raw data for later use
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(node_data, f, ensure_ascii=False, indent=2)
        else:
            logging.warning(f"Term ID {term_id} not found in nodes")

    logging.info(f"Saved term files to {terms_dir}")


_recursive_child_to_parents_map = {}
_recursive_all_term_ids_set = set()


@functools.lru_cache(maxsize=None)  # Use LRU cache for memoization
def get_term_ancestors_recursive(term_id: str, graph_dict_key: str) -> frozenset[str]:
    """
    Recursive helper with memoization to get ancestors for a single term.
    Uses a special key to identify the graph dictionary in global variables.
    """
    # We'll use global variables to avoid passing unhashable dictionaries
    global _recursive_child_to_parents_map
    global _recursive_all_term_ids_set

    # Base case: Term itself is always an ancestor
    current_ancestors = {term_id}

    # Get direct parents
    direct_parents = _recursive_child_to_parents_map.get(term_id, [])

    # Recursively get ancestors of parents and add them
    for parent_id in direct_parents:
        # Prevent infinite loops and ensure parent is valid & not self
        if parent_id in _recursive_all_term_ids_set and parent_id != term_id:
            # Make the recursive call to get parent's ancestors
            parent_ancestors = get_term_ancestors_recursive(parent_id, graph_dict_key)
            # Update the current set with all ancestors from this parent path
            current_ancestors.update(parent_ancestors)
        elif parent_id not in _recursive_all_term_ids_set:
            # logging.debug(f"Term {term_id} has an unknown parent {parent_id}. Ignoring.")
            pass  # Ignore parents not in the main HPO set

    # Return immutable frozenset for caching compatibility
    return frozenset(current_ancestors)


def compute_ancestors(
    child_to_parents: Dict[str, List[str]], all_term_ids: Set[str]
) -> Dict[str, Set[str]]:
    """
    Compute all ancestors (including self) for each HPO term using recursion with memoization.

    Args:
        child_to_parents: Dictionary mapping child term IDs to lists of direct parent IDs.
        all_term_ids: A set containing all known HPO term IDs in the ontology.

    Returns:
        Dictionary mapping term IDs to sets of all ancestor IDs (including self).
    """
    global _recursive_child_to_parents_map
    global _recursive_all_term_ids_set

    # Set up globals for the recursive function
    _recursive_child_to_parents_map = child_to_parents
    _recursive_all_term_ids_set = all_term_ids

    # Create a unique identifier for this graph to use in caching
    # We'll just use a timestamp as a unique key
    graph_dict_key = str(time.time())

    ancestors_map: Dict[str, Set[str]] = {}
    logging.info(
        f"Computing ancestors for {len(all_term_ids)} total terms using recursive approach."
    )

    # Pre-clear cache to ensure clean state if running multiple times
    get_term_ancestors_recursive.cache_clear()

    # Compute ancestors for all terms using the recursive helper with global state
    for term_id in tqdm(all_term_ids, desc="Computing ancestors", unit="term"):
        ancestors_map[term_id] = set(
            get_term_ancestors_recursive(term_id, graph_dict_key)
        )

    # Clear the cache after computation
    get_term_ancestors_recursive.cache_clear()

    # Clear globals to free memory
    _recursive_child_to_parents_map = {}
    _recursive_all_term_ids_set = set()

    computed_count = len(ancestors_map)
    logging.info(f"Computed ancestor sets for {computed_count} terms.")
    if computed_count != len(all_term_ids):
        logging.warning(
            f"Mismatch: Expected {len(all_term_ids)} ancestor sets, computed {computed_count}."
        )

    # Verification Step: Check if the universal root exists in most sets
    universal_root = "HP:0000001"
    root_check_count = 0
    if universal_root in all_term_ids:
        root_check_count = sum(
            1 for ancestors in ancestors_map.values() if universal_root in ancestors
        )
        logging.info(
            f"Number of terms tracing back to root {universal_root}: {root_check_count} / {computed_count}"
        )
        if root_check_count < computed_count * 0.98:  # Expect almost all to reach root
            logging.warning(
                f"Significantly fewer terms than expected trace back to the universal root {universal_root}. Check graph connectivity."
            )
    else:
        logging.warning(
            f"Universal root {universal_root} not found in parsed terms. Ancestry check might be incomplete."
        )

    return ancestors_map


def compute_term_depths(
    parent_to_children: Dict[str, List[str]], all_term_ids: Set[str]
) -> Dict[str, int]:
    """
    Compute the depth of each term from the true root (HP:0000001) using BFS.

    Args:
        parent_to_children: Dictionary mapping parent term IDs to lists of direct child IDs.
        all_term_ids: A set containing all known HPO term IDs.

    Returns:
        Dictionary mapping term IDs to their depths (shortest path from true root).
        Returns empty dict if true root is not found.
    """
    TRUE_ROOT_ID = "HP:0000001"
    logging.info(f"Calculating term depths from true root {TRUE_ROOT_ID}")

    # Initialize depths dictionary with -1 (not yet visited/unreachable)
    depths: Dict[str, int] = {term_id: -1 for term_id in all_term_ids}

    # Check if the true root exists in the set of all terms
    if TRUE_ROOT_ID not in all_term_ids:
        logging.error(
            f"True root term {TRUE_ROOT_ID} not found in the parsed HPO terms. Cannot calculate depths."
        )
        return {}  # Return empty dict indicating failure

    # Set root depth to 0
    depths[TRUE_ROOT_ID] = 0

    # Use BFS to compute depths, traversing DOWN using the parent_to_children map
    queue = deque([TRUE_ROOT_ID])
    visited_bfs = {TRUE_ROOT_ID}  # Keep track of visited nodes during BFS

    processed_count = 0
    while queue:
        term_id = queue.popleft()
        processed_count += 1

        current_depth = depths.get(term_id, -1)  # Should always exist if in queue
        if current_depth == -1:
            # Should ideally not happen if queue logic is correct
            logging.warning(
                f"Node {term_id} in BFS queue but has invalid depth. Skipping."
            )
            continue

        # Process all children using the provided parent_to_children map
        for child_id in parent_to_children.get(term_id, []):
            # Ensure child is a known term and hasn't been visited in this BFS
            if child_id in depths and child_id not in visited_bfs:
                depths[child_id] = current_depth + 1
                visited_bfs.add(child_id)
                queue.append(child_id)
            # Optional: Log if a child mentioned in edges isn't in all_term_ids
            # elif child_id not in depths:
            #    logging.debug(f"Child term {child_id} (parent {term_id}) not found in the initial set of all terms.")

    # Log statistics
    reachable_count = sum(1 for d in depths.values() if d >= 0)
    unreachable_count = len(all_term_ids) - reachable_count  # More accurate count
    logging.info(
        f"Calculated depths for {reachable_count} HPO terms starting from root {TRUE_ROOT_ID}."
    )
    if unreachable_count > 0:
        logging.warning(
            f"{unreachable_count} terms were unreachable from root {TRUE_ROOT_ID}."
        )
        # Log a few unreachable terms for debugging
        unreachable_sample = [tid for tid, d in depths.items() if d < 0][:10]
        logging.debug(f"Sample unreachable terms: {unreachable_sample}")

    max_depth_val = max((d for d in depths.values() if d >= 0), default=-1)
    logging.info(f"Maximum depth found: {max_depth_val}")

    return depths


def filter_phenotypic_terms(
    terms: Dict[str, dict],
    ancestors: Dict[str, Set[str]],
    root_id: str = PHENOTYPE_ROOT,
) -> Dict[str, dict]:
    """
    Filter HPO terms to include only phenotypic abnormalities.

    This keeps only terms that are descendants of the phenotypic abnormality root,
    excluding modes of inheritance and other non-phenotypic branches.

    Args:
        terms: Dictionary of all HPO terms
        ancestors: Dictionary mapping term IDs to sets of ancestor IDs
        root_id: The ID of the root term (default is Phenotypic abnormality HP:0000118)

    Returns:
        Dictionary of filtered terms
    """
    filtered_terms = {}

    # Check if root is in the ontology
    if root_id not in terms:
        print(f"Root term {root_id} not found in the ontology")
        return filtered_terms

    for term_id, term_data in terms.items():
        # Include a term if:
        # 1. It is the root OR
        # 2. The root is in its ancestors AND
        # 3. None of the excluded roots is in its ancestors
        is_root = term_id == root_id
        is_phenotype = root_id in ancestors.get(term_id, set())
        is_excluded = any(
            excluded in ancestors.get(term_id, set()) for excluded in EXCLUDED_ROOTS
        )

        if (is_root or is_phenotype) and not is_excluded:
            filtered_terms[term_id] = term_data

    return filtered_terms


def save_ancestors_to_file(
    ancestors: Dict[str, Set[str]], ancestors_file: Path
) -> None:
    """Save the ancestors dictionary to a pickle file.

    Args:
        ancestors: Dictionary mapping term IDs to their ancestors
        ancestors_file: Path to the file where ancestors will be saved
    """
    os.makedirs(os.path.dirname(ancestors_file), exist_ok=True)
    with open(ancestors_file, "wb") as f:
        pickle.dump(ancestors, f)


def save_depths_to_file(term_depths: Dict[str, int], depths_file: Path) -> None:
    """Save term depths to a pickle file.

    Args:
        term_depths: Dictionary mapping term IDs to their depths
        depths_file: Path to the file where depths will be saved
    """
    os.makedirs(os.path.dirname(depths_file), exist_ok=True)
    with open(depths_file, "wb") as f:
        pickle.dump(term_depths, f)


def load_ancestors(ancestors_file: Path) -> Dict[str, Set[str]]:
    """
    Load the precomputed ancestors from the pickle file.

    Args:
        ancestors_file: Path to the pickle file containing ancestor data

    Returns:
        Dictionary mapping term IDs to sets of ancestor IDs
    """
    if not os.path.exists(ancestors_file):
        print(f"Ancestors file not found: {ancestors_file}")
        return {}

    with open(ancestors_file, "rb") as f:
        return pickle.load(f)


def load_term_depths(depths_file: Path) -> Dict[str, int]:
    """
    Load the precomputed term depths from the pickle file.

    Args:
        depths_file: Path to the pickle file containing term depths data

    Returns:
        Dictionary mapping term IDs to their depths
    """
    if not os.path.exists(depths_file):
        print(f"Depths file not found: {depths_file}")
        return {}

    with open(depths_file, "rb") as f:
        return pickle.load(f)


def prepare_hpo_data(
    force_update: bool = False,
    hpo_file_path: Path = None,
    hpo_terms_dir: Path = None,
    ancestors_file: Path = None,
    depths_file: Path = None,
) -> Tuple[bool, Optional[str]]:
    """
    Prepare HPO data: download, parse, filter IDs, save term files, compute full graph data.
    (Refactored to follow original logic flow)

    Args:
        force_update: Force updating the data even if files exist
        hpo_file_path: Path to the HPO JSON file
        hpo_terms_dir: Directory to store individual HPO term JSON files
        ancestors_file: Path to save the ancestors pickle file
        depths_file: Path to save the depths pickle file

    Returns:
        Tuple containing success status and optional error message
    """
    # 1. Download/Load JSON
    if force_update or not os.path.exists(hpo_file_path):
        logging.info("Downloading HPO JSON file...")
        if not download_hpo_json(hpo_file_path):
            return False, "Failed to download HPO JSON file"
        logging.info("HPO JSON downloaded.")
    else:
        logging.info(f"Using existing HPO JSON: {hpo_file_path}")

    logging.info("Loading HPO data...")
    hpo_data = load_hpo_json(hpo_file_path)
    if not hpo_data:
        return False, "Failed to load HPO JSON file"

    # 2. Parse into nodes and parent->child edges
    logging.info("Parsing HPO JSON into nodes and edges...")
    nodes, parent_child_edges = _parse_hpo_json_to_nodes_and_edges(hpo_data)
    if not nodes or not parent_child_edges:
        return False, "Failed to parse nodes and edges from HPO JSON"
    all_term_ids = set(nodes.keys())
    logging.info(f"Parsed {len(all_term_ids)} total HPO terms.")

    # 3. Build Child -> Parent graph (needed for both id filtering and ancestor calculation)
    logging.info(
        "Building child-to-parent relationships for term filtering and ancestry..."
    )
    child_parent_graph = build_ontology_graph(hpo_data, nodes)
    if not child_parent_graph:
        return False, "Failed to build child-to-parent graph"

    # 4. Identify Phenotypic Term IDs using BFS on parent->child edges
    logging.info("Identifying phenotypic term IDs...")
    phenotypic_ids = _identify_phenotypic_term_ids(
        nodes, parent_child_edges, child_parent_graph
    )
    if not phenotypic_ids:
        return False, "No phenotypic term IDs were identified."

    # 5. Save *only* the identified phenotypic terms to individual files
    # Recreate directory if force_update or empty/non-existent
    if (
        force_update
        or not os.path.exists(hpo_terms_dir)
        or not os.listdir(hpo_terms_dir)
    ):
        if os.path.exists(hpo_terms_dir):
            logging.info(f"Removing existing terms directory: {hpo_terms_dir}")
            shutil.rmtree(hpo_terms_dir)
        save_terms_as_json_files(nodes, phenotypic_ids, hpo_terms_dir)
    else:
        logging.info(
            f"Skipping saving individual term files as directory exists and "
            f"force_update=False: {hpo_terms_dir}"
        )

    # --- Compute and Save Full Graph Data for Metrics ---
    # 6. Compute Ancestors for ALL terms (needed for metric calculations)
    logging.info("Computing ancestors for all terms...")
    ancestors = compute_ancestors(child_parent_graph, all_term_ids)
    save_ancestors_to_file(ancestors, ancestors_file)
    logging.info(f"Saved ancestors to {ancestors_file}")

    # 7. Compute Depths for ALL terms
    logging.info("Computing term depths...")
    depths = compute_term_depths(parent_child_edges, all_term_ids)
    save_depths_to_file(depths, depths_file)
    logging.info(f"Saved depths to {depths_file}")

    logging.info("HPO data preparation completed following revised flow.")
    return True, None


def orchestrate_hpo_preparation(
    debug: bool = False,
    force_update: bool = False,
    data_dir_override: Optional[str] = None,
) -> bool:
    """Orchestrates the HPO ontology data download, extraction, and precomputation.

    Args:
        debug: Enable debug logging
        force_update: Force updating the data even if files exist
        data_dir_override: Override directory for HPO data files

    Returns:
        True if the data preparation was successful, False otherwise
    """
    logging.info("Starting HPO ontology data preparation orchestration")

    try:
        # Resolve data directory path based on priority: CLI > Config > Default
        data_dir = resolve_data_path(
            data_dir_override, "data_dir", get_default_data_dir
        )
        logging.info(f"Using data directory: {data_dir}")

        # Construct full paths for files using the resolved data directory
        hpo_file_path = data_dir / DEFAULT_HPO_FILENAME
        hpo_terms_dir = data_dir / DEFAULT_HPO_TERMS_SUBDIR
        ancestors_file = data_dir / DEFAULT_ANCESTORS_FILENAME
        depths_file = data_dir / DEFAULT_DEPTHS_FILENAME

        # Make sure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Download HPO JSON if needed
        if not os.path.exists(hpo_file_path) or force_update:
            logging.info("Downloading HPO JSON file...")
            if not download_hpo_json(hpo_file_path):
                logging.error("Failed to download HPO JSON file.")
                return False
            logging.info("HPO JSON file downloaded successfully.")
        else:
            logging.info(f"Using existing HPO JSON file: {hpo_file_path}")

        # Run the full preparation process with resolved paths
        success, error = prepare_hpo_data(
            force_update=force_update,
            hpo_file_path=hpo_file_path,
            hpo_terms_dir=hpo_terms_dir,
            ancestors_file=ancestors_file,
            depths_file=depths_file,
        )

        if not success:
            logging.error(f"Failed to prepare HPO ontology data: {error}")
            return False

        logging.info("HPO data preparation orchestration completed successfully!")
        logging.info(f"HPO JSON file: {hpo_file_path}")
        logging.info(f"HPO terms directory: {hpo_terms_dir}")
        logging.info(f"HPO ancestors file: {ancestors_file}")
        logging.info(f"HPO depths file: {depths_file}")
        return True
    except Exception as e:
        logging.error(
            f"Error during HPO data preparation orchestration: {e}", exc_info=debug
        )
        return False


if __name__ == "__main__":
    # Create a simple command-line interface
    import argparse

    parser = argparse.ArgumentParser(description="Download and process HPO data")
    parser.add_argument(
        "--force", action="store_true", help="Force update even if files exist"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--data-dir", type=str, help="Override the data directory for HPO files"
    )
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Run the orchestration function
    if not orchestrate_hpo_preparation(
        debug=args.debug, force_update=args.force, data_dir_override=args.data_dir
    ):
        sys.exit(1)
