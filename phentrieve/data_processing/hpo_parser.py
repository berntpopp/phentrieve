#!/usr/bin/env python3
"""
HPO data processing and preparation module.

This module provides functions for downloading, parsing, and processing
Human Phenotype Ontology (HPO) data including:
- Downloading the HPO JSON file
- Extracting ALL individual HPO terms
- Building the HPO graph structure
- Precomputing graph properties (ancestor sets, term depths) for ALL terms
"""

import json
import logging
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import requests
from tqdm import tqdm

# Assuming config and utils are in the phentrieve package and accessible
from phentrieve.config import (
    DEFAULT_HPO_DB_FILENAME,
    DEFAULT_HPO_FILENAME,
    HPO_BASE_URL,
    HPO_CHUNK_SIZE,
    HPO_DOWNLOAD_TIMEOUT,
    HPO_VERSION,
)
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.utils import (
    get_default_data_dir,
    normalize_id,
    resolve_data_path,
)

# HPO download settings (constructed from config)
HPO_JSON_URL = f"{HPO_BASE_URL}/{HPO_VERSION}/hp.json"
TRUE_ONTOLOGY_ROOT = "HP:0000001"  # All of HPO

# HPO branches that are often excluded for *phenotypic abnormality* specific tasks,
# but all terms will be extracted and processed for graph properties.
EXCLUDED_ROOTS_FOR_PHENOTYPIC_FILTERING = {
    "HP:0000005",  # Mode of inheritance
    "HP:0012823",  # Clinical modifier
    "HP:0031797",  # Clinical course
    "HP:0040279",  # Frequency
    "HP:0025354",  # Evidence
}

logger = logging.getLogger(__name__)


def download_hpo_json(hpo_file_path: Path) -> bool:
    """
    Download the latest version of the HPO JSON file.
    """
    os.makedirs(os.path.dirname(hpo_file_path), exist_ok=True)
    logger.info(
        f"Attempting to download HPO JSON from {HPO_JSON_URL} to {hpo_file_path}"
    )
    try:
        response = requests.get(HPO_JSON_URL, stream=True, timeout=HPO_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        with open(hpo_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=HPO_CHUNK_SIZE):
                f.write(chunk)
        logger.info("HPO JSON file downloaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading HPO JSON file: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during HPO JSON download: {e}")
        return False


def load_hpo_json(hpo_file_path: Path) -> Optional[dict]:
    """
    Load the HPO JSON file.
    """
    try:
        if not os.path.exists(hpo_file_path):
            logger.warning(
                f"HPO JSON file not found at {hpo_file_path}. Attempting download."
            )
            if not download_hpo_json(hpo_file_path):
                logger.error("Failed to download HPO JSON file.")
                return None

        logger.info(f"Loading HPO JSON from {hpo_file_path}")
        with open(hpo_file_path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("HPO JSON file loaded successfully.")
        # Cast to dict to match return type annotation
        return dict(data) if isinstance(data, dict) else {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding HPO JSON file {hpo_file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading HPO JSON file {hpo_file_path}: {e}")
        return None


def _parse_hpo_json_to_graphs(
    hpo_data: dict,
) -> tuple[
    Optional[dict[str, dict]],
    Optional[dict[str, list[str]]],
    Optional[dict[str, list[str]]],
    Optional[set[str]],
]:
    """
    Parses raw HPO JSON data into term data, parent->child, and child->parent relationships.
    """
    all_nodes_data: dict[str, dict] = {}
    parent_to_children_map: dict[str, list[str]] = defaultdict(list)
    child_to_parents_map: dict[str, list[str]] = defaultdict(list)
    all_term_ids: set[str] = set()

    logger.debug("Parsing nodes and edges from HPO JSON...")

    graphs_data = hpo_data.get("graphs")
    if not graphs_data or not isinstance(graphs_data, list) or not graphs_data[0]:
        logger.error("Invalid HPO JSON structure: 'graphs' array is missing or empty.")
        return None, None, None, None

    graph = graphs_data[0]  # Assume single graph structure

    # Process nodes first
    raw_nodes = graph.get("nodes", [])
    if not raw_nodes:
        logger.warning("No nodes found in HPO graph data.")

    for node_obj in raw_nodes:
        original_id = node_obj.get("id")
        if not original_id:
            logger.warning(f"Node found without an ID: {node_obj.get('lbl', 'N/A')}")
            continue

        node_id_norm = normalize_id(original_id)
        if node_id_norm and node_id_norm.startswith("HP:"):
            all_nodes_data[node_id_norm] = node_obj
            all_term_ids.add(node_id_norm)
        # else: # Optionally log non-HP terms or terms that don't normalize
        # logger.debug(f"Skipping node with non-HP or unnormalizable ID: {original_id} -> {node_id_norm}")

    if (
        TRUE_ONTOLOGY_ROOT not in all_term_ids and TRUE_ONTOLOGY_ROOT in all_nodes_data
    ):  # double check if normalize_id was the issue
        all_term_ids.add(TRUE_ONTOLOGY_ROOT)  # ensure root is there
        logger.info(
            f"Manually added {TRUE_ONTOLOGY_ROOT} to all_term_ids as it was in all_nodes_data but not initially in set."
        )

    # Process edges for parent->child and child->parent mappings (is_a)
    raw_edges = graph.get("edges", [])
    if not raw_edges:
        logger.warning("No edges found in HPO graph data.")

    for edge_obj in raw_edges:
        pred = edge_obj.get("pred")
        # Standard 'is_a' or common URI for 'is_a'
        if (
            pred == "is_a"
            or pred == "http://www.w3.org/2000/01/rdf-schema#subClassOf"
            or pred == "http://purl.obolibrary.org/obo/BFO_0000050"
        ):
            subj_orig = edge_obj.get("sub", edge_obj.get("subj"))  # child
            obj_orig = edge_obj.get("obj")  # parent

            if not subj_orig or not obj_orig:
                logger.warning(f"Edge found with missing subject or object: {edge_obj}")
                continue

            subj_norm = normalize_id(subj_orig)
            obj_norm = normalize_id(obj_orig)

            if (
                subj_norm
                and obj_norm
                and subj_norm.startswith("HP:")
                and obj_norm.startswith("HP:")
            ):
                # Ensure both terms are known (were parsed as nodes)
                if subj_norm in all_term_ids and obj_norm in all_term_ids:
                    parent_to_children_map[obj_norm].append(subj_norm)
                    child_to_parents_map[subj_norm].append(obj_norm)
                # else: # Optionally log edges connecting to unknown terms
                # logger.debug(f"Edge connects unknown terms: {subj_norm} is_a {obj_norm}")

    num_parent_child_edges = sum(len(v) for v in parent_to_children_map.values())
    num_child_parent_edges = sum(len(v) for v in child_to_parents_map.values())

    logger.info(
        f"Parsed {len(all_nodes_data)} total HPO terms (nodes). Found {num_parent_child_edges} parent->child and {num_child_parent_edges} child->parent 'is_a' relationships."
    )

    # Ensure the true ontology root is present if defined.
    if TRUE_ONTOLOGY_ROOT not in all_term_ids:
        logger.warning(
            f"True ontology root {TRUE_ONTOLOGY_ROOT} not found among parsed term IDs. This may affect depth/ancestor calculations."
        )
        # If it was in the original nodes but got filtered by normalize_id or other logic, this is an issue.
        # For now, we proceed, but this is a critical warning.
        # One might consider adding it manually if it's absolutely essential and known to be the root.
        # e.g. if TRUE_ONTOLOGY_ROOT in all_nodes_data: all_term_ids.add(TRUE_ONTOLOGY_ROOT)

    return all_nodes_data, parent_to_children_map, child_to_parents_map, all_term_ids


def compute_ancestors_iterative(
    child_to_parents_map: dict[str, list[str]], all_term_ids: set[str]
) -> dict[str, set[str]]:
    """
    Compute all ancestors (including self) for each HPO term using iterative BFS.
    Args:
        child_to_parents_map: Dictionary mapping child term IDs to lists of direct parent IDs.
        all_term_ids: A set containing all known HPO term IDs in the ontology.
    Returns:
        Dictionary mapping term IDs to sets of all ancestor IDs (including self).
    """
    logger.info("Computing ancestors for all HPO terms (iterative BFS approach)...")
    ancestors_map: dict[str, set[str]] = {}

    for term_id in tqdm(all_term_ids, desc="Computing ancestors", unit="term"):
        current_term_ancestors = {term_id}  # Always include self

        queue: deque[str] = deque()
        # Add direct parents to the queue and to ancestors
        direct_parents = child_to_parents_map.get(term_id, [])
        for parent_id in direct_parents:
            if parent_id in all_term_ids:  # Ensure parent is a known term
                if parent_id not in current_term_ancestors:
                    queue.append(parent_id)
                    current_term_ancestors.add(parent_id)

        # BFS to find all further ancestors
        visited_bfs_for_term = set(
            current_term_ancestors
        )  # Keep track of visited nodes for this term's BFS

        while queue:
            current_ancestor_candidate = queue.popleft()
            # Get parents of this candidate (i.e., grandparents of the original term_id, or higher)
            parents_of_candidate = child_to_parents_map.get(
                current_ancestor_candidate, []
            )
            for p_of_c_id in parents_of_candidate:
                if p_of_c_id in all_term_ids and p_of_c_id not in visited_bfs_for_term:
                    current_term_ancestors.add(p_of_c_id)
                    visited_bfs_for_term.add(p_of_c_id)
                    queue.append(p_of_c_id)

        ancestors_map[term_id] = current_term_ancestors

    # Verification Step
    root_check_count = 0
    if TRUE_ONTOLOGY_ROOT in all_term_ids:
        root_check_count = sum(
            1
            for ancestors_set in ancestors_map.values()
            if TRUE_ONTOLOGY_ROOT in ancestors_set
        )
        logger.info(
            f"Number of terms tracing back to true root {TRUE_ONTOLOGY_ROOT}: {root_check_count} / {len(ancestors_map)}"
        )
        if (
            ancestors_map and root_check_count < len(ancestors_map) * 0.98
        ):  # Expect most to reach root
            logger.warning(
                f"Significantly fewer terms than expected trace back to the true ontology root {TRUE_ONTOLOGY_ROOT}. "
                "This might indicate issues with graph connectivity or parsing."
            )
    else:
        logger.warning(
            f"True ontology root {TRUE_ONTOLOGY_ROOT} not found in parsed terms. Ancestry check against it is incomplete."
        )

    avg_ancestor_count = (
        sum(len(s) for s in ancestors_map.values()) / len(ancestors_map)
        if ancestors_map
        else 0
    )
    logger.info(
        f"Computed ancestor sets for {len(ancestors_map)} HPO terms (avg {avg_ancestor_count:.2f} ancestors per term)."
    )
    return ancestors_map


def compute_term_depths(
    parent_to_children_map: dict[str, list[str]], all_term_ids: set[str]
) -> dict[str, int]:
    """
    Compute the depth of each term from the true ontology root (HP:0000001) using BFS.
    Args:
        parent_to_children_map: Dictionary mapping parent term IDs to lists of direct child IDs.
        all_term_ids: A set containing all known HPO term IDs.
    Returns:
        Dictionary mapping term IDs to their depths (shortest path from true root).
        Returns empty dict if true root is not found or no terms are reachable.
    """
    logger.info(f"Calculating term depths from true HPO root: {TRUE_ONTOLOGY_ROOT}")

    depths: dict[str, int] = dict.fromkeys(all_term_ids, -1)  # Initialize depths

    if TRUE_ONTOLOGY_ROOT not in all_term_ids:
        logger.error(
            f"True HPO root ({TRUE_ONTOLOGY_ROOT}) not found in the set of all parsed HPO terms. "
            "Cannot calculate depths accurately. Ensure HPO JSON is complete and parsing is correct."
        )
        return {}

    depths[TRUE_ONTOLOGY_ROOT] = 0
    queue = deque([(TRUE_ONTOLOGY_ROOT, 0)])
    visited_bfs = {TRUE_ONTOLOGY_ROOT}

    max_depth_found = 0
    processed_count = 0

    while queue:
        current_term_id, current_depth = queue.popleft()
        depths[current_term_id] = current_depth  # Set/update depth
        max_depth_found = max(max_depth_found, current_depth)
        processed_count += 1

        for child_id in parent_to_children_map.get(current_term_id, []):
            if child_id in all_term_ids and child_id not in visited_bfs:
                visited_bfs.add(child_id)
                queue.append((child_id, current_depth + 1))
            # elif child_id not in all_term_ids:
            # logger.debug(f"Child term {child_id} (of parent {current_term_id}) found in edges but not in all_term_ids set.")

    reachable_count = sum(1 for d_val in depths.values() if d_val >= 0)
    unreachable_count = len(all_term_ids) - reachable_count

    logger.info(
        f"Calculated depths for {reachable_count} HPO terms (out of {len(all_term_ids)} total) starting from root {TRUE_ONTOLOGY_ROOT}."
    )
    logger.info(f"Maximum depth found in the HPO graph: {max_depth_found}.")

    if unreachable_count > 0:
        logger.warning(
            f"{unreachable_count} HPO terms were unreachable from the root {TRUE_ONTOLOGY_ROOT}. "
            "These terms will have a depth of -1."
        )
        unreachable_sample = [tid for tid, d_val in depths.items() if d_val < 0][:5]
        if unreachable_sample:
            logger.debug(f"Sample unreachable terms: {unreachable_sample}")

    return depths


def _extract_term_data_for_db(all_nodes_data: dict[str, dict]) -> list[dict[str, Any]]:
    """
    Extract term data from raw HPO nodes for database storage.

    Converts raw node objects into structured term dictionaries suitable
    for bulk insert into the HPO database.

    Args:
        all_nodes_data: Dictionary mapping term IDs to raw node data

    Returns:
        List of term dictionaries with keys: id, label, definition, synonyms, comments
        Note: synonyms and comments are JSON-serialized strings for storage
    """
    terms_data = []

    for term_id, node_data in tqdm(
        all_nodes_data.items(), desc="Preparing HPO terms for database"
    ):
        # Skip non-HP terms (should already be filtered, but double-check)
        if not term_id.startswith("HP:"):
            continue

        # Extract label
        label = node_data.get("lbl", "")

        # Extract definition
        definition = ""
        if (
            "meta" in node_data
            and "definition" in node_data["meta"]
            and "val" in node_data["meta"]["definition"]
        ):
            definition = node_data["meta"]["definition"]["val"]

        # Extract synonyms
        synonyms = []
        if "meta" in node_data and "synonyms" in node_data["meta"]:
            for syn_obj in node_data["meta"]["synonyms"]:
                if "val" in syn_obj:
                    synonyms.append(syn_obj["val"])

        # Extract comments
        comments = []
        if "meta" in node_data and "comments" in node_data["meta"]:
            comments = [c for c in node_data["meta"]["comments"] if c]

        # Prepare term data for database (serialize JSON fields)
        terms_data.append(
            {
                "id": term_id,
                "label": label,
                "definition": definition,
                "synonyms": json.dumps(synonyms, ensure_ascii=False),
                "comments": json.dumps(comments, ensure_ascii=False),
            }
        )

    return terms_data


def prepare_hpo_data(
    force_update: bool = False,
    hpo_file_path: Path | None = None,
    db_path: Path | None = None,
) -> tuple[bool, Optional[str]]:
    """
    Core HPO data preparation: download, parse, save ALL terms to SQLite, compute graph data.

    Args:
        force_update: Force re-download of HPO JSON file
        hpo_file_path: Path to HPO JSON file
        db_path: Path to SQLite database file

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    # Validate required paths
    if hpo_file_path is None:
        return False, "hpo_file_path is required but was not provided"
    if db_path is None:
        return False, "db_path is required but was not provided"

    # 1. Download/Load HPO JSON
    if force_update or not os.path.exists(hpo_file_path):
        logger.info(
            f"Force update or file missing. Downloading HPO JSON to {hpo_file_path}..."
        )
        if not download_hpo_json(hpo_file_path):
            return False, f"Failed to download HPO JSON to {hpo_file_path}"
    else:
        logger.info(f"Using existing HPO JSON: {hpo_file_path}")

    hpo_data = load_hpo_json(hpo_file_path)
    if not hpo_data:
        return False, f"Failed to load HPO JSON from {hpo_file_path}"

    # 2. Parse HPO JSON into structured graphs and node data
    logger.info("Parsing HPO JSON into node data and graph structures...")
    all_nodes_data, parent_to_children_map, child_to_parents_map, all_term_ids = (
        _parse_hpo_json_to_graphs(hpo_data)
    )

    if (
        not all_nodes_data
        or not parent_to_children_map
        or not child_to_parents_map
        or not all_term_ids
    ):
        return (
            False,
            "Failed to parse HPO data into necessary graph structures or node data.",
        )
    logger.info(f"Successfully parsed data for {len(all_term_ids)} HPO terms.")

    # 3. Initialize SQLite database
    logger.info(f"Initializing HPO database at {db_path}...")
    try:
        # Remove existing database if force_update
        if force_update and os.path.exists(db_path):
            logger.info(f"Removing existing database: {db_path}")
            os.remove(db_path)

        db = HPODatabase(db_path)
        db.initialize_schema()
    except Exception as e:
        return False, f"Failed to initialize database: {e}"

    # 4. Extract and insert HPO terms into database
    logger.info("Extracting HPO term data for database storage...")
    terms_data = _extract_term_data_for_db(all_nodes_data)

    if not terms_data:
        logger.warning("No terms extracted for database storage")
        db.close()
        return False, "No HPO terms could be extracted for storage"

    logger.info(f"Inserting {len(terms_data)} HPO terms into database...")
    try:
        num_inserted = db.bulk_insert_terms(terms_data)
        logger.info(f"Successfully inserted {num_inserted} HPO terms")
    except Exception as e:
        db.close()
        return False, f"Failed to insert HPO terms: {e}"

    # 5. Compute Ancestors for ALL terms
    logger.info("Computing ancestor sets for all HPO terms...")
    ancestors_map = compute_ancestors_iterative(child_to_parents_map, all_term_ids)
    if not ancestors_map:
        logger.warning(
            "Ancestor computation resulted in an empty map. Semantic similarity might be affected."
        )

    # 6. Compute Depths for ALL terms from the true ontology root
    logger.info("Computing term depths for all HPO terms...")
    term_depths_map = compute_term_depths(parent_to_children_map, all_term_ids)
    if not term_depths_map:
        logger.warning(
            "Term depth computation resulted in an empty map. Semantic similarity might be affected."
        )

    # 7. Prepare and insert graph metadata
    logger.info("Preparing graph metadata for database storage...")
    graph_metadata = []
    for term_id in tqdm(all_term_ids, desc="Preparing graph metadata"):
        # Get ancestors and depth for this term
        ancestors = ancestors_map.get(term_id, set())
        depth = term_depths_map.get(term_id, 0)

        graph_metadata.append(
            {
                "term_id": term_id,
                "depth": depth,
                "ancestors": json.dumps(list(ancestors), ensure_ascii=False),
            }
        )

    logger.info(f"Inserting {len(graph_metadata)} graph metadata records...")
    try:
        num_graph_inserted = db.bulk_insert_graph_metadata(graph_metadata)
        logger.info(
            f"Successfully inserted {num_graph_inserted} graph metadata records"
        )
    except Exception as e:
        db.close()
        return False, f"Failed to insert graph metadata: {e}"

    # 8. Optimize database
    logger.info("Optimizing database with ANALYZE...")
    try:
        db.optimize()
    except Exception as e:
        logger.warning(f"Database optimization failed: {e}")

    # Close database connection
    db.close()

    logger.info("HPO data preparation completed successfully.")
    logger.info(f"  Total terms: {len(terms_data)}")
    logger.info(f"  Database: {db_path}")
    return True, None


def orchestrate_hpo_preparation(
    debug: bool = False,  # Logging level handled by CLI caller
    force_update: bool = False,
    data_dir_override: Optional[str] = None,
) -> bool:
    """
    Orchestrates HPO data download, extraction of ALL terms to SQLite, and precomputation of graph properties.

    Args:
        debug: Enable debug logging (handled by CLI caller)
        force_update: Force re-download and regeneration of all data
        data_dir_override: Override default data directory

    Returns:
        True if preparation succeeded, False otherwise
    """
    logger.info("Starting HPO ontology data preparation orchestration...")

    try:
        data_dir = resolve_data_path(
            data_dir_override, "data_dir", get_default_data_dir
        )
        logger.info(f"Using data directory: {data_dir}")

        hpo_file_path = data_dir / DEFAULT_HPO_FILENAME
        db_path = data_dir / DEFAULT_HPO_DB_FILENAME

        os.makedirs(data_dir, exist_ok=True)  # Ensure base data_dir exists

        success, error_message = prepare_hpo_data(
            force_update=force_update,
            hpo_file_path=hpo_file_path,
            db_path=db_path,
        )

        if not success:
            logger.error(f"HPO data preparation failed: {error_message}")
            return False

        logger.info("HPO data preparation orchestration completed successfully!")
        logger.info(f"  HPO JSON file: {hpo_file_path}")
        logger.info(f"  HPO database: {db_path}")
        return True

    except Exception as e:
        logger.error(
            f"Critical error during HPO data preparation orchestration: {e}",
            exc_info=True,
        )
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download, parse, and precompute HPO data."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update/re-download of HPO data and re-computation.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Override the default data directory for HPO files.",
    )
    args = parser.parse_args()

    # Setup logging (basic for direct script run, CLI in phentrieve.cli will handle richer setup)
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing root logger configuration
    )

    # Set higher level for noisy libraries if necessary
    # logging.getLogger("requests").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)

    if not orchestrate_hpo_preparation(
        debug=args.debug, force_update=args.force, data_dir_override=args.data_dir
    ):
        sys.exit(1)
    sys.exit(0)
