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

from phentrieve.config import (
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
    Download the latest version of the HPO JSON file.

    Returns:
        bool: True if the file was downloaded successfully, False otherwise
    """
    os.makedirs(os.path.dirname(HPO_FILE_PATH), exist_ok=True)

    try:
        response = requests.get(HPO_JSON_URL, stream=True)
        response.raise_for_status()  # Raise an error for bad responses

        with open(HPO_FILE_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    except Exception as e:
        print(f"Error downloading HPO JSON file: {str(e)}")
        return False


def load_hpo_json() -> Optional[dict]:
    """
    Load the HPO JSON file.

    Returns:
        dict: The loaded HPO JSON data or None if it fails
    """
    try:
        if not os.path.exists(HPO_FILE_PATH):
            print(f"HPO JSON file not found at {HPO_FILE_PATH}")
            if download_hpo_json():
                print("HPO JSON file downloaded successfully")
            else:
                print("Failed to download HPO JSON file")
                return None

        with open(HPO_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data

    except Exception as e:
        print(f"Error loading HPO JSON file: {str(e)}")
        return None


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


def save_terms_as_json_files(terms: Dict[str, dict]) -> None:
    """
    Save each HPO term as an individual JSON file.

    Args:
        terms: Dictionary mapping HPO IDs to term data
    """
    # Create the directory if it doesn't exist
    os.makedirs(HPO_TERMS_DIR, exist_ok=True)

    # Save each term as a JSON file
    for term_id, term_data in tqdm(terms.items(), desc="Saving HPO terms"):
        # Replace the colon with an underscore for the filename
        filename = term_id.replace(":", "_") + ".json"
        file_path = os.path.join(HPO_TERMS_DIR, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(term_data, f, ensure_ascii=False, indent=2)


def compute_ancestors(is_a_relationships: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """
    Compute all ancestors (transitive closure) for each HPO term.

    This uses a breadth-first search to find all ancestors of each term.

    Args:
        is_a_relationships: Dictionary mapping term IDs to lists of direct parent IDs

    Returns:
        Dictionary mapping term IDs to sets of all ancestor IDs
    """
    ancestors: Dict[str, Set[str]] = {}

    for term_id in tqdm(is_a_relationships, desc="Computing ancestors"):
        # Skip if already computed
        if term_id in ancestors:
            continue

        # Initialize the ancestor set for this term
        ancestors[term_id] = set()

        # Initialize the queue for BFS
        queue = deque(is_a_relationships[term_id])
        visited = set(is_a_relationships[term_id])

        # BFS to find all ancestors
        while queue:
            parent_id = queue.popleft()
            ancestors[term_id].add(parent_id)

            # Add parent's parents to queue if not visited
            for grandparent_id in is_a_relationships.get(parent_id, []):
                if grandparent_id not in visited:
                    queue.append(grandparent_id)
                    visited.add(grandparent_id)

    return ancestors


def compute_term_depths(
    is_a_relationships: Dict[str, List[str]], root_id: str = PHENOTYPE_ROOT
) -> Dict[str, int]:
    """
    Compute the depth of each term from the root.

    This uses a breadth-first search starting from the root.

    Args:
        is_a_relationships: Dictionary mapping term IDs to lists of parent IDs
        root_id: The ID of the root term (default is Phenotypic abnormality HP:0000118)

    Returns:
        Dictionary mapping term IDs to their depths
    """
    # Initialize depths dictionary with -1 (not yet visited)
    depths: Dict[str, int] = {term_id: -1 for term_id in is_a_relationships}

    # Check if root exists
    if root_id not in is_a_relationships:
        print(f"Root term {root_id} not found in the ontology graph")
        return depths

    # Set root depth to 0
    depths[root_id] = 0

    # Create a reverse mapping: child -> parents becomes parent -> children
    children: Dict[str, List[str]] = defaultdict(list)
    for child, parents in is_a_relationships.items():
        for parent in parents:
            children[parent].append(child)

    # Use BFS to compute depths
    queue = deque([root_id])
    while queue:
        term_id = queue.popleft()
        current_depth = depths[term_id]

        # Process all children
        for child_id in children[term_id]:
            # If not visited or found a shorter path
            if depths[child_id] == -1 or depths[child_id] > current_depth + 1:
                depths[child_id] = current_depth + 1
                queue.append(child_id)

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


def save_ancestors_to_file(ancestors: Dict[str, Set[str]]) -> None:
    """
    Save the computed ancestors to a pickle file.

    Args:
        ancestors: Dictionary mapping term IDs to sets of ancestor IDs
    """
    os.makedirs(os.path.dirname(HPO_ANCESTORS_FILE), exist_ok=True)
    with open(HPO_ANCESTORS_FILE, "wb") as f:
        pickle.dump(ancestors, f)


def save_depths_to_file(depths: Dict[str, int]) -> None:
    """
    Save the computed term depths to a pickle file.

    Args:
        depths: Dictionary mapping term IDs to their depths
    """
    os.makedirs(os.path.dirname(HPO_DEPTHS_FILE), exist_ok=True)
    with open(HPO_DEPTHS_FILE, "wb") as f:
        pickle.dump(depths, f)


def load_ancestors() -> Dict[str, Set[str]]:
    """
    Load the precomputed ancestors from the pickle file.

    Returns:
        Dictionary mapping term IDs to sets of ancestor IDs
    """
    if not os.path.exists(HPO_ANCESTORS_FILE):
        print(f"Ancestors file not found: {HPO_ANCESTORS_FILE}")
        return {}

    with open(HPO_ANCESTORS_FILE, "rb") as f:
        return pickle.load(f)


def load_term_depths() -> Dict[str, int]:
    """
    Load the precomputed term depths from the pickle file.

    Returns:
        Dictionary mapping term IDs to their depths
    """
    if not os.path.exists(HPO_DEPTHS_FILE):
        print(f"Depths file not found: {HPO_DEPTHS_FILE}")
        return {}

    with open(HPO_DEPTHS_FILE, "rb") as f:
        return pickle.load(f)


def prepare_hpo_data(force_update: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Prepare all HPO data: download, parse, and save.

    Args:
        force_update: Force updating the data even if files exist

    Returns:
        Tuple containing success status and optional error message
    """
    # Check if we need to download the HPO file
    if force_update or not os.path.exists(HPO_FILE_PATH):
        print("Downloading HPO JSON file...")
        if not download_hpo_json():
            return False, "Failed to download HPO JSON file"

    # Load the HPO JSON file
    print("Loading HPO data...")
    hpo_data = load_hpo_json()
    if not hpo_data:
        return False, "Failed to load HPO JSON file"

    # Extract terms
    print("Extracting HPO terms...")
    terms = extract_hpo_terms(hpo_data)
    if not terms:
        return False, "Failed to extract HPO terms"

    # Build the ontology graph
    print("Building ontology graph...")
    is_a_relationships = build_ontology_graph(hpo_data, terms)

    # Compute ancestors
    print("Computing ancestors...")
    ancestors = compute_ancestors(is_a_relationships)

    # Filter terms to phenotypic abnormalities
    print("Filtering phenotypic terms...")
    filtered_terms = filter_phenotypic_terms(terms, ancestors)
    if not filtered_terms:
        return False, "Failed to filter phenotypic terms"

    # Save filtered terms as individual JSON files
    if (
        force_update
        or not os.path.exists(HPO_TERMS_DIR)
        or not os.listdir(HPO_TERMS_DIR)
    ):
        print("Saving HPO terms as JSON files...")
        save_terms_as_json_files(filtered_terms)

    # Compute term depths
    print("Computing term depths...")
    depths = compute_term_depths(is_a_relationships)

    # Save ancestors and depths
    print("Saving ancestors and depths...")
    save_ancestors_to_file(ancestors)
    save_depths_to_file(depths)

    return True, None


if __name__ == "__main__":
    # Create a simple command-line interface
    import argparse

    parser = argparse.ArgumentParser(description="Download and process HPO data")
    parser.add_argument(
        "--force", action="store_true", help="Force update even if files exist"
    )
    args = parser.parse_args()

    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Prepare the HPO data
    success, error = prepare_hpo_data(force_update=args.force)
    if not success:
        print(f"Error: {error}")
        sys.exit(1)
    else:
        print("HPO data preparation completed successfully!")
