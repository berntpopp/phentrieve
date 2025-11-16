"""
Document creation module for HPO terms.

This module provides functionality for loading HPO terms and converting them
into indexable documents with appropriate metadata for vector storage.
"""

import glob
import json
import logging
import os
from typing import Any, Optional

from phentrieve.config import DEFAULT_HPO_TERMS_SUBDIR
from phentrieve.utils import get_default_data_dir, resolve_data_path


def load_hpo_terms(data_dir_override: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Load HPO terms from individual JSON files in the HPO_TERMS_DIR.

    Returns:
        List of dictionaries containing HPO term data
    """
    # Resolve data directory path using our dynamic path resolution system
    data_dir = resolve_data_path(data_dir_override, "data_dir", get_default_data_dir)

    # Construct path to HPO terms directory
    hpo_terms_dir = data_dir / DEFAULT_HPO_TERMS_SUBDIR

    # Check if terms directory exists
    if not os.path.exists(hpo_terms_dir) or not os.listdir(hpo_terms_dir):
        logging.error(f"HPO terms directory not found or empty: {hpo_terms_dir}")
        return []

    # Load all HPO terms from individual JSON files
    logging.info(f"Loading HPO terms from {hpo_terms_dir}...")
    hpo_terms = []

    # Get all JSON files in the directory
    term_files = glob.glob(os.path.join(str(hpo_terms_dir), "*.json"))
    logging.debug(f"Found {len(term_files)} term files")

    # Process each term file
    for file_path in term_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                node = json.load(f)

            # Extract the HP ID
            node_id = (
                node.get("id", "")
                .replace("http://purl.obolibrary.org/obo/HP_", "HP:")
                .replace("_", ":")
            )
            if not node_id.startswith("HP:"):
                continue

            # Extract the label
            label = node.get("lbl", "")

            # Extract definition
            definition = ""
            if (
                "meta" in node
                and "definition" in node["meta"]
                and "val" in node["meta"]["definition"]
            ):
                definition = node["meta"]["definition"]["val"]

            # Extract synonyms
            synonyms = []
            if "meta" in node and "synonyms" in node["meta"]:
                for syn_obj in node["meta"]["synonyms"]:
                    if "val" in syn_obj:
                        synonyms.append(syn_obj["val"])

            # Extract comments
            comments = []
            if "meta" in node and "comments" in node["meta"]:
                comments = [c for c in node["meta"]["comments"] if c]

            # Add to our collection
            hpo_terms.append(
                {
                    "id": node_id,
                    "label": label,
                    "definition": definition,
                    "synonyms": synonyms,
                    "comments": comments,
                }
            )

        except (OSError, json.JSONDecodeError) as e:
            logging.error(f"Error reading {file_path}: {e}")

    logging.info(f"Successfully loaded {len(hpo_terms)} HPO terms.")
    return hpo_terms


def create_hpo_documents(
    hpo_terms: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    Create descriptive documents for each HPO term suitable for embedding and indexing.

    This function generates a descriptive document for each HPO term by combining
    the label, synonyms, definition, and comments into a coherent text.

    Args:
        hpo_terms: List of HPO term dictionaries

    Returns:
        Tuple containing:
        - documents: List of text documents
        - metadatas: List of metadata dictionaries
        - ids: List of ID strings
    """
    logging.info("Creating HPO documents for indexing...")
    documents = []
    metadatas = []
    ids = []

    for term in hpo_terms:
        term_id = term["id"]
        label = term["label"]
        definition = term.get("definition", "")
        synonyms = term.get("synonyms", [])
        comments = term.get("comments", [])

        # Build document text
        doc_parts = []

        # Add label as the primary term
        doc_parts.append(f"Term: {label}")

        # Add synonyms if available
        if synonyms:
            doc_parts.append(f"Synonyms: {', '.join(synonyms)}")

        # Add definition if available
        if definition:
            doc_parts.append(f"Definition: {definition}")

        # Add comments if available
        if comments:
            doc_parts.append(f"Notes: {' '.join(comments)}")

        # Combine into a single document
        document = "\n".join(doc_parts)

        # Create metadata
        metadata = {
            "hpo_id": term_id,
            "label": label,
            "has_definition": bool(definition),
            "synonym_count": len(synonyms),
        }

        # Use the HPO ID as the document ID for easy lookup
        doc_id = term_id

        documents.append(document)
        metadatas.append(metadata)
        ids.append(doc_id)

    logging.info(f"Created {len(documents)} HPO documents for indexing")
    return documents, metadatas, ids
