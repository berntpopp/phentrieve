"""
Document creation module for HPO terms.

This module provides functionality for loading HPO terms and converting them
into indexable documents with appropriate metadata for vector storage.
"""

import logging
import os
from typing import Any, Optional

from phentrieve.config import DEFAULT_HPO_DB_FILENAME
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.utils import get_default_data_dir, resolve_data_path


def load_hpo_terms(data_dir_override: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Load HPO terms from SQLite database.

    Args:
        data_dir_override: Optional override for the data directory path

    Returns:
        List of dictionaries containing HPO term data with keys:
        - id: HPO term ID (e.g., "HP:0000001")
        - label: Term label/name
        - definition: Term definition text
        - synonyms: List of synonym strings
        - comments: List of comment strings

    Note:
        Returns empty list if database not found or loading fails.
    """
    # Resolve data directory path using our dynamic path resolution system
    data_dir = resolve_data_path(data_dir_override, "data_dir", get_default_data_dir)

    # Construct path to HPO database
    db_path = data_dir / DEFAULT_HPO_DB_FILENAME

    # Check if database exists
    if not os.path.exists(db_path):
        logging.error(f"HPO database not found: {db_path}")
        logging.error("Please run 'phentrieve data prepare' to generate the database.")
        return []

    # Load HPO terms from database
    logging.info(f"Loading HPO terms from database: {db_path}...")

    try:
        db = HPODatabase(db_path)
        hpo_terms = db.load_all_terms()
        db.close()

        logging.info(f"Successfully loaded {len(hpo_terms)} HPO terms from database.")
        return hpo_terms

    except Exception as e:
        logging.error(f"Error loading HPO terms from database: {e}", exc_info=True)
        return []


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
