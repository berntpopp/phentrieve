"""
Multi-vector document creation module for HPO terms.

This module provides functionality for creating separate embedding documents
for each component of an HPO term (label, synonyms, definition), enabling
fine-grained semantic matching with configurable aggregation strategies.

See issue #136 for design details.
"""

import logging
from typing import Any


def create_multi_vector_documents(
    hpo_terms: list[dict[str, Any]],
    include_label: bool = True,
    include_synonyms: bool = True,
    include_definition: bool = True,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    Create per-component documents for multi-vector HPO term indexing.

    Each HPO term generates multiple documents:
    - One for the label
    - One for each synonym (individual vectors)
    - One for the definition (if present)

    Args:
        hpo_terms: List of HPO term dictionaries with keys:
            - id: HPO term ID (e.g., "HP:0000001")
            - label: Term label/name
            - definition: Term definition text (optional)
            - synonyms: List of synonym strings (optional)
        include_label: Whether to create label vectors (default: True)
        include_synonyms: Whether to create synonym vectors (default: True)
        include_definition: Whether to create definition vectors (default: True)

    Returns:
        Tuple containing:
        - documents: List of text documents (one per component)
        - metadatas: List of metadata dictionaries with component info
        - ids: List of document IDs in format "{hpo_id}__{component}__{index}"

    Example:
        For HP:0001250 "Seizure" with synonyms ["Fits", "Convulsions"]:
        - Document: "Seizure", ID: "HP:0001250__label__0"
        - Document: "Fits", ID: "HP:0001250__synonym__0"
        - Document: "Convulsions", ID: "HP:0001250__synonym__1"
        - Document: "A seizure is...", ID: "HP:0001250__definition__0"
    """
    logging.info("Creating multi-vector HPO documents for indexing...")

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    label_count = 0
    synonym_count = 0
    definition_count = 0

    for term in hpo_terms:
        term_id = term["id"]
        label = term["label"]
        definition = term.get("definition", "")
        synonyms = term.get("synonyms", [])

        # Create label document
        if include_label and label:
            documents.append(label)
            metadatas.append(
                {
                    "hpo_id": term_id,
                    "component": "label",
                    "component_index": 0,
                    "label": label,
                }
            )
            ids.append(f"{term_id}__label__0")
            label_count += 1

        # Create individual synonym documents
        if include_synonyms and synonyms:
            for idx, synonym in enumerate(synonyms):
                if synonym:  # Skip empty synonyms
                    documents.append(synonym)
                    metadatas.append(
                        {
                            "hpo_id": term_id,
                            "component": "synonym",
                            "component_index": idx,
                            "label": label,
                            "synonym_text": synonym,
                        }
                    )
                    ids.append(f"{term_id}__synonym__{idx}")
                    synonym_count += 1

        # Create definition document
        if include_definition and definition:
            documents.append(definition)
            metadatas.append(
                {
                    "hpo_id": term_id,
                    "component": "definition",
                    "component_index": 0,
                    "label": label,
                }
            )
            ids.append(f"{term_id}__definition__0")
            definition_count += 1

    logging.info(
        f"Created {len(documents)} multi-vector documents: "
        f"{label_count} labels, {synonym_count} synonyms, {definition_count} definitions"
    )

    return documents, metadatas, ids


def get_component_stats(hpo_terms: list[dict[str, Any]]) -> dict[str, int]:
    """
    Get statistics about HPO term components for planning index size.

    Args:
        hpo_terms: List of HPO term dictionaries

    Returns:
        Dictionary with counts:
        - total_terms: Number of HPO terms
        - total_labels: Number of labels (same as total_terms)
        - total_synonyms: Total number of synonyms across all terms
        - total_definitions: Number of terms with definitions
        - estimated_documents: Total documents for multi-vector index
    """
    total_terms = len(hpo_terms)
    total_synonyms = sum(len(term.get("synonyms", [])) for term in hpo_terms)
    total_definitions = sum(1 for term in hpo_terms if term.get("definition"))

    return {
        "total_terms": total_terms,
        "total_labels": total_terms,
        "total_synonyms": total_synonyms,
        "total_definitions": total_definitions,
        "estimated_documents": total_terms + total_synonyms + total_definitions,
    }
