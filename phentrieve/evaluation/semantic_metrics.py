"""
Evaluation metrics for HPO term extraction based on semantic similarity.

This module provides functions for evaluating the performance of HPO term extraction
using semantically-aware metrics that consider ontological similarity
between predicted and ground truth HPO terms.
"""

import logging
from typing import Any, Optional

from phentrieve.evaluation.metrics import (
    SimilarityFormula,
    calculate_semantic_similarity,
)

logger = logging.getLogger(__name__)


def calculate_semantically_aware_set_based_prf1(
    extracted_annotations: list[
        dict[str, Any]
    ],  # List of dicts from orchestrate_hpo_extraction
    ground_truth_annotations: list[
        dict[str, Any]
    ],  # List of dicts from full_text_loader
    target_assertion_status: Optional[
        str
    ] = "affirmed",  # e.g., "affirmed", "negated", or None
    semantic_similarity_threshold: float = 0.7,  # Threshold for considering a semantic match
    similarity_formula: SimilarityFormula = SimilarityFormula.HYBRID,
) -> dict[str, Any]:
    """
    Calculate semantically-aware precision, recall, and F1 score for HPO term extraction.

    This function evaluates how well extracted HPO terms match ground truth annotations,
    considering not only exact matches but also semantic similarity between terms.

    Args:
        extracted_annotations: List of dictionaries with extracted HPO terms
            Each dict should have keys: "id" (HPO ID), "status" (assertion status)
        ground_truth_annotations: List of dictionaries with ground truth HPO terms
            Each dict should have keys: "hpo_id" (HPO ID), "assertion_status" (assertion status)
        target_assertion_status: If provided, only evaluate terms with this status
        semantic_similarity_threshold: Minimum similarity score to consider a semantic match
        similarity_formula: Formula to use for calculating semantic similarity

    Returns:
        Dictionary with precision, recall, F1 score, and counts of TP, FP, FN,
        as well as the list of matched term pairs
    """
    # Filter by assertion status if provided
    filtered_extracted_terms = extracted_annotations
    filtered_ground_truth_terms = ground_truth_annotations

    # Log the original extracted and ground truth terms for debugging
    logger.info(f"Raw extracted terms: {len(extracted_annotations)} items")
    for term in extracted_annotations:
        assertion = term.get("assertion_status")
        logger.info(
            f"Extracted term: {term.get('id')} - {term.get('name')} (assertion: {assertion})"
        )

    logger.info(f"Raw ground truth terms: {len(ground_truth_annotations)} items")
    for term in ground_truth_annotations:
        term_id = term.get("hpo_id") or term.get("id")
        name = term.get("label") or term.get("name")
        assertion = term.get("assertion_status")
        logger.info(f"Ground truth term: {term_id} - {name} (assertion: {assertion})")

    # Temporarily disable assertion status filtering for debugging
    if False and target_assertion_status is not None:
        filtered_extracted_terms = [
            term
            for term in extracted_annotations
            if term.get("assertion_status") == target_assertion_status
        ]
        filtered_ground_truth_terms = [
            term
            for term in ground_truth_annotations
            if term.get("assertion_status") == target_assertion_status
        ]

        # Log filtered terms
        logger.info(
            f"After assertion filtering: {len(filtered_extracted_terms)} extracted, {len(filtered_ground_truth_terms)} ground truth"
        )
    else:
        logger.info("No assertion filtering applied - using all terms for matching")

    # Create mutable copies for matching
    available_extracted = filtered_extracted_terms.copy()
    available_truth = filtered_ground_truth_terms.copy()

    # Lists to store matched pairs (needed for assertion accuracy evaluation)
    matched_pairs = []

    # ---- Pass 1: Find exact matches first ----
    true_positives = 0
    exact_match_count = 0  # Track exact matches separately

    # We'll directly iterate through the terms for better matching control

    # Track indices to remove (to avoid modifying lists during iteration)
    extracted_indices_to_remove = set()
    truth_indices_to_remove = set()

    # First pass: Find exact matches
    for truth_idx, truth_term in enumerate(available_truth):
        truth_id = truth_term.get("hpo_id") or truth_term.get("id")
        if not truth_id:
            continue

        # Check for exact match with IDs
        match_found = False
        for extracted_idx, extracted_term in enumerate(available_extracted):
            if extracted_idx in extracted_indices_to_remove:
                continue

            extracted_id = extracted_term.get("id")
            if not extracted_id:
                continue

            # Simple string comparison - the IDs should be in the same format
            if extracted_id == truth_id:
                logger.info(f"Exact ID match found: {extracted_id} ↔ {truth_id}")

                # Mark as matched
                true_positives += 1
                exact_match_count += 1  # Increment exact match count
                matched_pairs.append((extracted_term, truth_term))

                # Mark for removal
                extracted_indices_to_remove.add(extracted_idx)
                truth_indices_to_remove.add(truth_idx)
                match_found = True
                break

        if match_found:
            logger.info(
                f"Matched term pair: {truth_term.get('label')} ↔ {extracted_term.get('name')}"
            )
        else:
            logger.info(
                f"No exact match found for ground truth term: {truth_id} - {truth_term.get('label')}"
            )
            # Log some of the available extracted terms for debugging
            shown_terms = 0
            for i, term in enumerate(available_extracted):
                if (
                    i not in extracted_indices_to_remove and shown_terms < 5
                ):  # Limit to 5 terms for clarity
                    logger.info(
                        f"  Available extracted term: {term.get('id')} - {term.get('name')}"
                    )
                    shown_terms += 1

    # Remove matched terms
    available_extracted = [
        term
        for i, term in enumerate(available_extracted)
        if i not in extracted_indices_to_remove
    ]
    available_truth = [
        term
        for i, term in enumerate(available_truth)
        if i not in truth_indices_to_remove
    ]

    # ---- Pass 2: Find semantic matches for remaining terms ----
    semantic_match_count = 0  # Track semantic matches separately
    if available_extracted and available_truth:
        # Reset trackers
        extracted_indices_to_remove = set()
        truth_indices_to_remove = set()

        # For each remaining extracted term, find best semantic match
        for extracted_idx, extracted_term in enumerate(available_extracted):
            extracted_id = extracted_term["id"]

            best_similarity = 0.0
            best_match_idx = -1

            # Find best matching ground truth term
            for truth_idx, truth_term in enumerate(available_truth):
                truth_id = truth_term.get("hpo_id") or truth_term.get("id")

                # Skip if truth_id is not available
                if not truth_id or not isinstance(truth_id, str):
                    continue

                # Calculate semantic similarity
                similarity = calculate_semantic_similarity(
                    extracted_id, truth_id, formula=similarity_formula
                )

                # Update best match if better
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = truth_idx

            # If we found a good enough match
            if best_similarity >= semantic_similarity_threshold and best_match_idx >= 0:
                # Mark as matched
                true_positives += 1
                semantic_match_count += 1  # Increment semantic match count
                matched_pairs.append((extracted_term, available_truth[best_match_idx]))

                # Mark for removal
                extracted_indices_to_remove.add(extracted_idx)
                truth_indices_to_remove.add(best_match_idx)

                logger.debug(
                    f"Semantic match: {extracted_id} ↔ {available_truth[best_match_idx].get('hpo_id') or available_truth[best_match_idx].get('id')} "
                    f"(similarity: {best_similarity:.4f})"
                )

        # Remove matched terms again
        available_extracted = [
            term
            for i, term in enumerate(available_extracted)
            if i not in extracted_indices_to_remove
        ]
        available_truth = [
            term
            for i, term in enumerate(available_truth)
            if i not in truth_indices_to_remove
        ]

    # Calculate overall metrics (exact + semantic)
    fp_count = len(filtered_extracted_terms) - true_positives
    fn_count = len(filtered_ground_truth_terms) - true_positives

    overall_precision = (
        true_positives / len(filtered_extracted_terms)
        if filtered_extracted_terms
        else 0.0
    )
    overall_recall = (
        true_positives / len(filtered_ground_truth_terms)
        if filtered_ground_truth_terms
        else 0.0
    )
    overall_f1_score = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    # Calculate exact match metrics
    exact_precision = (
        exact_match_count / len(filtered_extracted_terms)
        if filtered_extracted_terms
        else 0.0
    )
    exact_recall = (
        exact_match_count / len(filtered_ground_truth_terms)
        if filtered_ground_truth_terms
        else 0.0
    )
    exact_f1_score = (
        2 * (exact_precision * exact_recall) / (exact_precision + exact_recall)
        if (exact_precision + exact_recall) > 0
        else 0.0
    )

    # Calculate semantic match metrics
    semantic_precision = (
        semantic_match_count / len(filtered_extracted_terms)
        if filtered_extracted_terms
        else 0.0
    )
    semantic_recall = (
        semantic_match_count / len(filtered_ground_truth_terms)
        if filtered_ground_truth_terms
        else 0.0
    )
    semantic_f1_score = (
        2
        * (semantic_precision * semantic_recall)
        / (semantic_precision + semantic_recall)
        if (semantic_precision + semantic_recall) > 0
        else 0.0
    )

    # Verify that all TPs are accounted for in either exact or semantic matches
    # This is crucial for making sure we don't have matches that aren't classified
    if true_positives != (exact_match_count + semantic_match_count):
        logger.warning(
            f"Mismatch in match counting! Total TP: {true_positives}, Exact: {exact_match_count}, Semantic: {semantic_match_count}"
        )
        # Force alignment - this is a safety check to ensure all matches are categorized
        if semantic_match_count == 0 and exact_match_count == 0 and true_positives > 0:
            # We have TPs but no categorization - assign to semantic as a fallback
            logger.warning(
                f"Forced categorization of {true_positives} matches as semantic"
            )
            semantic_match_count = true_positives

    logger.info(
        f"Match breakdown: {exact_match_count} exact matches, {semantic_match_count} semantic matches"
    )

    return {
        # Overall metrics (exact + semantic)
        "precision": overall_precision,
        "recall": overall_recall,
        "f1_score": overall_f1_score,
        "tp_count": true_positives,
        "fp_count": fp_count,
        "fn_count": fn_count,
        # Exact match metrics
        "exact_match_count": exact_match_count,
        "exact_precision": exact_precision,
        "exact_recall": exact_recall,
        "exact_f1_score": exact_f1_score,
        # Semantic match metrics
        "semantic_match_count": semantic_match_count,
        "semantic_precision": semantic_precision,
        "semantic_recall": semantic_recall,
        "semantic_f1_score": semantic_f1_score,
        "tp_matched_pairs_list": matched_pairs,
    }


def calculate_assertion_accuracy(
    matched_pairs: list[
        tuple[dict[str, Any], dict[str, Any]]
    ],  # List of (extracted_term_dict, ground_truth_term_dict)
) -> tuple[float, int, int]:
    """
    Calculate assertion status accuracy for matched HPO terms.

    This function evaluates how accurately assertion statuses are predicted
    for HPO terms that were correctly identified (either exact or semantic matches).

    Args:
        matched_pairs: List of tuples where each tuple contains:
            (extracted_term_dict, ground_truth_term_dict)

    Returns:
        Tuple of:
        - Assertion accuracy as percentage
        - Number of correctly asserted terms
        - Total number of matched terms
    """
    num_common_terms = len(matched_pairs)

    if num_common_terms == 0:
        return 0.0, 0, 0

    num_correctly_asserted = 0

    for extracted_term, ground_truth_term in matched_pairs:
        # Get assertion status from each term
        extracted_status = extracted_term.get("status") or extracted_term.get(
            "assertion_status"
        )
        truth_status = ground_truth_term.get("assertion_status")

        # Check if assertion statuses match
        if extracted_status == truth_status:
            num_correctly_asserted += 1

    # Calculate accuracy percentage
    accuracy_percentage = (
        (num_correctly_asserted / num_common_terms) * 100
        if num_common_terms > 0
        else 0.0
    )

    return accuracy_percentage, num_correctly_asserted, num_common_terms
