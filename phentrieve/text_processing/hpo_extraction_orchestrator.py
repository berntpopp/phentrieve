"""Module for orchestrating HPO term extraction from text.

This module provides functionality to extract HPO terms from text using a
pipeline-based approach with dense retrieval and optional cross-encoder reranking.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
from sentence_transformers import CrossEncoder

from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.utils import load_translation_text

logger = logging.getLogger(__name__)


def orchestrate_hpo_extraction(
    text_chunks: List[str],
    retriever: DenseRetriever,
    num_results_per_chunk: int = 10,
    similarity_threshold_per_chunk: float = 0.3,
    cross_encoder: Optional[CrossEncoder] = None,
    translation_dir_path: Optional[Path] = None,
    language: str = "en",
    reranker_mode: str = "cross-lingual",
    top_term_per_chunk: bool = False,
    min_confidence: float = 0.0,
    assertion_statuses: Optional[List[str]] = None,
) -> Tuple[
    List[Dict[str, Any]],  # aggregated results
    List[Dict[str, Any]],  # chunk results
]:
    """Orchestrate HPO term extraction from text.

    Process involves:
    1. Getting matches for each chunk
    2. Re-ranking matches if enabled
    3. Processing translations if needed
    4. Aggregating and deduplicating results

    Args:
        text_chunks: List of text chunks to process
        retriever: Dense retriever for HPO terms
        num_results_per_chunk: Number of results per chunk
        similarity_threshold_per_chunk: Min similarity threshold for matches
        cross_encoder: Optional cross-encoder model for re-ranking
        translation_dir_path: Path to translation files directory
        language: Language code (e.g. 'en', 'de')
        reranker_mode: Mode for re-ranking ('monolingual' or 'cross-lingual')
        top_term_per_chunk: If True, only keep top term per chunk
        min_confidence: Minimum confidence threshold for terms
        assertion_statuses: Optional list of assertion statuses per chunk

    Returns:
        Tuple containing:
        - List of aggregated HPO terms with scores and ranks
        - List of chunk-level results with matches
    """
    # Initialize results
    chunk_results = []  # Store results for each chunk
    aggregated_hpo_evidence_map = defaultdict(list)  # Group evidence by HPO ID

    # Check reranking config
    if cross_encoder and not retriever:
        logger.warning("Reranking enabled but no retriever provided")
        cross_encoder = None

    # Process chunks
    for chunk_idx, chunk_text in enumerate(text_chunks):
        try:
            # Get translation
            if (
                translation_dir_path
                and language != "en"
                and reranker_mode == "monolingual"
            ):
                try:
                    translation = load_translation_text(
                        chunk_text,
                        translation_dir_path,
                        language,
                    )
                    if translation:
                        chunk_text = translation
                except Exception as e:
                    logger.warning(f"Translation failed: {e}")

            # Get matches
            query_results = retriever.query(
                text=chunk_text,
                n_results=num_results_per_chunk,
                include_similarities=True,
            )

            # Convert to expected format
            current_hpo_matches = []
            if query_results.get("metadatas") and query_results["metadatas"][0]:
                for i, metadata in enumerate(query_results["metadatas"][0]):
                    similarity = (
                        query_results["similarities"][0][i]
                        if query_results.get("similarities")
                        else 0.0
                    )
                    if similarity >= similarity_threshold_per_chunk:
                        # Extract ID and name from metadata
                        hpo_id = metadata.get("id") or metadata.get("hpo_id")
                        name = metadata.get("label") or metadata.get("name")

                        if hpo_id and name:
                            current_hpo_matches.append(
                                {
                                    "id": hpo_id,
                                    "name": name,
                                    "score": similarity,
                                    "assertion_status": (
                                        assertion_statuses[chunk_idx]
                                        if assertion_statuses
                                        else None
                                    ),
                                }
                            )

            # Log matches
            logger.info(
                f"Found {len(current_hpo_matches)} matches"
                f" for chunk {chunk_idx + 1}"
            )

            # Re-rank if enabled
            if cross_encoder and current_hpo_matches:
                try:
                    # Prepare pairs for cross-encoder
                    pairs = [
                        (chunk_text, match["name"]) for match in current_hpo_matches
                    ]

                    # Get cross-encoder scores
                    scores = cross_encoder.predict(
                        pairs,
                        show_progress_bar=False,
                    )

                    # Add scores to candidates
                    for idx, match in enumerate(current_hpo_matches[:]):
                        match["score"] = float(scores[idx])

                    # Sort by score
                    current_hpo_matches.sort(key=lambda x: x["score"], reverse=True)
                except Exception as e:
                    logger.warning(f"Re-ranking failed: {e}")

            # Filter to top match per chunk if requested
            if top_term_per_chunk and current_hpo_matches:
                current_hpo_matches = [current_hpo_matches[0]]

            # Add matches to results
            chunk_results.append(
                {
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk_text,
                    "matches": current_hpo_matches,
                }
            )

            # Collect evidence for each HPO term from this chunk
            for term in current_hpo_matches:
                hpo_id = term["id"]
                # Create evidence detail for this match
                evidence_detail = {
                    "score": term["score"],
                    "chunk_id": chunk_idx,
                    "text": chunk_text,
                    "status": term.get("assertion_status"),
                    "name": term["name"],
                }
                # Add to evidence map keyed by HPO ID
                aggregated_hpo_evidence_map[hpo_id].append(evidence_detail)

        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_idx + 1}: {e}")
            continue

    # Process the aggregated evidence map into the final list format
    aggregated_results_list = []

    # Process each HPO term's evidence
    for hpo_id, evidence_list in aggregated_hpo_evidence_map.items():
        # Skip if no evidence (shouldn't happen, but just to be safe)
        if not evidence_list:
            continue

        # Calculate stats from evidence
        total_score = sum(evidence["score"] for evidence in evidence_list)
        avg_score = total_score / len(evidence_list)

        # Skip terms below confidence threshold
        if avg_score < min_confidence:
            continue

        # Determine the most common assertion status
        assertion_statuses = [e["status"] for e in evidence_list if e["status"]]
        assertion_status = None
        if assertion_statuses:
            assertion_status = max(
                set(assertion_statuses), key=assertion_statuses.count
            )

        # Create the aggregated term entry
        aggregated_term = {
            "id": hpo_id,
            "name": evidence_list[0]["name"],  # Use name from first evidence
            "score": max(
                evidence["score"] for evidence in evidence_list
            ),  # Highest score
            "count": len(evidence_list),
            "avg_score": avg_score,
            "chunks": sorted(
                list(set(evidence["chunk_id"] for evidence in evidence_list))
            ),
        }

        # Add assertion status if available
        if assertion_status:
            aggregated_term["assertion_status"] = assertion_status

        aggregated_results_list.append(aggregated_term)

    # Sort by score and count (descending)
    aggregated_results_list.sort(key=lambda x: (-x["avg_score"], -x["count"]))

    # Add ranks
    for idx, term in enumerate(aggregated_results_list):
        term["rank"] = idx + 1

    logger.info(
        f"Found {len(aggregated_results_list)} unique HPO terms "
        f"above threshold {min_confidence}"
    )

    return (aggregated_results_list, chunk_results)
