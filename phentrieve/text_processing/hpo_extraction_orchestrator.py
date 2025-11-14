"""Module for orchestrating HPO term extraction from text.

This module provides functionality to extract HPO terms from text using a
pipeline-based approach with dense retrieval and optional cross-encoder reranking.
"""

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from sentence_transformers import CrossEncoder

from phentrieve.data_processing.document_creator import load_hpo_terms
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.text_attribution import get_text_attributions
from phentrieve.utils import load_translation_text

logger = logging.getLogger(__name__)


def orchestrate_hpo_extraction(
    text_chunks: list[str],
    retriever: DenseRetriever,
    num_results_per_chunk: int = 10,
    chunk_retrieval_threshold: float = 0.3,
    cross_encoder: Optional[CrossEncoder] = None,
    translation_dir_path: Optional[Path] = None,
    language: str = "en",
    reranker_mode: str = "cross-lingual",
    top_term_per_chunk: bool = False,
    min_confidence_for_aggregated: float = 0.0,
    assertion_statuses: Optional[list[str]] = None,
) -> tuple[
    list[dict[str, Any]],  # aggregated results
    list[dict[str, Any]],  # chunk results
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
        chunk_retrieval_threshold: Min similarity threshold for HPO term matches per chunk
        cross_encoder: Optional cross-encoder model for re-ranking
        translation_dir_path: Path to translation files directory
        language: Language code (e.g. 'en', 'de')
        reranker_mode: Mode for re-ranking ('monolingual' or 'cross-lingual')
        top_term_per_chunk: If True, only keep top term per chunk
        min_confidence_for_aggregated: Minimum confidence threshold for aggregated terms
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
            # Note: In monolingual reranker mode, we keep the query text in its original
            # language and translate the HPO term candidates instead (done in reranker).
            # The chunk_text is not translated here.

            # Get matches
            # Query logging is already done in the retriever class, so don't log here
            query_results = retriever.query(
                text=chunk_text,
                n_results=num_results_per_chunk,
                include_similarities=True,
            )

            # Convert to expected format
            current_hpo_matches = []
            if query_results.get("metadatas") and query_results["metadatas"][0]:
                matches_added = 0
                for i, metadata in enumerate(query_results["metadatas"][0]):
                    # Stop if we've reached the desired number of results
                    if matches_added >= num_results_per_chunk:
                        break

                    similarity = (
                        query_results["similarities"][0][i]
                        if query_results.get("similarities")
                        else 0.0
                    )
                    if similarity >= chunk_retrieval_threshold:
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
                            matches_added += 1

            # Log matches
            logger.info(
                f"Found {len(current_hpo_matches)} matches for chunk {chunk_idx + 1}"
            )

            # Print detailed extraction results for debugging
            if current_hpo_matches:
                for idx, match in enumerate(current_hpo_matches):
                    score_str = f"{match['score']:.4f}" if "score" in match else "N/A"
                    logger.info(
                        f"  [{idx + 1}] {match['id']} - {match['name']} [score: {score_str}]"
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

            # Get HPO term synonyms for text attribution
            # Create a cache for HPO term synonyms to avoid repeated loading
            hpo_synonyms_cache = {}

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

                # Get synonyms for text attribution
                synonyms = []
                if hpo_id not in hpo_synonyms_cache:
                    try:
                        # Try to get synonyms from HPO terms
                        hpo_terms = load_hpo_terms()
                        for hpo_term in hpo_terms:
                            if hpo_term.get("id") == hpo_id:
                                synonyms = hpo_term.get("synonyms", [])
                                break
                        hpo_synonyms_cache[hpo_id] = synonyms
                    except Exception as e:
                        logger.warning(f"Failed to load synonyms for {hpo_id}: {e}")
                        hpo_synonyms_cache[hpo_id] = []
                else:
                    synonyms = hpo_synonyms_cache[hpo_id]

                # Get text attributions for this term in this chunk
                attributions_in_chunk = get_text_attributions(
                    source_chunk_text=chunk_text,
                    hpo_term_label=term["name"],
                    hpo_term_synonyms=synonyms,
                    hpo_term_id=hpo_id,
                )

                # Create evidence detail for this match
                evidence_detail = {
                    "score": term["score"],
                    "chunk_idx": chunk_idx,
                    "text": chunk_text,
                    "status": term.get("assertion_status"),
                    "name": term["name"],
                    "attributions_in_chunk": attributions_in_chunk,
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
        if avg_score < min_confidence_for_aggregated:
            continue

        # Get the highest score and corresponding chunk index
        max_score = max(evidence["score"] for evidence in evidence_list)
        top_evidence_chunk_idx = next(
            evidence["chunk_idx"]
            for evidence in evidence_list
            if evidence["score"] == max_score
        )

        # Determine the most common assertion status using Counter for better handling
        # of multiple statuses with same frequency
        status_counter = Counter([e["status"] for e in evidence_list if e["status"]])
        assertion_status = None
        if status_counter:
            assertion_status = status_counter.most_common(1)[0][0]

        # Collect all text attributions from all evidence chunks
        text_attributions = []
        for evidence in evidence_list:
            chunk_idx = evidence["chunk_idx"]
            for attribution in evidence.get("attributions_in_chunk", []):
                # Add chunk_idx to each attribution
                attribution_with_chunk = attribution.copy()
                attribution_with_chunk["chunk_idx"] = chunk_idx
                text_attributions.append(attribution_with_chunk)

        # Create the aggregated term entry with enhanced information
        aggregated_term = {
            "id": hpo_id,
            "name": evidence_list[0]["name"],  # Use name from first evidence
            "score": max_score,  # Highest score
            "count": len(evidence_list),
            "evidence_count": len(evidence_list),  # Alias for count for API consistency
            "avg_score": avg_score,
            "confidence": avg_score,  # Alias for avg_score for API consistency
            "chunks": sorted({evidence["chunk_idx"] for evidence in evidence_list}),
            "top_evidence_chunk_idx": top_evidence_chunk_idx,
            "text_attributions": text_attributions,
        }

        # Always include assertion status (even if None) for consistency
        aggregated_term["assertion_status"] = assertion_status
        aggregated_term["status"] = (
            assertion_status  # Alias for assertion_status for API consistency
        )

        aggregated_results_list.append(aggregated_term)

    # Sort by score and count (descending)
    aggregated_results_list.sort(key=lambda x: (-x["avg_score"], -x["count"]))

    # Add ranks
    for idx, term in enumerate(aggregated_results_list):
        term["rank"] = idx + 1

    logger.info(
        f"Found {len(aggregated_results_list)} unique HPO terms "
        f"above threshold {min_confidence_for_aggregated}"
    )

    return (aggregated_results_list, chunk_results)
