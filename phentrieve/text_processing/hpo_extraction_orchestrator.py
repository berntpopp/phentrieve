"""Module for orchestrating HPO term extraction from text.

This module provides functionality to extract HPO terms from text using a
pipeline-based approach with dense retrieval and optional cross-encoder reranking.
"""

import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

# NOTE: CrossEncoder is only imported for type hints (TYPE_CHECKING).
# This module receives CrossEncoder instances but doesn't create them,
# so we avoid the 18+ second import cost at module load time.
if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from phentrieve.config import DEFAULT_HPO_DB_FILENAME
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.text_attribution import get_text_attributions
from phentrieve.utils import get_default_data_dir, resolve_data_path

logger = logging.getLogger(__name__)


def orchestrate_hpo_extraction(
    text_chunks: list[str],
    retriever: DenseRetriever,
    num_results_per_chunk: int = 10,
    chunk_retrieval_threshold: float = 0.3,
    cross_encoder: Optional["CrossEncoder"] = None,
    language: str = "en",
    top_term_per_chunk: bool = False,
    min_confidence_for_aggregated: float = 0.0,
    assertion_statuses: Optional[list[str | None]] = None,
    include_details: bool = False,
) -> tuple[
    list[dict[str, Any]],  # aggregated results
    list[dict[str, Any]],  # chunk results
]:
    """Orchestrate HPO term extraction from text.

    Process involves:
    1. Getting matches for each chunk
    2. Re-ranking matches if enabled
    3. Aggregating and deduplicating results

    Args:
        text_chunks: List of text chunks to process
        retriever: Dense retriever for HPO terms
        num_results_per_chunk: Number of results per chunk
        chunk_retrieval_threshold: Min similarity threshold for HPO term matches per chunk
        cross_encoder: Optional cross-encoder model for re-ranking
        language: Language code (e.g. 'en', 'de')
        top_term_per_chunk: If True, only keep top term per chunk
        min_confidence_for_aggregated: Minimum confidence threshold for aggregated terms
        assertion_statuses: Optional list of assertion statuses per chunk
        include_details: If True, include HPO term definitions and synonyms in results

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

    # OPTIMIZATION: Query all chunks at once using batch API (10-20x faster!)
    # This replaces the sequential query loop with a single batch query to ChromaDB
    logger.info(f"Batch querying {len(text_chunks)} chunks at once")
    all_query_results = retriever.query_batch(
        texts=text_chunks,
        n_results=num_results_per_chunk,
        include_similarities=True,
    )

    # Process chunks with pre-fetched results
    for chunk_idx, chunk_text in enumerate(text_chunks):
        try:
            # Get pre-fetched results for this chunk from the batch query
            query_results = all_query_results[chunk_idx]

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
                    # Handle different output formats from various cross-encoder models:
                    # - NLI models return arrays: [P(entailment), P(neutral), P(contradiction)]
                    # - Rerankers return single float: relevance_score
                    for idx, match in enumerate(current_hpo_matches[:]):
                        raw_score = scores[idx]
                        if (
                            isinstance(raw_score, (list, np.ndarray))
                            and len(raw_score) > 1
                        ):
                            # NLI model: use entailment probability (index 0)
                            # Note: Suboptimal for semantic relevance - dedicated reranker recommended
                            match["score"] = float(raw_score[0])
                        else:
                            # Proper reranker: single relevance score
                            match["score"] = float(raw_score)

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

        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_idx + 1}: {e}")
            continue

    # OPTIMIZATION: Batch-load ALL synonyms (and optionally definitions) in ONE database query
    # This replaces the inefficient per-term loading that was loading all 19,534 terms repeatedly
    logger.info("Batch-loading synonyms for all HPO terms")
    hpo_synonyms_cache = {}
    hpo_definitions_cache = {}  # Only populated when include_details=True

    # Collect all unique HPO IDs from all chunks
    all_hpo_ids: set[str] = set()
    for chunk_result in chunk_results:
        matches: Any = chunk_result.get("matches", [])
        for match in matches:
            all_hpo_ids.add(match["id"])

    # Load synonyms (and optionally definitions) for all HPO IDs in ONE batch query
    if all_hpo_ids:
        try:
            data_dir = resolve_data_path(None, "data_dir", get_default_data_dir)
            db_path = data_dir / DEFAULT_HPO_DB_FILENAME

            if db_path.exists():
                logger.debug(
                    f"Loading {'synonyms and definitions' if include_details else 'synonyms'} "
                    f"for {len(all_hpo_ids)} unique HPO terms"
                )
                db = HPODatabase(db_path)
                terms_map = db.get_terms_by_ids(list(all_hpo_ids))
                db.close()

                # Build synonyms cache (always) and definitions cache (when requested)
                for hpo_id, term_data in terms_map.items():
                    hpo_synonyms_cache[hpo_id] = term_data.get("synonyms", [])
                    if include_details:
                        hpo_definitions_cache[hpo_id] = term_data.get("definition")

                logger.info(
                    f"Loaded {'synonyms and definitions' if include_details else 'synonyms'} "
                    f"for {len(hpo_synonyms_cache)} HPO terms"
                )
            else:
                logger.warning(
                    f"HPO database not found: {db_path}. Skipping synonym lookup."
                )
        except Exception as e:
            logger.warning(f"Failed to batch-load HPO term data: {e}")

    # Now collect evidence with pre-loaded synonyms
    for chunk_result in chunk_results:
        result_chunk_idx: Any = chunk_result["chunk_idx"]
        result_chunk_text: Any = chunk_result["chunk_text"]
        result_matches: Any = chunk_result["matches"]

        for term in result_matches:
            hpo_id = term["id"]

            # Get synonyms from cache (no database load!)
            synonyms = hpo_synonyms_cache.get(hpo_id, [])

            # Get text attributions for this term in this chunk
            attributions_in_chunk = get_text_attributions(
                source_chunk_text=result_chunk_text,
                hpo_term_label=term["name"],
                hpo_term_synonyms=synonyms,
                hpo_term_id=hpo_id,
            )

            # Create evidence detail for this match
            evidence_detail = {
                "score": term["score"],
                "chunk_idx": result_chunk_idx,
                "text": result_chunk_text,
                "status": term.get("assertion_status"),
                "name": term["name"],
                "attributions_in_chunk": attributions_in_chunk,
            }
            # Add to evidence map keyed by HPO ID
            aggregated_hpo_evidence_map[hpo_id].append(evidence_detail)

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

        # Include definition and synonyms when requested (for API include_details=True)
        if include_details:
            aggregated_term["definition"] = hpo_definitions_cache.get(hpo_id)
            aggregated_term["synonyms"] = hpo_synonyms_cache.get(hpo_id, [])

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
