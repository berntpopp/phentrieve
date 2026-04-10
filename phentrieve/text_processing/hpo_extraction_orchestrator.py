"""Module for orchestrating HPO term extraction from text.

This module provides functionality to extract HPO terms from text using a
pipeline-based approach with dense retrieval.
"""

import logging
from typing import Any

from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing._hpo_extraction_helpers import (
    aggregate_and_rank,
    build_evidence_map,
    load_term_details,
    process_chunk_matches,
)

logger = logging.getLogger(__name__)


def orchestrate_hpo_extraction(
    text_chunks: list[str],
    retriever: DenseRetriever,
    num_results_per_chunk: int = 10,
    chunk_retrieval_threshold: float = 0.3,
    language: str = "en",
    top_term_per_chunk: bool = False,
    min_confidence_for_aggregated: float = 0.0,
    assertion_statuses: list[str | None] | None = None,
    include_details: bool = False,
) -> tuple[
    list[dict[str, Any]],  # aggregated results
    list[dict[str, Any]],  # chunk results
]:
    """Orchestrate HPO term extraction from text.

    Process involves:
    1. Getting matches for each chunk
    2. Aggregating and deduplicating results

    Args:
        text_chunks: List of text chunks to process
        retriever: Dense retriever for HPO terms
        num_results_per_chunk: Number of results per chunk
        chunk_retrieval_threshold: Min similarity threshold for HPO term matches per chunk
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
    # Batch-query all chunks via the retriever in ONE call.
    logger.info(f"Batch querying {len(text_chunks)} chunks at once")
    all_query_results = retriever.query_batch(
        texts=text_chunks,
        n_results=num_results_per_chunk,
        include_similarities=True,
    )

    # Step 1 — Apply thresholds, top-term filter, assertion propagation.
    chunk_results = process_chunk_matches(
        text_chunks=text_chunks,
        all_query_results=all_query_results,
        num_results_per_chunk=num_results_per_chunk,
        chunk_retrieval_threshold=chunk_retrieval_threshold,
        top_term_per_chunk=top_term_per_chunk,
        assertion_statuses=assertion_statuses,
    )

    # Step 2 — Batch-load synonyms (and definitions if requested).
    all_hpo_ids: set[str] = {
        match["id"] for chunk in chunk_results for match in chunk.get("matches", [])
    }
    hpo_synonyms_cache, hpo_definitions_cache = load_term_details(
        all_hpo_ids=all_hpo_ids,
        include_details=include_details,
    )

    # Step 3 — Build evidence map with text attributions.
    evidence_map = build_evidence_map(
        chunk_results=chunk_results,
        hpo_synonyms_cache=hpo_synonyms_cache,
    )

    # Step 4 — Aggregate, rank, and render final output.
    aggregated_results_list = aggregate_and_rank(
        evidence_map=evidence_map,
        min_confidence_for_aggregated=min_confidence_for_aggregated,
        hpo_synonyms_cache=hpo_synonyms_cache,
        hpo_definitions_cache=hpo_definitions_cache,
        include_details=include_details,
    )

    logger.info(
        f"Found {len(aggregated_results_list)} unique HPO terms "
        f"above threshold {min_confidence_for_aggregated}"
    )
    return (aggregated_results_list, chunk_results)
