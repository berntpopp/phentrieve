"""Module for orchestrating HPO term extraction from text.

This module provides functionality to extract HPO terms from text using a
pipeline-based approach with dense retrieval.
"""

import logging

from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing._hpo_extraction_helpers import (
    aggregate_and_rank,
    build_evidence_map,
    load_term_details,
    process_chunk_matches,
)
from phentrieve.text_processing.orchestration_result import OrchestrationResult

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
    precomputed_query_results: list[dict] | None = None,
) -> OrchestrationResult:
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
        precomputed_query_results: Optional pre-fetched per-chunk raw retrieval
            results (same shape as ``DenseRetriever.query_batch`` output, one
            entry per chunk). When provided, retrieval is skipped and
            aggregation runs over the supplied data. Used by the adaptive
            rechunker to re-aggregate over a curated mix of original and
            child-chunk raw results without re-querying.

    Returns:
        OrchestrationResult dataclass exposing ``aggregated_results``,
        ``chunk_results``, and ``raw_query_results``. Iteration and indexing
        yield the legacy 2-tuple ``(aggregated_results, chunk_results)`` for
        backward compatibility with existing call sites.
    """
    # OPTIMIZATION: Query all chunks at once using batch API (10-20x faster!)
    # This replaces the sequential query loop with a single batch query to ChromaDB.
    # If precomputed_query_results is supplied (e.g. from the adaptive
    # rechunker's re-aggregation pass), skip retrieval and use it directly.
    if precomputed_query_results is not None:
        if len(precomputed_query_results) != len(text_chunks):
            raise ValueError(
                f"precomputed_query_results length ({len(precomputed_query_results)}) "
                f"does not match text_chunks length ({len(text_chunks)})"
            )
        all_query_results = precomputed_query_results
        logger.info(
            "Using %d precomputed query results (skipping retrieval)",
            len(all_query_results),
        )
    else:
        logger.info(f"Batch querying {len(text_chunks)} chunks at once")
        all_query_results = retriever.query_batch(
            texts=text_chunks,
            n_results=num_results_per_chunk,
            include_similarities=True,
        )

    chunk_results = process_chunk_matches(
        text_chunks=text_chunks,
        all_query_results=all_query_results,
        num_results_per_chunk=num_results_per_chunk,
        chunk_retrieval_threshold=chunk_retrieval_threshold,
        top_term_per_chunk=top_term_per_chunk,
        assertion_statuses=assertion_statuses,
    )
    all_hpo_ids = {
        match["id"]
        for chunk_result in chunk_results
        for match in chunk_result.get("matches", [])
    }
    hpo_synonyms_cache, hpo_definitions_cache = load_term_details(
        all_hpo_ids=all_hpo_ids,
        include_details=include_details,
    )
    evidence_map = build_evidence_map(
        chunk_results=chunk_results,
        hpo_synonyms_cache=hpo_synonyms_cache,
    )
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

    return OrchestrationResult(
        aggregated_results=aggregated_results_list,
        chunk_results=chunk_results,
        raw_query_results=list(all_query_results),
    )
