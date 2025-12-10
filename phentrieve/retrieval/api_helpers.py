import logging
from typing import Any, Optional

from sentence_transformers import CrossEncoder

from phentrieve.config import (
    DEFAULT_AGGREGATION_STRATEGY,
    DEFAULT_DENSE_TRUST_THRESHOLD,
)
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.reranker import protected_dense_rerank
from phentrieve.text_processing.assertion_detection import (
    CombinedAssertionDetector,
)
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)


def _convert_single_vector_results(
    query_results: dict[str, Any],
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """
    Convert single-vector ChromaDB results to candidate list format.

    Args:
        query_results: Raw ChromaDB query results
        similarity_threshold: Minimum similarity score to include

    Returns:
        List of candidate dicts with hpo_id, label, similarity, comparison_text
    """
    candidates: list[dict[str, Any]] = []

    if not query_results or not query_results.get("ids") or not query_results["ids"][0]:
        return candidates

    for i in range(len(query_results["ids"][0])):
        # Skip items below similarity threshold
        similarity = (
            query_results["similarities"][0][i]
            if "similarities" in query_results
            else None
        )
        if similarity is not None and similarity < similarity_threshold:
            continue

        metadata = (
            query_results["metadatas"][0][i] if query_results.get("metadatas") else {}
        )
        label_text = metadata.get("label", query_results["documents"][0][i])

        candidates.append(
            {
                "hpo_id": metadata.get("hpo_id", query_results["ids"][0][i]),
                "label": label_text,
                "similarity": similarity,
                "comparison_text": label_text,
            }
        )

    return candidates


def _convert_multi_vector_results(
    multi_vector_results: list[dict[str, Any]],
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """
    Convert multi-vector aggregated results to candidate list format.

    Args:
        multi_vector_results: Aggregated results from query_multi_vector()
        similarity_threshold: Minimum similarity score to include

    Returns:
        List of candidate dicts with hpo_id, label, similarity, comparison_text,
        and component_scores
    """
    candidates: list[dict[str, Any]] = []

    for result in multi_vector_results:
        similarity = result.get("similarity", 0.0)
        if similarity < similarity_threshold:
            continue

        candidates.append(
            {
                "hpo_id": result["hpo_id"],
                "label": result["label"],
                "similarity": similarity,
                "comparison_text": result["label"],
                "component_scores": result.get("component_scores"),
            }
        )

    return candidates


async def execute_hpo_retrieval_for_api(
    text: str,
    language: str,
    retriever: DenseRetriever,
    num_results: int,
    similarity_threshold: float,
    enable_reranker: bool,
    cross_encoder: Optional[CrossEncoder],
    rerank_count: int,
    include_details: bool = False,
    detect_query_assertion: bool = True,
    query_assertion_language: Optional[str] = None,
    query_assertion_preference: str = "dependency",
    debug: bool = False,
    # Multi-vector parameters (Issue #136)
    multi_vector: bool = False,
    aggregation_strategy: str = DEFAULT_AGGREGATION_STRATEGY,
    component_weights: Optional[dict[str, float]] = None,
    custom_formula: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute HPO term retrieval for API requests.

    This function adapts the core logic from process_query and orchestrate_query
    for API usage, removing CLI outputs and returning structured data.

    Args:
        text: The clinical text to process and query for HPO terms
        language: ISO 639-1 language code (e.g., 'en', 'de')
        retriever: Initialized DenseRetriever instance
        num_results: Number of HPO terms to return
        similarity_threshold: Minimum similarity score for results
        enable_reranker: Whether to apply cross-encoder reranking
        cross_encoder: Optional CrossEncoder instance for reranking
        rerank_count: Number of top dense results to rerank
        include_details: Include HPO term definitions and synonyms in results
        detect_query_assertion: Enable assertion detection on query text
        query_assertion_language: Language for assertion detection
        query_assertion_preference: Assertion detection strategy
        debug: Enable debug logging
        multi_vector: Use multi-vector index with aggregation (deduplicates results)
        aggregation_strategy: Strategy for combining component scores
        component_weights: Weights for 'all_weighted' strategy
        custom_formula: Formula for 'custom' strategy

    Returns:
        Dictionary with query results matching QueryResponse structure
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Processing API query with text: %s...", _sanitize(text[:50]))
    # Validate input
    if not text or not text.strip():
        return {
            "query_text_processed": text,
            "header": "Error: Empty query text",
            "results": [],
            "original_query_assertion_status": None,
        }

    # Detect assertion status if enabled
    original_query_assertion_status = None
    if detect_query_assertion:
        # Use the explicitly provided assertion language or fallback to the query language
        assertion_language = query_assertion_language or language
        try:
            # Log the language being used for assertion detection
            logger.info(
                "Using language '%s' for assertion detection",
                _sanitize(assertion_language),
            )

            # Create the assertion detector with the specified language
            assertion_detector = CombinedAssertionDetector(
                language=assertion_language, preference=query_assertion_preference
            )

            # Detect assertion status
            original_query_assertion_status, details = assertion_detector.detect(text)

            logger.info(
                "Query assertion status detected: %s",
                _sanitize(original_query_assertion_status),
            )
            if details:
                logger.debug("Assertion detection details: %s", _sanitize(details))
        except Exception as e:
            logger.warning("Error in assertion detection: %s", _sanitize(e))
            original_query_assertion_status = None
    if enable_reranker and not cross_encoder:
        logger.warning(
            "Reranking requested but no cross_encoder provided. Disabling reranking."
        )
        enable_reranker = False
    # Process as a single text segment for now (not sentence mode)
    # Could extract sentence mode logic from process_query if needed later
    segment_to_process = text.strip()

    # Determine query method based on multi_vector flag and index type
    # Multi-vector mode uses aggregation to deduplicate results by HPO ID
    n_results_for_query = rerank_count if enable_reranker else num_results

    if multi_vector:
        # Detect index type to ensure we're using a multi-vector index
        index_type = retriever.detect_index_type()
        if index_type != "multi_vector":
            logger.warning(
                "Multi-vector mode requested but index type is '%s'. "
                "Falling back to single-vector query.",
                _sanitize(index_type),
            )
            multi_vector = False

    if multi_vector:
        # Use multi-vector query with aggregation (deduplicates by HPO ID)
        logger.debug(
            "Using multi-vector query with aggregation strategy: %s",
            _sanitize(aggregation_strategy),
        )
        multi_vector_results = retriever.query_multi_vector(
            text=segment_to_process,
            n_results=n_results_for_query,
            aggregation_strategy=aggregation_strategy,
            component_weights=component_weights,
            custom_formula=custom_formula,
        )

        # Check for empty results
        if not multi_vector_results:
            logger.info(
                "No HPO terms found for query with threshold %s",
                _sanitize(similarity_threshold),
            )
            return {
                "query_text_processed": segment_to_process,
                "header": f"No HPO terms found with similarity threshold {similarity_threshold}.",
                "results": [],
                "original_query_assertion_status": (
                    original_query_assertion_status.value
                    if original_query_assertion_status
                    else None
                ),
            }

        # Convert multi-vector results to candidate list format
        hpo_embeddings_results = _convert_multi_vector_results(
            multi_vector_results, similarity_threshold
        )
    else:
        # Use single-vector query (original behavior)
        query_results = retriever.query(
            text=segment_to_process,
            n_results=n_results_for_query,
            include_similarities=True,
        )

        # Check for empty results
        if (
            not query_results
            or not query_results.get("ids")
            or not query_results["ids"][0]
        ):
            logger.info(
                "No HPO terms found for query with threshold %s",
                _sanitize(similarity_threshold),
            )
            return {
                "query_text_processed": segment_to_process,
                "header": f"No HPO terms found with similarity threshold {similarity_threshold}.",
                "results": [],
                "original_query_assertion_status": (
                    original_query_assertion_status.value
                    if original_query_assertion_status
                    else None
                ),
            }

        # Convert single-vector results to candidate list format
        hpo_embeddings_results = _convert_single_vector_results(
            query_results, similarity_threshold
        )
    # Apply reranking if enabled
    if enable_reranker and cross_encoder:
        logger.debug(
            "Reranking %s results with protected retrieval",
            _sanitize(len(hpo_embeddings_results)),
        )
        try:
            # Map "similarity" field to "bi_encoder_score" for protected_dense_rerank()
            for item in hpo_embeddings_results:
                item["bi_encoder_score"] = item["similarity"]

            # Use protected two-stage retrieval approach
            # This preserves high-confidence dense matches while refining uncertain candidates
            hpo_embeddings_results = protected_dense_rerank(
                query=segment_to_process,
                candidates=hpo_embeddings_results,
                cross_encoder_model=cross_encoder,
                trust_threshold=DEFAULT_DENSE_TRUST_THRESHOLD,
            )

        except Exception as e:
            logger.error("Error during reranking: %s", _sanitize(e))
            # Continue with dense retrieval results if reranking fails

    # Limit to requested number of results
    if len(hpo_embeddings_results) > num_results:
        hpo_embeddings_results = hpo_embeddings_results[:num_results]

    # Format results as HPOResultItem compatible structure
    formatted_results = []
    for item in hpo_embeddings_results:
        result_item = {
            "hpo_id": item["hpo_id"],
            "label": item["label"],
            "similarity": item["similarity"],
        }

        # Add reranking info if available
        if "cross_encoder_score" in item:
            result_item["cross_encoder_score"] = item["cross_encoder_score"]
        if "original_rank" in item:
            result_item["original_rank"] = item["original_rank"]

        formatted_results.append(result_item)

    # Enrich with HPO term details if requested (definitions and synonyms)
    if include_details:
        from phentrieve.retrieval.details_enrichment import enrich_results_with_details

        formatted_results = enrich_results_with_details(formatted_results)

    # Create result dictionary without using language as key to avoid the unhashable slice error
    result_dict = {
        "query_text_processed": segment_to_process,
        "results": formatted_results,
        "original_query_assertion_status": (
            original_query_assertion_status.value
            if original_query_assertion_status
            else None
        ),
    }

    return result_dict
