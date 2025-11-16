import logging
from typing import Any, Optional

from sentence_transformers import CrossEncoder

from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing.assertion_detection import (
    CombinedAssertionDetector,
)

logger = logging.getLogger(__name__)


async def execute_hpo_retrieval_for_api(
    text: str,
    language: str,
    retriever: DenseRetriever,
    num_results: int,
    similarity_threshold: float,
    enable_reranker: bool,
    cross_encoder: Optional[CrossEncoder],
    rerank_count: int,
    reranker_mode: str,
    translation_dir_path: Optional[str],
    detect_query_assertion: bool = True,
    query_assertion_language: Optional[str] = None,
    query_assertion_preference: str = "dependency",
    debug: bool = False,
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
        reranker_mode: Either "cross-lingual" or "monolingual"
        translation_dir_path: Path to translation directory for monolingual mode
        debug: Enable debug logging

    Returns:
        Dictionary with query results matching QueryResponse structure
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Processing API query with text: {text[:50]}...")
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
                f"Using language '{assertion_language}' for assertion detection"
            )

            # Create the assertion detector with the specified language
            assertion_detector = CombinedAssertionDetector(
                language=assertion_language, preference=query_assertion_preference
            )

            # Detect assertion status
            original_query_assertion_status, details = assertion_detector.detect(text)

            logger.info(
                f"Query assertion status detected: {original_query_assertion_status}"
            )
            if details:
                logger.debug(f"Assertion detection details: {details}")
        except Exception as e:
            logger.warning(f"Error in assertion detection: {e}")
            original_query_assertion_status = None
    if enable_reranker and not cross_encoder:
        logger.warning(
            "Reranking requested but no cross_encoder provided. Disabling reranking."
        )
        enable_reranker = False
    # Process as a single text segment for now (not sentence mode)
    # Could extract sentence mode logic from process_query if needed later
    segment_to_process = text.strip()
    # Perform dense retrieval
    query_results = retriever.query(
        text=segment_to_process,
        n_results=rerank_count if enable_reranker else num_results,
        include_similarities=True,
    )

    # Convert the dictionary results to a list of HPO items
    hpo_embeddings_results = []

    # Check if we have valid results
    if not query_results or not query_results.get("ids") or not query_results["ids"][0]:
        logger.info(
            f"No HPO terms found for query with threshold {similarity_threshold}"
        )
        return {
            "query_text_processed": segment_to_process,
            "header": f"No HPO terms found with similarity threshold {similarity_threshold}.",
            "results": [],
        }

    # Process the results from query into a usable format
    for i in range(len(query_results["ids"][0])):
        # Skip items below similarity threshold
        if (
            "similarities" in query_results
            and query_results["similarities"][0][i] < similarity_threshold
        ):
            continue

        # Get metadata
        metadata = (
            query_results["metadatas"][0][i] if query_results.get("metadatas") else {}
        )

        # Build the HPO item
        hpo_item = {
            "hpo_id": metadata.get("hpo_id", query_results["ids"][0][i]),
            "label": metadata.get("label", query_results["documents"][0][i]),
            "similarity": (
                query_results["similarities"][0][i]
                if "similarities" in query_results
                else None
            ),
        }

        hpo_embeddings_results.append(hpo_item)
    # Apply reranking if enabled
    if enable_reranker and cross_encoder:
        logger.debug(
            f"Reranking {len(hpo_embeddings_results)} results using {reranker_mode} mode"
        )
        try:
            # Store original ranks for reference
            for i, result in enumerate(hpo_embeddings_results):
                result["original_rank"] = i + 1

            if reranker_mode == "monolingual" and translation_dir_path:
                # Load translations for monolingual reranking
                # This would need the translation loading logic adapted from process_query
                # For now, placeholder for the translation logic
                pass

            # Create sentence pairs for the cross-encoder
            # Format: [(query_text, hpo_term_label), ...]
            sentence_pairs = [
                (segment_to_process, item["label"]) for item in hpo_embeddings_results
            ]

            # Get cross-encoder scores
            cross_encoder_scores = cross_encoder.predict(sentence_pairs)

            # Add scores to results
            for i, score in enumerate(cross_encoder_scores):
                hpo_embeddings_results[i]["cross_encoder_score"] = float(score)

            # Sort by cross-encoder score (descending)
            hpo_embeddings_results = sorted(
                hpo_embeddings_results,
                key=lambda x: x["cross_encoder_score"],
                reverse=True,
            )

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
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
