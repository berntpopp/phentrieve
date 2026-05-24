"""Shared retrieval utilities.

Functions used across multiple retrieval and evaluation modules.
"""

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


def convert_multi_vector_to_chromadb_format(
    multi_vector_results: list[dict[str, Any]],
    include_similarities: bool = False,
) -> dict[str, Any]:
    """
    Convert multi-vector aggregated results to ChromaDB-style format.

    This allows reusing existing format_results() and output formatters.

    Args:
        multi_vector_results: List of aggregated results from query_multi_vector()
        include_similarities: Whether to include nested similarity scores

    Returns:
        Dictionary in ChromaDB query result format
    """
    if not multi_vector_results:
        empty: dict[str, Any] = {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }
        if include_similarities:
            empty["similarities"] = [[]]
        return empty

    ids = []
    metadatas = []
    documents = []
    distances = []
    similarities = []

    for result in multi_vector_results:
        hpo_id = result["hpo_id"]
        ids.append(hpo_id)
        metadata = {
            "hpo_id": hpo_id,
            "label": result.get("label", ""),
        }
        if "component_scores" in result:
            metadata["component_scores"] = result["component_scores"]
        if "matched_component" in result:
            metadata["matched_component"] = result["matched_component"]
        if "matched_text" in result:
            metadata["matched_text"] = result["matched_text"]
        metadatas.append(metadata)
        documents.append(result.get("label", ""))

        similarity = float(result.get("similarity") or 0.0)
        similarities.append(similarity)
        distances.append(1.0 - similarity)

    converted: dict[str, Any] = {
        "ids": [ids],
        "metadatas": [metadatas],
        "documents": [documents],
        "distances": [distances],
    }
    if include_similarities:
        converted["similarities"] = [similarities]
    return converted


def query_chunk_candidates(
    *,
    retriever: Any,
    text_chunks: list[str],
    n_results: int,
    include_similarities: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve per-chunk HPO candidates using the retriever's index mode."""
    index_type = "single_vector"
    detect_index_type = getattr(retriever, "detect_index_type", None)
    if callable(detect_index_type):
        try:
            detected = detect_index_type()
            if detected in {"single_vector", "multi_vector"}:
                index_type = detected
            else:
                logger.debug("Unknown index type %r; using query_batch", detected)
        except Exception:
            logger.warning(
                "Could not detect retriever index type; using query_batch",
                exc_info=True,
            )

    query_batch_multi_vector = getattr(retriever, "query_batch_multi_vector", None)
    if index_type == "multi_vector" and callable(query_batch_multi_vector):
        logger.info(
            "Batch querying %d chunks with multi-vector aggregation",
            len(text_chunks),
        )
        return cast(
            list[dict[str, Any]],
            query_batch_multi_vector(
                texts=text_chunks,
                n_results=n_results,
            ),
        )

    if index_type == "multi_vector":
        logger.warning(
            "Retriever is connected to a multi-vector index but does not expose "
            "query_batch_multi_vector(); using query_batch"
        )

    logger.info("Batch querying %d chunks with query_batch", len(text_chunks))
    return cast(
        list[dict[str, Any]],
        retriever.query_batch(
            texts=text_chunks,
            n_results=n_results,
            include_similarities=include_similarities,
        ),
    )
