"""Shared retrieval utilities.

Functions used across multiple retrieval and evaluation modules.
"""

from typing import Any

from phentrieve.retrieval.dense_retriever import calculate_similarity


def convert_multi_vector_to_chromadb_format(
    multi_vector_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convert multi-vector aggregated results to ChromaDB-style format.

    This allows reusing existing format_results() and output formatters.

    Args:
        multi_vector_results: List of aggregated results from query_multi_vector()

    Returns:
        Dictionary in ChromaDB query result format
    """
    if not multi_vector_results:
        return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

    ids = []
    metadatas = []
    documents = []
    distances = []

    for result in multi_vector_results:
        ids.append(result["hpo_id"])
        # Build metadata with component scores
        metadata = {
            "hpo_id": result["hpo_id"],
            "label": result.get("label", ""),
        }
        # Add component scores if present
        if "component_scores" in result:
            metadata["component_scores"] = result["component_scores"]
        metadatas.append(metadata)
        documents.append(result.get("label", ""))
        # Convert similarity to distance (1 - similarity)
        distances.append(1.0 - result.get("similarity", 0.0))

    return {
        "ids": [ids],
        "metadatas": [metadatas],
        "documents": [documents],
        "distances": [distances],
    }


def convert_results_to_candidates(
    results: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Convert ChromaDB query results to candidate format for reranking.

    Args:
        results: ChromaDB query results dictionary

    Returns:
        List of candidate dictionaries ready for reranking
    """
    candidates: list[dict[str, Any]] = []

    if not results or not results.get("ids") or not results["ids"][0]:
        return candidates

    ids = results["ids"][0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Use min length to handle partial ChromaDB responses gracefully
    n = min(len(ids), len(metadatas), len(documents), len(distances))
    for j in range(n):
        candidate = {
            "hpo_id": ids[j],
            "english_doc": documents[j],
            "metadata": metadatas[j],
            "rank": j + 1,
            "bi_encoder_score": calculate_similarity(distances[j]),
            "comparison_text": documents[j],  # Always use English document
        }

        candidates.append(candidate)

    return candidates
