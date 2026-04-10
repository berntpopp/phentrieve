"""Shared retrieval utilities.

Functions used across multiple retrieval and evaluation modules.
"""

from typing import Any


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
        # Convert similarity to distance (1 - similarity).
        # Normalize to float: guard against None (key present but null) which
        # would raise TypeError in `1.0 - None`.  The legitimate 0.0 case is
        # handled correctly because `0.0 or 0.0` evaluates to 0.0.
        similarity = result.get("similarity") or 0.0
        distances.append(1.0 - similarity)

    return {
        "ids": [ids],
        "metadatas": [metadatas],
        "documents": [documents],
        "distances": [distances],
    }
