"""Shared retrieval utilities.

Functions used across multiple retrieval and evaluation modules.
"""

from typing import Any


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
