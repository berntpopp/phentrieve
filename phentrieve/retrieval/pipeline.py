"""Single-vector retrieval pipeline.

Extracts the retrieve -> format_to_chromadb sequence
that was duplicated 3x in query_orchestrator.py (sentence mode, fallback,
full-text mode).
"""

from typing import Any

from phentrieve.retrieval.dense_retriever import DenseRetriever


def execute_single_vector_pipeline(
    retriever: DenseRetriever,
    text: str,
    num_results: int,
    debug: bool = False,
    output_func: Any = print,
) -> dict[str, Any]:
    """Execute single-vector retrieval.

    Returns results in ChromaDB format.

    Args:
        retriever: Dense retriever instance
        text: Query text
        num_results: Number of results to return
        debug: Enable debug output
        output_func: Function for debug output

    Returns:
        ChromaDB-format result dict
    """
    query_count = num_results * 2

    # Query the retriever
    results = retriever.query(text, n_results=query_count)

    return results
