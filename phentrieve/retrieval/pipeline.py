"""Single-vector retrieval pipeline.

Extracts the retrieve -> convert -> rerank -> format_to_chromadb sequence
that was duplicated 3x in query_orchestrator.py (sentence mode, fallback,
full-text mode).
"""

from typing import Any

from phentrieve.config import DEFAULT_DENSE_TRUST_THRESHOLD
from phentrieve.retrieval import reranker as reranker_module
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.utils import convert_results_to_candidates


def execute_single_vector_pipeline(
    retriever: DenseRetriever,
    text: str,
    num_results: int,
    cross_encoder: Any | None = None,
    rerank_count: int | None = None,
    debug: bool = False,
    output_func: Any = print,
) -> dict[str, Any]:
    """Execute single-vector retrieval with optional reranking.

    Returns results in ChromaDB format (reranked if cross-encoder provided).

    Args:
        retriever: Dense retriever instance
        text: Query text
        num_results: Number of results to return
        cross_encoder: Optional cross-encoder model for reranking
        rerank_count: Number of candidates to rerank (None = no reranking)
        debug: Enable debug output
        output_func: Function for debug output

    Returns:
        ChromaDB-format result dict (possibly reranked)
    """
    # Determine query count
    if cross_encoder and rerank_count is not None:
        query_count = rerank_count * 2
    else:
        query_count = num_results * 2

    # Query the retriever
    results = retriever.query(text, n_results=query_count)

    # Rerank with cross-encoder if available
    if cross_encoder and rerank_count is not None:
        if debug:
            output_func("[DEBUG] Reranking with protected dense retrieval")
        try:
            candidates = convert_results_to_candidates(results)
            reranked_candidates = reranker_module.protected_dense_rerank(
                text,
                candidates,
                cross_encoder,
                trust_threshold=DEFAULT_DENSE_TRUST_THRESHOLD,
            )
            # Convert back to ChromaDB format
            results = {
                "ids": [[c["hpo_id"] for c in reranked_candidates]],
                "metadatas": [[c["metadata"] for c in reranked_candidates]],
                "documents": [[c["english_doc"] for c in reranked_candidates]],
                "distances": [
                    [1.0 - c.get("bi_encoder_score", 0.0) for c in reranked_candidates]
                ],
            }
        except Exception as e:
            if debug:
                output_func(f"[DEBUG] Error during re-ranking: {str(e)}")
            # Fall through to return unranked results

    return results
