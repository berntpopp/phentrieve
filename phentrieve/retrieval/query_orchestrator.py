"""
HPO Query Orchestrator

This module provides functionality for querying HPO terms with natural language
descriptions and processing the results. It migrates the functionality from the
interactive query script to be usable from the CLI interface.
"""

import logging
import os
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

import pysbd

from phentrieve.config import (
    DEFAULT_MODEL,
    MIN_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_RERANKER_MODE,
    DEFAULT_TRANSLATIONS_SUBDIR,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    DEFAULT_ENABLE_RERANKER,
)
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.dense_retriever import (
    DenseRetriever,
    calculate_similarity,
)
from phentrieve.retrieval import reranker
from phentrieve.utils import (
    generate_collection_name,
    load_german_translation_text,
)


def segment_text(text: str, lang: str = "de") -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Text to segment
        lang: Language code for sentence segmentation

    Returns:
        List of sentences
    """
    segmenter = pysbd.Segmenter(language=lang, clean=False)
    return segmenter.segment(text)


def format_results(
    results: Dict[str, Any],
    threshold: float = MIN_SIMILARITY_THRESHOLD,
    max_results: int = DEFAULT_TOP_K,
    query: str = None,
    reranked: bool = False,
) -> Dict[str, Any]:
    """
    Format the query results for display, filtering by similarity threshold.

    Args:
        results: Raw results dictionary from retriever.query
        threshold: Minimum similarity score to display
        max_results: Maximum number of results to display
        query: Original query text
        reranked: Whether the results were re-ranked by a cross-encoder

    Returns:
        Dictionary with formatted results information
    """
    logging.debug(
        f"format_results called with: threshold={threshold} ({type(threshold)}), max_results={max_results} ({type(max_results)})"
    )

    # Ensure max_results is an integer
    if not isinstance(max_results, int):
        if max_results is None:
            max_results = DEFAULT_TOP_K
        else:
            try:
                max_results = int(max_results)
            except (ValueError, TypeError):
                logging.warning(
                    f"max_results is not an integer: {type(max_results)}, value: {max_results}"
                )
                max_results = DEFAULT_TOP_K
    if not results or not results.get("ids") or not results["ids"][0]:
        return {"results": [], "query": query, "header": "No matching HPO terms found."}

    formatted_output = []

    # Check if this is a re-ranked result by looking for cross_encoder_score in first metadata
    is_reranked = False
    if (
        results.get("metadatas")
        and len(results["metadatas"]) > 0
        and len(results["metadatas"][0]) > 0
    ):
        is_reranked = "cross_encoder_score" in results["metadatas"][0][0]

    # Create a list of result tuples with all necessary information
    result_tuples = []

    # Check if distances are available in the results
    has_distances = (
        "distances" in results
        and len(results["distances"]) > 0
        and len(results["distances"][0]) > 0
    )

    # Prepare data for iteration
    ids = results["ids"][0] if results.get("ids") and len(results["ids"]) > 0 else []
    metadatas = (
        results["metadatas"][0]
        if results.get("metadatas") and len(results["metadatas"]) > 0
        else []
    )
    distances = results["distances"][0] if has_distances else [0.0] * len(ids)

    # Iterate through all results
    for i, (doc_id, metadata, distance) in enumerate(zip(ids, metadatas, distances)):
        # Calculate bi-encoder similarity from distance
        bi_encoder_similarity = calculate_similarity(distance)

        # Get HPO ID and label from metadata
        hpo_id = metadata.get("hpo_id", "Unknown")
        label = metadata.get("label", "Unknown")

        # Get cross-encoder score if available
        cross_encoder_score = None
        if "cross_encoder_score" in metadata:
            cross_encoder_score = metadata["cross_encoder_score"]

        # Get original rank if available
        original_rank = None
        if "original_rank" in metadata:
            original_rank = metadata["original_rank"]

        result_tuples.append(
            (hpo_id, label, bi_encoder_similarity, cross_encoder_score, original_rank)
        )

    # If results were re-ranked, they're already in correct order, just limit to max_results
    if is_reranked:
        # Make sure max_results is an integer
        if not isinstance(max_results, int):
            logging.warning(
                f"max_results is not an integer: {type(max_results)}, value: {max_results}"
            )
            max_results = int(max_results) if max_results is not None else DEFAULT_TOP_K

        filtered_results = result_tuples[:max_results]
    else:
        # Sort by bi-encoder similarity (highest first)
        result_tuples.sort(key=lambda x: x[2], reverse=True)

        # Filter by threshold
        threshold_filtered = [res for res in result_tuples if res[2] >= threshold]

        # Make sure max_results is an integer
        if not isinstance(max_results, int):
            logging.warning(
                f"max_results is not an integer: {type(max_results)}, value: {max_results}"
            )
            max_results = int(max_results) if max_results is not None else DEFAULT_TOP_K

        # Apply limit safely
        filtered_results = threshold_filtered[:max_results]

    # Format the results
    if not filtered_results:
        return {
            "results": [],
            "query": query,
            "header": "No results above the similarity threshold.",
        }

    header = f"Found {len(filtered_results)} matching HPO terms:"
    results_list = []

    # Create results list with formatted entries
    filtered_results = filtered_results if filtered_results else []
    for i, (
        hpo_id,
        label,
        bi_encoder_similarity,
        cross_encoder_score,
        original_rank,
    ) in enumerate(filtered_results):
        # For each result, create a dictionary with relevant information
        entry = {
            "hpo_id": hpo_id,
            "label": label,
            "similarity": bi_encoder_similarity,
        }

        # Add cross-encoder score if present
        if cross_encoder_score is not None:
            entry["cross_encoder_score"] = cross_encoder_score

        # Add original rank if present
        if original_rank is not None:
            entry["original_rank"] = original_rank

        results_list.append(entry)

    return {"results": results_list, "query": query, "header": header}


def print_results(results: Dict[str, Any], output_func: Callable = print) -> None:
    """
    Print formatted results to the console or using a custom output function.

    Args:
        results: Formatted results dictionary
        output_func: Function to use for output (defaults to print)
    """
    if not results or not results.get("results"):
        output_func("No matching HPO terms found.")
        return

    output_func(f"\n{results.get('header', 'Results:')}")

    for i, result in enumerate(results["results"]):
        hpo_id = result.get("hpo_id", "Unknown")
        label = result.get("label", "Unknown")

        # Get the cross-encoder score and original rank if available
        if "cross_encoder_score" in result:
            original_rank = result.get("original_rank", "?")
            output_func(
                f"{i+1}. {hpo_id}: {label} (was #{original_rank})\n   Cross-encoder score: {result['cross_encoder_score']:.4f}"
            )
        else:
            similarity = result["similarity"]
            output_func(f"{i+1}. {hpo_id}: {label}\n   Similarity: {similarity:.4f}")


def process_query(
    text: str,
    retriever: DenseRetriever,
    num_results: int = DEFAULT_TOP_K,
    sentence_mode: bool = False,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    debug: bool = False,
    cross_encoder=None,
    rerank_count: int = None,
    reranker_mode: str = DEFAULT_RERANKER_MODE,
    translation_dir: str = DEFAULT_TRANSLATIONS_SUBDIR,
    output_func: Callable = print,
) -> List[Dict[str, Any]]:
    """
    Process input text, either as a whole or sentence by sentence.

    Args:
        text: The input text to process
        retriever: DenseRetriever instance for querying
        num_results: Number of results to display for each query
        sentence_mode: Whether to process text sentence by sentence
        similarity_threshold: Minimum similarity threshold for results
        debug: Whether to enable debug logging
        cross_encoder: Optional cross-encoder model for re-ranking
        rerank_count: Number of candidates to re-rank (if cross_encoder is provided)
        reranker_mode: Mode for re-ranking ('cross-lingual' or 'monolingual')
        translation_dir: Directory containing German translations of HPO terms
        output_func: Function to use for output (defaults to print)

    Returns:
        List of result dictionaries, one per query (or sentence if sentence_mode is True)
    """
    all_results = []

    # Process in sentence mode if enabled
    if sentence_mode:
        # Split into sentences and process
        sentences = segment_text(text)

        if debug:
            output_func(f"[DEBUG] Split into {len(sentences)} sentences")

        for i, sentence in enumerate(sentences):
            if debug:
                output_func(f"[DEBUG] Processing sentence {i+1}: {sentence}")

            # Set query count - need more results for reranking
            if cross_encoder and rerank_count is not None:
                query_count = rerank_count * 2
            else:
                query_count = num_results * 2

            # Query the retriever
            results = retriever.query(sentence, n_results=query_count)

            # Rerank with cross-encoder if available
            if cross_encoder and rerank_count:
                if debug:
                    output_func(f"[DEBUG] Reranking with cross-encoder")
                reranked_results = reranker.rerank_with_cross_encoder(
                    query=sentence,
                    results=results,
                    cross_encoder=cross_encoder,
                    top_k=rerank_count,
                )
                formatted = format_results(
                    results=reranked_results,
                    threshold=similarity_threshold,
                    max_results=num_results,
                    query=sentence,
                    reranked=True,
                )
            else:
                formatted = format_results(
                    results=results,
                    threshold=similarity_threshold,
                    max_results=num_results,
                    query=sentence,
                )

            # Only add if we have valid results
            if formatted and formatted["results"]:
                all_results.append(formatted)

        # If we have no results at all, try processing the whole text
        if not all_results:
            if debug:
                output_func("[DEBUG] No results from sentence mode, trying full text")

            # Set query count - need more results for reranking
            if cross_encoder and rerank_count is not None:
                query_count = rerank_count * 2
            else:
                query_count = num_results * 2

            # Query the retriever
            results = retriever.query(text, n_results=query_count)

            # Rerank with cross-encoder if available
            if cross_encoder and rerank_count:
                if debug:
                    output_func(f"[DEBUG] Reranking with cross-encoder")
                reranked_results = reranker.rerank_with_cross_encoder(
                    query=text,
                    results=results,
                    cross_encoder=cross_encoder,
                    top_k=rerank_count,
                )
                formatted = format_results(
                    results=reranked_results,
                    threshold=similarity_threshold,
                    max_results=num_results,
                    query=text,
                    reranked=True,
                )
            else:
                formatted = format_results(
                    results=results,
                    threshold=similarity_threshold,
                    max_results=num_results,
                    query=text,
                )

            if formatted and formatted["results"]:
                all_results.append(formatted)

        # Print results for each sentence
        if all_results:
            for i, result_set in enumerate(all_results):
                output_func(f"\n==== Results for: {result_set['query']} ====")
                print_results(result_set, output_func=output_func)
        else:
            output_func("\nNo matching HPO terms found.")

    else:
        # Process the entire text at once
        # Set query count - need more results for reranking
        if cross_encoder and rerank_count is not None:
            query_count = rerank_count * 2
        else:
            query_count = num_results * 2

        # Query the retriever
        results = retriever.query(text, n_results=query_count)

        # Format original results first
        original_formatted = format_results(
            results=results,
            threshold=similarity_threshold,
            max_results=num_results,
            query=text,
        )

        # Only proceed with re-ranking if we have original results
        if not original_formatted or not original_formatted.get("results"):
            # No results passed the threshold, so just return the original results
            formatted = original_formatted
            if formatted and formatted["results"]:
                print_results(formatted, output_func=output_func)
                all_results.append(formatted)
            else:
                output_func("\nNo matching HPO terms found.")
            return all_results

        # Rerank with cross-encoder if available
        if (
            cross_encoder
            and rerank_count
            and results
            and results.get("ids")
            and len(results["ids"]) > 0
            and len(results["ids"][0]) > 0
        ):
            if debug:
                output_func(f"[DEBUG] Reranking with cross-encoder")

            # Extract only the top candidates from original results that passed the threshold
            candidates = []
            original_results = original_formatted["results"]

            if debug:
                output_func(
                    f"[DEBUG] Original formatted results structure: {original_formatted.keys()}"
                )
                output_func(
                    f"[DEBUG] Original results count: {len(original_results) if original_results else 0}"
                )
                if original_results and len(original_results) > 0:
                    output_func(
                        f"[DEBUG] First result keys: {original_results[0].keys()}"
                    )

            # We only want to re-rank the candidates that passed the bi-encoder filtering
            for result in original_results:
                if debug:
                    output_func(
                        f"[DEBUG] Processing result: {result.get('hpo_id', 'unknown')} with type {type(result)}"
                    )

                # Extract the HPO ID from the result
                hpo_id = result.get("hpo_id", None)
                if not hpo_id:
                    if debug:
                        output_func(f"[DEBUG] Missing hpo_id in result: {result}")
                    continue

                # Find the full document and metadata for this ID
                # Use the HPO ID as-is since we store them with the same format in ChromaDB
                chroma_compatible_id = hpo_id
                if debug:
                    output_func(f"[DEBUG] Looking for HPO ID: {chroma_compatible_id}")

                found = False
                for i, doc_id in enumerate(results["ids"][0]):
                    if doc_id == chroma_compatible_id:
                        found = True
                        if debug:
                            output_func(f"[DEBUG] Found matching document for {hpo_id}")
                        original_rank = original_results.index(result) + 1
                        # Get metadata for this document
                        metadata = results["metadatas"][0][i]

                        # Get the document text to use for comparison based on the reranker mode
                        if reranker_mode == "monolingual":
                            # For monolingual mode, load the German translation of the HPO term
                            german_text = load_german_translation_text(
                                hpo_id, translation_dir
                            )

                            if german_text is None:
                                if debug:
                                    output_func(
                                        f"[DEBUG] No German translation found for {hpo_id}, skipping"
                                    )
                                continue

                            if debug:
                                output_func(
                                    f"[DEBUG] Loaded German translation for {hpo_id}: {german_text[:50]}..."
                                )

                            comparison_text = german_text

                        else:  # cross-lingual mode
                            # For cross-lingual mode, use the simplified English label
                            comparison_text = metadata.get("label", "")

                        candidates.append(
                            {
                                "id": doc_id,
                                "comparison_text": comparison_text,  # Text to use for re-ranking
                                "english_doc": metadata.get(
                                    "label", ""
                                ),  # Keep the English label
                                "original_document": results["documents"][0][
                                    i
                                ],  # Keep the original for reference
                                "metadata": metadata,
                                "distance": (
                                    results["distances"][0][i]
                                    if "distances" in results
                                    and len(results["distances"]) > 0
                                    else 0.0
                                ),
                                "embedding": None,  # Safer to omit embeddings
                                "bi_encoder_score": result.get("similarity", 0.0),
                                "original_rank": original_rank,
                            }
                        )
                        break

                if not found and debug:
                    output_func(
                        f"[DEBUG] Could not find document for {hpo_id} in results"
                    )

            if debug:
                output_func(f"[DEBUG] Candidates for re-ranking: {len(candidates)}")
                if candidates:
                    output_func(f"[DEBUG] First candidate keys: {candidates[0].keys()}")
                    output_func(
                        f"[DEBUG] First candidate simplified english_doc (raw label): {candidates[0]['english_doc']}"
                    )
                    output_func(
                        f"[DEBUG] Original document: {candidates[0]['original_document'][:50]}..."
                    )
                else:
                    output_func(f"[DEBUG] No candidates to re-rank")
                    output_func(
                        f"[DEBUG] Results shape: ids={len(results['ids'][0])}, documents={len(results['documents'][0])}, metadatas={len(results['metadatas'][0])}"
                    )
                    # Sample a few IDs for comparison
                    sample_ids = results["ids"][0][:5]
                    output_func(f"[DEBUG] Sample IDs from results: {sample_ids}")
                    output_func(
                        f"[DEBUG] Original result IDs: {[r.get('hpo_id', 'unknown') for r in original_results[:5]]}"
                    )
                    # Try to match them directly
                    for orig_result in original_results[:5]:
                        orig_id = orig_result.get("hpo_id", "unknown")
                        found = orig_id in sample_ids
                        output_func(
                            f"[DEBUG] Original ID {orig_id} found in results: {found}"
                        )

            # Call the reranker with the correct parameters
            reranked_candidates = reranker.rerank_with_cross_encoder(
                query=text, candidates=candidates, cross_encoder_model=cross_encoder
            )

            # Convert back to ChromaDB format
            reranked_results = {
                "ids": [[c["id"] for c in reranked_candidates[:rerank_count]]],
                "documents": [
                    [c["english_doc"] for c in reranked_candidates[:rerank_count]]
                ],
                "metadatas": [
                    [c["metadata"] for c in reranked_candidates[:rerank_count]]
                ],
                "distances": [
                    [
                        1.0 - c.get("cross_encoder_score", 0.0)
                        for c in reranked_candidates[:rerank_count]
                    ]
                ],
            }

            # Add cross-encoder scores and original rank to metadata
            for i, candidate in enumerate(reranked_candidates[:rerank_count]):
                reranked_results["metadatas"][0][i]["cross_encoder_score"] = (
                    candidate.get("cross_encoder_score", 0.0)
                )
                reranked_results["metadatas"][0][i]["original_rank"] = candidate.get(
                    "original_rank", 0
                )

            reranked_formatted = format_results(
                results=reranked_results,
                threshold=similarity_threshold,
                max_results=num_results,
                query=text,
                reranked=True,
            )

            # Show both results
            output_func("\n---------- Original Results (Bi-Encoder) ----------")
            if original_formatted and original_formatted["results"]:
                print_results(original_formatted, output_func=output_func)
                all_results.append(original_formatted)
            else:
                output_func("No matching HPO terms found.")

            output_func("\n---------- Re-Ranked Results (Cross-Encoder) ----------")
            formatted = reranked_formatted
            if formatted and formatted["results"]:
                print_results(formatted, output_func=output_func)
                all_results.append(formatted)
            else:
                output_func("\nNo matching HPO terms found.")
        else:
            # Just use the original results
            formatted = original_formatted
            if formatted and formatted["results"]:
                print_results(formatted, output_func=output_func)
                all_results.append(formatted)
            else:
                output_func("\nNo matching HPO terms found.")

    return all_results


# Global variables for sharing models and retriever between interactive queries
_global_model = None
_global_retriever = None
_global_cross_encoder = None


def orchestrate_query(
    query_text: str = None,
    model_name: str = DEFAULT_MODEL,
    num_results: int = DEFAULT_TOP_K,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    sentence_mode: bool = False,
    trust_remote_code: bool = False,
    enable_reranker: bool = DEFAULT_ENABLE_RERANKER,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    monolingual_reranker_model: str = DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    reranker_mode: str = DEFAULT_RERANKER_MODE,
    translation_dir: str = DEFAULT_TRANSLATIONS_SUBDIR,
    rerank_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
    device_override: Optional[str] = None,
    debug: bool = False,
    output_func: Callable = print,
    interactive_setup: bool = False,
    interactive_mode: bool = False,
) -> Union[List[Dict[str, Any]], bool]:
    """
    Main orchestration function for HPO term queries.

    Args:
        query_text: The clinical text to query
        model_name: Name of the embedding model to use
        num_results: Number of results to return
        similarity_threshold: Minimum similarity score threshold
        sentence_mode: Process text sentence-by-sentence
        trust_remote_code: Trust remote code when loading models
        enable_reranker: Enable cross-encoder reranking
        reranker_model: Cross-encoder model name for reranking
        monolingual_reranker_model: Monolingual cross-encoder model
        reranker_mode: Reranking mode (cross-lingual or monolingual)
        translation_dir: Directory with German HPO translations
        rerank_count: Number of candidates to rerank
        device_override: Override device (cpu/cuda)
        debug: Enable debug output
        output_func: Function to use for output (defaults to print)
        interactive_setup: Whether this is just setting up models for interactive mode
        interactive_mode: Whether this is an interactive query using shared models

    Returns:
        List of result dictionaries, or bool if in interactive_setup mode
    """
    global _global_model, _global_retriever, _global_cross_encoder

    # If in interactive mode, use the cached models
    if interactive_mode:
        if not all([_global_model, _global_retriever]):
            error_msg = "Interactive mode requires initialized models. Run with interactive_setup first."
            logging.error(error_msg)
            output_func(error_msg)
            return []

        # Process the query using the global models
        return process_query(
            text=query_text,
            retriever=_global_retriever,
            num_results=num_results,
            sentence_mode=sentence_mode,
            similarity_threshold=similarity_threshold,
            debug=debug,
            cross_encoder=_global_cross_encoder,
            rerank_count=rerank_count if _global_cross_encoder else None,
            reranker_mode=reranker_mode,
            translation_dir=translation_dir,
            output_func=output_func,
        )

    # Determine device
    if device_override:
        device = device_override
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Log the device being used
    logging.info(f"Using device: {device}")

    try:
        # Load embedding model
        model = load_embedding_model(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            device=device,
        )

        # Initialize retriever
        retriever = DenseRetriever.from_model_name(
            model=model,
            model_name=model_name,
            min_similarity=similarity_threshold,
        )

        if not retriever:
            error_msg = f"Failed to initialize retriever. Make sure you have built an index for {model_name}"
            logging.error(error_msg)
            output_func(error_msg)
            return [] if not interactive_setup else False

        # Load cross-encoder model if re-ranking is enabled
        cross_encoder = None
        if enable_reranker:
            # Select the appropriate model based on the reranker mode
            ce_model_name = reranker_model
            if reranker_mode == "monolingual":
                # For monolingual mode, use the German-specific model
                ce_model_name = monolingual_reranker_model

                # Check if translation directory exists
                if not os.path.exists(translation_dir):
                    warning_msg = (
                        f"Translation directory not found: {translation_dir}. "
                        "Monolingual re-ranking will not work properly."
                    )
                    logging.warning(warning_msg)
                    output_func(warning_msg)

            # Load the selected cross-encoder model
            cross_encoder = reranker.load_cross_encoder(ce_model_name, device)
            if cross_encoder:
                logging.info(
                    f"Cross-encoder re-ranking enabled in {reranker_mode} mode with model: {ce_model_name}"
                )
            else:
                warning_msg = f"Failed to load cross-encoder model {ce_model_name}, re-ranking will be disabled"
                logging.warning(warning_msg)
                output_func(warning_msg)

        # Get collection information
        collection_name = generate_collection_name(model_name)
        collection_count = retriever.collection.count()

        output_func(f"Model: {model_name}")
        output_func(f"Collection: {collection_name}")
        output_func(f"Index entries: {collection_count}")
        output_func(f"Similarity threshold: {similarity_threshold}")
        if cross_encoder:
            model_display = (
                reranker_model
                if reranker_mode == "cross-lingual"
                else monolingual_reranker_model
            )
            output_func(f"Cross-encoder re-ranking: Enabled (using {model_display})")
        else:
            output_func("Cross-encoder re-ranking: Disabled")

        # If this is just interactive setup, store models and return
        if interactive_setup:
            _global_model = model
            _global_retriever = retriever
            _global_cross_encoder = cross_encoder
            return True

        # For non-interactive mode, process the single query
        if query_text is None and not interactive_setup:
            error_msg = "Query text is required for non-interactive mode"
            logging.error(error_msg)
            output_func(error_msg)
            return []

        # Process the query
        return process_query(
            text=query_text,
            retriever=retriever,
            num_results=num_results,
            sentence_mode=sentence_mode,
            similarity_threshold=similarity_threshold,
            debug=debug,
            cross_encoder=cross_encoder,
            rerank_count=rerank_count if enable_reranker else None,
            reranker_mode=reranker_mode,
            translation_dir=translation_dir,
            output_func=output_func,
        )

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logging.error(error_msg)
        if debug:
            import traceback

            traceback.print_exc()

        output_func(error_msg)
        return []
