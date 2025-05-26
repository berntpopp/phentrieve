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
    load_translation_text,
)


def segment_text(text: str, lang: str = None) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Text to segment
        lang: Language code for sentence segmentation

    Returns:
        List of sentences
    """
    # Use detected language or default to English if not specified
    if lang is None:
        # Try to detect language from text content
        if len(text) > 20 and text[:20].strip().isascii():
            lang = "en"  # Default to English for ASCII text
        else:
            lang = "en"  # Fallback to English

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
    Format the query results into a structured format, filtering by similarity threshold.

    Args:
        results: Raw results dictionary from retriever.query
        threshold: Minimum similarity score to display
        max_results: Maximum number of results to display
        query: Original query text
        reranked: Whether the results were re-ranked by a cross-encoder

    Returns:
        Dictionary with structured results information suitable for different output formats
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
        return {
            "results": [],
            "query_text_processed": query,
            "header_info": "No matching HPO terms found.",
        }

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
            "query_text_processed": query,
            "header_info": "No results above the similarity threshold.",
        }

    header_info = f"Found {len(filtered_results)} matching HPO terms:"
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
            "rank": i + 1,  # Add rank for better structured output
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

    return {
        "results": results_list,
        "query_text_processed": query,
        "header_info": header_info,
    }


def _format_structured_results_to_text_display(results: Dict[str, Any]) -> str:
    """
    Format structured results into a human-readable text string.

    Args:
        results: Structured results dictionary

    Returns:
        Formatted text string for display
    """
    output_lines = []

    # Get the formatted results
    header = results.get("header_info", "")
    results_list = results.get("results", [])

    # Add the header
    if header:
        output_lines.append(header)

    # Format each result
    for entry in results_list:
        hpo_id = entry.get("hpo_id", "")
        label = entry.get("label", "")
        similarity = entry.get("similarity", 0.0)
        similarity_str = f"{similarity:.2f}"
        rank_display = f"{entry.get('rank', '?')}."

        # Get re-ranking info if available
        reranking_info = ""
        if "cross_encoder_score" in entry:
            ce_score = f"{entry['cross_encoder_score']:.2f}"
            original_rank = entry.get("original_rank", "?")
            reranking_info = (
                f" [re-ranked from #{original_rank}, cross-encoder: {ce_score}]"
            )

        output_lines.append(
            f"{rank_display:3} {hpo_id:11} {label} (similarity: {similarity_str}){reranking_info}"
        )

    return "\n".join(output_lines)


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
        translation_dir: Directory containing translations of HPO terms in target language
        output_func: Function to use for output (for debug messages only, not for final results)

    Returns:
        List of structured result dictionaries, one per query (or sentence if sentence_mode is True)
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

            # Single query mode, no sentence splitting
            query_result = retriever.query(
                query_embedding=retriever.encode_query(text),
                top_k=rerank_count if cross_encoder else num_results,
            )

            # Perform re-ranking if a cross-encoder is provided
            if cross_encoder:
                # Extract metadata for re-ranking
                rerank_query_result = query_result

                # Apply cross-encoder re-ranking based on the selected mode
                if reranker_mode == "monolingual":
                    # Monolingual mode requires translations of HPO terms
                    lang_code = detect_language(text)
                    translations = load_translation_text(translation_dir, lang_code)

                    # Re-rank using the original language
                    reranked_result = reranker.rerank_with_cross_encoder_monolingual(
                        query=text,
                        query_result=rerank_query_result,
                        cross_encoder=cross_encoder,
                        translations=translations,
                        top_k=num_results,
                    )
                else:
                    # Cross-lingual mode (default): uses cross-encoder directly
                    reranked_result = reranker.rerank_with_cross_encoder(
                        query=text,
                        query_result=rerank_query_result,
                        cross_encoder=cross_encoder,
                        top_k=num_results,
                    )

                query_result = reranked_result

            # Format the results into a structured format
            formatted_result = format_results(
                query_result,
                threshold=similarity_threshold,
                max_results=num_results,
                query=text,
                reranked=cross_encoder is not None,
            )
            structured_results = [formatted_result]

            # Add to the list of results
            all_results.extend(structured_results)

            output_func("\n---------- Re-Ranked Results (Cross-Encoder) ----------")
            # Return all results collected from sentences
        return all_results
    else:
        # Process the entire text at once
        if debug:
            output_func(f"[DEBUG] Processing complete text: {text[:50]}...")

        # Set query count - need more results for reranking
        if cross_encoder and rerank_count is not None:
            query_count = rerank_count * 2
        else:
            query_count = num_results * 2

        # Query the retriever
        results = retriever.query(text, n_results=query_count)

        # Perform re-ranking if a cross-encoder is provided
        if cross_encoder and rerank_count is not None:
            if debug:
                output_func(f"[DEBUG] Reranking with cross-encoder")

            reranked_result = None
            try:
                # Apply cross-encoder re-ranking based on the selected mode
                if reranker_mode == "monolingual":
                    # Monolingual mode requires translations of HPO terms
                    from phentrieve.utils import detect_language

                    lang_code = detect_language(text)
                    translations = load_translation_text(translation_dir, lang_code)

                    # Re-rank using the original language
                    reranked_result = reranker.rerank_with_cross_encoder_monolingual(
                        query=text,
                        query_result=results,
                        cross_encoder=cross_encoder,
                        translations=translations,
                        top_k=num_results,
                    )
                else:
                    # Cross-lingual mode (default): uses cross-encoder directly
                    reranked_result = reranker.rerank_with_cross_encoder(
                        query=text,
                        query_result=results,
                        cross_encoder=cross_encoder,
                        top_k=num_results,
                    )
            except Exception as e:
                if debug:
                    output_func(f"[DEBUG] Error during re-ranking: {str(e)}")

            # Format the reranked results if available
            if reranked_result:
                formatted_result = format_results(
                    results=reranked_result,
                    threshold=similarity_threshold,
                    max_results=num_results,
                    query=text,
                    reranked=True,
                )
                if formatted_result and formatted_result["results"]:
                    all_results.append(formatted_result)

        # Format the original results (or use them if no reranking was done)
        if not all_results or not cross_encoder:
            formatted_result = format_results(
                results=results,
                threshold=similarity_threshold,
                max_results=num_results,
                query=text,
                reranked=False,
            )
            if formatted_result and formatted_result["results"]:
                all_results.append(formatted_result)
            # Return the structured results (could be empty list if no results were found)
    return all_results


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
        translation_dir: Directory with HPO translations in target language
        rerank_count: Number of candidates to rerank
        device_override: Override device (cpu/cuda)
        debug: Enable debug output
        output_func: Function to use for output (for setup and debug messages)
        interactive_setup: Whether this is just setting up models for interactive mode
        interactive_mode: Whether this is an interactive query using shared models

    Returns:
        List of structured result dictionaries, or bool if in interactive_setup mode
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
                # For monolingual mode, use the language-specific model
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
