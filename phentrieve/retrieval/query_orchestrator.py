"""
HPO Query Orchestrator

This module provides functionality for querying HPO terms with natural language
descriptions and processing the results. It migrates the functionality from the
interactive query script to be usable from the CLI interface.
"""

import logging
from collections.abc import Callable
from typing import Any

import pysbd
import torch

from phentrieve.config import (
    DEFAULT_AGGREGATION_STRATEGY,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
)
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.dense_retriever import (
    DenseRetriever,
    calculate_similarity,
)
from phentrieve.retrieval.interactive_state import interactive_state
from phentrieve.retrieval.utils import convert_multi_vector_to_chromadb_format

# Note: CombinedAssertionDetector is imported lazily when needed
# to avoid requiring spacy (optional dependency) for basic query operations
from phentrieve.utils import (
    detect_language,
    generate_collection_name,
)

_interactive_state = interactive_state


def _execute_multi_vector_query(
    retriever: DenseRetriever,
    text: str,
    num_results: int,
    aggregation_strategy: str,
    component_weights: dict[str, float] | None,
    custom_formula: str | None,
) -> dict[str, Any]:
    """
    Execute a multi-vector query and convert results to ChromaDB format.

    This helper function encapsulates the multi-vector query pattern to avoid
    code duplication (DRY principle). It performs the query using the retriever's
    multi-vector method and converts the results to ChromaDB-compatible format.

    Args:
        retriever: DenseRetriever instance connected to multi-vector index
        text: Query text to search for
        num_results: Number of results to return
        aggregation_strategy: Strategy for aggregating component scores
        component_weights: Weights for "all_weighted" strategy
        custom_formula: Formula for "custom" strategy

    Returns:
        Dictionary in ChromaDB result format with ids, metadatas, documents, distances
    """
    multi_results = retriever.query_multi_vector(
        text,
        n_results=num_results,
        aggregation_strategy=aggregation_strategy,
        component_weights=component_weights,
        custom_formula=custom_formula,
    )
    return convert_multi_vector_to_chromadb_format(multi_results)


def segment_text(text: str, lang: str | None = None) -> list[str]:
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
    return list(segmenter.segment(text))


def format_results(
    results: dict[str, Any],
    threshold: float = MIN_SIMILARITY_THRESHOLD,
    max_results: int = DEFAULT_TOP_K,
    query: str | None = None,
    original_query_assertion_status=None,
    original_query_assertion_details=None,
) -> dict[str, Any]:
    """
    Format the query results into a structured format, filtering by similarity threshold.

    Args:
        results: Raw results dictionary from retriever.query
        threshold: Minimum similarity score to display
        max_results: Maximum number of results to display
        query: Original query text

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

    # Iterate through all results (use min length for defensive handling)
    n = min(len(ids), len(metadatas), len(distances))
    for i in range(n):
        metadata = metadatas[i]
        distance = distances[i]
        # Calculate bi-encoder similarity from distance
        bi_encoder_similarity = calculate_similarity(distance)

        # Get HPO ID and label from metadata
        hpo_id = metadata.get("hpo_id", "Unknown")
        label = metadata.get("label", "Unknown")

        result_tuples.append((hpo_id, label, bi_encoder_similarity))

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
    ) in enumerate(filtered_results):
        # For each result, create a dictionary with relevant information
        entry = {
            "rank": i + 1,  # Add rank for better structured output
            "hpo_id": hpo_id,
            "label": label,
            "similarity": bi_encoder_similarity,
        }

        results_list.append(entry)

    return {
        "results": results_list,
        "query_text_processed": query,
        "header_info": header_info,
        "original_query_assertion_status": (
            original_query_assertion_status.value
            if original_query_assertion_status
            else None
        ),
        "original_query_assertion_status_value": (
            original_query_assertion_status.value
            if original_query_assertion_status
            else None
        ),  # Keep for backward compatibility
        "original_query_assertion_details": original_query_assertion_details,  # Can be None
    }


def _format_structured_results_to_text_display(results: dict[str, Any]) -> str:
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

        output_lines.append(
            f"{rank_display:3} {hpo_id:11} {label} (similarity: {similarity_str})"
        )

    return "\n".join(output_lines)


def process_query(
    text: str,
    retriever: DenseRetriever,
    num_results: int = DEFAULT_TOP_K,
    sentence_mode: bool = False,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    debug: bool = False,
    output_func: Callable = print,
    query_assertion_detector=None,
    # Multi-vector parameters (Issue #136)
    multi_vector: bool = False,
    aggregation_strategy: str = DEFAULT_AGGREGATION_STRATEGY,
    component_weights: dict[str, float] | None = None,
    custom_formula: str | None = None,
) -> list[dict[str, Any]]:
    """
    Process input text, either as a whole or sentence by sentence.

    Args:
        text: The input text to process
        retriever: DenseRetriever instance for querying
        num_results: Number of results to display for each query
        sentence_mode: Whether to process text sentence by sentence
        similarity_threshold: Minimum similarity threshold for results
        debug: Whether to enable debug logging
        output_func: Function to use for output (for debug messages only, not for final results)
        query_assertion_detector: Optional query assertion detector

    Returns:
        List of structured result dictionaries, one per query (or sentence if sentence_mode is True)
    """
    all_results = []

    # Detect assertion on the entire original input query if a detector is provided
    original_query_assertion_status = None
    original_query_assertion_details = None

    if query_assertion_detector and text:
        try:
            # Explicitly log detector info
            if debug:
                output_func(
                    f"[DEBUG] Using assertion detector with language: {query_assertion_detector.language}"
                )
                output_func(
                    f"[DEBUG] Assertion detector config: keyword={query_assertion_detector.enable_keyword}, dependency={query_assertion_detector.enable_dependency}"
                )

            # Detect assertion status
            original_query_assertion_status, original_query_assertion_details = (
                query_assertion_detector.detect(text)
            )

            # Make sure we log a successful detection
            if original_query_assertion_status:
                output_func(
                    f"Query assertion detection enabled (lang: {query_assertion_detector.language}, pref: {query_assertion_detector.preference})"
                )

            # Log detailed information
            if debug:
                output_func(
                    f"[DEBUG] Query: '{text}' - Raw assertion result: {original_query_assertion_status}"
                )
                output_func(
                    f"[DEBUG] Assertion status type: {type(original_query_assertion_status)}"
                )
                output_func(
                    f"[DEBUG] Assertion value: {original_query_assertion_status.value if original_query_assertion_status else 'None'}"
                )
                output_func(
                    f"[DEBUG] Assertion details: {original_query_assertion_details}"
                )
        except Exception as e:
            if debug:
                output_func(f"[DEBUG] Error during assertion detection: {str(e)}")
                import traceback

                traceback.print_exc()
            # Ensure we have a fallback
            original_query_assertion_status = None
            original_query_assertion_details = {"error": str(e)}

    # If sentence_mode is True, split the text into sentences and process each one separately
    if sentence_mode:
        # Split into sentences and process
        sentences = segment_text(text)

        if debug:
            output_func(f"[DEBUG] Split into {len(sentences)} sentences")

        for i, sentence in enumerate(sentences):
            if debug:
                output_func(f"[DEBUG] Processing sentence {i + 1}: {sentence}")

            # Multi-vector query path for sentence mode (Issue #136)
            if multi_vector:
                results = _execute_multi_vector_query(
                    retriever=retriever,
                    text=sentence,
                    num_results=num_results,
                    aggregation_strategy=aggregation_strategy,
                    component_weights=component_weights,
                    custom_formula=custom_formula,
                )

                formatted = format_results(
                    results=results,
                    threshold=similarity_threshold,
                    max_results=num_results,
                    query=sentence,
                    original_query_assertion_status=original_query_assertion_status,
                    original_query_assertion_details=original_query_assertion_details,
                )
                if formatted and formatted["results"]:
                    all_results.append(formatted)
                continue  # Skip the single-vector path

            # Single-vector query path
            results = retriever.query(sentence, n_results=num_results)
            formatted = format_results(
                results=results,
                threshold=similarity_threshold,
                max_results=num_results,
                query=sentence,
                original_query_assertion_status=original_query_assertion_status,
                original_query_assertion_details=original_query_assertion_details,
            )

            # Only add if we have valid results
            if formatted and formatted["results"]:
                all_results.append(formatted)

        # If we have no results at all, try processing the whole text
        if not all_results:
            if debug:
                output_func("[DEBUG] No results from sentence mode, trying full text")

            # Multi-vector fallback path (Issue #136)
            if multi_vector:
                query_result = _execute_multi_vector_query(
                    retriever=retriever,
                    text=text,
                    num_results=num_results,
                    aggregation_strategy=aggregation_strategy,
                    component_weights=component_weights,
                    custom_formula=custom_formula,
                )

                formatted_result = format_results(
                    query_result,
                    threshold=similarity_threshold,
                    max_results=num_results,
                    query=text,
                    original_query_assertion_status=original_query_assertion_status,
                    original_query_assertion_details=original_query_assertion_details,
                )
                all_results.extend([formatted_result])
                return all_results

            # Single-vector fallback
            query_result = retriever.query(text, n_results=num_results)
            formatted_result = format_results(
                query_result,
                threshold=similarity_threshold,
                max_results=num_results,
                query=text,
                original_query_assertion_status=original_query_assertion_status,
                original_query_assertion_details=original_query_assertion_details,
            )
            all_results.extend([formatted_result])
            # Return all results collected from sentences
        return all_results
    else:
        # Process the entire text at once
        if debug:
            output_func(f"[DEBUG] Processing complete text: {text[:50]}...")

        # Multi-vector query path (Issue #136)
        if multi_vector:
            if debug:
                output_func(
                    f"[DEBUG] Using multi-vector query with strategy: {aggregation_strategy}"
                )

            # Query using multi-vector aggregation via helper function (DRY)
            results = _execute_multi_vector_query(
                retriever=retriever,
                text=text,
                num_results=num_results,
                aggregation_strategy=aggregation_strategy,
                component_weights=component_weights,
                custom_formula=custom_formula,
            )

            formatted_result = format_results(
                results=results,
                threshold=similarity_threshold,
                max_results=num_results,
                query=text,
                original_query_assertion_status=original_query_assertion_status,
                original_query_assertion_details=original_query_assertion_details,
            )
            if formatted_result and formatted_result["results"]:
                all_results.append(formatted_result)
            return all_results

        # Single-vector query path
        results = retriever.query(text, n_results=num_results)
        formatted_result = format_results(
            results=results,
            threshold=similarity_threshold,
            max_results=num_results,
            query=text,
            original_query_assertion_status=original_query_assertion_status,
            original_query_assertion_details=original_query_assertion_details,
        )
        if formatted_result and formatted_result["results"]:
            all_results.append(formatted_result)
    return all_results


def orchestrate_query(
    query_text: str | None = None,
    model_name: str = DEFAULT_MODEL,
    num_results: int = DEFAULT_TOP_K,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    sentence_mode: bool = False,
    trust_remote_code: bool = False,
    device_override: str | None = None,
    debug: bool = False,
    output_func: Callable = print,
    interactive_setup: bool = False,
    interactive_mode: bool = False,
    detect_query_assertion: bool = False,
    query_assertion_language: str | None = None,
    query_assertion_preference: str = "dependency",
    # Multi-vector parameters (Issue #136)
    multi_vector: bool = False,
    aggregation_strategy: str = DEFAULT_AGGREGATION_STRATEGY,
    component_weights: dict[str, float] | None = None,
    custom_formula: str | None = None,
) -> list[dict[str, Any]] | bool:
    """
    Main orchestration function for HPO term queries.

    Args:
        query_text: The clinical text to query
        model_name: Name of the embedding model to use
        num_results: Number of results to return
        similarity_threshold: Minimum similarity score threshold
        sentence_mode: Process text sentence-by-sentence
        trust_remote_code: Trust remote code when loading models
        device_override: Override device (cpu/cuda)
        debug: Enable debug output
        output_func: Function to use for output (for setup and debug messages)
        interactive_setup: Whether this is just setting up models for interactive mode
        interactive_mode: Whether this is an interactive query using shared models
        detect_query_assertion: Whether to detect query assertions
        query_assertion_language: Language of the query assertions
        query_assertion_preference: Preference for query assertion detection (dependency or rule-based)
        multi_vector: Use multi-vector index with component-level aggregation (Issue #136)
        aggregation_strategy: Strategy for aggregating component scores
        component_weights: Weights for "all_weighted" strategy
        custom_formula: Custom formula for "custom" strategy

    Returns:
        List of structured result dictionaries, or bool if in interactive_setup mode
    """
    # If in interactive mode, use the cached models from _interactive_state
    if interactive_mode:
        if not all(
            [_interactive_state.model, _interactive_state.retriever, query_text]
        ):
            error_msg = "Interactive mode requires initialized models and query text. Run with interactive_setup first."
            logging.error(error_msg)
            output_func(error_msg)
            return []

        # Type narrowing: at this point we know these are not None
        assert _interactive_state.retriever is not None
        assert query_text is not None

        # Process the query using the cached models
        return process_query(
            text=query_text,
            retriever=_interactive_state.retriever,
            num_results=num_results,
            sentence_mode=sentence_mode,
            similarity_threshold=similarity_threshold,
            debug=debug,
            output_func=output_func,
            query_assertion_detector=_interactive_state.query_assertion_detector,
            # Multi-vector parameters (Issue #136)
            multi_vector=_interactive_state.multi_vector,
            aggregation_strategy=_interactive_state.aggregation_strategy,
            component_weights=_interactive_state.component_weights,
            custom_formula=_interactive_state.custom_formula,
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

        # Initialize retriever (use multi-vector index if requested)
        retriever = DenseRetriever.from_model_name(
            model=model,
            model_name=model_name,
            min_similarity=similarity_threshold,
            multi_vector=multi_vector,
        )

        if not retriever:
            error_msg = f"Failed to initialize retriever. Make sure you have built an index for {model_name}"
            logging.error(error_msg)
            output_func(error_msg)
            return [] if not interactive_setup else False

        # Get collection information
        collection_name = generate_collection_name(model_name)
        collection_count = retriever.collection.count()

        output_func(f"Model: {model_name}")
        output_func(f"Collection: {collection_name}")
        output_func(f"Index entries: {collection_count}")
        output_func(f"Similarity threshold: {similarity_threshold}")

        # Initialize assertion detector for the query if requested
        query_assertion_detector_to_use = None

        if detect_query_assertion:
            # Lazy import to avoid requiring spacy for basic queries
            from phentrieve.text_processing.assertion_detection import (
                CombinedAssertionDetector,
            )

            actual_query_assertion_lang = query_assertion_language
            if not actual_query_assertion_lang and query_text:
                actual_query_assertion_lang = detect_language(
                    query_text, default_lang="en"
                )
            elif not actual_query_assertion_lang:
                actual_query_assertion_lang = "en"  # Fallback for interactive_setup

            assertion_config_for_query = {
                "enable_keyword": True,  # Keep these simple for query, or make configurable later
                "enable_dependency": True,
                "preference": query_assertion_preference,
            }
            query_assertion_detector_to_use = CombinedAssertionDetector(
                language=actual_query_assertion_lang,
                # Pass only the expected kwargs to CombinedAssertionDetector
                enable_keyword=bool(assertion_config_for_query["enable_keyword"]),
                enable_dependency=bool(assertion_config_for_query["enable_dependency"]),
                preference=str(assertion_config_for_query["preference"]),
            )
            if not interactive_setup:  # Avoid logging during mere setup
                output_func(
                    f"Query assertion detection enabled (lang: {actual_query_assertion_lang}, pref: {query_assertion_preference})"
                )

        # If this is just interactive setup, store models in state container and return
        if interactive_setup:
            _interactive_state.model = model
            _interactive_state.retriever = retriever
            _interactive_state.query_assertion_detector = (
                query_assertion_detector_to_use  # Store for interactive use
            )
            # Store multi-vector settings (Issue #136)
            _interactive_state.multi_vector = multi_vector
            _interactive_state.aggregation_strategy = aggregation_strategy
            _interactive_state.component_weights = component_weights
            _interactive_state.custom_formula = custom_formula

            # Log that assertion detection is ready if requested
            if detect_query_assertion:
                output_func(
                    f"Query assertion detection enabled (lang: {actual_query_assertion_lang}, pref: {query_assertion_preference})"
                )

            # Log multi-vector status
            if multi_vector:
                output_func(
                    f"Multi-vector mode enabled (strategy: {aggregation_strategy})"
                )

            return True

        # For non-interactive mode, process the single query
        if query_text is None and not interactive_setup:
            error_msg = "Query text is required for non-interactive mode"
            logging.error(error_msg)
            output_func(error_msg)
            return []

        # Determine which assertion detector to use (cached for interactive, local for non-interactive)
        active_query_assertion_detector = (
            _interactive_state.query_assertion_detector
            if interactive_mode
            else query_assertion_detector_to_use
        )

        # Process the query
        return process_query(
            text=query_text,
            retriever=retriever
            if not interactive_mode
            else _interactive_state.retriever,
            num_results=num_results,
            sentence_mode=sentence_mode,
            similarity_threshold=similarity_threshold,
            debug=debug,
            output_func=output_func,
            query_assertion_detector=active_query_assertion_detector,
            # Multi-vector parameters (Issue #136)
            multi_vector=multi_vector,
            aggregation_strategy=aggregation_strategy,
            component_weights=component_weights,
            custom_formula=custom_formula,
        )

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logging.error(error_msg)
        if debug:
            import traceback

            traceback.print_exc()

        output_func(error_msg)
        return []
