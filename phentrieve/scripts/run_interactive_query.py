#!/usr/bin/env python3
"""
Interactive HPO Query Tool

This script provides an interactive CLI for querying the HPO index with
multilingual clinical text descriptions to find matching HPO terms.
"""

import argparse
import logging
import os
import sys
import time
import traceback
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Set up import paths for the phentrieve modules
# The correct structure is: repo_root/phentrieve/phentrieve/[modules]
script_dir = Path(__file__).parent.absolute()
phentrieve_dir = script_dir.parent  # This is 'phentrieve' directory
sys.path.insert(0, str(phentrieve_dir))
print(f"Added phentrieve directory to path: {phentrieve_dir}")

import pysbd

# Now directly import from the phentrieve directory
from phentrieve.config import (
    DEFAULT_MODEL,
    INDEX_DIR,
    MIN_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_RERANKER_MODE,
    DEFAULT_TRANSLATION_DIR,
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


def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    # Also set root logger level
    logging.getLogger().setLevel(level)


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
    # Debug print to trace calls with incorrect parameters
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


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted results to the console."""
    if not results or not results.get("results"):
        print("No matching HPO terms found.")
        return

    print(f"\n{results.get('header', 'Results:')}")

    for i, result in enumerate(results["results"]):
        hpo_id = result.get("hpo_id", "Unknown")
        label = result.get("label", "Unknown")

        # Get the cross-encoder score and original rank if available
        if "cross_encoder_score" in result:
            original_rank = result.get("original_rank", "?")
            print(
                f"{i+1}. {hpo_id}: {label} (was #{original_rank})\n   Cross-encoder score: {result['cross_encoder_score']:.4f}"
            )
        else:
            similarity = result["similarity"]
            print(f"{i+1}. {hpo_id}: {label}\n   Similarity: {similarity:.4f}")


def process_input(
    text: str,
    retriever: DenseRetriever,
    num_results: int = DEFAULT_TOP_K,
    sentence_mode: bool = False,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    debug: bool = False,
    cross_encoder=None,
    rerank_count: int = None,
    reranker_mode: str = DEFAULT_RERANKER_MODE,
    translation_dir: str = DEFAULT_TRANSLATION_DIR,
) -> None:
    """Process input text, either as a whole or sentence by sentence.

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
    """

    # Process in sentence mode if enabled
    if sentence_mode:
        # Split into sentences and process
        sentences = segment_text(text)
        all_results = []

        if debug:
            print(f"[DEBUG] Split into {len(sentences)} sentences")

        for i, sentence in enumerate(sentences):
            if debug:
                print(f"[DEBUG] Processing sentence {i+1}: {sentence}")

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
                    print(f"[DEBUG] Reranking with cross-encoder")
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
                print("[DEBUG] No results from sentence mode, trying full text")

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
                    print(f"[DEBUG] Reranking with cross-encoder")
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
                print(f"\n==== Results for: {result_set['query']} ====")
                print_results(result_set)
        else:
            print("\nNo matching HPO terms found.")

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
                print_results(formatted)
            else:
                print("\nNo matching HPO terms found.")
            return

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
                print(f"[DEBUG] Reranking with cross-encoder")

            # Extract only the top candidates from original results that passed the threshold
            candidates = []
            original_results = original_formatted["results"]

            if debug:
                print(
                    f"[DEBUG] Original formatted results structure: {original_formatted.keys()}"
                )
                print(
                    f"[DEBUG] Original results count: {len(original_results) if original_results else 0}"
                )
                if original_results and len(original_results) > 0:
                    print(f"[DEBUG] First result keys: {original_results[0].keys()}")

            # We only want to re-rank the candidates that passed the bi-encoder filtering
            for result in original_results:
                if debug:
                    print(
                        f"[DEBUG] Processing result: {result.get('hpo_id', 'unknown')} with type {type(result)}"
                    )

                # Extract the HPO ID from the result
                hpo_id = result.get("hpo_id", None)
                if not hpo_id:
                    if debug:
                        print(f"[DEBUG] Missing hpo_id in result: {result}")
                    continue

                # Find the full document and metadata for this ID
                # Use the HPO ID as-is since we store them with the same format in ChromaDB
                chroma_compatible_id = (
                    hpo_id  # No need to replace colons with underscores
                )
                if debug:
                    print(f"[DEBUG] Looking for HPO ID: {chroma_compatible_id}")

                found = False
                for i, doc_id in enumerate(results["ids"][0]):
                    if doc_id == chroma_compatible_id:
                        found = True
                        if debug:
                            print(f"[DEBUG] Found matching document for {hpo_id}")
                        original_rank = original_results.index(result) + 1
                        # Get metadata for this document
                        metadata = results["metadatas"][0][i]

                        # Get the document text to use for comparison based on the reranker mode
                        if reranker_mode == "monolingual":
                            # For monolingual mode, load the translation of the HPO term
                            translated_text = load_translation_text(
                                hpo_id, translation_dir
                            )

                            if translated_text is None:
                                if debug:
                                    print(
                                        f"[DEBUG] No translation found for {hpo_id}, skipping"
                                    )
                                continue

                            if debug:
                                print(
                                    f"[DEBUG] Loaded translation for {hpo_id}: {translated_text[:50]}..."
                                )

                            comparison_text = translated_text

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
                    print(f"[DEBUG] Could not find document for {hpo_id} in results")

            if debug:
                print(f"[DEBUG] Candidates for re-ranking: {len(candidates)}")
                if candidates:
                    print(f"[DEBUG] First candidate keys: {candidates[0].keys()}")
                    print(
                        f"[DEBUG] First candidate simplified english_doc (raw label): {candidates[0]['english_doc']}"
                    )
                    print(
                        f"[DEBUG] Original document: {candidates[0]['original_document'][:50]}..."
                    )
                else:
                    print(f"[DEBUG] No candidates to re-rank")
                    print(
                        f"[DEBUG] Results shape: ids={len(results['ids'][0])}, documents={len(results['documents'][0])}, metadatas={len(results['metadatas'][0])}"
                    )
                    # Sample a few IDs for comparison
                    sample_ids = results["ids"][0][:5]
                    print(f"[DEBUG] Sample IDs from results: {sample_ids}")
                    print(
                        f"[DEBUG] Original result IDs: {[r.get('hpo_id', 'unknown') for r in original_results[:5]]}"
                    )
                    # Try to match them directly
                    for orig_result in original_results[:5]:
                        orig_id = orig_result.get("hpo_id", "unknown")
                        found = orig_id in sample_ids
                        print(
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
            print("\n---------- Original Results (Bi-Encoder) ----------")
            if original_formatted and original_formatted["results"]:
                print_results(original_formatted)
            else:
                print("No matching HPO terms found.")

            print("\n---------- Re-Ranked Results (Cross-Encoder) ----------")
            formatted = reranked_formatted
        else:
            # Just use the original results
            formatted = original_formatted

        if formatted and formatted["results"]:
            print_results(formatted)
        else:
            print("\nNo matching HPO terms found.")


def process_single_query(
    query_text: str,
    retriever: DenseRetriever,
    num_results: int = DEFAULT_TOP_K,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    cross_encoder=None,
    rerank_count: int = None,
) -> None:
    """
    Process a single query with optional re-ranking.

    Args:
        query_text: The text to query
        retriever: DenseRetriever instance for querying
        num_results: Number of results to display
        similarity_threshold: Minimum similarity threshold for filtering results
        cross_encoder: Optional cross-encoder model for re-ranking
        rerank_count: Number of candidates to re-rank
    """
    # Determine how many results to retrieve initially
    if cross_encoder and rerank_count is not None:
        initial_result_count = rerank_count * 2
    else:
        initial_result_count = num_results * 3

    # Query with the bi-encoder
    logging.info(f"Querying with text: '{query_text}'")
    results = retriever.query(query_text, n_results=initial_result_count)

    # Apply cross-encoder re-ranking if enabled
    if cross_encoder and rerank_count and results["ids"] and results["ids"][0]:
        logging.info(
            f"Re-ranking top {min(rerank_count, len(results['ids'][0]))} results"
        )

        # Prepare candidates list for re-ranking
        candidates = []
        for i, (doc_id, metadata, distance) in enumerate(
            zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
        ):
            if i >= rerank_count:
                break

            # Create English document text from metadata
            english_doc = f"{metadata['label']}. {metadata.get('definition', '')}"
            if metadata.get("synonyms"):
                english_doc += f" Synonyms: {metadata['synonyms']}"

            # Store candidate information
            candidates.append(
                {
                    "id": doc_id,
                    "metadata": metadata,
                    "distance": distance,
                    "english_doc": english_doc,
                }
            )

        # Only proceed with re-ranking if we have candidates
        if candidates:
            try:
                # Perform re-ranking
                reranked_candidates = reranker.rerank_with_cross_encoder(
                    query=query_text,
                    candidates=candidates,
                    cross_encoder_model=cross_encoder,
                )

                # Reconstruct results in re-ranked order
                reranked_results = {
                    "ids": [[c["id"] for c in reranked_candidates]],
                    "metadatas": [[]],
                    "distances": [[]],
                }

                # Add cross-encoder scores to the metadata
                for candidate in reranked_candidates:
                    # Create a copy of the metadata and add cross-encoder score
                    metadata_copy = candidate["metadata"].copy()
                    metadata_copy["cross_encoder_score"] = candidate[
                        "cross_encoder_score"
                    ]
                    reranked_results["metadatas"][0].append(metadata_copy)

                    # Keep the original distance for reference
                    reranked_results["distances"][0].append(candidate["distance"])

                # Format and display re-ranked results
                formatted = format_results(
                    results=reranked_results,
                    threshold=similarity_threshold,
                    max_results=num_results,
                    query=query_text,
                    reranked=True,
                )
                print(formatted)
                return

            except Exception as e:
                logging.error(f"Error during re-ranking: {e}")
                logging.warning("Falling back to bi-encoder results")

    # Format and display regular bi-encoder results if no re-ranking or if re-ranking failed
    formatted = format_results(
        results=results,
        threshold=similarity_threshold,
        max_results=num_results,
        query=query_text,
    )
    print(formatted)


def main() -> None:
    """Main function for the interactive query tool."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Interactive HPO query tool for clinical text"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to process (if not provided, interactive mode is used)",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to show (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--sentence-mode",
        action="store_true",
        help="Process input text sentence by sentence (helps with longer texts)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=MIN_SIMILARITY_THRESHOLD,
        help=f"Minimum similarity threshold (default: {MIN_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Sentence transformer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging with more verbose output",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for model loading (required for some models)",
    )

    # Cross-encoder re-ranking arguments
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help=f"Enable cross-encoder re-ranking of results (default: {DEFAULT_ENABLE_RERANKER})",
    )
    parser.add_argument(
        "--reranker-mode",
        type=str,
        choices=["cross-lingual", "monolingual"],
        default=DEFAULT_RERANKER_MODE,
        help=f"Mode for re-ranking: cross-lingual (source->English) or monolingual (source->source) (default: {DEFAULT_RERANKER_MODE})",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=DEFAULT_RERANKER_MODEL,
        help=f"Cross-encoder model to use for re-ranking (default: {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--monolingual-reranker-model",
        type=str,
        default=DEFAULT_MONOLINGUAL_RERANKER_MODEL,
        help=f"Language-specific cross-encoder model for monolingual re-ranking (default: {DEFAULT_MONOLINGUAL_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--translation-dir",
        type=str,
        default=DEFAULT_TRANSLATION_DIR,
        help=f"Directory containing HPO term translations in target language (default: {DEFAULT_TRANSLATION_DIR})",
    )
    parser.add_argument(
        "--rerank-count",
        type=int,
        default=DEFAULT_RERANK_CANDIDATE_COUNT,
        help=f"Number of candidates to re-rank (default: {DEFAULT_RERANK_CANDIDATE_COUNT})",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    # Load embedding model
    try:
        model = load_embedding_model(
            model_name=args.model_name, trust_remote_code=args.trust_remote_code
        )
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)

    # Initialize retriever
    retriever = DenseRetriever.from_model_name(
        model=model,
        model_name=args.model_name,
        index_dir=INDEX_DIR,
        min_similarity=args.similarity_threshold,
    )

    if not retriever:
        logging.error(
            f"Failed to initialize retriever. Make sure you have built an index for {args.model_name}"
        )
        sys.exit(1)

    # Load cross-encoder model if re-ranking is enabled
    cross_encoder = None
    if args.enable_reranker:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Select the appropriate model based on the reranker mode
        model_name = args.reranker_model
        if args.reranker_mode == "monolingual":
            # For monolingual mode, use the language-specific model
            model_name = args.monolingual_reranker_model

            # Check if translation directory exists
            if not os.path.exists(args.translation_dir):
                logging.warning(
                    f"Translation directory not found: {args.translation_dir}. "
                    "Monolingual re-ranking will not work properly without translations."
                )

        # Load the selected cross-encoder model
        cross_encoder = reranker.load_cross_encoder(model_name, device)
        if cross_encoder:
            logging.info(
                f"Cross-encoder re-ranking enabled in {args.reranker_mode} mode with model: {model_name}"
            )
        else:
            logging.warning(
                f"Failed to load cross-encoder model {model_name}, re-ranking will be disabled"
            )

    # Get collection information
    collection_name = generate_collection_name(args.model_name)
    collection_count = retriever.collection.count()

    # Display summary
    print(f"Model: {args.model_name}")
    print(f"Collection: {collection_name}")
    print(f"Index entries: {collection_count}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    if cross_encoder:
        print(f"Cross-encoder re-ranking: Enabled (using {args.reranker_model})")
    else:
        print("Cross-encoder re-ranking: Disabled")

    # Process a one-time query if provided via command line
    if args.text:
        process_input(
            args.text,
            retriever,
            args.num_results,
            args.sentence_mode,
            args.similarity_threshold,
            debug=args.debug,
            cross_encoder=cross_encoder,
            rerank_count=args.rerank_count if args.enable_reranker else None,
            reranker_mode=args.reranker_mode,
            translation_dir=args.translation_dir,
        )
        return

    # Interactive mode
    print("\n===== Phentrieve HPO RAG Query Tool =====")
    print("Enter clinical descriptions to find matching HPO terms.")
    print("Type 'exit', 'quit', or 'q' to exit the program.\n")

    while True:
        try:
            user_input = input("\nEnter text (or 'q' to quit): ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting.")
                break

            if not user_input.strip():
                continue

            # Print debug information
            if args.debug:
                print(f"[DEBUG] Using num_results: {args.num_results}")
                print(
                    f"[DEBUG] Using rerank_count: {args.rerank_count if args.enable_reranker else None}"
                )

            process_input(
                user_input,
                retriever,
                args.num_results,
                args.sentence_mode,
                args.similarity_threshold,
                debug=args.debug,
                cross_encoder=cross_encoder,
                rerank_count=args.rerank_count if args.enable_reranker else None,
                reranker_mode=args.reranker_mode,
                translation_dir=args.translation_dir,
            )
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            # Print detailed traceback for debugging
            print("\nDetailed error information:")
            traceback.print_exc()
            print()
            continue


if __name__ == "__main__":
    main()
