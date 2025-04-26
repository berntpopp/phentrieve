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
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import pysbd

from multilingual_hpo_rag.config import (
    DEFAULT_MODEL,
    INDEX_DIR,
    MIN_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)
from multilingual_hpo_rag.embeddings import load_embedding_model
from multilingual_hpo_rag.retrieval.dense_retriever import (
    DenseRetriever,
    calculate_similarity,
)
from multilingual_hpo_rag.utils import generate_collection_name


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
) -> str:
    """
    Format the query results for display, filtering by similarity threshold.

    Args:
        results: Raw results dictionary from retriever.query
        threshold: Minimum similarity score to display
        max_results: Maximum number of results to display

    Returns:
        Formatted string for display
    """
    if not results or not results.get("ids") or not results["ids"][0]:
        return "No matching HPO terms found."

    formatted_output = []

    # Create a list of (HPO ID, label, similarity) tuples
    result_tuples = []
    for i, (doc_id, metadata, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        # Calculate similarity from distance
        similarity = calculate_similarity(distance)

        # Get HPO ID and label from metadata
        hpo_id = metadata.get("hpo_id", "Unknown")
        label = metadata.get("label", "Unknown")

        # Add to result tuples
        result_tuples.append((hpo_id, label, similarity))

    # Sort by similarity (highest first)
    result_tuples.sort(key=lambda x: x[2], reverse=True)

    # Filter by threshold and limit to max_results
    filtered_results = [res for res in result_tuples if res[2] >= threshold][
        :max_results
    ]

    # Format the results
    if not filtered_results:
        return "No results above the similarity threshold."

    formatted_output.append(f"Found {len(filtered_results)} matching HPO terms:")

    # Format each result
    for i, (hpo_id, label, similarity) in enumerate(filtered_results):
        formatted_output.append(
            f"{i+1}. {hpo_id} - {label} (similarity: {similarity:.3f})"
        )

    return "\n".join(formatted_output)


def process_input(
    text: str,
    retriever: DenseRetriever,
    num_results: int = DEFAULT_TOP_K,
    sentence_mode: bool = False,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    debug: bool = False,
) -> None:
    """
    Process input text, either as a whole or sentence by sentence.

    Args:
        text: The input text to process
        retriever: DenseRetriever instance for querying
        num_results: Number of results to display for each query
        sentence_mode: Whether to process text sentence by sentence
        similarity_threshold: Minimum similarity threshold for results
        debug: Whether to enable debug logging
    """
    if sentence_mode:
        sentences = segment_text(text)
        logging.info(f"Text split into {len(sentences)} sentences")

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            print(f"\nSentence {i+1}: {sentence}")

            # Query for this sentence
            results = retriever.query(sentence, n_results=num_results * 3)

            # Format and display results
            formatted = format_results(
                results, threshold=similarity_threshold, max_results=num_results
            )
            print(formatted)
    else:
        # Query for the entire text
        results = retriever.query(text, n_results=num_results * 3)

        # Format and display results
        formatted = format_results(
            results, threshold=similarity_threshold, max_results=num_results
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

    # Get collection information
    collection_name = generate_collection_name(args.model_name)
    collection_count = retriever.collection.count()

    # Display summary
    print(f"Model: {args.model_name}")
    print(f"Collection: {collection_name}")
    print(f"Index entries: {collection_count}")
    print(f"Similarity threshold: {args.similarity_threshold}")

    # Process a one-time query if provided via command line
    if args.text:
        process_input(
            args.text,
            retriever,
            args.num_results,
            args.sentence_mode,
            args.similarity_threshold,
            debug=args.debug,
        )
        return

    # Interactive mode
    print("\n===== Multilingual HPO RAG Query Tool =====")
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

            process_input(
                user_input,
                retriever,
                args.num_results,
                args.sentence_mode,
                args.similarity_threshold,
                debug=args.debug,
            )
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            logging.error(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
