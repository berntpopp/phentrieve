import chromadb
import os
import sys
import re
import pysbd
import argparse
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import (
    get_model_slug,
    get_index_dir,
    generate_collection_name,
    get_embedding_dimension,
)
import torch


# Set up device - use CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# Logging will be configured based on debug flag
def configure_logging(debug=False):
    """Configure logging based on debug flag"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    # Also set root logger level
    logging.getLogger().setLevel(level)

    if debug:
        logging.debug("Debug logging enabled in german_hpo_rag.py")


# Default to INFO level initially
configure_logging(False)

# Default values
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to display results


def calculate_similarity(distance):
    """Converts cosine distance (0 to 2) to similarity score (1 to -1, clamped to 0-1)."""
    # Cosine distance = 1 - Cosine Similarity
    # Similarity = 1 - Cosine Distance
    similarity = 1.0 - distance
    # Clamp the result between 0.0 and 1.0 as similarity scores typically range from 0 to 1
    # (though cosine similarity technically ranges from -1 to 1, negative values are unlikely here)
    return max(0.0, min(1.0, similarity))


def query_hpo(sentence, model, collection, n_results=10, debug=False):
    """Generates embedding and queries the HPO index.

    Args:
        sentence (str): The German input sentence
        model: Loaded SentenceTransformer model instance
        collection: ChromaDB collection instance
        n_results (int): Number of results to fetch initially

    Returns:
        dict: ChromaDB query results dictionary
    """
    logging.info(f"Query: '{sentence}'")
    if debug:
        logging.debug(
            f"Using model: {model.__class__.__name__}, n_results: {n_results}"
        )

    try:
        # Generate embedding for the query sentence
        query_embedding = model.encode([sentence])[
            0
        ]  # Encode returns a list, get the first element

        # Query the collection - get more results than we need to filter by similarity
        query_n_results = n_results * 3  # Get more results to allow better filtering
        if debug:
            logging.debug(f"Querying collection with {query_n_results} initial results")

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=query_n_results,
            include=["documents", "metadatas", "distances"],
        )

        if debug and results:
            logging.debug(f"Got {len(results['ids'][0])} results from collection")

        return results
    except Exception as e:
        logging.error(f"Error querying HPO: {e}")
        return None


def format_results(results, threshold=MIN_SIMILARITY_THRESHOLD, max_results=5):
    """Format the query results for display, filtering by similarity threshold.

    Args:
        results (dict): Raw results dictionary from collection.query
        threshold (float): Minimum similarity score to display
        max_results (int): Maximum number of results to display

    Returns:
        str: Formatted string for display
    """
    if not results or not results["ids"] or not results["ids"][0]:
        return "No matching HPO terms found."

    formatted_output = []
    count = 0

    # Print raw distances for debugging
    raw_distances = results["distances"][0][:5]  # Just look at first 5
    raw_similarities = [calculate_similarity(d) for d in raw_distances]
    formatted_output.append(f"DEBUG - Raw distances: {raw_distances}")
    formatted_output.append(f"DEBUG - Raw similarities: {raw_similarities}")
    formatted_output.append(f"DEBUG - Similarity threshold: {threshold}\n")

    # Prepare items with calculated similarity scores
    items = []
    for i, (hpo_id, metadata, distance, document) in enumerate(
        zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["documents"][0],
        )
    ):
        # Calculate similarity score from distance (cosine distance)
        similarity_score = calculate_similarity(distance)
        items.append((similarity_score, hpo_id, metadata, document))

    # Sort by similarity score (descending)
    items.sort(reverse=True)

    # Debug top 5 items regardless of threshold
    formatted_output.append("DEBUG - Top 5 items regardless of threshold:")
    for i, (similarity_score, hpo_id, metadata, document) in enumerate(items[:5]):
        formatted_output.append(
            f"DEBUG {i+1}. {metadata['hpo_id']} - {metadata['hpo_name']}\n"
            f"   Similarity: {similarity_score:.5f}"
        )
    formatted_output.append("")

    # Format the regular results
    count = 0
    regular_results = []
    for similarity_score, hpo_id, metadata, document in items:
        # Skip results with low similarity
        if similarity_score < threshold:
            continue

        # Limit to max_results
        if count >= max_results:
            break

        count += 1

        # Get definition directly from metadata
        definition = metadata.get("definition", "No definition")
        if definition:
            # Truncate definition to first sentence for display clarity
            if "." in definition:
                definition = definition.split(".", 1)[0] + "."

        # Get synonyms directly from metadata
        synonyms = ""
        synonyms_text = metadata.get("synonyms_text", "")
        if synonyms_text:
            # Display the first few synonyms for readability
            syn_parts = synonyms_text.split("; ", 3)
            if len(syn_parts) > 3:
                display_syns = "; ".join(syn_parts[:3])
                count = metadata.get("synonyms_count", 0) - 3
                if count > 0:
                    display_syns += f" (+ {count} more)"
            else:
                display_syns = synonyms_text

            synonyms = f"\n   Synonyms: {display_syns}"

        # Format the result with more detail
        regular_results.append(
            f"{count}. {metadata['hpo_id']} - {metadata['hpo_name']}\n"
            f"   Similarity: {similarity_score:.5f}\n"
            f"   Definition: {definition}{synonyms}"
        )

    if regular_results:
        formatted_output.append("REGULAR RESULTS (with threshold filtering):")
        formatted_output.extend(regular_results)
    else:
        formatted_output.append(
            "No matching HPO terms found with sufficient similarity.\n\nTry lowering the similarity threshold with --similarity-threshold."
        )

    return "\n\n".join(formatted_output)


def segment_text(text, lang="de"):
    """Split text into sentences."""
    segmenter = pysbd.Segmenter(language=lang, clean=False)
    return segmenter.segment(text)


def process_input(
    text,
    model,
    collection,
    num_results=5,
    sentence_mode=False,
    similarity_threshold=MIN_SIMILARITY_THRESHOLD,
    debug=False,
):
    """Process input text, either as a whole or sentence by sentence.

    Args:
        text (str): The input text to process
        model: Loaded SentenceTransformer model instance
        collection: ChromaDB collection instance
        num_results (int): Number of results to display per query
        sentence_mode (bool): Whether to process text sentence by sentence
        similarity_threshold (float): Minimum similarity threshold for results
    """
    if not sentence_mode:
        print("\n[Processing text as a whole]")
        results = query_hpo(
            text.strip(), model, collection, num_results * 2, debug=debug
        )
        print("\nMatches:\n")
        print(
            format_results(
                results, threshold=similarity_threshold, max_results=num_results
            )
        )
    else:
        # Process text sentence by sentence
        sentences = segment_text(text)
        if len(sentences) > 1:
            print(f"\nText segmented into {len(sentences)} sentences.")

        # Dictionary to track the best similarity score for each HPO ID
        aggregated_results = {}
        sentence_results = []

        # Process each sentence and collect results
        for i, sentence in enumerate(sentences, 1):
            if len(sentences) > 1:
                print(f"\n--- Sentence {i}/{len(sentences)} ---")
                print(f'"{sentence}"\n')

            results = query_hpo(
                sentence, model, collection, num_results * 2
            )  # Get more results to aggregate

            # Show individual sentence results if multiple sentences
            if len(sentences) > 1:
                print(
                    format_results(
                        results, threshold=similarity_threshold, max_results=num_results
                    )
                )

            # Store the results for aggregation
            sentence_results.append((sentence, results))

            # Process results for aggregation
            if results and results["ids"] and results["ids"][0]:
                for hpo_id, metadata, distance, document in zip(
                    results["ids"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                    results["documents"][0],
                ):
                    similarity = calculate_similarity(distance)

                    # Skip results below threshold
                    if similarity < similarity_threshold:
                        continue

                    # Update aggregated results with max similarity
                    if (
                        metadata["hpo_id"] not in aggregated_results
                        or similarity
                        > aggregated_results[metadata["hpo_id"]]["similarity"]
                    ):
                        aggregated_results[metadata["hpo_id"]] = {
                            "similarity": similarity,
                            "metadata": metadata,
                            "document": document,
                            "source_sentence": sentence,
                        }

        # If we processed multiple sentences, show aggregated results
        if len(sentences) > 1 and aggregated_results:
            print("\n\n===== AGGREGATED RESULTS ACROSS ALL SENTENCES =====\n")

            # Convert to list and sort by similarity
            ranked_results = list(aggregated_results.values())
            ranked_results.sort(key=lambda x: x["similarity"], reverse=True)

            # Format and display aggregated results
            for i, result in enumerate(ranked_results[:num_results], 1):
                metadata = result["metadata"]
                similarity = result["similarity"]

                # Get definition from metadata
                definition = metadata.get("definition", "No definition")
                if definition and "." in definition:
                    definition = definition.split(".", 1)[0] + "."

                # Get synonyms from metadata
                synonyms = ""
                synonym_list = metadata.get("synonyms", [])
                if synonym_list:
                    synonyms = f"\n   Synonyms: {'; '.join(synonym_list[:3])}"
                    if len(synonym_list) > 3:
                        synonyms += f" (+ {len(synonym_list) - 3} more)"

                print(f"{i}. {metadata['hpo_id']} - {metadata['hpo_name']}")
                print(f"   Similarity: {similarity:.5f}")
                print(f"   Definition: {definition}{synonyms}")
                print(f"   Found in sentence: \"{result['source_sentence']}\"\n")


def connect_to_chroma(index_dir, collection_name, model_name=None):
    """Connect to the ChromaDB index.

    Args:
        index_dir: Directory where ChromaDB indices are stored
        collection_name: Name of the collection to connect to
        model_name: Optional model name to handle dimension-specific collections
    """
    logging.info(f"Connecting to ChromaDB at {index_dir}")

    try:
        client = chromadb.PersistentClient(path=index_dir)

        # Only use the model-specific collection - no fallback
        try:
            logging.info(f"Using model-specific collection: {collection_name}")
            collection = client.get_collection(name=collection_name)
            return collection
        except ValueError as e:
            logging.error(f"Error: Collection '{collection_name}' not found")
            logging.error(
                f"You need to run setup_hpo_index.py with model {model_name} to create a collection compatible with its embedding dimension {get_embedding_dimension(model_name) if model_name else 'unknown'}."
            )
            return None
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {e}")
        return None


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Query HPO terms using German clinical descriptions."
    )
    parser.add_argument(
        "-t",
        "--text",
        help="German text to query (if not provided, will use interactive mode)",
    )
    parser.add_argument(
        "-n",
        "--num-results",
        type=int,
        default=5,
        help="Number of HPO terms to retrieve (default: 5)",
    )
    parser.add_argument(
        "-s",
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
    args = parser.parse_args()

    # Configure logging based on debug flag
    configure_logging(args.debug)
    if args.debug:
        logging.debug("Debug mode enabled")

    # Get index directory and collection name based on model
    index_dir = get_index_dir()
    collection_name = generate_collection_name(args.model_name)
    model_slug = get_model_slug(args.model_name)

    # Check if index exists
    if not os.path.exists(index_dir):
        logging.error(
            f"Error: Index directory '{index_dir}' not found. Please run setup_hpo_index.py first."
        )
        sys.exit(1)

    logging.info(f"Loading embedding model: {args.model_name}")
    try:
        # Special handling for Jina model which requires trust_remote_code=True
        jina_model_id = "jinaai/jina-embeddings-v2-base-de"
        if args.model_name == jina_model_id:
            logging.info(
                f"Loading Jina model '{args.model_name}' with trust_remote_code=True on {device}"
            )
            # Security note: Only use trust_remote_code=True for trusted sources
            model = SentenceTransformer(args.model_name, trust_remote_code=True)
        else:
            logging.info(f"Loading model '{args.model_name}' on {device}")
            model = SentenceTransformer(args.model_name)

        # Move model to GPU if available
        model = model.to(device)
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer model: {e}")
        logging.error("Make sure you have run: pip install -r requirements.txt")
        sys.exit(1)
    logging.info("Model loaded successfully.")

    # Connect to ChromaDB
    collection = connect_to_chroma(index_dir, collection_name, args.model_name)
    if not collection:
        sys.exit(1)

    # Display summary
    print(f"Model: {args.model_name}")
    print(f"Collection: {collection_name}")
    print(f"Index entries: {collection.count()}")
    print(f"Similarity threshold: {args.similarity_threshold}")

    # Process a one-time query if provided via command line
    if args.text:
        process_input(
            args.text,
            model,
            collection,
            args.num_results,
            args.sentence_mode,
            args.similarity_threshold,
            debug=args.debug,
        )
        return

    # Interactive mode
    print("\n===== German HPO RAG Query Tool =====")
    print("Enter German clinical descriptions to find matching HPO terms.")
    print("Type 'exit', 'quit', or 'q' to exit the program.\n")

    while True:
        try:
            user_input = input("\nEnter German text (or 'q' to quit): ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting.")
                break

            if not user_input.strip():
                continue

            process_input(
                user_input,
                model,
                collection,
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
