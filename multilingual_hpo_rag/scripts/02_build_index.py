#!/usr/bin/env python3
"""
HPO Index Building Script

This script builds a ChromaDB vector index for HPO terms using the specified
embedding model.

It performs the following steps:
1. Load HPO terms from individual JSON files
2. Create descriptive documents for each HPO term
3. Load the specified embedding model
4. Build a ChromaDB index with the embeddings
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from multilingual_hpo_rag.config import DEFAULT_MODEL, INDEX_DIR, BENCHMARK_MODELS
from multilingual_hpo_rag.data_processing.document_creator import (
    load_hpo_terms,
    create_hpo_documents,
)
from multilingual_hpo_rag.embeddings import load_embedding_model
from multilingual_hpo_rag.indexing.chromadb_indexer import build_chromadb_index
from multilingual_hpo_rag.utils import get_model_slug, generate_collection_name


def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    """Main function for building the HPO index."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Build a ChromaDB index for HPO terms using a multilingual sentence transformer model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Sentence transformer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Build indices for all models defined in BENCHMARK_MODELS",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to process at once (default: 100)",
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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the index even if it already exists",
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    # Record start time
    start_time = time.time()

    # 1. Load HPO terms
    logging.info("Loading HPO terms...")
    hpo_terms = load_hpo_terms()
    if not hpo_terms:
        logging.error("Failed to load HPO terms. Run 01_prepare_hpo_data.py first.")
        sys.exit(1)

    # 2. Create documents for indexing
    logging.info("Creating HPO documents for indexing...")
    documents, metadatas, ids = create_hpo_documents(hpo_terms)

    # 3. Load embedding model
    logging.info(f"Loading embedding model: {args.model_name}")
    try:
        model = load_embedding_model(
            model_name=args.model_name, trust_remote_code=args.trust_remote_code
        )
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)

    # Make sure index directory exists
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 4. Build the index
    logging.info(f"Building ChromaDB index with {len(documents)} documents...")

    # Track success/failure count
    success_count = 0
    failure_count = 0
    processed_models = []

    if args.all_models:
        # Build indices for all benchmark models
        logging.info(f"Building indices for {len(BENCHMARK_MODELS)} models")

        for model_name in BENCHMARK_MODELS:
            logging.info(f"Building index for model: {model_name}")
            try:
                # Load model for each iteration
                current_model = load_embedding_model(
                    model_name=model_name,
                    trust_remote_code=args.trust_remote_code,
                    device="cpu" if args.cpu else None,
                )

                result = build_chromadb_index(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    model=current_model,
                    model_name=model_name,
                    batch_size=args.batch_size,
                    recreate=args.recreate,
                )

                if result:
                    logging.info(f"✓ Index built successfully for model: {model_name}")
                    success_count += 1
                    processed_models.append(model_name)
                else:
                    logging.error(f"✗ Failed to build index for model: {model_name}")
                    failure_count += 1
            except Exception as e:
                logging.error(f"✗ Error building index for {model_name}: {e}")
                failure_count += 1
                if args.debug:
                    import traceback

                    traceback.print_exc()
    else:
        # Build index for a single model
        try:
            result = build_chromadb_index(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                model=model,
                model_name=args.model_name,
                batch_size=args.batch_size,
                recreate=args.recreate,
            )

            if result:
                logging.info(f"✓ Index built successfully for model: {args.model_name}")
                success_count += 1
                processed_models.append(args.model_name)
            else:
                logging.error(f"✗ Failed to build index for model: {args.model_name}")
                failure_count += 1
        except Exception as e:
            logging.error(f"Error building index: {e}")
            failure_count += 1
            if args.debug:
                import traceback

                traceback.print_exc()

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Report results
    if success_count > 0:
        logging.info(
            f"Index building completed: {success_count} successful, {failure_count} failed"
        )
        logging.info(f"Successfully built indices for: {', '.join(processed_models)}")
        logging.info(f"Index building completed in {elapsed_time:.2f} seconds!")
        logging.info(f"Index location: {os.path.abspath(INDEX_DIR)}")
        logging.info(f"Document count: {len(documents)}")
        logging.info(f"You can now use the interactive query tool or run benchmarks.")
        return True
    else:
        logging.error("All index building attempts failed.")
        return False


if __name__ == "__main__":
    main()
