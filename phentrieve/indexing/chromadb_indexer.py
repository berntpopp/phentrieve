"""
ChromaDB index management for HPO terms.

This module provides functionality for building and managing ChromaDB vector indices
for HPO terms, allowing efficient semantic search.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from phentrieve.utils import (
    generate_collection_name,
    get_embedding_dimension,
    get_model_slug,
)


def build_chromadb_index(
    documents: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    model: SentenceTransformer,
    model_name: str,
    batch_size: int = 100,
    recreate: bool = False,
    index_dir: Path = None,
) -> bool:
    """
    Build a ChromaDB index for the given documents using the specified embedding model.

    Args:
        documents: List of document strings to index
        metadatas: List of metadata dictionaries
        ids: List of document IDs
        model: SentenceTransformer model instance
        model_name: Name of the model (used for collection naming)
        batch_size: Number of documents to process at once
        recreate: Whether to recreate the collection if it exists

    Returns:
        bool: True if indexing was successful, False otherwise
    """
    if not documents:
        logging.error("No documents provided for indexing")
        return False

    # Get collection name (index_dir should be passed in)
    collection_name = generate_collection_name(model_name)
    get_model_slug(model_name)

    # Make sure index directory exists
    os.makedirs(index_dir, exist_ok=True)

    # Record start time for performance measurement
    start_time = time.time()

    # Initialize ChromaDB
    logging.info(f"Initializing ChromaDB at {index_dir}")
    try:
        # Convert Path to string and ensure it exists
        index_dir_str = str(index_dir)
        os.makedirs(index_dir_str, exist_ok=True)

        # Initialize with proper settings to avoid tenant issues
        client = chromadb.PersistentClient(
            path=index_dir_str,
            settings=chromadb.Settings(
                anonymized_telemetry=False, allow_reset=True, is_persistent=True
            ),
        )

        # Initialize skip collection creation flag
        skip_collection_creation = False

        # Check if collection already exists
        try:
            existing_collection = client.get_collection(name=collection_name)

            if recreate:
                logging.info(f"Deleting existing collection: {collection_name}")
                client.delete_collection(name=collection_name)
            else:
                logging.info(
                    f"Collection {collection_name} already exists with {existing_collection.count()} documents"
                )
                if existing_collection.count() > 0:
                    logging.info(
                        "Using existing collection (use recreate=True to rebuild)"
                    )
                    return True
                else:
                    logging.info("Collection exists but is empty, will populate it")
                    # Use the existing collection instead of creating a new one
                    collection = existing_collection
                    # Set a flag to skip collection creation below
                    skip_collection_creation = True
        except Exception as e:
            # Collection didn't exist or some other error
            logging.debug(f"Note: {e}")

        # Check model embedding dimension
        model_dimension = get_embedding_dimension(model_name)
        logging.info(
            f"Using embedding dimension {model_dimension} for model {model_name}"
        )

        # Create a new collection with specified metadata if needed
        if not skip_collection_creation:
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "hpo_version": "latest",
                    "model": model_name,
                    "dimension": model_dimension,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hnsw:space": "cosine",
                },
            )
        if not skip_collection_creation:
            logging.info(
                f"Created new collection: {collection_name} with dimension {model_dimension}"
            )
        else:
            logging.info(
                f"Using existing collection: {collection_name} with dimension {model_dimension}"
            )
    except Exception as e:
        logging.error(f"Error initializing ChromaDB: {e}")
        return False

    # Get the device the model is on
    device = next(model.parameters()).device
    device_name = "cuda" if device.type == "cuda" else "cpu"

    # Generate embeddings and add to ChromaDB in batches
    total_batches = len(documents) // batch_size + (
        1 if len(documents) % batch_size > 0 else 0
    )
    logging.info(
        f"Computing embeddings for {len(documents)} HPO terms using {device_name}..."
    )

    # Add a clear progress bar for batch processing
    for i in tqdm(
        range(0, len(documents), batch_size),
        desc="Batches",
        total=total_batches,
        unit="batch",
    ):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        try:
            # Generate embeddings for the batch
            embeddings = model.encode(batch_docs, device=device_name)

            # Add to ChromaDB
            collection.add(
                documents=batch_docs,
                embeddings=embeddings.tolist(),
                metadatas=batch_meta,
                ids=batch_ids,
            )
        except Exception as e:
            logging.error(
                f"Error processing batch {i // batch_size + 1}/{total_batches}: {e}"
            )
            continue

    end_time = time.time()
    logging.info(f"Index built successfully in {end_time - start_time:.2f} seconds!")
    logging.info(f"Indexed {len(documents)} HPO terms.")
    logging.info(f"Index location: {os.path.abspath(str(index_dir))}")
    return True
