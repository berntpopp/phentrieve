"""
Index building orchestration module.

This module provides orchestration functions for building ChromaDB indexes
from HPO term data using embedding models.
"""

import logging
import os
import time
from typing import Optional

from phentrieve.config import BENCHMARK_MODELS, DEFAULT_MODEL
from phentrieve.data_processing.document_creator import (
    create_hpo_documents,
    load_hpo_terms,
)
from phentrieve.embeddings import load_embedding_model
from phentrieve.indexing.chromadb_indexer import build_chromadb_index
from phentrieve.utils import get_default_index_dir, resolve_data_path


def orchestrate_index_building(
    model_name_arg: Optional[str] = None,
    all_models: bool = False,
    batch_size: int = 100,
    trust_remote_code: bool = False,
    device_override: Optional[str] = None,
    recreate: bool = False,
    debug: bool = False,
    index_dir_override: Optional[str] = None,
    data_dir_override: Optional[str] = None,
) -> bool:
    """Orchestrates loading data, models, and building ChromaDB indexes.

    Args:
        model_name_arg: Name of the model to use for embeddings, or None to use default
        all_models: Whether to build indices for all benchmark models
        batch_size: Number of documents to process at once
        trust_remote_code: Whether to trust remote code when loading models
        device_override: Device to use ('cpu', 'cuda', etc.), or None for auto-detection
        recreate: Whether to recreate the index even if it exists
        debug: Enable debug logging

    Returns:
        True if all requested indexes were built successfully, False otherwise
    """
    start_time = time.time()
    logging.info("Loading HPO terms for indexing...")
    hpo_terms = load_hpo_terms(data_dir_override=data_dir_override)
    if not hpo_terms:
        logging.error("Failed to load HPO terms. Run 'phentrieve data prepare' first.")
        return False

    logging.info("Creating HPO documents for indexing...")
    documents, metadatas, ids = create_hpo_documents(hpo_terms)
    if not documents:
        logging.error("Failed to create documents from HPO terms.")
        return False

    # Resolve index directory path based on priority: CLI > Config > Default
    index_dir = resolve_data_path(
        index_dir_override, "index_dir", get_default_index_dir
    )
    logging.info(f"Using index directory: {index_dir}")
    os.makedirs(index_dir, exist_ok=True)

    models_to_process = []
    if all_models:
        models_to_process = BENCHMARK_MODELS
        logging.info(
            f"Building indices for all {len(models_to_process)} benchmark models."
        )
    elif model_name_arg:
        models_to_process = [model_name_arg]
    else:
        models_to_process = [DEFAULT_MODEL]  # Default if nothing specified

    success_count = 0
    failure_count = 0
    processed_model_names = []

    for model_name in models_to_process:
        logging.info(f"--- Processing model: {model_name} ---")
        try:
            model = load_embedding_model(
                model_name=model_name,
                trust_remote_code=trust_remote_code,
                device=device_override,  # Pass CPU/GPU preference
            )
            if not model:
                raise ValueError(f"Failed to load model {model_name}")

            result = build_chromadb_index(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                model=model,
                model_name=model_name,
                batch_size=batch_size,
                recreate=recreate,
                index_dir=index_dir,
            )
            if result:
                logging.info(f"✓ Index built successfully for model: {model_name}")
                success_count += 1
                processed_model_names.append(model_name)
            else:
                logging.error(f"✗ Failed to build index for model: {model_name}")
                failure_count += 1
        except Exception as e:
            logging.error(
                f"✗ Error building index for {model_name}: {e}", exc_info=debug
            )
            failure_count += 1

    elapsed_time = time.time() - start_time
    logging.info("--- Index Building Summary ---")
    logging.info(f"Completed in {elapsed_time:.2f} seconds.")
    logging.info(f"Successful: {success_count}, Failed: {failure_count}")
    if processed_model_names:
        logging.info(f"Successfully processed: {', '.join(processed_model_names)}")

    return failure_count == 0
