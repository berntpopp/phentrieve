"""
Index building orchestration module.

This module provides orchestration functions for building ChromaDB indexes
from HPO term data using embedding models.

Supports both single-vector and multi-vector index types (see issue #136).
"""

import logging
import os
import time

from phentrieve.config import BENCHMARK_MODELS, DEFAULT_HPO_DB_FILENAME, DEFAULT_MODEL
from phentrieve.data_processing.document_creator import (
    create_hpo_documents,
    load_hpo_terms,
)
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.data_processing.multi_vector_document_creator import (
    create_multi_vector_documents,
    get_component_stats,
)
from phentrieve.embeddings import load_embedding_model
from phentrieve.indexing.chromadb_indexer import build_chromadb_index
from phentrieve.utils import (
    get_default_data_dir,
    get_default_index_dir,
    resolve_data_path,
)


def orchestrate_index_building(
    model_name_arg: str | None = None,
    all_models: bool = False,
    batch_size: int = 100,
    trust_remote_code: bool = False,
    device_override: str | None = None,
    recreate: bool = False,
    debug: bool = False,
    index_dir_override: str | None = None,
    data_dir_override: str | None = None,
    multi_vector: bool = False,
    model_revision: str | None = None,
    code_revision: str | None = None,
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
        index_dir_override: Override for index directory path
        data_dir_override: Override for data directory path
        multi_vector: Build multi-vector index (separate vectors per component)
        model_revision: Optional immutable Hugging Face revision for one model build.
        code_revision: Optional immutable revision for model custom code.

    Returns:
        True if all requested indexes were built successfully, False otherwise
    """
    start_time = time.time()
    index_type = "multi_vector" if multi_vector else "single_vector"

    data_dir = resolve_data_path(data_dir_override, "data_dir", get_default_data_dir)
    db_path = data_dir / DEFAULT_HPO_DB_FILENAME
    if not db_path.exists():
        logging.error("HPO database not found: %s", db_path)
        return False
    with HPODatabase(db_path) as db:
        hpo_version = db.get_metadata("hpo_version")
        hpo_source_sha256 = db.get_metadata("hpo_source_sha256")
    if not hpo_version or hpo_version == "latest":
        logging.error("HPO database is missing a resolved release version")
        return False
    if not hpo_source_sha256:
        logging.error(
            "HPO database is missing its source digest. Re-run 'phentrieve data prepare'."
        )
        return False

    logging.info("Loading HPO terms for indexing...")
    hpo_terms = load_hpo_terms(data_dir_override=data_dir_override)
    if not hpo_terms:
        logging.error("Failed to load HPO terms. Run 'phentrieve data prepare' first.")
        return False

    # Create documents based on index type
    if multi_vector:
        logging.info("Creating multi-vector HPO documents for indexing...")
        # Show component statistics
        stats = get_component_stats(hpo_terms)
        logging.info(
            f"Component stats: {stats['total_labels']} labels, "
            f"{stats['total_synonyms']} synonyms, "
            f"{stats['total_definitions']} definitions"
        )
        logging.info(f"Estimated total documents: {stats['estimated_documents']}")
        documents, metadatas, ids = create_multi_vector_documents(hpo_terms)
    else:
        logging.info("Creating single-vector HPO documents for indexing...")
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

    models_to_process: list[str] = []
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
                revision=model_revision,
                code_revision=code_revision,
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
                index_type=index_type,
                hpo_version=hpo_version,
                hpo_source_sha256=hpo_source_sha256,
                model_revision=model_revision,
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
