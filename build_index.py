"""
Script to build the HPO term vector index using the configuration system.
This uses our dynamic path resolution to properly locate data and index directories.
"""

import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Import our utilities after configuring logging
from phentrieve.indexing.chromadb_orchestrator import orchestrate_index_building
from phentrieve.utils import (
    resolve_data_path,
    get_default_index_dir,
    get_default_data_dir,
    load_user_config,
)


def build_index_with_config(
    model_name=None, recreate=False, all_models=False, data_dir=None, index_dir=None
):
    """Build vector index using our configuration system for path resolution"""

    # Resolve data directory path based on CLI > Config > Default
    data_dir_path = resolve_data_path(data_dir, "data_dir", get_default_data_dir)
    logging.info(f"Using data directory: {data_dir_path}")

    # Resolve index directory path based on CLI > Config > Default
    index_dir_path = resolve_data_path(index_dir, "index_dir", get_default_index_dir)
    logging.info(f"Using index directory: {index_dir_path}")

    # Ensure both directories exist
    os.makedirs(data_dir_path, exist_ok=True)
    os.makedirs(index_dir_path, exist_ok=True)

    # Build the index using our orchestrator with explicit paths
    success = orchestrate_index_building(
        model_name_arg=model_name,
        all_models=all_models,
        recreate=recreate,
        index_dir_override=str(index_dir_path),
        data_dir_override=str(data_dir_path),
        debug=True,
    )

    if success:
        logging.info("Index building completed successfully!")
    else:
        logging.error("Index building failed for one or more models.")

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build HPO term vector index")
    parser.add_argument("--model-name", help="Model name to use for embeddings")
    parser.add_argument(
        "--all-models", action="store_true", help="Run for all benchmark models"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Recreate index even if it exists"
    )
    parser.add_argument("--data-dir", help="Custom directory for HPO data")
    parser.add_argument("--index-dir", help="Custom directory for vector indexes")

    args = parser.parse_args()

    # Use our custom function to build the index with proper path resolution
    build_index_with_config(
        model_name=args.model_name,
        recreate=args.recreate,
        all_models=args.all_models,
        data_dir=args.data_dir,
        index_dir=args.index_dir,
    )
