#!/usr/bin/env python
"""
Local API Server Launcher for Phentrieve

This script sets up the necessary environment variables and data directories
before launching the FastAPI server using uvicorn.

Usage:
    python run_api_local.py
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist."""
    path = Path(directory_path)
    if not path.exists():
        logger.info(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return path


def setup_environment():
    """Load environment variables and ensure directories exist."""
    # Load environment from config file
    config_path = Path(__file__).parent / "local_api_config.env"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create the configuration file based on the template.")
        sys.exit(1)

    load_dotenv(config_path)

    # Get environment variables with defaults
    data_root = os.environ.get(
        "PHENTRIEVE_DATA_ROOT_DIR", str(Path.home() / ".phentrieve")
    )
    data_dir = os.environ.get(
        "PHENTRIEVE_DATA_DIR",
        data_root,  # No longer appending hpo_core_data subdirectory.
    )
    index_dir = os.environ.get(
        "PHENTRIEVE_INDEX_DIR", os.path.join(data_root, "indexes")
    )
    results_dir = os.environ.get(
        "PHENTRIEVE_RESULTS_DIR", os.path.join(data_root, "results")
    )

    # Ensure directories exist
    ensure_directory_exists(data_root)
    ensure_directory_exists(data_dir)
    ensure_directory_exists(index_dir)
    ensure_directory_exists(results_dir)

    # Set PYTHONPATH to include the project root
    project_root = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(project_root))
    os.environ["PYTHONPATH"] = str(project_root)

    logger.info(f"Data Root: {data_root}")
    logger.info(f"Data Dir: {data_dir}")
    logger.info(f"Index Dir: {index_dir}")
    logger.info(f"Results Dir: {results_dir}")

    # Check if required data files exist
    ancestors_file = Path(data_dir) / "hpo_ancestors.pkl"
    if not ancestors_file.exists():
        logger.warning(f"HPO ancestors file not found: {ancestors_file}")
        logger.warning("You may need to run the setup script to initialize HPO data.")
        logger.warning("Try: python -m phentrieve.setup_hpo_index")


def main():
    """Set up environment and start the API server."""
    logger.info("Starting Phentrieve API server setup...")
    setup_environment()

    # API server configuration
    host = os.environ.get("API_HOST", "0.0.0.0")  # Listen on all interfaces
    port = int(os.environ.get("API_PORT", "8000"))  # Default to 8000

    logger.info(f"Starting uvicorn on {host}:{port}")
    logger.info("Press CTRL+C to stop the server")

    # Start uvicorn server
    uvicorn.run(
        "main:app",  # Modified to use main.py in the current directory
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
