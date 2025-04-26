#!/usr/bin/env python3
"""
HPO Ontology Data Preparation Script

This script downloads the Human Phenotype Ontology (HPO) data,
extracts individual HPO terms, and precomputes graph properties
needed for similarity calculations.

It performs the following steps:
1. Download HPO JSON data if not present
2. Extract individual HPO terms as separate JSON files
3. Build HPO graph structure
4. Precompute ancestor sets and term depths for each HPO term
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from multilingual_hpo_rag.data_processing.hpo_parser import (
    download_hpo_json,
    extract_hpo_terms,
    prepare_hpo_ontology_data,
)
from multilingual_hpo_rag.config import (
    HPO_FILE_PATH,
    HPO_TERMS_DIR,
    HPO_ANCESTORS_FILE,
    HPO_DEPTHS_FILE,
)


def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    """Main function for HPO data preparation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download and prepare HPO ontology data"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    logging.info("Starting HPO ontology data preparation")

    # Run the full preparation process
    success = prepare_hpo_ontology_data()

    if success:
        logging.info("HPO data preparation completed successfully!")
        logging.info(f"HPO JSON file: {HPO_FILE_PATH}")
        logging.info(f"HPO terms directory: {HPO_TERMS_DIR}")
        logging.info(f"HPO ancestors file: {HPO_ANCESTORS_FILE}")
        logging.info(f"HPO depths file: {HPO_DEPTHS_FILE}")
    else:
        logging.error("HPO data preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
