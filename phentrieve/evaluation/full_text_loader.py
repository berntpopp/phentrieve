"""
Ground truth loader for Phentrieve.

This module provides functionality for loading annotated ground truth data
from files to support evaluation of HPO term extraction.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


def load_full_text_annotations(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load and parse full-text HPO annotations from a JSONL file.

    The JSONL file should contain one JSON object per line, where each object
    represents an annotated document with HPO terms. Each line should be a valid
    JSON object containing at minimum:
    - text: The full text content
    - hpo_terms: List of HPO term annotations
    - metadata: Optional additional document metadata

    Args:
        filepath: Path to the JSONL file containing annotated documents

    Returns:
        List of dictionaries, where each dictionary represents an annotated document
        with its text content, HPO term annotations, and optional metadata

    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not filepath.exists():
        logger.error(f"Ground truth file not found: {filepath}")
        raise FileNotFoundError(f"Ground truth file does not exist: {filepath}")

    loaded_documents = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    document = json.loads(line)
                    loaded_documents.append(document)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping malformed JSON line {line_number + 1} "
                        f"in {filepath}: {e}"
                    )

        logger.info(
            f"Successfully loaded {len(loaded_documents)} documents from {filepath}"
        )
    except IOError as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise

    return loaded_documents
