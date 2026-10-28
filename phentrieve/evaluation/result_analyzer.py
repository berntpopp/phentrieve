#!/usr/bin/env python3
"""
Utility functions for loading, processing, and analyzing benchmark summary files.
"""

import json
import logging
from pathlib import Path
from typing import Any

from phentrieve.benchmark.result_store import discover_artifacts

logger = logging.getLogger(__name__)


def load_summary_files(directory: str) -> list[dict[str, Any]]:
    """
    Load all retrieval summary JSON files from the specified directory.

    Args:
        directory: Directory containing summary JSON files

    Returns:
        List of dictionaries containing benchmark summaries
    """
    summaries = []
    json_files = discover_artifacts(
        Path(directory), "summary", benchmark_type="retrieval"
    )
    logger.debug(f"Found {len(json_files)} summary files in {directory}")

    for file_path in json_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                summary = json.load(f)
            # str(), not the Path itself: these summaries cross JSON boundaries.
            summary["file_path"] = str(file_path)
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Error loading summary file {file_path}: {e}")

    if summaries:
        summaries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    logger.info(f"Loaded {len(summaries)} summary files successfully.")
    return summaries
