#!/usr/bin/env python3
"""
Utility functions for loading, processing, and analyzing benchmark summary files.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def deduplicate_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate model entries, keeping only the most recent result for each model.
    Assumes summaries are sorted by timestamp (newest first).
    """
    if not summaries:
        return []
    unique_summaries_dict = {}
    for summary in summaries:
        composite_key = (summary.get("model", "Unknown"),)

        if composite_key not in unique_summaries_dict:
            unique_summaries_dict[composite_key] = summary

    deduplicated = list(unique_summaries_dict.values())
    logger.info(
        f"Deduplicated summaries: {len(summaries)} -> {len(deduplicated)} unique runs."
    )
    return deduplicated


def prepare_comparison_dataframe(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Prepare a Pandas DataFrame for comparing benchmark summaries.
    Handles dense retrieval metrics.
    """
    comparison_data = []
    k_values = [1, 3, 5, 10]

    for summary in summaries:
        row = {
            "Model": summary.get("model", "Unknown"),
            "Test File": summary.get("test_file", "Unknown"),
            "Test Cases": summary.get("num_test_cases", 0),
            "Date": summary.get("timestamp", ""),
        }

        # MRR
        row["MRR (Dense)"] = summary.get(
            "avg_mrr_dense", summary.get("avg_mrr", summary.get("mrr", 0.0))
        )
        # Hit Rate and Maximum Ontology Similarity
        for metric_prefix, display_prefix in [
            ("hit_rate", "HR"),
            ("max_ont_similarity", "MaxOntSim"),
        ]:
            for k in k_values:
                dense_key_avg = f"avg_{metric_prefix}_dense@{k}"
                dense_key_legacy = f"avg_{metric_prefix}@{k}"  # for older files
                dense_key_direct = f"{metric_prefix}@{k}"  # for even older files

                # Get dense value, checking multiple possible keys
                dense_val = summary.get(
                    dense_key_avg,
                    summary.get(dense_key_legacy, summary.get(dense_key_direct, 0.0)),
                )
                row[f"{display_prefix}@{k} (Dense)"] = dense_val
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    if "MRR (Dense)" in df.columns:
        df = df.sort_values(by="MRR (Dense)", ascending=False)
    return df


def prepare_flat_dataframe_for_plotting(
    summaries: list[dict[str, Any]],
    metric_prefix: str,  # e.g., "hit_rate", "max_ont_similarity"
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """
    Prepares a long-form DataFrame for plotting metrics like Hit@K or MaxOntSim@K.
    This structure is suitable for seaborn plots using `hue` for 'Method'.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    plot_data = []
    for summary in summaries:
        model_name = summary.get("model", "Unknown")

        for k in k_values:
            # Dense metrics
            dense_key_avg = f"avg_{metric_prefix}_dense@{k}"
            dense_key_legacy = f"avg_{metric_prefix}@{k}"
            dense_key_direct = f"{metric_prefix}@{k}"
            dense_value = summary.get(
                dense_key_avg,
                summary.get(dense_key_legacy, summary.get(dense_key_direct, np.nan)),
            )

            dense_std_dev = np.nan
            dense_per_case_key = f"{metric_prefix}_dense@{k}_per_case"
            legacy_per_case_key = f"{metric_prefix}@{k}_per_case"  # for older files

            per_case_data_dense = summary.get(
                dense_per_case_key, summary.get(legacy_per_case_key)
            )
            if (
                per_case_data_dense
                and isinstance(per_case_data_dense, list)
                and len(per_case_data_dense) > 1
            ):
                dense_std_dev = float(np.std(per_case_data_dense))

            if not np.isnan(dense_value):
                plot_data.append(
                    {
                        "model": model_name,
                        "k": k,
                        "method": "Dense",
                        "value": dense_value,
                        "std_dev": dense_std_dev,
                    }
                )

    df = pd.DataFrame(plot_data)
    if not df.empty:
        df["k"] = df["k"].astype(int)  # Ensure k is integer for plotting
    return df
