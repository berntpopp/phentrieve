#!/usr/bin/env python3
"""
Utility functions for loading, processing, and analyzing benchmark summary files.
"""
import glob
import json
import logging
import os
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_summary_files(directory: str) -> List[Dict[str, Any]]:
    """
    Load all summary JSON files from the specified directory.

    Args:
        directory: Directory containing summary JSON files

    Returns:
        List of dictionaries containing benchmark summaries
    """
    summaries = []
    json_files = glob.glob(os.path.join(directory, "*.json"))
    logger.debug(f"Found {len(json_files)} summary files in {directory}")

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary["file_path"] = file_path  # Add file path for reference
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Error loading summary file {file_path}: {e}")

    if summaries:
        summaries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    logger.info(f"Loaded {len(summaries)} summary files successfully.")
    return summaries


def deduplicate_summaries(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate model entries, keeping only the most recent result for each model.
    Assumes summaries are sorted by timestamp (newest first).
    """
    if not summaries:
        return []
    unique_summaries_dict = {}
    for summary in summaries:
        # Use a composite key if re-ranking info is present to differentiate runs
        key_parts = [summary.get("model", "Unknown")]
        if summary.get("reranker_enabled", False):
            key_parts.append(summary.get("reranker_model", "Unknown"))
            key_parts.append(summary.get("reranker_mode", "Unknown"))

        composite_key = tuple(key_parts)

        if composite_key not in unique_summaries_dict:
            unique_summaries_dict[composite_key] = summary

    deduplicated = list(unique_summaries_dict.values())
    logger.info(
        f"Deduplicated summaries: {len(summaries)} -> {len(deduplicated)} unique runs."
    )
    return deduplicated


def prepare_comparison_dataframe(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Prepare a Pandas DataFrame for comparing benchmark summaries.
    Handles both dense and re-ranked metrics.
    """
    comparison_data = []
    k_values = [1, 3, 5, 10]

    for summary in summaries:
        row = {
            "Model": summary.get("model", "Unknown"),
            "Test File": summary.get("test_file", "Unknown"),
            "Test Cases": summary.get("num_test_cases", 0),
            "Date": summary.get("timestamp", ""),
            "Reranker": "Enabled" if summary.get("reranker_enabled") else "Disabled",
            "Reranker Model": (
                summary.get("reranker_model", "N/A")
                if summary.get("reranker_enabled")
                else "N/A"
            ),
            "Reranker Mode": (
                summary.get("reranker_mode", "N/A")
                if summary.get("reranker_enabled")
                else "N/A"
            ),
        }

        # MRR
        row["MRR (Dense)"] = summary.get(
            "avg_mrr_dense", summary.get("avg_mrr", summary.get("mrr", 0.0))
        )
        if summary.get("reranker_enabled"):
            row["MRR (ReRanked)"] = summary.get("avg_mrr_reranked", 0.0)
            row["MRR (Diff)"] = row["MRR (ReRanked)"] - row["MRR (Dense)"]

        # Hit Rate and Maximum Ontology Similarity
        for metric_prefix, display_prefix in [
            ("hit_rate", "HR"),
            ("max_ont_similarity", "MaxOntSim"),
        ]:
            for k in k_values:
                dense_key_avg = f"avg_{metric_prefix}_dense@{k}"
                dense_key_legacy = f"avg_{metric_prefix}@{k}"  # for older files
                dense_key_direct = f"{metric_prefix}@{k}"  # for even older files

                reranked_key_avg = f"avg_{metric_prefix}_reranked@{k}"

                # Get dense value, checking multiple possible keys
                dense_val = summary.get(
                    dense_key_avg,
                    summary.get(dense_key_legacy, summary.get(dense_key_direct, 0.0)),
                )
                row[f"{display_prefix}@{k} (Dense)"] = dense_val

                if summary.get("reranker_enabled"):
                    reranked_val = summary.get(reranked_key_avg, 0.0)
                    row[f"{display_prefix}@{k} (ReRanked)"] = reranked_val
                    row[f"{display_prefix}@{k} (Diff)"] = reranked_val - dense_val
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    if "MRR (Dense)" in df.columns:
        df = df.sort_values(by="MRR (Dense)", ascending=False)
    return df


def prepare_flat_dataframe_for_plotting(
    summaries: List[Dict[str, Any]],
    metric_prefix: str,  # e.g., "hit_rate", "max_ont_similarity"
    k_values: List[int] = [1, 3, 5, 10],
) -> pd.DataFrame:
    """
    Prepares a long-form DataFrame for plotting metrics like Hit@K or MaxOntSim@K.
    This structure is suitable for seaborn plots using `hue` for 'Method'.
    """
    plot_data = []
    for summary in summaries:
        model_name = summary.get("model", "Unknown")
        reranker_enabled = summary.get("reranker_enabled", False)

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
                dense_std_dev = np.std(per_case_data_dense)

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

            # Re-ranked metrics (if enabled and present)
            if reranker_enabled:
                reranked_key_avg = f"avg_{metric_prefix}_reranked@{k}"
                reranked_value = summary.get(reranked_key_avg, np.nan)

                reranked_std_dev = np.nan
                reranked_per_case_key = f"{metric_prefix}_reranked@{k}_per_case"
                per_case_data_reranked = summary.get(reranked_per_case_key)
                if (
                    per_case_data_reranked
                    and isinstance(per_case_data_reranked, list)
                    and len(per_case_data_reranked) > 1
                ):
                    reranked_std_dev = np.std(per_case_data_reranked)

                if not np.isnan(reranked_value):
                    plot_data.append(
                        {
                            "model": model_name,
                            "k": k,
                            "method": "Re-Ranked",
                            "value": reranked_value,
                            "std_dev": reranked_std_dev,
                        }
                    )

    df = pd.DataFrame(plot_data)
    if not df.empty:
        df["k"] = df["k"].astype(int)  # Ensure k is integer for plotting
    return df
