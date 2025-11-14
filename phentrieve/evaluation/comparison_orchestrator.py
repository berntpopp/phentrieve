"""
Benchmark comparison orchestration module for the CLI.

This module provides functionality for comparing benchmark results
from multiple model runs and generating comparison tables and charts.
"""

import glob
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from phentrieve.config import (
    DEFAULT_SUMMARIES_SUBDIR,
    DEFAULT_VISUALIZATIONS_SUBDIR,
)
from phentrieve.utils import get_default_results_dir, resolve_data_path

# Set up logging
logger = logging.getLogger(__name__)


def load_benchmark_summaries(summaries_dir: str) -> list[dict[str, Any]]:
    """
    Load benchmark summary files from the specified directory.

    Args:
        summaries_dir: Directory containing benchmark summary files

    Returns:
        List of loaded benchmark summaries
    """

    # Find all JSON files in the summaries directory
    summary_files = glob.glob(os.path.join(summaries_dir, "*.json"))

    if not summary_files:
        logger.warning(f"No summary files found in {summaries_dir}")
        return []

    # Load each summary file
    summaries = []
    for file_path in summary_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                summary = json.load(f)
                summaries.append(summary)
                logger.debug(f"Loaded summary from {file_path}")
        except Exception as e:
            logger.error(f"Error loading summary file {file_path}: {e}")

    logger.info(f"Loaded {len(summaries)} benchmark summaries")
    return summaries


def compare_benchmark_summaries(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Compare benchmark summaries and create a DataFrame with the comparison.

    Args:
        summaries: List of benchmark summary dictionaries

    Returns:
        DataFrame with benchmark comparison
    """
    if not summaries:
        logger.warning("No summaries provided for comparison")
        return pd.DataFrame()

    # Extract comparison data
    comparison_data = []

    for summary in summaries:
        # Basic model information
        model_data = {
            "Model": summary.get("model", "Unknown"),
            "Original Model Name": summary.get("original_model_name", "Unknown"),
            "Timestamp": summary.get("timestamp", "Unknown"),
            "Test Cases": summary.get("num_test_cases", 0),
        }

        # Add re-ranking configuration if present
        if summary.get("reranker_enabled", False):
            model_data["Reranker"] = summary.get("reranker_model", "Unknown")
            model_data["Rerank Mode"] = summary.get("reranker_mode", "Unknown")

        # Add dense retrieval metrics
        model_data["MRR (Dense)"] = summary.get("mrr_dense", 0)

        # Add Hit Rate metrics for dense retrieval
        for k in [1, 3, 5, 10]:
            key = f"hit_rate_dense@{k}"
            if key in summary:
                model_data[f"HR@{k} (Dense)"] = summary[key]

        # Add maximum ontology similarity metrics for dense retrieval
        for k in [1, 3, 5, 10]:
            key = f"max_ont_similarity_dense@{k}"
            if key in summary:
                model_data[f"MaxOntSim@{k} (Dense)"] = summary[key]

        # Add re-ranked metrics if available
        if summary.get("reranker_enabled", False):
            model_data["MRR (Reranked)"] = summary.get("mrr_reranked", 0)
            model_data["MRR (Improvement)"] = (
                model_data["MRR (Reranked)"] - model_data["MRR (Dense)"]
            )

            # Add Hit Rate metrics for re-ranked results
            for k in [1, 3, 5, 10]:
                key = f"hit_rate_reranked@{k}"
                if key in summary:
                    model_data[f"HR@{k} (Reranked)"] = summary[key]

                    # Calculate improvement
                    dense_key = f"hit_rate_dense@{k}"
                    if dense_key in summary:
                        model_data[f"HR@{k} (Improvement)"] = (
                            summary[key] - summary[dense_key]
                        )

            # Add maximum ontology similarity metrics for re-ranked results
            for k in [1, 3, 5, 10]:
                key = f"max_ont_similarity_reranked@{k}"
                if key in summary:
                    model_data[f"MaxOntSim@{k} (Reranked)"] = summary[key]

                    # Calculate improvement
                    dense_key = f"max_ont_similarity_dense@{k}"
                    if dense_key in summary:
                        model_data[f"MaxOntSim@{k} (Improvement)"] = (
                            summary[key] - summary[dense_key]
                        )

        comparison_data.append(model_data)

    # Create DataFrame from the collected data
    df = pd.DataFrame(comparison_data)

    # Sort by MRR (Dense) by default
    if "MRR (Dense)" in df.columns:
        df = df.sort_values("MRR (Dense)", ascending=False)

    return df


def orchestrate_benchmark_comparison(
    summaries_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    visualize: bool = False,
    output_dir: Optional[str] = None,
    metrics: str = "all",
    debug: bool = False,
    results_dir_override: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Orchestrate the benchmark comparison process.

    Args:
        summaries_dir: Directory containing benchmark summary files
        output_csv: Path to save comparison CSV
        visualize: Whether to generate visualizations
        output_dir: Directory to save visualizations
        metrics: Metrics to include in visualizations, comma-separated or 'all'
        debug: Enable debug logging

    Returns:
        DataFrame with benchmark comparison or None if no summaries found
    """
    # Set up logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Resolve paths
    base_results_dir = resolve_data_path(
        results_dir_override, "results_dir", get_default_results_dir
    )

    # Resolve summaries_dir (if provided explicitly, use it; otherwise construct from base_results_dir)
    summaries_load_path = (
        Path(summaries_dir).expanduser().resolve()
        if summaries_dir
        else base_results_dir / DEFAULT_SUMMARIES_SUBDIR
    )

    # Resolve output_dir for visualizations (if provided, use it; otherwise construct from base_results_dir)
    output_dir_viz = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else base_results_dir / DEFAULT_VISUALIZATIONS_SUBDIR
    )

    # Determine the output path for comparison CSV
    if output_csv:
        csv_save_path = Path(output_csv).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save comparison directly in the results directory, not in detailed subdirectory
        os.makedirs(base_results_dir, exist_ok=True)
        csv_save_path = base_results_dir / f"benchmark_comparison_{timestamp}.csv"

    # Load benchmark summaries
    summaries = load_benchmark_summaries(str(summaries_load_path))

    if not summaries:
        logger.warning(f"No benchmark summaries found in {summaries_load_path}")
        return None

    # Generate comparison DataFrame
    comparison_df = compare_benchmark_summaries(summaries)

    # Save comparison to CSV
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    comparison_df.to_csv(str(csv_save_path))
    logger.info(f"Comparison table saved to {csv_save_path}")

    # Generate visualizations if requested
    if visualize:
        generate_visualizations(comparison_df, metrics, str(output_dir_viz), debug)

    return comparison_df


def generate_visualizations(
    comparison_df: pd.DataFrame,
    metrics: str = "all",
    output_dir: str | None = None,
    debug: bool = False,
) -> bool:
    """
    Generate visualizations from benchmark comparison data.

    Args:
        comparison_df: DataFrame with benchmark comparison data
        metrics: Metrics to include in visualizations, comma-separated or 'all'
        output_dir: Directory to save visualizations
        debug: Enable debug logging

    Returns:
        True if visualizations were generated successfully, False otherwise
    """
    if comparison_df.empty:
        logger.warning("Cannot generate visualizations from empty comparison data")
        return False

    # Determine which metrics to visualize
    if metrics.lower() == "all":
        metric_columns = [
            col
            for col in comparison_df.columns
            if any(col.startswith(prefix) for prefix in ["MRR", "HR@", "MaxOntSim@"])
        ]
    else:
        metric_columns = [m.strip() for m in metrics.split(",")]
        # Filter to only include columns that exist in the DataFrame
        metric_columns = [col for col in metric_columns if col in comparison_df.columns]

    if not metric_columns:
        logger.warning("No valid metrics found for visualization")
        return False

    # Validate output directory
    if output_dir is None:
        logger.error("Output directory must be specified for visualizations")
        return False

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up plot style
    sns.set(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "font.size": 12,
        }
    )

    # GENERATE COMBINED MULTI-PANEL VISUALIZATIONS
    try:
        # Organize metrics by type
        mrr_metrics = [col for col in comparison_df.columns if col.startswith("MRR")]
        hr_metrics = [col for col in comparison_df.columns if col.startswith("HR@")]
        ont_metrics = [
            col for col in comparison_df.columns if col.startswith("MaxOntSim@")
        ]

        # Check if we have both dense and reranked metrics
        has_dense = any("Dense" in m for m in comparison_df.columns)
        has_reranked = any("Re-Ranked" in m for m in comparison_df.columns)
        has_comparison = has_dense and has_reranked

        # Organize HR and OntSim metrics by k value
        hr_by_k: dict[str, list[str]] = {}
        for metric in hr_metrics:
            k_match = re.search(r"HR@(\d+)", metric)
            if k_match:
                k = k_match.group(1)
                if k not in hr_by_k:
                    hr_by_k[k] = []
                hr_by_k[k].append(metric)

        ont_by_k: dict[str, list[str]] = {}
        for metric in ont_metrics:
            k_match = re.search(r"MaxOntSim@(\d+)", metric)
            if k_match:
                k = k_match.group(1)
                if k not in ont_by_k:
                    ont_by_k[k] = []
                ont_by_k[k].append(metric)

        # Sort models by MRR for consistent display across all plots
        primary_sort = (
            "MRR (Dense)"
            if "MRR (Dense)" in comparison_df.columns
            else comparison_df.columns[1]
        )
        sorted_df = comparison_df.sort_values(
            by=primary_sort, ascending=False
        ).reset_index(drop=True)

        # Create a colormap for models that will be consistent across all charts
        # Use a colorblind-friendly palette with distinct colors
        model_names = sorted_df["Model"].tolist()
        n_models = len(model_names)
        model_colors = plt.cm.get_cmap(
            "tab10", n_models
        )  # tab10 is a good discrete colormap
        color_map = {model: model_colors(i) for i, model in enumerate(model_names)}

        # 1. GENERATE MRR COMPARISON (special visualization)
        if mrr_metrics:
            plt.figure(figsize=(14, 8))
            x = np.arange(len(sorted_df))
            ax = plt.axes()

            # If we have both dense and reranked MRR
            if (
                has_comparison
                and "MRR (Dense)" in mrr_metrics
                and "MRR (Re-Ranked)" in mrr_metrics
            ):
                bar_width = 0.35
                # Use the model colors for the bars
                dense_bars = ax.bar(
                    x - bar_width / 2,
                    sorted_df["MRR (Dense)"],
                    width=bar_width,
                    label="Dense",
                    color=[color_map[m] for m in sorted_df["Model"]],
                    alpha=0.9,
                )
                reranked_bars = ax.bar(
                    x + bar_width / 2,
                    sorted_df["MRR (Re-Ranked)"],
                    width=bar_width,
                    label="Re-Ranked",
                    color=[color_map[m] for m in sorted_df["Model"]],
                    alpha=0.6,
                )

                # Add value labels on bars
                for i, bar in enumerate(dense_bars):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

                for i, bar in enumerate(reranked_bars):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

                # Create legend for dense vs reranked
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc="upper right")

                # Add a second legend for model colors
                model_patches = [
                    plt.Rectangle((0, 0), 1, 1, color=color_map[model])
                    for model in model_names
                ]
                plt.legend(
                    model_patches,
                    model_names,
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1),
                )

                title = "MRR Comparison: Dense vs Re-Ranked Retrieval"
            else:
                # Single MRR type, create simple bar chart with model-colored bars
                metric = mrr_metrics[0]
                bars = ax.bar(
                    x,
                    sorted_df[metric],
                    color=[color_map[m] for m in sorted_df["Model"]],
                )

                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

                # Create legend for model colors
                model_patches = [
                    plt.Rectangle((0, 0), 1, 1, color=color_map[model])
                    for model in model_names
                ]
                plt.legend(model_patches, model_names, loc="upper right")

                title = "MRR Comparison Across Models"

            # Common formatting
            plt.xlabel("Model", fontsize=14)
            plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=14)
            plt.title(title, fontsize=16)
            plt.xticks(x, sorted_df["Model"], rotation=45, ha="right")
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "mrr_comparison.png"), dpi=300)
            plt.close()
            logger.info("Generated MRR comparison visualization")

        # 2. CREATE COMBINED HR@K MULTI-PANEL CHART
        if hr_by_k:
            # Determine how many k-values we have for HR metrics
            hr_k_values = sorted([int(k) for k in hr_by_k.keys()])
            n_hr_plots = len(hr_k_values)

            if n_hr_plots > 0:
                # Create a figure with multiple subplots for each k value
                fig, axes = plt.subplots(
                    n_hr_plots, 1, figsize=(12, 5 * n_hr_plots), squeeze=False
                )

                for i, k in enumerate(hr_k_values):
                    ax = axes[i, 0]
                    metrics: list[str] = hr_by_k[str(k)]

                    # Check if we have both dense and reranked for this k value
                    has_dense_k = any("Dense" in m for m in metrics)
                    has_reranked_k = any("Re-Ranked" in m for m in metrics)
                    has_comparison_k = has_dense_k and has_reranked_k

                    x = np.arange(len(sorted_df))

                    # Case 1: If we have both dense and reranked
                    if has_comparison_k:
                        dense_metric = f"HR@{k} (Dense)"
                        reranked_metric = f"HR@{k} (Re-Ranked)"

                        bar_width = 0.35
                        # Apply model-based colors with different alpha for Dense vs Reranked
                        dense_bars = ax.bar(
                            x - bar_width / 2,
                            sorted_df[dense_metric],
                            width=bar_width,
                            label="Dense",
                            color=[color_map[m] for m in sorted_df["Model"]],
                            alpha=0.9,
                        )
                        reranked_bars = ax.bar(
                            x + bar_width / 2,
                            sorted_df[reranked_metric],
                            width=bar_width,
                            label="Re-Ranked",
                            color=[color_map[m] for m in sorted_df["Model"]],
                            alpha=0.6,
                        )

                        # Add value labels (only for a reasonable number of models)
                        if len(sorted_df) <= 8:
                            for _j, bar in enumerate(dense_bars):
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.01,
                                    f"{height:.3f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=9,
                                )

                        # Add model color legend to the first plot
                        if i == 0 and len(model_names) <= 8:
                            model_patches = [
                                plt.Rectangle((0, 0), 1, 1, color=color_map[model])
                                for model in model_names
                            ]
                            ax.legend(
                                model_patches,
                                model_names,
                                loc="upper right",
                                title="Models",
                            )

                            for _j, bar in enumerate(reranked_bars):
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.01,
                                    f"{height:.3f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=9,
                                )

                        if i == 0:  # Only add legend to first plot
                            # Create legend for dense vs reranked types
                            handles, labels = ax.get_legend_handles_labels()
                            ax.legend(handles, labels, loc="upper right")

                            # Only add model color legend to the first plot
                            if (
                                len(model_names) <= 8
                            ):  # Only if we have a reasonable number of models
                                model_patches = [
                                    plt.Rectangle((0, 0), 1, 1, color=color_map[model])
                                    for model in model_names
                                ]
                                ax2 = ax.twinx()  # Create a second y-axis
                                ax2.set_yticks([])
                                ax2.legend(
                                    model_patches,
                                    model_names,
                                    loc="upper left",
                                    bbox_to_anchor=(1.05, 1),
                                    title="Models",
                                )

                    # Case 2: Single metric type
                    else:
                        metric = metrics[0]
                        # Use model-specific colors for better visualization
                        bars = ax.bar(
                            x,
                            sorted_df[metric],
                            color=[color_map[m] for m in sorted_df["Model"]],
                        )

                        # Add value labels (only for a reasonable number of models)
                        if len(sorted_df) <= 8:
                            for _j, bar in enumerate(bars):
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.01,
                                    f"{height:.3f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=9,
                                )

                        # Add model color legend to the first plot
                        if i == 0 and len(model_names) <= 8:
                            model_patches = [
                                plt.Rectangle((0, 0), 1, 1, color=color_map[model])
                                for model in model_names
                            ]
                            ax.legend(
                                model_patches,
                                model_names,
                                loc="upper right",
                                title="Models",
                            )

                    # Common formatting for this subplot
                    ax.set_ylim(0, 1.0)
                    ax.set_title(f"Hit Rate: Dense vs Re-Ranked - HR@{k}", fontsize=12)
                    ax.set_xticks(x)
                    ax.set_xticklabels(sorted_df["Model"], rotation=45, ha="right")
                    ax.set_ylabel(f"HR@{k}")

                    # Only show x-label on bottom plot
                    if i == n_hr_plots - 1:
                        ax.set_xlabel("Model")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, "combined_hr_comparison.png"), dpi=300
                )
                plt.close()
                logger.info("Generated combined HR comparison visualization")

        # 3. CREATE COMBINED ONTSIM@K MULTI-PANEL CHART
        if ont_by_k:
            # Determine how many k-values we have for OntSim metrics
            ont_k_values = sorted([int(k) for k in ont_by_k.keys()])
            n_ont_plots = len(ont_k_values)

            if n_ont_plots > 0:
                # Create a figure with multiple subplots for each k value
                fig, axes = plt.subplots(
                    n_ont_plots, 1, figsize=(12, 5 * n_ont_plots), squeeze=False
                )

                for i, k in enumerate(ont_k_values):
                    ax = axes[i, 0]
                    metrics: list[str] = ont_by_k[str(k)]

                    # Check if we have both dense and reranked for this k value
                    has_dense_k = any("Dense" in m for m in metrics)
                    has_reranked_k = any("Re-Ranked" in m for m in metrics)
                    has_comparison_k = has_dense_k and has_reranked_k

                    x = np.arange(len(sorted_df))

                    # Case 1: If we have both dense and reranked
                    if has_comparison_k:
                        dense_metric = f"MaxOntSim@{k} (Dense)"
                        reranked_metric = f"MaxOntSim@{k} (Re-Ranked)"

                        bar_width = 0.35
                        # Apply model-based colors with different alpha for Dense vs Reranked
                        dense_bars = ax.bar(
                            x - bar_width / 2,
                            sorted_df[dense_metric],
                            width=bar_width,
                            label="Dense",
                            color=[color_map[m] for m in sorted_df["Model"]],
                            alpha=0.9,
                        )
                        reranked_bars = ax.bar(
                            x + bar_width / 2,
                            sorted_df[reranked_metric],
                            width=bar_width,
                            label="Re-Ranked",
                            color=[color_map[m] for m in sorted_df["Model"]],
                            alpha=0.6,
                        )

                        # Add value labels (only for a reasonable number of models)
                        if len(sorted_df) <= 8:
                            for _j, bar in enumerate(dense_bars):
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.01,
                                    f"{height:.3f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=9,
                                )

                        # Add model color legend to the first plot
                        if i == 0 and len(model_names) <= 8:
                            model_patches = [
                                plt.Rectangle((0, 0), 1, 1, color=color_map[model])
                                for model in model_names
                            ]
                            ax.legend(
                                model_patches,
                                model_names,
                                loc="upper right",
                                title="Models",
                            )

                            for _j, bar in enumerate(reranked_bars):
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.01,
                                    f"{height:.3f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=9,
                                )

                        if i == 0:  # Only add legend to first plot
                            # Create legend for dense vs reranked types
                            handles, labels = ax.get_legend_handles_labels()
                            ax.legend(handles, labels, loc="upper right")

                            # Only add model color legend to the first plot
                            if (
                                len(model_names) <= 8
                            ):  # Only if we have a reasonable number of models
                                model_patches = [
                                    plt.Rectangle((0, 0), 1, 1, color=color_map[model])
                                    for model in model_names
                                ]
                                ax2 = ax.twinx()  # Create a second y-axis
                                ax2.set_yticks([])
                                ax2.legend(
                                    model_patches,
                                    model_names,
                                    loc="upper left",
                                    bbox_to_anchor=(1.05, 1),
                                    title="Models",
                                )

                    # Case 2: Single metric type
                    else:
                        metric = metrics[0]
                        # Use model-specific colors for better visualization
                        bars = ax.bar(
                            x,
                            sorted_df[metric],
                            color=[color_map[m] for m in sorted_df["Model"]],
                        )

                        # Add value labels (only for a reasonable number of models)
                        if len(sorted_df) <= 8:
                            for _j, bar in enumerate(bars):
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height + 0.01,
                                    f"{height:.3f}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=9,
                                )

                        # Add model color legend to the first plot
                        if i == 0 and len(model_names) <= 8:
                            model_patches = [
                                plt.Rectangle((0, 0), 1, 1, color=color_map[model])
                                for model in model_names
                            ]
                            ax.legend(
                                model_patches,
                                model_names,
                                loc="upper right",
                                title="Models",
                            )

                    # Common formatting for this subplot
                    ax.set_ylim(0, 1.0)
                    ax.set_title(
                        f"Ontology Similarity: Dense vs Re-Ranked - OntSim@{k}",
                        fontsize=12,
                    )
                    ax.set_xticks(x)
                    ax.set_xticklabels(sorted_df["Model"], rotation=45, ha="right")
                    ax.set_ylabel(f"OntSim@{k}")

                    # Only show x-label on bottom plot
                    if i == n_ont_plots - 1:
                        ax.set_xlabel("Model")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, "combined_ontsim_comparison.png"), dpi=300
                )
                plt.close()
                logger.info("Generated combined OntSim comparison visualization")

    except Exception as e:
        logger.error(f"Error generating comparison visualizations: {str(e)}")
        if debug:
            import traceback

            traceback.print_exc()

    # ADD HEATMAP - The one visualization you liked!
    try:
        # Select key metrics for heatmap
        key_metrics = [
            "HR@1 (Dense)",
            "HR@3 (Dense)",
            "HR@5 (Dense)",
            "HR@10 (Dense)",
            "MaxOntSim@1 (Dense)",
            "MaxOntSim@3 (Dense)",
            "MaxOntSim@5 (Dense)",
            "MaxOntSim@10 (Dense)",
        ]

        available_metrics = [m for m in key_metrics if m in comparison_df.columns]

        if available_metrics:
            # Sort models by MRR for consistent ordering
            if "MRR (Dense)" in comparison_df.columns:
                sorted_df = comparison_df.sort_values(
                    by="MRR (Dense)", ascending=False
                ).reset_index(drop=True)
            else:
                sorted_df = comparison_df

            plt.figure(figsize=(12, len(sorted_df) * 0.8 + 2))
            heatmap_data = sorted_df.set_index("Model")[available_metrics]

            # Create a divider between hit rate and ontology metrics
            divider_position = 4  # After first 4 metrics (HR@1, HR@3, HR@5, HR@10)

            # Draw the heatmap with a vertical line separator
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                linewidths=0.5,
                cbar_kws={"label": "Score"},
            )

            # Add a vertical line to separate metrics types
            plt.axvline(x=divider_position, color="black", linewidth=3)

            plt.title("Performance Metrics Heatmap", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "performance_heatmap.png"), dpi=300)
            plt.close()
            logger.info("Generated performance heatmap")

            # We no longer generate a separate ontology heatmap as requested

    except Exception as e:
        logger.error(f"Error generating heatmap: {str(e)}")
        if debug:
            import traceback

            traceback.print_exc()

    # Generate line plots for k-value trends (showing how metrics change with k)
    try:
        # Extract Hit Rate metrics for different k values
        hr_pattern = re.compile(r"HR@(\d+) \(Dense\)")
        hr_metrics = [col for col in comparison_df.columns if hr_pattern.match(col)]

        # Extract Ontology Similarity metrics for different k values
        ont_pattern = re.compile(r"MaxOntSim@(\d+) \(Dense\)")
        ont_metrics = [col for col in comparison_df.columns if ont_pattern.match(col)]

        # Check if we have enough metrics to make line plots
        if len(hr_metrics) >= 2 or len(ont_metrics) >= 2:
            # Create figure with subplots for each model
            n_models = len(comparison_df)
            fig, axes = plt.subplots(
                1, n_models, figsize=(n_models * 4, 5), sharey=True
            )

            # If there's only one model, axes won't be an array
            if n_models == 1:
                axes = [axes]  # type: ignore[assignment]

            # For each model, create a line plot showing how metrics vary with k
            for i, (_idx, row) in enumerate(comparison_df.iterrows()):
                model_name = row["Model"]
                ax = axes[i]

                # Prepare data for HR@k metrics
                if len(hr_metrics) >= 2:
                    hr_k_values = [
                        int(m.group(1)) for col in hr_metrics if (m := hr_pattern.match(col))
                    ]
                    hr_scores = [row[col] for col in hr_metrics]

                    # Sort by k value
                    hr_points = sorted(zip(hr_k_values, hr_scores))
                    hr_k_values = [p[0] for p in hr_points]
                    hr_scores = [p[1] for p in hr_points]

                    # Plot HR@k line with consistent color
                    ax.plot(
                        hr_k_values, hr_scores, "o-", color="tab:blue", label="Hit Rate"
                    )

                # Prepare data for OntSim@k metrics
                if len(ont_metrics) >= 2:
                    ont_k_values = [
                        int(m.group(1)) for col in ont_metrics if (m := ont_pattern.match(col))
                    ]
                    ont_scores = [row[col] for col in ont_metrics]

                    # Sort by k value
                    ont_points = sorted(zip(ont_k_values, ont_scores))
                    ont_k_values = [p[0] for p in ont_points]
                    ont_scores = [p[1] for p in ont_points]

                    # Plot OntSim@k line with consistent color
                    ax.plot(
                        ont_k_values,
                        ont_scores,
                        "s--",
                        color="tab:orange",
                        label="Ontology Similarity",
                    )

                # Configure subplot
                ax.set_title(model_name, fontsize=10)
                ax.set_xlabel("k value")
                if i == 0:  # Only add y-label to first subplot
                    ax.set_ylabel("Score")
                ax.grid(True, linestyle="--", alpha=0.7)

                # Set common y-axis limits
                ax.set_ylim(0, 1.05)

                # Only add legend to first plot to avoid redundancy
                if i == 0 and (len(hr_metrics) >= 2 and len(ont_metrics) >= 2):
                    ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "metric_by_k_trends.png"), dpi=300)
            plt.close()
            logger.info("Generated k-value trend line plots")

    except Exception as e:
        logger.error(f"Error generating k-value trend plots: {str(e)}")
        if debug:
            import traceback

            traceback.print_exc()

    return True
