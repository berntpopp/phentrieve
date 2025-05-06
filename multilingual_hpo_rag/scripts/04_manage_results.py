#!/usr/bin/env python3
"""
Benchmark Results Management Script

This script provides functionality for:
1. Comparing results across different model runs
2. Visualizing benchmark results with plots
3. Generating reports from benchmark data
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from multilingual_hpo_rag.config import (
    SUMMARIES_DIR,
    VISUALIZATIONS_DIR,
    DETAILED_DIR,
)


def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def ensure_directories_exist() -> None:
    """Create necessary output directories if they don't exist."""
    for directory in [SUMMARIES_DIR, VISUALIZATIONS_DIR, DETAILED_DIR]:
        if not os.path.exists(directory):
            logging.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)


def load_summary_files(directory: str = SUMMARIES_DIR) -> List[Dict[str, Any]]:
    """
    Load all summary JSON files from the specified directory.

    Args:
        directory: Directory containing summary JSON files

    Returns:
        List of dictionaries containing benchmark summaries
    """
    summaries = []

    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory, "*.json"))
    logging.debug(f"Found {len(json_files)} summary files in {directory}")

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            # Add file path for reference
            summary["file_path"] = file_path
            summaries.append(summary)
        except Exception as e:
            logging.error(f"Error loading summary file {file_path}: {e}")

    # Sort by timestamp (newest first)
    if summaries:
        summaries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return summaries


def deduplicate_summaries(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate model entries, keeping only the most recent result for each model.

    Args:
        summaries: List of benchmark summary dictionaries

    Returns:
        List of deduplicated benchmark summaries (one per model)
    """
    if not summaries:
        return []

    # Since summaries are already sorted by timestamp (newest first),
    # we can use a dict to keep track of models we've seen
    unique_summaries = {}

    for summary in summaries:
        model = summary.get("model", "Unknown")
        if model not in unique_summaries:
            unique_summaries[model] = summary

    return list(unique_summaries.values())


def compare_summaries(
    summaries: List[Dict[str, Any]], output: Optional[str] = None
) -> None:
    """
    Compare benchmark summaries and generate a comparison table.

    Args:
        summaries: List of benchmark summary dictionaries
        output: Optional output CSV file path
    """
    if not summaries:
        logging.error("No summaries to compare")
        return

    # Create comparison data
    comparison_data = []

    for summary in summaries:
        # Basic row data
        row = {
            "Model": summary.get("model", "Unknown"),
            "Test File": summary.get("test_file", "Unknown"),
            "Test Cases": summary.get("num_test_cases", 0),
            "Date": summary.get("timestamp", ""),
        }

        # Add re-ranking configuration if available
        reranker_enabled = summary.get("reranker_enabled", False)
        if reranker_enabled:
            row["Re-Ranking"] = "Enabled"
            row["Re-Ranker Model"] = summary.get("reranker_model", "Unknown")
            row["Re-Rank Mode"] = summary.get("reranker_mode", "cross-lingual")
            row["Re-Rank Count"] = summary.get("rerank_count", 0)
        else:
            row["Re-Ranking"] = "Disabled"

        # Add comparison metrics (both dense and re-ranked if available)

        # MRR metrics
        row["MRR (Dense)"] = summary.get(
            "mrr_dense", summary.get("mrr", 0.0)
        )  # Backward compatibility

        if reranker_enabled:
            row["MRR (Re-Ranked)"] = summary.get("mrr_reranked", 0.0)
            row["MRR Diff"] = (
                row["MRR (Re-Ranked)"] - row["MRR (Dense)"]
            )  # Positive is good

        # Add Hit Rate metrics - both dense and re-ranked if available
        for k in [1, 3, 5, 10]:
            # Try both new and legacy format for backward compatibility
            dense_key = f"hit_rate_dense@{k}"
            legacy_key = f"hit_rate@{k}"

            if dense_key in summary:
                row[f"HR@{k} (Dense)"] = summary[dense_key]
            elif legacy_key in summary:
                row[f"HR@{k} (Dense)"] = summary[legacy_key]

            # Add re-ranked metrics if available
            if reranker_enabled:
                reranked_key = f"hit_rate_reranked@{k}"
                if reranked_key in summary:
                    row[f"HR@{k} (Re-Ranked)"] = summary[reranked_key]
                    row[f"HR@{k} Diff"] = (
                        row[f"HR@{k} (Re-Ranked)"] - row[f"HR@{k} (Dense)"]
                    )  # Positive is good

        # Add Ontology Similarity metrics - both dense and re-ranked if available
        for k in [1, 3, 5, 10]:
            # Try both new and legacy format for backward compatibility
            dense_key = f"ont_similarity_dense@{k}"
            legacy_key = f"ont_similarity@{k}"

            if dense_key in summary:
                row[f"OntSim@{k} (Dense)"] = summary[dense_key]
            elif legacy_key in summary:
                row[f"OntSim@{k} (Dense)"] = summary[legacy_key]

            # Add re-ranked metrics if available
            if reranker_enabled:
                reranked_key = f"ont_similarity_reranked@{k}"
                if reranked_key in summary:
                    row[f"OntSim@{k} (Re-Ranked)"] = summary[reranked_key]
                    row[f"OntSim@{k} Diff"] = (
                        row[f"OntSim@{k} (Re-Ranked)"] - row[f"OntSim@{k} (Dense)"]
                    )  # Positive is good

        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Sort by MRR (descending)
    if "MRR" in df.columns:
        df = df.sort_values("MRR", ascending=False)

    # Display the table
    pd.options.display.float_format = "{:.4f}".format
    print("\n===== Benchmark Comparison =====")
    print(df.to_string(index=False))

    # Save to CSV if output is specified
    if output:
        df.to_csv(output, index=False)
        print(f"\nComparison saved to: {output}")


def visualize_results(
    summaries: List[Dict[str, Any]],
    metric: str = "all",
    include_models: Optional[List[str]] = None,
    output_dir: str = VISUALIZATIONS_DIR,
) -> None:
    """
    Create visualizations from benchmark results.

    Args:
        summaries: List of benchmark summary dictionaries
        metric: Metric to visualize ("mrr", "hit_rate", "ont_similarity", or "all")
        include_models: Optional list of models to include (None for all)
        output_dir: Directory to save visualization images
    """
    if not summaries:
        logging.error("No summaries to visualize")
        return

    # Import visualization libraries
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError:
        logging.error(
            "Visualization requires matplotlib, seaborn, and numpy. "
            "Please install these packages."
        )
        return

    # Filter models if specified
    if include_models:
        summaries = [s for s in summaries if s.get("model", "") in include_models]

    if not summaries:
        logging.error("No matching summaries to visualize")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up the style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Set specific metrics based on parameter
    metrics_to_visualize = []
    if metric == "all" or metric == "mrr":
        metrics_to_visualize.append("mrr")
    if metric == "all" or metric == "hit_rate":
        metrics_to_visualize.append("hit_rate")
    if metric == "all" or metric == "ont_similarity":
        metrics_to_visualize.append("ont_similarity")

    # Create timestamp for output files
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process the data for visualization
    model_names = [s.get("model", "Unknown") for s in summaries]
    model_slugs = [s.get("model_slug", s.get("model", "Unknown")) for s in summaries]

    # Create visualization for each requested metric
    for metric_name in metrics_to_visualize:
        if metric_name == "mrr":
            # ---------------------------------------------------------------------
            # MRR Bar Chart with Error Bars if available
            # ---------------------------------------------------------------------
            plt.figure(figsize=(10, 6))
            mrr_values = [s.get("mrr", 0) for s in summaries]

            # Check if we have raw data for error bars
            has_raw_data = False
            error_data = []

            for i, summary in enumerate(summaries):
                # Look for raw MRR values (single value or list)
                mrr_raw = summary.get("mrr_per_case") or summary.get("raw_mrr") or None

                if isinstance(mrr_raw, list) and len(mrr_raw) > 1:
                    has_raw_data = True
                    error_data.append(
                        {
                            "model": model_names[i],
                            "mean": np.mean(mrr_raw),
                            "std": np.std(mrr_raw),
                        }
                    )

            # Create bar plot with or without error bars
            if has_raw_data:
                # We have error data for standard deviation
                error_df = pd.DataFrame(error_data)
                plt.bar(
                    error_df["model"],
                    error_df["mean"],
                    yerr=error_df["std"],
                    capsize=5,
                    color=sns.color_palette("viridis", len(error_df)),
                )
                # Ensure y-axis doesn't go below 0 when showing error bars
                ymin = max(
                    0,
                    min([m - s for m, s in zip(error_df["mean"], error_df["std"])])
                    - 0.05,
                )
                plt.ylim(
                    bottom=ymin,
                    top=min(
                        1.0,
                        max([m + s for m, s in zip(error_df["mean"], error_df["std"])])
                        + 0.05,
                    ),
                )
            else:
                # Create simple bar chart without error bars
                bar_colors = sns.color_palette("viridis", len(model_names))
                plt.bar(model_names, mrr_values, color=bar_colors)
                plt.ylim(
                    0, min(1.0, max(mrr_values) * 1.1)
                )  # Add some headroom, but cap at 1.0

            plt.title("Mean Reciprocal Rank (MRR) by Model")
            plt.xlabel("Model")
            plt.ylabel("MRR")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save figure
            output_file = os.path.join(output_dir, f"mrr_comparison_{timestamp}.png")
            plt.savefig(output_file)
            plt.close()
            print(f"MRR visualization saved to: {output_file}")

        if metric_name == "hit_rate":
            # ---------------------------------------------------------------------
            # 1. Hit Rate Bar Chart with Error Bars if available
            # ---------------------------------------------------------------------
            plt.figure(figsize=(12, 8))
            k_values = [1, 3, 5, 10]

            # Check if we have raw data for error bars
            has_raw_data = False
            hit_rate_error_data = []

            for i, summary in enumerate(summaries):
                for k in k_values:
                    # Look for raw Hit Rate values
                    key = f"hit_rate@{k}"
                    raw_key = f"hit_rate@{k}_per_case"
                    hr_raw = summary.get(raw_key) or summary.get(f"raw_{key}") or None

                    if isinstance(hr_raw, list) and len(hr_raw) > 1:
                        has_raw_data = True
                        hit_rate_error_data.append(
                            {
                                "model": model_names[i],
                                "metric": f"Hit@{k}",
                                "mean": np.mean(hr_raw),
                                "std": np.std(hr_raw),
                            }
                        )

            # Create a visualization based on whether we have error data
            if has_raw_data and len(hit_rate_error_data) > 0:
                # Create grouped bar chart with error bars
                hr_error_df = pd.DataFrame(hit_rate_error_data)

                # Group by model and metric for positioning
                unique_models = sorted(hr_error_df["model"].unique())
                metrics = [f"Hit@{k}" for k in k_values]
                metrics = [m for m in metrics if m in hr_error_df["metric"].unique()]

                # Calculate bar positions
                bar_width = 0.2 if len(unique_models) <= 3 else 0.15
                x = np.arange(len(unique_models))

                # Plot each metric as a group
                viridis_colors = sns.color_palette("viridis", len(metrics))
                color_map = {
                    metric: viridis_colors[i] for i, metric in enumerate(metrics)
                }

                for i, metric in enumerate(metrics):
                    metric_df = hr_error_df[hr_error_df["metric"] == metric]
                    metric_df = (
                        metric_df.set_index("model")
                        .reindex(unique_models)
                        .reset_index()
                    )

                    # Position the bars for this metric
                    offset = (i - len(metrics) / 2 + 0.5) * bar_width

                    plt.bar(
                        x + offset,
                        metric_df["mean"],
                        bar_width,
                        yerr=metric_df["std"],
                        capsize=3,
                        label=metric,
                        color=color_map[metric],
                    )

                plt.xticks(x, unique_models, rotation=45, ha="right")
                plt.legend(title="Metric", loc="upper left")

                # Set y-axis limits (0 to 1 with padding for error bars)
                min_y = max(
                    0,
                    min([r["mean"] - r["std"] for _, r in hr_error_df.iterrows()])
                    - 0.05,
                )
                plt.ylim(bottom=min_y, top=1.05)
            else:
                # Create standard grouped bar chart without error bars
                hit_rate_data = []
                for i, model in enumerate(model_names):
                    for k in k_values:
                        key = f"hit_rate@{k}"
                        value = summaries[i].get(key, 0)
                        hit_rate_data.append(
                            {"model": model, "metric": f"Hit@{k}", "value": value}
                        )

                hr_df = pd.DataFrame(hit_rate_data)
                sns.barplot(
                    x="model", y="value", hue="metric", data=hr_df, palette="viridis"
                )
                plt.ylim(0, 1.05)

            plt.title("Hit Rate at K by Model")
            plt.xlabel("Model")
            plt.ylabel("Hit Rate")
            plt.tight_layout()

            # Save figure
            output_file = os.path.join(output_dir, f"hit_rate_barplot_{timestamp}.png")
            plt.savefig(output_file)
            plt.close()
            print(f"Hit Rate bar plot saved to: {output_file}")

            # ---------------------------------------------------------------------
            # 2. Hit Rate Line Plot with connected dots (by K value)
            # ---------------------------------------------------------------------
            plt.figure(figsize=(12, 8))

            # Extract data for line plot (one line per model, x-axis = k values)
            for i, model in enumerate(model_names):
                hit_rates_by_k = []
                for k in k_values:
                    key = f"hit_rate@{k}"
                    hit_rates_by_k.append(summaries[i].get(key, 0))

                if any(hit_rates_by_k):
                    model_color = plt.cm.viridis(i / max(1, len(model_names) - 1))
                    plt.plot(
                        k_values,
                        hit_rates_by_k,
                        "o-",
                        linewidth=2.5,
                        markersize=8,
                        label=model,
                        color=model_color,
                    )

            plt.title("Hit Rate by K Value", fontsize=14)
            plt.xlabel("K", fontsize=12)
            plt.ylabel("Hit Rate", fontsize=12)
            plt.xticks(k_values, fontsize=11)
            plt.yticks(fontsize=11)
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(title="Model", title_fontsize=12, fontsize=11, loc="lower right")
            plt.tight_layout()

            # Save line plot
            output_file = os.path.join(output_dir, f"hit_rate_lineplot_{timestamp}.png")
            plt.savefig(output_file)
            plt.close()
            print(f"Hit Rate line plot saved to: {output_file}")

        if metric_name == "ont_similarity":
            # ---------------------------------------------------------------------
            # 1. Ontology Similarity Bar Chart with Error Bars if available
            # ---------------------------------------------------------------------
            plt.figure(figsize=(12, 8))
            k_values = [1, 3, 5, 10]

            # Check if we have raw data for error bars
            has_raw_data = False
            ont_sim_error_data = []

            for i, summary in enumerate(summaries):
                for k in k_values:
                    # Look for raw Ontology Similarity values
                    key = f"ont_similarity@{k}"
                    raw_key = f"ont_similarity@{k}_per_case"
                    os_raw = summary.get(raw_key) or summary.get(f"raw_{key}") or None

                    if isinstance(os_raw, list) and len(os_raw) > 1:
                        has_raw_data = True
                        ont_sim_error_data.append(
                            {
                                "model": model_names[i],
                                "metric": f"OntSim@{k}",
                                "mean": np.mean(os_raw),
                                "std": np.std(os_raw),
                                "k": k,
                            }
                        )

            if has_raw_data and len(ont_sim_error_data) > 0:
                # Create grouped bar chart with error bars similar to Hit Rate
                os_error_df = pd.DataFrame(ont_sim_error_data)

                # Group by model and metric for positioning
                unique_models = sorted(os_error_df["model"].unique())
                metrics = [f"OntSim@{k}" for k in k_values]
                metrics = [m for m in metrics if m in os_error_df["metric"].unique()]

                # Calculate bar positions
                bar_width = 0.2 if len(unique_models) <= 3 else 0.15
                x = np.arange(len(unique_models))

                # Plot each metric as a group
                viridis_colors = sns.color_palette("viridis", len(metrics))
                color_map = {
                    metric: viridis_colors[i] for i, metric in enumerate(metrics)
                }

                for i, metric in enumerate(metrics):
                    metric_df = os_error_df[os_error_df["metric"] == metric]
                    metric_df = (
                        metric_df.set_index("model")
                        .reindex(unique_models)
                        .reset_index()
                    )

                    # Position the bars for this metric
                    offset = (i - len(metrics) / 2 + 0.5) * bar_width

                    plt.bar(
                        x + offset,
                        metric_df["mean"],
                        bar_width,
                        yerr=metric_df["std"],
                        capsize=3,
                        label=metric,
                        color=color_map[metric],
                    )

                plt.xticks(x, unique_models, rotation=45, ha="right")
                plt.legend(title="Metric", loc="upper left")

                # Set y-axis limits (0 to 1 with padding for error bars)
                min_y = max(
                    0,
                    min([r["mean"] - r["std"] for _, r in os_error_df.iterrows()])
                    - 0.05,
                )
                plt.ylim(bottom=min_y, top=1.05)
            else:
                # Check if we can do a line chart (one line per model)
                ont_sim_data = []
                for i, model in enumerate(model_names):
                    values = []
                    for k in k_values:
                        key = f"ont_similarity@{k}"
                        values.append(summaries[i].get(key, 0))

                    if any(values):
                        plt.plot(k_values, values, marker="o", label=model, linewidth=2)

                plt.xticks(k_values)
                plt.legend(title="Model")
                plt.ylim(0, 1.05)

            plt.title("Ontology Similarity at K by Model")
            plt.xlabel("K")
            plt.ylabel("Ontology Similarity")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save figure
            output_file = os.path.join(
                output_dir, f"ont_similarity_barplot_{timestamp}.png"
            )
            plt.savefig(output_file)
            plt.close()
            print(f"Ontology Similarity bar plot saved to: {output_file}")

            # ---------------------------------------------------------------------
            # 2. Ontology Similarity Line Plot with connected dots (by K value)
            # ---------------------------------------------------------------------
            plt.figure(figsize=(12, 8))

            # Extract data for line plot (one line per model, x-axis = k values)
            for i, model in enumerate(model_names):
                ont_sim_by_k = []
                for k in k_values:
                    key = f"ont_similarity@{k}"
                    ont_sim_by_k.append(summaries[i].get(key, 0))

                if any(ont_sim_by_k):
                    model_color = plt.cm.viridis(i / max(1, len(model_names) - 1))
                    plt.plot(
                        k_values,
                        ont_sim_by_k,
                        "o-",
                        linewidth=2.5,
                        markersize=8,
                        label=model,
                        color=model_color,
                    )

            plt.title("Ontology Similarity by K Value", fontsize=14)
            plt.xlabel("K", fontsize=12)
            plt.ylabel("Ontology Similarity", fontsize=12)
            plt.xticks(k_values, fontsize=11)
            plt.yticks(fontsize=11)
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(title="Model", title_fontsize=12, fontsize=11, loc="lower right")
            plt.tight_layout()

            # Save line plot
            output_file = os.path.join(
                output_dir, f"ont_similarity_lineplot_{timestamp}.png"
            )
            plt.savefig(output_file)
            plt.close()
            print(f"Ontology Similarity line plot saved to: {output_file}")

    return os.path.join(output_dir, f"mrr_comparison_{timestamp}.png")


def main() -> None:
    """Main function for managing benchmark results."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Manage and visualize benchmark results"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file for the comparison table",
    )
    compare_parser.add_argument(
        "--summaries-dir",
        type=str,
        default=SUMMARIES_DIR,
        help=f"Directory containing summary JSON files (default: {SUMMARIES_DIR})",
    )

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Create visualizations")
    visualize_parser.add_argument(
        "--metric",
        type=str,
        choices=["mrr", "hit_rate", "ont_similarity", "all"],
        default="all",
        help="Metric to visualize (default: all plots)",
    )
    visualize_parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to include",
    )
    visualize_parser.add_argument(
        "--summaries-dir",
        type=str,
        default=SUMMARIES_DIR,
        help=f"Directory containing summary JSON files (default: {SUMMARIES_DIR})",
    )
    visualize_parser.add_argument(
        "--output-dir",
        type=str,
        default=VISUALIZATIONS_DIR,
        help=f"Directory to save visualization images (default: {VISUALIZATIONS_DIR})",
    )

    # Debug option for both commands
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    # Ensure output directories exist
    ensure_directories_exist()

    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        return

    # Load summary files
    summaries_dir = (
        args.summaries_dir if hasattr(args, "summaries_dir") else SUMMARIES_DIR
    )

    # Make sure the summaries directory exists
    os.makedirs(summaries_dir, exist_ok=True)

    summaries = load_summary_files(summaries_dir)

    if not summaries:
        logging.error(f"No summary files found in {summaries_dir}")
        return

    logging.info(f"Loaded {len(summaries)} summary files")

    # Execute the specified command
    if args.command == "compare":
        compare_summaries(summaries, args.output)

    elif args.command == "visualize":
        # Parse models list if provided
        include_models = None
        if hasattr(args, "models") and args.models:
            include_models = [m.strip() for m in args.models.split(",")]

        # Ensure output directory exists
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Deduplicate summaries to avoid issues with duplicate model labels
        deduplicated_summaries = deduplicate_summaries(summaries)
        logging.info(
            f"Using {len(deduplicated_summaries)} unique models for visualization"
        )

        visualize_results(
            deduplicated_summaries,
            metric=args.metric,
            include_models=include_models,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    # Import necessary libraries here to avoid unnecessarily importing them when not needed
    import pandas as pd
    import matplotlib.pyplot as plt

    main()
