#!/usr/bin/env python3
"""
Benchmark Results Management Script

This script provides functionality for:
1. Comparing results across different model runs (generates CSV).
2. Visualizing benchmark results with plots.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Any, Dict
from datetime import datetime
import pandas as pd

# Determine the absolute paths needed for proper imports
script_file = os.path.abspath(__file__)
script_dir = os.path.dirname(script_file)

# Navigate from scripts dir to the project root
project_root = os.path.dirname(os.path.dirname(script_dir))

# Path to the actual package containing the modules
actual_package_dir = os.path.join(
    project_root, "multilingual_hpo_rag", "multilingual_hpo_rag"
)

# Add all necessary paths to sys.path
paths_to_add = [project_root, actual_package_dir, os.path.dirname(project_root)]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Debug: Print the sys.path to help diagnose issues
# print(f"Script directory: {script_dir}")
# print(f"Project root: {project_root}")
# print(f"Actual package directory: {actual_package_dir}")
# print("Python sys.path:")
# for idx, path in enumerate(sys.path):
#     print(f"  {idx}: {path}")

# Import configuration constants
try:
    # Try direct import first
    from config import SUMMARIES_DIR, VISUALIZATIONS_DIR, DETAILED_DIR
except ImportError:
    # Fall back to full package path
    from multilingual_hpo_rag.config import (
        SUMMARIES_DIR,
        VISUALIZATIONS_DIR,
        DETAILED_DIR,
    )

# Import required modules with fallback mechanisms
try:
    # Try direct imports first (works when the script directory is in sys.path)
    from evaluation.result_analyzer import (
        load_summary_files,
        deduplicate_summaries,
        prepare_comparison_dataframe,
        prepare_flat_dataframe_for_plotting,
    )
except ImportError:
    # Fall back to full package path
    from multilingual_hpo_rag.evaluation.result_analyzer import (
        load_summary_files,
        deduplicate_summaries,
        prepare_comparison_dataframe,
        prepare_flat_dataframe_for_plotting,
    )

try:
    # Try direct imports first
    from visualization.plot_utils import (
        plot_mrr_comparison,
        plot_metric_at_k_bars,
        plot_metric_at_k_lines,
        K_VALUES_DEFAULT,
    )
except ImportError:
    # Fall back to full package path
    from multilingual_hpo_rag.visualization.plot_utils import (
        plot_mrr_comparison,
        plot_metric_at_k_bars,
        plot_metric_at_k_lines,
        K_VALUES_DEFAULT,
    )


def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",  # Added module to format
        handlers=[logging.StreamHandler()],
    )
    # Ensure root logger is also set if other modules are importing logging
    logging.getLogger().setLevel(level)


def ensure_directories_exist() -> None:
    """Create necessary output directories if they don't exist."""
    for directory in [SUMMARIES_DIR, VISUALIZATIONS_DIR, DETAILED_DIR]:
        if not os.path.exists(directory):
            logging.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)


def compare_summaries_cli(
    summaries: List[Dict[str, Any]], output_csv_path: Optional[str] = None
) -> None:
    """
    Prepares, prints, and optionally saves a comparison table from benchmark summaries.
    This function is intended for direct CLI output and CSV saving.
    """
    if not summaries:
        logging.error("No summaries to compare")
        return

    comparison_df = prepare_comparison_dataframe(summaries)

    if comparison_df.empty:
        logging.warning("No data to compare after processing summaries.")
        return

    # Display the table
    pd.options.display.float_format = "{:.4f}".format
    print("\n===== Benchmark Comparison =====")
    print(comparison_df.to_string(index=False))

    if output_csv_path:
        try:
            comparison_df.to_csv(output_csv_path, index=False)
            print(f"\nComparison table saved to: {output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save comparison CSV to {output_csv_path}: {e}")


def visualize_results_cli(
    summaries: List[Dict[str, Any]],
    metrics_to_plot: List[str],  # e.g., ["mrr", "hit_rate", "ont_similarity"]
    output_dir: str = VISUALIZATIONS_DIR,
) -> None:
    """
    Generates and saves visualizations for the specified metrics.
    """
    if not summaries:
        logging.error("No summaries provided for visualization.")
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Log the keys available in the first summary to help with debugging
    if len(summaries) > 0 and logging.getLogger().level <= logging.DEBUG:
        first_summary = summaries[0]
        logging.debug(f"Available keys in first summary: {list(first_summary.keys())}")

    # Prepare dataframes once
    # For MRR, we can derive from the comparison_df or pass summaries directly
    mrr_plot_df_data = []

    # Debug information to help understand what's in the summaries
    if len(summaries) > 0:
        logging.info(f"First summary keys: {list(summaries[0].keys())}")

    for s in summaries:
        model_name = s.get("model", "Unknown")
        reranker_enabled = s.get("reranker_enabled", False)

        # Direct access to the MRR values from summary
        dense_mrr = s.get("mrr_dense", 0.0)
        reranked_mrr = s.get("mrr_reranked", 0.0) if reranker_enabled else None

        # Create a row for this model
        row_data = {
            "model": model_name,
            "metric": "MRR",
            "Dense": dense_mrr,
        }

        # Add reranked value if available
        if reranked_mrr is not None:
            row_data["Re-Ranked"] = reranked_mrr

        mrr_plot_df_data.append(row_data)
    # Convert to DataFrame for MRR plot
    mrr_plot_df = pd.DataFrame(mrr_plot_df_data)

    if "mrr" in metrics_to_plot:
        logging.info("Creating MRR comparison plots...")
        if "model" in mrr_plot_df.columns:
            logging.info(f"MRR data: {mrr_plot_df}")
            plot_mrr_comparison(mrr_plot_df, output_dir, timestamp)

    if "hit_rate" in metrics_to_plot or "all" in metrics_to_plot:
        # Work directly with the summary data instead of the comparison DataFrame
        plot_data = []
        k_values = (
            K_VALUES_DEFAULT if "K_VALUES_DEFAULT" in globals() else [1, 3, 5, 10]
        )

        # Debug information to help understand what's in the summaries
        if len(summaries) > 0:
            logging.info(f"First summary keys: {list(summaries[0].keys())}")

        for summary in summaries:
            model_name = summary.get("model", "Unknown")
            reranker_enabled = summary.get("reranker_enabled", False)

            for k in k_values:
                # Directly access hit rate values from the summary with appropriate key names
                dense_key = f"hit_rate_dense@{k}"
                if dense_key in summary:
                    plot_data.append(
                        {
                            "model": model_name,
                            "k": k,
                            "method": "Dense",
                            "value": float(summary[dense_key]),
                            "std_dev": 0.0,
                        }
                    )

                if reranker_enabled:
                    reranked_key = f"hit_rate_reranked@{k}"
                    if reranked_key in summary:
                        plot_data.append(
                            {
                                "model": model_name,
                                "k": k,
                                "method": "Re-Ranked",
                                "value": float(summary[reranked_key]),
                                "std_dev": 0.0,
                            }
                        )

        flat_df_hr = pd.DataFrame(plot_data)
        if not flat_df_hr.empty:
            logging.info(f"Created Hit Rate plot data with {len(flat_df_hr)} entries")
            plot_metric_at_k_bars(
                flat_df_hr, "Hit Rate", "Hit Rate", output_dir, timestamp
            )
            plot_metric_at_k_lines(
                flat_df_hr, "Hit Rate", "Hit Rate", output_dir, timestamp
            )
        else:
            logging.warning("No data available to plot Hit Rate.")

    if "ont_similarity" in metrics_to_plot or "all" in metrics_to_plot:
        # Work directly with the summary data for ontology similarity too
        plot_data = []
        k_values = (
            K_VALUES_DEFAULT if "K_VALUES_DEFAULT" in globals() else [1, 3, 5, 10]
        )

        for summary in summaries:
            model_name = summary.get("model", "Unknown")
            reranker_enabled = summary.get("reranker_enabled", False)

            for k in k_values:
                # Directly access ontology similarity values from the summary with appropriate key names
                dense_key = f"ont_similarity_dense@{k}"
                if dense_key in summary:
                    plot_data.append(
                        {
                            "model": model_name,
                            "k": k,
                            "method": "Dense",
                            "value": float(summary[dense_key]),
                            "std_dev": 0.0,
                        }
                    )

                if reranker_enabled:
                    reranked_key = f"ont_similarity_reranked@{k}"
                    if reranked_key in summary:
                        plot_data.append(
                            {
                                "model": model_name,
                                "k": k,
                                "method": "Re-Ranked",
                                "value": float(summary[reranked_key]),
                                "std_dev": 0.0,
                            }
                        )

        flat_df_os = pd.DataFrame(plot_data)
        if not flat_df_os.empty:
            logging.info(
                f"Created Ontology Similarity plot data with {len(flat_df_os)} entries"
            )
            plot_metric_at_k_bars(
                flat_df_os,
                "Ontology Similarity",
                "Ontology Similarity",
                output_dir,
                timestamp,
            )
            plot_metric_at_k_lines(
                flat_df_os,
                "Ontology Similarity",
                "Ontology Similarity",
                output_dir,
                timestamp,
            )
        else:
            logging.warning("No data available to plot Ontology Similarity.")


def main() -> None:
    """Main function for managing benchmark results."""
    parser = argparse.ArgumentParser(
        description="Manage and visualize benchmark results"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute", required=True
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(
            DETAILED_DIR,
            f"benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        ),
        help="Output CSV file for the comparison table (default: benchmark_results/detailed/benchmark_comparison_TIMESTAMP.csv)",
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
        "--metrics",
        type=str,
        choices=["mrr", "hit_rate", "ont_similarity", "all"],
        default="all",
        help="Comma-separated list of metrics to visualize (e.g., mrr,hit_rate) or 'all' (default: all plots)",
    )
    # --models filter was removed from visualize for now, can be re-added if complex filtering on summaries is needed
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(args.debug)
    ensure_directories_exist()

    summaries_dir = args.summaries_dir
    os.makedirs(summaries_dir, exist_ok=True)  # Ensure it exists before loading

    all_summaries = load_summary_files(summaries_dir)
    if not all_summaries:
        logging.error(f"No summary files found in {summaries_dir}. Exiting.")
        return

    # Deduplicate summaries to get the latest run for each model configuration
    # This is important if multiple runs for the same model exist.
    # The definition of "unique run" might need adjustment if you run same model with different reranker configs.
    # For now, deduplicating by 'model' slug primarily.
    unique_model_summaries = deduplicate_summaries(all_summaries)
    logging.info(
        f"Using {len(unique_model_summaries)} unique model summaries for processing."
    )

    if args.command == "compare":
        compare_summaries_cli(unique_model_summaries, args.output_csv)

    elif args.command == "visualize":
        metrics_to_plot_list = []
        if args.metrics == "all":
            metrics_to_plot_list = ["mrr", "hit_rate", "ont_similarity"]
        else:
            metrics_to_plot_list = [m.strip() for m in args.metrics.split(",")]

        os.makedirs(args.output_dir, exist_ok=True)  # Ensure plot output dir exists

        visualize_results_cli(
            unique_model_summaries,
            metrics_to_plot=metrics_to_plot_list,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
