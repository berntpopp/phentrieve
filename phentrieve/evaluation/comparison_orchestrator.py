"""
Benchmark comparison orchestration module for the CLI.

This module compares benchmark summary files and can generate simple charts for
dense retrieval metrics.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from phentrieve.benchmark.result_store import discover_artifacts
from phentrieve.config import DEFAULT_VISUALIZATIONS_SUBDIR
from phentrieve.utils import get_default_results_dir, resolve_data_path

logger = logging.getLogger(__name__)


def load_benchmark_summaries(summaries_dir: str) -> list[dict[str, Any]]:
    """Load benchmark summary files from the specified directory."""
    summary_files = discover_artifacts(Path(summaries_dir), "summary")
    if not summary_files:
        logger.warning("No summary files found in %s", summaries_dir)
        return []

    summaries = []
    for file_path in summary_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                summaries.append(json.load(f))
            logger.debug("Loaded summary from %s", file_path)
        except Exception as e:
            logger.error("Error loading summary file %s: %s", file_path, e)

    logger.info("Loaded %s benchmark summaries", len(summaries))
    return summaries


def compare_benchmark_summaries(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Compare benchmark summaries and create a dense retrieval metrics table."""
    if not summaries:
        logger.warning("No summaries provided for comparison")
        return pd.DataFrame()

    comparison_data = []
    for summary in summaries:
        model_data = {
            "Model": summary.get("model", "Unknown"),
            "Original Model Name": summary.get("original_model_name", "Unknown"),
            "Timestamp": summary.get("timestamp", "Unknown"),
            "Dataset": summary.get(
                "dataset_name", summary.get("test_file", "Unknown")
            ),
            "Run ID": summary.get("run_id", "legacy"),
            "Test Cases": summary.get("num_test_cases", 0),
            "MRR (Dense)": summary.get("mrr_dense", 0),
        }

        for k in [1, 3, 5, 10]:
            hit_key = f"hit_rate_dense@{k}"
            if hit_key in summary:
                model_data[f"HR@{k} (Dense)"] = summary[hit_key]

            similarity_key = f"max_ont_similarity_dense@{k}"
            if similarity_key in summary:
                model_data[f"MaxOntSim@{k} (Dense)"] = summary[similarity_key]

        comparison_data.append(model_data)

    df = pd.DataFrame(comparison_data)
    if "MRR (Dense)" in df.columns:
        df = df.sort_values("MRR (Dense)", ascending=False)
    return df


def orchestrate_benchmark_comparison(
    summaries_dir: str | None = None,
    output_csv: str | None = None,
    visualize: bool = False,
    output_dir: str | None = None,
    metrics: str = "all",
    debug: bool = False,
    results_dir_override: str | None = None,
) -> pd.DataFrame | None:
    """Load summaries, write a comparison CSV, and optionally generate charts."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    base_results_dir = resolve_data_path(
        results_dir_override, "results_dir", get_default_results_dir
    )
    summaries_load_path = (
        Path(summaries_dir).expanduser().resolve()
        if summaries_dir
        else base_results_dir
    )
    output_dir_viz = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else base_results_dir / DEFAULT_VISUALIZATIONS_SUBDIR
    )

    if output_csv:
        csv_save_path = Path(output_csv).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(base_results_dir, exist_ok=True)
        csv_save_path = base_results_dir / f"benchmark_comparison_{timestamp}.csv"

    summaries = load_benchmark_summaries(str(summaries_load_path))
    if not summaries:
        logger.warning("No benchmark summaries found in %s", summaries_load_path)
        return None

    comparison_df = compare_benchmark_summaries(summaries)
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    comparison_df.to_csv(str(csv_save_path))
    logger.info("Comparison table saved to %s", csv_save_path)

    if visualize:
        generate_visualizations(comparison_df, metrics, str(output_dir_viz), debug)

    return comparison_df


def generate_visualizations(
    comparison_df: pd.DataFrame,
    metrics: str = "all",
    output_dir: str | None = None,
    debug: bool = False,
) -> bool:
    """Generate bar charts for requested dense retrieval metrics."""
    if comparison_df.empty:
        logger.warning("Cannot generate visualizations from empty comparison data")
        return False

    metric_columns = [
        column
        for column in comparison_df.columns
        if column == "MRR (Dense)"
        or column.startswith("HR@")
        or column.startswith("MaxOntSim@")
    ]
    if metrics.lower() != "all":
        requested = {metric.strip() for metric in metrics.split(",") if metric.strip()}
        metric_columns = [column for column in metric_columns if column in requested]

    if not metric_columns:
        logger.warning("No matching metrics found for visualization")
        return False

    output_path = Path(output_dir or ".").expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    generated_any = False
    for metric in metric_columns:
        try:
            plot_df = (
                comparison_df[["Model", metric]]
                .dropna()
                .sort_values(metric, ascending=False)
            )
            if plot_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(plot_df["Model"], plot_df[metric], color="#2f6f9f")
            ax.set_title(metric)
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()

            safe_metric = (
                metric.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("@", "_at_")
            )
            fig.savefig(output_path / f"{safe_metric}.png", dpi=300)
            plt.close(fig)
            generated_any = True
        except Exception as e:
            logger.error("Failed to generate visualization for %s: %s", metric, e)
            if debug:
                raise

    return generated_any
