#!/usr/bin/env python3
"""
German HPO RAG Benchmark Management Tool

This script provides an integrated command-line interface for:
1. Setting up HPO indexes with various embedding models
2. Running benchmarks using the configured models
3. Comparing results across models
4. Visualizing performance metrics

Usage examples:
  # Setup indexes for all supported models
  python manage_benchmarks.py setup --all

  # Setup an index for a specific model
  python manage_benchmarks.py setup --model-name "FremyCompany/BioLORD-2023-M"

  # Run benchmarks on all models
  python manage_benchmarks.py run --all

  # Run benchmark on a specific model
  python manage_benchmarks.py run --model-name "FremyCompany/BioLORD-2023-M"

  # Compare all benchmark results
  python manage_benchmarks.py compare

  # Compare specific models
  python manage_benchmarks.py compare --models "biolord_2023_m" "jina_embeddings_v2_base_de"
"""

import os
import argparse
import subprocess
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from utils import get_model_slug
import time
import glob

# Import run_benchmark directly from benchmark_rag.py
from benchmark_rag import run_benchmark, load_test_data, DEFAULT_TEST_FILE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Default models to use
DEFAULT_MODELS = [
    "FremyCompany/BioLORD-2023-M",
    "jinaai/jina-embeddings-v2-base-de",
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
]

# Directories for saving results
RESULTS_DIR = "benchmark_results"
SUMMARIES_DIR = os.path.join(RESULTS_DIR, "summaries")
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, "visualizations")


def ensure_directories():
    """Create necessary directories for results storage."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    os.makedirs("data/test_cases", exist_ok=True)


def run_command(command, desc=None, capture_output=True):
    """Run a subprocess command and log the output."""
    if desc:
        logging.info(desc)

    logging.debug(f"Running: {' '.join(command)}")

    try:
        if capture_output:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return True, result.stdout, result.stderr
        else:
            # Run without capturing output (directly to console)
            result = subprocess.run(command)
            return result.returncode == 0, None, None
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            logging.error(f"Error output: {e.stderr}")
        return False, e.stdout if e.stdout else None, e.stderr if e.stderr else None


def setup_model(model_name, batch_size=100):
    """Set up HPO index with the specified model."""
    logging.info(f"Setting up HPO index with model: {model_name}")

    command = [
        "python",
        "setup_hpo_index.py",
        "--model-name",
        model_name,
        "--batch-size",
        str(batch_size),
    ]

    success, stdout, stderr = run_command(command, f"Setting up index for {model_name}")
    if success:
        logging.info(f"Successfully set up index for {model_name}")
    else:
        logging.error(f"Failed to set up index for {model_name}")

    return success


def run_benchmark_wrapper(
    model_name, similarity_threshold=0.1, test_file=None, detailed=False
):
    """Run benchmark for a single model and save results."""
    start_time = time.time()
    logging.info(f"Running benchmark for {model_name}")

    # Use the default test file if none provided
    if not test_file:
        test_file = DEFAULT_TEST_FILE

    # Load test data
    test_cases = load_test_data(test_file)
    if not test_cases:
        logging.error(f"Failed to load test cases from {test_file}")
        return None

    # Call run_benchmark directly from benchmark_rag.py
    results = run_benchmark(
        model_name, test_cases, similarity_threshold=similarity_threshold
    )

    if not results:
        logging.error(f"Benchmark failed for model {model_name}")
        return None

    end_time = time.time()
    logging.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")

    # The results are already saved by benchmark_rag.py, just return them
    return results


def save_results(results, model_name):
    """Save benchmark results to CSV and JSON files."""
    ensure_directories()

    model_slug = results["model"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON with detailed metrics
    json_path = os.path.join(SUMMARIES_DIR, f"{model_slug}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save as CSV for compatibility
    df = pd.DataFrame([results])
    csv_path = os.path.join(RESULTS_DIR, f"{model_slug}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    logging.info(f"Results saved as JSON: {json_path}")
    logging.info(f"Results saved as CSV: {csv_path}")

    return json_path, csv_path


def load_all_results(filter_models=None):
    """Load all benchmark result summaries.

    Args:
        filter_models: Optional list of model slugs to filter results

    Returns:
        List of benchmark result dictionaries with the latest results for each model
    """
    ensure_directories()

    # Get all JSON summary files
    json_files = glob.glob(os.path.join(SUMMARIES_DIR, "*.json"))

    if not json_files:
        logging.warning("No benchmark results found")
        return None

    # Track the latest result for each model by timestamp
    latest_results = {}

    # Load results
    for json_file in json_files:
        with open(json_file, "r") as f:
            try:
                results = json.load(f)

                # Skip if we're filtering and this model isn't in the filter list
                model_name = results.get("model", "")
                if filter_models and model_name not in filter_models:
                    continue

                # Get timestamp
                timestamp = results.get("timestamp", "")

                # If we don't have this model yet, or this is a newer result
                if model_name not in latest_results or timestamp > latest_results[
                    model_name
                ].get("timestamp", ""):
                    latest_results[model_name] = results
            except json.JSONDecodeError:
                logging.warning(f"Error parsing JSON file: {json_file}")
                continue

    # Convert dictionary to list
    all_results = list(latest_results.values())

    if not all_results:
        logging.warning("No matching benchmark results found")
        return None

    # Sort results by model name
    all_results.sort(key=lambda x: x.get("model", ""))

    return all_results


def compare_models(models_to_compare=None):
    """Compare models based on benchmark results.

    Args:
        models_to_compare: Either a list of model slugs or a list of result dictionaries

    Returns:
        DataFrame with model comparison metrics
    """
    # Load results based on what was provided
    if models_to_compare is None:
        # If nothing provided, load all available results
        results_list = load_all_results()
    elif (
        isinstance(models_to_compare, list)
        and models_to_compare
        and isinstance(models_to_compare[0], str)
    ):
        # If list of model slugs provided, load those specific results
        results_list = load_all_results(models_to_compare)
    else:
        # If list of result dictionaries provided, use as is
        results_list = models_to_compare

    # Check if we have any results to work with
    if not results_list or len(results_list) == 0:
        logging.warning("No results provided for comparison")
        return None

    # Create a comparison dataframe with the metrics
    comparison_data = []
    for result in results_list:
        # Use the right key based on what's available (model or model_slug)
        model_name = result.get("model_slug", result.get("model", "Unknown"))
        # Extract MRR directly from the result (avg_mrr key might not exist)
        row = {"model": model_name, "mrr": result.get("mrr", 0)}

        # Add hit rates
        for k in [1, 3, 5, 10]:
            # First check for the direct key, then fallback to avg_ prefix
            if f"hit_rate@{k}" in result:
                row[f"hit_rate@{k}"] = result[f"hit_rate@{k}"]
            elif f"avg_hit_rate@{k}" in result:
                row[f"hit_rate@{k}"] = result[f"avg_hit_rate@{k}"]

        # Add ontology similarity metrics
        for k in [1, 3, 5, 10]:
            # First check for the direct key, then fallback to avg_ prefix
            if f"ont_similarity@{k}" in result:
                row[f"ont_similarity@{k}"] = result[f"ont_similarity@{k}"]
            elif f"avg_ont_similarity@{k}" in result:
                row[f"ont_similarity@{k}"] = result[f"avg_ont_similarity@{k}"]

        comparison_data.append(row)

    comparison = pd.DataFrame(comparison_data)

    # Select columns to display
    display_cols = ["model", "mrr"]

    # Add hit rate columns
    for k in [1, 3, 5, 10]:
        hr_col = f"hit_rate@{k}"
        if hr_col in comparison.columns:
            display_cols.append(hr_col)

    # Add ontology similarity columns
    for k in [1, 3, 5, 10]:
        os_col = f"ont_similarity@{k}"
        if os_col in comparison.columns:
            display_cols.append(os_col)

    # Create final comparison dataframe with selected columns
    comparison_df = comparison[display_cols]

    # Rename columns for better display
    rename_map = {f"hit_rate@{k}": f"Hit@{k}" for k in [1, 3, 5, 10]}
    rename_map.update({f"ont_similarity@{k}": f"OntSim@{k}" for k in [1, 3, 5, 10]})
    rename_map["mrr"] = "MRR"
    rename_map["model"] = "Model"

    comparison_df = comparison_df.rename(columns=rename_map)

    return comparison_df


def visualize_results(results_list):
    """Visualize benchmark results directly from the results list.

    Args:
        results_list: Either a list of model slugs or benchmark result dictionaries

    Returns:
        Path to saved visualization file
    """
    # Load results if models_to_compare is a list of model slugs
    if (
        isinstance(results_list, list)
        and results_list
        and isinstance(results_list[0], str)
    ):
        loaded_results = load_all_results(results_list)
    else:
        loaded_results = results_list

    if not loaded_results:
        logging.warning("No results provided for visualization")
        return None

    # Build a dataframe from the results
    visualization_data = []
    for result in loaded_results:
        row = {
            "model": result.get("model_slug", result.get("model", "Unknown")),
            "mrr": result.get("mrr", 0),  # Use direct key, not avg_mrr
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
        }

        # Add hit rates
        for k in [1, 3, 5, 10]:
            # First check for the direct key, then fallback to avg_ prefix
            if f"hit_rate@{k}" in result:
                row[f"hit_rate@{k}"] = result[f"hit_rate@{k}"]
            elif f"avg_hit_rate@{k}" in result:
                row[f"hit_rate@{k}"] = result[f"avg_hit_rate@{k}"]

        # Add ontology similarity metrics
        for k in [1, 3, 5, 10]:
            # First check for the direct key, then fallback to avg_ prefix
            if f"ont_similarity@{k}" in result:
                row[f"ont_similarity@{k}"] = result[f"ont_similarity@{k}"]
            elif f"avg_ont_similarity@{k}" in result:
                row[f"ont_similarity@{k}"] = result[f"avg_ont_similarity@{k}"]

        visualization_data.append(row)

    # Create dataframe and sort by model name
    df = pd.DataFrame(visualization_data)
    df = df.sort_values("model")

    # Set timestamp for the plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Verify we have at least one model with results
    if len(df) == 0:
        logging.warning("No valid results for visualization")
        return None

    # Create the visualization directory if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    # Visualize MRR score
    plt.figure(figsize=(10, 6))
    sns.barplot(x="model", y="mrr", data=df, palette="viridis")
    plt.title("Mean Reciprocal Rank (MRR) by Model")
    plt.xlabel("Model")
    plt.ylabel("MRR")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    mrr_plot_path = os.path.join(VISUALIZATIONS_DIR, f"mrr_comparison_{timestamp}.png")
    plt.savefig(mrr_plot_path)
    plt.close()

    # Visualize Hit@K rates
    hit_rate_cols = [col for col in df.columns if col.startswith("hit_rate@")]
    if hit_rate_cols:
        plt.figure(figsize=(12, 8))
        hit_rate_df = df.melt(
            id_vars=["model"],
            value_vars=hit_rate_cols,
            var_name="Metric",
            value_name="Value",
        )
        # Clean up metric names for plot
        hit_rate_df["Metric"] = hit_rate_df["Metric"].str.replace("hit_rate@", "Hit@")

        sns.barplot(
            x="model", y="Value", hue="Metric", data=hit_rate_df, palette="viridis"
        )
        plt.title("Hit Rate by Model")
        plt.xlabel("Model")
        plt.ylabel("Hit Rate")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Metric")
        plt.tight_layout()

        # Save plot
        hit_rate_plot_path = os.path.join(
            VISUALIZATIONS_DIR, f"hit_rate_comparison_{timestamp}.png"
        )
        plt.savefig(hit_rate_plot_path)
        plt.close()

    # Visualize Ontology Similarity@K rates
    ont_sim_cols = [col for col in df.columns if col.startswith("ont_similarity@")]
    if ont_sim_cols:
        plt.figure(figsize=(12, 8))
        ont_sim_df = df.melt(
            id_vars=["model"],
            value_vars=ont_sim_cols,
            var_name="Metric",
            value_name="Value",
        )
        # Clean up metric names for plot
        ont_sim_df["Metric"] = ont_sim_df["Metric"].str.replace(
            "ont_similarity@", "OntSim@"
        )

        sns.barplot(
            x="model", y="Value", hue="Metric", data=ont_sim_df, palette="viridis"
        )
        plt.title("Ontology Similarity by Model")
        plt.xlabel("Model")
        plt.ylabel("Ontology Similarity")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Metric")
        plt.tight_layout()

        # Save plot
        ont_sim_plot_path = os.path.join(
            VISUALIZATIONS_DIR, f"ont_similarity_comparison_{timestamp}.png"
        )
        plt.savefig(ont_sim_plot_path)
        plt.close()

    # Create a combined metrics comparison chart (side-by-side)
    plt.figure(figsize=(15, 10))

    # Create 3 subplots
    plt.subplot(3, 1, 1)
    sns.barplot(x="model", y="mrr", data=df, palette="viridis")
    plt.title("Mean Reciprocal Rank (MRR) by Model")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    if hit_rate_cols:
        sns.barplot(
            x="model", y="Value", hue="Metric", data=hit_rate_df, palette="viridis"
        )
        plt.title("Hit Rate by Model")
        plt.xlabel("")
        plt.legend(title="Metric")
        plt.tight_layout()

    plt.subplot(3, 1, 3)
    if ont_sim_cols:
        sns.barplot(
            x="model", y="Value", hue="Metric", data=ont_sim_df, palette="viridis"
        )
        plt.title("Ontology Similarity by Model")
        plt.xlabel("Model")
        plt.legend(title="Metric")
        plt.tight_layout()

    # Save the combined plot
    combined_plot_path = os.path.join(
        VISUALIZATIONS_DIR, f"combined_metrics_comparison_{timestamp}.png"
    )
    plt.savefig(combined_plot_path)
    plt.close()

    # Return path to main visualizations
    return combined_plot_path


def compare_subcommand(args):
    """Handle the compare subcommand."""
    models_to_compare = args.models if args.models else None

    # Get comparison
    comparison_df = compare_models(models_to_compare)

    if comparison_df is not None:
        print("\n===== Model Comparison =====")
        print(f"Models compared: {len(comparison_df)}")
        print("\nPerformance Metrics:")

        # Format and display
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(comparison_df)

        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(SUMMARIES_DIR, f"comparison_{timestamp}.csv")
        comparison_df.to_csv(csv_path)
        print(f"\nComparison table saved to {csv_path}")

        # Generate visualization
        # Pass the actual loaded results to visualize_results, not just the model names
        results_list = load_all_results(models_to_compare)
        vis_path = visualize_results(results_list)
        if vis_path:
            print(f"Visualization saved to {vis_path}")

        return True
    else:
        logging.error("No results available for comparison")
        return False


def main():
    # Create main parser
    parser = argparse.ArgumentParser(
        description="German HPO RAG Benchmark Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Setup subcommand
    setup_parser = subparsers.add_parser(
        "setup", help="Set up HPO indexes with embedding models"
    )
    setup_group = setup_parser.add_mutually_exclusive_group(required=True)
    setup_group.add_argument(
        "--all", action="store_true", help="Set up all default models"
    )
    setup_group.add_argument("--model-name", type=str, help="Model name to set up")
    setup_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing documents (default: 100)",
    )

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run benchmarks on models")
    run_group = run_parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        "--all", action="store_true", help="Run benchmarks on all default models"
    )
    run_group.add_argument("--model-name", type=str, help="Model name to benchmark")
    run_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.1,
        help="Similarity threshold for filtering results (default: 0.1)",
    )
    run_parser.add_argument(
        "--test-file", type=str, help="Custom test file to use for benchmarking"
    )
    run_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed per-test-case results"
    )

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument(
        "--models", nargs="+", help="Models to compare (model slugs)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Ensure results directories exist
    ensure_directories()

    # Execute appropriate subcommand
    if args.command == "setup":
        success = setup_subcommand(args)
    elif args.command == "run":
        success = run_subcommand(args)
    elif args.command == "compare":
        success = compare_subcommand(args)
    else:
        parser.print_help()
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
