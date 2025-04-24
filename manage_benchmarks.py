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
        DataFrame with benchmark results
    """
    ensure_directories()

    # Get all JSON summary files
    json_files = glob.glob(os.path.join(SUMMARIES_DIR, "*.json"))

    if not json_files:
        logging.warning("No benchmark results found")
        return None

    # Load results
    all_results = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            results = json.load(f)

            # Filter by model if specified
            if filter_models and results["model"] not in filter_models:
                continue

            all_results.append(results)

    if not all_results:
        logging.warning("No matching benchmark results found")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Sort by model name
    if "model" in df.columns:
        df = df.sort_values(by="model")

    return df


def compare_models(results_list):
    """Compare models directly from the results_list."""
    if not results_list:
        logging.warning("No results provided for comparison")
        return None

    # Create a comparison dataframe with the metrics
    comparison_data = []
    for result in results_list:
        row = {"model": result["model_slug"], "mrr": result.get("avg_mrr", 0)}
        # Add hit rates
        for k in [1, 3, 5, 10]:
            if f"avg_hit_rate@{k}" in result:
                row[f"hit_rate@{k}"] = result[f"avg_hit_rate@{k}"]
        comparison_data.append(row)

    comparison = pd.DataFrame(comparison_data)

    # Select columns to display
    display_cols = ["model", "mrr"]

    # Add hit rate columns
    for k in [1, 3, 5, 10]:
        hr_col = f"hit_rate@{k}"
        if hr_col in comparison.columns:
            display_cols.append(hr_col)

    # Create final comparison dataframe with selected columns
    comparison_df = comparison[display_cols]

    # Rename columns for better display
    rename_map = {f"hit_rate@{k}": f"HR@{k}" for k in [1, 3, 5, 10]}
    rename_map["mrr"] = "MRR"
    rename_map["model"] = "Model"

    comparison_df = comparison_df.rename(columns=rename_map)

    return comparison_df


def visualize_results(results_list):
    """Visualize benchmark results directly from the results list.

    Args:
        results_list: List of benchmark result dictionaries

    Returns:
        Path to saved visualization file
    """
    if not results_list:
        logging.warning("No results provided for visualization")
        return None

    # Build a dataframe from the results
    visualization_data = []
    for result in results_list:
        row = {
            "model": result["model_slug"],
            "mrr": result.get("avg_mrr", 0),
            "timestamp": datetime.now().isoformat(),
        }

        # Add hit rates
        for k in [1, 3, 5, 10]:
            if f"avg_hit_rate@{k}" in result:
                row[f"hit_rate@{k}"] = result[f"avg_hit_rate@{k}"]

        visualization_data.append(row)

    # Create dataframe
    latest_results = pd.DataFrame(visualization_data)

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)

    # Plot MRR
    sns.barplot(x="model", y="mrr", data=latest_results, ax=axes[0])
    axes[0].set_title("Mean Reciprocal Rank (MRR)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("MRR")
    axes[0].tick_params(axis="x", rotation=45)

    # Plot Hit Rate @ K
    hr_cols = [col for col in latest_results.columns if col.startswith("hit_rate@")]
    if hr_cols:
        hr_data = []
        for model in latest_results["model"]:
            model_data = latest_results[latest_results["model"] == model]
            for hr_col in hr_cols:
                k = hr_col.split("@")[1]
                hr_data.append(
                    {
                        "model": model,
                        "k": int(k),
                        "hit_rate": model_data[hr_col].values[0],
                    }
                )

        hr_df = pd.DataFrame(hr_data)
        sns.lineplot(
            x="k",
            y="hit_rate",
            hue="model",
            markers=True,
            dashes=False,
            data=hr_df,
            ax=axes[1],
        )
        axes[1].set_title("Hit Rate @ K")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("Hit Rate")

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        VISUALIZATIONS_DIR, f"benchmark_comparison_{timestamp}.png"
    )

    plt.savefig(output_file)
    logging.info(f"Visualization saved to {output_file}")

    return output_file


def setup_subcommand(args):
    """Handle the setup subcommand."""
    models_to_setup = []

    if args.all:
        models_to_setup = DEFAULT_MODELS
    elif args.model_name:
        models_to_setup = [args.model_name]
    else:
        logging.error("Please specify either --all or --model-name")
        return False

    logging.info(f"Setting up {len(models_to_setup)} models")

    success_count = 0
    for model in models_to_setup:
        if setup_model(model, args.batch_size):
            success_count += 1

    if success_count == len(models_to_setup):
        logging.info("All models were set up successfully")
        return True
    else:
        logging.warning(
            f"{success_count}/{len(models_to_setup)} models were set up successfully"
        )
        return success_count > 0


def run_subcommand(args):
    """Handle the run subcommand."""
    models_to_run = []

    if args.all:
        models_to_run = DEFAULT_MODELS
    elif args.model_name:
        models_to_run = [args.model_name]
    else:
        logging.error("Please specify either --all or --model-name")
        return False

    logging.info(f"Running benchmarks for {len(models_to_run)} models")

    # Run benchmarks
    results_list = []
    for model in models_to_run:
        results = run_benchmark_wrapper(
            model,
            similarity_threshold=args.similarity_threshold,
            test_file=args.test_file,
            detailed=args.detailed,
        )
        if results:
            results_list.append(results)

    # Compare results
    if results_list:
        comparison_df = compare_models(results_list)
        if comparison_df is not None:
            print("\n===== Benchmark Results =====")
            print(f"Models evaluated: {len(results_list)}")
            print(f"Similarity threshold: {args.similarity_threshold}")
            print("\nModel Performance Comparison:")

            # Format and display
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            pd.set_option("display.float_format", "{:.4f}".format)
            print(comparison_df)

            # Also save the comparison
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(SUMMARIES_DIR, f"comparison_{timestamp}.csv")
            comparison_df.to_csv(csv_path)
            print(f"\nComparison table saved to {csv_path}")

            # Generate visualization
            vis_path = visualize_results(results_list)
            if vis_path:
                print(f"Visualization saved to {vis_path}")

        return True
    else:
        logging.error("No benchmarks completed successfully")
        return False


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
        vis_path = visualize_results(models_to_compare)
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
