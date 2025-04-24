#!/usr/bin/env python3
"""
Script to set up multiple embedding models and benchmark them in one go.

This utility helps with:
1. Setting up HPO indexes with multiple embedding models
2. Running benchmarks across all models
3. Comparing performance metrics
"""

import os
import argparse
import logging
import subprocess
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Default models to compare
DEFAULT_MODELS = [
    # Default model - multilingual MPNet
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # Alternative multilingual models
    "sentence-transformers/distiluse-base-multilingual-cased-v1",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/LaBSE",
    # Smaller model for faster processing
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def run_command(command, desc=None):
    """Run a subprocess command and log the output."""
    try:
        if desc:
            logging.info(desc)

        logging.debug(f"Running: {command}")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return False, e.stderr


def setup_model(model_name):
    """Set up HPO index with the specified model."""
    logging.info(f"Setting up HPO index with model: {model_name}")

    command = ["python", "setup_hpo_index.py", "--model-name", model_name]

    success, output = run_command(command, f"Setting up index for {model_name}")
    return success


def benchmark_models(models, test_file=None):
    """Run benchmarks on all specified models."""
    logging.info(f"Benchmarking {len(models)} models")

    command = ["python", "benchmark_rag.py", "--model-names"]
    command.extend(models)

    if test_file:
        command.extend(["--test-file", test_file])

    success, output = run_command(command, "Running benchmarks")
    return success, output


def visualize_results(csv_file="benchmark_results.csv"):
    """Create visualizations of benchmark results."""
    try:
        df = pd.read_csv(csv_file)

        # Group by model to get averages
        model_metrics = df.groupby("model").mean().reset_index()

        # Set up the figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.3)

        # Plot MRR
        sns.barplot(x="model", y="mrr", data=model_metrics, ax=axes[0, 0])
        axes[0, 0].set_title("Mean Reciprocal Rank (MRR)")
        axes[0, 0].set_xlabel("")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Plot F1
        sns.barplot(x="model", y="f1", data=model_metrics, ax=axes[0, 1])
        axes[0, 1].set_title("F1 Score")
        axes[0, 1].set_xlabel("")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot Hit Rate@k
        hr_cols = [col for col in model_metrics.columns if col.startswith("hit_rate@")]
        if hr_cols:
            hr_data = []
            for model in model_metrics["model"].unique():
                model_data = model_metrics[model_metrics["model"] == model]
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
                ax=axes[1, 0],
            )
            axes[1, 0].set_title("Hit Rate@k")
            axes[1, 0].set_xlabel("k")
            axes[1, 0].set_ylabel("Hit Rate")

        # Plot Precision/Recall
        if "precision" in model_metrics.columns and "recall" in model_metrics.columns:
            for i, model in enumerate(model_metrics["model"].unique()):
                model_data = model_metrics[model_metrics["model"] == model]
                axes[1, 1].scatter(
                    model_data["recall"].values,
                    model_data["precision"].values,
                    label=model,
                    s=100,
                )

            axes[1, 1].set_title("Precision vs. Recall")
            axes[1, 1].set_xlabel("Recall")
            axes[1, 1].set_ylabel("Precision")
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig("benchmark_results.png")
        logging.info(f"Visualizations saved to benchmark_results.png")

        return True
    except Exception as e:
        logging.error(f"Error visualizing results: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Set up multiple embedding models and benchmark them"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="List of model names to set up and benchmark",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip the setup phase and only run benchmarks",
    )
    parser.add_argument(
        "--test-file", type=str, help="Custom test file to use for benchmarking"
    )

    args = parser.parse_args()

    # Create directory for test cases if it doesn't exist
    os.makedirs("data/test_cases", exist_ok=True)

    # Set up models
    if not args.skip_setup:
        for model in tqdm(args.models, desc="Setting up models"):
            setup_model(model)

    # Run benchmarks
    success, output = benchmark_models(args.models, args.test_file)
    if success:
        # Visualize results
        visualize_results()

    logging.info("Setup and benchmarking complete")


if __name__ == "__main__":
    main()
