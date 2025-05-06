#!/usr/bin/env python3
"""
Compare Dense vs Re-ranked Results Visualization

This script creates a bar chart to compare dense retrieval vs re-ranked metrics
from a benchmark results summary file.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def visualize_dense_vs_reranked(summary_file, output_dir=None):
    """
    Create visualizations comparing dense retrieval and re-ranked metrics.

    Args:
        summary_file: Path to summary JSON file
        output_dir: Directory to save visualization images
    """
    # Load the summary file
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Check if re-ranking was enabled
    if not summary.get("reranker_enabled", False):
        print("Re-ranking was not enabled in this benchmark run.")
        return

    # Set up output directory
    if output_dir is None:
        benchmark_dir = os.path.join(
            os.path.dirname(os.path.abspath(summary_file)), ".."
        )
        output_dir = os.path.join(benchmark_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)

    # Extract model info
    model_name = summary.get("original_model_name", summary.get("model", "Unknown"))
    reranker_model = summary.get("reranker_model", "Unknown")
    reranker_mode = summary.get("reranker_mode", "Unknown")
    timestamp = summary.get("timestamp", "unknown_time")

    # Create metrics comparison plots

    # 1. MRR comparison
    plt.figure(figsize=(10, 6))
    mrr_dense = summary.get("mrr_dense", 0)
    mrr_reranked = summary.get("mrr_reranked", 0)

    metrics = ["MRR"]
    dense_values = [mrr_dense]
    reranked_values = [mrr_reranked]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, dense_values, width, label="Dense Retrieval")
    rects2 = ax.bar(
        x + width / 2, reranked_values, width, label=f"Re-ranked ({reranker_mode})"
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Score")
    ax.set_title(f"MRR: Dense vs Re-ranked Comparison\n{model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, f"mrr_dense_vs_reranked_{timestamp}.png")
    plt.savefig(output_file)
    print(f"MRR comparison saved to: {output_file}")

    # 2. Hit Rate comparison
    plt.figure(figsize=(12, 7))

    # Extract Hit Rate metrics
    k_values = [1, 3, 5, 10]
    hr_dense_values = [summary.get(f"hit_rate_dense@{k}", 0) for k in k_values]
    hr_reranked_values = [summary.get(f"hit_rate_reranked@{k}", 0) for k in k_values]

    metrics = [f"Hit@{k}" for k in k_values]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width / 2, hr_dense_values, width, label="Dense Retrieval")
    rects2 = ax.bar(
        x + width / 2, hr_reranked_values, width, label=f"Re-ranked ({reranker_mode})"
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Hit Rate")
    ax.set_title(f"Hit Rate: Dense vs Re-ranked Comparison\n{model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend()

    # Add delta values
    for i, (dense, reranked) in enumerate(zip(hr_dense_values, hr_reranked_values)):
        delta = reranked - dense
        color = "green" if delta >= 0 else "red"
        ax.annotate(
            f"{delta:+.4f}",
            xy=(x[i], max(dense, reranked) + 0.02),
            ha="center",
            va="bottom",
            color=color,
            fontweight="bold",
        )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(
        output_dir, f"hit_rate_dense_vs_reranked_{timestamp}.png"
    )
    plt.savefig(output_file)
    print(f"Hit Rate comparison saved to: {output_file}")

    # 3. Ontology Similarity comparison
    plt.figure(figsize=(12, 7))

    # Extract Ontology Similarity metrics
    ont_dense_values = [summary.get(f"ont_similarity_dense@{k}", 0) for k in k_values]
    ont_reranked_values = [
        summary.get(f"ont_similarity_reranked@{k}", 0) for k in k_values
    ]

    metrics = [f"OntSim@{k}" for k in k_values]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width / 2, ont_dense_values, width, label="Dense Retrieval")
    rects2 = ax.bar(
        x + width / 2, ont_reranked_values, width, label=f"Re-ranked ({reranker_mode})"
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Ontology Similarity")
    ax.set_title(f"Ontology Similarity: Dense vs Re-ranked Comparison\n{model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.85, 1.0)  # Adjusted to focus on the range where most values are
    ax.legend()

    # Add delta values
    for i, (dense, reranked) in enumerate(zip(ont_dense_values, ont_reranked_values)):
        delta = reranked - dense
        color = "green" if delta >= 0 else "red"
        ax.annotate(
            f"{delta:+.4f}",
            xy=(x[i], max(dense, reranked) + 0.005),
            ha="center",
            va="bottom",
            color=color,
            fontweight="bold",
        )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(
        output_dir, f"ont_similarity_dense_vs_reranked_{timestamp}.png"
    )
    plt.savefig(output_file)
    print(f"Ontology Similarity comparison saved to: {output_file}")

    # Print a summary of improvements
    print("\nPerformance Summary:")
    print(
        f"MRR: Dense={mrr_dense:.4f}, Re-ranked={mrr_reranked:.4f}, Diff={mrr_reranked-mrr_dense:+.4f}"
    )

    for k in k_values:
        dense_hr = summary.get(f"hit_rate_dense@{k}", 0)
        reranked_hr = summary.get(f"hit_rate_reranked@{k}", 0)
        print(
            f"Hit@{k}: Dense={dense_hr:.4f}, Re-ranked={reranked_hr:.4f}, Diff={reranked_hr-dense_hr:+.4f}"
        )

    for k in k_values:
        dense_ont = summary.get(f"ont_similarity_dense@{k}", 0)
        reranked_ont = summary.get(f"ont_similarity_reranked@{k}", 0)
        print(
            f"OntSim@{k}: Dense={dense_ont:.4f}, Re-ranked={reranked_ont:.4f}, Diff={reranked_ont-dense_ont:+.4f}"
        )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Visualize comparison between dense retrieval and re-ranked metrics"
    )
    parser.add_argument(
        "summary_file",
        type=str,
        help="Path to the benchmark summary JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualization images (default: benchmark_results/visualizations)",
    )

    args = parser.parse_args()

    # Run visualization
    visualize_dense_vs_reranked(args.summary_file, args.output_dir)


if __name__ == "__main__":
    main()
