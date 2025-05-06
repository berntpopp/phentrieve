#!/usr/bin/env python3
"""
HPO Benchmark Evaluation Script

This script evaluates the performance of the HPO RAG system using a set of
test cases and multiple evaluation metrics.

Metrics include:
- Mean Reciprocal Rank (MRR)
- Hit Rate at K (HR@K)
- Ontology-based semantic similarity
"""

import argparse
import logging
import os
import json
import time
import re
from typing import Dict, List, Tuple, Any

import torch
from tqdm import tqdm

from multilingual_hpo_rag.config import (
    MIN_SIMILARITY_THRESHOLD,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    DEFAULT_ENABLE_RERANKER,
    DEFAULT_BIOLORD_MODEL,
    DEFAULT_MODEL,
    BENCHMARK_MODELS,
    TEST_CASES_DIR,
    RESULTS_DIR,
    SUMMARIES_DIR,
    DETAILED_DIR,
)
from multilingual_hpo_rag.data_processing.test_data_loader import (
    create_sample_test_data,
)
from multilingual_hpo_rag.evaluation.runner import run_evaluation, compare_models


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
    for directory in [RESULTS_DIR, SUMMARIES_DIR, DETAILED_DIR]:
        if not os.path.exists(directory):
            logging.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)


def main() -> None:
    """Main function for running benchmark evaluations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark tool for evaluating HPO RAG retrieval performance"
    )

    # Test data options
    parser.add_argument(
        "--test-file",
        type=str,
        default=os.path.join(TEST_CASES_DIR, "sample_test_cases.json"),
        help="Path to test cases JSON file (default: sample_test_cases.json)",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample test dataset if none exists",
    )

    # Model options
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_BIOLORD_MODEL,
        help=f"Sentence transformer model name (default: {DEFAULT_BIOLORD_MODEL})",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        help="Comma-separated list of models to benchmark",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run benchmarks for all models defined in BENCHMARK_MODELS",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for model loading (required for some models)",
    )

    # Benchmark options
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.1,
        help="Minimum similarity threshold (default: 0.1)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-test-case results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "detailed", "benchmark_results.csv"),
        help="Output CSV file for detailed results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Cross-encoder re-ranking arguments
    reranker_group = parser.add_argument_group("Re-ranking options")
    reranker_group.add_argument(
        "--enable-reranker",
        action="store_true",
        help=f"Enable cross-encoder re-ranking of results (default: {DEFAULT_ENABLE_RERANKER})",
    )
    reranker_group.add_argument(
        "--reranker-model",
        type=str,
        default=DEFAULT_RERANKER_MODEL,
        help=f"Cross-encoder model to use for re-ranking (default: {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--rerank-count",
        type=int,
        default=DEFAULT_RERANK_CANDIDATE_COUNT,
        help=f"Number of candidates to re-rank (default: {DEFAULT_RERANK_CANDIDATE_COUNT})",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    # Ensure output directories exist
    ensure_directories_exist()

    # Create sample test data if requested or if test file doesn't exist
    if args.create_sample or not os.path.exists(args.test_file):
        logging.info(f"Creating sample test data: {args.test_file}")
        create_sample_test_data(args.test_file)

    # Determine which models to run
    models_to_run: List[str] = []
    if args.all_models:
        models_to_run = BENCHMARK_MODELS
        logging.info(f"Running benchmarks for all {len(models_to_run)} models")
    elif args.model_list:
        models_to_run = [model.strip() for model in args.model_list.split(",")]
        logging.info(
            f"Running benchmark on {len(models_to_run)} models: {', '.join(models_to_run)}"
        )
    else:
        models_to_run = [args.model_name]
        logging.info(f"Running benchmark on single model: {args.model_name}")

    # Determine device
    device = "cpu" if args.cpu else None

    # Run benchmark for each model
    results_list = []
    for model_name in models_to_run:
        logging.info(f"Benchmarking model: {model_name}")

        results = run_evaluation(
            model_name=model_name,
            test_file=args.test_file,
            similarity_threshold=args.similarity_threshold,
            debug=args.debug,
            device=device,
            trust_remote_code=args.trust_remote_code,
            save_results=True,
            enable_reranker=args.enable_reranker,
            reranker_model=args.reranker_model,
            rerank_count=args.rerank_count,
        )

        if results:
            results_list.append(results)

    # Generate comparison if we have results
    if len(results_list) > 0:
        comparison_df = compare_models(results_list)

        # Display results
        print("\n===== Benchmark Results =====")
        print(f"Models evaluated: {len(results_list)}")
        print(f"Test file: {args.test_file}")
        print(f"Similarity threshold: {args.similarity_threshold}")
        print("\nModel Comparison:")

        # Format and display results in a clean table format
        pd.options.display.float_format = "{:.4f}".format
        print(comparison_df)

        # Save comparison to CSV
        csv_path = os.path.join(RESULTS_DIR, "benchmark_comparison.csv")
        comparison_df.to_csv(csv_path)
        print(f"\nComparison table saved to {csv_path}")
    else:
        print("No benchmark results collected.")


if __name__ == "__main__":
    # Import pandas here to avoid unnecessarily importing it when not needed
    import pandas as pd

    main()
