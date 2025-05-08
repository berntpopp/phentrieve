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
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add the parent directory to sys.path to make the package importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# For direct script execution, we also need to add the package directory itself
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, package_dir)

# For debugging import path issues
if "--debug" in sys.argv:
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print(f"Package dir: {package_dir}")

from phentrieve.config import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_RERANKER_MODE,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    DEFAULT_ENABLE_RERANKER,
    DEFAULT_BIOLORD_MODEL,
    BENCHMARK_MODELS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_TEST_CASES_SUBDIR,
    DEFAULT_SUMMARIES_SUBDIR,
    DEFAULT_DETAILED_SUBDIR,
)
from phentrieve.utils import get_default_data_dir
from phentrieve.evaluation.runner import run_evaluation, compare_models


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
    data_dir = get_default_data_dir()
    results_dir = data_dir / "results"
    summaries_dir = results_dir / DEFAULT_SUMMARIES_SUBDIR
    detailed_dir = results_dir / DEFAULT_DETAILED_SUBDIR
    for directory in [results_dir, summaries_dir, detailed_dir]:
        if not directory.exists():
            logging.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Main function for running benchmark evaluations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark tool for evaluating HPO RAG retrieval performance"
    )

    # Test data options
    data_dir = get_default_data_dir()
    test_cases_dir = data_dir / DEFAULT_TEST_CASES_SUBDIR
    parser.add_argument(
        "--test-file",
        type=str,
        default=str(test_cases_dir / "sample_test_cases.json"),
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
    reranker_group.add_argument(
        "--monolingual-reranker-model",
        type=str,
        default=DEFAULT_MONOLINGUAL_RERANKER_MODEL,
        help=f"German cross-encoder model for monolingual re-ranking (default: {DEFAULT_MONOLINGUAL_RERANKER_MODEL})",
    )
    reranker_group.add_argument(
        "--rerank-mode",
        type=str,
        choices=["cross-lingual", "monolingual"],
        default=DEFAULT_RERANKER_MODE,
        help=f"Re-ranking mode: cross-lingual (German->English) or monolingual (German->German) (default: {DEFAULT_RERANKER_MODE})",
    )
    data_dir = get_default_data_dir()
    translations_dir = data_dir / DEFAULT_TRANSLATIONS_SUBDIR
    reranker_group.add_argument(
        "--translation-dir",
        type=str,
        default=str(translations_dir),
        help=f"Directory containing German translations of HPO terms (default: {translations_dir})",
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
        # Importing here to avoid circular imports
        from phentrieve.data_processing.test_data_loader import create_sample_test_data

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
            reranker_model=(
                args.reranker_model
                if args.rerank_mode == "cross-lingual"
                else args.monolingual_reranker_model
            ),
            rerank_count=args.rerank_count,
            reranker_mode=args.rerank_mode,
            translation_dir=args.translation_dir,
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
        data_dir = get_default_data_dir()
        results_dir = data_dir / "results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = str(results_dir / f"benchmark_comparison_{timestamp}.csv")
        comparison_df.to_csv(csv_path)
        print(f"\nComparison table saved to {csv_path}")
    else:
        print("No benchmark results collected.")


if __name__ == "__main__":
    # Import pandas here to avoid unnecessarily importing it when not needed
    import pandas as pd

    main()
