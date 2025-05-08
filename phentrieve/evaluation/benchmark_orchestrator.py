"""
Benchmark orchestration module for running HPO retrieval evaluations from the CLI.

This module provides a high-level interface for running benchmark evaluations
for HPO term retrieval, supporting both single-model and multi-model evaluations.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple, Union

from phentrieve.config import (
    DEFAULT_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_RERANKER_MODE,
    DEFAULT_TRANSLATION_DIR,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    DEFAULT_ENABLE_RERANKER,
    BENCHMARK_MODELS,
    TEST_CASES_DIR,
    RESULTS_DIR,
    SUMMARIES_DIR,
    DETAILED_DIR,
)
from phentrieve.evaluation.runner import run_evaluation, compare_models
from phentrieve.data_processing.test_data_loader import create_sample_test_data

# Set up logging
logger = logging.getLogger(__name__)


def ensure_directories_exist() -> None:
    """Create necessary output directories if they don't exist."""
    for directory in [RESULTS_DIR, SUMMARIES_DIR, DETAILED_DIR]:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)


def orchestrate_benchmark(
    test_file: str = None,
    model_name: str = DEFAULT_MODEL,
    model_list: str = None,
    all_models: bool = False,
    similarity_threshold: float = 0.1,
    cpu: bool = False,
    detailed: bool = False,
    output: str = None,
    debug: bool = False,
    create_sample: bool = False,
    trust_remote_code: bool = False,
    enable_reranker: bool = DEFAULT_ENABLE_RERANKER,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    monolingual_reranker_model: str = DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    rerank_mode: str = DEFAULT_RERANKER_MODE,
    translation_dir: str = DEFAULT_TRANSLATION_DIR,
    rerank_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """
    Run benchmark evaluations for HPO term retrieval.

    Args:
        test_file: Path to test cases JSON file
        model_name: Name of the embedding model to evaluate
        model_list: Comma-separated list of models to benchmark
        all_models: Run benchmarks for all models defined in BENCHMARK_MODELS
        similarity_threshold: Minimum similarity threshold for results filtering
        cpu: Force CPU usage even if GPU is available
        detailed: Show detailed per-test-case results
        output: Output CSV file for detailed results
        debug: Whether to enable debug logging
        create_sample: Create a sample test dataset if none exists
        trust_remote_code: Whether to trust remote code when loading the model
        enable_reranker: Whether to enable cross-encoder re-ranking
        reranker_model: Model name for the cross-encoder for cross-lingual reranking
        monolingual_reranker_model: Model name for the cross-encoder for monolingual reranking
        rerank_mode: Re-ranking mode ('cross-lingual' or 'monolingual')
        translation_dir: Directory containing German translations of HPO terms
        rerank_count: Number of candidates to re-rank

    Returns:
        Dictionary or list of dictionaries containing benchmark results, or None if evaluation failed
    """
    # Set up logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Ensure output directories exist
    ensure_directories_exist()

    # Set default test file if not provided
    if not test_file:
        test_file = os.path.join(TEST_CASES_DIR, "sample_test_cases.json")

    # Create sample test data if requested or if test file doesn't exist
    if create_sample or not os.path.exists(test_file):
        logger.info(f"Creating sample test data: {test_file}")
        create_sample_test_data(test_file)

    # Determine device
    device = "cpu" if cpu else None

    # Determine which models to run
    models_to_run: List[str] = []
    if all_models:
        models_to_run = BENCHMARK_MODELS
        logger.info(f"Running benchmarks for all {len(models_to_run)} models")
    elif model_list:
        models_to_run = [model.strip() for model in model_list.split(",")]
        logger.info(
            f"Running benchmark on {len(models_to_run)} models: {', '.join(models_to_run)}"
        )
    else:
        models_to_run = [model_name]
        logger.info(f"Running benchmark on single model: {model_name}")

    # Run benchmark for each model
    results_list = []
    for model_name in models_to_run:
        logger.info(f"Benchmarking model: {model_name}")

        # Select appropriate reranker model based on mode
        active_reranker_model = (
            reranker_model
            if rerank_mode == "cross-lingual"
            else monolingual_reranker_model
        )

        results = run_evaluation(
            model_name=model_name,
            test_file=test_file,
            similarity_threshold=similarity_threshold,
            debug=debug,
            device=device,
            trust_remote_code=trust_remote_code,
            save_results=True,
            enable_reranker=enable_reranker,
            reranker_model=active_reranker_model,
            rerank_count=rerank_count,
            reranker_mode=rerank_mode,
            translation_dir=translation_dir,
        )

        if results:
            results_list.append(results)

    # Generate comparison if we have multiple results
    if len(results_list) > 1:
        comparison_df = compare_models(results_list)

        # Display and save comparison results
        logger.info("\n===== Benchmark Comparison =====")
        logger.info(f"Models evaluated: {len(results_list)}")
        logger.info(f"Test file: {test_file}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df}")

        # Save comparison to CSV if file path provided
        if output:
            output_path = output
        else:
            import pandas as pd
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                RESULTS_DIR, f"benchmark_comparison_{timestamp}.csv"
            )

        # Save comparison to CSV
        comparison_df.to_csv(output_path)
        logger.info(f"\nComparison table saved to {output_path}")

        return results_list
    elif len(results_list) == 1:
        # Return single model results
        return results_list[0]
    else:
        logger.warning("No benchmark results collected.")
        return None
