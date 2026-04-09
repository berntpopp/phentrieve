"""
Benchmark runner for evaluating HPO retrieval performance.

This module provides functionality for running comprehensive benchmark
evaluations of the HPO retrieval system using various metrics.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from phentrieve.config import (
    DEFAULT_AGGREGATION_STRATEGY,
    DEFAULT_K_VALUES,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from phentrieve.data_processing.test_data_loader import load_test_data
from phentrieve.embeddings import load_embedding_model
from phentrieve.evaluation.metrics import (
    SimilarityFormula,
    average_precision_at_k,
    calculate_test_case_max_ont_sim,
    hit_rate_at_k,
    load_hpo_graph_data,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from phentrieve.evaluation.statistics import (
    calculate_bootstrap_ci_for_metrics,
    compare_models_with_significance,
)
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.utils import convert_multi_vector_to_chromadb_format
from phentrieve.utils import get_model_slug

# Alias for backward compatibility within this module
_convert_multi_vector_to_chromadb_format = convert_multi_vector_to_chromadb_format


def run_evaluation(
    model_name: str,
    test_file: str,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    debug: bool = False,
    device: str | None = None,
    trust_remote_code: bool = False,
    save_results: bool = True,
    results_dir: Path | None = None,
    index_dir: Path | None = None,
    similarity_formula: str = "hybrid",
    # Multi-vector parameters (Issue #136)
    multi_vector: bool = False,
    aggregation_strategy: str = DEFAULT_AGGREGATION_STRATEGY,
) -> dict[str, Any] | None:
    """
    Run a complete benchmark evaluation for a model on a test dataset.

    Args:
        model_name: Name of the embedding model to evaluate
        test_file: Path to the test cases JSON file
        k_values: Tuple of k values for Hit@K metrics
        similarity_threshold: Minimum similarity threshold for results filtering
        debug: Whether to enable debug logging
        device: Device to use for model inference (None for auto-detection)
        trust_remote_code: Whether to trust remote code when loading the model
        save_results: Whether to save results to files
        results_dir: Directory to save results (default: configured RESULTS_DIR)
        index_dir: Directory containing the index files
        similarity_formula: Which similarity formula to use for ontology similarity calculations
        multi_vector: Use multi-vector index with component-level aggregation (Issue #136)
        aggregation_strategy: Strategy for aggregating component scores in multi-vector mode

    Returns:
        Dictionary containing benchmark results or None if evaluation failed
    """
    # Create output directory structure
    detailed_results_dir = None
    summaries_dir = None
    if save_results:
        if results_dir is None:
            # Log error if results_dir is None
            logging.error("results_dir must be provided when save_results is True")
            raise ValueError("results_dir must be provided when save_results is True")

        # Create results directories if they don't exist
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)

        detailed_results_dir = results_dir / "detailed"
        summaries_dir = results_dir / "summaries"
        os.makedirs(detailed_results_dir, exist_ok=True)
        os.makedirs(summaries_dir, exist_ok=True)

    # Load test data
    test_cases = load_test_data(test_file)
    if not test_cases:
        logging.error(f"Failed to load test data from {test_file}")
        return None

    # Create a descriptive name for the benchmark run
    model_slug = get_model_slug(model_name)
    datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.info(f"Starting benchmark evaluation for model '{model_name}'")
    logging.info(f"Test file: {test_file} ({len(test_cases)} test cases)")

    try:
        # Load embedding model
        model = load_embedding_model(
            model_name=model_name, trust_remote_code=trust_remote_code, device=device
        )

        # Load the HPO graph data for similarity metrics
        logging.info("Loading HPO graph data for semantic similarity calculation")
        load_hpo_graph_data()

        # Convert similarity formula string to enum
        formula = SimilarityFormula.from_string(similarity_formula)

        # Initialize retrieval system
        if index_dir is None and not os.path.exists("data"):
            raise ValueError(
                "No index directory provided and default 'data' directory not found."
                " Please provide a valid index_dir."
            )

        (
            index_dir / f"{model_name}.faiss"
            if index_dir
            else Path("data") / f"{model_name}.faiss"
        )
        retriever = DenseRetriever.from_model_name(
            model=model,
            model_name=model_name,
            index_dir=index_dir,
            min_similarity=similarity_threshold,
            multi_vector=multi_vector,
        )

        if retriever is None:
            logging.error("Failed to initialize retriever. Is the index built?")
            return None

        # Initialize result containers for both dense and re-ranked metrics
        # Baseline dense metrics
        mrr_dense_values = []
        hit_rate_dense_values: dict[int, list[float]] = {k: [] for k in k_values}
        max_ont_sim_dense_values: dict[int, list[float]] = {k: [] for k in k_values}
        ndcg_dense_values: dict[int, list[float]] = {k: [] for k in k_values}
        recall_dense_values: dict[int, list[float]] = {k: [] for k in k_values}
        precision_dense_values: dict[int, list[float]] = {k: [] for k in k_values}
        map_dense_values: dict[int, list[float]] = {k: [] for k in k_values}

        detailed_results = []

        # Run benchmark for each test case
        logging.info(f"Running benchmark on {len(test_cases)} test cases")
        start_time = time.time()

        for i, test_case in enumerate(tqdm(test_cases, desc="Evaluating test cases")):
            # Extract test case data
            text = test_case["text"]
            expected_ids = test_case.get("expected_hpo_ids", [])
            description = test_case.get("description", f"Case {i + 1}")

            # Skip test cases with no expected IDs
            if not expected_ids:
                logging.warning(f"Skipping test case {i + 1} with no expected HPO IDs")
                continue

            try:
                # Query the index with bi-encoder
                n_query_results = max(k_values) * 3

                if multi_vector:
                    # Use multi-vector query with aggregation
                    multi_results = retriever.query_multi_vector(
                        text,
                        n_results=n_query_results,
                        aggregation_strategy=aggregation_strategy,
                    )
                    # Convert to ChromaDB format for metrics calculation
                    dense_results = _convert_multi_vector_to_chromadb_format(
                        multi_results
                    )
                else:
                    # Standard single-vector query
                    dense_results = retriever.query(
                        text,
                        n_results=n_query_results,
                    )

                #### BASELINE DENSE METRICS CALCULATION ####

                # Extract HPO IDs from the results
                dense_term_ids = []
                dense_metadatas = []

                if (
                    dense_results
                    and "metadatas" in dense_results
                    and dense_results["metadatas"]
                    and dense_results["metadatas"][0]
                ):
                    # Extract data from results
                    dense_metadatas = dense_results["metadatas"][0]
                    dense_term_ids = [
                        meta.get("hpo_id", "") for meta in dense_metadatas
                    ]

                # Calculate baseline MRR
                mrr_dense = mean_reciprocal_rank(dense_results, expected_ids)
                mrr_dense_values.append(mrr_dense)

                # Calculate baseline metrics for each k value
                dense_hit_rates = {}
                dense_max_ont_sims = {}
                dense_ndcgs = {}
                dense_recalls = {}
                dense_precisions = {}
                dense_maps = {}
                for k in k_values:
                    hit = hit_rate_at_k(dense_results, expected_ids, k=k)
                    hit_rate_dense_values[k].append(hit)
                    dense_hit_rates[f"hit_rate_dense@{k}"] = hit

                    # New metrics
                    ndcg = ndcg_at_k(dense_results, expected_ids, k=k)
                    ndcg_dense_values[k].append(ndcg)
                    dense_ndcgs[f"ndcg_dense@{k}"] = ndcg

                    recall = recall_at_k(dense_results, expected_ids, k=k)
                    recall_dense_values[k].append(recall)
                    dense_recalls[f"recall_dense@{k}"] = recall

                    precision = precision_at_k(dense_results, expected_ids, k=k)
                    precision_dense_values[k].append(precision)
                    dense_precisions[f"precision_dense@{k}"] = precision

                    map_score = average_precision_at_k(dense_results, expected_ids, k=k)
                    map_dense_values[k].append(map_score)
                    dense_maps[f"map_dense@{k}"] = map_score

                # Calculate baseline ontology similarity at different K values
                for k in k_values:
                    if dense_term_ids:
                        # Extract HPO IDs from results
                        retrieved_ids = dense_term_ids[:k]

                        # Calculate the maximum ontology similarity using the new function
                        # This gets the single highest similarity between any expected ID and any retrieved ID
                        max_ont_sim = calculate_test_case_max_ont_sim(
                            expected_ids, retrieved_ids, formula
                        )

                        # Store the values using the new variable names
                        max_ont_sim_dense_values[k].append(max_ont_sim)
                        dense_max_ont_sims[f"max_ont_similarity_dense@{k}"] = (
                            max_ont_sim
                        )
                    else:
                        max_ont_sim_dense_values[k].append(0.0)
                        dense_max_ont_sims[f"max_ont_similarity_dense@{k}"] = 0.0

                # Record detailed results for this test case
                case_result = {
                    "case_id": i,
                    "description": description,
                    "text": text,
                    "expected_ids": expected_ids,
                    "mrr_dense": mrr_dense,
                    **dense_hit_rates,
                    **dense_max_ont_sims,
                }

                detailed_results.append(case_result)

            except Exception as e:
                logging.error(f"Error processing test case {i + 1}: {e}")
                # Add a failed entry
                detailed_results.append(
                    {
                        "case_id": i,
                        "description": description,
                        "text": text,
                        "expected_ids": expected_ids,
                        "mrr": 0.0,
                        "error": str(e),
                    }
                )

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Calculate average metrics for both dense and re-ranked results
        # Baseline dense metrics
        avg_mrr_dense = (
            sum(mrr_dense_values) / len(mrr_dense_values) if mrr_dense_values else 0
        )
        avg_hit_rates_dense = {
            k: (
                sum(hit_rate_dense_values[k]) / len(hit_rate_dense_values[k])
                if hit_rate_dense_values[k]
                else 0
            )
            for k in k_values
        }
        avg_max_ont_sim_dense = {
            k: (
                sum(max_ont_sim_dense_values[k]) / len(max_ont_sim_dense_values[k])
                if max_ont_sim_dense_values[k]
                else 0
            )
            for k in k_values
        }
        avg_ndcg_dense = {
            k: (
                sum(ndcg_dense_values[k]) / len(ndcg_dense_values[k])
                if ndcg_dense_values[k]
                else 0
            )
            for k in k_values
        }
        avg_recall_dense = {
            k: (
                sum(recall_dense_values[k]) / len(recall_dense_values[k])
                if recall_dense_values[k]
                else 0
            )
            for k in k_values
        }
        avg_precision_dense = {
            k: (
                sum(precision_dense_values[k]) / len(precision_dense_values[k])
                if precision_dense_values[k]
                else 0
            )
            for k in k_values
        }
        avg_map_dense = {
            k: (
                sum(map_dense_values[k]) / len(map_dense_values[k])
                if map_dense_values[k]
                else 0
            )
            for k in k_values
        }

        # Compile final results with dense metrics
        results = {
            "model_name": model_name,
            "model_slug": model_slug,
            "test_file": test_file,
            "num_test_cases": len(test_cases),
            "similarity_threshold": similarity_threshold,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            # Multi-vector configuration (Issue #136)
            "multi_vector": multi_vector,
            "aggregation_strategy": aggregation_strategy if multi_vector else None,
            # Raw values - dense
            "mrr_dense": mrr_dense_values,
            # Average metrics - dense
            "avg_mrr_dense": avg_mrr_dense,
        }

        # Add Hit Rate metrics - dense
        for k in k_values:
            results[f"hit_rate_dense@{k}"] = hit_rate_dense_values[k]
            results[f"avg_hit_rate_dense@{k}"] = avg_hit_rates_dense[k]

        # Add maximum ontology similarity metrics - dense
        for k in k_values:
            results[f"max_ont_similarity_dense@{k}"] = max_ont_sim_dense_values[k]
            results[f"avg_max_ont_similarity_dense@{k}"] = avg_max_ont_sim_dense[k]

        # Add NDCG metrics - dense
        for k in k_values:
            results[f"ndcg_dense@{k}"] = ndcg_dense_values[k]
            results[f"avg_ndcg_dense@{k}"] = avg_ndcg_dense[k]

        # Add Recall metrics - dense
        for k in k_values:
            results[f"recall_dense@{k}"] = recall_dense_values[k]
            results[f"avg_recall_dense@{k}"] = avg_recall_dense[k]

        # Add Precision metrics - dense
        for k in k_values:
            results[f"precision_dense@{k}"] = precision_dense_values[k]
            results[f"avg_precision_dense@{k}"] = avg_precision_dense[k]

        # Add MAP metrics - dense
        for k in k_values:
            results[f"map_dense@{k}"] = map_dense_values[k]
            results[f"avg_map_dense@{k}"] = avg_map_dense[k]

        # Add detailed results
        results["detailed_results"] = detailed_results

        # Calculate bootstrap confidence intervals for all metrics
        logging.info("Calculating bootstrap confidence intervals...")
        confidence_intervals = calculate_bootstrap_ci_for_metrics(results, k_values)

        # Add confidence intervals to results
        results["confidence_intervals"] = confidence_intervals

        # Save results if requested
        if save_results:
            # Create summary dictionary with dense metrics
            summary: dict[str, Any] = {
                "model": model_slug,
                "original_model_name": model_name,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "test_file": os.path.basename(test_file),
                "num_test_cases": len(test_cases),
                # Multi-vector configuration (Issue #136)
                "multi_vector": multi_vector,
                "aggregation_strategy": aggregation_strategy if multi_vector else None,
                # MRR metric
                "mrr_dense": avg_mrr_dense,
                "mrr_dense_per_case": mrr_dense_values,
            }

            # Add all K-dependent metrics
            for k in k_values:
                # Hit rate
                summary[f"hit_rate_dense@{k}"] = avg_hit_rates_dense[k]
                summary[f"hit_rate_dense@{k}_per_case"] = hit_rate_dense_values[k]
                # Ontology similarity
                summary[f"max_ont_similarity_dense@{k}"] = avg_max_ont_sim_dense[k]
                summary[f"max_ont_similarity_dense@{k}_per_case"] = (
                    max_ont_sim_dense_values[k]
                )
                # NDCG
                summary[f"ndcg_dense@{k}"] = avg_ndcg_dense[k]
                summary[f"ndcg_dense@{k}_per_case"] = ndcg_dense_values[k]
                # Recall
                summary[f"recall_dense@{k}"] = avg_recall_dense[k]
                summary[f"recall_dense@{k}_per_case"] = recall_dense_values[k]
                # Precision
                summary[f"precision_dense@{k}"] = avg_precision_dense[k]
                summary[f"precision_dense@{k}_per_case"] = precision_dense_values[k]
                # MAP
                summary[f"map_dense@{k}"] = avg_map_dense[k]
                summary[f"map_dense@{k}_per_case"] = map_dense_values[k]

            # Add confidence intervals
            summary["confidence_intervals"] = confidence_intervals

            # Save summary to file
            # Type narrowing: these are not None inside save_results block
            assert summaries_dir is not None
            assert detailed_results_dir is not None

            os.makedirs(summaries_dir, exist_ok=True)
            # Use model_slug instead of creating a new safe name
            # This ensures consistency with collection naming
            # Add suffix for multi-vector to distinguish from single-vector results
            file_suffix = "_multi" if multi_vector else ""
            summary_path = summaries_dir / f"{model_slug}{file_suffix}_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            # Save detailed results as CSV
            os.makedirs(detailed_results_dir, exist_ok=True)
            detailed_df = pd.DataFrame(detailed_results)
            # Use model_slug for CSV files
            csv_path = detailed_results_dir / f"{model_slug}{file_suffix}_detailed.csv"
            detailed_df.to_csv(csv_path, index=False)

        # Log summary of results
        logging.info(f"Benchmark results for {model_name}:")
        logging.info("  === Dense Retrieval Metrics ====")
        logging.info(f"  MRR (Dense): {avg_mrr_dense:.4f}")
        if "mrr_dense" in confidence_intervals:
            ci = confidence_intervals["mrr_dense"]
            logging.info(f"    95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        for k in k_values:
            logging.info(f"  Hit@{k} (Dense): {avg_hit_rates_dense[k]:.4f}")
            if f"hit_rate_dense@{k}" in confidence_intervals:
                ci = confidence_intervals[f"hit_rate_dense@{k}"]
                logging.info(
                    f"    95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
                )
            logging.info(f"  NDCG@{k} (Dense): {avg_ndcg_dense[k]:.4f}")
            if f"ndcg_dense@{k}" in confidence_intervals:
                ci = confidence_intervals[f"ndcg_dense@{k}"]
                logging.info(
                    f"    95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
                )
            logging.info(f"  Recall@{k} (Dense): {avg_recall_dense[k]:.4f}")
            if f"recall_dense@{k}" in confidence_intervals:
                ci = confidence_intervals[f"recall_dense@{k}"]
                logging.info(
                    f"    95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
                )
            logging.info(f"  Precision@{k} (Dense): {avg_precision_dense[k]:.4f}")
            if f"precision_dense@{k}" in confidence_intervals:
                ci = confidence_intervals[f"precision_dense@{k}"]
                logging.info(
                    f"    95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
                )
            logging.info(f"  MAP@{k} (Dense): {avg_map_dense[k]:.4f}")
            if f"map_dense@{k}" in confidence_intervals:
                ci = confidence_intervals[f"map_dense@{k}"]
                logging.info(
                    f"    95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
                )
            logging.info(f"  MaxOntSim@{k} (Dense): {avg_max_ont_sim_dense[k]:.4f}")
            if f"max_ont_similarity_dense@{k}" in confidence_intervals:
                ci = confidence_intervals[f"max_ont_similarity_dense@{k}"]
                logging.info(
                    f"    95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
                )

        return results

    except Exception as e:
        logging.error(f"Error during benchmark evaluation: {e}")
        return None


def compare_models(results_list: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Compare results across different models.

    Args:
        results_list: List of benchmark result dictionaries

    Returns:
        DataFrame: Comparison table
    """
    if not results_list:
        return pd.DataFrame()

    # Extract comparison data
    comparison_data = []

    for result in results_list:
        model_data = {
            "Model": result["model_slug"],
        }

        # Add Dense Retrieval metrics
        if "mrr_dense" in result:
            # Check if the value is a list and get the average if so
            if isinstance(result["mrr_dense"], list):
                model_data["MRR (Dense)"] = (
                    sum(result["mrr_dense"]) / len(result["mrr_dense"])
                    if result["mrr_dense"]
                    else 0
                )
            else:
                model_data["MRR (Dense)"] = result["mrr_dense"]

        # Legacy: handle old results that may contain reranked data
        if "mrr_reranked" in result:
            if result.get("reranker_enabled", False) and result.get("mrr_reranked"):
                dense_mrr = (
                    sum(result["mrr_dense"]) / len(result["mrr_dense"])
                    if isinstance(result["mrr_dense"], list) and result["mrr_dense"]
                    else result["mrr_dense"]
                )
                if True:  # keep indentation level
                    reranked_mrr = (
                        sum(result["mrr_reranked"]) / len(result["mrr_reranked"])
                        if isinstance(result["mrr_reranked"], list)
                        and result["mrr_reranked"]
                        else result["mrr_reranked"]
                    )
                    if isinstance(reranked_mrr, (int, float)):
                        model_data["MRR (Diff)"] = reranked_mrr - dense_mrr

        # Add Hit Rate metrics
        for k in [1, 3, 5, 10]:
            # Dense retrieval metrics
            dense_key = f"hit_rate_dense@{k}"
            if dense_key in result:
                if isinstance(result[dense_key], list):
                    model_data[f"HR@{k} (Dense)"] = (
                        sum(result[dense_key]) / len(result[dense_key])
                        if result[dense_key]
                        else 0
                    )
                else:
                    model_data[f"HR@{k} (Dense)"] = result[dense_key]

            # Re-ranked metrics if available
            reranked_key = f"hit_rate_reranked@{k}"
            if reranked_key in result:
                if isinstance(result[reranked_key], list):
                    model_data[f"HR@{k} (ReRanked)"] = (
                        sum(result[reranked_key]) / len(result[reranked_key])
                        if result[reranked_key]
                        else 0
                    )
                else:
                    model_data[f"HR@{k} (ReRanked)"] = result[reranked_key]

                # Calculate difference if both metrics are available
                if dense_key in result:
                    dense_val = (
                        sum(result[dense_key]) / len(result[dense_key])
                        if isinstance(result[dense_key], list) and result[dense_key]
                        else result[dense_key]
                    )
                    reranked_val = (
                        sum(result[reranked_key]) / len(result[reranked_key])
                        if isinstance(result[reranked_key], list)
                        and result[reranked_key]
                        else result[reranked_key]
                    )
                    model_data[f"HR@{k} (Diff)"] = reranked_val - dense_val

        # Add Maximum Ontology Similarity metrics
        for k in [1, 3, 5, 10]:
            # Dense retrieval metrics
            dense_key = f"max_ont_similarity_dense@{k}"
            if dense_key in result:
                if isinstance(result[dense_key], list):
                    model_data[f"MaxOntSim@{k} (Dense)"] = (
                        sum(result[dense_key]) / len(result[dense_key])
                        if result[dense_key]
                        else 0
                    )
                else:
                    model_data[f"MaxOntSim@{k} (Dense)"] = result[dense_key]

            # Re-ranked metrics if available
            reranked_key = f"max_ont_similarity_reranked@{k}"
            if reranked_key in result:
                if isinstance(result[reranked_key], list):
                    model_data[f"MaxOntSim@{k} (ReRanked)"] = (
                        sum(result[reranked_key]) / len(result[reranked_key])
                        if result[reranked_key]
                        else 0
                    )
                else:
                    model_data[f"MaxOntSim@{k} (ReRanked)"] = result[reranked_key]

                # Calculate difference if both metrics are available
                if dense_key in result:
                    dense_val = (
                        sum(result[dense_key]) / len(result[dense_key])
                        if isinstance(result[dense_key], list) and result[dense_key]
                        else result[dense_key]
                    )
                    reranked_val = (
                        sum(result[reranked_key]) / len(result[reranked_key])
                        if isinstance(result[reranked_key], list)
                        and result[reranked_key]
                        else result[reranked_key]
                    )
                    model_data[f"OntSim@{k} (Diff)"] = reranked_val - dense_val

        # Add NDCG metrics
        for k in [1, 3, 5, 10]:
            # Dense retrieval metrics
            dense_key = f"ndcg_dense@{k}"
            if dense_key in result:
                if isinstance(result[dense_key], list):
                    model_data[f"NDCG@{k} (Dense)"] = (
                        sum(result[dense_key]) / len(result[dense_key])
                        if result[dense_key]
                        else 0
                    )
                else:
                    model_data[f"NDCG@{k} (Dense)"] = result[dense_key]

            # Re-ranked metrics if available
            reranked_key = f"ndcg_reranked@{k}"
            if reranked_key in result:
                if isinstance(result[reranked_key], list):
                    model_data[f"NDCG@{k} (ReRanked)"] = (
                        sum(result[reranked_key]) / len(result[reranked_key])
                        if result[reranked_key]
                        else 0
                    )
                else:
                    model_data[f"NDCG@{k} (ReRanked)"] = result[reranked_key]

        # Add Recall metrics
        for k in [1, 3, 5, 10]:
            # Dense retrieval metrics
            dense_key = f"recall_dense@{k}"
            if dense_key in result:
                if isinstance(result[dense_key], list):
                    model_data[f"Recall@{k} (Dense)"] = (
                        sum(result[dense_key]) / len(result[dense_key])
                        if result[dense_key]
                        else 0
                    )
                else:
                    model_data[f"Recall@{k} (Dense)"] = result[dense_key]

            # Re-ranked metrics if available
            reranked_key = f"recall_reranked@{k}"
            if reranked_key in result:
                if isinstance(result[reranked_key], list):
                    model_data[f"Recall@{k} (ReRanked)"] = (
                        sum(result[reranked_key]) / len(result[reranked_key])
                        if result[reranked_key]
                        else 0
                    )
                else:
                    model_data[f"Recall@{k} (ReRanked)"] = result[reranked_key]

        # Add Precision metrics
        for k in [1, 3, 5, 10]:
            # Dense retrieval metrics
            dense_key = f"precision_dense@{k}"
            if dense_key in result:
                if isinstance(result[dense_key], list):
                    model_data[f"Precision@{k} (Dense)"] = (
                        sum(result[dense_key]) / len(result[dense_key])
                        if result[dense_key]
                        else 0
                    )
                else:
                    model_data[f"Precision@{k} (Dense)"] = result[dense_key]

            # Re-ranked metrics if available
            reranked_key = f"precision_reranked@{k}"
            if reranked_key in result:
                if isinstance(result[reranked_key], list):
                    model_data[f"Precision@{k} (ReRanked)"] = (
                        sum(result[reranked_key]) / len(result[reranked_key])
                        if result[reranked_key]
                        else 0
                    )
                else:
                    model_data[f"Precision@{k} (ReRanked)"] = result[reranked_key]

        # Add MAP metrics
        for k in [1, 3, 5, 10]:
            # Dense retrieval metrics
            dense_key = f"map_dense@{k}"
            if dense_key in result:
                if isinstance(result[dense_key], list):
                    model_data[f"MAP@{k} (Dense)"] = (
                        sum(result[dense_key]) / len(result[dense_key])
                        if result[dense_key]
                        else 0
                    )
                else:
                    model_data[f"MAP@{k} (Dense)"] = result[dense_key]

            # Re-ranked metrics if available
            reranked_key = f"map_reranked@{k}"
            if reranked_key in result:
                if isinstance(result[reranked_key], list):
                    model_data[f"MAP@{k} (ReRanked)"] = (
                        sum(result[reranked_key]) / len(result[reranked_key])
                        if result[reranked_key]
                        else 0
                    )
                else:
                    model_data[f"MAP@{k} (ReRanked)"] = result[reranked_key]

        comparison_data.append(model_data)

    # Create and return DataFrame
    df = pd.DataFrame(comparison_data)

    # Sort by MRR (descending)
    if "MRR (Dense)" in df.columns:
        df = df.sort_values("MRR (Dense)", ascending=False)

    return df


def compare_models_with_statistics(
    results_list: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Compare results across different models with statistical significance testing.

    Args:
        results_list: List of benchmark result dictionaries

    Returns:
        Tuple of (comparison_dataframe, significance_tests_list)
    """
    if len(results_list) < 2:
        return compare_models(results_list), []

    # Get the basic comparison table
    comparison_df = compare_models(results_list)

    # Perform pairwise significance tests
    significance_tests = []
    for i in range(len(results_list)):
        for j in range(i + 1, len(results_list)):
            model_a = results_list[i]
            model_b = results_list[j]
            model_a_name = model_a.get("model_slug", f"Model {i + 1}")
            model_b_name = model_b.get("model_slug", f"Model {j + 1}")

            # Perform significance testing
            significance = compare_models_with_significance(
                model_a, model_b, model_a_name, model_b_name
            )
            significance_tests.append(significance)

    return comparison_df, significance_tests
