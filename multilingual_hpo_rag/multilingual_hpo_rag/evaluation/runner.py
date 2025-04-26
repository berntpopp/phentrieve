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
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
from tqdm import tqdm

from multilingual_hpo_rag.config import (
    INDEX_DIR,
    RESULTS_DIR,
    SUMMARIES_DIR,
    DETAILED_DIR,
    DEFAULT_TOP_K,
)
from multilingual_hpo_rag.data_processing.test_data_loader import load_test_data
from multilingual_hpo_rag.embeddings import load_embedding_model
from multilingual_hpo_rag.evaluation.metrics import (
    average_max_similarity,
    hit_rate_at_k,
    load_hpo_graph_data,
    mean_reciprocal_rank,
)
from multilingual_hpo_rag.retrieval.dense_retriever import DenseRetriever
from multilingual_hpo_rag.utils import get_model_slug


def run_evaluation(
    model_name: str,
    test_file: str,
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
    similarity_threshold: float = 0.1,
    debug: bool = False,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    save_results: bool = True,
    output_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
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
        output_dir: Directory to save results (default: configured RESULTS_DIR)

    Returns:
        Dictionary containing benchmark results or None if evaluation failed
    """
    # Set up results directory
    if output_dir is None:
        output_dir = RESULTS_DIR

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "summaries"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "detailed"), exist_ok=True)

    # Load test data
    test_cases = load_test_data(test_file)
    if not test_cases:
        logging.error(f"Failed to load test data from {test_file}")
        return None

    # Create a descriptive name for the benchmark run
    model_slug = get_model_slug(model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_slug}_{timestamp}"

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

        # Create retriever
        retriever = DenseRetriever.from_model_name(
            model=model,
            model_name=model_name,
            index_dir=INDEX_DIR,
            min_similarity=similarity_threshold,
        )

        if retriever is None:
            logging.error("Failed to initialize retriever. Is the index built?")
            return None

        # Initialize result containers
        mrr_values = []
        hit_rate_values = {k: [] for k in k_values}
        ont_similarity_values = {k: [] for k in k_values}
        detailed_results = []

        # Run benchmark for each test case
        logging.info(f"Running benchmark on {len(test_cases)} test cases")
        start_time = time.time()

        for i, test_case in enumerate(tqdm(test_cases, desc="Evaluating test cases")):
            # Extract test case data
            text = test_case["text"]
            expected_ids = test_case.get("expected_hpo_ids", [])
            description = test_case.get("description", f"Case {i+1}")

            # Skip test cases with no expected IDs
            if not expected_ids:
                logging.warning(f"Skipping test case {i+1} with no expected HPO IDs")
                continue

            try:
                # Query the index
                results = retriever.query(text, n_results=max(k_values) * 3)

                # Calculate MRR
                mrr = mean_reciprocal_rank(results, expected_ids)
                mrr_values.append(mrr)

                # Calculate Hit Rate at different K values
                hit_rates = {}
                for k in k_values:
                    hit = hit_rate_at_k(results, expected_ids, k=k)
                    hit_rate_values[k].append(hit)
                    hit_rates[f"hit_rate@{k}"] = hit

                # Calculate ontology similarity at different K values
                ont_similarities = {}
                for k in k_values:
                    if results["ids"][0]:
                        # Extract HPO IDs from results
                        retrieved_ids = [
                            metadata.get("hpo_id")
                            for metadata in results["metadatas"][0][:k]
                        ]
                        ont_sim = average_max_similarity(expected_ids, retrieved_ids)
                        ont_similarity_values[k].append(ont_sim)
                        ont_similarities[f"ont_similarity@{k}"] = ont_sim
                    else:
                        ont_similarity_values[k].append(0.0)
                        ont_similarities[f"ont_similarity@{k}"] = 0.0

                # Record detailed results for this test case
                case_result = {
                    "case_id": i,
                    "description": description,
                    "text": text,
                    "expected_ids": expected_ids,
                    "mrr": mrr,
                    **hit_rates,
                    **ont_similarities,
                }

                if debug:
                    # Add top retrieved terms for debugging
                    if results["ids"][0]:
                        top_retrieved = []
                        for j, (term_id, metadata, distance) in enumerate(
                            zip(
                                results["ids"][0][:DEFAULT_TOP_K],
                                results["metadatas"][0][:DEFAULT_TOP_K],
                                results["distances"][0][:DEFAULT_TOP_K],
                            )
                        ):
                            hpo_id = metadata.get("hpo_id", "")
                            label = metadata.get("label", "")
                            similarity = 1.0 - distance
                            top_retrieved.append(
                                {
                                    "rank": j + 1,
                                    "hpo_id": hpo_id,
                                    "label": label,
                                    "similarity": similarity,
                                }
                            )
                        case_result["top_retrieved"] = top_retrieved

                detailed_results.append(case_result)

            except Exception as e:
                logging.error(f"Error processing test case {i+1}: {e}")
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

        # Calculate average metrics
        avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0
        avg_hit_rates = {
            k: (
                sum(hit_rate_values[k]) / len(hit_rate_values[k])
                if hit_rate_values[k]
                else 0
            )
            for k in k_values
        }
        avg_ont_similarities = {
            k: (
                sum(ont_similarity_values[k]) / len(ont_similarity_values[k])
                if ont_similarity_values[k]
                else 0
            )
            for k in k_values
        }

        # Compile final results
        results = {
            "model_name": model_name,
            "model_slug": model_slug,
            "test_file": test_file,
            "num_test_cases": len(test_cases),
            "similarity_threshold": similarity_threshold,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            # Raw values
            "mrr": mrr_values,
            # Average metrics
            "avg_mrr": avg_mrr,
        }

        # Add Hit Rate metrics
        for k in k_values:
            results[f"hit_rate@{k}"] = hit_rate_values[k]
            results[f"avg_hit_rate@{k}"] = avg_hit_rates[k]

        # Add ontology similarity metrics
        for k in k_values:
            results[f"ont_similarity@{k}"] = ont_similarity_values[k]
            results[f"avg_ont_similarity@{k}"] = avg_ont_similarities[k]

        # Note: Recall metric removed as it was redundant with hit_rate@max_k

        # Add detailed results
        results["detailed_results"] = detailed_results

        # Save results if requested
        if save_results:
            # Create summary dictionary
            summary = {
                "model": model_slug,
                "original_model_name": model_name,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "test_file": os.path.basename(test_file),
                "num_test_cases": len(test_cases),
                "mrr": avg_mrr,
                # Add raw metrics for error bar calculation
                "mrr_per_case": mrr_values,
            }

            # Add hit rates and ontology similarity
            for k in k_values:
                summary[f"hit_rate@{k}"] = avg_hit_rates[k]
                summary[f"ont_similarity@{k}"] = avg_ont_similarities[k]
                # Add raw metrics for each k value
                summary[f"hit_rate@{k}_per_case"] = hit_rate_values[k]
                summary[f"ont_similarity@{k}_per_case"] = ont_similarity_values[k]

            # Save summary to file
            summary_file = os.path.join(SUMMARIES_DIR, f"{run_id}.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logging.info(f"Summary saved to {summary_file}")

            # Save detailed results as CSV
            detailed_df = pd.DataFrame(detailed_results)
            csv_path = os.path.join(DETAILED_DIR, f"{run_id}_detailed.csv")
            detailed_df.to_csv(csv_path, index=False)
            logging.info(f"Detailed results saved to {csv_path}")

        # Log summary of results
        logging.info(f"Benchmark results for {model_name}:")
        logging.info(f"  MRR: {avg_mrr:.4f}")
        for k in k_values:
            logging.info(f"  Hit@{k}: {avg_hit_rates[k]:.4f}")
            logging.info(f"  OntSim@{k}: {avg_ont_similarities[k]:.4f}")
        # Recall logging removed as it was redundant with hit_rate@max_k

        return results

    except Exception as e:
        logging.error(f"Error during benchmark evaluation: {e}")
        return None


def compare_models(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
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
            "MRR": result["avg_mrr"],
        }

        # Add Hit Rate metrics
        for k in [1, 3, 5, 10]:
            key = f"avg_hit_rate@{k}"
            if key in result:
                model_data[f"HR@{k}"] = result[key]

        # Add Ontology Similarity metrics
        for k in [1, 3, 5, 10]:
            key = f"avg_ont_similarity@{k}"
            if key in result:
                model_data[f"OntSim@{k}"] = result[key]

        # Recall metric has been removed as it was redundant with hit_rate@max_k

        comparison_data.append(model_data)

    # Create and return DataFrame
    df = pd.DataFrame(comparison_data)

    # Sort by MRR (descending)
    if "MRR" in df.columns:
        df = df.sort_values("MRR", ascending=False)

    return df
