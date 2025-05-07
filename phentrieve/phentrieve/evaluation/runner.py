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
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
from tqdm import tqdm

from phentrieve.config import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_CANDIDATE_COUNT,
    DEFAULT_TOP_K,
    RESULTS_DIR,
    SUMMARIES_DIR,
    DETAILED_DIR,
    INDEX_DIR,
    DEFAULT_RERANKER_MODE,
    DEFAULT_TRANSLATION_DIR,
)
from phentrieve.data_processing.test_data_loader import load_test_data
from phentrieve.evaluation.metrics import (
    mean_reciprocal_rank,
    hit_rate_at_k,
    average_max_similarity,
    calculate_semantic_similarity,
    load_hpo_graph_data,
)
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.reranker import (
    load_cross_encoder,
    rerank_with_cross_encoder,
)
from phentrieve.utils import get_model_slug, load_german_translation_text


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
    enable_reranker: bool = False,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    rerank_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
    reranker_mode: str = DEFAULT_RERANKER_MODE,
    translation_dir: str = DEFAULT_TRANSLATION_DIR,
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
        enable_reranker: Whether to enable cross-encoder re-ranking
        reranker_model: Model name for the cross-encoder
        rerank_count: Number of candidates to re-rank
        reranker_mode: Re-ranking mode ('cross-lingual' or 'monolingual')
        translation_dir: Directory containing German translations of HPO terms

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

        # Load cross-encoder model if re-ranking is enabled
        cross_encoder = None
        if enable_reranker:
            device_str = (
                "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
            )
            logging.info(
                f"Loading cross-encoder model for {reranker_mode} re-ranking on {device_str}"
            )
            cross_encoder = load_cross_encoder(reranker_model, device=device_str)

            if cross_encoder is None:
                logging.warning(
                    f"Failed to load cross-encoder model {reranker_model}. Re-ranking will be disabled."
                )
                enable_reranker = False
            elif reranker_mode == "monolingual" and not os.path.exists(translation_dir):
                logging.warning(
                    f"Translation directory not found: {translation_dir}. Re-ranking will be disabled."
                )
                enable_reranker = False
            else:
                logging.info(
                    f"Successfully loaded cross-encoder model: {reranker_model}"
                )

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

        # Initialize result containers for both dense and re-ranked metrics
        # Baseline dense metrics
        mrr_dense_values = []
        hit_rate_dense_values = {k: [] for k in k_values}
        ont_similarity_dense_values = {k: [] for k in k_values}

        # Re-ranked metrics (will remain empty if re-ranking is disabled)
        mrr_reranked_values = []
        hit_rate_reranked_values = {k: [] for k in k_values}
        ont_similarity_reranked_values = {k: [] for k in k_values}

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
                # Query the index with bi-encoder
                dense_results = retriever.query(
                    text,
                    n_results=max(
                        max(k_values) * 3, rerank_count if enable_reranker else 0
                    ),
                )

                #### BASELINE DENSE METRICS CALCULATION ####

                # Extract HPO IDs from the results
                dense_term_ids = []
                dense_docs = []
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

                    if (
                        "documents" in dense_results
                        and dense_results["documents"]
                        and dense_results["documents"][0]
                    ):
                        dense_docs = dense_results["documents"][0]

                # Calculate baseline MRR
                mrr_dense = mean_reciprocal_rank(dense_results, expected_ids)
                mrr_dense_values.append(mrr_dense)

                # Calculate baseline Hit Rate at different K values
                dense_hit_rates = {}
                for k in k_values:
                    hit = hit_rate_at_k(dense_results, expected_ids, k=k)
                    hit_rate_dense_values[k].append(hit)
                    dense_hit_rates[f"hit_rate_dense@{k}"] = hit

                # Calculate baseline ontology similarity at different K values
                dense_ont_similarities = {}
                for k in k_values:
                    if dense_term_ids:
                        # Extract HPO IDs from results
                        retrieved_ids = dense_term_ids[:k]

                        # Calculate similarity scores for each expected ID
                        max_similarities = []
                        for expected_id in expected_ids:
                            # Check for exact match first
                            if expected_id in retrieved_ids:
                                # If there's an exact match, use 1.0 (ensures OntSim >= HR)
                                max_similarities.append(1.0)
                            else:
                                # Otherwise calculate semantic similarity
                                similarities = []
                                for retrieved_id in retrieved_ids:
                                    sim = calculate_semantic_similarity(
                                        expected_id, retrieved_id
                                    )
                                    similarities.append(sim)
                                # Take the max similarity for this expected ID
                                max_similarities.append(
                                    max(similarities) if similarities else 0.0
                                )

                        # Use the maximum similarity across all expected terms
                        # This aligns with HR@k which checks if ANY expected term is retrieved
                        ont_sim = max(max_similarities) if max_similarities else 0.0

                        ont_similarity_dense_values[k].append(ont_sim)
                        dense_ont_similarities[f"ont_similarity_dense@{k}"] = ont_sim
                    else:
                        ont_similarity_dense_values[k].append(0.0)
                        dense_ont_similarities[f"ont_similarity_dense@{k}"] = 0.0

                #### RE-RANKING (if enabled) ####
                reranked_hit_rates = {}
                reranked_ont_similarities = {}
                mrr_reranked = None
                reranked_results = None
                reranked_candidates = []

                # Perform re-ranking if enabled
                if (
                    enable_reranker
                    and cross_encoder
                    and dense_term_ids
                    and len(dense_term_ids) > 0
                ):
                    # Prepare candidates for re-ranking
                    candidates_to_rerank = []
                    num_to_rerank = min(rerank_count, len(dense_term_ids))

                    for j in range(num_to_rerank):
                        candidate = {
                            "hpo_id": dense_term_ids[j],
                            "english_doc": dense_docs[j] if j < len(dense_docs) else "",
                            "metadata": (
                                dense_metadatas[j] if j < len(dense_metadatas) else {}
                            ),
                            "bi_encoder_score": 0.0,  # Will be calculated if distances are available
                            "rank": j + 1,
                        }

                        # Add bi-encoder score if available
                        if (
                            "distances" in dense_results
                            and dense_results["distances"]
                            and dense_results["distances"][0]
                        ):
                            distance = dense_results["distances"][0][j]
                            score = (
                                1.0 - distance
                            )  # Convert distance to similarity score
                            candidate["bi_encoder_score"] = score

                        # For monolingual re-ranking, load German translation text
                        if reranker_mode == "monolingual":
                            german_text = load_german_translation_text(
                                candidate["hpo_id"], translation_dir
                            )
                            if german_text:
                                candidate["comparison_text"] = german_text
                            else:
                                # If translation not found, skip this candidate
                                logging.debug(
                                    f"No German translation found for {candidate['hpo_id']}"
                                )
                                continue
                        else:  # cross-lingual mode
                            candidate["comparison_text"] = candidate["english_doc"]

                        candidates_to_rerank.append(candidate)

                    # Re-rank candidates if we have valid candidates
                    if candidates_to_rerank:
                        reranked_candidates = rerank_with_cross_encoder(
                            text, candidates_to_rerank, cross_encoder
                        )

                        # Create results structure for metrics calculation (compatible with original format)
                        reranked_results = {
                            "ids": [[c["hpo_id"] for c in reranked_candidates]],
                            "metadatas": [[c["metadata"] for c in reranked_candidates]],
                            "documents": [
                                [c["english_doc"] for c in reranked_candidates]
                            ],
                            "distances": [
                                [
                                    1.0 - c["cross_encoder_score"]
                                    for c in reranked_candidates
                                ]
                            ],  # Convert score to distance
                        }

                        # Calculate re-ranked MRR
                        mrr_reranked = mean_reciprocal_rank(
                            reranked_results, expected_ids
                        )
                        mrr_reranked_values.append(mrr_reranked)

                        # Extract re-ranked term IDs for other metrics
                        reranked_term_ids = [c["hpo_id"] for c in reranked_candidates]

                        # Calculate re-ranked Hit Rate
                        for k in k_values:
                            hit = hit_rate_at_k(reranked_results, expected_ids, k=k)
                            hit_rate_reranked_values[k].append(hit)
                            reranked_hit_rates[f"hit_rate_reranked@{k}"] = hit

                        # Calculate re-ranked ontology similarity
                        for k in k_values:
                            if reranked_term_ids and len(reranked_term_ids) > 0:
                                retrieved_ids = (
                                    reranked_term_ids[:k]
                                    if k <= len(reranked_term_ids)
                                    else reranked_term_ids
                                )
                                ont_sim = average_max_similarity(
                                    expected_ids, retrieved_ids
                                )
                                ont_similarity_reranked_values[k].append(ont_sim)
                                reranked_ont_similarities[
                                    f"ont_similarity_reranked@{k}"
                                ] = ont_sim
                            else:
                                ont_similarity_reranked_values[k].append(0.0)
                                reranked_ont_similarities[
                                    f"ont_similarity_reranked@{k}"
                                ] = 0.0

                # Record detailed results for this test case with both baseline and re-ranked metrics
                case_result = {
                    "case_id": i,
                    "description": description,
                    "text": text,
                    "expected_ids": expected_ids,
                    "mrr_dense": mrr_dense,
                    "mrr_reranked": mrr_reranked,
                    **dense_hit_rates,
                    **dense_ont_similarities,
                }

                # Add re-ranked metrics if available
                if enable_reranker and reranked_results:
                    case_result.update(reranked_hit_rates)
                    case_result.update(reranked_ont_similarities)

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
        avg_ont_similarities_dense = {
            k: (
                sum(ont_similarity_dense_values[k])
                / len(ont_similarity_dense_values[k])
                if ont_similarity_dense_values[k]
                else 0
            )
            for k in k_values
        }

        # Re-ranked metrics (will be 0 if re-ranking was disabled)
        avg_mrr_reranked = (
            sum(mrr_reranked_values) / len(mrr_reranked_values)
            if mrr_reranked_values
            else 0
        )
        avg_hit_rates_reranked = {
            k: (
                sum(hit_rate_reranked_values[k]) / len(hit_rate_reranked_values[k])
                if hit_rate_reranked_values[k]
                else 0
            )
            for k in k_values
        }
        avg_ont_similarities_reranked = {
            k: (
                sum(ont_similarity_reranked_values[k])
                / len(ont_similarity_reranked_values[k])
                if ont_similarity_reranked_values[k]
                else 0
            )
            for k in k_values
        }

        # Compile final results with both dense and re-ranked metrics
        results = {
            "model_name": model_name,
            "model_slug": model_slug,
            "test_file": test_file,
            "num_test_cases": len(test_cases),
            "similarity_threshold": similarity_threshold,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            # Re-ranking configuration
            "reranker_enabled": enable_reranker,
            "reranker_model": reranker_model if enable_reranker else None,
            "reranker_mode": reranker_mode if enable_reranker else None,
            "rerank_count": rerank_count if enable_reranker else None,
            # Raw values - dense
            "mrr_dense": mrr_dense_values,
            # Average metrics - dense
            "avg_mrr_dense": avg_mrr_dense,
            # Raw values - reranked (if enabled)
            "mrr_reranked": mrr_reranked_values if enable_reranker else [],
            # Average metrics - reranked (if enabled)
            "avg_mrr_reranked": avg_mrr_reranked if enable_reranker else 0,
        }

        # Add Hit Rate metrics - dense
        for k in k_values:
            results[f"hit_rate_dense@{k}"] = hit_rate_dense_values[k]
            results[f"avg_hit_rate_dense@{k}"] = avg_hit_rates_dense[k]

        # Add ontology similarity metrics - dense
        for k in k_values:
            results[f"ont_similarity_dense@{k}"] = ont_similarity_dense_values[k]
            results[f"avg_ont_similarity_dense@{k}"] = avg_ont_similarities_dense[k]

        # Add re-ranked metrics if enabled
        if enable_reranker:
            # Add Hit Rate metrics - reranked
            for k in k_values:
                results[f"hit_rate_reranked@{k}"] = hit_rate_reranked_values[k]
                results[f"avg_hit_rate_reranked@{k}"] = avg_hit_rates_reranked[k]

            # Add ontology similarity metrics - reranked
            for k in k_values:
                results[f"ont_similarity_reranked@{k}"] = (
                    ont_similarity_reranked_values[k]
                )
                results[f"avg_ont_similarity_reranked@{k}"] = (
                    avg_ont_similarities_reranked[k]
                )

        # Add detailed results
        results["detailed_results"] = detailed_results

        # Save results if requested
        if save_results:
            # Create summary dictionary with both dense and re-ranked metrics
            summary = {
                "model": model_slug,
                "original_model_name": model_name,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "test_file": os.path.basename(test_file),
                "num_test_cases": len(test_cases),
                # Re-ranking configuration
                "reranker_enabled": enable_reranker,
                "reranker_model": reranker_model if enable_reranker else None,
                "reranker_mode": reranker_mode if enable_reranker else None,
                "rerank_count": rerank_count if enable_reranker else None,
                # Dense metrics
                "mrr_dense": avg_mrr_dense,
                "mrr_dense_per_case": mrr_dense_values,
                # Re-ranked metrics (if enabled)
                "mrr_reranked": avg_mrr_reranked if enable_reranker else None,
                "mrr_reranked_per_case": (
                    mrr_reranked_values if enable_reranker else None
                ),
            }

            # Add hit rates and ontology similarity for dense metrics
            for k in k_values:
                summary[f"hit_rate_dense@{k}"] = avg_hit_rates_dense[k]
                summary[f"ont_similarity_dense@{k}"] = avg_ont_similarities_dense[k]
                # Add raw metrics for each k value
                summary[f"hit_rate_dense@{k}_per_case"] = hit_rate_dense_values[k]
                summary[f"ont_similarity_dense@{k}_per_case"] = (
                    ont_similarity_dense_values[k]
                )

            # Add hit rates and ontology similarity for re-ranked metrics (if enabled)
            if enable_reranker:
                for k in k_values:
                    summary[f"hit_rate_reranked@{k}"] = avg_hit_rates_reranked[k]
                    summary[f"ont_similarity_reranked@{k}"] = (
                        avg_ont_similarities_reranked[k]
                    )
                    # Add raw metrics for each k value
                    summary[f"hit_rate_reranked@{k}_per_case"] = (
                        hit_rate_reranked_values[k]
                    )
                    summary[f"ont_similarity_reranked@{k}_per_case"] = (
                        ont_similarity_reranked_values[k]
                    )

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
        logging.info(f"  === Dense Retrieval Metrics ====")
        logging.info(f"  MRR (Dense): {avg_mrr_dense:.4f}")
        for k in k_values:
            logging.info(f"  Hit@{k} (Dense): {avg_hit_rates_dense[k]:.4f}")
            logging.info(f"  OntSim@{k} (Dense): {avg_ont_similarities_dense[k]:.4f}")

        if enable_reranker:
            logging.info(f"  === Re-ranked Metrics ({reranker_mode} mode) ===")
            logging.info(f"  MRR (Re-ranked): {avg_mrr_reranked:.4f}")
            for k in k_values:
                logging.info(f"  Hit@{k} (Re-ranked): {avg_hit_rates_reranked[k]:.4f}")
                logging.info(
                    f"  OntSim@{k} (Re-ranked): {avg_ont_similarities_reranked[k]:.4f}"
                )

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

        # Add Re-ranked metrics if available
        if "mrr_reranked" in result:
            # Check if the value is a list and get the average if so
            if isinstance(result["mrr_reranked"], list):
                model_data["MRR (ReRanked)"] = (
                    sum(result["mrr_reranked"]) / len(result["mrr_reranked"])
                    if result["mrr_reranked"]
                    else 0
                )
            else:
                model_data["MRR (ReRanked)"] = result["mrr_reranked"]

            # Calculate difference
            if "mrr_dense" in result:
                dense_mrr = (
                    sum(result["mrr_dense"]) / len(result["mrr_dense"])
                    if isinstance(result["mrr_dense"], list) and result["mrr_dense"]
                    else result["mrr_dense"]
                )
                # Only calculate reranked MRR and diff if reranking was enabled
                if result.get("reranker_enabled", False) and result.get("mrr_reranked"):
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

        # Add Ontology Similarity metrics
        for k in [1, 3, 5, 10]:
            # Dense retrieval metrics
            dense_key = f"ont_similarity_dense@{k}"
            if dense_key in result:
                if isinstance(result[dense_key], list):
                    model_data[f"OntSim@{k} (Dense)"] = (
                        sum(result[dense_key]) / len(result[dense_key])
                        if result[dense_key]
                        else 0
                    )
                else:
                    model_data[f"OntSim@{k} (Dense)"] = result[dense_key]

            # Re-ranked metrics if available
            reranked_key = f"ont_similarity_reranked@{k}"
            if reranked_key in result:
                if isinstance(result[reranked_key], list):
                    model_data[f"OntSim@{k} (ReRanked)"] = (
                        sum(result[reranked_key]) / len(result[reranked_key])
                        if result[reranked_key]
                        else 0
                    )
                else:
                    model_data[f"OntSim@{k} (ReRanked)"] = result[reranked_key]

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

        comparison_data.append(model_data)

    # Create and return DataFrame
    df = pd.DataFrame(comparison_data)

    # Sort by MRR (descending)
    if "MRR (Dense)" in df.columns:
        df = df.sort_values("MRR (Dense)", ascending=False)

    return df
