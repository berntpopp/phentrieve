#!/usr/bin/env python3
"""
Benchmark tool for the German HPO RAG system.

This script evaluates the performance of the RAG system using retrieval metrics:
- Mean Reciprocal Rank (MRR)
- Hit Rate at K (HR@K)

Multiple embedding models can be compared if they have been indexed.
"""

import os
import argparse
import json
import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from utils import (
    get_model_slug,
    get_index_dir,
    generate_collection_name,
    get_embedding_dimension,
)
from german_hpo_rag import query_hpo, calculate_similarity

# Set up device - use CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Default model
DEFAULT_MODEL = "FremyCompany/BioLORD-2023-M"
DEFAULT_TEST_FILE = os.path.join("data", "test_cases", "sample_test_cases.json")
RESULTS_DIR = "benchmark_results"
SUMMARIES_DIR = os.path.join(RESULTS_DIR, "summaries")


# Define test data directory
TEST_DATA_DIR = "data/test_cases"


def load_test_data(test_file):
    """
    Load test cases from a JSON file.

    Expected format:
    [
        {
            "text": "German clinical text",
            "expected_hpo_ids": ["HP:0000123", "HP:0000456"],
            "description": "Optional description of the case"
        },
        ...
    ]
    """
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        logging.info(f"Loaded {len(test_cases)} test cases from {test_file}")
        return test_cases
    except Exception as e:
        logging.error(f"Error loading test data from {test_file}: {e}")
        return None


def mean_reciprocal_rank(results, expected_ids):
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        results: Results from query_hpo
        expected_ids: List of expected HPO IDs

    Returns:
        float: MRR value (0 if no matches)
    """
    if not results or not results["ids"] or not results["ids"][0]:
        return 0.0

    # Get all retrieved HPO IDs
    retrieved_ids = []
    for i, (hpo_id, metadata, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        # Add HPO ID with its actual rank (1-based)
        retrieved_ids.append(
            (metadata["hpo_id"], i + 1, calculate_similarity(distance))
        )

    # Sort by similarity score (descending)
    retrieved_ids.sort(key=lambda x: x[2], reverse=True)

    # Re-rank based on similarity
    ranked_ids = [
        (hpo_id, i + 1, sim) for i, (hpo_id, _, sim) in enumerate(retrieved_ids)
    ]

    # Find the first match
    for hpo_id, rank, _ in ranked_ids:
        if hpo_id in expected_ids:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(results, expected_ids, k=5):
    """
    Calculate Hit Rate at K.

    Args:
        results: Results from query_hpo
        expected_ids: List of expected HPO IDs
        k: Number of top results to consider

    Returns:
        float: 1.0 if any expected ID is in top K, 0.0 otherwise
    """
    if not results or not results["ids"] or not results["ids"][0]:
        return 0.0

    # Get all retrieved HPO IDs with similarity scores
    retrieved_ids = []
    for i, (hpo_id, metadata, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        retrieved_ids.append((metadata["hpo_id"], calculate_similarity(distance)))

    # Sort by similarity score (descending)
    retrieved_ids.sort(key=lambda x: x[1], reverse=True)

    # Check if any expected ID is in top K
    top_k_ids = [item[0] for item in retrieved_ids[:k]]

    for expected_id in expected_ids:
        if expected_id in top_k_ids:
            return 1.0

    return 0.0


def run_benchmark(
    model_name, test_cases, k_values=(1, 3, 5, 10), similarity_threshold=0.1
):
    """
    Run benchmark with given model and test cases.

    Args:
        model_name: Name of the embedding model
        test_cases: List of test case dictionaries
        k_values: Tuple of k values for Hit Rate@K
        similarity_threshold: Threshold for similarity scores

    Returns:
        dict: Benchmark results
    """
    model_slug = get_model_slug(model_name)
    index_dir = get_index_dir()
    collection_name = generate_collection_name(model_name)

    # Load the embedding model
    logging.info(f"Loading embedding model: {model_name}")
    try:
        # Special handling for Jina model which requires trust_remote_code=True
        jina_model_id = "jinaai/jina-embeddings-v2-base-de"
        if model_name == jina_model_id:
            logging.info(
                f"Loading Jina model '{model_name}' with trust_remote_code=True on {device}"
            )
            # Security note: Only use trust_remote_code=True for trusted sources
            model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            logging.info(f"Loading model '{model_name}' on {device}")
            model = SentenceTransformer(model_name)

        # Move model to GPU if available
        model = model.to(device)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

    logging.info("Model loaded successfully.")

    # Connect to ChromaDB
    logging.info(f"Connecting to ChromaDB at {index_dir}")
    try:
        client = chromadb.PersistentClient(path=index_dir)

        # Only use the model-specific collection - no fallback
        try:
            logging.info(f"Using model-specific collection: {collection_name}")
            collection = client.get_collection(collection_name)
        except Exception as e:
            logging.error(f"Error: Collection '{collection_name}' not found")
            logging.error(
                f"You need to run setup_hpo_index.py with model {model_name} to create a collection compatible with its embedding dimension {get_embedding_dimension(model_name)}."
            )
            return None

        logging.info(
            f"Connected to ChromaDB collection with {collection.count()} entries."
        )
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {e}")
        return None

    # Initialize result containers
    mrr_values = []
    hit_rates = {k: [] for k in k_values}

    # For detailed per-test-case results
    detailed_results = []

    # Run the benchmark
    logging.info(
        f"Running benchmark for model {model_name} with {len(test_cases)} test cases at similarity threshold {similarity_threshold}"
    )

    # Set up a progress bar
    pbar = tqdm(test_cases, desc=f"Model: {model_slug}")

    for test_case in pbar:
        german_text = test_case["text"]
        expected_ids = test_case["expected_hpo_ids"]

        # Log the query only in debug mode to avoid cluttering the output
        logging.info(f"Query: '{german_text}'")

        # Query the HPO index with the German text
        # The query_hpo function will handle the encoding internally
        query_results = query_hpo(german_text, model, collection)

        # Initialize detail dict for this test case
        test_detail = {
            "query": german_text,
            "expected_ids": expected_ids,
            "expected_labels": test_case.get("description", ""),
            "top_hits": [],
            "correct_hits": [],
            "similarity_threshold": similarity_threshold,
        }

        if query_results is None:
            # Log error and skip this test case
            logging.error(f"Query failed for: '{german_text}'")
            mrr_values.append(0.0)
            for k in k_values:
                hit_rates[k].append(0.0)

            test_detail["error"] = "Query failed"
            detailed_results.append(test_detail)
            continue

        # Process results to get ranked hits
        if query_results and query_results["ids"] and query_results["ids"][0]:
            ranked_hits = []
            for i, (hpo_id, metadata, distance) in enumerate(
                zip(
                    query_results["ids"][0],
                    query_results["metadatas"][0],
                    query_results["distances"][0],
                )
            ):
                similarity = calculate_similarity(distance)
                hpo_term = metadata["hpo_id"]
                hpo_name = metadata.get("hpo_name", "Unknown")
                ranked_hits.append(
                    {
                        "rank": i + 1,
                        "hpo_id": hpo_term,
                        "name": hpo_name,
                        "similarity": similarity,
                    }
                )

            # Sort by similarity (descending)
            ranked_hits.sort(key=lambda x: x["similarity"], reverse=True)

            # Re-rank after sorting
            for i, hit in enumerate(ranked_hits):
                hit["rank"] = i + 1

            # Store top 5 hits for detailed results
            test_detail["top_hits"] = ranked_hits[:5]

            # Find where the expected IDs are in the results
            for expected_id in expected_ids:
                found = False
                for hit in ranked_hits:
                    if hit["hpo_id"] == expected_id:
                        test_detail["correct_hits"].append(
                            {
                                "expected_id": expected_id,
                                "rank": hit["rank"],
                                "similarity": hit["similarity"],
                                "above_threshold": hit["similarity"]
                                >= similarity_threshold,
                            }
                        )
                        found = True
                        break
                if not found:
                    test_detail["correct_hits"].append(
                        {
                            "expected_id": expected_id,
                            "rank": float("inf"),
                            "similarity": 0.0,
                            "above_threshold": False,
                        }
                    )

        # Calculate MRR
        mrr = mean_reciprocal_rank(query_results, expected_ids)
        mrr_values.append(mrr)
        test_detail["mrr"] = mrr

        # Calculate Hit Rate @ K
        for k in k_values:
            hit_rate = hit_rate_at_k(query_results, expected_ids, k=k)
            hit_rates[k].append(hit_rate)
            test_detail[f"hit_rate@{k}"] = hit_rate

        detailed_results.append(test_detail)

    # Aggregate results
    results = {
        "model_name": model_name,
        "model_slug": model_slug,
        "mrr": mrr_values,
        "detailed_results": detailed_results,  # Add detailed results
        "similarity_threshold": similarity_threshold,
    }

    # Add Hit Rate results
    for k in k_values:
        results[f"hit_rate@{k}"] = hit_rates[k]

    # Calculate averages
    results["avg_mrr"] = np.mean(mrr_values)

    for k in k_values:
        results[f"avg_hit_rate@{k}"] = np.mean(results[f"hit_rate@{k}"])

    return results


def create_sample_test_data():
    """
    Create a sample test dataset if none exists.
    """
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)

    sample_file = os.path.join(TEST_DATA_DIR, "sample_test_cases.json")

    # Don't overwrite existing test file
    if os.path.exists(sample_file):
        return sample_file

    # Create sample test cases with single terms only
    sample_data = [
        {
            "text": "Kleinwuchs",
            "expected_hpo_ids": ["HP:0004322"],
            "description": "Short stature",
        },
        {
            "text": "Makrozephalie",
            "expected_hpo_ids": ["HP:0000256"],
            "description": "Macrocephaly",
        },
        {
            "text": "Hemiplegie",
            "expected_hpo_ids": ["HP:0002301"],
            "description": "Hemiplegia",
        },
        {
            "text": "Krampfanfälle",
            "expected_hpo_ids": ["HP:0001250"],
            "description": "Seizures",
        },
        {
            "text": "Niedriger Blutdruck",
            "expected_hpo_ids": ["HP:0002615"],
            "description": "Hypotension",
        },
        {
            "text": "Schwindel",
            "expected_hpo_ids": ["HP:0002321"],
            "description": "Vertigo",
        },
        {
            "text": "Fortschreitende Muskelschwäche",
            "expected_hpo_ids": ["HP:0001324"],
            "description": "Progressive muscle weakness",
        },
        {
            "text": "Geistige Behinderung",
            "expected_hpo_ids": ["HP:0001249"],
            "description": "Intellectual disability",
        },
        {
            "text": "Mikrozephalie",
            "expected_hpo_ids": ["HP:0000252"],
            "description": "Microcephaly",
        },
    ]

    logging.info(f"Creating sample test data at {sample_file}")
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    return sample_file


def display_test_case_results(results):
    """
    Display detailed per-test-case results.

    Args:
        results: Benchmark results dictionary with detailed_results
    """
    if "detailed_results" not in results:
        print("No detailed results available.")
        return

    threshold = results.get("similarity_threshold", 0.3)
    print(f"\n===== Detailed Test Case Results =====")
    print(f"Model: {results['model_slug']}")
    print(f"Similarity threshold: {threshold}")
    print("\n")

    for i, test_detail in enumerate(results["detailed_results"]):
        # Print test case information
        print(f"[{i+1}] Query: '{test_detail['query']}'")
        print(
            f"    Expected HPO terms: {', '.join(test_detail['expected_ids'])} ({test_detail['expected_labels']})"
        )

        # Print top hits summary
        if test_detail.get("top_hits"):
            print(f"    Top hits:")
            # Show top 10 hits for comprehensive analysis
            display_count = min(10, len(test_detail["top_hits"]))
            for hit in test_detail["top_hits"][:display_count]:
                print(
                    f"      {hit['rank']}. {hit['hpo_id']} - {hit['name']} (Similarity: {hit['similarity']:.4f})"
                )

        # Print correct hits information if any
        if test_detail.get("correct_hits"):
            print(f"    Correct term positions:")
            for hit in test_detail["correct_hits"]:
                if hit["rank"] == float("inf"):
                    print(f"      {hit['expected_id']} - Not found")
                else:
                    threshold_status = "✓" if hit["above_threshold"] else "✗"
                    print(
                        f"      {hit['expected_id']} - Rank: {hit['rank']} (Similarity: {hit['similarity']:.4f}) {threshold_status}"
                    )

        # Print metrics
        print(
            f"    Metrics: MRR: {test_detail.get('mrr', 0):.4f}, "
            f"Hit@1: {test_detail.get('hit_rate@1', 0):.1f}, "
            f"Hit@3: {test_detail.get('hit_rate@3', 0):.1f}, "
            f"Hit@5: {test_detail.get('hit_rate@5', 0):.1f}, "
            f"Hit@10: {test_detail.get('hit_rate@10', 0):.1f}"
        )
        print("\n")


def compare_models(results_list):
    """
    Compare results across different models.

    Args:
        results_list: List of benchmark result dictionaries

    Returns:
        DataFrame: Comparison table
    """
    if not results_list:
        return None

    comparison = []
    for results in results_list:
        model_metrics = {
            "Model": results["model_slug"],
            "MRR": results["avg_mrr"],
        }

        # Add Hit Rate metrics
        for k in [1, 3, 5, 10]:
            if f"avg_hit_rate@{k}" in results:
                model_metrics[f"HR@{k}"] = results[f"avg_hit_rate@{k}"]

        comparison.append(model_metrics)

    df = pd.DataFrame(comparison)

    # Set Model as index for better display
    df.set_index("Model", inplace=True)

    return df


def main():
    """
    Main function for the benchmark tool.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark German HPO RAG system with different models."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Single model name to benchmark (default: only the default model)",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="List of model names to benchmark (overrides --model-name if provided)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run benchmark on all available models",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="JSON file with test cases (if not provided, a sample will be used)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file for detailed results",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.1,
        help="Minimum similarity threshold (default: 0.1)",
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed per-test-case results"
    )

    args = parser.parse_args()

    # Load or create test data
    if args.test_file and os.path.exists(args.test_file):
        test_file = args.test_file
    else:
        test_file = create_sample_test_data()

    test_cases = load_test_data(test_file)
    if not test_cases:
        return

    # Determine which models to run
    models_to_run = []
    if args.all_models:
        # Define all available models
        models_to_run = [
            "FremyCompany/BioLORD-2023-M",
            "jinaai/jina-embeddings-v2-base-de",
            "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/distiluse-base-multilingual-cased-v2",
        ]
        logging.info(f"Running benchmark on all {len(models_to_run)} available models")
    elif args.model_names:
        models_to_run = args.model_names
        logging.info(f"Running benchmark on {len(models_to_run)} specified models")
    else:
        models_to_run = [args.model_name]
        logging.info(f"Running benchmark on single model: {args.model_name}")

    # Run benchmark for each model
    results_list = []
    for model_name in models_to_run:
        logging.info(f"Benchmarking model: {model_name}")
        results = run_benchmark(
            model_name, test_cases, similarity_threshold=args.similarity_threshold
        )
        if results:
            results_list.append(results)
            # Show detailed results if requested
            if args.detailed:
                display_test_case_results(results)

    # Generate comparison
    if len(results_list) > 0:
        comparison_df = compare_models(results_list)

        # Display results
        print("\n===== Benchmark Results =====")
        print(f"Test cases: {len(test_cases)}")
        print(f"Models evaluated: {len(results_list)}")
        print(f"Similarity threshold: {args.similarity_threshold}")
        print("\nModel Comparison:")

        # Format and display results in a clean table format
        comparison_table = comparison_df.round(4)
        pd.set_option("display.max_columns", None)  # Show all columns
        pd.set_option("display.width", 200)  # Wide display
        pd.set_option("display.float_format", "{:.4f}".format)  # Consistent formatting
        print(comparison_table)

        # Also save as CSV for easier parsing by other tools
        csv_path = "benchmark_comparison.csv"
        comparison_df.to_csv(csv_path)
        print(f"\nComparison table saved to {csv_path}")

        # Save summary JSON files for each model
        os.makedirs(SUMMARIES_DIR, exist_ok=True)
        for result in results_list:
            model_slug = result["model_slug"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(SUMMARIES_DIR, f"{model_slug}_{timestamp}.json")

            # Create summary dictionary
            summary = {
                "model": model_slug,
                "original_model_name": result["model_name"],
                "timestamp": datetime.now().isoformat(),
                "mrr": result.get("avg_mrr", 0),
            }

            # Add hit rates
            for k in [1, 3, 5, 10]:
                if f"avg_hit_rate@{k}" in result:
                    summary[f"hit_rate@{k}"] = result[f"avg_hit_rate@{k}"]

            # Save to JSON file
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"Summary saved to {summary_file}")

        # Save detailed results
        detailed_results = []
        for result in results_list:
            for i in range(len(test_cases)):
                row = {
                    "model": result["model_slug"],
                    "case_id": i,
                    "text": test_cases[i]["text"],
                    "expected_ids": ", ".join(test_cases[i]["expected_hpo_ids"]),
                    "mrr": result["mrr"][i],
                }

                for k in [1, 3, 5, 10]:
                    if f"hit_rate@{k}" in result:
                        row[f"hit_rate@{k}"] = result[f"hit_rate@{k}"][i]

                detailed_results.append(row)

        # Save results to CSV
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to {args.output}")
    else:
        print("No benchmark results collected.")


if __name__ == "__main__":
    main()
