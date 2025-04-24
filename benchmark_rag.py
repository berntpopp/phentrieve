#!/usr/bin/env python3
"""
Benchmark tool for the German HPO RAG system.

This script evaluates the performance of the RAG system using various metrics:
- Mean Reciprocal Rank (MRR)
- Hit Rate at K (HR@K)
- Precision, Recall, and F1 score

Multiple embedding models can be compared if they have been indexed.
"""

import argparse
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from utils import get_model_slug, get_index_dir, get_collection_name
from german_hpo_rag import query_hpo, calculate_similarity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Default model
DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

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
        with open(test_file, 'r', encoding='utf-8') as f:
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
    if not results or not results['ids'] or not results['ids'][0]:
        return 0.0
    
    # Get all retrieved HPO IDs
    retrieved_ids = []
    for i, (hpo_id, metadata, distance) in enumerate(zip(
        results['ids'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        # Add HPO ID with its actual rank (1-based)
        retrieved_ids.append((metadata['hpo_id'], i + 1, calculate_similarity(distance)))
    
    # Sort by similarity score (descending)
    retrieved_ids.sort(key=lambda x: x[2], reverse=True)
    
    # Re-rank based on similarity
    ranked_ids = [(hpo_id, i + 1, sim) for i, (hpo_id, _, sim) in enumerate(retrieved_ids)]
    
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
    if not results or not results['ids'] or not results['ids'][0]:
        return 0.0
    
    # Get all retrieved HPO IDs with similarity scores
    retrieved_ids = []
    for i, (hpo_id, metadata, distance) in enumerate(zip(
        results['ids'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        retrieved_ids.append((metadata['hpo_id'], calculate_similarity(distance)))
    
    # Sort by similarity score (descending)
    retrieved_ids.sort(key=lambda x: x[1], reverse=True)
    
    # Check if any expected ID is in top K
    top_k_ids = [item[0] for item in retrieved_ids[:k]]
    
    for expected_id in expected_ids:
        if expected_id in top_k_ids:
            return 1.0
    
    return 0.0


def precision_recall_f1(results, expected_ids, similarity_threshold=0.3, k=None):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        results: Results from query_hpo
        expected_ids: List of expected HPO IDs
        similarity_threshold: Minimum similarity score to consider
        k: If set, only consider top k results after sorting by similarity
    
    Returns:
        tuple: (precision, recall, f1)
    """
    if not results or not results['ids'] or not results['ids'][0]:
        return 0.0, 0.0, 0.0
    
    # Get all retrieved HPO IDs with similarity scores
    retrieved_items = []
    for i, (hpo_id, metadata, distance) in enumerate(zip(
        results['ids'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        similarity = calculate_similarity(distance)
        if similarity >= similarity_threshold:
            retrieved_items.append((metadata['hpo_id'], similarity))
    
    # Sort by similarity score (descending)
    retrieved_items.sort(key=lambda x: x[1], reverse=True)
    
    # Limit to top K if specified
    if k is not None:
        retrieved_items = retrieved_items[:k]
    
    retrieved_ids = [item[0] for item in retrieved_items]
    
    # Calculate metrics
    true_positives = len(set(retrieved_ids).intersection(set(expected_ids)))
    
    if not retrieved_ids:
        precision = 0.0
    else:
        precision = true_positives / len(retrieved_ids)
    
    if not expected_ids:
        recall = 0.0
    else:
        recall = true_positives / len(expected_ids)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def run_benchmark(model_name, test_cases, k_values=(1, 3, 5, 10)):
    """
    Run benchmark with given model and test cases.
    
    Args:
        model_name: Name of the embedding model
        test_cases: List of test case dictionaries
        k_values: Tuple of k values for Hit Rate@K
    
    Returns:
        dict: Benchmark results
    """
    # Get index directory and collection name
    index_dir = get_index_dir()
    model_collection_name = get_collection_name(model_name)
    
    # Check if index exists
    if not os.path.exists(index_dir):
        logging.error(f"Error: Index directory '{index_dir}' not found. Please run setup_hpo_index.py first.")
        return None
    
    # Load model
    try:
        logging.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer model: {e}")
        return None
    
    # Connect to ChromaDB
    try:
        logging.info(f"Connecting to ChromaDB at {index_dir}")
        client = chromadb.PersistentClient(path=index_dir)
        
        # First try model-specific collection name
        try:
            logging.info(f"Trying model-specific collection: {model_collection_name}")
            collection = client.get_collection(model_collection_name)
            logging.info(f"Connected to ChromaDB collection '{model_collection_name}' with {collection.count()} entries.")
        except Exception as e:
            # If that fails, try the default collection name
            logging.info(f"Model-specific collection not found, trying default collection 'hpo_multilingual'")
            try:
                collection = client.get_collection('hpo_multilingual')
                logging.info(f"Connected to ChromaDB collection 'hpo_multilingual' with {collection.count()} entries.")
            except Exception as e:
                logging.error(f"Error: No suitable collection found for model '{model_name}'")
                logging.error(f"Make sure you've run setup_hpo_index.py with this model or the default model")
                return None
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {e}")
        return None
    
    # Prepare results
    results = {
        'model': model_name,
        'model_slug': get_model_slug(model_name),
        'total_cases': len(test_cases),
        'mrr': [],
        'f1_scores': [],
        'precision': [],
        'recall': []
    }
    
    # Add Hit Rate@K entries
    for k in k_values:
        results[f'hit_rate@{k}'] = []
    
    # Process each test case
    logging.info(f"Running benchmark with {len(test_cases)} test cases...")
    for case in tqdm(test_cases, desc=f"Model: {get_model_slug(model_name)}"):
        text = case['text']
        expected_ids = case['expected_hpo_ids']
        
        # Query the collection
        query_results = query_hpo(text, model, collection, n_results=20)
        
        # Calculate metrics
        if query_results:
            # MRR
            mrr = mean_reciprocal_rank(query_results, expected_ids)
            results['mrr'].append(mrr)
            
            # Hit Rate@K
            for k in k_values:
                hit_rate = hit_rate_at_k(query_results, expected_ids, k=k)
                results[f'hit_rate@{k}'].append(hit_rate)
            
            # Precision, Recall, F1
            precision, recall, f1 = precision_recall_f1(
                query_results, expected_ids, 
                similarity_threshold=0.3, k=10
            )
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_scores'].append(f1)
        else:
            # No results - all metrics are 0
            results['mrr'].append(0.0)
            for k in k_values:
                results[f'hit_rate@{k}'].append(0.0)
            results['precision'].append(0.0)
            results['recall'].append(0.0)
            results['f1_scores'].append(0.0)
    
    # Calculate averages
    results['avg_mrr'] = np.mean(results['mrr'])
    results['avg_precision'] = np.mean(results['precision'])
    results['avg_recall'] = np.mean(results['recall'])
    results['avg_f1'] = np.mean(results['f1_scores'])
    
    for k in k_values:
        results[f'avg_hit_rate@{k}'] = np.mean(results[f'hit_rate@{k}'])
    
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
    
    # Create sample test cases
    sample_data = [
        {
            "text": "Das Kind hat Kleinwuchs und eine Makrozephalie.",
            "expected_hpo_ids": ["HP:0004322", "HP:0000098"],
            "description": "Short stature and macrocephaly"
        },
        {
            "text": "Der Patient zeigt eine Hemiplegie und Krampfanfälle.",
            "expected_hpo_ids": ["HP:0002301", "HP:0001250"],
            "description": "Hemiplegia and seizures"
        },
        {
            "text": "Das Kind hat einen niedrigen Blutdruck und Schwindel.",
            "expected_hpo_ids": ["HP:0002615", "HP:0002321"],
            "description": "Hypotension and vertigo"
        },
        {
            "text": "Der Patient hat eine fortschreitende Muskelschwäche.",
            "expected_hpo_ids": ["HP:0001324"],
            "description": "Progressive muscle weakness"
        },
        {
            "text": "Die Patientin zeigt geistige Behinderung und eine Mikrozephalie.",
            "expected_hpo_ids": ["HP:0001249", "HP:0000252"],
            "description": "Intellectual disability and microcephaly"
        }
    ]
    
    logging.info(f"Creating sample test data at {sample_file}")
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    return sample_file


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
            'Model': results['model_slug'],
            'MRR': results['avg_mrr'],
            'Precision': results['avg_precision'],
            'Recall': results['avg_recall'],
            'F1': results['avg_f1']
        }
        
        # Add Hit Rate metrics
        for k in [1, 3, 5, 10]:
            if f'avg_hit_rate@{k}' in results:
                model_metrics[f'HR@{k}'] = results[f'avg_hit_rate@{k}']
        
        comparison.append(model_metrics)
    
    df = pd.DataFrame(comparison)
    
    # Set Model as index for better display
    df.set_index('Model', inplace=True)
    
    return df


def main():
    """
    Main function for the benchmark tool.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark German HPO RAG system with different models."
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="List of model names to benchmark (default: only the default model)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="JSON file with test cases (if not provided, a sample will be used)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file for detailed results"
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
    
    # Run benchmark for each model
    results_list = []
    for model_name in args.model_names:
        logging.info(f"Benchmarking model: {model_name}")
        results = run_benchmark(model_name, test_cases)
        if results:
            results_list.append(results)
    
    # Generate comparison
    if len(results_list) > 0:
        comparison_df = compare_models(results_list)
        
        # Display results
        print("\n===== Benchmark Results =====")
        print(f"Test cases: {len(test_cases)}")
        print(f"Models evaluated: {len(results_list)}")
        print("\nModel Comparison:")
        print(comparison_df.round(4))
        
        # Save detailed results
        detailed_results = []
        for result in results_list:
            for i in range(len(test_cases)):
                row = {
                    'model': result['model_slug'],
                    'case_id': i,
                    'text': test_cases[i]['text'],
                    'expected_ids': ', '.join(test_cases[i]['expected_hpo_ids']),
                    'mrr': result['mrr'][i],
                    'precision': result['precision'][i],
                    'recall': result['recall'][i],
                    'f1': result['f1_scores'][i]
                }
                
                for k in [1, 3, 5, 10]:
                    if f'hit_rate@{k}' in result:
                        row[f'hit_rate@{k}'] = result[f'hit_rate@{k}'][i]
                
                detailed_results.append(row)
        
        # Save results to CSV
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to {args.output}")
    else:
        print("No benchmark results collected.")


if __name__ == "__main__":
    main()
