#!/usr/bin/env python
"""
Example script for running full-text HPO extraction evaluation.

This script demonstrates how to use the Phentrieve evaluation system to:
1. Load ground truth annotated documents
2. Initialize Phentrieve components (pipeline, retriever, etc.)
3. Run the evaluation on each document
4. Calculate and display overall results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.evaluation.semantic_metrics import (
    calculate_semantically_aware_set_based_prf1,
    calculate_assertion_accuracy,
    SimilarityFormula,
)
from phentrieve.evaluation.full_text_runner import evaluate_single_document_extraction
from phentrieve.evaluation.full_text_loader import load_full_text_annotations


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run HPO extraction evaluation on annotated documents"
    )

    # Input/output
    parser.add_argument(
        "--ground-truth-file",
        type=str,
        required=True,
        help="Path to JSONL file with annotated ground truth documents",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="FremyCompany/BioLORD-2023-M",
        help="Dense retriever model to use",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default="ncbi/MedCPT-Cross-Encoder",
        help="Cross-encoder reranker model to use",
    )
    parser.add_argument(
        "--enable-reranker", action="store_true", help="Enable cross-encoder reranking"
    )

    # Text processing
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language of the documents (en, de, etc.)",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="detailed",
        choices=["simple", "paragraph", "sentence", "detailed"],
        help="Chunking strategy for text processing",
    )
    parser.add_argument(
        "--translation-dir",
        type=str,
        help="Path to translation directory for non-English documents",
    )

    # Extraction parameters
    parser.add_argument(
        "--num-results",
        type=int,
        default=10,
        help="Number of HPO terms to retrieve per chunk",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.3,
        help="Minimum similarity score for retrieved terms",
    )
    parser.add_argument(
        "--reranker-mode",
        type=str,
        default="cross-lingual",
        choices=["monolingual", "cross-lingual"],
        help="Mode for cross-encoder reranking",
    )
    parser.add_argument(
        "--top-term-per-chunk",
        action="store_true",
        help="Keep only the top term per chunk",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence for aggregated results",
    )

    # Evaluation parameters
    parser.add_argument(
        "--target-assertion-status",
        type=str,
        default="affirmed",
        help="Target assertion status for evaluation (affirmed, negated, or leave empty for all)",
    )
    parser.add_argument(
        "--semantic-similarity-threshold",
        type=float,
        default=0.7,
        help="Threshold for considering semantic matches in evaluation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include additional debug information in results",
    )

    return parser.parse_args()


def initialize_components(args):
    """Initialize Phentrieve components based on arguments."""
    # Initialize dense retriever
    logger.info(f"Initializing dense retriever with model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load the sentence transformer model
    logger.info(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model, device=device)

    # Initialize the retriever with the model
    retriever = DenseRetriever.from_model_name(model=model, model_name=args.model)

    # Initialize cross-encoder if enabled
    cross_encoder = None
    if args.enable_reranker:
        logger.info(f"Initializing cross-encoder with model: {args.reranker}")
        cross_encoder = CrossEncoder(args.reranker, device=device)
        logger.info(f"Successfully loaded cross-encoder model: {args.reranker}")

    # Initialize text processing pipeline with selected strategy
    logger.info(
        f"Initializing text processing pipeline with strategy: {args.chunking_strategy}"
    )

    # Configure chunking pipeline based on selected strategy
    if args.chunking_strategy == "simple":
        chunking_config = [{"type": "paragraph"}]
    elif args.chunking_strategy == "paragraph":
        chunking_config = [{"type": "paragraph"}]
    elif args.chunking_strategy == "sentence":
        chunking_config = [{"type": "paragraph"}, {"type": "sentence"}]
    elif args.chunking_strategy == "detailed":
        chunking_config = [
            {"type": "paragraph"},
            {"type": "sentence"},
            {"type": "fine_grained_punctuation"},
        ]
    else:
        chunking_config = [{"type": "paragraph"}]  # Default to paragraph

    # Configure assertion detection
    assertion_config = {
        "enable_keyword": True,
        "enable_dependency": True,
        "preference": "dependency",
    }

    # Create pipeline with configured strategy
    pipeline = TextProcessingPipeline(
        language=args.language,
        chunking_pipeline_config=chunking_config,
        assertion_config=assertion_config,
        sbert_model_for_semantic_chunking=model,
    )

    return retriever, cross_encoder, pipeline


def run_evaluation(args, retriever, cross_encoder, pipeline):
    """Run evaluation on all documents."""
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load ground truth documents
    ground_truth_docs = load_full_text_annotations(Path(args.ground_truth_file))
    logger.info(f"Loaded {len(ground_truth_docs)} ground truth documents")

    # Prepare translation directory
    translation_dir_path = Path(args.translation_dir) if args.translation_dir else None

    # Process each document
    all_results = []
    for idx, doc in enumerate(ground_truth_docs):
        logger.info(
            f"Processing document {idx+1}/{len(ground_truth_docs)}: {doc.get('doc_id', f'doc_{idx}')}"
        )

        # Run evaluation for this document
        result = evaluate_single_document_extraction(
            ground_truth_doc=doc,
            language=args.language,
            pipeline=pipeline,
            retriever=retriever,
            cross_encoder=cross_encoder,
            similarity_threshold_per_chunk=args.similarity_threshold,
            num_results_per_chunk=args.num_results,
            enable_reranker=args.enable_reranker,
            reranker_mode=args.reranker_mode,
            translation_dir_path=translation_dir_path,
            rerank_count_per_chunk=args.num_results,
            min_confidence_for_aggregated=args.min_confidence,
            top_term_per_chunk_for_aggregated=args.top_term_per_chunk,
            metrics_target_assertion_status=(
                args.target_assertion_status
                if args.target_assertion_status != "all"
                else None
            ),
            metrics_semantic_similarity_threshold=args.semantic_similarity_threshold,
            metrics_similarity_formula=SimilarityFormula.HYBRID,
            debug=args.debug,
        )

        all_results.append(result)

    return all_results


def calculate_summary_metrics(all_results):
    """Calculate summary metrics across all documents."""
    # Extract overall metrics
    metrics = {
        "precision": [r["precision"] for r in all_results if "precision" in r],
        "recall": [r["recall"] for r in all_results if "recall" in r],
        "f1_score": [r["f1_score"] for r in all_results if "f1_score" in r],
        "assertion_accuracy": [
            r["assertion_accuracy"] for r in all_results if "assertion_accuracy" in r
        ],
    }

    # Extract exact match metrics
    exact_metrics = {
        "exact_precision": [],
        "exact_recall": [],
        "exact_f1_score": [],
    }

    # Extract semantic match metrics
    semantic_metrics = {
        "semantic_precision": [],
        "semantic_recall": [],
        "semantic_f1_score": [],
    }

    # Process each document's results individually to calculate proper metrics
    for r in all_results:
        if "exact_match_count" in r and "semantic_match_count" in r:
            # Get counts from the document
            exact_count = r.get("exact_match_count", 0)
            semantic_count = r.get("semantic_match_count", 0)
            extracted_count = r.get("num_extracted_final", 0)
            truth_count = r.get("num_ground_truth_total", 0)

            # Calculate exact match metrics for this document
            if extracted_count > 0 and exact_count > 0:
                doc_exact_precision = exact_count / extracted_count
            else:
                doc_exact_precision = 0.0

            if truth_count > 0 and exact_count > 0:
                # Ensure recall never exceeds 1.0
                doc_exact_recall = min(1.0, exact_count / truth_count)
            else:
                doc_exact_recall = 0.0

            if doc_exact_precision + doc_exact_recall > 0:
                doc_exact_f1 = (
                    2
                    * (doc_exact_precision * doc_exact_recall)
                    / (doc_exact_precision + doc_exact_recall)
                )
            else:
                doc_exact_f1 = 0.0

            # Calculate semantic match metrics for this document
            if extracted_count > 0 and semantic_count > 0:
                doc_semantic_precision = semantic_count / extracted_count
            else:
                doc_semantic_precision = 0.0

            if truth_count > 0 and semantic_count > 0:
                # Ensure recall never exceeds 1.0
                doc_semantic_recall = min(1.0, semantic_count / truth_count)
            else:
                doc_semantic_recall = 0.0

            if doc_semantic_precision + doc_semantic_recall > 0:
                doc_semantic_f1 = (
                    2
                    * (doc_semantic_precision * doc_semantic_recall)
                    / (doc_semantic_precision + doc_semantic_recall)
                )
            else:
                doc_semantic_f1 = 0.0

            # Add to exact metrics
            exact_metrics["exact_precision"].append(doc_exact_precision)
            exact_metrics["exact_recall"].append(doc_exact_recall)
            exact_metrics["exact_f1_score"].append(doc_exact_f1)

            # Add to semantic metrics
            semantic_metrics["semantic_precision"].append(doc_semantic_precision)
            semantic_metrics["semantic_recall"].append(doc_semantic_recall)
            semantic_metrics["semantic_f1_score"].append(doc_semantic_f1)

    # Count documents
    total_docs = len(all_results)
    successful_docs = len(metrics["precision"])

    # Calculate aggregated counts
    total_tp = sum(r.get("tp_count", 0) for r in all_results)
    total_fp = sum(r.get("fp_count", 0) for r in all_results)
    total_ground_truth = sum(r.get("num_ground_truth_total", 0) for r in all_results)
    total_fn = max(0, total_ground_truth - total_tp)
    total_exact_matches = sum(r.get("exact_match_count", 0) for r in all_results)
    total_semantic_matches = sum(r.get("semantic_match_count", 0) for r in all_results)

    # Calculate micro-average precision/recall/f1 for overall metrics
    if total_tp + total_fp > 0:
        micro_precision = total_tp / (total_tp + total_fp)
    else:
        micro_precision = 0.0

    if total_ground_truth > 0:
        # Ensure recall never exceeds 1.0
        micro_recall = min(1.0, total_tp / total_ground_truth)
    else:
        micro_recall = 0.0

    micro_f1 = (
        2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    # Initialize total counts for micro-average calculations
    total_extracted_count = sum(r.get("num_extracted_final", 0) for r in all_results)
    total_ground_truth_count = sum(
        r.get("num_ground_truth_total", 0) for r in all_results
    )

    # For exact match metrics, calculate precision, recall, and F1 based on exact matches only
    if total_exact_matches > 0 and total_extracted_count > 0:
        micro_exact_precision = total_exact_matches / total_extracted_count
    else:
        micro_exact_precision = 0.0

    if total_exact_matches > 0 and total_ground_truth_count > 0:
        # Ensure recall never exceeds 1.0
        micro_exact_recall = min(1.0, total_exact_matches / total_ground_truth_count)
    else:
        micro_exact_recall = 0.0

    micro_exact_f1 = (
        2
        * (micro_exact_precision * micro_exact_recall)
        / (micro_exact_precision + micro_exact_recall)
        if (micro_exact_precision + micro_exact_recall) > 0
        else 0.0
    )

    # For semantic match metrics, calculate similarly using only the semantic matches
    if total_semantic_matches > 0 and total_extracted_count > 0:
        micro_semantic_precision = total_semantic_matches / total_extracted_count
    else:
        micro_semantic_precision = 0.0

    if total_semantic_matches > 0 and total_ground_truth_count > 0:
        # Ensure recall never exceeds 1.0
        micro_semantic_recall = min(
            1.0, total_semantic_matches / total_ground_truth_count
        )
    else:
        micro_semantic_recall = 0.0

    micro_semantic_f1 = (
        2
        * (micro_semantic_precision * micro_semantic_recall)
        / (micro_semantic_precision + micro_semantic_recall)
        if (micro_semantic_precision + micro_semantic_recall) > 0
        else 0.0
    )

    # Calculate macro-average (mean across documents) for all metrics
    macro_metrics = {
        metric: np.mean(values) if values else 0.0 for metric, values in metrics.items()
    }

    macro_exact_metrics = {
        metric: np.mean(values) if values else 0.0
        for metric, values in exact_metrics.items()
    }

    macro_semantic_metrics = {
        metric: np.mean(values) if values else 0.0
        for metric, values in semantic_metrics.items()
    }

    # Create summary
    summary = {
        "total_documents": total_docs,
        "successful_documents": successful_docs,
        # Overall counts
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_exact_matches": total_exact_matches,
        "total_semantic_matches": total_semantic_matches,
        # Micro-average (pooled across all documents)
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        # Micro-average for exact matches
        "micro_exact_precision": micro_exact_precision,
        "micro_exact_recall": micro_exact_recall,
        "micro_exact_f1": micro_exact_f1,
        # Micro-average for semantic matches
        "micro_semantic_precision": micro_semantic_precision,
        "micro_semantic_recall": micro_semantic_recall,
        "micro_semantic_f1": micro_semantic_f1,
        # Macro-average (mean of document metrics)
        "macro_precision": macro_metrics["precision"],
        "macro_recall": macro_metrics["recall"],
        "macro_f1": macro_metrics["f1_score"],
        "macro_assertion_accuracy": macro_metrics["assertion_accuracy"],
        # Macro-average for exact matches
        "macro_exact_precision": macro_exact_metrics.get("exact_precision", 0.0),
        "macro_exact_recall": macro_exact_metrics.get("exact_recall", 0.0),
        "macro_exact_f1": macro_exact_metrics.get("exact_f1_score", 0.0),
        # Macro-average for semantic matches
        "macro_semantic_precision": macro_semantic_metrics.get(
            "semantic_precision", 0.0
        ),
        "macro_semantic_recall": macro_semantic_metrics.get("semantic_recall", 0.0),
        "macro_semantic_f1": macro_semantic_metrics.get("semantic_f1_score", 0.0),
    }

    return summary


def save_results(all_results, summary, args):
    """Save evaluation results and summary to files."""
    output_dir = Path(args.output_dir)

    # Save individual document results
    results_file = output_dir / "document_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Create a dataframe for easier analysis
    df_results = []
    for r in all_results:
        if "error" in r or "warning" in r:
            continue

        df_results.append(
            {
                "doc_id": r.get("doc_id", "unknown"),
                "precision": r.get("precision", 0.0),
                "recall": r.get("recall", 0.0),
                "f1_score": r.get("f1_score", 0.0),
                "assertion_accuracy": r.get("assertion_accuracy", 0.0),
                "tp_count": r.get("tp_count", 0),
                "fp_count": r.get("fp_count", 0),
                "fn_count": r.get("fn_count", 0),
                "extracted_count": r.get("num_extracted_final", 0),
                "ground_truth_count": r.get("num_ground_truth_total", 0),
            }
        )

    if df_results:
        df = pd.DataFrame(df_results)
        csv_file = output_dir / "results.csv"
        df.to_csv(csv_file, index=False)

    logger.info(f"Results saved to {output_dir}")

    return results_file, summary_file


def display_summary(summary):
    """Display summary results in console."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Total documents: {summary['total_documents']}")
    print(f"Successfully evaluated: {summary['successful_documents']}")
    print(
        f"Total matches: {summary['total_tp']} (Exact: {summary['total_exact_matches']}, Semantic: {summary['total_semantic_matches']})"
    )
    print(f"False Positives: {summary['total_fp']}")
    print(f"False Negatives: {summary['total_fn']}")

    print("\n" + "-" * 80)
    print("OVERALL METRICS (EXACT + SEMANTIC MATCHES)")
    print("-" * 80)

    print("\nMicro-averaged metrics (pooled):")
    print(f"  Precision: {summary['micro_precision']:.4f}")
    print(f"  Recall: {summary['micro_recall']:.4f}")
    print(f"  F1 Score: {summary['micro_f1']:.4f}")

    print("\nMacro-averaged metrics (mean per document):")
    print(f"  Precision: {summary['macro_precision']:.4f}")
    print(f"  Recall: {summary['macro_recall']:.4f}")
    print(f"  F1 Score: {summary['macro_f1']:.4f}")
    print(f"  Assertion Accuracy: {summary['macro_assertion_accuracy']:.2f}%")

    print("\n" + "-" * 80)
    print("EXACT MATCH METRICS")
    print("-" * 80)

    print("\nMicro-averaged metrics (pooled):")
    print(f"  Precision: {summary['micro_exact_precision']:.4f}")
    print(f"  Recall: {summary['micro_exact_recall']:.4f}")
    print(f"  F1 Score: {summary['micro_exact_f1']:.4f}")

    print("\nMacro-averaged metrics (mean per document):")
    print(f"  Precision: {summary['macro_exact_precision']:.4f}")
    print(f"  Recall: {summary['macro_exact_recall']:.4f}")
    print(f"  F1 Score: {summary['macro_exact_f1']:.4f}")

    print("\n" + "-" * 80)
    print("SEMANTIC MATCH METRICS")
    print("-" * 80)

    print("\nMicro-averaged metrics (pooled):")
    print(f"  Precision: {summary['micro_semantic_precision']:.4f}")
    print(f"  Recall: {summary['micro_semantic_recall']:.4f}")
    print(f"  F1 Score: {summary['micro_semantic_f1']:.4f}")

    print("\nMacro-averaged metrics (mean per document):")
    print(f"  Precision: {summary['macro_semantic_precision']:.4f}")
    print(f"  Recall: {summary['macro_semantic_recall']:.4f}")
    print(f"  F1 Score: {summary['macro_semantic_f1']:.4f}")

    print("=" * 80)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Check if ground truth file exists
    if not Path(args.ground_truth_file).exists():
        logger.error(f"Ground truth file not found: {args.ground_truth_file}")
        sys.exit(1)

    try:
        # Initialize components
        retriever, cross_encoder, pipeline = initialize_components(args)

        # Run evaluation
        all_results = run_evaluation(args, retriever, cross_encoder, pipeline)

        # Calculate summary metrics
        summary = calculate_summary_metrics(all_results)

        # Save results
        save_results(all_results, summary, args)

        # Display summary
        display_summary(summary)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
