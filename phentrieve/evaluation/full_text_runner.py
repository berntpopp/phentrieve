"""
Document-level evaluation runner for Phentrieve HPO extraction.

This module provides functionality for evaluating Phentrieve's HPO extraction
performance on a document-by-document basis, using the orchestration pipelines
and semantic metrics.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from sentence_transformers import CrossEncoder

from phentrieve.evaluation.metrics import SimilarityFormula
from phentrieve.evaluation.semantic_metrics import (
    calculate_assertion_accuracy,
    calculate_semantically_aware_set_based_prf1,
)
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline

logger = logging.getLogger(__name__)


def evaluate_single_document_extraction(
    ground_truth_doc: dict[str, Any],
    language: str,  # Often from ground_truth_doc, but can be overridden
    # --- Phentrieve Components (pre-initialized) ---
    pipeline: TextProcessingPipeline,
    retriever: DenseRetriever,
    cross_encoder: Optional[CrossEncoder] = None,
    # --- Extraction Configs (passed to orchestrate_hpo_extraction) ---
    similarity_threshold_per_chunk: float = 0.3,
    num_results_per_chunk: int = 10,
    enable_reranker: bool = False,
    reranker_mode: str = "cross-lingual",
    translation_dir_path: Optional[Path] = None,
    rerank_count_per_chunk: int = 3,
    min_confidence_for_aggregated: float = 0.0,
    top_term_per_chunk_for_aggregated: bool = False,
    # --- Metrics Configs ---
    metrics_target_assertion_status: Optional[str] = None,  # For PRF1 calculation
    metrics_semantic_similarity_threshold: float = 0.7,  # For PRF1
    metrics_similarity_formula: SimilarityFormula = SimilarityFormula.HYBRID,  # For PRF1
    debug: bool = False,
) -> dict[str, Any]:
    """
    Evaluate Phentrieve's HPO extraction performance on a single document.

    This function orchestrates the full evaluation process:
    1. Processes the document text through the chunking pipeline
    2. Extracts HPO terms using the orchestrator
    3. Compares extracted terms to ground truth using semantic metrics
    4. Evaluates assertion status accuracy for correctly identified terms

    Args:
        ground_truth_doc: Document with ground truth annotations
        language: Language of the document
        pipeline: Initialized text processing pipeline
        retriever: Initialized dense retriever for HPO terms
        cross_encoder: Optional cross-encoder for re-ranking
        similarity_threshold_per_chunk: Min similarity for retrieval
        num_results_per_chunk: Number of HPO terms to retrieve per chunk
        enable_reranker: Whether to use cross-encoder re-ranking
        reranker_mode: Re-ranking mode (monolingual or cross-lingual)
        translation_dir_path: Directory with translations for cross-lingual
        rerank_count_per_chunk: Number of terms to re-rank per chunk
        min_confidence_for_aggregated: Min confidence for final results
        top_term_per_chunk_for_aggregated: Whether to keep only top term per chunk
        metrics_target_assertion_status: Target status for metrics
        metrics_semantic_similarity_threshold: Similarity threshold for metrics
        metrics_similarity_formula: Formula for semantic similarity
        debug: Whether to include additional debug information

    Returns:
        Dictionary with evaluation metrics and details
    """
    # Get document ID and text
    doc_id = ground_truth_doc.get("doc_id", "unknown_doc_id")
    full_text = ground_truth_doc.get("full_text") or ground_truth_doc.get("text", "")

    if not full_text:
        logger.error(f"No text content found in document {doc_id}")
        return {"doc_id": doc_id, "error": "No text content found in document"}

    logger.info(f"Evaluating document: {doc_id}")

    # Run text processing pipeline
    processed_chunks_from_pipeline = pipeline.process(full_text)
    text_chunks_for_orchestrator = [
        chunk["text"] for chunk in processed_chunks_from_pipeline
    ]

    # Get assertion statuses if available in the pipeline output
    assertion_statuses_for_orchestrator: list[str | None] | None = None
    if processed_chunks_from_pipeline and "status" in processed_chunks_from_pipeline[0]:
        # Extract assertion statuses from each chunk
        assertion_statuses_for_orchestrator = []
        for chunk in processed_chunks_from_pipeline:
            status = chunk.get("status")
            if status is None:
                assertion_statuses_for_orchestrator.append(None)
            elif hasattr(status, "value"):
                # If it's an enum, get the value
                assertion_statuses_for_orchestrator.append(status.value)
            else:
                # Otherwise convert to string
                assertion_statuses_for_orchestrator.append(str(status))

        # Debug logging for assertion statuses
        if assertion_statuses_for_orchestrator:
            logger.info(
                f"Found assertion statuses: {assertion_statuses_for_orchestrator}"
            )
        else:
            logger.info("No assertion statuses found in processed chunks")

    # Use HPO extraction orchestrator
    try:
        # Only use cross encoder if enabled
        cross_encoder_to_use = None
        if enable_reranker:
            cross_encoder_to_use = cross_encoder

        # Call orchestrate_hpo_extraction
        aggregated_results, chunk_results = orchestrate_hpo_extraction(
            text_chunks=text_chunks_for_orchestrator,
            retriever=retriever,
            num_results_per_chunk=num_results_per_chunk,
            chunk_retrieval_threshold=similarity_threshold_per_chunk,
            cross_encoder=cross_encoder_to_use,
            translation_dir_path=translation_dir_path,
            language=language,
            reranker_mode=reranker_mode,
            top_term_per_chunk=top_term_per_chunk_for_aggregated,
            min_confidence_for_aggregated=min_confidence_for_aggregated,
            assertion_statuses=assertion_statuses_for_orchestrator,
        )

        # Debug: Print chunk-by-chunk results if debug mode is enabled
        if debug:
            logger.info(f"\n=== CHUNK EXTRACTION DETAILS FOR DOCUMENT: {doc_id} ===")
            for i, chunk_data in enumerate(chunk_results):
                chunk_idx = chunk_data.get("chunk_idx", i)
                chunk_text = chunk_data.get("chunk_text", "")
                matches = chunk_data.get("matches", [])

                logger.info(
                    f"\nChunk {chunk_idx + 1}: '{chunk_text}' (extracted {len(matches)} terms)"
                )

                for j, term in enumerate(matches):
                    term_id = term.get("id", "unknown")
                    name = term.get("name", "")
                    score = term.get("score", 0.0)  # Score is the confidence value
                    status = term.get("assertion_status")

                    score_str = (
                        f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                    )
                    status_str = status if status else "None"
                    logger.info(
                        f"  [{j + 1}] {term_id} - {name} ({status_str}) [score: {score_str}]"
                    )

            logger.info("\n=== END CHUNK EXTRACTION DETAILS ===")

        # Get ground truth annotations
        ground_truth_annotations = ground_truth_doc.get("annotations", [])

        if not ground_truth_annotations:
            logger.warning(f"No ground truth annotations found for document {doc_id}")
            return {
                "doc_id": doc_id,
                "warning": "No ground truth annotations found",
                "num_extracted": len(aggregated_results),
                "extracted_terms": aggregated_results,
            }

        # Calculate precision, recall, F1
        metrics_results = calculate_semantically_aware_set_based_prf1(
            extracted_annotations=aggregated_results,
            ground_truth_annotations=ground_truth_annotations,
            target_assertion_status=metrics_target_assertion_status,
            semantic_similarity_threshold=metrics_semantic_similarity_threshold,
            similarity_formula=metrics_similarity_formula,
        )

        # Extract metrics
        metrics_results.get("precision", 0.0)
        metrics_results.get("recall", 0.0)
        metrics_results.get("f1_score", 0.0)
        metrics_results.get("tp_count", 0)
        metrics_results.get("fp_count", 0)
        metrics_results.get("fn_count", 0)

        # Extract exact and semantic match counts
        exact_match_count = metrics_results.get("exact_match_count", 0)
        semantic_match_count = metrics_results.get("semantic_match_count", 0)

        # Get match-related metrics
        exact_precision = metrics_results.get("exact_precision", 0.0)
        exact_recall = metrics_results.get("exact_recall", 0.0)
        exact_f1_score = metrics_results.get("exact_f1_score", 0.0)
        semantic_precision = metrics_results.get("semantic_precision", 0.0)
        semantic_recall = metrics_results.get("semantic_recall", 0.0)
        semantic_f1_score = metrics_results.get("semantic_f1_score", 0.0)

        tp_matched_pairs = metrics_results.get("tp_matched_pairs_list", [])

        # Calculate assertion accuracy
        assertion_acc, correct_assert, common_assert = calculate_assertion_accuracy(
            tp_matched_pairs
        )

        # Compile results
        doc_metrics = {
            "doc_id": doc_id,  # Include document ID
        }
        doc_metrics.update(
            {
                "precision": metrics_results.get("precision", 0.0),
                "recall": metrics_results.get("recall", 0.0),
                "f1_score": metrics_results.get("f1_score", 0.0),
                "tp_count": metrics_results.get("tp_count", 0),
                "fp_count": metrics_results.get("fp_count", 0),
                "fn_count": metrics_results.get("fn_count", 0),
                # Include exact and semantic match counts
                "exact_match_count": exact_match_count,
                "semantic_match_count": semantic_match_count,
                # Include match-specific metrics
                "exact_precision": exact_precision,
                "exact_recall": exact_recall,
                "exact_f1_score": exact_f1_score,
                "semantic_precision": semantic_precision,
                "semantic_recall": semantic_recall,
                "semantic_f1_score": semantic_f1_score,
                "assertion_accuracy": assertion_acc,
                "correctly_asserted_count": correct_assert,
                "common_terms_for_assertion_eval": common_assert,
                "num_extracted_final": len(aggregated_results),
                "num_ground_truth_total": len(ground_truth_annotations),
            }
        )

        # Add debug information if requested
        if debug:
            # Get the extracted and ground truth IDs for better analysis
            extracted_ids = [term["id"] for term in aggregated_results]
            ground_truth_ids = [
                term.get("hpo_id") or term.get("id")
                for term in ground_truth_annotations
            ]

            # Print more detailed debug info if requested
            logger.info(f"\n=== DEBUG INFO FOR DOCUMENT: {doc_id} ===")

            # Print ground truth terms
            logger.info(f"Ground Truth Terms ({len(ground_truth_annotations)}):")
            for i, term in enumerate(ground_truth_annotations):
                hpo_id = term.get("hpo_id") or term.get("id")
                name = term.get("label") or term.get("name")
                status = term.get("assertion_status")
                logger.info(f"  [{i + 1}] {hpo_id} - {name} ({status})")

            # Print extracted terms
            logger.info(f"\nExtracted Terms ({len(aggregated_results)}):")
            for i, term in enumerate(aggregated_results):
                term_id = term.get("id", "unknown")
                name = term.get("name", "")
                score = term.get("score")
                status = term.get("assertion_status")

                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                logger.info(
                    f"  [{i + 1}] {term_id} - {name} ({status}) [score: {score_str}]"
                )

            # Print matched pairs
            logger.info(f"\nMatched Term Pairs ({len(tp_matched_pairs)}):")
            for i, (extracted, ground_truth) in enumerate(tp_matched_pairs):
                extracted_id = extracted.get("id")
                ground_truth_id = ground_truth.get("hpo_id") or ground_truth.get("id")
                extracted_name = extracted.get("name")
                ground_truth_name = ground_truth.get("label") or ground_truth.get(
                    "name"
                )
                logger.info(
                    f"  [{i + 1}] {extracted_id} - {extracted_name} ↔ {ground_truth_id} - {ground_truth_name}"
                )

            # Print false positives
            true_positive_extracted_ids = [pair[0]["id"] for pair in tp_matched_pairs]
            false_positive_ids = [
                term_id
                for term_id in extracted_ids
                if term_id not in true_positive_extracted_ids
            ]
            logger.info(f"\nFalse Positives ({len(false_positive_ids)}):")
            for i, fp_id in enumerate(false_positive_ids):
                term = next((t for t in aggregated_results if t["id"] == fp_id), None)
                if term:
                    name = term.get("name")
                    logger.info(f"  [{i + 1}] {fp_id} - {name}")

            # Print false negatives
            true_positive_ground_truth_ids = [
                pair[1].get("hpo_id") or pair[1].get("id") for pair in tp_matched_pairs
            ]
            false_negative_ids = [
                term_id
                for term_id in ground_truth_ids
                if term_id not in true_positive_ground_truth_ids
            ]
            logger.info(f"\nFalse Negatives ({len(false_negative_ids)}):")
            for i, fn_id in enumerate(false_negative_ids):
                term = next(
                    (
                        t
                        for t in ground_truth_annotations
                        if t.get("hpo_id") == fn_id or t.get("id") == fn_id
                    ),
                    None,
                )
                if term:
                    label = term.get("label") or term.get("name")
                    logger.info(f"  [{i + 1}] {fn_id} - {label}")

            logger.info("\n=== END DEBUG INFO ===\n")

            # Add to results
            doc_metrics.update(
                {
                    "extracted_ids": extracted_ids,
                    "ground_truth_ids": ground_truth_ids,
                    "true_positive_pairs": [
                        (pair[0]["id"], pair[1].get("hpo_id") or pair[1].get("id"))
                        for pair in tp_matched_pairs
                    ],
                    "false_positive_ids": false_positive_ids,
                    "false_negative_ids": false_negative_ids,
                    "extracted_terms_full": aggregated_results,
                    "ground_truth_full": ground_truth_annotations,
                    "processed_chunks_count": len(text_chunks_for_orchestrator),
                }
            )

        return doc_metrics

    except Exception as e:
        logger.error(f"Error evaluating document {doc_id}: {str(e)}", exc_info=True)
        return {"doc_id": doc_id, "error": f"Evaluation failed: {str(e)}"}
