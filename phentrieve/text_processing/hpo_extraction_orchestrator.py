"""HPO term extraction orchestrator.

This module handles the orchestration of HPO term extraction from text,
separating the core processing logic from CLI concerns.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from sentence_transformers import CrossEncoder

from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.utils import load_translation_text


logger = logging.getLogger(__name__)


def orchestrate_hpo_extraction(
    raw_text: str,
    language: str,
    # Pre-loaded models & components
    pipeline: TextProcessingPipeline,
    retriever: DenseRetriever,
    cross_encoder: Optional[CrossEncoder] = None,
    # Processing configurations
    similarity_threshold_per_chunk: float = 0.3,
    num_results_per_chunk: int = 10,
    enable_reranker: bool = False,
    reranker_mode: str = "cross-lingual",
    translation_dir_path: Optional[Path] = None,
    rerank_count_per_chunk: int = 50,
    min_confidence_for_aggregated: float = 0.0,
    top_term_per_chunk_for_aggregated: bool = False,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Orchestrate the extraction of HPO terms from text.

    Args:
        raw_text: The input text to process
        language: Language of the text
        pipeline: Initialized TextProcessingPipeline
        retriever: Initialized DenseRetriever
        cross_encoder: Optional CrossEncoder for reranking
        similarity_threshold_per_chunk: Minimum similarity score for matches
        num_results_per_chunk: Maximum number of results per chunk
        enable_reranker: Whether to use cross-encoder reranking
        reranker_mode: Mode for reranking ("cross-lingual" or "monolingual")
        translation_dir_path: Path to HPO term translations for monolingual mode
        rerank_count_per_chunk: Number of candidates to consider for reranking
        min_confidence_for_aggregated: Minimum confidence for aggregated results
        top_term_per_chunk_for_aggregated: Whether to keep only the top term per chunk
        debug: Enable debug logging

    Returns:
        Tuple containing:
        - List of aggregated HPO term results
        - List of chunk-level results
        - List of processed chunks
    """
    # Process the text through the pipeline
    logger.info("Starting HPO extraction orchestration process")
    logger.info(f"Processing text ({len(raw_text)} chars) in language: {language}")
    logger.info("Running text through chunking and assertion pipeline...")
    processed_chunks = pipeline.process(raw_text)
    logger.info(
        f"Generated {len(processed_chunks)} chunks from text processing pipeline"
    )

    # Extract HPO terms from each chunk
    logger.info("Extracting HPO terms from processed chunks")

    # Storage for results
    chunk_results = []
    all_hpo_terms = defaultdict(list)  # HPO ID -> list of evidence

    logger.info("Beginning HPO term extraction for each processed chunk")

    # Process each chunk for HPO terms
    for i, chunk_data in enumerate(processed_chunks):
        chunk_text = chunk_data["text"]
        assertion_status = chunk_data["status"]
        # Get assertion details if needed for future use
        # assertion_details = chunk_data.get("assertion_details", {})

        # Get status value for logging
        status_value = getattr(assertion_status, "value", assertion_status)
        logger.info(f"Processing chunk {i+1}: Status is {status_value}")

        # Retrieve HPO terms for this chunk
        try:
            # Get matching HPO terms with the chunk text as query
            logger.debug(f"Retrieving HPO terms for chunk {i+1} using DenseRetriever")
            results = retriever.query(
                text=chunk_text,
                n_results=max(
                    num_results_per_chunk,
                    rerank_count_per_chunk if enable_reranker else 0,
                ),
                include_similarities=True,
            )

            # Filter results based on similarity threshold
            logger.debug(
                f"Filtering retrieval results with threshold: {similarity_threshold_per_chunk}"
            )
            filtered_results = retriever.filter_results(
                results,
                min_similarity=similarity_threshold_per_chunk,
                max_results=max(
                    num_results_per_chunk,
                    rerank_count_per_chunk if enable_reranker else 0,
                ),
            )

            # Format the results into a list of dictionaries
            hpo_matches = []
            if filtered_results.get("ids") and filtered_results["ids"][0]:
                for i, doc_id in enumerate(filtered_results["ids"][0]):
                    if i < len(filtered_results["metadatas"][0]):
                        metadata = filtered_results["metadatas"][0][i]
                        similarity = (
                            filtered_results["similarities"][0][i]
                            if filtered_results.get("similarities")
                            else None
                        )

                        # Extract the term name from metadata
                        term_name = metadata.get("name", "Unknown")

                        # Try to extract from nested properties if available
                        if not term_name or term_name == "Unknown":
                            if "properties" in metadata and isinstance(
                                metadata["properties"], dict
                            ):
                                props = metadata["properties"]
                                if "name" in props and props["name"]:
                                    term_name = props["name"]
                                elif "label" in props and props["label"]:
                                    term_name = props["label"]

                        hpo_match = {
                            "id": doc_id,
                            "name": term_name,
                            "score": similarity,
                            "rank": i,
                            "metadata": metadata,
                        }
                        hpo_matches.append(hpo_match)

                # Apply re-ranking if enabled
                if enable_reranker and cross_encoder and hpo_matches:
                    try:
                        # Prepare candidates for re-ranking
                        candidates_for_reranking = []

                        # Limit to the number of candidates specified by rerank_count
                        candidates_to_rerank = hpo_matches[
                            : min(len(hpo_matches), rerank_count_per_chunk)
                        ]

                        for candidate in candidates_to_rerank:
                            # Get the appropriate comparison text based on reranker mode
                            if (
                                translation_dir_path
                                and language != "en"
                                and reranker_mode == "monolingual"
                            ):
                                # For monolingual mode, load the translation in the target language
                                try:
                                    comparison_text = load_translation_text(
                                        hpo_id=candidate["id"],
                                        language=language,
                                        translation_dir=(
                                            str(translation_dir_path)
                                            if translation_dir_path
                                            else None
                                        ),
                                    )
                                except Exception as e:
                                    # If translation fails, fall back to English text
                                    comparison_text = candidate["name"]
                                    if debug:
                                        logger.debug(
                                            f"Translation for {candidate['id']} failed: {str(e)}. "
                                            f"Using English text."
                                        )
                            else:  # cross-lingual mode
                                # Use the English label directly
                                comparison_text = candidate["name"]

                            # Re-load HPO term in original language for monolingual
                            # re-ranking
                            rerank_candidate = {
                                "hpo_id": candidate["id"],
                                "english_doc": candidate["name"],
                                "metadata": candidate["metadata"],
                                "bi_encoder_score": candidate["score"],
                                "rank": candidate["rank"],
                                "comparison_text": comparison_text,
                            }
                            candidates_for_reranking.append(rerank_candidate)

                        # Perform re-ranking
                        if candidates_for_reranking:
                            from phentrieve.retrieval import reranker as reranker_module

                            reranked_candidates = (
                                reranker_module.rerank_with_cross_encoder(
                                    query=chunk_text,
                                    candidates=candidates_for_reranking,
                                    cross_encoder_model=cross_encoder,
                                )
                            )

                            # Update hpo_matches with reranked results
                            if reranked_candidates:
                                # Create a mapping of hpo_id to reranked score
                                reranked_scores = {}
                                for i, candidate in enumerate(reranked_candidates):
                                    reranked_scores[candidate["hpo_id"]] = {
                                        "reranker_score": candidate[
                                            "cross_encoder_score"
                                        ],
                                        "new_rank": i,
                                    }

                                # Update original matches with reranker scores
                                for match in hpo_matches:
                                    if match["id"] in reranked_scores:
                                        match["reranker_score"] = reranked_scores[
                                            match["id"]
                                        ]["reranker_score"]
                                        match["reranked_rank"] = reranked_scores[
                                            match["id"]
                                        ]["new_rank"]

                                # If we're using reranking for final ordering, resort matches by reranker score
                                hpo_matches = sorted(
                                    hpo_matches,
                                    key=lambda x: x.get(
                                        "reranker_score", -float("inf")
                                    ),
                                    reverse=True,
                                )[
                                    :num_results_per_chunk
                                ]  # Limit to the requested number of results

                                if debug:
                                    logger.debug(
                                        f"Re-ranked {len(reranked_candidates)} candidates"
                                    )
                    except Exception as e:
                        logger.warning(f"Error during re-ranking: {str(e)}")
                        if debug:
                            import traceback

                            traceback.print_exc()

            logger.info(f"Chunk {i+1}: Found {len(hpo_matches)} HPO terms")

            # Store chunk results
            chunk_result = {
                "chunk_id": i + 1,
                "text": chunk_text,
                "status": status_value,
                "hpo_terms": [],
            }

            # If top_term_per_chunk is enabled, keep only the highest-scoring match
            if top_term_per_chunk_for_aggregated and hpo_matches:
                # Sort by score descending
                sorted_matches = sorted(
                    hpo_matches, key=lambda x: x["score"], reverse=True
                )
                # Keep only the top-scoring match
                hpo_matches = [sorted_matches[0]]
                logger.info(
                    f"Chunk {i+1}: Taking only top term (score: {hpo_matches[0]['score']:.4f})"
                )

            for match in hpo_matches:
                # Basic match information
                hpo_term_info = {
                    "hpo_id": match["id"],
                    "name": match["name"],
                    "score": match["score"],
                    "reranker_score": match.get("reranker_score"),
                }

                # Add to chunk result
                chunk_result["hpo_terms"].append(hpo_term_info)

                # Add to aggregate results
                evidence = {
                    "chunk_id": i + 1,
                    "chunk_text": chunk_text,
                    "status": status_value,
                    "score": match["score"],
                    "reranker_score": match.get("reranker_score"),
                    "name": match["name"],  # Include the HPO term name in the evidence
                }
                all_hpo_terms[match["id"]].append(evidence)

            chunk_results.append(chunk_result)

        except Exception as e:
            logger.error(f"Error retrieving HPO terms for chunk {i+1}: {str(e)}")
            if debug:
                import traceback

                traceback.print_exc()

    # Aggregate results
    logger.info(
        f"Aggregating HPO terms from all chunks ({len(all_hpo_terms)} unique HPO IDs found)"
    )
    aggregated_results = []

    for hpo_id, evidence_list in all_hpo_terms.items():
        # Get basic HPO information from the first evidence match
        if not evidence_list or not evidence_list[0].get("chunk_id"):
            continue  # Skip if we don't have any evidence for this HPO term

        # Determine the best score
        max_score = max(evidence["score"] for evidence in evidence_list)
        max_reranker_score = max(
            (
                evidence.get("reranker_score", -float("inf"))
                for evidence in evidence_list
            ),
            default=None,
        )

        # Count status types
        affirmed_count = sum(1 for e in evidence_list if e["status"] == "affirmed")
        negated_count = sum(1 for e in evidence_list if e["status"] == "negated")
        normal_count = sum(1 for e in evidence_list if e["status"] == "normal")
        uncertain_count = sum(1 for e in evidence_list if e["status"] == "uncertain")

        # Determine overall status
        if affirmed_count > 0 and negated_count == 0:
            overall_status = "affirmed"
        elif negated_count > 0 and affirmed_count == 0:
            overall_status = "negated"
        elif normal_count > 0 and affirmed_count == 0 and negated_count == 0:
            overall_status = "normal"
        elif uncertain_count > 0:
            overall_status = "uncertain"
        else:
            # Mixed status (both affirmed and negated)
            overall_status = "mixed"

        # Calculate a confidence score based on all evidence
        evidence_count = len(evidence_list)

        # Confidence formula: We combine scores from bi-encoder and cross-encoder if available
        if max_reranker_score is not None and max_reranker_score != -float("inf"):
            # If we have reranker scores, include them in confidence calculation
            confidence_score = (max_score + max(0, max_reranker_score)) / 2
        else:
            # Otherwise just use the bi-encoder similarity score
            confidence_score = max_score

        # Extract term name from evidence
        term_name = "Unknown"
        for evidence in evidence_list:
            if "name" in evidence:
                term_name = evidence["name"]
                break

        aggregated_results.append(
            {
                "hpo_id": hpo_id,
                "name": term_name,  # Use the name we found in the evidence
                "score": max_score,
                "reranker_score": max_reranker_score,
                "confidence": confidence_score,
                "evidence_count": evidence_count,
                "status": overall_status,
                "affirmed_count": affirmed_count,
                "negated_count": negated_count,
                "normal_count": normal_count,
                "uncertain_count": uncertain_count,
                "evidence": evidence_list,
            }
        )

    # Sort by confidence score (descending)
    logger.info("Sorting aggregated results by confidence score")
    aggregated_results.sort(key=lambda x: x["confidence"], reverse=True)

    # Apply min_confidence filtering if specified
    if min_confidence_for_aggregated > 0.0:
        logger.info(f"Applying min_confidence filter: {min_confidence_for_aggregated}")
        filtered_count = len(aggregated_results)
        aggregated_results = [
            result
            for result in aggregated_results
            if result["confidence"] >= min_confidence_for_aggregated
        ]
        filtered_count -= len(aggregated_results)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} results below threshold")

    logger.info(
        f"HPO extraction completed: {len(aggregated_results)} terms across "
        f"{len(processed_chunks)} chunks"
    )
    return aggregated_results, chunk_results, processed_chunks
