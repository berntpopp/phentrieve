"""
Cross-encoder re-ranking module for Phentrieve.

This module provides functionality to re-rank initial retrieval results
using a cross-encoder model, which directly compares query-document pairs
to provide more accurate relevance scores.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)


def load_cross_encoder(
    model_name: str, device: Optional[str] = None
) -> Optional[CrossEncoder]:
    """
    Load a cross-encoder model for re-ranking.

    Args:
        model_name: Name of the cross-encoder model to load (HuggingFace model ID)
        device: Device to load the model on ('cpu', 'cuda', 'cuda:0', etc.)
                If None, will use CUDA if available, otherwise CPU

    Returns:
        Loaded CrossEncoder model or None if loading fails

    Note:
        The function handles the model loading gracefully and returns None on failure,
        allowing the calling code to fall back to dense retrieval only.
    """
    try:
        # Determine device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "Loading cross-encoder model '%s' on %s",
            _sanitize(model_name),
            _sanitize(device),
        )

        # Load the cross-encoder model
        model: CrossEncoder = CrossEncoder(model_name, device=device)
        logger.info(
            "Successfully loaded cross-encoder model: %s", _sanitize(model_name)
        )
        return model

    except Exception as e:
        logger.error(
            "Failed to load cross-encoder model '%s': %s",
            _sanitize(model_name),
            _sanitize(e),
        )
        logger.warning("Re-ranking will be disabled")
        return None


def rerank_with_cross_encoder(
    query: str, candidates: list[dict[str, Any]], cross_encoder_model: CrossEncoder
) -> list[dict[str, Any]]:
    """
    Re-rank retrieval candidates using a cross-encoder model.

    Args:
        query: Original query string (in source language)
        candidates: List of candidate dictionaries with at least 'english_doc' field
        cross_encoder_model: Loaded CrossEncoder model

    Returns:
        List of candidates with added 'cross_encoder_score' field, sorted by this score

    Note:
        - Each candidate must contain an 'english_doc' field with the document text
        - The function adds a 'cross_encoder_score' field to each candidate
        - Returned list is sorted by cross_encoder_score in descending order
        - Original ordering is preserved in case of errors
    """
    try:
        if not candidates:
            logger.warning("No candidates provided for re-ranking")
            return []

        # Prepare pairs for the cross-encoder
        # Use comparison_text if available, falling back to english_doc for backward compatibility
        pairs = [
            (query, candidate.get("comparison_text", candidate.get("english_doc", "")))
            for candidate in candidates
        ]

        # Get scores from the cross-encoder
        scores = cross_encoder_model.predict(pairs)

        # Add scores to candidates
        for i, candidate in enumerate(candidates):
            # Handle different output formats from various cross-encoder models
            if isinstance(scores[i], (list, np.ndarray)) and len(scores[i]) > 1:
                # For models returning array outputs (e.g., multi-class classifiers)
                # Use the first score as the relevance score
                candidate["cross_encoder_score"] = float(scores[i][0])
            else:
                # For standard cross-encoders that return a single relevance score
                candidate["cross_encoder_score"] = float(scores[i])

        # Sort by cross_encoder_score in descending order
        reranked_candidates = sorted(
            candidates, key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True
        )

        logger.debug(
            "Re-ranked %s candidates using cross-encoder", _sanitize(len(candidates))
        )
        return reranked_candidates

    except Exception as e:
        logger.error("Error during cross-encoder re-ranking: %s", _sanitize(e))
        logger.warning("Falling back to original candidate order")
        return candidates


def protected_dense_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    cross_encoder_model: CrossEncoder,
    trust_threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Protected two-stage retrieval that preserves high-confidence dense retrieval matches.

    This implements a research-backed approach for cross-lingual medical retrieval:
    - Stage 1 (Dense Retrieval): BioLORD provides high-recall semantic matching
    - Protection: High-confidence matches (≥trust_threshold) are protected from demotion
    - Stage 2 (Cross-Encoder): Refines only uncertain mid-tier results
    - Merge: Protected results stay at top, reranked results fill below

    This addresses the problem where cross-encoders can demote correct cross-lingual
    matches due to lexical bias (e.g., promoting "Bladder stones" over "Nephrolithiasis"
    for German query "Steine der Niere").

    References:
        - BioLORD-2023: Designed as DPR model for biomedical RAG pipelines
        - Multistage BiCross: Multilingual medical information access (PMC8423231)
        - Two-stage retrieval: Bi-encoder for recall, cross-encoder for precision

    Args:
        query: Original query string (in source language)
        candidates: List of candidate dictionaries with 'bi_encoder_score' field
        cross_encoder_model: Loaded CrossEncoder model
        trust_threshold: Minimum bi_encoder_score to protect from demotion (default: 0.7)

    Returns:
        List of candidates with 'cross_encoder_score' added and optimal ordering:
        - Protected high-confidence dense matches at top (preserved order)
        - Reranked uncertain matches below (sorted by cross_encoder_score)

    Example:
        Query: "Steine der Niere" (German)

        Before protection:
        1. Urolithiasis (dense: 0.91) → Cross-encoder demotes to #10
        2. Bladder stones (dense: 0.45) → Cross-encoder promotes to #1

        After protection:
        1. Urolithiasis (dense: 0.91) ← PROTECTED (above 0.7 threshold)
        2. [Other reranked results...]

    Note:
        - Candidates must have 'bi_encoder_score' field from dense retrieval
        - Protected candidates retain original dense retrieval ordering
        - Cross-encoder scores are still added for all candidates
        - Protection threshold should be tuned based on dense retriever performance
    """
    try:
        if not candidates:
            logger.warning("No candidates provided for protected re-ranking")
            return []

        # ============ STAGE 1: Identify Protected Candidates ============
        # High-confidence dense matches are protected from cross-encoder demotion
        protected_candidates = [
            c for c in candidates if c.get("bi_encoder_score", 0.0) >= trust_threshold
        ]

        # Lower-confidence candidates can be reranked
        rerank_candidates = [
            c for c in candidates if c.get("bi_encoder_score", 0.0) < trust_threshold
        ]

        logger.info(
            "Protected dense retrieval: %s candidates above threshold %.2f, %s candidates for re-ranking",
            _sanitize(len(protected_candidates)),
            _sanitize(trust_threshold),
            _sanitize(len(rerank_candidates)),
        )

        # ============ STAGE 2: Cross-Encoder Refinement ============
        # ONLY rerank the uncertain mid-tier candidates
        if rerank_candidates:
            # Prepare pairs for the cross-encoder
            pairs = [
                (
                    query,
                    candidate.get("comparison_text", candidate.get("english_doc", "")),
                )
                for candidate in rerank_candidates
            ]

            # Get scores from the cross-encoder
            scores = cross_encoder_model.predict(pairs)

            # Add scores to rerank candidates
            for i, candidate in enumerate(rerank_candidates):
                # Handle different output formats from various cross-encoder models
                if isinstance(scores[i], (list, np.ndarray)) and len(scores[i]) > 1:
                    # For models returning array outputs: use first score
                    candidate["cross_encoder_score"] = float(scores[i][0])
                else:
                    # For standard cross-encoders: single relevance score
                    candidate["cross_encoder_score"] = float(scores[i])

            # Sort reranked candidates by cross-encoder score
            rerank_candidates.sort(
                key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True
            )

            logger.debug(
                "Re-ranked %s uncertain candidates", _sanitize(len(rerank_candidates))
            )

        # Also add cross-encoder scores to protected candidates for transparency
        if protected_candidates:
            pairs = [
                (
                    query,
                    candidate.get("comparison_text", candidate.get("english_doc", "")),
                )
                for candidate in protected_candidates
            ]
            scores = cross_encoder_model.predict(pairs)

            for i, candidate in enumerate(protected_candidates):
                # Handle different output formats from various cross-encoder models
                if isinstance(scores[i], (list, np.ndarray)) and len(scores[i]) > 1:
                    candidate["cross_encoder_score"] = float(scores[i][0])
                else:
                    candidate["cross_encoder_score"] = float(scores[i])

        # ============ STAGE 3: Merge with Protection ============
        # Protected results stay at top (preserve dense retrieval order)
        # Reranked results fill in below (sorted by cross-encoder)
        final_candidates = protected_candidates + rerank_candidates

        logger.info(
            "Protected re-ranking complete: %s protected, %s reranked",
            _sanitize(len(protected_candidates)),
            _sanitize(len(rerank_candidates)),
        )

        return final_candidates

    except Exception as e:
        logger.error("Error during protected re-ranking: %s", _sanitize(e))
        logger.warning("Falling back to original candidate order")
        return candidates
