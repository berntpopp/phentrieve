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

        logger.info(f"Loading cross-encoder model '{model_name}' on {device}")

        # Load the cross-encoder model
        model: CrossEncoder = CrossEncoder(model_name, device=device)
        logger.info(f"Successfully loaded cross-encoder model: {model_name}")
        return model

    except Exception as e:
        logger.error(f"Failed to load cross-encoder model '{model_name}': {str(e)}")
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
                # For NLI models that return probabilities for entailment/neutral/contradiction
                # Use the entailment score (usually the first index) as the relevance score
                candidate["cross_encoder_score"] = float(scores[i][0])
            else:
                # For traditional cross-encoders that return a single score
                candidate["cross_encoder_score"] = float(scores[i])

        # Sort by cross_encoder_score in descending order
        reranked_candidates = sorted(
            candidates, key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True
        )

        logger.debug(f"Re-ranked {len(candidates)} candidates using cross-encoder")
        return reranked_candidates

    except Exception as e:
        logger.error(f"Error during cross-encoder re-ranking: {str(e)}")
        logger.warning("Falling back to original candidate order")
        return candidates
