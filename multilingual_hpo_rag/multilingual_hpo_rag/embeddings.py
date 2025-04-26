"""
Embedding model handling for the multilingual HPO RAG package.

This module provides functionality for loading and managing embedding models
used for encoding text into vector representations.
"""

import logging
from typing import Optional, Union

import torch
from sentence_transformers import SentenceTransformer

from multilingual_hpo_rag.config import JINA_MODEL_ID, DEFAULT_BIOLORD_MODEL


def load_embedding_model(
    model_name: str, trust_remote_code: bool = False, device: Optional[str] = None
) -> SentenceTransformer:
    """
    Load a sentence transformer embedding model with support for GPU acceleration.

    Args:
        model_name: Name or path of the sentence transformer model to load
        trust_remote_code: Whether to trust remote code (needed for some models like Jina)
        device: Device to load the model on ('cuda' or 'cpu'). If None, will use CUDA if available

    Returns:
        Loaded SentenceTransformer model instance

    Raises:
        ValueError: If the model couldn't be loaded
    """
    # Set device - use CUDA if available and not explicitly set to CPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Loading embedding model: {model_name} on {device}")

    try:
        # Special handling for Jina model which requires trust_remote_code=True
        if model_name == JINA_MODEL_ID:
            logging.info(
                f"Loading Jina model '{model_name}' with trust_remote_code=True on {device}"
            )
            # Security note: Only use trust_remote_code=True for trusted sources
            model = SentenceTransformer(model_name, trust_remote_code=True)
        # Special handling for BioLORD model which may require special authentication
        elif model_name == DEFAULT_BIOLORD_MODEL or "BioLORD" in model_name:
            logging.info(
                f"Loading BioLORD model '{model_name}' with trust_remote_code=True on {device}"
            )
            # BioLORD models often require trust_remote_code for custom layers
            model = SentenceTransformer(model_name, trust_remote_code=True)
        # Handle other models that might require trust_remote_code
        elif trust_remote_code:
            logging.info(
                f"Loading model '{model_name}' with trust_remote_code=True on {device}"
            )
            model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            logging.info(f"Loading model '{model_name}' on {device}")
            model = SentenceTransformer(model_name)

        # Move model to specified device
        model = model.to(device)
        logging.info(f"Successfully loaded model {model_name} on {device}")
        return model

    except Exception as e:
        error_msg = f"Error loading SentenceTransformer model '{model_name}': {e}"
        logging.error(error_msg)
        logging.error("Make sure you have run: pip install -r requirements.txt")
        raise ValueError(error_msg) from e
