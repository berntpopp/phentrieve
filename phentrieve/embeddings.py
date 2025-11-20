"""
Embedding model handling for the Phentrieve package.

This module provides functionality for loading and managing embedding models
used for encoding text into vector representations. It implements a thread-safe
singleton registry to prevent loading the same model into VRAM multiple times.

Thread Safety:
    All functions in this module are thread-safe. Concurrent calls to
    load_embedding_model() with the same model name will return the same
    instance without loading the model multiple times.

Memory Management:
    Models are cached globally within the Python process. Use
    clear_model_registry() to free memory when models are no longer needed.

Examples:
    >>> from phentrieve.embeddings import load_embedding_model
    >>>
    >>> # Load model (cached for reuse)
    >>> model = load_embedding_model("all-MiniLM-L6-v2")
    >>>
    >>> # Second call returns cached instance (fast!)
    >>> same_model = load_embedding_model("all-MiniLM-L6-v2")
    >>> assert model is same_model  # Same object in memory
    >>>
    >>> # Force reload if needed
    >>> fresh_model = load_embedding_model("all-MiniLM-L6-v2", force_reload=True)
    >>>
    >>> # Clear cache to free memory
    >>> from phentrieve.embeddings import clear_model_registry
    >>> clear_model_registry()
"""

import logging
import threading
import warnings
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from phentrieve.config import DEFAULT_BIOLORD_MODEL, JINA_MODEL_ID

# Configure logger
logger = logging.getLogger(__name__)

# Catch and log NVML warning nicely
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    # Trigger any torch.cuda initialization warnings
    _ = torch.cuda.is_available() if hasattr(torch, "cuda") else False

    # Log NVML warnings nicely
    for warning in caught_warnings:
        if "NVML" in str(warning.message):
            logger.debug(
                "PyTorch CUDA/NVML initialization: GPU monitoring unavailable. "
                "Using CPU device."
            )

# Thread-safe model registry
# Key: model_name (str), Value: SentenceTransformer instance
_MODEL_REGISTRY: dict[str, SentenceTransformer] = {}
_REGISTRY_LOCK = threading.Lock()


def _devices_match(device1: str, device2: str) -> bool:
    """
    Compare two device strings for equality using torch.device normalization.

    This handles cases like 'cuda' vs 'cuda:0' properly by normalizing
    both to torch.device objects before comparison.

    Args:
        device1: First device string (e.g., 'cuda', 'cpu', 'cuda:0')
        device2: Second device string (e.g., 'cuda', 'cpu', 'cuda:0')

    Returns:
        True if devices are equivalent, False otherwise
    """
    try:
        return torch.device(device1) == torch.device(device2)
    except RuntimeError:
        # Fallback to string comparison if torch.device() fails
        return str(device1) == str(device2)


def load_embedding_model(
    model_name: Optional[str] = None,
    trust_remote_code: bool = False,
    device: Optional[str] = None,
    force_reload: bool = False,
) -> SentenceTransformer:
    """
    Load a sentence transformer embedding model with caching and GPU support.

    This function implements a thread-safe singleton pattern to avoid loading
    the same model into memory multiple times. The first call loads the model,
    subsequent calls with the same model_name return the cached instance.

    Args:
        model_name: Name or path of the sentence transformer model to load.
            If None, uses DEFAULT_BIOLORD_MODEL.
        trust_remote_code: Whether to trust remote code (needed for some models like Jina).
            Automatically set to True for known models (Jina, BioLORD).
        device: Device to load the model on ('cuda', 'cpu', 'mps'). If None, auto-detects
            best available device (CUDA > MPS > CPU).
        force_reload: If True, forces a fresh load even if model is cached.
            Useful for testing or when model files have been updated.

    Returns:
        Loaded SentenceTransformer model instance. Multiple calls with the same
        model_name return the same cached instance (unless force_reload=True).

    Raises:
        ValueError: If the model couldn't be loaded

    Examples:
        >>> # First call loads model (~2-5 seconds)
        >>> model1 = load_embedding_model("all-MiniLM-L6-v2")
        >>>
        >>> # Second call returns cached instance (~instant)
        >>> model2 = load_embedding_model("all-MiniLM-L6-v2")
        >>> assert model1 is model2  # Same object
        >>>
        >>> # Force reload if needed
        >>> fresh = load_embedding_model("all-MiniLM-L6-v2", force_reload=True)

    Thread Safety:
        This function is thread-safe. Concurrent calls will properly synchronize
        to ensure only one instance is loaded per model name.
    """
    # Use default model if None is provided
    if model_name is None:
        model_name = DEFAULT_BIOLORD_MODEL
        logging.info(f"No model specified, using default model: {model_name}")

    # Determine device - auto-detect if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Fast path: Check cache without lock (double-check locking pattern)
    if not force_reload and model_name in _MODEL_REGISTRY:
        cached_model = _MODEL_REGISTRY[model_name]
        logging.debug(
            f"Returning cached embedding model: {model_name} "
            f"(current device: {cached_model.device})"
        )
        # Move to requested device if different (lightweight operation if already there)
        if not _devices_match(str(cached_model.device), device):
            logging.debug(
                f"Moving cached model {model_name} from {cached_model.device} to {device}"
            )
            cached_model = cached_model.to(device)
            # Update registry with device-moved model
            with _REGISTRY_LOCK:
                _MODEL_REGISTRY[model_name] = cached_model
        return cached_model

    # Slow path: Acquire lock and load model
    with _REGISTRY_LOCK:
        # Re-check cache after acquiring lock (another thread might have loaded it)
        if not force_reload and model_name in _MODEL_REGISTRY:
            cached_model = _MODEL_REGISTRY[model_name]
            logging.debug(f"Returning cached embedding model (post-lock): {model_name}")
            # Move to requested device if needed
            if not _devices_match(str(cached_model.device), device):
                logging.debug(f"Moving cached model {model_name} to {device}")
                cached_model = cached_model.to(device)
                _MODEL_REGISTRY[model_name] = cached_model
            return cached_model

        # No cached version available - load fresh
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

            # Store in registry for future reuse
            _MODEL_REGISTRY[model_name] = model
            logging.info(
                f"Successfully loaded and cached model {model_name} on {device}"
            )

            return model

        except Exception as e:
            error_msg = f"Error loading SentenceTransformer model '{model_name}': {e}"
            logging.error(error_msg)
            logging.error("Make sure you have run: pip install -r requirements.txt")
            raise ValueError(error_msg) from e


def clear_model_registry() -> None:
    """
    Clear the model registry to free up VRAM/RAM.

    This function removes all cached models from memory and attempts to
    free GPU memory if CUDA is available. Useful for:
    - Testing: Ensuring fresh model loads
    - Long-running processes: Freeing memory when models are no longer needed
    - Model switching: Clearing old models before loading different ones

    Thread Safety:
        This function is thread-safe and will properly synchronize with
        concurrent load_embedding_model() calls.

    Examples:
        >>> from phentrieve.embeddings import load_embedding_model, clear_model_registry
        >>>
        >>> # Load and cache a model
        >>> model = load_embedding_model("all-MiniLM-L6-v2")
        >>>
        >>> # Clear cache to free memory
        >>> clear_model_registry()
        >>>
        >>> # Next load will be fresh (not from cache)
        >>> fresh_model = load_embedding_model("all-MiniLM-L6-v2")

    Note:
        After calling this function, subsequent calls to load_embedding_model()
        will need to reload models from disk, which takes longer.
    """
    with _REGISTRY_LOCK:
        num_models = len(_MODEL_REGISTRY)
        _MODEL_REGISTRY.clear()
        logging.info(f"Model registry cleared ({num_models} model(s) removed)")

    # Attempt to free CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.debug("GPU memory cache emptied")


def get_cached_models() -> list[str]:
    """
    Get a list of currently cached model names.

    Returns:
        List of model names that are currently cached in memory.

    Examples:
        >>> from phentrieve.embeddings import load_embedding_model, get_cached_models
        >>>
        >>> load_embedding_model("model-a")
        >>> load_embedding_model("model-b")
        >>>
        >>> cached = get_cached_models()
        >>> print(cached)  # ['model-a', 'model-b']
    """
    with _REGISTRY_LOCK:
        return list(_MODEL_REGISTRY.keys())
