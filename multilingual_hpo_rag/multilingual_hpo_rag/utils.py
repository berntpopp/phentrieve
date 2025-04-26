"""
Utility functions for the multilingual HPO RAG package.

This module contains utility functions used throughout the multilingual_hpo_rag
package for tasks like file handling, string processing, and configuration.
"""

import re
from typing import Dict, Optional

from multilingual_hpo_rag.config import INDEX_DIR


def get_embedding_dimension(model_name: str) -> int:
    """
    Get the embedding dimension for a given model.

    Different models produce embeddings with different dimensions.

    Args:
        model_name: String representing the sentence-transformer model name

    Returns:
        The embedding dimension as an integer
    """
    # Models with non-standard dimensions
    dimension_map: Dict[str, int] = {
        "sentence-transformers/distiluse-base-multilingual-cased-v2": 512,
        "BAAI/bge-m3": 1024,  # BGE-M3 uses 1024-dimensional embeddings
        "sentence-transformers/LaBSE": 768,  # LaBSE uses 768-dimensional embeddings
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,  # MiniLM uses 384-dimensional embeddings
        # "Alibaba-NLP/gte-multilingual-base": 768,  # GTE uses standard 768-dimensional embeddings (covered by default)
    }

    # Default dimension for most sentence transformer models
    return dimension_map.get(model_name, 768)


def get_model_slug(model_name: str) -> str:
    """
    Create a simple, filesystem-safe string (slug) from the model name.

    Args:
        model_name: String representing the sentence-transformer model name
            (e.g., 'sentence-transformers/model-name')

    Returns:
        Filesystem-safe string
    """
    # Extract the last part of the model path if it contains slashes
    if "/" in model_name:
        slug = model_name.split("/")[-1]
    else:
        slug = model_name

    # Replace any non-alphanumeric characters with underscores
    slug = re.sub(r"[^a-zA-Z0-9]", "_", slug).lower()

    # Remove any duplicate underscores
    slug = re.sub(r"_+", "_", slug)

    # Remove leading/trailing underscores
    slug = slug.strip("_")

    return slug


def get_index_dir() -> str:
    """
    Return the path for the ChromaDB persistent storage.

    Returns:
        Directory path string
    """
    return INDEX_DIR


def generate_collection_name(model_name: str) -> str:
    """
    Generate a unique collection name based on the model slug.

    Args:
        model_name: Model name string

    Returns:
        Collection name string
    """
    return f"hpo_multilingual_{get_model_slug(model_name)}"


def calculate_similarity(distance: float) -> float:
    """
    Convert cosine distance to similarity score.

    Args:
        distance: Cosine distance (0 to 2) from ChromaDB

    Returns:
        Similarity score (0 to 1)
    """
    # Cosine distance = 1 - Cosine Similarity
    # Similarity = 1 - Cosine Distance
    similarity = 1.0 - distance
    # Clamp the result between 0.0 and 1.0 as similarity scores typically range from 0 to 1
    # (though cosine similarity technically ranges from -1 to 1, negative values are unlikely here)
    return max(0.0, min(1.0, similarity))
