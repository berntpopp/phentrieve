"""
Utility functions for the Phentrieve package.

This module contains utility functions used throughout the phentrieve
package for tasks like file handling, string processing, and configuration.
"""

import functools
import json
import logging
import os
import re
from typing import Dict, Optional

from phentrieve.config import INDEX_DIR

logger = logging.getLogger(__name__)


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
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,  # Mini
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
    return f"phentrieve_{get_model_slug(model_name)}"


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
    # (though cosine similarity ranges from -1 to 1, negative values are unlikely here)
    return max(0.0, min(1.0, similarity))


@functools.lru_cache(maxsize=512)
def load_german_translation_text(hpo_id: str, translation_dir: str) -> Optional[str]:
    """
    Load German translation text for a given HPO ID from a JSON file.

    The function extracts ONLY the German label and synonyms
    (not the original English terms).

    Args:
        hpo_id: HPO ID (e.g., "HP:0004241")
        translation_dir: Path to directory containing translation JSON files

    Returns:
        Formatted German translation text or None if not found/error
        Format: "[German Label]. Synonyms: [German Syn1]; [German Syn2]; ..."
    """
    # Convert HPO ID format from HP:NNNNNNN to HP_NNNNNNN for filename
    file_id = hpo_id.replace(":", "_")
    json_path = os.path.join(translation_dir, f"{file_id}.json")

    if not os.path.exists(json_path):
        logger.warning(f"Translation file not found for {hpo_id} at {json_path}")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            translation_data = json.load(f)

        # Extract German label (required field)
        german_label = translation_data.get("lbl", "")
        if not german_label:
            logger.warning(f"Missing German label in translation for {hpo_id}")
            return None

        # Get German synonyms if available (optional field)
        german_synonyms = []
        if "meta" in translation_data and "synonyms" in translation_data["meta"]:
            # Extract ONLY the German synonym values ("val" field)
            for syn in translation_data["meta"]["synonyms"]:
                if "val" in syn:
                    german_synonyms.append(syn["val"])

        # Construct the combined German text
        result = german_label
        if german_synonyms:
            # Add synonyms separated by semicolons
            synonyms_text = "; ".join(german_synonyms)
            result += f". Synonyms: {synonyms_text}"

        return result

    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading translation for {hpo_id}: {str(e)}")
        return None
