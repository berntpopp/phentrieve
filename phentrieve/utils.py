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
import sys
from pathlib import Path
from typing import Callable, Optional

import yaml

try:
    from langdetect import LangDetectException, detect

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_logging_cli(debug: bool = False):
    """Configure logging for CLI commands."""
    level = logging.DEBUG if debug else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Optional: Silence overly verbose libraries if needed
    # logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.info(f"Logging level set to {'DEBUG' if debug else 'INFO'}")


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
    dimension_map: dict[str, int] = {
        "sentence-transformers/distiluse-base-multilingual-cased-v2": 512,
        "BAAI/bge-m3": 1024,  # BGE-M3 uses 1024-dimensional embeddings
        "sentence-transformers/LaBSE": 768,  # LaBSE uses 768-dimensional embeddings
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        # "Alibaba-NLP/gte-multilingual-base": 768,  # Standard dimensions
    }

    # Default dimension for most sentence transformer models
    return dimension_map.get(model_name, 768)


def get_model_slug(model_name: Optional[str]) -> str:
    """
    Create a simple, filesystem-safe string (slug) from the model name.

    Args:
        model_name: String representing the sentence-transformer model name
            (e.g., 'sentence-transformers/model-name')

    Returns:
        Filesystem-safe string
    """
    # Handle None case
    if model_name is None:
        return "biolord_2023_m"

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


def get_user_config_dir() -> Path:
    """Gets the path to the user-specific config/data directory."""
    return Path.home() / ".phentrieve"


def get_config_paths() -> list[Path]:
    """Gets all potential configuration file paths in priority order."""
    # Local project directory - highest priority
    local_config = Path.cwd() / "phentrieve.yaml"
    local_config_alt = Path.cwd() / "phentrieve.yml"  # Alternative extension

    # User home directory - secondary priority
    user_config = get_user_config_dir() / "phentrieve.yaml"
    user_config_alt = (
        get_user_config_dir() / "config.yaml"
    )  # For backward compatibility

    # Return in priority order
    return [local_config, local_config_alt, user_config, user_config_alt]


def get_config_file_path() -> Optional[Path]:
    """Gets the first existing config file path from the priority list.

    Returns:
        Path to existing config file, or None if no config file exists
    """
    for path in get_config_paths():
        if path.exists():
            return path
    return None


def get_default_data_dir() -> Path:
    """Gets the default path for storing HPO data.

    Prioritizes environment variables in this order:
    1. PHENTRIEVE_DATA_DIR - specific data directory
    2. PHENTRIEVE_DATA_ROOT_DIR/hpo_core_data - subfolder of root data directory
    3. User home directory fallback (~/.phentrieve/data)
    """
    # Check for specific data directory environment variable
    env_specific = os.getenv("PHENTRIEVE_DATA_DIR")
    if env_specific:
        return Path(env_specific)

    # Check for root data directory and add subfolder
    env_root = os.getenv("PHENTRIEVE_DATA_ROOT_DIR")
    if env_root:
        return Path(env_root) / "hpo_core_data"

    # Fallback to user config directory
    return get_user_config_dir() / "data"


def get_default_index_dir() -> Path:
    """Gets the default path for storing ChromaDB indexes.

    Prioritizes environment variables in this order:
    1. PHENTRIEVE_INDEX_DIR - specific index directory
    2. PHENTRIEVE_DATA_ROOT_DIR/indexes - subfolder of root data directory
    3. User home directory fallback (~/.phentrieve/hpo_chroma_index)
    """
    # Check for specific index directory environment variable
    env_specific = os.getenv("PHENTRIEVE_INDEX_DIR")
    if env_specific:
        return Path(env_specific)

    # Check for root data directory and add subfolder
    env_root = os.getenv("PHENTRIEVE_DATA_ROOT_DIR")
    if env_root:
        return Path(env_root) / "indexes"

    # Fallback to user config directory
    return get_user_config_dir() / "hpo_chroma_index"


def get_default_results_dir() -> Path:
    """Gets the default path for storing benchmark results.

    Prioritizes environment variables in this order:
    1. PHENTRIEVE_RESULTS_DIR - specific results directory
    2. PHENTRIEVE_DATA_ROOT_DIR/results - subfolder of root data directory
    3. User home directory fallback (~/.phentrieve/benchmark_results)
    """
    # Check for specific results directory environment variable
    env_specific = os.getenv("PHENTRIEVE_RESULTS_DIR")
    if env_specific:
        return Path(env_specific)

    # Check for root data directory and add subfolder
    env_root = os.getenv("PHENTRIEVE_DATA_ROOT_DIR")
    if env_root:
        return Path(env_root) / "results"

    # Fallback to user config directory
    return get_user_config_dir() / "benchmark_results"


def get_index_dir() -> Path:
    """Return the path for the ChromaDB persistent storage.

    Note: This is maintained for backward compatibility.
    New code should use resolve_data_path with get_default_index_dir.

    Returns:
        Directory path
    """
    # Create the directory if it doesn't exist
    index_dir = get_default_index_dir()
    os.makedirs(index_dir, exist_ok=True)
    return index_dir


@functools.lru_cache(maxsize=1)  # Cache the config for a single run
def load_user_config() -> dict:
    """Loads the user configuration from the YAML file."""
    config_path = get_config_file_path()
    config = {}
    if config_path and config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                logging.warning(
                    f"Config file {config_path} is not a valid dictionary. "
                    f"Using defaults."
                )
                config = {}
            logging.info(f"Loaded user configuration from {config_path}")
        except Exception as e:
            logging.warning(
                f"Could not load user config file {config_path}: {e}. Using defaults."
            )
            config = {}
    else:
        logging.debug(f"User config file not found at {config_path}. Using defaults.")
    return config if config else {}  # Ensure returning dict


def resolve_data_path(
    cli_path: Optional[str] = None,
    config_key: Optional[str] = None,
    default_func: Optional[Callable] = None,
) -> Path:
    """Resolves data paths based on priority: CLI > User Config > Default."""
    user_config = load_user_config()

    # 1. CLI Argument
    if cli_path:
        path = Path(cli_path).expanduser().resolve()
        logging.debug(f"Using path from CLI arg '{cli_path}': {path}")
        os.makedirs(path, exist_ok=True)  # Ensure dir exists if specified via CLI
        return path

    # 2. User Config File
    if config_key and config_key in user_config:
        path_str = user_config[config_key]
        if isinstance(path_str, str):
            path = Path(path_str).expanduser().resolve()
            logging.debug(f"Using path from config file key '{config_key}': {path}")
            os.makedirs(path, exist_ok=True)  # Ensure dir exists via config
            return path
        else:
            logging.warning(
                f"Invalid path value '{path_str}' for key '{config_key}' "
                f"in config file. Using default."
            )

    # 3. Default Function
    if default_func:
        result = default_func()
        path = (
            Path(result).resolve() if result else Path.cwd()
        )  # Call function to get Path object
        logging.debug(
            f"Using default path from function {default_func.__name__}: {path}"
        )
        # Ensure default directories exist
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            logging.error(f"Could not create default directory {path}: {e}")
            raise OSError(f"Failed to establish default data directory: {path}") from e
        return path

    # Fallback
    logging.error(
        "Could not resolve data path: No CLI arg, config key, "
        "or default function provided."
    )
    raise ValueError("Unable to resolve data path")


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
def normalize_id(term_id: str) -> str:
    """
    Normalize HPO term ID to the standard format (HP:NNNNNNN).

    This handles different ID formats found in HPO data:
    - URIs like http://purl.obolibrary.org/obo/HP_0000001
    - Already normalized HP:0000001 format

    Args:
        term_id: An HPO term ID in any format

    Returns:
        Normalized ID in HP:NNNNNNN format
    """
    # Handle URI format (HP_0000001)
    if "obo/HP_" in term_id:
        hp_number = term_id.split("HP_")[1]
        return f"HP:{hp_number}"
    # Handle colon format (already normalized)
    elif term_id.startswith("HP:"):
        return term_id
    # Return as-is for other formats (which may not be valid HPO IDs)
    return term_id


def load_translation_text(hpo_id: str, translation_dir: str) -> Optional[str]:
    """
    Load translation text for a given HPO ID from a JSON file.

    The function extracts the translated label and synonyms in the target language
    (not the original English terms).

    Args:
        hpo_id: HPO ID (e.g., "HP:0004241")
        translation_dir: Path to directory containing translation JSON files

    Returns:
        Formatted translation text or None if not found/error
        Format: "[Translated Label]. Synonyms: [Synonym 1]; [Synonym 2]; ..."
    """
    # Convert HPO ID format from HP:NNNNNNN to HP_NNNNNNN for filename
    file_id = hpo_id.replace(":", "_")
    json_path = os.path.join(translation_dir, f"{file_id}.json")

    if not os.path.exists(json_path):
        logger.warning(f"Translation file not found for {hpo_id} at {json_path}")
        return None

    try:
        with open(json_path, encoding="utf-8") as f:
            translation_data = json.load(f)

        # Extract translated label (required field)
        translated_label = str(translation_data.get("lbl", ""))
        if not translated_label:
            logger.warning(f"Missing label in translation for {hpo_id}")
            return None

        # Get translated synonyms if available (optional field)
        translated_synonyms: list[str] = []
        if "meta" in translation_data and "synonyms" in translation_data["meta"]:
            # Extract ONLY the translated synonym values ("val" field)
            for syn in translation_data["meta"]["synonyms"]:
                if "val" in syn:
                    translated_synonyms.append(str(syn["val"]))

        # Construct the combined translation text
        result: str = translated_label
        if translated_synonyms:
            # Add synonyms separated by semicolons
            synonyms_text = "; ".join(translated_synonyms)
            result += f". Synonyms: {synonyms_text}"

        return result

    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Error loading translation for {hpo_id}: {str(e)}")
        return None


# Alias for backward compatibility
def load_german_translation_text(hpo_id: str, translation_dir: str) -> Optional[str]:
    """
    Legacy function for backward compatibility. Use load_translation_text instead.
    """
    return load_translation_text(hpo_id, translation_dir)


def detect_language(text: str, default_lang: str = "en") -> str:
    """
    Detect the language of a given text string.

    Args:
        text: The text to analyze for language detection
        default_lang: Fallback language code if detection fails or is unavailable

    Returns:
        ISO 639-1 language code (e.g., 'en', 'de', 'fr', etc.)

    Note:
        Requires langdetect package to be installed. If not available, returns default_lang.
        For very short texts or ambiguous content, detection may not be reliable.
    """
    # Check if langdetect is available
    if not LANGDETECT_AVAILABLE:
        logger.warning(
            "langdetect package not installed. Defaulting to '" + default_lang + "'. "
            "Install with 'pip install langdetect' for automatic language detection."
        )
        return default_lang

    # Text must be long enough for reliable detection
    if not text or len(text.strip()) < 20:
        logger.info(
            f"Text too short for reliable language detection, using default language: {default_lang}"
        )
        return default_lang

    try:
        # Detect language
        detected = str(detect(text))
        logger.info(f"Detected language: {detected}")
        return detected
    except LangDetectException as e:
        logger.warning(
            f"Language detection failed: {str(e)}. Using default: {default_lang}"
        )
        return default_lang
