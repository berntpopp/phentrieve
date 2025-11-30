"""
Central configuration for the Phentrieve package.

This module contains constants, defaults, and configuration parameters used
throughout the phentrieve package.

Configuration values are loaded from phentrieve.yaml if present, otherwise
fall back to the defaults defined in this module.
"""

import copy
import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import chromadb

# Public API exports - these are imported by other modules
__all__ = [
    # Directory and file paths
    "DEFAULT_HPO_TERMS_SUBDIR",
    "BENCHMARK_DATA_DIR",
    "DEFAULT_BENCHMARK_FILE",
    "DEFAULT_HPO_FILENAME",
    "DEFAULT_HPO_DB_FILENAME",
    # Benchmark result subdirectories
    "DEFAULT_SUMMARIES_SUBDIR",
    "DEFAULT_DETAILED_SUBDIR",
    "DEFAULT_VISUALIZATIONS_SUBDIR",
    # Models
    "DEFAULT_MODEL",
    "DEFAULT_BIOLORD_MODEL",
    "JINA_MODEL_ID",
    "BENCHMARK_MODELS",
    # Retrieval settings
    "DEFAULT_TOP_K",
    "DEFAULT_K_VALUES",
    "MIN_SIMILARITY_THRESHOLD",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_SIMILARITY_FORMULA",
    # Reranker settings
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RERANK_CANDIDATE_COUNT",
    "DEFAULT_ENABLE_RERANKER",
    "DEFAULT_DENSE_TRUST_THRESHOLD",
    # Device and language
    "DEFAULT_DEVICE",
    "DEFAULT_LANGUAGE",
    # HPO ontology
    "PHENOTYPE_ROOT",
    # HPO data configuration
    "HPO_VERSION",
    "HPO_BASE_URL",
    "HPO_DOWNLOAD_TIMEOUT",
    "HPO_CHUNK_SIZE",
    # Chunking configurations
    "DEFAULT_CHUNK_PIPELINE_CONFIG",
    "SIMPLE_CHUNKING_CONFIG",
    "SEMANTIC_CHUNKING_CONFIG",
    "SLIDING_WINDOW_CONFIG",
    # Assertion detection
    "DEFAULT_ASSERTION_CONFIG",
    # Helper functions
    "get_config_value",
]

# Default directory sub-paths and filenames (relative to base dirs)
# Sub-directories (for data_dir)
DEFAULT_HPO_TERMS_SUBDIR = "hpo_terms"

# Benchmark test data configuration (relative to project root)
BENCHMARK_DATA_DIR = Path("tests/data/benchmarks")
DEFAULT_BENCHMARK_FILE = "german/tiny_v1.json"

# HPO data filenames (relative to data_dir)
DEFAULT_HPO_FILENAME = "hp.json"
DEFAULT_HPO_DB_FILENAME = "hpo_data.db"  # SQLite database for HPO terms and graph data

# Benchmark result subdirectories
DEFAULT_SUMMARIES_SUBDIR = "summaries"
DEFAULT_DETAILED_SUBDIR = "detailed"
DEFAULT_VISUALIZATIONS_SUBDIR = "visualizations"

# Default models (loaded from YAML with fallbacks)
_DEFAULT_MODEL_FALLBACK = "FremyCompany/BioLORD-2023-M"
DEFAULT_BIOLORD_MODEL = "FremyCompany/BioLORD-2023-M"  # Kept for backward compatibility
JINA_MODEL_ID = "jinaai/jina-embeddings-v2-base-de"  # German embeddings model

# All models for benchmarking (loaded from YAML with fallback)
_BENCHMARK_MODELS_FALLBACK = [
    "FremyCompany/BioLORD-2023-M",  # Domain-specific biomedical model
    "jinaai/jina-embeddings-v2-base-de",  # Language-specific embeddings model (German)
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",  # Cross-lingual model (English-German),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "BAAI/bge-m3",
    "Alibaba-NLP/gte-multilingual-base",
    "sentence-transformers/LaBSE",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

# Default parameters (loaded from YAML with fallbacks)
_MIN_SIMILARITY_THRESHOLD_FALLBACK = 0.3  # Minimum similarity score to display results
_DEFAULT_SIMILARITY_THRESHOLD_FALLBACK = (
    0.1  # Default threshold for benchmark evaluations
)
_DEFAULT_TOP_K_FALLBACK = 10  # Default number of results to return
_DEFAULT_K_VALUES_FALLBACK = (1, 3, 5, 10)  # Default k values for hit rate calculation
_DEFAULT_DEVICE_FALLBACK: str | None = None  # Default device (None = auto-detect)

# Cross-encoder re-ranking settings (loaded from YAML with fallbacks)
# BAAI/bge-reranker-v2-m3: Dedicated multilingual reranker (568M parameters)
# - Fine-tuned from BGE-M3 on multilingual datasets (bge-m3-data, Quora, FEVER)
# - Supports 100+ languages for cross-lingual retrieval (XLM-RoBERTa base)
# - Used with protected two-stage retrieval to preserve dense retrieval quality
_DEFAULT_RERANKER_MODEL_FALLBACK = "BAAI/bge-reranker-v2-m3"
_DEFAULT_RERANK_CANDIDATE_COUNT_FALLBACK = 50
_DEFAULT_ENABLE_RERANKER_FALLBACK = False
# Protected dense retrieval threshold: Minimum bi-encoder score to protect from cross-encoder demotion
# This implements research-backed two-stage retrieval for cross-lingual medical queries
# - High-confidence dense matches (≥threshold) are protected from reranker demotion
# - Lower-confidence matches (<threshold) are refined by cross-encoder
# - Default 0.7 preserves BioLORD's strong cross-lingual semantic matches
# See: BioLORD-2023 RAG design, Multistage BiCross multilingual medical retrieval
_DEFAULT_DENSE_TRUST_THRESHOLD_FALLBACK = 0.7

# Root for HPO term extraction and depth calculations
PHENOTYPE_ROOT = "HP:0000118"

# Text Processing Configuration

# Predefined chunking strategies
SIMPLE_CHUNKING_CONFIG = [{"type": "paragraph"}, {"type": "sentence"}]

# Strategy: "semantic" - Uses paragraph -> sentence -> semantic splitting
SEMANTIC_CHUNKING_CONFIG = [
    {"type": "paragraph"},
    {"type": "sentence"},
    {
        "type": "sliding_window",
        "config": {
            "window_size_tokens": 2,
            "step_size_tokens": 1,
            "splitting_threshold": 0.6,
            "min_split_segment_length_words": 1,
        },
    },
]

# Strategy: "detailed" - Most granular: paragraph -> sentence -> punctuation -> semantic split
DETAILED_CHUNKING_CONFIG = [
    {"type": "paragraph"},
    {"type": "sentence"},
    {"type": "fine_grained_punctuation"},
    {
        "type": "sliding_window",
        "config": {
            "window_size_tokens": 2,
            "step_size_tokens": 1,
            "splitting_threshold": 0.6,
            "min_split_segment_length_words": 1,
        },
    },
]


# Most detailed chunking strategy using the sliding window semantic splitter
def get_sliding_window_config_with_params(
    window_size=7, step_size=1, threshold=0.5, min_segment_length=3
):
    """Get a sliding window config with custom parameters.

    Args:
        window_size: Number of tokens in each sliding window
        step_size: Number of tokens to step between windows
        threshold: Cosine similarity threshold below which to split (0-1)
        min_segment_length: Minimum number of words in a split segment

    Returns:
        Sliding window configuration with custom parameters
    """
    return [
        {"type": "paragraph"},  # First split by paragraphs
        {"type": "sentence"},  # Then split into sentences
        {
            "type": "sliding_window",  # Finally apply sliding window semantic splitting
            "config": {
                "window_size_tokens": window_size,
                "step_size_tokens": step_size,
                "splitting_threshold": threshold,
                "min_split_segment_length_words": min_segment_length,
            },
        },
    ]


# Default sliding window config
SLIDING_WINDOW_CONFIG = get_sliding_window_config_with_params()

# Default chunking pipeline configuration (using sliding window for better results)
DEFAULT_CHUNK_PIPELINE_CONFIG = SLIDING_WINDOW_CONFIG


# Functions to get fresh copies of the configs to avoid mutation issues
def get_default_chunk_pipeline_config():
    return copy.deepcopy(DEFAULT_CHUNK_PIPELINE_CONFIG)


def get_simple_chunking_config():
    return copy.deepcopy(SIMPLE_CHUNKING_CONFIG)


def get_semantic_chunking_config():
    return copy.deepcopy(SEMANTIC_CHUNKING_CONFIG)


def get_detailed_chunking_config():
    return copy.deepcopy(DETAILED_CHUNKING_CONFIG)


def get_sliding_window_config():
    return copy.deepcopy(SLIDING_WINDOW_CONFIG)


# Strategy: "sliding_window_cleaned" - Adds FinalChunkCleaner to the sliding window strategy
SLIDING_WINDOW_CLEANED_CONFIG = [
    {"type": "paragraph"},  # First split by paragraphs
    {"type": "sentence"},  # Then split into sentences
    {
        "type": "sliding_window",  # Apply sliding window semantic splitting
        "config": {
            "window_size_tokens": 7,
            "step_size_tokens": 1,
            "splitting_threshold": 0.55,
            "min_split_segment_length_words": 3,
        },
    },
    {
        "type": "final_chunk_cleaner",  # Clean up the chunks
        "config": {
            "min_cleaned_chunk_length_chars": 2,  # Minimum length of cleaned chunks in characters
            "max_cleanup_passes": 3,  # Maximum number of cleanup passes
        },
    },
]


def get_sliding_window_cleaned_config():
    return copy.deepcopy(SLIDING_WINDOW_CLEANED_CONFIG)


# Strategy: "sliding_window_punct_cleaned" - Adds FineGrainedPunctuationChunker before SlidingWindowSemanticSplitter and FinalChunkCleaner
SLIDING_WINDOW_PUNCT_CLEANED_CONFIG = [
    {"type": "paragraph"},  # First split by paragraphs
    {"type": "sentence"},  # Then split into sentences
    {"type": "fine_grained_punctuation"},  # Split by punctuation (commas, semicolons)
    {
        "type": "sliding_window",  # Apply sliding window semantic splitting
        "config": {
            "window_size_tokens": 5,  # Slightly smaller window due to pre-punctuation split
            "step_size_tokens": 1,
            "splitting_threshold": 0.45,  # Slightly lower threshold for more sensitive splits
            "min_split_segment_length_words": 2,  # Allow slightly shorter segments
        },
    },
    {
        "type": "final_chunk_cleaner",  # Clean up the chunks
        "config": {
            "min_cleaned_chunk_length_chars": 2,  # Minimum length of cleaned chunks in characters
            "max_cleanup_passes": 3,  # Maximum number of cleanup passes
        },
    },
]


def get_sliding_window_punct_cleaned_config():
    return copy.deepcopy(SLIDING_WINDOW_PUNCT_CLEANED_CONFIG)


# Strategy: "sliding_window_punct_conj_cleaned" - Adds ConjunctionChunker between punctuation and sliding window
SLIDING_WINDOW_PUNCT_CONJ_CLEANED_CONFIG = [
    {"type": "paragraph"},  # First split by paragraphs
    {"type": "sentence"},  # Then split into sentences
    {"type": "fine_grained_punctuation"},  # Split by punctuation (commas, semicolons)
    {"type": "conjunction"},  # Split at coordinating conjunctions
    {
        "type": "sliding_window",  # Apply sliding window semantic splitting
        "config": {
            "window_size_tokens": 5,  # Keep reasonable defaults after more aggressive pre-splitting
            "step_size_tokens": 1,
            "splitting_threshold": 0.50,  # Similar to punct_cleaned but may need adjustment
            "min_split_segment_length_words": 2,  # Allow shorter segments due to extra splitting
        },
    },
    {
        "type": "final_chunk_cleaner",  # Clean up the chunks
        "config": {
            "min_cleaned_chunk_length_chars": 2,  # Minimum length of cleaned chunks in characters
            "max_cleanup_passes": 3,  # Maximum number of cleanup passes
        },
    },
]


def get_sliding_window_punct_conj_cleaned_config():
    return copy.deepcopy(SLIDING_WINDOW_PUNCT_CONJ_CLEANED_CONFIG)


# Default formula for semantic similarity calculations (loaded from YAML with fallback)
_DEFAULT_SIMILARITY_FORMULA_FALLBACK = "hybrid"

# Default assertion detection configuration
DEFAULT_ASSERTION_CONFIG = {
    "enable_keyword": True,
    "enable_dependency": True,
    "preference": "dependency",
}

# Default language for text processing (loaded from YAML with fallback)
_DEFAULT_LANGUAGE_FALLBACK = "en"

# Default HPO data configuration (loaded from YAML with fallbacks)
_DEFAULT_HPO_VERSION_FALLBACK = "v2025-03-03"
_DEFAULT_HPO_BASE_URL_FALLBACK = (
    "https://github.com/obophenotype/human-phenotype-ontology/releases/download"
)
_DEFAULT_HPO_DOWNLOAD_TIMEOUT_FALLBACK = 60
_DEFAULT_HPO_CHUNK_SIZE_FALLBACK = 8192


# =============================================================================
# YAML Configuration Loading
# =============================================================================


@functools.lru_cache(maxsize=1)
def _load_yaml_config() -> dict:
    """
    Load configuration from phentrieve.yaml.

    Uses lazy import to avoid circular dependency with utils module.
    Cached to load only once per Python session.

    Returns:
        dict: Configuration dictionary from YAML, or empty dict if not found
    """
    try:
        from phentrieve.utils import load_user_config

        return load_user_config()
    except Exception as e:
        # Log configuration loading errors for debugging
        # but continue with defaults rather than failing
        import logging

        logging.getLogger(__name__).debug(
            f"Failed to load configuration: {e}. Using defaults."
        )
        return {}


def get_config_value(key: str, default: Any, nested_key: str | None = None) -> Any:
    """
    Get a configuration value from YAML config with fallback to default.

    Args:
        key: Top-level key in YAML config
        default: Default value if key not found in config
        nested_key: Optional nested key for hierarchical configs (e.g., benchmark.models)

    Returns:
        Configuration value from YAML, or default if not found

    Examples:
        >>> get_config_value("default_model", "FremyCompany/BioLORD-2023-M")
        >>> get_config_value("benchmark", {}, "similarity_threshold")
    """
    config = _load_yaml_config()

    if key not in config:
        return default

    if nested_key is None:
        return config.get(key, default)

    # Handle nested keys
    if isinstance(config[key], dict):
        return config[key].get(nested_key, default)

    return default


# =============================================================================
# Public Configuration Constants (loaded from YAML with fallbacks)
# =============================================================================
# These constants are loaded from phentrieve.yaml if present, otherwise use
# the fallback values defined above. This allows for runtime configuration
# while maintaining backward compatibility.

# Models
DEFAULT_MODEL: str = get_config_value("default_model", _DEFAULT_MODEL_FALLBACK)
_loaded_models = get_config_value("benchmark", _BENCHMARK_MODELS_FALLBACK, "models")
# Validate BENCHMARK_MODELS is a list of strings
if not isinstance(_loaded_models, list) or not all(
    isinstance(m, str) for m in _loaded_models
):
    BENCHMARK_MODELS = _BENCHMARK_MODELS_FALLBACK
else:
    BENCHMARK_MODELS = _loaded_models

# Retrieval parameters
MIN_SIMILARITY_THRESHOLD = get_config_value(
    "min_similarity_threshold", _MIN_SIMILARITY_THRESHOLD_FALLBACK
)
DEFAULT_SIMILARITY_THRESHOLD = get_config_value(
    "benchmark", _DEFAULT_SIMILARITY_THRESHOLD_FALLBACK, "similarity_threshold"
)
DEFAULT_TOP_K = get_config_value("default_top_k", _DEFAULT_TOP_K_FALLBACK)
DEFAULT_K_VALUES: tuple[int, ...] = tuple(
    get_config_value("benchmark", list(_DEFAULT_K_VALUES_FALLBACK), "k_values")
)
DEFAULT_DEVICE: str | None = get_config_value("device", _DEFAULT_DEVICE_FALLBACK)

# Re-ranking settings
DEFAULT_RERANKER_MODEL = get_config_value(
    "reranker_model", _DEFAULT_RERANKER_MODEL_FALLBACK
)
DEFAULT_RERANK_CANDIDATE_COUNT = get_config_value(
    "rerank_candidate_count", _DEFAULT_RERANK_CANDIDATE_COUNT_FALLBACK
)
DEFAULT_ENABLE_RERANKER = get_config_value(
    "enable_reranker", _DEFAULT_ENABLE_RERANKER_FALLBACK
)
DEFAULT_DENSE_TRUST_THRESHOLD = get_config_value(
    "dense_trust_threshold", _DEFAULT_DENSE_TRUST_THRESHOLD_FALLBACK
)

# Text processing
DEFAULT_SIMILARITY_FORMULA = get_config_value(
    "similarity_formula", _DEFAULT_SIMILARITY_FORMULA_FALLBACK
)
DEFAULT_LANGUAGE = get_config_value("default_language", _DEFAULT_LANGUAGE_FALLBACK)

# HPO data configuration
HPO_VERSION: str = get_config_value(
    "hpo_data", _DEFAULT_HPO_VERSION_FALLBACK, "version"
)
HPO_BASE_URL: str = get_config_value(
    "hpo_data", _DEFAULT_HPO_BASE_URL_FALLBACK, "base_url"
)
HPO_DOWNLOAD_TIMEOUT: int = int(
    get_config_value(
        "hpo_data", _DEFAULT_HPO_DOWNLOAD_TIMEOUT_FALLBACK, "download_timeout"
    )
)
HPO_CHUNK_SIZE: int = int(
    get_config_value("hpo_data", _DEFAULT_HPO_CHUNK_SIZE_FALLBACK, "chunk_size")
)


# =============================================================================
# Vector Store Configuration
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorStoreConfig:
    """
    Configuration for vector store connection and settings.

    This class centralizes vector store configuration to reduce coupling with
    ChromaDB-specific implementation details. It follows the Dependency Inversion
    Principle (SOLID) by depending on configuration rather than implementation.

    Design Principles:
    - Immutable (frozen=True): Prevents accidental modification after creation
    - Memory-efficient (slots=True): Reduces memory overhead
    - Explicit (kw_only=True): Requires keyword arguments for clarity

    Attributes:
        path: Directory path where vector store data is persisted
        collection_name: Name of the collection within the vector store
        distance_metric: Distance metric for similarity calculations (default: "cosine")
        settings: Backend-specific settings dictionary

    Examples:
        >>> config = VectorStoreConfig.for_chromadb(
        ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
        ...     index_dir=Path("/data/indexes")
        ... )
        >>> client = chromadb.PersistentClient(
        ...     path=config.path,
        ...     settings=config.to_chromadb_settings()
        ... )
    """

    path: str
    collection_name: str
    distance_metric: str = "cosine"
    settings: dict[str, Any] = field(
        default_factory=lambda: {
            "anonymized_telemetry": False,
            "allow_reset": True,
            "is_persistent": True,
        }
    )

    @classmethod
    def for_chromadb(
        cls,
        model_name: str,
        index_dir: Path,
        distance_metric: str = "cosine",
        custom_settings: dict[str, Any] | None = None,
    ) -> "VectorStoreConfig":
        """
        Create a VectorStoreConfig for ChromaDB backend.

        Factory method that encapsulates ChromaDB-specific defaults and
        collection naming conventions. This allows easy creation of configs
        without coupling client code to ChromaDB details.

        Args:
            model_name: Name of the embedding model (used for collection naming)
            index_dir: Directory where ChromaDB will store data
            distance_metric: Distance metric for similarity ("cosine", "l2", "ip")
            custom_settings: Optional settings to override defaults

        Returns:
            VectorStoreConfig instance configured for ChromaDB

        Examples:
            >>> config = VectorStoreConfig.for_chromadb(
            ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
            ...     index_dir=Path("/data/indexes")
            ... )
            >>> config.collection_name
            'hpo_sentence-transformers_all-MiniLM-L6-v2'
        """
        # Lazy import to avoid circular dependency
        from phentrieve.utils import generate_collection_name

        collection_name = generate_collection_name(model_name)

        # Default ChromaDB settings
        default_settings = {
            "anonymized_telemetry": False,
            "allow_reset": True,
            "is_persistent": True,
        }

        # Merge custom settings with defaults (custom settings override)
        final_settings = default_settings.copy()
        if custom_settings:
            final_settings.update(custom_settings)

        return cls(
            path=str(index_dir),
            collection_name=collection_name,
            distance_metric=distance_metric,
            settings=final_settings,
        )

    def to_chromadb_settings(self) -> "chromadb.Settings":
        """
        Convert configuration to ChromaDB Settings object.

        This method encapsulates ChromaDB-specific settings creation. If we
        migrate to a different vector store backend, this method can be
        replaced with an equivalent for the new backend.

        Returns:
            chromadb.Settings: ChromaDB settings object

        Examples:
            >>> config = VectorStoreConfig.for_chromadb(...)
            >>> settings = config.to_chromadb_settings()
            >>> client = chromadb.PersistentClient(path=config.path, settings=settings)
        """
        import chromadb

        return chromadb.Settings(**self.settings)
