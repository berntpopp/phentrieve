"""
Central configuration for the Phentrieve package.

This module contains constants, defaults, and configuration parameters used
throughout the phentrieve package.
"""

import copy

# Note: This module intentionally does not import path resolution functions
# We avoid importing from utils to prevent circular imports

# Default directory sub-paths and filenames (relative to base dirs)
# Sub-directories (for data_dir)
DEFAULT_HPO_TERMS_SUBDIR = "hpo_terms"
DEFAULT_TEST_CASES_SUBDIR = "test_cases"
DEFAULT_TRANSLATIONS_SUBDIR = "hpo_translations"  # Directory for HPO term translations

# HPO data filenames (relative to data_dir)
DEFAULT_HPO_FILENAME = "hp.json"
DEFAULT_ANCESTORS_FILENAME = "hpo_ancestors.pkl"
DEFAULT_DEPTHS_FILENAME = "hpo_term_depths.pkl"

# Benchmark result subdirectories
DEFAULT_SUMMARIES_SUBDIR = "summaries"
DEFAULT_DETAILED_SUBDIR = "detailed"
DEFAULT_VISUALIZATIONS_SUBDIR = "visualizations"

# Default models
DEFAULT_MODEL = "FremyCompany/BioLORD-2023-M"
DEFAULT_BIOLORD_MODEL = "FremyCompany/BioLORD-2023-M"
JINA_MODEL_ID = (
    "jinaai/jina-embeddings-v2-base-de"  # Current default is a German embeddings model
)

# All models for benchmarking
BENCHMARK_MODELS = [
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

# Default parameters
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to display results
# Default similarity threshold used in benchmark evaluations
DEFAULT_SIMILARITY_THRESHOLD = 0.1
DEFAULT_TOP_K = 10  # Default number of results to return
DEFAULT_K_VALUES = (1, 3, 5, 10)  # Default k values for hit rate calculation
DEFAULT_DEVICE = None  # Default device (None = auto-detect)

# Cross-encoder re-ranking settings
# Multilingual cross-encoder model for re-ranking
DEFAULT_RERANKER_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# Language-specific cross-encoder model for monolingual re-ranking
DEFAULT_MONOLINGUAL_RERANKER_MODEL = (
    "ml6team/cross-encoder-mmarco-german-distilbert-base"
)
# Re-ranking mode options:
# - 'cross-lingual': Query in target language -> English HPO terms
# - 'monolingual': Query in target language -> HPO terms in same language
DEFAULT_RERANKER_MODE = "cross-lingual"
# Default directory for HPO term translations
# Note: Path is resolved at runtime using DEFAULT_TRANSLATIONS_SUBDIR
DEFAULT_TRANSLATION_DIR = None  # Will be resolved at runtime
DEFAULT_RERANK_CANDIDATE_COUNT = 50
DEFAULT_ENABLE_RERANKER = False

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


# Default formula for semantic similarity calculations
DEFAULT_SIMILARITY_FORMULA = "hybrid"

# Default assertion detection configuration
DEFAULT_ASSERTION_CONFIG = {
    "enable_keyword": True,
    "enable_dependency": True,
    "preference": "dependency",
}

# Default language for text processing
DEFAULT_LANGUAGE = "en"
