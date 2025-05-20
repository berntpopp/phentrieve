"""
Central configuration for the Phentrieve package.

This module contains constants, defaults, and configuration parameters used
throughout the phentrieve package.
"""

import sys
import warnings
import copy
from pathlib import Path

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
# Default chunking pipeline configuration
DEFAULT_CHUNK_PIPELINE_CONFIG = [{"type": "paragraph"}, {"type": "sentence"}]

# Predefined chunking strategies
SIMPLE_CHUNKING_CONFIG = [{"type": "paragraph"}, {"type": "sentence"}]

SEMANTIC_CHUNKING_CONFIG = [
    {"type": "paragraph"},
    {
        "type": "semantic",
        "config": {
            "similarity_threshold": 0.4,
            "min_chunk_sentences": 1,
            "max_chunk_sentences": 3,
        },
    },
]

DETAILED_CHUNKING_CONFIG = [
    {"type": "paragraph"},
    {"type": "fine_grained_punctuation"},
    {
        "type": "pre_chunk_semantic_grouper",
        "config": {
            "similarity_threshold": 0.5,
            "min_group_size": 1,
            "max_group_size": 7,
        },
    },
]


# Functions to get fresh copies of the configs to avoid mutation issues
def get_default_chunk_pipeline_config():
    return copy.deepcopy(DEFAULT_CHUNK_PIPELINE_CONFIG)


def get_simple_chunking_config():
    return copy.deepcopy(SIMPLE_CHUNKING_CONFIG)


def get_semantic_chunking_config():
    return copy.deepcopy(SEMANTIC_CHUNKING_CONFIG)


def get_detailed_chunking_config():
    return copy.deepcopy(DETAILED_CHUNKING_CONFIG)


# Default assertion detection configuration
DEFAULT_ASSERTION_CONFIG = {
    "enable_keyword": True,
    "enable_dependency": True,
    "preference": "dependency",
}

# Default language for text processing
DEFAULT_LANGUAGE = "en"
