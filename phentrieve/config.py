"""
Central configuration for the Phentrieve package.

This module contains constants, defaults, and configuration parameters used
throughout the phentrieve package.
"""

# Note: This module intentionally does not import path resolution functions
# We avoid importing from utils to prevent circular imports

# Default directory sub-paths and filenames (relative to base dirs)
# Sub-directories (for data_dir)
DEFAULT_HPO_TERMS_SUBDIR = "hpo_terms"
DEFAULT_TEST_CASES_SUBDIR = "test_cases"
DEFAULT_TRANSLATIONS_SUBDIR = "hpo_translations_de"

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
JINA_MODEL_ID = "jinaai/jina-embeddings-v2-base-de"

# All models for benchmarking
BENCHMARK_MODELS = [
    "FremyCompany/BioLORD-2023-M",
    "jinaai/jina-embeddings-v2-base-de",
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
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
DEFAULT_TOP_K = 5  # Default number of results to return
DEFAULT_DEVICE = None  # Default device (None = auto-detect)

# Cross-encoder re-ranking settings
# Multilingual cross-encoder model for re-ranking
DEFAULT_RERANKER_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# German cross-encoder model for monolingual re-ranking
DEFAULT_MONOLINGUAL_RERANKER_MODEL = (
    "ml6team/cross-encoder-mmarco-german-distilbert-base"
)
# Re-ranking mode options:
# - 'cross-lingual': German query -> English HPO
# - 'monolingual': German query -> German HPO
DEFAULT_RERANKER_MODE = "cross-lingual"
# Default directory for German HPO term translations
# Note: Path is resolved at runtime using DEFAULT_TRANSLATIONS_SUBDIR
DEFAULT_RERANK_CANDIDATE_COUNT = 50
DEFAULT_ENABLE_RERANKER = False

# Root for HPO term extraction and depth calculations
PHENOTYPE_ROOT = "HP:0000118"
