# Algorithms and Packages Reference

This document provides a comprehensive reference of the algorithms, packages, methods, and functions used throughout Phentrieve's implementation.

## Overview

Phentrieve implements a multi-stage pipeline combining natural language processing, vector embeddings, and information retrieval to map clinical text to HPO terms. This reference documents the technical implementation details.

## Core Algorithms

### 1. Semantic Chunking

#### Algorithm: Sliding Window Semantic Splitting

**Mathematical Foundation:**

Given a text segment with tokens $T = [t_1, t_2, ..., t_n]$:

1. **Window Creation**: Create overlapping windows of size $w$ with step $s$:
   - $W_i = [t_i, t_{i+1}, ..., t_{i+w-1}]$ where $i \in \{1, s+1, 2s+1, ...\}$

2. **Embedding**: Embed each window using function $E$:
   - $v_i = E(W_i)$ where $v_i \in \mathbb{R}^d$

3. **Similarity Computation**: Calculate cosine similarity between adjacent windows:
   $$\text{sim}(v_i, v_{i+1}) = \frac{v_i \cdot v_{i+1}}{||v_i|| \cdot ||v_{i+1}||}$$

4. **Split Detection**: Mark split point when:
   $$\text{sim}(v_i, v_{i+1}) < \theta$$
   where $\theta$ is the splitting threshold (default: 0.5)

**Implementation:**
- **Class**: `SlidingWindowSemanticSplitter`
- **Module**: `phentrieve.text_processing.chunkers`
- **Key Methods**:
  - `chunk(text_segments: list[str]) -> list[str]`
  - `_split_one_segment_by_sliding_window(text: str) -> list[str]`

**Parameters:**
- `window_size_tokens`: Size of sliding window (default: 7)
- `step_size_tokens`: Step between windows (default: 1)
- `splitting_threshold`: Similarity threshold for splitting (default: 0.5)
- `min_split_segment_length_words`: Minimum segment length (default: 3)

**Dependencies:**
- `sentence-transformers` - For window embedding
- `numpy` - For cosine similarity computation
- `torch` - For tensor operations

### 2. Dense Retrieval

#### Algorithm: Bi-Encoder Semantic Search

**Mathematical Foundation:**

Given a query $q$ and HPO term documents $D = \{d_1, d_2, ..., d_m\}$:

1. **Query Embedding**:
   $$q_v = E_\text{query}(q)$$

2. **Document Embedding** (pre-computed):
   $$d_{v_i} = E_\text{doc}(d_i)$$

3. **Similarity Scoring**:
   $$\text{score}(q, d_i) = \frac{q_v \cdot d_{v_i}}{||q_v|| \cdot ||d_{v_i}||}$$

4. **Top-K Retrieval**:
   $$\text{results} = \text{argsort}_k\{\text{score}(q, d_i) : i \in [1, m]\}$$

**Implementation:**
- **Class**: `DenseRetriever`
- **Module**: `phentrieve.retrieval.dense_retriever`
- **Key Methods**:
  - `query(text: str, k: int) -> list[dict]`
  - `query_batch(texts: list[str], k: int) -> list[list[dict]]`
  - `_embed_queries(texts: list[str]) -> np.ndarray`

**Batch Processing Optimization:**
- Processes multiple queries in single forward pass
- Uses matrix multiplication for parallel similarity computation
- 10-20x faster than sequential processing

**Dependencies:**
- `sentence-transformers.SentenceTransformer` - Bi-encoder models
- `chromadb` - Vector database for storage and retrieval
- `torch` - GPU acceleration for embedding generation
- `numpy` - Matrix operations

### 3. Cross-Encoder Re-Ranking

#### Algorithm: Cross-Attention Scoring

**Mathematical Foundation:**

Given query $q$ and candidate documents $C = \{c_1, c_2, ..., c_k\}$ from dense retrieval:

1. **Pair Construction**:
   $$\text{pairs} = \{(q, c_1), (q, c_2), ..., (q, c_k)\}$$

2. **Cross-Attention Scoring**:
   $$\text{score}_\text{cross}(q, c_i) = \text{CrossEncoder}([q; c_i])$$
   where $[q; c_i]$ denotes concatenation with separator token

3. **Re-Ranking**:
   $$\text{results}_\text{reranked} = \text{argsort}_k\{\text{score}_\text{cross}(q, c_i) : c_i \in C\}$$

**Implementation:**
- **Class**: `CrossEncoder`
- **Module**: `phentrieve.retrieval.reranker`
- **Key Methods**:
  - `rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]`
  - `_compute_scores(pairs: list[tuple[str, str]]) -> list[float]`

**Mode Options:**
- `crosslingual`: Query and documents in different languages
- `monolingual`: Same language matching

**Dependencies:**
- `sentence-transformers.CrossEncoder` - Cross-encoder models
- `torch` - GPU acceleration

### 4. Assertion Detection

#### Algorithm: Dependency-Based Negation Detection

**Rule-Based Logic:**

1. **Dependency Parsing**: Parse sentence into dependency tree using spaCy
2. **Negation Detection**: Identify negation markers (`neg`, `no`, `without`)
3. **Scope Determination**: Find all tokens within negation scope via dependency traversal
4. **Classification**:
   - `NEGATED` if chunk in negation scope
   - `UNCERTAIN` if uncertainty markers present
   - `NORMAL` if normal markers present
   - `AFFIRMED` otherwise

**Implementation:**
- **Class**: `CombinedAssertionDetector`
- **Module**: `phentrieve.text_processing.assertion_detection`
- **Key Methods**:
  - `detect_assertion(text: str, chunk: str) -> AssertionType`
  - `_dependency_based_detection(doc, chunk_text: str) -> AssertionType`
  - `_keyword_based_detection(text: str, chunk: str) -> AssertionType`

**Dependencies:**
- `spacy` - Dependency parsing and NLP pipeline
- Custom language resources (JSON files with negation keywords)

## Key Packages and Their Roles

### Embedding and Retrieval

#### sentence-transformers (v2.2.2+)

**Purpose**: State-of-the-art embedding models for semantic similarity

**Key Classes:**
- `SentenceTransformer`: Bi-encoder for dense retrieval
  - Methods: `encode()`, `encode_multi_process()`
- `CrossEncoder`: Cross-attention re-ranking
  - Methods: `predict()`, `rank()`

**Models Used:**
- `FremyCompany/BioLORD-2023-M` - Biomedical domain (default)
- `jinaai/jina-embeddings-v2-base-de` - German specialized
- `BAAI/bge-m3` - Multilingual general purpose
- `BAAI/bge-reranker-v2-m3` - Multilingual re-ranker

**Usage in Phentrieve:**
```python
# Dense retrieval
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("FremyCompany/BioLORD-2023-M")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# Re-ranking
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
scores = reranker.predict([(query, doc) for doc in candidates])
```

#### ChromaDB (v0.4.0+)

**Purpose**: Vector database for efficient similarity search

**Key Classes:**
- `chromadb.Client`: Database client
- `chromadb.Collection`: Vector collection management

**Key Methods:**
- `create_collection()` - Initialize vector store
- `add()` - Batch insert embeddings
- `query()` - Similarity search with filtering
- `get()` - Retrieve by ID
- `update()` - Update embeddings

**Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Time Complexity**: O(log N) for queries
- **Space Complexity**: O(N × M × d) where N=docs, M=neighbors, d=dimensions

**Usage in Phentrieve:**
```python
import chromadb
client = chromadb.PersistentClient(path="./data/indexes")
collection = client.get_or_create_collection(
    name="hpo_embeddings",
    metadata={"hnsw:space": "cosine"}
)
results = collection.query(
    query_embeddings=query_vectors,
    n_results=20,
    include=["documents", "distances", "metadatas"]
)
```

#### PyTorch (v2.0+)

**Purpose**: Deep learning framework for model inference

**Key Functions:**
- `torch.cuda.is_available()` - GPU detection
- `torch.device()` - Device management
- `torch.nn.functional.cosine_similarity()` - Similarity computation
- `model.to(device)` - Device placement

**Automatic Device Selection:**
```python
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
else:
    device = "cpu"
```

### Text Processing

#### spaCy (v3.5+)

**Purpose**: Industrial-strength NLP for text processing

**Models Used:**
- `en_core_web_sm` - English
- `de_core_news_sm` - German
- `es_core_news_sm` - Spanish
- `fr_core_news_sm` - French
- `nl_core_news_sm` - Dutch

**Key Components:**
- **Tokenizer**: Word and sentence tokenization
- **POS Tagger**: Part-of-speech tagging
- **Dependency Parser**: Grammatical dependency trees
- **NER**: Named entity recognition (optional)

**Methods Used:**
```python
import spacy
nlp = spacy.load("en_core_web_sm")

# Sentence splitting
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]

# Dependency parsing for negation detection
for token in doc:
    if token.dep_ == "neg":
        # Find negation scope via dependency tree
        scope = [child for child in token.head.subtree]
```

#### NumPy (v1.24+)

**Purpose**: Numerical computing for vector operations

**Key Functions:**
- `np.dot()` - Dot product for similarity
- `np.linalg.norm()` - Vector magnitude
- `np.argsort()` - Sorting for top-k retrieval
- `np.concatenate()` - Batch concatenation

**Cosine Similarity Implementation:**
```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Data Processing

#### SQLite3 (Python stdlib)

**Purpose**: Lightweight database for HPO metadata storage

**Key Functions:**
- `sqlite3.connect()` - Database connection
- `cursor.execute()` - SQL execution
- `cursor.executemany()` - Batch inserts
- `cursor.executescript()` - Schema creation

**Optimizations Applied:**
```sql
-- Write-Ahead Logging for concurrent reads
PRAGMA journal_mode = WAL;

-- WITHOUT ROWID for 20% space savings
CREATE TABLE hpo_terms (...) WITHOUT ROWID;

-- Indexes for fast queries
CREATE INDEX idx_hpo_terms_label ON hpo_terms(label);
```

**Methods in Phentrieve:**
```python
from phentrieve.data_processing.hpo_database import HPODatabase

db = HPODatabase("data/hpo_data.db")
db.initialize_schema()
terms = db.load_all_terms()  # Returns list[dict]
ancestors, depths = db.load_graph_data()  # Returns tuple[dict, dict]
```

### API and Frontend

#### FastAPI (v0.104+)

**Purpose**: High-performance async web framework

**Key Decorators:**
- `@app.get()`, `@app.post()` - Route handlers
- `@lru_cache` - Response caching
- Dependency injection via `Depends()`

**Features Used:**
- Automatic OpenAPI documentation
- Pydantic validation
- Async request handling
- CORS middleware
- Background tasks for long-running operations

#### Pydantic (v2.0+)

**Purpose**: Data validation and serialization

**Usage:**
```python
from pydantic import BaseModel, Field

class HPOQueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    num_results: int = Field(default=5, ge=1, le=100)
    enable_reranker: bool = Field(default=False)
```

#### Vue 3 + Vuetify

**Purpose**: Reactive frontend framework

**Key Features:**
- Composition API for component logic
- Pinia for state management
- Vue Router for navigation
- Vue i18n for internationalization

## Performance Characteristics

### Algorithmic Complexity

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Semantic Chunking | O(n × w × d) | O(n × d) | n=tokens, w=window, d=dims |
| Dense Retrieval | O(k × d) | O(m × d) | k=queries, m=docs, d=dims |
| HNSW Search | O(log m) | O(m × M × d) | M=neighbors per layer |
| Cross-Encoder | O(k × L²) | O(k × L) | L=sequence length |
| Dependency Parsing | O(n²) | O(n) | n=sentence tokens |

### Optimization Techniques

**Batch Processing:**
```python
# Instead of: for text in texts: model.encode(text)
# Use batch processing:
embeddings = model.encode(texts, batch_size=32)  # 10-20x faster
```

**Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model(model_name: str):
    """Cache loaded models to avoid reloading."""
    return SentenceTransformer(model_name)
```

**Lazy Loading:**
```python
# Don't load models at import time
# Load only when first used
_model_registry: dict[str, SentenceTransformer] = {}

def get_or_load_model(name: str):
    if name not in _model_registry:
        _model_registry[name] = SentenceTransformer(name)
    return _model_registry[name]
```

## Implementation Modules

### Phentrieve Package Structure

```
phentrieve/
├── embeddings.py              # Model loading and caching
├── retrieval/
│   ├── dense_retriever.py     # DenseRetriever class
│   └── reranker.py            # CrossEncoder re-ranking
├── text_processing/
│   ├── chunkers.py            # All chunking classes
│   ├── assertion_detection.py # Assertion detection
│   └── pipeline.py            # TextProcessingPipeline
├── data_processing/
│   ├── hpo_database.py        # HPODatabase class
│   └── hpo_parser.py          # HPO JSON parsing
└── evaluation/
    └── metrics.py             # Benchmark metrics
```

### Key Classes and Methods

#### DenseRetriever

**File**: `phentrieve/retrieval/dense_retriever.py`

**Methods:**
```python
class DenseRetriever:
    def query(self, text: str, k: int = 5) -> list[dict]:
        """Single query retrieval."""

    def query_batch(self, texts: list[str], k: int = 5) -> list[list[dict]]:
        """Batch query retrieval (10-20x faster)."""

    def _embed_queries(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for queries."""

    def _compute_similarities(self, query_emb, doc_embs) -> np.ndarray:
        """Compute cosine similarities."""
```

#### SlidingWindowSemanticSplitter

**File**: `phentrieve/text_processing/chunkers.py`

**Methods:**
```python
class SlidingWindowSemanticSplitter(TextChunker):
    def chunk(self, text_segments: list[str]) -> list[str]:
        """Process and split text segments."""

    def _split_one_segment_by_sliding_window(self, text: str) -> list[str]:
        """Split single segment using sliding window."""

    def _create_sliding_windows(self, tokens: list[str]) -> list[str]:
        """Create overlapping windows."""

    def _compute_window_embeddings(self, windows: list[str]) -> np.ndarray:
        """Embed all windows in batch."""
```

#### HPODatabase

**File**: `phentrieve/data_processing/hpo_database.py`

**Methods:**
```python
class HPODatabase:
    def initialize_schema(self) -> None:
        """Create database schema with optimizations."""

    def bulk_insert_terms(self, terms: list[dict]) -> int:
        """Batch insert HPO terms."""

    def load_all_terms(self) -> list[dict]:
        """Load terms with JSON deserialization."""

    def load_graph_data(self) -> tuple[dict, dict]:
        """Load ancestors and depths."""
```

## Configuration

### Model Configuration (phentrieve.yaml)

```yaml
embeddings:
  model_name: "FremyCompany/BioLORD-2023-M"
  device: "auto"  # auto, cuda, mps, cpu
  batch_size: 32
  normalize_embeddings: true

retrieval:
  num_results: 20
  enable_reranker: true
  reranker_model: "BAAI/bge-reranker-v2-m3"
  reranker_mode: "crosslingual"
  reranker_top_k: 20

chunking_pipeline:
  - type: paragraph
  - type: sentence
  - type: fine_grained_punctuation
  - type: conjunction
  - type: sliding_window
    config:
      window_size_tokens: 7
      step_size_tokens: 1
      splitting_threshold: 0.5
  - type: final_chunk_cleaner
```

## References

### Academic Papers

1. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.

2. **Dense Passage Retrieval**: Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." EMNLP 2020.

3. **Cross-Encoders**: Nogueira, R., & Cho, K. (2019). "Passage Re-ranking with BERT." arXiv:1901.04085.

4. **HNSW Algorithm**: Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." TPAMI.

### Package Documentation

- **sentence-transformers**: https://www.sbert.net/
- **ChromaDB**: https://docs.trychroma.com/
- **spaCy**: https://spacy.io/api
- **FastAPI**: https://fastapi.tiangolo.com/
- **PyTorch**: https://pytorch.org/docs/

## Version Compatibility

| Package | Minimum Version | Tested Version | Notes |
|---------|----------------|----------------|-------|
| sentence-transformers | 2.2.2 | 2.7.0 | Core functionality |
| chromadb | 0.4.0 | 0.5.0 | Vector storage |
| spacy | 3.5.0 | 3.7.0 | NLP pipeline |
| torch | 2.0.0 | 2.3.0 | GPU acceleration |
| fastapi | 0.104.0 | 0.110.0 | API framework |
| pydantic | 2.0.0 | 2.6.0 | Validation |
| numpy | 1.24.0 | 1.26.0 | Numerical ops |
| sqlite3 | 3.35.0 | 3.45.0 | Database (stdlib) |

!!! note "GPU Acceleration"
    All embedding and retrieval operations automatically utilize GPU when available via PyTorch's device management. CPU fallback is automatic and seamless.

!!! tip "Performance Optimization"
    For production deployments:
    - Use batch processing wherever possible (10-20x speedup)
    - Enable GPU acceleration for embedding generation (3-5x speedup)
    - Use model caching to avoid reloading weights
    - Configure appropriate batch sizes based on available memory
