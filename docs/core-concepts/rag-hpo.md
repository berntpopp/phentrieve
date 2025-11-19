# RAG-HPO Approach

Phentrieve uses a Retrieval-Augmented Generation (RAG) approach specifically adapted for mapping clinical text to Human Phenotype Ontology (HPO) terms.

## What is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that combines:

1. A retrieval system that finds relevant documents from a knowledge base
2. A generation system that uses both the query and retrieved documents to produce a response

In the context of Phentrieve, we adapt this approach to retrieve relevant HPO terms for a given clinical description.

## The Phentrieve RAG-HPO Architecture

### 1. Setup Phase (One-time)

- **HPO Data Acquisition**: Downloads the official HPO data in JSON format from JAX
- **HPO Term Extraction**: Processes the HPO data, extracting HPO terms with a focus on phenotypic abnormalities
- **Document Creation**: For each HPO term, creates a comprehensive document containing:
  - HPO ID (e.g., HP:0000123)
  - Primary label/name in English
  - Definition
  - Synonyms
- **Vector Embedding**: Converts each HPO term document into a vector representation using a multilingual embedding model
- **Index Creation**: Stores these vectors in a vector database (ChromaDB) for efficient retrieval

### 2. Query Phase (Runtime)

- **Query Embedding**: Converts the clinical text (in any supported language) into a vector representation using the same embedding model
- **Semantic Search**: Finds HPO term vectors that are most similar to the query vector
- **Re-ranking** (Optional): Uses a cross-encoder model to re-rank the results for improved precision
- **Result Processing**: Formats and returns the matched HPO terms with their similarity scores

## Advantages of this Approach

- **Direct semantic matching**: No error-prone intermediate translation step
- **Language-independent**: The model understands the meaning across languages
- **Scalable**: New HPO terms can be easily added to the knowledge base
- **Flexible**: Multiple embedding models can be used and compared
- **Extensible**: Additional processing steps (like assertion detection) can be integrated

## Technical Implementation

Phentrieve's implementation has been modernized to ensure performance and scalability:

### 1. Storage Layer

**Metadata Store (SQLite)**
- HPO terms, definitions, synonyms, and comments stored in optimized SQLite database (`hpo_data.db`)
- Pre-computed graph data (ancestors, depths) for fast ontology traversal
- Write-Ahead Logging (WAL) mode enabled for concurrent reads
- WITHOUT ROWID optimization for 20% storage savings
- ~12 MB database size for 19,534+ HPO terms

**Schema Design:**
```sql
-- Core terms table
CREATE TABLE hpo_terms (
    id TEXT PRIMARY KEY,          -- HP:0000123
    label TEXT NOT NULL,
    definition TEXT,
    synonyms TEXT,                -- JSON array
    comments TEXT                 -- JSON array
) WITHOUT ROWID;

-- Pre-computed graph metadata
CREATE TABLE hpo_graph_metadata (
    term_id TEXT PRIMARY KEY,
    depth INTEGER NOT NULL,       -- Distance from root HP:0000001
    ancestors TEXT NOT NULL,      -- JSON array of ancestor IDs
    FOREIGN KEY (term_id) REFERENCES hpo_terms(id)
) WITHOUT ROWID;
```

**Vector Store (ChromaDB)**
- Stores dense vector embeddings of HPO terms for semantic similarity search
- Persistent storage in `data/indexes/{model_name}/`
- Supports multiple concurrent indexes for different embedding models
- Fast approximate nearest neighbor (ANN) search using HNSW algorithm

### 2. Retrieval Layer (`DenseRetriever`)

**Batch Processing**
- Queries processed in batches using `query_batch` method
- Leverages matrix operations in PyTorch for 10-20x faster performance
- Configurable batch size to balance memory and throughput

**Model Management**
- Uses `sentence-transformers` for bi-encoder retrieval
- Models cached in memory using thread-safe registry pattern
- Lazy loading prevents unnecessary model initialization
- Automatic device selection (CUDA → MPS → CPU)

**Optimization Features:**
- Connection pooling for ChromaDB client
- Query result caching for repeated queries
- Parallel embedding generation for large batches

### 3. Re-Ranking Layer (`CrossEncoder`)

**Precision Enhancement**
- Optional second-stage re-ranking of top-K results from dense retriever
- Uses cross-encoder models for more accurate relevance scoring
- Trades computational cost for improved precision

**Cross-Lingual Capabilities**
- `crosslingual` mode: Re-ranks non-English queries against English HPO terms directly
- `monolingual` mode: Best for same-language matching
- Supports multiple re-ranker models:
  - `BAAI/bge-reranker-v2-m3` (multilingual)
  - Biomedical cross-encoders
  - Domain-specific models

**Configuration:**
```yaml
retrieval:
  enable_reranker: true
  reranker_mode: crosslingual
  reranker_model: "BAAI/bge-reranker-v2-m3"
  reranker_top_k: 20  # Re-rank top 20 from dense retrieval
```

### 4. Application Layer

**FastAPI**
- High-performance async REST API
- Automatic OpenAPI documentation at `/docs`
- Request validation via Pydantic schemas
- Dependency injection for model loading and caching

**Concurrency Handling**
- Heavy NLP tasks offloaded to thread pools
- Keeps API responsive under load
- Configurable worker processes and threads

**Vue 3 Frontend**
- Reactive UI with Composition API
- Communicates with API via REST
- State management with Pinia
- Internationalization (i18n) for 5+ languages

## Performance Characteristics

### Loading Speed (SQLite vs Legacy)

| Metric | Legacy (Pickle) | SQLite | Improvement |
|--------|-----------------|---------|-------------|
| **Load Time** | 10-15 seconds | 0.87 seconds | 10-15x faster |
| **Storage Size** | 60 MB | 12 MB | 80% reduction |
| **Memory Usage** | High (full load) | Low (lazy) | ~75% less |
| **Query Speed** | O(n) scan | O(log n) indexed | Logarithmic |

### Retrieval Performance

| Operation | Batch Size | CPU Time | GPU Time |
|-----------|-----------|----------|----------|
| Dense Retrieval | 1 | ~50ms | ~20ms |
| Dense Retrieval | 32 | ~200ms | ~40ms |
| Re-ranking | 20 results | ~100ms | ~30ms |
| Full Pipeline | 1 query | ~150ms | ~50ms |

### Hardware Acceleration

**GPU Support:**
- CUDA-enabled GPUs: 3-5x speedup for embedding generation
- Apple Silicon (MPS): 2-3x speedup
- Automatic device selection and fallback

**Optimization Tips:**
1. Use GPU for production workloads (5x faster)
2. Enable re-ranking only when precision is critical (adds ~100ms)
3. Batch queries together (up to 32x for throughput)
4. Cache frequently used models in memory

!!! note "GPU Acceleration"
    Phentrieve supports GPU acceleration with CUDA when available and gracefully falls back to CPU when unavailable. Detection is automatic via PyTorch's device management.

!!! tip "Performance Tuning"
    For maximum throughput, enable GPU, use batch processing, and adjust `num_results` to retrieve only what you need. For latency-sensitive applications, consider disabling re-ranking or using a smaller re-ranker model.
