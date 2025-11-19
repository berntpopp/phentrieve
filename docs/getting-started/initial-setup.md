# Initial Setup

Before using Phentrieve, you must prepare the HPO data and build vector indexes.

## Configuration

Phentrieve uses a `phentrieve.yaml` file for configuration.
1. Copy the template: `cp phentrieve.yaml.template phentrieve.yaml`
2. Edit `phentrieve.yaml` to customize model selection or data paths.

## 1. Data Preparation (SQLite)

Phentrieve uses a local SQLite database (`hpo_data.db`) to store HPO terms and graph metadata for high-performance retrieval.

```bash
phentrieve data prepare
```

This command:
1. Downloads the official `hp.json` from the [HPO ontology repository](https://github.com/obophenotype/human-phenotype-ontology)
2. Parses 19,534+ HPO terms with labels, definitions, synonyms, and comments
3. Pre-computes ontology hierarchy (ancestor graphs and term depths)
4. **Stores structured data** in `data/hpo_data.db` using optimized SQLite schema

### What Gets Created

The data preparation process generates a compact, high-performance SQLite database:

- **Database Size**: ~12 MB (compared to 60 MB with previous file-based storage)
- **Performance**: 10-15x faster loading (0.87s vs 10-15s)
- **Schema Optimizations**:
  - Write-Ahead Logging (WAL) mode for concurrent reads
  - `WITHOUT ROWID` optimization for 20% storage savings
  - Memory-mapped I/O for faster access
  - Indexed columns for common queries

### Database Schema

The SQLite database contains three main tables:

1. **`hpo_terms`**: Core HPO term data
   - HPO ID (e.g., HP:0000123)
   - Label/name
   - Definition
   - Synonyms (JSON array)
   - Comments (JSON array)

2. **`hpo_graph_metadata`**: Pre-computed graph structure
   - Term depth (distance from root HP:0000001)
   - Ancestor set (JSON array)

3. **`generation_metadata`**: Tracking and versioning
   - Schema version
   - Data source information
   - Generation timestamps

## 2. Building Vector Indexes

Build the ChromaDB vector index for your chosen embedding model.

```bash
# Build index for the default model (BioLORD)
phentrieve index build

# Or specify a model explicitly
phentrieve index build --model-name "FremyCompany/BioLORD-2023-M"

# Build indexes for all supported models (for benchmarking)
phentrieve index build --all-models
```

### Index Building Process

The index builder:
1. Loads all HPO terms from the SQLite database
2. Creates rich document representations combining labels, definitions, and synonyms
3. Generates embeddings using the specified model
4. Stores vectors in ChromaDB persistent storage

**Time Estimates**:
- First model: 5-10 minutes (downloads model weights)
- Subsequent models: 2-5 minutes (cached weights)
- With GPU: 1-3 minutes

## 3. Supported Embedding Models

Phentrieve supports several multilingual embedding models optimized for different use cases:

### Domain-Specific Models (Recommended)

- **`FremyCompany/BioLORD-2023-M`** (Default)
  - Biomedical domain specialization
  - Excellent performance on clinical terminology
  - Multilingual support

### Language-Specific Models

- **`jinaai/jina-embeddings-v2-base-de`**
  - German language specialization
  - High precision for German clinical text

### General Multilingual Models

- **`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`**
  - 50+ languages supported
  - Good general-purpose performance

- **`BAAI/bge-m3`**
  - State-of-the-art multilingual embeddings
  - Excellent for cross-lingual retrieval

- **`Alibaba-NLP/gte-multilingual-base`**
  - Optimized for retrieval tasks
  - Fast inference

!!! note "Model Selection"
    The BioLORD model is recommended for most use cases as it provides excellent performance specifically tuned for biomedical terminology. For non-English text, consider using language-specific models or cross-lingual models like BGE-M3.

## 4. Language Resources (Text Processing)

If using text processing features, ensure language resources (spaCy models) are installed. If you used `make install-dev` or the Docker image, these are already included.

Otherwise, you can manually install them:
```bash
python -m spacy download en_core_web_sm  # English
python -m spacy download de_core_news_sm # German
python -m spacy download es_core_news_sm # Spanish
python -m spacy download fr_core_news_sm # French
python -m spacy download nl_core_news_sm # Dutch
```

## Data Storage Locations

By default, Phentrieve stores its data in these locations:

```text
data/
├── hpo_data.db           # SQLite database (HPO terms and graph)
├── hp.json               # Source HPO JSON file
├── indexes/              # ChromaDB persistent storage
│   └── {model_name}/    # Per-model vector stores
├── results/              # Benchmark results
├── hf_cache/             # HuggingFace model cache
└── hpo_translations/     # Translation files (if used)
```

You can configure these locations through environment variables or in `phentrieve.yaml`:

```yaml
# In phentrieve.yaml
data_dir: "./data"
index_dir: "./data/indexes"
results_dir: "./data/results"
```

Or via environment variables:
```bash
export PHENTRIEVE_DATA_ROOT_DIR=/path/to/data
export PHENTRIEVE_INDEX_DIR=/path/to/indexes
export PHENTRIEVE_RESULTS_DIR=/path/to/results
```

## Verification

Verify your setup is complete:

```bash
# Check database exists and has content
ls -lh data/hpo_data.db

# Test interactive query mode
phentrieve query --interactive

# Try a simple query
phentrieve query --text "seizures and small head"
```

If everything is working, you should see HPO term suggestions like:
- HP:0001250 (Seizure)
- HP:0000252 (Microcephaly)

## Next Steps

Once you've completed the initial setup:

1. Try [Interactive Querying](../user-guide/cli-usage.md#interactive-querying) to test your setup
2. Explore the [Text Processing Guide](../user-guide/text-processing-guide.md) to learn how to process clinical text
3. Check out the [Benchmarking Guide](../user-guide/benchmarking-guide.md) to evaluate model performance

## Troubleshooting

### Database Not Found
```
ERROR - HPO database not found: data/hpo_data.db
```
**Solution**: Run `phentrieve data prepare` to generate the database.

### Slow Index Building
**Solution**: Use GPU acceleration if available. Check with `python -c "import torch; print(torch.cuda.is_available())"`.

### Out of Memory
**Solution**: Reduce batch size in configuration or use a smaller embedding model.

!!! warning "Data Migration"
    If upgrading from a version prior to 0.2.0, the old file-based storage (`hpo_core_data/`) is obsolete. Run `phentrieve data prepare` to regenerate the SQLite database. Old pickle files can be safely deleted after migration.
