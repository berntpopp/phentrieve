# Initial Setup

After installing Phentrieve, you need to prepare the HPO data and build the vector indexes for the embedding models you plan to use.

## Data Preparation

The first step is to download and process the Human Phenotype Ontology (HPO) data:

```bash
# Download and process HPO data
phentrieve data prepare
```

This command:

1. Downloads the official HPO data in JSON format from JAX
2. Extracts HPO terms with a focus on phenotypic abnormalities
3. Creates comprehensive documents for each HPO term containing:
   - HPO ID (e.g., HP:0000123)
   - Primary label/name in English
   - Definition
   - Synonyms
   - Parent and child term relationships

## Building Indexes

After preparing the HPO data, you need to build the vector indexes for the embedding models you want to use:

```bash
# Build the index for a specific model
phentrieve index build --model-name "FremyCompany/BioLORD-2023-M"

# Or build indexes for all supported models (for benchmarking)
phentrieve index build --all-models
```

!!! note "Model Selection"
    The BioLORD model is recommended for most use cases as it provides excellent performance for biomedical terminology.

### Supported Models

Phentrieve supports several multilingual embedding models:

- **Domain-specific models**:
  - `FremyCompany/BioLORD-2023-M` (biomedical specialized)
- **Language-specific models**:
  - `jinaai/jina-embeddings-v2-base-de` (German specialized)
- **General multilingual models**:
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - `BAAI/bge-m3`
  - `Alibaba-NLP/gte-multilingual-base`
  - And others

## Data Storage Locations

By default, Phentrieve stores its data in these locations:

```text
/your/data/dir/
├── hpo_core_data/    # HPO source files (hp.json, etc.)
├── indexes/          # ChromaDB persistent storage
├── results/          # Benchmark results
└── hpo_translations/ # Translation files (if used)
```

You can configure these locations through environment variables or in the configuration file.

## Next Steps

Once you've completed the initial setup:

1. Try [Interactive Querying](../user-guide/cli-usage.md#interactive-querying) to test your setup
2. Explore the [Text Processing Guide](../user-guide/text-processing-guide.md) to learn how to process clinical text
3. Check out the [Benchmarking Guide](../user-guide/benchmarking-guide.md) to evaluate model performance
