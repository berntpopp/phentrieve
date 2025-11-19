# Core Concepts

This section explains the key concepts and technologies behind Phentrieve's approach to HPO term mapping.

## Multilingual Retrieval Augmented Generation (RAG)

Phentrieve uses a Retrieval Augmented Generation (RAG) approach to map clinical text to Human Phenotype Ontology (HPO) terms. The key innovation is doing this in a multilingual context without requiring translation.

## Key Components

### Embedding Models

At the heart of Phentrieve are multilingual embedding models that convert text from any supported language into vector representations. These models are trained to map semantically similar concepts from different languages to nearby points in the embedding space.

### HPO Term Vectors

The system pre-processes HPO terms into comprehensive documents containing:
- HPO ID (e.g., HP:0000123)
- Primary label/name in English
- Definition
- Synonyms

These documents are converted into vector embeddings and stored in a vector database for efficient retrieval.

### Semantic Similarity

When clinical text is processed, it's embedded using the same model and compared to the stored HPO term vectors using semantic similarity. This allows direct matching between clinical descriptions in any language and English-based HPO terminology.

### Cross-Encoder Re-ranking

To improve retrieval precision, Phentrieve can use cross-encoder models to re-rank the initial results, providing more accurate mappings between clinical text and HPO terms.

## Section Contents

- [Multilingual Embeddings](multilingual-embeddings.md): How multilingual models enable cross-language matching
- [Reranking](reranking.md): Advanced re-ranking techniques for improved precision
- [Ontology Similarity](ontology-similarity.md): Methods for calculating semantic similarity between HPO terms
