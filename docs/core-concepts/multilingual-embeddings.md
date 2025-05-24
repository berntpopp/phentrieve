# Multilingual Embeddings

Multilingual embedding models are at the core of Phentrieve's ability to map clinical text in different languages to HPO terms without requiring translation.

## How Multilingual Embeddings Work

Multilingual embedding models are trained on text from multiple languages simultaneously. They learn to map semantically similar concepts from different languages to nearby points in a shared vector space.

For example, the words "headache" (English), "Kopfschmerzen" (German), and "céphalée" (French) all refer to the same medical concept. A well-trained multilingual model will place these terms close together in the embedding space despite their different languages.

This property enables Phentrieve to:
1. Embed clinical text in any supported language
2. Embed HPO terms (primarily in English)
3. Find matches based on semantic similarity in the shared embedding space

## Supported Models in Phentrieve

Phentrieve supports several multilingual embedding models, each with different strengths:

### Domain-Specific Models

- **FremyCompany/BioLORD-2023-M**: Specialized for biomedical terminology, this model shows the strongest performance for HPO term mapping across languages.

### Language-Specific Models

- **jinaai/jina-embeddings-v2-base-de**: Optimized for German language understanding, this model performs well for German clinical text but lacks the cross-lingual capabilities of other models.

### General Multilingual Models

- **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**: A general-purpose multilingual model that performs reasonably well across many languages.
- **BAAI/bge-m3**: Supports a wide range of languages with good performance.
- **Alibaba-NLP/gte-multilingual-base**: Another strong general-purpose multilingual embedding model.

## Benchmark Results

Based on our memories, GPU-accelerated benchmarking shows significant performance differences between models:

- **BioLORD-2023-M** (domain-specific biomedical model):
  - MRR: 0.5361
  - HR@1: 0.3333
  - HR@3: 0.6667
  - HR@5: 0.7778
  - HR@10: 1.0
  - Recall: 1.0

- **Jina-v2-base-de** (German language-specific model):
  - MRR: 0.3708
  - HR@1: 0.2222
  - HR@3: 0.4444
  - HR@5: 0.5556
  - HR@10: 0.7778
  - Recall: 0.7778

These results demonstrate that domain-specific models (BioLORD) consistently outperform language-specific models for medical terminology retrieval, highlighting the importance of domain expertise over language specialization.

## Model Selection Guide

When choosing a model for your use case, consider:

1. **Language coverage**: Does the model support all the languages you need?
2. **Domain expertise**: Is the model trained on medical/biomedical text?
3. **Performance**: How well does the model perform in benchmarks?
4. **Resource requirements**: Larger models require more memory and computational resources.

!!! tip "Recommended Default"
    For most medical text processing use cases, the BioLORD model is recommended as it consistently outperforms other models in HPO term mapping tasks.

## GPU Acceleration

Phentrieve supports GPU acceleration with CUDA when available and gracefully falls back to CPU when unavailable. Using GPU acceleration can significantly improve processing speed, especially for larger models and batch processing.

To ensure GPU acceleration is utilized:
- Install PyTorch with CUDA support
- Have a compatible NVIDIA GPU
- Phentrieve will automatically detect and use GPU resources
