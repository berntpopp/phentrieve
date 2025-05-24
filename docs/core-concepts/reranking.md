# Cross-Encoder Reranking

Phentrieve supports re-ranking of retrieved candidate HPO terms using cross-encoder models, which can significantly improve the ranking precision of results.

## What is a Cross-Encoder?

Unlike bi-encoder models (which create separate embeddings for query and document), cross-encoders take both the query and document as input simultaneously. This allows them to directly model the interaction between the query and document, leading to more accurate relevance judgments.

However, cross-encoders are computationally more expensive as they need to process each query-document pair separately, making them unsuitable for initial retrieval from large collections but perfect for re-ranking a small set of candidates.

## Reranking Modes

Phentrieve supports two reranking modes:

### 1. Cross-lingual Reranking (Default)

Compares non-English queries directly with English HPO term labels using a multilingual cross-encoder model. This approach:

- Avoids the need for translation
- Maintains the multilingual capability of the system
- Provides consistent results across languages

### 2. Monolingual Reranking

Uses translated HPO term labels in the target language (if available) for comparison. This approach:

- May provide better performance for specific languages
- Requires pre-translated HPO terms
- Is less flexible across languages

## Supported Cross-Encoder Models

Phentrieve has been tested with three different cross-encoder models:

### 1. cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (Original Implementation)

- General multilingual retrieval model
- Returns negative scores (higher/less negative = better)
- Small and efficient, but not domain-specific

### 2. MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (Current Default)

- Multilingual natural language inference model
- Returns probability distributions for entailment/neutral/contradiction
- Adapted to extract entailment scores for ranking
- Strong multilingual capabilities but not domain-specific

### 3. ncbi/MedCPT-Cross-Encoder

- Biomedical domain-specific cross-encoder
- Developed by NCBI specifically for medical text matching
- Provides better understanding of medical relationships
- Complements BioLORD bi-encoder which already performs well on medical terminology

## Using Reranking

### CLI Usage

```bash
# Enable reranking with default settings (cross-lingual mode)
phentrieve query --enable-reranker

# Specify a particular cross-encoder model
phentrieve query --enable-reranker --reranker-model "ncbi/MedCPT-Cross-Encoder"

# Use monolingual reranking (requires translations)
phentrieve query --enable-reranker --reranker-mode monolingual --translation-dir path/to/translations
```

### Performance Considerations

Reranking improves precision but adds computational overhead:

- Without reranking: Faster but potentially less accurate ranking
- With reranking: Better ranking quality but slower processing

!!! tip "GPU Acceleration"
    When using reranking, GPU acceleration can significantly improve processing speed. Phentrieve automatically uses GPU acceleration with CUDA when available.
