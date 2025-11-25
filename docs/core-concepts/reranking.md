# Cross-Encoder Reranking

Phentrieve uses a two-stage retrieval architecture that combines fast bi-encoder retrieval with precise cross-encoder reranking to improve ranking quality for HPO term matching.

## How It Works

### Stage 1: Dense Retrieval (Bi-Encoder)

The bi-encoder (BioLORD-2023-M) creates separate embeddings for queries and HPO terms, enabling fast similarity search across the entire ontology. This provides high recall but may occasionally rank similar-but-incorrect terms highly.

### Stage 2: Protected Reranking (Cross-Encoder)

The cross-encoder (BAAI/bge-reranker-v2-m3) processes query-document pairs together, enabling more accurate relevance judgments. Phentrieve implements **protected two-stage retrieval**:

1. **High-confidence matches are protected**: Dense retrieval results with similarity ≥0.7 are preserved at top positions
2. **Uncertain matches are refined**: Lower-confidence candidates are reranked by the cross-encoder
3. **Results are merged**: Protected matches stay on top, reranked results fill below

This approach prevents cross-encoders from demoting correct cross-lingual matches due to lexical bias.

## Default Model

**BAAI/bge-reranker-v2-m3** (568M parameters)

- Fine-tuned from [BGE-M3](https://huggingface.co/BAAI/bge-m3) on multilingual datasets
- Supports 100+ languages via XLM-RoBERTa architecture
- Optimized for semantic matching without requiring translation

**Reference**: Chen et al. (2024). [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216). arXiv:2402.03216.

## Configuration

### CLI Usage

```bash
# Enable reranking with default settings
phentrieve query --enable-reranker

# Specify a different cross-encoder model
phentrieve query --enable-reranker --reranker-model "ncbi/MedCPT-Cross-Encoder"

# Control number of candidates to rerank
phentrieve query --enable-reranker --rerank-count 50
```

### Configuration File

```yaml
# phentrieve.yaml
enable_reranker: true
reranker_model: "BAAI/bge-reranker-v2-m3"
rerank_candidate_count: 50
dense_trust_threshold: 0.7  # Protect matches above this similarity
```

## Alternative Models

Other cross-encoder models compatible with Phentrieve:

| Model | Domain | Notes |
|-------|--------|-------|
| `ncbi/MedCPT-Cross-Encoder` | Biomedical | NCBI's medical text matcher |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | General | Fast, English-focused |

!!! tip "GPU Acceleration"
    Cross-encoder reranking benefits significantly from GPU acceleration. Phentrieve automatically uses CUDA when available.
