# Embedding Model Expansion Plan

**Status:** Draft
**Created:** 2025-12-10
**Target Completion:** TBD
**Priority:** Medium
**Owner:** Development Team

---

## Executive Summary

This plan outlines the benchmarking of three additional embedding models for the Phentrieve HPO term retrieval system. The goal is to evaluate whether specialized biomedical models or state-of-the-art generalist models can outperform the current default (BioLORD-2023-M) for multilingual HPO term matching.

**Key Finding:** The current codebase has hardcoded embedding dimensions that will cause failures with new models. This plan addresses both the infrastructure fix and the benchmarking strategy.

---

## Objective

Evaluate and benchmark three embedding models to determine optimal HPO term retrieval performance across multiple languages (primarily German, with English support).

## Success Criteria

- [ ] Infrastructure: Dynamic embedding dimension detection implemented
- [ ] Infrastructure: All existing tests pass after changes
- [ ] Benchmarking: All three models successfully build ChromaDB indexes
- [ ] Benchmarking: Comparative metrics (MRR@10, Hit@k) collected for all models
- [ ] Documentation: Results documented with recommendations

---

## Model Selection

### Proposed Models (Corrected)

| Model | Dimensions | Size | Languages | Rationale |
|-------|-----------|------|-----------|-----------|
| [`cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR) | **768** | ~1.1GB | 100+ | Purpose-built for biomedical entity linking (UMLS trained) |
| [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | **1024** | ~1.3GB | English | SOTA on MTEB benchmark (March 2024), tests if generalist beats specialist |
| [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | **1024** | ~3.3GB | 100+ | Multi-function (dense+sparse+ColBERT), 8192 token context |

### Important Corrections from Original Proposal

| Original (Incorrect) | Correct Model | Issue |
|---------------------|---------------|-------|
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext-multilingual` | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | Model name doesn't exist; English-only version was confused with multilingual |
| `BAAI/bge-m3-large` | `BAAI/bge-m3` | No "large" variant exists; BGE-M3 IS the large model |

### Model Characteristics

#### 1. SapBERT (Multilingual) - The Biomedical Specialist

- **Base Model:** XLM-RoBERTa-base
- **Training:** UMLS 2020AB self-alignment pretraining with 4M+ biomedical concepts
- **Publications:** [NAACL'21 & ACL'21](https://github.com/cambridgeltl/sapbert)
- **Input Limit:** 25 tokens (short entity names)
- **Best For:** HPO term matching where clinical terminology precision matters

#### 2. mxbai-embed-large-v1 - The High-Performance Generalist

- **Architecture:** BERT-large (24 layers, 16 attention heads)
- **Performance:** Outperforms OpenAI text-embedding-3-large on MTEB
- **Features:** Matryoshka representations (flexible dimension reduction), binary quantization support
- **Note:** Requires query prompt prefix for optimal retrieval:
  ```
  "Represent this sentence for searching relevant passages: "
  ```
- **Best For:** Testing if raw embedding power beats domain specialization

#### 3. BGE-M3 - The Multi-Function Multilingual Model

- **Features:**
  - Dense retrieval (standard)
  - Sparse retrieval (lexical matching, BM25-like)
  - Multi-vector retrieval (ColBERT-style)
- **Context Window:** 8192 tokens (excellent for long clinical notes)
- **Memory:** ~6.6GB VRAM required
- **Best For:** Long clinical notes and cross-lingual queries

### Hardware Compatibility

Based on current BioLORD-2023-M (~420MB) operation:

| Model | VRAM Required | CPU RAM | Compatibility |
|-------|--------------|---------|---------------|
| SapBERT-multilingual | ~2.2GB | ~4GB | ✅ Safe for all systems |
| mxbai-embed-large-v1 | ~2.6GB | ~5GB | ✅ Safe for all systems |
| BAAI/bge-m3 | ~6.6GB | ~10GB | ⚠️ High-memory systems only |

---

## Implementation Phases

### Phase 1: Infrastructure Fix (Critical - Required First)

**Objective:** Make embedding dimension detection dynamic to support any model.

#### Task 1.1: Update ChromaDB Indexer

**File:** `phentrieve/indexing/chromadb_indexer.py`

**Current Code (Lines 19-22):**
```python
from phentrieve.utils import (
    get_embedding_dimension,
    get_model_slug,
)
```

**Change To:**
```python
from phentrieve.utils import (
    get_model_slug,
)
```

**Current Code (Line 121):**
```python
model_dimension = get_embedding_dimension(model_name)
```

**Change To:**
```python
model_dimension = model.get_sentence_embedding_dimension()
```

**Rationale:** The `SentenceTransformer.get_sentence_embedding_dimension()` method ([documentation](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)) dynamically returns the actual embedding dimension from the loaded model, eliminating the need for a hardcoded lookup table.

#### Task 1.2: Deprecate Static Dimension Function

**File:** `phentrieve/utils.py`

Add deprecation warning to `get_embedding_dimension()` function:

```python
import warnings

def get_embedding_dimension(model_name: str) -> int:
    """
    Get the embedding dimension for a given model.

    .. deprecated::
        Use `model.get_sentence_embedding_dimension()` instead for dynamic
        dimension detection. This function will be removed in a future version.

    Args:
        model_name: String representing the sentence-transformer model name

    Returns:
        The embedding dimension as an integer
    """
    warnings.warn(
        "get_embedding_dimension() is deprecated. Use "
        "model.get_sentence_embedding_dimension() for dynamic dimension detection.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Keep existing logic for backwards compatibility
    dimension_map: dict[str, int] = {
        "sentence-transformers/distiluse-base-multilingual-cased-v2": 512,
        "BAAI/bge-m3": 1024,
        "sentence-transformers/LaBSE": 768,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        # Add new models for fallback
        "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": 768,
        "mixedbread-ai/mxbai-embed-large-v1": 1024,
    }
    return dimension_map.get(model_name, 768)
```

#### Task 1.3: Unit Tests

**File:** `tests/unit/test_chromadb_indexer.py`

```python
import pytest
from unittest.mock import MagicMock, patch


def test_dynamic_dimension_detection():
    """Verify dimension is read from model, not hardcoded."""
    from sentence_transformers import SentenceTransformer

    # Use a small, known model for testing
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dimension = model.get_sentence_embedding_dimension()
    assert dimension == 384  # Known dimension for this model


@pytest.mark.parametrize(
    "model_name,expected_dim",
    [
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
        ("sentence-transformers/all-mpnet-base-v2", 768),
    ],
)
def test_various_model_dimensions(model_name, expected_dim):
    """Test dimension detection for various models."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    assert model.get_sentence_embedding_dimension() == expected_dim
```

#### Task 1.4: Verify Existing Tests Pass

```bash
make test
make typecheck-fast
make check
```

### Phase 2: Model Configuration

**Objective:** Add new models to configuration and ensure smooth loading.

#### Task 2.1: Update phentrieve.yaml.template

Add new models to benchmark configuration:

```yaml
# Benchmark models for comparison
benchmark:
  models:
    - "FremyCompany/BioLORD-2023-M"  # Current default (MRR@10: 0.823)
    - "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"  # Biomedical multilingual
    - "mixedbread-ai/mxbai-embed-large-v1"  # SOTA generalist
    # Uncomment for high-memory systems (requires ~6.6GB VRAM):
    # - "BAAI/bge-m3"  # Large multilingual model
```

#### Task 2.2: Update Dimension Map Fallback (Optional)

If keeping the dimension map as fallback, add correct entries to `phentrieve/utils.py`:

```python
dimension_map: dict[str, int] = {
    # Existing entries
    "sentence-transformers/distiluse-base-multilingual-cased-v2": 512,
    "BAAI/bge-m3": 1024,
    "sentence-transformers/LaBSE": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    # New entries
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": 768,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
}
```

### Phase 3: Integration Testing

**Objective:** Verify new models load and index correctly.

#### Task 3.1: Model Loading Tests

**File:** `tests/integration/test_new_embedding_models.py`

```python
"""Integration tests for new embedding models."""

import pytest
from phentrieve.embeddings import load_embedding_model


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name,expected_dim",
    [
        ("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR", 768),
        ("mixedbread-ai/mxbai-embed-large-v1", 1024),
    ],
)
def test_model_loads_with_correct_dimension(model_name: str, expected_dim: int):
    """Verify new models load and report correct dimensions."""
    model = load_embedding_model(model_name, device="cpu")
    assert model.get_sentence_embedding_dimension() == expected_dim


@pytest.mark.slow
def test_sapbert_encodes_biomedical_terms():
    """Verify SapBERT can encode biomedical entity names."""
    model = load_embedding_model(
        "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        device="cpu",
    )
    embeddings = model.encode(["Hypotonie", "Muskelhypotonie", "Seizure"])
    assert embeddings.shape == (3, 768)


@pytest.mark.slow
def test_mxbai_encodes_with_query_prompt():
    """Verify mxbai model works with query prompt."""
    model = load_embedding_model(
        "mixedbread-ai/mxbai-embed-large-v1",
        device="cpu",
    )
    query = "Represent this sentence for searching relevant passages: muscle weakness"
    embeddings = model.encode([query])
    assert embeddings.shape == (1, 1024)
```

#### Task 3.2: Index Building Test

```python
@pytest.mark.slow
def test_index_build_with_new_model(tmp_path):
    """Verify index building works with new model dimensions."""
    from phentrieve.embeddings import load_embedding_model
    from phentrieve.indexing.chromadb_indexer import build_chromadb_index

    model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    model = load_embedding_model(model_name, device="cpu")

    # Minimal test data
    documents = ["Hypotonie", "Muskelschwäche", "Krampfanfälle"]
    metadatas = [{"hpo_id": f"HP:000000{i}"} for i in range(3)]
    ids = [f"term_{i}" for i in range(3)]

    success = build_chromadb_index(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        model=model,
        model_name=model_name,
        index_dir=tmp_path / "test_index",
        recreate=True,
    )

    assert success is True
```

### Phase 4: Benchmarking

**Objective:** Run comparative benchmarks and collect metrics.

#### Task 4.1: Build Indexes for New Models

```bash
# Build indexes (run sequentially to avoid memory issues)
phentrieve index build --model "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
phentrieve index build --model "mixedbread-ai/mxbai-embed-large-v1"

# For high-memory systems only:
# phentrieve index build --model "BAAI/bge-m3"
```

#### Task 4.2: Run Benchmarks

```bash
# Run on small dataset first
phentrieve benchmark run --test-file german/tiny_v1.json

# Run on medium dataset for more reliable metrics
phentrieve benchmark run --test-file german/70cases_gemini_v1.json

# Compare results
phentrieve benchmark compare

# Generate visualizations
phentrieve benchmark visualize
```

#### Task 4.3: Collect Metrics

Document the following metrics for each model:
- MRR@10 (primary metric)
- Hit@1, Hit@3, Hit@5, Hit@10
- Index build time
- Query latency (average, P95)
- Memory usage

### Phase 5: Optional Enhancements

#### Task 5.1: Query Prompting for mxbai-embed-large-v1

If benchmarks show improvement potential, add query prompt support:

**File:** `phentrieve/retrieval/dense_retriever.py`

```python
# Query prompt prefixes for models that benefit from them
QUERY_PROMPTS: dict[str, str] = {
    "mixedbread-ai/mxbai-embed-large-v1": (
        "Represent this sentence for searching relevant passages: "
    ),
}


def _prepare_query(self, query: str) -> str:
    """Prepare query with model-specific prompt if needed."""
    prompt = QUERY_PROMPTS.get(self.model_name, "")
    return prompt + query
```

#### Task 5.2: BGE-M3 Hybrid Retrieval (Future)

For advanced users, expose hybrid dense+sparse retrieval:

```python
# Future enhancement: BGE-M3 hybrid mode
# This would require changes to the retrieval pipeline
# to support sparse retrieval alongside dense
```

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| BGE-M3 OOM on low-VRAM systems | High | Medium | Document requirements, default to CPU, provide warnings |
| Dimension mismatch corrupts index | Critical | Low | Dynamic detection eliminates this risk |
| Query prompting adds latency | Low | High | Optional feature, benchmark first |
| Existing indexes become invalid | Medium | None | Indexes are versioned by model slug (already implemented) |
| SapBERT 25-token limit truncates queries | Medium | Medium | Document limitation, suitable for entity names |

---

## Rollback Plan

### Phase 1 Rollback

If dynamic dimension detection causes issues:

```bash
# Revert chromadb_indexer.py changes
git checkout HEAD -- phentrieve/indexing/chromadb_indexer.py

# Re-run tests
make test
```

### Phase 2-4 Rollback

Model configuration changes are additive and don't affect existing functionality:

```bash
# Simply remove new model entries from phentrieve.yaml
# Existing BioLORD indexes remain functional
```

---

## Dependencies

### Prerequisites

- [ ] Phase 1 must complete before Phase 3-4 (infrastructure before benchmarking)
- [ ] Sufficient disk space for new indexes (~500MB per model)
- [ ] VRAM availability documented for users

### Related Issues/PRs

- Related: Current hardcoded dimension approach in `get_embedding_dimension()`
- Blocks: Any future model additions without this fix

---

## Timeline Estimate

| Phase | Tasks | Complexity |
|-------|-------|------------|
| Phase 1 | Infrastructure fix | Low (3-5 code changes) |
| Phase 2 | Configuration | Low (config file updates) |
| Phase 3 | Integration tests | Medium (new test files) |
| Phase 4 | Benchmarking | Medium (execution time) |
| Phase 5 | Optional enhancements | Low priority |

---

## Resources

### Documentation

- [SentenceTransformer API Documentation](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)
- [SapBERT GitHub Repository](https://github.com/cambridgeltl/sapbert)
- [mxbai-embed-large-v1 Model Card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- [Mixedbread Blog: Model Release](https://www.mixedbread.com/blog/mxbai-embed-large-v1)
- [BGE-M3 Model Card](https://huggingface.co/BAAI/bge-m3)
- [BGE-M3 Paper (arXiv)](https://arxiv.org/abs/2402.03216)

### Related Plans

- `plan/02-completed/MASTER-PLAN.md` - Tooling modernization context
- `plan/02-completed/TESTING-MODERNIZATION-PLAN.md` - Testing patterns

---

## Appendix: Model Comparison Summary

### Expected Performance Characteristics

| Aspect | BioLORD (Current) | SapBERT | mxbai-embed | BGE-M3 |
|--------|-------------------|---------|-------------|--------|
| **Domain Specificity** | Biomedical | Biomedical (UMLS) | General | General |
| **Multilingual** | Yes | Yes (100+) | English only | Yes (100+) |
| **Dimensions** | 768 | 768 | 1024 | 1024 |
| **Max Tokens** | 512 | 25 | 512 | 8192 |
| **Memory** | ~420MB | ~1.1GB | ~1.3GB | ~3.3GB |
| **Expected Strength** | Clinical text | Entity names | General retrieval | Long documents |

### Benchmark Priority

1. **SapBERT-multilingual** - Highest priority (targeted for HPO domain)
2. **mxbai-embed-large-v1** - Second priority (test generalist vs specialist)
3. **BGE-M3** - Third priority (high-memory systems only)

---

**Last Updated:** 2025-12-10
**Status:** Draft - Awaiting Review
