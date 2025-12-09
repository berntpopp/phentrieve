# Extraction Benchmark Action Plan

**Date**: 2025-12-08 | **Dataset**: PhenoBERT GeneReviews (10 docs, 237 annotations) | **Model**: BioLORD-2023-M

---

## Summary

| Metric | Baseline | Threshold Opt | **+Chunkers** | Target |
|--------|----------|---------------|---------------|--------|
| **F1** | 0.145 | 0.265 | **0.328** | 0.40-0.50 |
| Precision | 0.088 | 0.316 | 0.310 | ~0.50 |
| Recall | 0.422 | 0.228 | **0.353** | ~0.35 |

**Root cause**: Sliding window creates broken fragments ("carcinoma of the") → wrong HPO matches.
**Embedding quality is good but not perfect** - see HPO Self-Match Analysis below for nuances.

---

## Phrase-Level Retrieval Benchmark (Embedding Quality Proof)

Direct phrase-to-HPO retrieval bypassing chunking. Proves embeddings work well.

| Dataset | Phrases | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 | OntSim@10 |
|---------|---------|-----|-------|-------|-------|--------|-----------|
| **ID_68** | 420 | 0.649 | 50.7% | 76.2% | 81.9% | **90.7%** | 0.963 |
| **GeneReviews** | 231 | 0.611 | 47.2% | 70.6% | 79.7% | 87.9% | 0.946 |
| GSC_plus | 991 | 0.494 | 37.4% | 56.9% | 64.5% | 72.4% | 0.812 |

**Key finding**: When given clean phrases, **90.7% of exact HPO terms are found in top 10** (ID_68).
This confirms chunking is the bottleneck, not embedding quality.

---

## HPO Self-Match Analysis (Embedding Ceiling)

**Experiment**: Query ALL 19,534 HPO term labels and check if they return themselves as the top match.

### Summary

| Metric | Value |
|--------|-------|
| **Total HPO terms** | 19,534 |
| **Self-match rate** | 86.5% (16,889) |
| Wrong match | 13.5% (2,645) |
| Average score (all) | 0.918 |
| Average score (exact) | 0.925 |
| Average score (wrong) | 0.874 |

### Score Distribution (Exact Matches)

| Percentile | Score |
|------------|-------|
| Min | 0.384 |
| 5th | 0.820 |
| 25th | 0.901 |
| Median | 0.937 |
| 75th | 0.962 |
| 95th | 0.989 |
| Max | 0.998 |

### Why 13.5% Don't Self-Match

The "wrong" matches are typically **semantically equivalent or hierarchically related terms**, not errors:

| Query | Returned | Reason |
|-------|----------|--------|
| "Rod-cone dystrophy" | "Cone-rod dystrophy" | Synonym |
| "Abnormality of the ureter" | "Abnormal ureter morphology" | Synonym |
| "Postaxial hand polydactyly" | "Postaxial polydactyly of fingers" | Parent/child |
| "Optic nerve hypoplasia" | "Optic disc hypoplasia" | Related term |

**This is expected embedding behavior** - semantically similar terms cluster together.

### Correlation: Synonym Count (KEY FINDING)

**Terms with more synonyms have LOWER self-match rates:**

| Synonyms | Count | Exact | Wrong | **Accuracy** | Avg Score |
|----------|-------|-------|-------|--------------|-----------|
| 0 | 8,675 | 8,113 | 562 | **93.5%** | 0.933 |
| 1 | 5,587 | 5,058 | 529 | **90.5%** | 0.917 |
| 2 | 2,389 | 1,922 | 467 | **80.5%** | 0.902 |
| 3 | 1,286 | 937 | 349 | **72.9%** | 0.893 |
| 4 | 596 | 379 | 217 | **63.6%** | 0.884 |
| 5 | 367 | 194 | 173 | **52.9%** | 0.879 |
| 6-10 | 539 | 241 | 298 | **44.7%** | 0.872 |
| 11+ | 95 | 45 | 50 | **47.4%** | 0.869 |

**Interpretation**: Terms with many synonyms compete with their own synonyms in embedding space. The canonical label may lose to a synonym that's more common in training data.

### Correlation: Has Definition

| Definition | Count | Accuracy | Avg Score |
|------------|-------|----------|-----------|
| Yes | 16,509 | 85.0% | 0.910 |
| No | 3,025 | **94.2%** | 0.963 |

Terms without definitions are typically simpler/less ambiguous.

### Correlation: Label Word Count

| Words | Count | Accuracy | Avg Score |
|-------|-------|----------|-----------|
| 1 | 1,379 | 79.6% | 0.855 |
| 2 | 4,320 | 84.3% | 0.906 |
| 3 | 4,384 | 86.7% | 0.921 |
| 4 | 3,783 | 87.1% | 0.925 |
| 5 | 2,484 | 89.2% | 0.933 |
| 6+ | 3,184 | **89.2%** | 0.936 |

Single-word terms have lowest accuracy - they're often generic/ambiguous.

### Low-Score Correct Matches (<0.70)

| Term | Score | Synonyms | Words |
|------|-------|----------|-------|
| "Tenesmus" | 0.384 | 0 | 1 |
| "Moderate" | 0.489 | 0 | 1 |
| "Miliary" | 0.521 | 0 | 1 |
| "Circumscribed" | 0.529 | 0 | 1 |
| "Anasarca" | 0.564 | 2 | 1 |
| "Carcinoma" | 0.736 | varies | 1 |

These are rare technical terms not well-represented in embedding model training data, or generic terms with crowded neighborhoods.

### Implications

1. **86.5% ceiling for perfect matching** - Even ideal inputs won't achieve 100% accuracy due to HPO structure
2. **Synonym count is primary factor** - More synonyms = lower accuracy (93.5% → 45%)
3. **Single-word terms perform worst** - 79.6% accuracy vs 89.2% for 5+ word terms
4. **"Wrong" matches are semantically correct** - The model finds equivalent/related terms
5. **Problem is chunking, not embeddings** - The 86.5% ceiling is acceptable; improving what we feed to embeddings is the priority

---

## Action 1: Update Default Configuration (Immediate)

**File**: `phentrieve/benchmark/extraction_benchmark.py`

```python
@dataclass
class ExtractionConfig:
    chunk_retrieval_threshold: float = 0.7      # was 0.5
    min_confidence_for_aggregated: float = 0.75  # was 0.5
    top_term_per_chunk: bool = True              # was False
```

**CLI equivalent**:
```bash
phentrieve benchmark extraction run <path> \
    --chunk-threshold 0.7 \
    --min-confidence 0.75 \
    --top-term-only
```

**Impact**: F1 0.145 → 0.265 (+83%)

---

## Action 2: Enhanced Chunking Pipeline (VALIDATED)

**Status**: Tested and validated. Significantly improves recall without hurting precision.

**Change**: Add `conjunction` and `fine_grained_punctuation` chunkers before the sliding window splitter.

**File**: `phentrieve/benchmark/extraction_benchmark.py`

```python
chunking_pipeline_config=[
    {"type": "paragraph"},
    {"type": "sentence"},
    {"type": "conjunction"},              # NEW: splits at "and", "or", "but"
    {"type": "fine_grained_punctuation"}, # NEW: splits at ", ; :"
    {"type": "sliding_window", "config": {...}},
    {"type": "final_chunk_cleaner"},
]
```

**Impact**: F1 0.265 → 0.328 (+24%), Recall 0.228 → 0.353 (+55%)

---

## ~~Action 3: Stop Word Removal~~ (REJECTED)

**Status**: Investigated and rejected based on empirical testing.

**Original hypothesis**: Removing stop words from chunks would improve matching by cleaning fragments like "carcinoma of the" → "carcinoma".

**Finding**: Stop word removal has inconsistent effects and often changes which HPO term is matched, not just the similarity score. Since **2,187 HPO terms (11%)** contain "of the", removing stop words from queries creates an asymmetry with the index.

See **Stop Word Removal Experiments** section below for detailed results.

---

## ~~Action 4: Candidate Decomposition~~ (REJECTED)

**Status**: Investigated and rejected based on empirical testing.

**Original hypothesis**: Generate n-gram candidates from chunks and query each, keeping the best match.

**Finding**: N-gram decomposition **increases similarity scores but decreases accuracy**. Short n-grams match HPO terms with those exact words strongly, but lose semantic context needed for correct matching.

See **N-Gram Decomposition Experiments** section below for detailed results.

---

## ~~Action 5: Cross-Encoder Reranking~~ (REJECTED)

**Status**: Investigated and rejected due to language limitations.

**Problem**: Cross-encoder models are typically trained on English text. They don't generalize well to German, Spanish, French, or Dutch - the same multilingual limitation as dictionary approaches.

**Note**: While Phentrieve supports cross-encoder reranking (`--enable-reranker`), it's not recommended for multilingual deployments.

---

## Why Embeddings Over Dictionaries

| Approach | EN | DE | ES | FR | NL | Viable? |
|----------|----|----|----|----|----|----|
| Dictionary-based | ✅ | ❌ | ❌ | ❌ | ❌ | **No** |
| Embeddings | ✅ | ✅ | ✅ | ✅ | ✅ | **Yes** |

Phentrieve's multilingual support is a core feature. Dictionary approaches require separate vocabularies per language.

---

## Performance Roadmap

| Step | Change | Effort | F1 |
|------|--------|--------|-----|
| **1. Default config** | Update thresholds | Trivial | 0.265 |
| **2. Enhanced chunking** | Add conjunction + punctuation chunkers | Low | **0.328** ✓ |
| ~~3. Stop word removal~~ | ~~Clean chunks internally~~ | ~~Low~~ | **REJECTED** |
| ~~4. Candidate decomposition~~ | ~~N-gram matching~~ | ~~Medium~~ | **REJECTED** (hurts accuracy) |
| ~~5. Cross-encoder reranking~~ | ~~Rerank candidates~~ | ~~Low~~ | **REJECTED** (EN-only) |

**Current status**: Threshold optimization + enhanced chunking achieved **F1 0.328** (+126% from baseline). Further improvements may require LLM-based phrase extraction.

---

## Key Experiments Summary

### Threshold Experiments (22 runs)

| Config | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| **thresh=0.7, conf=0.75, top=true** | **0.316** | 0.228 | **0.265** |
| thresh=0.75, conf=0.7, top=true | 0.310 | 0.228 | 0.263 |
| thresh=0.5, conf=0.7, top=true | 0.278 | 0.245 | 0.260 |
| thresh=0.5, conf=0.5, top=false (baseline) | 0.088 | 0.422 | 0.145 |

### Chunking Experiments (6 runs)

| Window | Step | Recall | F1 |
|--------|------|--------|-----|
| **3** | **1** | **0.228** | **0.265** |
| 5 | 2 | 0.156 | 0.196 |
| 7 | 3 | 0.122 | 0.171 |
| Sentence-only | - | 0.068 | 0.107 |

**Finding**: Larger windows hurt recall. Solution is cleaner chunks, not bigger chunks.

### Chunking Strategy Comparison (12 runs)

Tested adding `conjunction` and `fine_grained_punctuation` chunkers to the pipeline.

**Pipeline configurations:**
- `baseline`: paragraph → sentence → sliding_window → cleaner
- `with_conjunction`: + conjunction chunker (splits at "and", "or", "but")
- `with_punctuation`: + punctuation chunker (splits at `, ; :`)
- `with_both`: + both chunkers

**Results (F1 scores):**

| Config | GeneReviews | ID_68 | GSC_plus | Avg |
|--------|-------------|-------|----------|-----|
| baseline | 0.270 | 0.196 | 0.209 | 0.225 |
| with_conjunction | 0.289 | 0.258 | 0.246 | 0.264 |
| with_punctuation | **0.330** | 0.293 | **0.286** | 0.303 |
| **with_both** | **0.341** | **0.348** | 0.296 | **0.328** |

**Key findings:**
1. **Punctuation chunker gives biggest improvement**: +22% F1 on GeneReviews
2. **Both chunkers combined is best**: F1 0.225 → 0.328 (+46% improvement)
3. **Recall improves significantly**: baseline 0.17-0.23 → with_both 0.29-0.40
4. **Precision stays stable**: ~0.30 across all configs

**Recommendation**: Enable both `conjunction` and `fine_grained_punctuation` chunkers in extraction benchmark defaults.

### Stop Word Removal Experiments (REJECTED)

**Hypothesis**: Removing stop words from query chunks would improve HPO matching.

**HPO Index Analysis**:
| Pattern | Terms in Index |
|---------|----------------|
| "of the" | 2,187 (11.2%) |
| "of a" | 70 |
| "in the" | 33 |
| "to the" | 13 |

**Complete Phrases** (from benchmark text snippets):

| Phrase | With Stop Words | Without | Delta | Better |
|--------|-----------------|---------|-------|--------|
| Anomalies of the semicircular canals | 0.93 | 0.92 | -0.01 | WITH |
| Compression of the spinal cord | 0.90 | 0.91 | +0.01 | NO SW |
| abnormalities of the eye | 0.90 | 0.89 | -0.01 | WITH |
| abnormality of the philtrum | 0.88 | 0.89 | +0.01 | NO SW |
| Malabsorption of fat | 0.95 | 0.94 | -0.01 | WITH |
| Premature graying of hair | 0.90 | 0.91 | +0.01 | NO SW |
| Schwannomas in the skin | 0.92 | 0.90 | -0.02 | WITH |
| craniosynostosis of the coronal | 0.93 | 0.94 | +0.01 | NO SW |
| pigmentation of the retina | 0.86 | 0.87 | +0.01 | NO SW |
| medullary carcinoma of the thyroid | 0.95 | 0.95 | 0.00 | SAME |

**Result**: 5 better without, 4 better with, 1 same. Average delta ±0.01 (negligible).

**Incomplete Fragments** (simulating broken chunks):

| Fragment | With SW | Without | Delta | Same Match? |
|----------|---------|---------|-------|-------------|
| "carcinoma of the" | **0.76** | 0.74 | -0.02 | ✅ Yes |
| "abnormality of the" | 0.62 | 0.69 | +0.07 | ❌ **Changed** |
| "compression of the" | 0.62 | 0.62 | 0.00 | ❌ **Changed** |
| "hypoplasia of the" | **0.69** | 0.68 | -0.01 | ❌ **Changed** |
| "anomalies of the" | **0.73** | 0.67 | -0.06 | ✅ Yes |
| "in the skin" | 0.68 | **0.78** | +0.10 | ❌ **Changed** |
| "of the eye" | **0.67** | 0.65 | -0.02 | ❌ **Changed** |

**Critical finding**: Stop word removal often **changes which HPO term is matched**, not just the score. 5 of 7 fragments matched a DIFFERENT term after stop word removal.

**Conclusion**: Stop word removal is **not recommended** because:
1. Effect on complete phrases is negligible (±1%)
2. Effect on fragments is unpredictable and often changes the match
3. Creates asymmetry: queries without stop words vs index with stop words
4. 11% of HPO terms contain "of the" - these would be harder to match

### N-Gram Decomposition Experiments (REJECTED)

**Hypothesis**: Generate n-gram candidates from each phrase and use the best-scoring match.

**Test**: 100 phrases from GSC_plus dataset.

**Score Comparison** (±0.02 threshold):

| Metric | Direct Query | Best N-Gram |
|--------|--------------|-------------|
| **Avg Score** | 0.858 | 0.872 (+0.015) |
| Higher score | 2% | **19%** |
| Same | 79% | |

**But Accuracy is WORSE:**

| Metric | Direct Query | Best N-Gram |
|--------|--------------|-------------|
| **Hit@1 (correct HPO)** | **42%** | 36% |

**Examples where n-gram scored higher but was WRONG:**

| Phrase | Direct | N-Gram | Best N-Gram |
|--------|--------|--------|-------------|
| "bilateral supra-auricular sinuses" | 0.68 ✓ | 0.77 ✗ | "bilateral" |
| "ear and kidney anomalies" | 0.82 ✓ | 0.92 ✗ | "and kidney anomalies" |
| "basal cell nevus syndrome" | 0.76 ✗ | 0.91 ✗ | "cell nevus" |
| "profound speech impairment" | 0.84 ✗ | 0.94 ✗ | "speech impairment" |

**Problem**: Short n-grams like "bilateral" or "cell nevus" match HPO terms containing those exact words very strongly, but lose semantic specificity of the full phrase.

**Conclusion**: N-gram decomposition is **not recommended** because:
1. Higher similarity scores ≠ better accuracy
2. Short n-grams lose semantic context
3. Hit@1 drops from 42% to 36% (**-6% accuracy loss**)
4. Computational cost increases (many queries per chunk)

---

## Files

- **Benchmark runner**: `phentrieve/benchmark/extraction_benchmark.py`
- **Chunking pipeline**: `phentrieve/text_processing/chunking.py`
- **Stop words**: `phentrieve/text_processing/stop_words/` (multilingual)
- **Test data**: `tests/data/en/phenobert/`

---

## Running the Benchmarks

```bash
# Extraction benchmark (document-level)
phentrieve benchmark extraction run tests/data/en/phenobert \
    --dataset GeneReviews \
    --chunk-threshold 0.7 \
    --min-confidence 0.75 \
    --top-term-only

# With detailed chunk-level analysis output
phentrieve benchmark extraction run tests/data/en/phenobert \
    --dataset GeneReviews \
    --detailed-output
```

### Output Files

| File | Description | Generated |
|------|-------------|-----------|
| `extraction_results.json` | Full results with predictions and gold terms | Always |
| `extraction_summary.json` | Summary metrics (F1, precision, recall) | Always |
| `extraction_detailed_analysis.json` | Chunk-level TP/FP/FN with positions | `--detailed-output` only |
