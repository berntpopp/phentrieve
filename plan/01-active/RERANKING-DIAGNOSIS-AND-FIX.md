# Re-ranking System Diagnosis and Fix Plan

**Status:** 📋 Analysis Complete - Ready for Implementation
**Created:** 2025-11-24
**Author:** Claude Code (AI-assisted analysis)
**Priority:** High
**Issue:** Re-ranking feature "never really worked well"

---

## Executive Summary

Investigation revealed **critical architectural flaws** in the re-ranking implementation that explain why it doesn't improve retrieval quality. The core problem: **the system uses a Natural Language Inference (NLI) model where a proper re-ranker model is required**.

### Key Findings

| Finding | Severity | Impact |
|---------|----------|--------|
| Wrong model type (NLI vs Re-ranker) | 🔴 Critical | Fundamentally wrong output semantics |
| Inconsistent score handling | 🔴 Critical | NLI output misinterpreted in orchestrator |
| Score replacement vs fusion | 🟡 Medium | Discards valuable dense retrieval signal |
| Asymmetric input lengths | 🟡 Medium | Suboptimal for cross-encoder architecture |

### Recommended Actions

1. **Replace NLI model with proper re-ranker** (e.g., `BAAI/bge-reranker-v2-m3`)
2. **Fix score handling inconsistency** between `reranker.py` and `hpo_extraction_orchestrator.py`
3. **Implement score fusion** instead of replacement
4. **Add medical-domain re-ranker option** (`ncbi/MedCPT-Cross-Encoder`)

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Code Investigation](#2-code-investigation)
3. [Model Comparison](#3-model-comparison)
4. [Recommended Fixes](#4-recommended-fixes)
5. [Implementation Plan](#5-implementation-plan)
6. [Testing Strategy](#6-testing-strategy)
7. [Rollback Plan](#7-rollback-plan)

---

## 1. Problem Analysis

### 1.1 The Core Issue: NLI vs Re-ranker

The default re-ranker model is `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`. This is a **Natural Language Inference** model, not a re-ranker.

#### What NLI Models Do

```
Input: (premise, hypothesis)
Output: [P(entailment), P(neutral), P(contradiction)]

Example:
  Premise: "Patient presents with recurrent seizures and loss of consciousness"
  Hypothesis: "Seizures"

  Output: [0.72, 0.25, 0.03]
           ↑      ↑      ↑
     entailment neutral contradiction
```

NLI answers: "Does the premise logically entail the hypothesis?"

#### What Re-ranker Models Do

```
Input: (query, document)
Output: relevance_score (single float)

Example:
  Query: "Patient presents with recurrent seizures and loss of consciousness"
  Document: "Seizures"

  Output: 0.89 (relevance score)
```

Re-rankers answer: "How relevant is this document to the query?"

#### Why This Matters for HPO Retrieval

| Scenario | NLI Interpretation | Re-ranker Interpretation |
|----------|-------------------|-------------------------|
| "Patient has seizures" → "Seizures" | Entailment (correct) | High relevance (correct) |
| "Patient has seizures" → "Epilepsy" | Neutral (wrong!) | High relevance (correct) |
| "No seizures observed" → "Seizures" | Contradiction (useful) | Low relevance (correct) |

The NLI model marks semantically related but not logically entailed terms as "neutral", giving them low scores. A re-ranker understands semantic relevance beyond logical entailment.

### 1.2 Score Interpretation Problem

Current code assumes `scores[0]` is the relevance score, but for NLI models:
- `scores[0]` = entailment probability
- `scores[1]` = neutral probability
- `scores[2]` = contradiction probability

Using entailment probability as relevance score is semantically incorrect.

---

## 2. Code Investigation

### 2.1 Model Configuration

**Location:** `phentrieve/config.py:66-71`

```python
_DEFAULT_RERANKER_MODEL_FALLBACK = (
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"  # ← NLI model!
)
_DEFAULT_MONOLINGUAL_RERANKER_MODEL_FALLBACK = (
    "ml6team/cross-encoder-mmarco-german-distilbert-base"  # ← This IS a re-ranker
)
```

**Issue:** The multilingual default is an NLI model, while the monolingual German default is a proper re-ranker. Inconsistent.

### 2.2 Score Handling in reranker.py

**Location:** `phentrieve/retrieval/reranker.py:91-99`

```python
for i, candidate in enumerate(candidates):
    # Handle different output formats from various cross-encoder models
    if isinstance(scores[i], (list, np.ndarray)) and len(scores[i]) > 1:
        # For NLI models that return probabilities for entailment/neutral/contradiction
        # Use the entailment score (usually the first index) as the relevance score
        candidate["cross_encoder_score"] = float(scores[i][0])
    else:
        # For traditional cross-encoders that return a single score
        candidate["cross_encoder_score"] = float(scores[i])
```

**Status:** Attempts to handle NLI output, but using entailment as relevance is conceptually wrong.

### 2.3 Score Handling in hpo_extraction_orchestrator.py

**Location:** `phentrieve/text_processing/hpo_extraction_orchestrator.py:154-165`

```python
# Get cross-encoder scores
scores = cross_encoder.predict(
    pairs,
    show_progress_bar=False,
)

# Add scores to candidates
for idx, match in enumerate(current_hpo_matches[:]):
    match["score"] = float(scores[idx])  # ← NO NLI handling!

# Sort by score
current_hpo_matches.sort(key=lambda x: x["score"], reverse=True)
```

**Critical Bug:** This code does NOT handle NLI model output. When the NLI model returns `[0.72, 0.25, 0.03]`, this code tries to convert the entire array to float, which either:
1. Fails with an error
2. Uses only the first element implicitly
3. Produces unexpected behavior

### 2.4 Score Replacement Problem

Both locations **replace** the original dense retrieval score:

```python
# Original score from BioLORD embedding similarity: 0.85
# After re-ranking:
match["score"] = float(scores[idx])  # Now 0.72 (entailment prob)
# Original 0.85 similarity is LOST
```

This discards valuable semantic similarity information in favor of a (mis)interpreted NLI score.

### 2.5 Input Asymmetry

**Location:** `phentrieve/text_processing/hpo_extraction_orchestrator.py:150-151`

```python
pairs = [(chunk_text, match["name"]) for match in current_hpo_matches]
```

Example pair:
```
(
  "Der Patient zeigt wiederkehrende Krampfanfälle mit Bewusstseinsverlust,
   die seit dem Kindesalter bestehen. Keine familiäre Vorgeschichte bekannt.",

  "Seizures"
)
```

Cross-encoders work best with balanced input lengths. This extreme asymmetry (paragraph vs single word) may degrade performance.

---

## 3. Model Comparison

### 3.1 Available Re-ranker Models

| Model | Type | Multilingual | Medical | Size | Speed |
|-------|------|--------------|---------|------|-------|
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | NLI ❌ | ✅ 100+ langs | ❌ | 278M | Medium |
| `BAAI/bge-reranker-v2-m3` | Re-ranker ✅ | ✅ 100+ langs | ❌ | 568M | Medium |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranker ✅ | ❌ English | ❌ | 22M | Fast |
| `ncbi/MedCPT-Cross-Encoder` | Re-ranker ✅ | ❌ English | ✅ **Yes** | 110M | Medium |
| `BAAI/bge-reranker-v2-gemma` | Re-ranker ✅ | ✅ | ❌ | 2B | Slow |
| `ml6team/cross-encoder-mmarco-german-distilbert-base` | Re-ranker ✅ | German only | ❌ | 66M | Fast |

### 3.2 Recommended Models for Phentrieve

**Primary Recommendation: `BAAI/bge-reranker-v2-m3`**
- Proper re-ranker architecture (outputs single relevance score)
- Multilingual (100+ languages) - matches Phentrieve's multilingual support
- Good balance of quality and speed
- Active maintenance by BAAI

**Medical Domain Option: `ncbi/MedCPT-Cross-Encoder`**
- Trained on medical/clinical text
- Would excel at HPO term matching
- English only - use in cross-lingual mode with translated HPO terms

### 3.3 Output Format Comparison

```python
# NLI Model (current - WRONG)
model = CrossEncoder("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
scores = model.predict([("query", "document")])
# scores = [[0.72, 0.25, 0.03]]  ← 3 probabilities

# Proper Re-ranker (recommended)
model = CrossEncoder("BAAI/bge-reranker-v2-m3")
scores = model.predict([("query", "document")])
# scores = [0.89]  ← single relevance score
```

---

## 4. Recommended Fixes

### 4.1 Fix 1: Replace Default Re-ranker Model (Critical)

**File:** `phentrieve/config.py`

```python
# Before:
_DEFAULT_RERANKER_MODEL_FALLBACK = (
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)

# After:
_DEFAULT_RERANKER_MODEL_FALLBACK = (
    "BAAI/bge-reranker-v2-m3"
)
```

### 4.2 Fix 2: Add NLI Handling to Orchestrator (Critical)

**File:** `phentrieve/text_processing/hpo_extraction_orchestrator.py`

```python
# Before (line 160-162):
for idx, match in enumerate(current_hpo_matches[:]):
    match["score"] = float(scores[idx])

# After:
for idx, match in enumerate(current_hpo_matches[:]):
    raw_score = scores[idx]
    # Handle NLI model output (3 probabilities) vs re-ranker (single score)
    if isinstance(raw_score, (list, np.ndarray)) and len(raw_score) > 1:
        # NLI model: use entailment probability (index 0)
        # Note: This is suboptimal - proper re-ranker recommended
        match["rerank_score"] = float(raw_score[0])
    else:
        # Proper re-ranker: single relevance score
        match["rerank_score"] = float(raw_score)
```

### 4.3 Fix 3: Implement Score Fusion (Medium Priority)

**File:** `phentrieve/text_processing/hpo_extraction_orchestrator.py`

```python
# Instead of replacing score, combine dense + rerank scores
RERANK_WEIGHT = 0.4  # Configurable

for idx, match in enumerate(current_hpo_matches[:]):
    original_score = match["score"]  # Dense retrieval score
    rerank_score = _extract_rerank_score(scores[idx])

    # Normalize rerank score to [0, 1] if needed (sigmoid for unbounded scores)
    if rerank_score < 0 or rerank_score > 1:
        rerank_score = 1 / (1 + np.exp(-rerank_score))  # Sigmoid normalization

    # Score fusion
    match["dense_score"] = original_score
    match["rerank_score"] = rerank_score
    match["score"] = (1 - RERANK_WEIGHT) * original_score + RERANK_WEIGHT * rerank_score
```

### 4.4 Fix 4: Add Medical Re-ranker Option (Enhancement)

**File:** `phentrieve/config.py`

```python
# Add new config option
_DEFAULT_MEDICAL_RERANKER_MODEL_FALLBACK = "ncbi/MedCPT-Cross-Encoder"

# In CLI, add option:
# --reranker-model medical  → uses MedCPT
# --reranker-model multilingual  → uses bge-reranker-v2-m3
```

---

## 5. Implementation Plan

### Phase 1: Critical Fixes (Immediate)

| Task | File | Effort | Risk |
|------|------|--------|------|
| Replace default model | `config.py` | 5 min | Low |
| Fix orchestrator score handling | `hpo_extraction_orchestrator.py` | 30 min | Medium |
| Update CLI defaults | `text_commands.py`, `query_commands.py` | 15 min | Low |
| Add unit tests | `tests/unit/retrieval/` | 1 hour | Low |

### Phase 2: Score Fusion (Short-term)

| Task | File | Effort | Risk |
|------|------|--------|------|
| Implement score fusion | `hpo_extraction_orchestrator.py` | 1 hour | Medium |
| Add fusion weight config | `config.py`, `phentrieve.yaml.template` | 30 min | Low |
| Benchmark fusion vs replacement | `tests/integration/` | 2 hours | Low |

### Phase 3: Medical Re-ranker (Medium-term)

| Task | File | Effort | Risk |
|------|------|--------|------|
| Add MedCPT support | `config.py`, CLI files | 1 hour | Low |
| Benchmark MedCPT vs BGE | Benchmark framework | 2 hours | Low |
| Documentation update | `CLAUDE.md`, README | 30 min | Low |

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# tests/unit/retrieval/test_reranker_models.py

def test_bge_reranker_output_format():
    """BGE reranker should return single float scores."""
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    scores = model.predict([("query", "doc")])
    assert isinstance(scores[0], (int, float))
    assert not isinstance(scores[0], (list, np.ndarray))

def test_nli_model_output_format():
    """NLI model returns 3 probabilities - verify handling."""
    model = CrossEncoder("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    scores = model.predict([("query", "doc")])
    assert len(scores[0]) == 3  # entailment, neutral, contradiction

def test_score_fusion():
    """Score fusion should combine dense and rerank scores."""
    dense_score = 0.85
    rerank_score = 0.72
    weight = 0.4
    expected = 0.6 * 0.85 + 0.4 * 0.72  # 0.798
    # ... test implementation
```

### 6.2 Integration Tests

```python
# tests/integration/test_reranking_quality.py

def test_reranking_improves_mrr():
    """Re-ranking with proper model should improve MRR."""
    # Run benchmark with and without re-ranking
    # Assert MRR improves or stays same (never degrades significantly)

def test_german_clinical_text_reranking():
    """German clinical text should be properly re-ranked."""
    text = "Patient zeigt wiederkehrende Krampfanfälle"
    # Verify "Seizures" ranks higher after re-ranking
```

### 6.3 Benchmark Comparison

Create benchmark comparing:
1. Dense retrieval only (baseline)
2. Dense + NLI model (current broken state)
3. Dense + BGE reranker (recommended fix)
4. Dense + MedCPT (medical domain)
5. Dense + score fusion

---

## 7. Rollback Plan

### 7.1 Configuration Rollback

If new re-ranker causes issues:

```yaml
# phentrieve.yaml - revert to NLI model (with known limitations)
reranker_model: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
```

### 7.2 Code Rollback

All changes are backward compatible:
- Score fusion preserves original score in `dense_score` field
- NLI handling is additive (doesn't break existing behavior)
- Model change is configuration-only

### 7.3 Feature Flag

Consider adding feature flag for gradual rollout:

```python
# config.py
USE_NEW_RERANKER = get_config_value("use_new_reranker", False)
```

---

## 8. References

### 8.1 Model Documentation

- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) - Recommended multilingual re-ranker
- [ncbi/MedCPT-Cross-Encoder](https://huggingface.co/ncbi/MedCPT-Cross-Encoder) - Medical domain re-ranker
- [mDeBERTa-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) - Current NLI model (not a re-ranker)

### 8.2 Related Issues

- Re-ranking "never really worked well" - User report (2025-11-24)

### 8.3 Related Files

| File | Purpose |
|------|---------|
| `phentrieve/config.py:66-71` | Default model configuration |
| `phentrieve/retrieval/reranker.py` | Re-ranking implementation |
| `phentrieve/text_processing/hpo_extraction_orchestrator.py:147-167` | Orchestrator re-ranking logic |
| `phentrieve/cli/text_commands.py:138-174` | CLI re-ranking options |
| `phentrieve/cli/query_commands.py:93-114` | Query CLI re-ranking options |

---

## Appendix A: Quick Diagnostic Test

Run this to verify the problem:

```python
from sentence_transformers import CrossEncoder

# Current model (NLI - wrong type)
nli_model = CrossEncoder("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
nli_scores = nli_model.predict([
    ("Patient has seizures", "Seizures"),
    ("Patient has seizures", "Epilepsy"),
])
print(f"NLI output: {nli_scores}")
# Expected: [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]] - 3 probs each

# Recommended model (proper re-ranker)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
rerank_scores = reranker.predict([
    ("Patient has seizures", "Seizures"),
    ("Patient has seizures", "Epilepsy"),
])
print(f"Reranker output: {rerank_scores}")
# Expected: [0.92, 0.85] - single relevance scores
```

---

## Appendix B: Embedding Gemma Consideration

During this investigation, `google/embeddinggemma-300m` was also evaluated. **It is NOT suitable for re-ranking** because:

1. It's a **bi-encoder** (encodes query and document separately)
2. Re-ranking requires **cross-encoders** (joint encoding for attention between query and document)
3. It uses asymmetric encoding (`encode_query` vs `encode_document`) which adds complexity

However, Embedding Gemma could be considered for:
- **Initial dense retrieval** (replacing BioLORD for multilingual scenarios)
- **Semantic chunking** similarity computation

See separate analysis if pursuing Embedding Gemma for retrieval.
