# Single-Term HPO Retrieval Benchmark Improvements

**Status:** 📋 Research Complete - Ready for Implementation
**Created:** 2025-01-21
**Author:** Claude Code (AI-assisted analysis)
**Priority:** Medium
**Scope:** Independent from Full-Text Extraction Benchmark

---

## Executive Summary

This document analyzes the **single-term HPO retrieval benchmark** used for comparing embedding models. This benchmark is **distinct from** the full-text extraction benchmark and serves a different purpose:

| Aspect | Single-Term Benchmark | Full-Text Benchmark |
|--------|----------------------|---------------------|
| **Purpose** | Compare embedding models | Evaluate extraction pipelines |
| **Input** | Single clinical phrase | Full clinical document |
| **Output** | Ranked list of HPO terms | Set of extracted HPO terms |
| **Key Metrics** | MRR, HR@K, NDCG | Precision, Recall, F1 |
| **Language** | Language-specific | Language-specific |
| **Scope** | Retrieval quality | End-to-end accuracy |

### Key Recommendations

1. **Add NDCG@K** - Industry standard metric missing from current implementation
2. **Add Recall@K** - Complement HR@K for multi-relevant scenarios
3. **Bootstrap confidence intervals** - Statistical rigor for model comparisons
4. **Language-stratified reporting** - Explicit separation by language
5. **Difficulty stratification** - Group test cases by complexity
6. **Expanded test data** - More test cases with varied difficulty

---

## Table of Contents

1. [Current Implementation Analysis](#1-current-implementation-analysis)
2. [Industry Standards: MTEB and BEIR](#2-industry-standards-mteb-and-beir)
3. [Recommended Metric Additions](#3-recommended-metric-additions)
4. [Statistical Significance](#4-statistical-significance)
5. [Language-Specific Considerations](#5-language-specific-considerations)
6. [Test Data Improvements](#6-test-data-improvements)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [References](#8-references)

---

## 1. Current Implementation Analysis

### 1.1 Existing Metrics

The current benchmark (`phentrieve/evaluation/runner.py`) implements:

| Metric | Implementation | Status |
|--------|---------------|--------|
| **MRR (Mean Reciprocal Rank)** | `mean_reciprocal_rank()` | ✅ Implemented |
| **HR@K (Hit Rate at K)** | `hit_rate_at_k()` | ✅ Implemented |
| **MaxOntSim@K** | `calculate_test_case_max_ont_sim()` | ✅ Implemented |
| **NDCG@K** | - | ❌ Missing |
| **Recall@K** | - | ❌ Missing |
| **Precision@K** | - | ❌ Missing |
| **MAP@K** | - | ❌ Missing |

### 1.2 Current MRR Implementation

```python
# From metrics.py:556-597
def mean_reciprocal_rank(results, expected_ids) -> float:
    """Find the first match and return 1/rank"""
    for hpo_id, rank, _ in ranked_ids:
        if hpo_id in expected_ids:
            return 1.0 / rank
    return 0.0
```

**Observation:** MRR only considers the **first relevant result**. This is appropriate when there's typically one correct answer, but HPO retrieval often has multiple valid terms.

### 1.3 Current Hit Rate Implementation

```python
# From metrics.py:600-641
def hit_rate_at_k(results, expected_ids, k=5) -> float:
    """Return 1.0 if ANY expected ID is in top K, else 0.0"""
    for hpo_id, _ in top_k_ids:
        if hpo_id in expected_ids:
            return 1.0
    return 0.0
```

**Observation:** HR@K is binary (0 or 1). It doesn't reward finding **more** relevant results or penalize finding them at lower ranks.

### 1.4 Current MaxOntSim Implementation

```python
# From metrics.py:496-553
def calculate_test_case_max_ont_sim(expected_ids, retrieved_ids, formula) -> float:
    """Find the maximum semantic similarity between any expected-retrieved pair"""
    # Returns 1.0 for exact match, else calculates ontology-based similarity
```

**Observation:** MaxOntSim is a unique metric that accounts for ontological relationships. This is valuable and should be retained.

### 1.5 Test Data Format

Current format (`tests/data/benchmarks/german/tiny_v1.json`):

```json
{
  "description": "Hypertrophic cardiomyopathy with septal hypertrophy",
  "text": "Hypertrophe Kardiomyopathie mit Septumhypertrophie",
  "expected_hpo_ids": ["HP:0001639", "HP:0001712"]
}
```

**Observations:**
- Simple, clean format
- Supports multiple expected IDs per test case
- No relevance grading (all expected IDs equally relevant)
- No difficulty classification
- Language implicit (folder-based: `german/`)

---

## 2. Industry Standards: MTEB and BEIR

### 2.1 MTEB (Massive Text Embedding Benchmark)

MTEB is the de facto standard for embedding model evaluation, covering 56+ tasks including:

- **Retrieval** (primary focus for Phentrieve)
- Classification
- Clustering
- Semantic Textual Similarity
- Re-ranking

**Key Retrieval Datasets in MTEB:**
- TREC-COVID (biomedical)
- BioASQ (biomedical Q&A)
- NFCorpus (nutrition/medical)
- SciFact (scientific claims)

**Primary Metric:** NDCG@10

### 2.2 BEIR (Benchmark for Zero-shot IR)

BEIR focuses specifically on retrieval with 18 diverse datasets. It includes biomedical datasets:

| Dataset | Domain | Queries | Documents |
|---------|--------|---------|-----------|
| TREC-COVID | Biomedical | 50 | 171K |
| BioASQ | Biomedical | 500 | 14.9M |
| NFCorpus | Medical/Nutrition | 323 | 3.6K |

**Primary Metric:** NDCG@10

### 2.3 Sentence Transformers Evaluation

The `InformationRetrievalEvaluator` from sentence-transformers measures:

1. **MRR@K** - Mean Reciprocal Rank
2. **Recall@K** - Proportion of relevant docs found
3. **NDCG@K** - Normalized Discounted Cumulative Gain
4. **Precision@K** - Proportion of retrieved docs that are relevant
5. **MAP@K** - Mean Average Precision

### 2.4 Gap Analysis

| Metric | MTEB/BEIR | Phentrieve | Priority |
|--------|-----------|------------|----------|
| NDCG@K | ✅ Primary | ❌ Missing | **High** |
| MRR | ✅ | ✅ | - |
| Recall@K | ✅ | ❌ Missing | **High** |
| Precision@K | ✅ | ❌ Missing | Medium |
| MAP@K | ✅ | ❌ Missing | Low |
| HR@K | Variant | ✅ | - |
| MaxOntSim | ❌ | ✅ Unique | - |

---

## 3. Recommended Metric Additions

### 3.1 NDCG@K (Normalized Discounted Cumulative Gain)

**Why it matters:** NDCG is the industry standard for retrieval evaluation. It:
- Rewards relevant results at higher ranks
- Handles graded relevance (not just binary)
- Normalizes for query difficulty (number of relevant docs)

**Implementation:**

```python
import math
from typing import Any

def ndcg_at_k(
    results: dict[str, Any],
    expected_ids: list[str],
    k: int = 10,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        results: Retrieval results from dense retriever
        expected_ids: List of relevant HPO IDs
        k: Number of results to consider
        relevance_scores: Optional dict mapping HPO ID to relevance score (default: binary)

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if not results or not results.get("metadatas") or not expected_ids:
        return 0.0

    # Extract retrieved IDs
    retrieved_ids = [
        meta.get("hpo_id", "")
        for meta in results["metadatas"][0][:k]
    ]

    # Calculate DCG
    dcg = 0.0
    for i, hpo_id in enumerate(retrieved_ids):
        if hpo_id in expected_ids:
            # Binary relevance (1 if relevant, 0 otherwise)
            # Or use graded relevance if provided
            rel = relevance_scores.get(hpo_id, 1.0) if relevance_scores else 1.0
            # Discount factor: 1 / log2(rank + 1)
            dcg += rel / math.log2(i + 2)  # +2 because rank is 1-indexed

    # Calculate ideal DCG (all relevant docs at top ranks)
    ideal_rels = sorted(
        [relevance_scores.get(hpo_id, 1.0) if relevance_scores else 1.0
         for hpo_id in expected_ids],
        reverse=True
    )[:k]

    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0

    return dcg / idcg
```

### 3.2 Recall@K

**Why it matters:** Unlike HR@K (binary), Recall@K measures **what proportion** of relevant items were found.

**Implementation:**

```python
def recall_at_k(
    results: dict[str, Any],
    expected_ids: list[str],
    k: int = 10,
) -> float:
    """
    Calculate Recall at K.

    Args:
        results: Retrieval results from dense retriever
        expected_ids: List of relevant HPO IDs
        k: Number of results to consider

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not results or not results.get("metadatas") or not expected_ids:
        return 0.0

    # Extract retrieved IDs
    retrieved_ids = set(
        meta.get("hpo_id", "")
        for meta in results["metadatas"][0][:k]
    )

    # Count relevant items found
    relevant_found = len(retrieved_ids.intersection(set(expected_ids)))

    return relevant_found / len(expected_ids)
```

**Difference from HR@K:**

| Scenario | HR@K | Recall@K |
|----------|------|----------|
| 0/3 relevant found in top K | 0.0 | 0.0 |
| 1/3 relevant found in top K | 1.0 | 0.33 |
| 2/3 relevant found in top K | 1.0 | 0.67 |
| 3/3 relevant found in top K | 1.0 | 1.0 |

### 3.3 Precision@K

**Why it matters:** Measures retrieval efficiency - how many of the top K results are actually relevant.

```python
def precision_at_k(
    results: dict[str, Any],
    expected_ids: list[str],
    k: int = 10,
) -> float:
    """
    Calculate Precision at K.

    Args:
        results: Retrieval results from dense retriever
        expected_ids: List of relevant HPO IDs
        k: Number of results to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if not results or not results.get("metadatas"):
        return 0.0

    # Extract retrieved IDs (up to k)
    retrieved_ids = [
        meta.get("hpo_id", "")
        for meta in results["metadatas"][0][:k]
    ]

    if not retrieved_ids:
        return 0.0

    # Count relevant items in top K
    relevant_in_top_k = sum(1 for hpo_id in retrieved_ids if hpo_id in expected_ids)

    return relevant_in_top_k / len(retrieved_ids)
```

### 3.4 MAP@K (Mean Average Precision)

**Why it matters:** Considers rank positions of **all** relevant results, not just the first.

```python
def average_precision_at_k(
    results: dict[str, Any],
    expected_ids: list[str],
    k: int = 10,
) -> float:
    """
    Calculate Average Precision at K for a single query.

    Args:
        results: Retrieval results from dense retriever
        expected_ids: List of relevant HPO IDs
        k: Number of results to consider

    Returns:
        AP@K score (0.0 to 1.0)
    """
    if not results or not results.get("metadatas") or not expected_ids:
        return 0.0

    # Extract retrieved IDs
    retrieved_ids = [
        meta.get("hpo_id", "")
        for meta in results["metadatas"][0][:k]
    ]

    expected_set = set(expected_ids)
    relevant_count = 0
    precision_sum = 0.0

    for i, hpo_id in enumerate(retrieved_ids):
        if hpo_id in expected_set:
            relevant_count += 1
            # Precision at this rank
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    if relevant_count == 0:
        return 0.0

    # Normalize by number of relevant docs (up to k)
    num_relevant = min(len(expected_ids), k)
    return precision_sum / num_relevant
```

### 3.5 Updated Metrics Summary

```python
@dataclass
class SingleTermRetrievalMetrics:
    """Complete metrics for single-term retrieval evaluation."""

    # Existing metrics (keep)
    mrr: float                    # Mean Reciprocal Rank
    hit_rate_at_k: dict[int, float]  # HR@1, HR@3, HR@5, HR@10
    max_ont_sim_at_k: dict[int, float]  # MaxOntSim@1, @3, @5, @10

    # New metrics (add)
    ndcg_at_k: dict[int, float]   # NDCG@1, @3, @5, @10
    recall_at_k: dict[int, float] # Recall@1, @3, @5, @10
    precision_at_k: dict[int, float]  # Precision@1, @3, @5, @10
    map_at_k: dict[int, float]    # MAP@1, @3, @5, @10 (optional)
```

---

## 4. Statistical Significance

### 4.1 The Problem

Current benchmark reports single averages without confidence intervals:
```
MRR (Dense): 0.5361
Hit@1 (Dense): 0.3333
```

This doesn't tell us:
- Is Model A significantly better than Model B?
- What's the variance in performance?
- How reliable are these numbers?

### 4.2 Bootstrap Confidence Intervals

From IR evaluation literature (Cormack & Lynam, SIGIR 2006):
> "Bootstrap Hypothesis Tests deserve more attention from the IR community, as they are based on fewer assumptions than traditional statistical significance tests."

**Implementation:**

```python
import numpy as np
from typing import Callable

def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    metric_fn: Callable[[list[float]], float] = np.mean,
) -> tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        values: Per-query metric values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        metric_fn: Aggregation function (default: mean)

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    values_array = np.array(values)
    n = len(values_array)

    # Bootstrap resampling
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample_indices = np.random.choice(n, size=n, replace=True)
        sample = values_array[sample_indices]
        bootstrap_estimates.append(metric_fn(sample))

    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    point_estimate = metric_fn(values_array)

    return point_estimate, ci_lower, ci_upper


def paired_bootstrap_test(
    values_a: list[float],
    values_b: list[float],
    n_bootstrap: int = 10000,
) -> tuple[float, bool]:
    """
    Paired bootstrap significance test for model comparison.

    Args:
        values_a: Per-query metrics for model A
        values_b: Per-query metrics for model B
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (p_value, is_significant_at_0.05)
    """
    assert len(values_a) == len(values_b), "Must have same number of queries"

    differences = np.array(values_a) - np.array(values_b)
    observed_diff = np.mean(differences)

    # Null hypothesis: no difference (center at 0)
    centered_diffs = differences - observed_diff

    # Bootstrap under null
    count_extreme = 0
    for _ in range(n_bootstrap):
        sample = np.random.choice(centered_diffs, size=len(centered_diffs), replace=True)
        if abs(np.mean(sample)) >= abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_bootstrap
    is_significant = p_value < 0.05

    return p_value, is_significant
```

### 4.3 Reporting Format

```markdown
## Model Comparison Results

### BioLORD-2023-M (German)

| Metric | Value | 95% CI |
|--------|-------|--------|
| MRR | 0.536 | [0.482, 0.591] |
| NDCG@10 | 0.612 | [0.558, 0.667] |
| Recall@10 | 0.778 | [0.722, 0.833] |
| HR@1 | 0.333 | [0.278, 0.389] |

### Statistical Significance (vs. Baseline)

| Comparison | Metric | Diff | p-value | Significant? |
|------------|--------|------|---------|--------------|
| BioLORD vs multilingual-e5 | MRR | +0.08 | 0.012 | ✅ |
| BioLORD vs all-MiniLM | MRR | +0.15 | 0.001 | ✅ |
```

---

## 5. Language-Specific Considerations

### 5.1 Language Isolation Principle

**Critical:** Single-term benchmarks must be **strictly language-specific**.

**Why:**
1. Embedding models have different multilingual capabilities
2. HPO terminology varies by language
3. Clinical phrasing differs across languages
4. Fair comparison requires language control

### 5.2 Recommended Test Data Structure

```
tests/data/benchmarks/
├── german/
│   ├── tiny_v1.json          # 9 cases (quick testing)
│   ├── standard_v1.json      # 70 cases (standard eval)
│   └── comprehensive_v1.json # 200 cases (full eval)
├── english/
│   ├── tiny_v1.json
│   ├── standard_v1.json
│   └── comprehensive_v1.json
├── french/
│   └── ...
├── spanish/
│   └── ...
└── dutch/
    └── ...
```

### 5.3 German-Specific Resources

**Key Model:** medBERT.de
- Trained on 4.7M German medical documents
- State-of-the-art for German clinical NLP
- Available: `GerMedBERT/medbert-512`

**Benchmark Consideration:** German clinical terminology may not align 1:1 with HPO's English-centric design. Consider:
- Testing with German HPO translations
- Including layperson terminology variants
- Testing abbreviation handling

### 5.4 Multilingual Model Evaluation

For multilingual embedding models (e.g., `multilingual-e5-large`):

```python
def evaluate_multilingual_model(
    model_name: str,
    languages: list[str] = ["de", "en", "fr", "es", "nl"],
) -> dict[str, dict]:
    """Evaluate a multilingual model across all supported languages."""

    results_by_language = {}

    for lang in languages:
        test_file = f"tests/data/benchmarks/{lang}/standard_v1.json"
        if not os.path.exists(test_file):
            continue

        results = run_evaluation(
            model_name=model_name,
            test_file=test_file,
            save_results=True,
        )

        results_by_language[lang] = results

    # Calculate cross-lingual consistency
    mrr_values = [r['avg_mrr_dense'] for r in results_by_language.values()]
    cross_lingual_variance = np.var(mrr_values)

    return {
        'per_language': results_by_language,
        'cross_lingual_variance': cross_lingual_variance,
        'average_mrr': np.mean(mrr_values),
    }
```

---

## 6. Test Data Improvements

### 6.1 Current Limitations

| Issue | Current State | Impact |
|-------|---------------|--------|
| Small size | 9 test cases (tiny) | Unreliable statistics |
| No difficulty levels | All cases equal | Can't analyze failure modes |
| No relevance grading | Binary (relevant/not) | NDCG less informative |
| Limited coverage | Random selection | May miss edge cases |

### 6.2 Proposed Test Case Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "metadata": {
      "type": "object",
      "properties": {
        "dataset_id": { "type": "string" },
        "version": { "type": "string" },
        "language": { "type": "string", "enum": ["de", "en", "fr", "es", "nl"] },
        "description": { "type": "string" }
      }
    },
    "test_cases": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["text", "expected_hpo_ids"],
        "properties": {
          "case_id": { "type": "string" },
          "text": { "type": "string" },
          "description": { "type": "string" },
          "expected_hpo_ids": {
            "type": "array",
            "items": { "type": "string", "pattern": "^HP:\\d{7}$" }
          },
          "difficulty": {
            "type": "string",
            "enum": ["easy", "medium", "hard"],
            "description": "Subjective difficulty rating"
          },
          "category": {
            "type": "string",
            "enum": ["exact_match", "synonym", "abbreviation", "layperson", "complex"],
            "description": "Type of matching required"
          },
          "hpo_domain": {
            "type": "string",
            "description": "HPO branch (e.g., 'Abnormality of the nervous system')"
          }
        }
      }
    }
  }
}
```

### 6.3 Difficulty Categories

| Category | Description | Example |
|----------|-------------|---------|
| **exact_match** | Direct HPO label | "Seizure" → HP:0001250 |
| **synonym** | HPO synonym | "Fits" → HP:0001250 |
| **abbreviation** | Medical abbreviation | "VSD" → HP:0001629 |
| **layperson** | Non-technical language | "bad hearing" → HP:0000365 |
| **complex** | Requires inference | "fails to thrive" → HP:0001508 |

### 6.4 Example Enhanced Test Case

```json
{
  "case_id": "de_neuro_001",
  "text": "Krampfanfälle",
  "description": "German word for seizures",
  "expected_hpo_ids": ["HP:0001250"],
  "difficulty": "easy",
  "category": "exact_match",
  "hpo_domain": "Abnormality of the nervous system"
}
```

### 6.5 Recommended Dataset Sizes

| Dataset | Cases | Purpose | Statistical Power |
|---------|-------|---------|-------------------|
| **tiny** | 9 | Quick sanity check | Very low |
| **small** | 50 | Development testing | Moderate |
| **standard** | 200 | Standard evaluation | Good |
| **comprehensive** | 500+ | Full benchmark | High |

**Minimum for publication:** 200 test cases with bootstrap CI

---

## 7. Implementation Roadmap

### Phase 1: Core Metric Additions (1 week)

**Tasks:**
1. [ ] Implement `ndcg_at_k()` function
2. [ ] Implement `recall_at_k()` function
3. [ ] Implement `precision_at_k()` function
4. [ ] Update `run_evaluation()` to calculate new metrics
5. [ ] Update `compare_models()` to display new metrics

**Files to modify:**
- `phentrieve/evaluation/metrics.py` (add functions)
- `phentrieve/evaluation/runner.py` (integrate metrics)

### Phase 2: Statistical Significance (1 week)

**Tasks:**
1. [ ] Implement `bootstrap_confidence_interval()`
2. [ ] Implement `paired_bootstrap_test()`
3. [ ] Add CI reporting to summary output
4. [ ] Create model comparison utility with significance

**Files to create/modify:**
- New: `phentrieve/evaluation/statistics.py`
- `phentrieve/evaluation/runner.py` (add CI calculation)
- `phentrieve/cli/benchmark_commands.py` (update output)

### Phase 3: Test Data Enhancement (1-2 weeks)

**Tasks:**
1. [ ] Define enhanced test case JSON schema
2. [ ] Add difficulty and category fields to existing data
3. [ ] Expand German dataset to 200 cases
4. [ ] Create English dataset (200 cases)
5. [ ] Add stratified reporting by difficulty/category

**Files to create/modify:**
- `tests/data/benchmarks/schema.json`
- `tests/data/benchmarks/german/*.json`
- New: `tests/data/benchmarks/english/*.json`

### Phase 4: Documentation & Reporting (0.5 weeks)

**Tasks:**
1. [ ] Update benchmark documentation
2. [ ] Create comparison report template
3. [ ] Add visualization for metric comparisons

**Files to modify:**
- `docs/advanced-topics/benchmarking-framework.md`

---

## 8. References

### Benchmarks

1. **MTEB (Massive Text Embedding Benchmark)**
   - https://github.com/embeddings-benchmark/mteb
   - Primary metric: NDCG@10

2. **BEIR (Benchmark for Zero-shot IR)**
   - https://github.com/beir-cellar/beir
   - 18 datasets including biomedical

3. **Sentence Transformers Evaluation**
   - https://sbert.net/docs/package_reference/evaluation.html

### Statistical Methods

4. **Bootstrap Methods for IR Evaluation**
   - Cormack & Lynam, SIGIR 2006: "Statistical Precision of Information Retrieval Evaluation"

5. **Comparing Statistical Tests**
   - Sakai, SIGIR 2006: "Evaluating evaluation metrics based on the bootstrap"

### German Medical NLP

6. **medBERT.de**
   - https://huggingface.co/GerMedBERT/medbert-512
   - State-of-the-art German medical BERT

7. **German Medical NLP Study (2024)**
   - "Comprehensive Study on German Language Models for Clinical and Biomedical Text Understanding"

### Retrieval Metrics

8. **NDCG Explained**
   - https://en.wikipedia.org/wiki/Discounted_cumulative_gain

9. **Retrieval Evaluation Metrics**
   - https://weaviate.io/blog/retrieval-evaluation-metrics

---

## Appendix A: Metric Formulas

### MRR (Mean Reciprocal Rank)
$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

### NDCG@K (Normalized Discounted Cumulative Gain)
$$DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$
$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

### Recall@K
$$Recall@K = \frac{|Retrieved@K \cap Relevant|}{|Relevant|}$$

### Precision@K
$$Precision@K = \frac{|Retrieved@K \cap Relevant|}{K}$$

### MAP@K (Mean Average Precision)
$$AP@K = \frac{1}{\min(K, |Relevant|)} \sum_{k=1}^{K} Precision@k \times rel_k$$

---

## Appendix B: K Values Recommendation

| K | Use Case | Rationale |
|---|----------|-----------|
| 1 | Top result quality | User often looks at first result |
| 3 | Quick scan | Typical user attention span |
| 5 | Standard | Balance of depth and efficiency |
| 10 | Comprehensive | Industry standard (MTEB) |
| 20 | Deep dive | For thorough evaluation |

**Default K values:** `[1, 3, 5, 10]`

---

**Document Status:** Complete
**Last Updated:** 2025-01-21
**Next Steps:** Implementation Phase 1 upon approval
