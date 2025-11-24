# Full-Text HPO Extraction Benchmarking Framework

**Status:** 📋 Research Complete - Design Phase
**Created:** 2025-01-21
**Updated:** 2025-01-21
**Author:** Claude Code (AI-assisted analysis)
**Priority:** High
**Estimated Effort:** 3-4 weeks implementation
**Scope:** Document-level HPO extraction from clinical text (distinct from single-term retrieval)

---

## Scope Clarification

This document covers **full-text HPO extraction** evaluation - the end-to-end process of extracting phenotype terms from clinical documents.

**This framework is INDEPENDENT from the Single-Term Retrieval Benchmark** (see `SINGLE-TERM-RETRIEVAL-BENCHMARK.md`).

| Aspect | Full-Text Extraction (THIS DOC) | Single-Term Retrieval (SEPARATE) |
|--------|--------------------------------|----------------------------------|
| **Input** | Clinical document (paragraphs) | Single clinical phrase |
| **Process** | Chunking → Retrieval → Aggregation → Assertion | Direct embedding retrieval |
| **Output** | Set of HPO terms with assertions | Ranked list of HPO terms |
| **Metrics** | Precision, Recall, F1, Assertion Accuracy | MRR, NDCG, HR@K |
| **Challenges** | Negation, chunking errors, aggregation | Embedding quality |
| **Gold Standard** | Annotated clinical documents | Term-to-term mappings |

---

## Executive Summary

This document presents a comprehensive analysis of **document-level HPO extraction** benchmarking challenges and proposes an improved evaluation framework.

### Core Challenges Addressed

1. **Document-level aggregation** - How to aggregate metrics across documents (micro/macro/weighted)
2. **Negation/assertion detection** - Distinguishing affirmed vs. negated phenotypes
3. **Partial annotations** - Handling incomplete gold standards
4. **Ontology-aware matching** - Credit for semantically similar extractions
5. **Span detection** - Strict vs. relaxed boundary matching
6. **Pipeline error propagation** - Errors in chunking affecting downstream extraction

### Benchmarks from Literature

| System | Dataset | Precision | Recall | F1 | Notes |
|--------|---------|-----------|--------|-----|-------|
| RAG-HPO (Llama-3 70B) | 120 case reports | 0.84 | 0.78 | **0.80** | State-of-the-art 2024 |
| PhenoBCBERT | Clinical notes | 0.75 | 0.71 | 0.73 | BERT-based |
| GPT-4 | OMIM summaries | 0.70 | 0.68 | 0.69 | Zero-shot LLM |
| Doc2Hpo | Benchmark | 0.62 | 0.58 | 0.60 | Traditional NLP |
| ClinPhen | Benchmark | 0.55 | 0.52 | 0.54 | Rule-based |
| BioCreative VIII Track 3 | Dysmorphology notes | Varies | Varies | 0.65-0.85 | Competition results |

### Key Recommendations Summary

| Challenge | Current State | Recommendation | Priority |
|-----------|---------------|----------------|----------|
| Aggregation | Macro only | Micro + Macro + Weighted | High |
| Negation | Post-hoc accuracy | Joint F1 (term + assertion) | High |
| Partial Gold | Not handled | Relaxed matching + CI | Medium |
| Ontology | Fixed 0.7 threshold | Data-driven threshold | Medium |
| Span | Not tracked | SemEval-style 4-tier | Low |
| Pipeline | Not analyzed | Component-wise metrics | Low |

---

## Table of Contents

1. [Current Implementation Analysis](#1-current-implementation-analysis)
2. [Challenge 1: Document-Level Aggregation](#2-challenge-1-document-level-aggregation)
3. [Challenge 2: Negation and Assertion Detection](#3-challenge-2-negation-and-assertion-detection)
4. [Challenge 3: Partial/Incomplete Annotations](#4-challenge-3-partialincomplete-annotations)
5. [Challenge 4: Ontology-Aware Matching](#5-challenge-4-ontology-aware-matching)
6. [Challenge 5: Span Detection](#6-challenge-5-span-detection)
7. [Challenge 6: Pipeline Error Analysis](#7-challenge-6-pipeline-error-analysis)
8. [Proposed Evaluation Framework](#8-proposed-evaluation-framework)
9. [Test Data Specification](#9-test-data-specification)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [References](#11-references)

---

## 1. Current Implementation Analysis

### 1.1 Existing Full-Text Evaluation Code

The current full-text extraction evaluation is in `phentrieve/evaluation/full_text_runner.py`:

```python
def evaluate_single_document_extraction(
    ground_truth_doc: dict,
    pipeline: TextProcessingPipeline,
    retriever: DenseRetriever,
    ...
) -> dict[str, Any]:
    """
    Pipeline:
    1. Chunk document via TextProcessingPipeline
    2. Extract HPO terms via orchestrate_hpo_extraction()
    3. Compare aggregated results to ground truth
    4. Calculate PRF1 + assertion accuracy
    """
```

### 1.2 Current Metrics in semantic_metrics.py

```python
def calculate_semantically_aware_set_based_prf1(
    extracted_annotations: list[dict],
    ground_truth_annotations: list[dict],
    semantic_similarity_threshold: float = 0.7,
) -> dict:
    """
    Two-pass matching:
    1. Exact ID matches
    2. Semantic matches (similarity >= threshold)

    Returns: precision, recall, f1, exact_count, semantic_count
    """
```

### 1.3 Current Metrics Summary

| Metric | Implemented | Notes |
|--------|-------------|-------|
| Exact Precision/Recall/F1 | ✅ | Exact HPO ID match |
| Semantic Precision/Recall/F1 | ✅ | With ontology similarity |
| Combined (Exact + Semantic) F1 | ✅ | Default reported metric |
| Assertion Accuracy | ✅ | For matched pairs only |
| Micro-averaging | ❌ | Missing |
| Weighted-averaging | ❌ | Missing |
| Confidence Intervals | ❌ | Missing |
| Span-based metrics | ❌ | Not applicable (no spans) |

### 1.4 Gap Analysis

| Feature | Current | Needed | Priority |
|---------|---------|--------|----------|
| Micro-averaged corpus metrics | ❌ | ✅ | High |
| Weighted-averaged corpus metrics | ❌ | ✅ | High |
| Joint F1 (term + assertion) | ❌ | ✅ | High |
| Assertion-stratified metrics | Partial | Full | High |
| Relaxed matching for partial gold | ❌ | ✅ | Medium |
| Bootstrap confidence intervals | ❌ | ✅ | Medium |
| Per-HPO-branch analysis | ❌ | ✅ | Low |

---

## 2. Challenge 1: Document-Level Aggregation

### 2.1 The Problem

When evaluating across multiple clinical documents, how do we aggregate metrics?

**Example Scenario:**

| Document | Extracted | Ground Truth | TP | FP | FN | P | R | F1 |
|----------|-----------|--------------|----|----|-----|-----|-----|-----|
| Doc A | 3 | 3 | 2 | 1 | 1 | 0.67 | 0.67 | 0.67 |
| Doc B | 10 | 5 | 4 | 6 | 1 | 0.40 | 0.80 | 0.53 |
| Doc C | 2 | 8 | 2 | 0 | 6 | 1.00 | 0.25 | 0.40 |

**Different aggregations yield different conclusions:**

| Method | Precision | Recall | F1 | Interpretation |
|--------|-----------|--------|-----|----------------|
| **Macro** (avg per doc) | 0.69 | 0.57 | 0.53 | Equal weight per document |
| **Micro** (pooled TP/FP/FN) | 0.53 | 0.50 | 0.52 | Equal weight per term |
| **Weighted** (by GT count) | 0.59 | 0.42 | 0.48 | Proportional to phenotype richness |

### 2.2 Literature Guidance

**From MedTric (PMC 2023):**
> "The micro average favors classifiers with stronger performance on predominant classes whereas the macro average favors classifiers suited to detecting rarely occurring classes. In clinical settings where rare diseases are of most concern, micro average measures are less meaningful."

**From ICD Multi-Label Classification Literature:**
> "Macro-averaged values are calculated by averaging metrics computed per-label. Micro-averaged values treat each (document, code) pair as a separate prediction. Macro metrics place much more emphasis on rare code prediction."

### 2.3 Recommendation

**Report all three aggregation strategies:**

```python
@dataclass
class CorpusExtractionMetrics:
    """Aggregated metrics across document corpus."""

    # Per-document metrics (for analysis)
    per_document: list[DocumentMetrics]

    # Micro-averaged (pooled across all documents)
    micro_precision: float  # sum(TP) / sum(TP + FP)
    micro_recall: float     # sum(TP) / sum(TP + FN)
    micro_f1: float

    # Macro-averaged (average of per-document metrics)
    macro_precision: float  # mean(doc_precision)
    macro_recall: float     # mean(doc_recall)
    macro_f1: float

    # Standard deviations
    std_precision: float
    std_recall: float
    std_f1: float

    # Weighted by ground truth count
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float

    # Sample size
    n_documents: int
    total_ground_truth: int
    total_extracted: int
```

**Primary metric recommendation:**
- **Macro-F1** for clinical applications (rare phenotypes equally important)
- Always report standard deviation and sample size

---

## 3. Challenge 2: Negation and Assertion Detection

### 3.1 The Problem

HPO extraction must distinguish:
- **Affirmed**: "Patient has seizures" → HP:0001250 (PRESENT)
- **Negated**: "No history of seizures" → HP:0001250 (ABSENT)
- **Uncertain**: "Possible seizures" → HP:0001250 (UNCERTAIN)

**BioCreative VIII Track 3 Definition:**
> "Normal findings (NORMFs) typically denote the absence of phenotypic abnormalities. Key findings (KEYFs) indicate the presence of phenotypic abnormalities. Tasks involve extracting KEYFs while excluding NORMFs."

### 3.2 Current Implementation Gap

Current approach calculates assertion accuracy **only for matched pairs**:

```python
# Current: Only evaluates assertion for TRUE POSITIVES
assertion_accuracy = correct_assertions / matched_pairs
```

**Problem:** This misses:
- Extracting affirmed when should be negated (assertion-based FP)
- Missing negated terms entirely

### 3.3 Recommended Metrics

#### 3.3.1 Joint F1 (Term + Assertion)

A true positive requires BOTH correct HPO ID AND correct assertion status:

```python
def calculate_joint_f1(
    extracted: list[dict],
    ground_truth: list[dict],
) -> dict:
    """
    Joint evaluation: term must match AND assertion must match.
    """
    joint_tp = 0

    for gt in ground_truth:
        gt_id = gt.get("hpo_id")
        gt_assertion = gt.get("assertion_status")

        # Find matching extraction with same ID AND assertion
        for ext in extracted:
            if (ext.get("id") == gt_id and
                ext.get("assertion_status") == gt_assertion):
                joint_tp += 1
                break

    joint_precision = joint_tp / len(extracted) if extracted else 0
    joint_recall = joint_tp / len(ground_truth) if ground_truth else 0
    joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall) if (joint_precision + joint_recall) > 0 else 0

    return {
        "joint_precision": joint_precision,
        "joint_recall": joint_recall,
        "joint_f1": joint_f1,
    }
```

#### 3.3.2 Assertion-Stratified Metrics

Evaluate performance separately for each assertion class:

```python
def calculate_assertion_stratified_metrics(
    extracted: list[dict],
    ground_truth: list[dict],
) -> dict:
    """Evaluate F1 for each assertion status separately."""

    results = {}
    for status in ["affirmed", "negated", "uncertain"]:
        # Filter by assertion status
        ext_filtered = [e for e in extracted if e.get("assertion_status") == status]
        gt_filtered = [g for g in ground_truth if g.get("assertion_status") == status]

        # Calculate PRF1 for this status
        metrics = calculate_set_prf1(ext_filtered, gt_filtered)
        results[status] = metrics

    return results
```

#### 3.3.3 Confusion Matrix for Assertions

For matched term pairs, show assertion confusion:

```
                    Predicted
                 Affirmed | Negated | Uncertain
Actual Affirmed    85     |   10    |    5
       Negated      8     |   42    |    0
       Uncertain    3     |    2    |   15
```

### 3.4 Benchmark Targets

From literature on assertion detection:

| System | Affirmed | Negated | Overall | Source |
|--------|----------|---------|---------|--------|
| Fine-tuned LLM | 0.96 | 0.92 | 0.962 | 2024 study |
| GPT-4o | 0.92 | 0.84 | 0.901 | Zero-shot |
| NegEx | 0.88 | 0.84 | 0.84 | Rule-based |
| cTAKES | 0.91 | 0.87 | 0.89 | Hybrid |

---

## 4. Challenge 3: Partial/Incomplete Annotations

### 4.1 The Problem

Gold standard annotations are often incomplete:
- Annotator fatigue
- Ambiguous phenotypes
- Granularity disagreement (HP:0001166 vs HP:0001238)
- Domain expertise variation

**From Alzheimer's NLP study:**
> "Inter-annotator agreement (Cohen's kappa) ranged from 0.72 to 1.0, positively correlated with NLP pipeline performance (F1 = 0.65-0.99)."

### 4.2 Annotation Quality Metrics

Before evaluation, assess gold standard quality:

```python
@dataclass
class AnnotationQualityMetrics:
    """Metrics for gold standard quality assessment."""

    # Inter-annotator agreement (if multiple annotators)
    cohen_kappa: float | None
    fleiss_kappa: float | None  # For >2 annotators

    # Coverage metrics
    is_exhaustively_annotated: bool
    annotation_density: float  # annotations per 100 tokens

    # Confidence
    annotator_confidence: float | None  # Self-reported
```

### 4.3 Relaxed Matching Strategies

Based on SemEval and i2b2 evaluation approaches:

```python
class MatchType(Enum):
    EXACT = "exact"           # Same HPO ID
    SEMANTIC = "semantic"     # Similarity >= threshold
    HIERARCHICAL = "hierarchical"  # Parent/child relationship
    NO_MATCH = "no_match"

def calculate_relaxed_prf1(
    extracted: list[dict],
    ground_truth: list[dict],
    semantic_threshold: float = 0.7,
    hierarchical_credit: float = 0.5,
) -> dict:
    """
    Calculate PRF1 with partial credit for near-matches.

    Credit scheme:
    - Exact match: 1.0
    - Semantic match: similarity_score
    - Hierarchical (parent/child): 0.5
    - No match: 0.0
    """
    total_credit = 0.0
    match_details = []

    for ext in extracted:
        best_match = find_best_match(ext, ground_truth, semantic_threshold)

        if best_match.type == MatchType.EXACT:
            total_credit += 1.0
        elif best_match.type == MatchType.SEMANTIC:
            total_credit += best_match.similarity
        elif best_match.type == MatchType.HIERARCHICAL:
            total_credit += hierarchical_credit

        match_details.append(best_match)

    relaxed_precision = total_credit / len(extracted) if extracted else 0
    # Similar for recall...

    return {
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_f1": ...,
        "match_breakdown": {
            "exact": count_exact,
            "semantic": count_semantic,
            "hierarchical": count_hierarchical,
            "no_match": count_no_match,
        }
    }
```

### 4.4 Handling Unknown Completeness

When gold standard completeness is uncertain:

```python
def calculate_metrics_with_uncertainty(
    extracted: list[dict],
    ground_truth: list[dict],
    confidence_scores: list[float] | None = None,
) -> dict:
    """
    Report conservative and optimistic bounds.
    """
    # Conservative: All unmatched extractions are FP
    conservative_metrics = calculate_standard_prf1(extracted, ground_truth)

    # Optimistic: Only high-confidence unmatched are FP
    if confidence_scores:
        high_conf_fp = [e for e, c in zip(extracted, confidence_scores)
                       if c >= 0.8 and not matches_ground_truth(e, ground_truth)]
        optimistic_fp = len(high_conf_fp)
    else:
        optimistic_fp = 0  # Assume no FP if uncertain

    optimistic_precision = tp / (tp + optimistic_fp)

    return {
        "conservative": conservative_metrics,
        "optimistic": {"precision": optimistic_precision, ...},
        "expected": average_of_bounds,
    }
```

---

## 5. Challenge 4: Ontology-Aware Matching

### 5.1 The Problem

Standard F1 treats all mismatches equally:
- **Near miss**: HP:0001250 (Seizure) vs HP:0020219 (Focal seizure) → Related!
- **Far miss**: HP:0001250 (Seizure) vs HP:0001626 (Cardiac) → Unrelated

### 5.2 Current Implementation

Phentrieve uses a HYBRID formula with fixed 0.7 threshold:

```python
# From metrics.py
def calculate_semantic_similarity(expected, retrieved, formula=HYBRID):
    """Combines LCA depth and path length factors."""
```

### 5.3 Optimal Threshold Selection

From PLOS ONE (2015) - "Optimal Threshold Determination":
> "Interpretation is frequently based on an arbitrary threshold (typically 0.5) even though no mathematical property supports this choice."

**Recommendation: Data-driven threshold**

```python
def find_optimal_similarity_threshold(
    validation_pairs: list[tuple[str, str, bool]],  # (term1, term2, are_semantically_equivalent)
) -> float:
    """
    Find threshold that minimizes false matches + missed matches.
    """
    from sklearn.metrics import roc_curve

    similarities = [calculate_semantic_similarity(t1, t2) for t1, t2, _ in validation_pairs]
    labels = [int(equiv) for _, _, equiv in validation_pairs]

    fpr, tpr, thresholds = roc_curve(labels, similarities)

    # Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)

    return thresholds[optimal_idx]
```

### 5.4 Multi-Threshold Reporting

Report performance across multiple thresholds:

```python
def analyze_threshold_sensitivity(
    extracted: list[dict],
    ground_truth: list[dict],
    thresholds: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
) -> dict:
    """Evaluate at multiple similarity thresholds."""
    results = {}

    for thresh in thresholds:
        metrics = calculate_prf1_at_threshold(extracted, ground_truth, thresh)
        results[f"threshold_{thresh}"] = metrics

    # Calculate AUPRC (Area Under Precision-Recall Curve)
    precisions = [results[f"threshold_{t}"]["precision"] for t in thresholds]
    recalls = [results[f"threshold_{t}"]["recall"] for t in thresholds]
    auprc = calculate_auc(recalls, precisions)

    results["auprc"] = auprc

    return results
```

---

## 6. Challenge 5: Span Detection

### 6.1 The Problem

Phentrieve currently extracts HPO terms without tracking source text spans:
- Cannot evaluate boundary detection accuracy
- Cannot trace errors to specific text regions

### 6.2 SemEval Evaluation Tiers (if spans are added)

Based on SemEval 2013 NER evaluation:

| Tier | Requirement | Use Case |
|------|-------------|----------|
| **Strict** | Exact span + Exact HPO ID | Gold standard alignment |
| **Exact** | Exact span, any HPO | Span detection quality |
| **Partial** | Overlapping span, any HPO | Tolerant boundary |
| **Type** | Any overlap, Exact HPO ID | Concept detection |

### 6.3 Recommendation

**For current system (no spans):** Not applicable - skip span-based metrics

**If spans are added in future:**

```python
def calculate_span_based_metrics(
    extracted_spans: list[dict],  # {start, end, hpo_id}
    ground_truth_spans: list[dict],
) -> dict:
    """
    SemEval-style span evaluation.
    """
    strict_tp, exact_tp, partial_tp, type_tp = 0, 0, 0, 0

    for ext in extracted_spans:
        for gt in ground_truth_spans:
            span_match = classify_span_match(ext, gt)

            if span_match == "exact" and ext["hpo_id"] == gt["hpo_id"]:
                strict_tp += 1
            if span_match == "exact":
                exact_tp += 1
            if span_match in ["exact", "partial"]:
                partial_tp += 1
            if span_match != "none" and ext["hpo_id"] == gt["hpo_id"]:
                type_tp += 1

    return {
        "strict_f1": ...,
        "exact_f1": ...,
        "partial_f1": ...,
        "type_f1": ...,
    }
```

---

## 7. Challenge 6: Pipeline Error Analysis

### 7.1 The Problem

Phentrieve has a multi-stage pipeline:
1. **Chunking** - Split document into semantic chunks
2. **Retrieval** - Find HPO terms per chunk
3. **Re-ranking** (optional) - Cross-encoder refinement
4. **Aggregation** - Combine chunk results
5. **Assertion** - Detect negation/affirmation

Errors can propagate through stages.

### 7.2 Component-Wise Evaluation

```python
@dataclass
class PipelineAnalysis:
    """Analyze error contribution by pipeline stage."""

    # Chunking quality (if gold chunks available)
    chunking_precision: float | None
    chunking_recall: float | None

    # Per-chunk retrieval (before aggregation)
    per_chunk_recall: float  # What % of chunk phenotypes found?

    # Aggregation quality
    aggregation_precision: float  # False positives from merging?

    # Assertion detection
    assertion_accuracy: float

    # Error attribution
    errors_from_chunking: int
    errors_from_retrieval: int
    errors_from_aggregation: int
    errors_from_assertion: int
```

### 7.3 Ablation Study Design

To identify bottlenecks:

| Experiment | Setup | Measures |
|------------|-------|----------|
| Gold chunks | Use manually segmented text | Retrieval quality in isolation |
| No aggregation | Evaluate per-chunk | Aggregation impact |
| No reranker | Disable cross-encoder | Re-ranking value |
| Gold assertions | Use annotated statuses | Assertion system quality |

---

## 8. Proposed Evaluation Framework

### 8.1 Complete Metrics Suite

```python
@dataclass
class DocumentExtractionMetrics:
    """Complete metrics for single document evaluation."""

    doc_id: str

    # === Counts ===
    num_extracted: int
    num_ground_truth: int

    # === Standard Metrics ===
    # Exact match only
    exact_precision: float
    exact_recall: float
    exact_f1: float

    # Ontology-aware (exact + semantic)
    semantic_precision: float
    semantic_recall: float
    semantic_f1: float

    # === Relaxed Metrics ===
    relaxed_precision: float  # With partial credit
    relaxed_recall: float
    relaxed_f1: float

    # Match breakdown
    exact_match_count: int
    semantic_match_count: int
    hierarchical_match_count: int
    no_match_count: int

    # === Assertion Metrics ===
    # For matched pairs
    assertion_accuracy: float
    assertion_confusion: dict  # 3x3 matrix

    # Joint evaluation
    joint_precision: float  # Term AND assertion correct
    joint_recall: float
    joint_f1: float

    # Stratified by assertion class
    affirmed_f1: float
    negated_f1: float
    uncertain_f1: float


@dataclass
class CorpusExtractionMetrics:
    """Aggregated metrics across document corpus."""

    # === Aggregation Strategies ===
    micro: AggregatedMetrics  # Pooled TP/FP/FN
    macro: AggregatedMetrics  # Mean of per-document
    weighted: AggregatedMetrics  # Weighted by GT count

    # === Variability ===
    std_precision: float
    std_recall: float
    std_f1: float

    # === Confidence Intervals ===
    ci_precision: tuple[float, float]  # 95% CI
    ci_recall: tuple[float, float]
    ci_f1: tuple[float, float]

    # === Corpus Statistics ===
    n_documents: int
    total_ground_truth: int
    total_extracted: int

    # === Stratified Analysis ===
    by_assertion: dict[str, AggregatedMetrics]
    by_document_length: dict[str, AggregatedMetrics]
    by_hpo_branch: dict[str, AggregatedMetrics]

    # === Threshold Sensitivity ===
    threshold_analysis: dict[float, AggregatedMetrics]
    auprc: float
```

### 8.2 Evaluation Pipeline

```python
def evaluate_full_text_extraction_corpus(
    documents: list[dict],
    extraction_pipeline: Callable,
    config: EvaluationConfig,
) -> CorpusExtractionMetrics:
    """
    Complete evaluation pipeline for full-text HPO extraction.
    """
    doc_results = []

    for doc in documents:
        # Run extraction pipeline
        extracted = extraction_pipeline(doc["text"])
        ground_truth = doc["annotations"]

        # Calculate document-level metrics
        doc_metrics = calculate_document_metrics(
            extracted=extracted,
            ground_truth=ground_truth,
            semantic_threshold=config.semantic_threshold,
        )
        doc_results.append(doc_metrics)

    # Aggregate to corpus level
    corpus_metrics = aggregate_corpus_metrics(
        doc_results,
        bootstrap_samples=config.bootstrap_samples,
    )

    return corpus_metrics
```

### 8.3 Reporting Template

```markdown
# Full-Text HPO Extraction Evaluation Report

## Configuration
- Pipeline: Phentrieve v{version}
- Embedding Model: {model}
- Re-ranker: {enabled/disabled}
- Semantic Threshold: {threshold}

## Corpus Statistics
- Documents: {n_docs}
- Total Ground Truth Terms: {n_gt}
- Total Extracted Terms: {n_ext}

## Primary Metrics (Macro-averaged)

| Metric | Value | 95% CI | Std Dev |
|--------|-------|--------|---------|
| Precision | {p:.3f} | [{p_lo:.3f}, {p_hi:.3f}] | {p_std:.3f} |
| Recall | {r:.3f} | [{r_lo:.3f}, {r_hi:.3f}] | {r_std:.3f} |
| F1 | {f1:.3f} | [{f1_lo:.3f}, {f1_hi:.3f}] | {f1_std:.3f} |

## Aggregation Comparison

| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Micro | {micro_p:.3f} | {micro_r:.3f} | {micro_f1:.3f} |
| Macro | {macro_p:.3f} | {macro_r:.3f} | {macro_f1:.3f} |
| Weighted | {weighted_p:.3f} | {weighted_r:.3f} | {weighted_f1:.3f} |

## Match Type Breakdown

| Match Type | Count | % of Matches |
|------------|-------|--------------|
| Exact | {exact} | {exact_pct:.1f}% |
| Semantic | {semantic} | {semantic_pct:.1f}% |
| Hierarchical | {hier} | {hier_pct:.1f}% |
| No Match (FP) | {no_match} | {no_match_pct:.1f}% |

## Assertion Detection

### Joint F1 (Term + Assertion)
| Metric | Value |
|--------|-------|
| Joint Precision | {joint_p:.3f} |
| Joint Recall | {joint_r:.3f} |
| Joint F1 | {joint_f1:.3f} |

### Stratified by Assertion

| Assertion | Precision | Recall | F1 | Support |
|-----------|-----------|--------|-----|---------|
| Affirmed | {aff_p:.3f} | {aff_r:.3f} | {aff_f1:.3f} | {aff_n} |
| Negated | {neg_p:.3f} | {neg_r:.3f} | {neg_f1:.3f} | {neg_n} |
| Uncertain | {unc_p:.3f} | {unc_r:.3f} | {unc_f1:.3f} | {unc_n} |

### Assertion Confusion Matrix (for matched terms)

|  | Pred: Affirmed | Pred: Negated | Pred: Uncertain |
|--|----------------|---------------|-----------------|
| **GT: Affirmed** | {aa} | {an} | {au} |
| **GT: Negated** | {na} | {nn} | {nu} |
| **GT: Uncertain** | {ua} | {un} | {uu} |

## Threshold Sensitivity

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.5 | ... | ... | ... |
| 0.6 | ... | ... | ... |
| 0.7 | ... | ... | ... |
| 0.8 | ... | ... | ... |
| 0.9 | ... | ... | ... |
| 1.0 (exact) | ... | ... | ... |

**AUPRC**: {auprc:.3f}
```

---

## 9. Test Data Specification

### 9.1 Enhanced Schema for Full-Text Evaluation

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
        "language": { "type": "string" },
        "document_type": {
          "type": "string",
          "enum": ["case_report", "discharge_summary", "progress_note", "physical_exam"]
        },
        "annotation_guidelines": { "type": "string" },
        "inter_annotator_kappa": { "type": "number" },
        "is_exhaustively_annotated": { "type": "boolean" }
      }
    },
    "documents": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["doc_id", "text", "annotations"],
        "properties": {
          "doc_id": { "type": "string" },
          "text": { "type": "string" },
          "annotations": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["hpo_id", "assertion_status"],
              "properties": {
                "hpo_id": { "type": "string", "pattern": "^HP:\\d{7}$" },
                "label": { "type": "string" },
                "assertion_status": {
                  "type": "string",
                  "enum": ["affirmed", "negated", "uncertain"]
                },
                "text_span": { "type": "string" },
                "start_offset": { "type": "integer" },
                "end_offset": { "type": "integer" }
              }
            }
          }
        }
      }
    }
  }
}
```

### 9.2 Example Document

```json
{
  "doc_id": "case_001",
  "text": "Der 5-jährige Junge präsentiert sich mit rezidivierenden Krampfanfällen seit dem 3. Lebensjahr. Die körperliche Untersuchung zeigt keine Mikrozephalie. Verdacht auf leichte geistige Behinderung.",
  "annotations": [
    {
      "hpo_id": "HP:0001250",
      "label": "Seizure",
      "assertion_status": "affirmed",
      "text_span": "rezidivierenden Krampfanfällen"
    },
    {
      "hpo_id": "HP:0000252",
      "label": "Microcephaly",
      "assertion_status": "negated",
      "text_span": "keine Mikrozephalie"
    },
    {
      "hpo_id": "HP:0001249",
      "label": "Intellectual disability",
      "assertion_status": "uncertain",
      "text_span": "Verdacht auf leichte geistige Behinderung"
    }
  ]
}
```

### 9.3 Recommended Dataset Sizes

| Dataset | Documents | Annotations | Purpose | Statistical Power |
|---------|-----------|-------------|---------|-------------------|
| tiny | 5 | ~25 | Quick testing | Very low |
| small | 20 | ~100 | Development | Low |
| standard | 100 | ~500 | Evaluation | Good |
| comprehensive | 300+ | ~1500+ | Publication | High |

**Minimum for publication:** 100 documents with bootstrap CI

---

## 10. Implementation Roadmap

### Phase 1: Core Metrics (Week 1-2)

**Tasks:**
1. [ ] Implement `CorpusExtractionMetrics` dataclass
2. [ ] Add micro/macro/weighted aggregation functions
3. [ ] Implement bootstrap confidence intervals
4. [ ] Update `full_text_runner.py` to return complete metrics

**Files:**
- `phentrieve/evaluation/corpus_metrics.py` (new)
- `phentrieve/evaluation/full_text_runner.py` (update)

### Phase 2: Assertion Metrics (Week 2)

**Tasks:**
1. [ ] Implement `calculate_joint_f1()`
2. [ ] Implement `calculate_assertion_stratified_metrics()`
3. [ ] Add assertion confusion matrix
4. [ ] Integrate into evaluation pipeline

**Files:**
- `phentrieve/evaluation/semantic_metrics.py` (update)
- `phentrieve/evaluation/assertion_metrics.py` (new)

### Phase 3: Relaxed Matching (Week 3)

**Tasks:**
1. [ ] Implement `MatchType` classification
2. [ ] Implement hierarchical match detection
3. [ ] Add relaxed PRF1 calculation
4. [ ] Add match breakdown reporting

**Files:**
- `phentrieve/evaluation/semantic_metrics.py` (update)

### Phase 4: Test Data & Documentation (Week 4)

**Tasks:**
1. [ ] Define enhanced test data JSON schema
2. [ ] Create sample annotated documents (German)
3. [ ] Update benchmark CLI commands
4. [ ] Write comprehensive documentation

**Files:**
- `tests/data/benchmarks/schema.json` (new)
- `tests/data/benchmarks/german/fulltext_v1.json` (new)
- `docs/advanced-topics/benchmarking-framework.md` (update)

---

## 11. References

### Full-Text Extraction Evaluation

1. **BioCreative VIII Track 3**
   - Phenotype extraction from dysmorphology notes
   - HPO normalization task
   - https://biocreative.bioinformatics.udel.edu/tasks/biocreative-viii/track-3/

2. **RAG-HPO (2024)**
   - medRxiv preprint on LLM-based phenotyping
   - 120 case report benchmark, F1=0.80
   - https://www.medrxiv.org/content/10.1101/2024.12.01.24318253v1

3. **PhenoBCBERT and PhenoGPT (2024)**
   - PMC article on clinical phenotype recognition
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC10801236/

### Evaluation Methodology

4. **SemEval 2013 Task 9.1**
   - NER evaluation schemas (Strict, Exact, Partial, Type)
   - https://github.com/MantisAI/nervaluate

5. **Clinical NER Evaluation (2024)**
   - JMIR Medical Informatics
   - https://medinform.jmir.org/2024/1/e59782/

6. **i2b2 2010 Challenge**
   - Assertion detection benchmark
   - Relaxed F-score methodology

### Statistical Methods

7. **Bootstrap Methods for IR**
   - Cormack & Lynam, SIGIR 2006
   - Bootstrap confidence intervals for evaluation

8. **Inter-Annotator Agreement**
   - Cohen's Kappa for clinical annotations
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC10294690/

### Multi-Label Classification

9. **ICD Multi-Label Classification**
   - MIMIC-III benchmark
   - Micro/Macro/Weighted averaging

---

## Appendix A: Assertion Status Definitions

| Status | Definition | Example | HPO Interpretation |
|--------|------------|---------|-------------------|
| **affirmed** | Phenotype is present | "Patient has seizures" | Include in patient phenotype |
| **negated** | Phenotype is absent | "No seizures" | Exclude from differential |
| **uncertain** | Possibly present | "Possible seizures" | Flag for follow-up |

---

## Appendix B: Comparison with Related Tasks

| Task | Input | Output | Key Metrics |
|------|-------|--------|-------------|
| **Full-Text HPO Extraction** (this doc) | Clinical document | Set of HPO terms | Macro-F1, Joint-F1 |
| **Single-Term Retrieval** (separate) | Clinical phrase | Ranked HPO list | MRR, NDCG@K |
| **ICD Coding** | Discharge summary | ICD codes | Micro-F1, P@K |
| **Clinical NER** | Clinical text | Named entities + spans | Strict/Relaxed F1 |

---

**Document Status:** Complete
**Last Updated:** 2025-01-21
**Next Steps:** Implementation Phase 1 upon approval
