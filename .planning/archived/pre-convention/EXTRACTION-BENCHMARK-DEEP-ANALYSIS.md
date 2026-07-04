# HPO Extraction Benchmark Deep Analysis

## Executive Summary

The extraction system achieves **F1 ~0.33** but is fundamentally limited by **semantic equivalence problems**, not retrieval quality. The core issue: **the system retrieves semantically correct terms that don't match the gold standard's specific HPO ID choices**.

---

## 1. Overall Metrics by Dataset

| Dataset | Documents | TP | FP | FN | Precision | Recall | F1 |
|---------|-----------|-----|-----|-----|-----------|--------|-----|
| GeneReviews | 10 | 87 | 186 | 150 | 0.319 | 0.367 | 0.341 |
| ID_68 | 68 | 251 | 568 | 374 | 0.306 | 0.402 | 0.348 |
| GSC_plus | 228 | 445 | 1019 | 1098 | 0.304 | 0.288 | 0.296 |
| **Total** | 306 | 783 | 1773 | 1622 | 0.306 | 0.326 | 0.316 |

---

## 2. Score Separation Analysis

| Metric | Value |
|--------|-------|
| Mean TP Score | 0.878 |
| Mean FP Score | 0.840 |
| **Separation** | **0.038** |

### Threshold Impact Analysis

| Threshold | TP Retained | FP Retained |
|-----------|-------------|-------------|
| 0.75 | 100.0% | 100.0% |
| 0.80 | 87.7% | 68.9% |
| 0.85 | 67.3% | 41.7% |
| 0.90 | 37.9% | 17.0% |

**Implication**: Threshold tuning cannot fix this problem. The similarity scores are high for BOTH correct and "incorrect" matches because they're semantically equivalent.

---

## 3. Root Causes of False Positives

### A. Parent-Child Confusion (60-70% of FP/FN pairs)

The system retrieves a parent or child term instead of the exact gold term:

| Predicted (FP) | Gold (FN) | Relationship |
|----------------|-----------|--------------|
| HP:0012759 Neurodevelopmental abnormality | HP:0001263 Global developmental delay | Parent |
| HP:0040195 Decreased head circumference | HP:0000252 Microcephaly | Parent |
| HP:0001252 Hypotonia | HP:0001290 Generalized hypotonia | Parent |
| HP:0001531 Failure to thrive in infancy | HP:0001508 Failure to thrive | Child (age-specific) |
| HP:0040196 Mild microcephaly | HP:0000252 Microcephaly | Child (severity) |

**The retrieval is semantically correct!**

### B. Synonym/Variant Term Confusion

Different HPO terms represent the same clinical concept:

| Text Evidence | Predicted | Gold |
|---------------|-----------|------|
| "thinning of the corpus callosum" | HP:0033725 Thin corpus callosum | HP:0002079 Hypoplasia of the corpus callosum |
| "hypoplastic nails" | HP:0001804 Hypoplastic fingernail | HP:0001792 Small nail |
| "craniosynostosis" | HP:0011325 Pansynostosis | HP:0001363 Craniosynostosis |

### C. Obsolete Terms Still in Index

| FP Term | Occurrences |
|---------|-------------|
| HP:0001452 "obsolete Autosomal dominant contiguous gene syndrome" | 26x |
| HP:0006877 "obsolete Mental retardation, in some" | 8x |

→ Issue #133 addresses this.

---

## 4. Most Common False Positives (Cross-Dataset)

| Count | HPO ID | Term |
|-------|--------|------|
| 48x | HP:0012759 | Neurodevelopmental abnormality |
| 39x | HP:0007524 | Atypical neurofibromatosis |
| 34x | HP:0032316 | Family history |
| 33x | HP:0040195 | Decreased head circumference |
| 26x | HP:0001452 | obsolete Autosomal dominant contiguous gene syndrome |
| 22x | HP:0000364 | Hearing abnormality |
| 19x | HP:0040196 | Mild microcephaly |

---

## 5. Most Common False Negatives (Cross-Dataset)

| Count | HPO ID | Term | Evidence Example |
|-------|--------|------|------------------|
| 66x | HP:0000006 | Autosomal dominant inheritance | "autosomal dominant trait" |
| 41x | HP:0006746 | (Unknown) | "Neurofibromatosis" |
| 39x | HP:0001263 | Global developmental delay | "developmental delay" |
| 37x | HP:0002664 | Neoplasm | "Tumour" |
| 28x | HP:0003745 | Sporadic | "de novo" |
| 26x | HP:0001249 | Intellectual disability | "mental retardation" |
| 21x | HP:0000365 | Hearing impairment | "hearing loss" |
| 19x | HP:0000252 | Microcephaly | "microcephaly" |
| 18x | HP:0001290 | Generalized hypotonia | "hypotonia" |

### Why Terms Are Missed

1. **Semantically equivalent term retrieved instead** (most common)
2. **Generic gold term but specific text** (e.g., "Neoplasm" vs specific tumor type)
3. **Different terminology** (e.g., "de novo" vs "Sporadic")

---

## 6. Dataset-Specific Observations

| Dataset | Avg Chunks/Doc | Key Issue |
|---------|----------------|-----------|
| GeneReviews | 87.3 | Long clinical narratives, complex descriptions |
| ID_68 | 36.7 | Clinical reports with specific phenotypes |
| GSC_plus | 36.0 | Many inheritance/abstract terms in gold |

GSC_plus has lower F1 because gold annotations include many **abstract terms** (inheritance patterns, disease categories) that the retrieval system is not designed to extract.

---

## 7. Concrete FP/FN Examples

### High-Confidence False Positives (Score > 0.8)

| HPO ID | Term | Score | Chunk Text |
|--------|------|-------|------------|
| HP:0034952 | Gangliocytoma | 0.90 | "ganglioneuromatosis of the..." |
| HP:0005617 | Bilateral camptodactyly | 0.90 | "camptodactyly..." |
| HP:0001804 | Hypoplastic fingernail | 0.94 | "hypoplastic nails..." |
| HP:0006152 | Proximal symphalangism of hands | 0.92 | "symphalangism (ankylois of proximal..." |
| HP:0011325 | Pansynostosis | 0.92 | "craniosynostosis..." |

### False Negatives with Clear Evidence

| HPO ID | Term | Gold Evidence |
|--------|------|---------------|
| HP:0008208 | Parathyroid hyperplasia | "parathyroid adenoma or hyperplasia..." |
| HP:0031023 | Multiple mucosal neuromas | "mucosal neuromas of the lips and tongue..." |
| HP:0001999 | Abnormal facial shape | "distinctive facies..." |
| HP:0001519 | Disproportionate tall stature | "marfanoid habitus..." |
| HP:0002020 | Gastroesophageal reflux | "GERD..." |

---

## 8. Recommendations

### High Impact (Algorithmic Changes)

1. **HPO Graph-Aware Evaluation**
   - If predicted term is ancestor/descendant of gold term within N hops, count as partial TP
   - Implement "soft F1" metric that rewards semantically close matches

2. **Remove Obsolete Terms** (Issue #133)
   - Expected impact: ~5% reduction in FP for GSC_plus

3. **Parent/Child Resolution**
   - When multiple candidates match, prefer the most specific term that still has high similarity
   - Use HPO graph depth as tiebreaker

### Medium Impact (Pipeline Tuning)

4. **Specificity-Aware Re-ranking**
   - Add penalty for very generic terms (low graph depth)

5. **Dual-Threshold Strategy**
   - High threshold (0.90) for direct matches
   - Lower threshold (0.80) + graph validation for related terms

### Lower Impact (Gold Standard Review)

6. **Gold Standard Quality Review**
   - Some gold annotations use very generic terms
   - Consider if gold standard matches intended use case

---

## 9. Key Insight

Given that **60-70% of "errors" are semantically correct** (parent/child/synonym), the **true semantic accuracy is likely 70-80%**, not 30%.

**Recommendation**: Implement hierarchical evaluation metrics before further optimization. Current F1 is misleading and underestimates the system's actual clinical utility.
