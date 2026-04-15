# Default Settings Optimization Based on Benchmark Findings

**Status:** ✅ Completed
**Date:** 2025-12-10
**Completed:** 2025-12-10
**PR:** #141
**Related Analysis:** `plan/05-analysis/EXTRACTION-BENCHMARK-ACTION-PLAN.md`, `plan/05-analysis/EXTRACTION-BENCHMARK-DEEP-ANALYSIS.md`
**Priority:** High
**Design Principles:** DRY, KISS, SOLID, Modularization

---

## Executive Summary

Benchmark analysis proves optimized thresholds and enhanced chunking significantly improve HPO extraction:

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **F1** | 0.145 | **0.328** | +126% |
| Precision | 0.088 | 0.310 | +252% |
| Recall | 0.422 | 0.353 | -16% (acceptable) |

**Root Cause:** Sliding window creates broken fragments → wrong HPO matches.

---

## Critical Issue: CLI/API Defaults Are Inconsistent!

### Current Discrepancy Matrix

| Parameter | config.py | CLI utils | CLI commands | API Schema | API Router | Frontend |
|-----------|-----------|-----------|--------------|------------|------------|----------|
| `window_size` | **7** | 3 | 3 | 2 | 7 | 2 |
| `step_size` | 1 | 1 | 1 | 1 | 1 | 1 |
| `split_threshold` | 0.5 | 0.5 | 0.5 | **0.3** | 0.5 | **0.25** |
| `min_segment_length` | 3 | **2** | 2 | **1** | 3 | **1** |
| `chunk_retrieval_threshold` | N/A | N/A | **0.3** | **0.3** | 0.3 | **0.5** |
| `min_confidence_aggregated` | N/A | N/A | **0.35** | **0.35** | 0.35 | **0.4** |
| `default_strategy` | `SLIDING_WINDOW` | N/A | varies* | `punct_conj_cleaned` | N/A | `punct_conj_cleaned` |

*CLI `chunk` uses `sliding_window`, others use `sliding_window_punct_conj_cleaned`

### Problems Identified

1. **6 different sources of truth** for the same parameters
2. **CLI `chunk` command** uses different default strategy than all others
3. **Frontend thresholds** don't match API defaults
4. **config.py** has one value, but API/CLI override it locally
5. **No centralized constants** for retrieval thresholds

---

## Best Practices Research (2024-2025)

Based on industry research from [Unstructured](https://unstructured.io/blog/chunking-for-rag-best-practices), [Databricks](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089), and [IBM](https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai):

| Recommendation | Industry Standard | Our Benchmark Finding |
|----------------|-------------------|----------------------|
| Chunk size | 250-500 tokens | Small chunks (phenotype phrases) |
| Overlap | 10-20% | Step size 1 (high overlap) |
| Threshold strategy | Percentile-based splitting | 0.5-0.7 for similarity |
| Preserve boundaries | Sentence/topic coherence | Conjunction + punctuation splitting |

**Key insight:** Clinical phenotype extraction differs from general RAG - we need very small, focused chunks containing single phenotype mentions.

---

## Unified Default Values (Target)

### Sliding Window Parameters

| Parameter | Target Value | Rationale |
|-----------|--------------|-----------|
| `window_size_tokens` | **3** | Validated by CLI usage; small for phenotype focus |
| `step_size_tokens` | **1** | Maximum overlap for boundary detection |
| `splitting_threshold` | **0.5** | Balance between over/under-splitting |
| `min_segment_length_words` | **2** | Allow short phenotype phrases |

### Retrieval Thresholds (Benchmark-Validated)

| Parameter | Target Value | Rationale |
|-----------|--------------|-----------|
| `chunk_retrieval_threshold` | **0.7** | +83% F1 improvement |
| `min_confidence_aggregated` | **0.75** | Reduces false positives |

### Strategy Default

| Component | Target Value | Rationale |
|-----------|--------------|-----------|
| All components | `sliding_window_punct_conj_cleaned` | Benchmark validated; handles lists |

---

## Implementation Plan (Revised)

### Phase 0: Add Centralized Constants (DRY - Single Source of Truth)

**File:** `phentrieve/config.py`

Add new constants section after line 119:

```python
# =============================================================================
# Sliding Window Chunking Defaults (Unified)
# =============================================================================
# These defaults are used by all components: CLI, API, Frontend
# Validated via benchmarking - see plan/05-analysis/EXTRACTION-BENCHMARK-ACTION-PLAN.md

DEFAULT_WINDOW_SIZE_TOKENS = 3
DEFAULT_STEP_SIZE_TOKENS = 1
DEFAULT_SPLITTING_THRESHOLD = 0.5
DEFAULT_MIN_SEGMENT_LENGTH_WORDS = 2

# =============================================================================
# HPO Extraction Thresholds (Benchmark-Validated)
# =============================================================================
# Higher thresholds reduce false positives at slight cost to recall
# F1 improvement: 0.145 → 0.328 (+126%)

_DEFAULT_CHUNK_RETRIEVAL_THRESHOLD_FALLBACK = 0.7
_DEFAULT_MIN_CONFIDENCE_AGGREGATED_FALLBACK = 0.75

# Public constants (configurable via phentrieve.yaml)
DEFAULT_CHUNK_RETRIEVAL_THRESHOLD: float = get_config_value(
    "extraction", _DEFAULT_CHUNK_RETRIEVAL_THRESHOLD_FALLBACK, "chunk_threshold"
)
DEFAULT_MIN_CONFIDENCE_AGGREGATED: float = get_config_value(
    "extraction", _DEFAULT_MIN_CONFIDENCE_AGGREGATED_FALLBACK, "min_confidence"
)

# =============================================================================
# Default Chunking Strategy
# =============================================================================
DEFAULT_CHUNKING_STRATEGY = "sliding_window_punct_conj_cleaned"
```

Update `__all__` exports:

```python
    # Sliding window defaults
    "DEFAULT_WINDOW_SIZE_TOKENS",
    "DEFAULT_STEP_SIZE_TOKENS",
    "DEFAULT_SPLITTING_THRESHOLD",
    "DEFAULT_MIN_SEGMENT_LENGTH_WORDS",
    # HPO extraction thresholds
    "DEFAULT_CHUNK_RETRIEVAL_THRESHOLD",
    "DEFAULT_MIN_CONFIDENCE_AGGREGATED",
    # Chunking strategy
    "DEFAULT_CHUNKING_STRATEGY",
```

Update `get_sliding_window_config_with_params()` to use constants:

```python
def get_sliding_window_config_with_params(
    window_size=DEFAULT_WINDOW_SIZE_TOKENS,
    step_size=DEFAULT_STEP_SIZE_TOKENS,
    threshold=DEFAULT_SPLITTING_THRESHOLD,
    min_segment_length=DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
):
```

Update `DEFAULT_CHUNK_PIPELINE_CONFIG`:

```python
DEFAULT_CHUNK_PIPELINE_CONFIG = SLIDING_WINDOW_PUNCT_CONJ_CLEANED_CONFIG
```

---

### Phase 1: Update CLI Utils (Use Central Constants)

**File:** `phentrieve/cli/utils.py` (lines 55-62)

```python
from phentrieve.config import (
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)

def resolve_chunking_pipeline_config(
    chunking_pipeline_config_file: Optional[Path],
    strategy_arg: str,
    window_size: int = DEFAULT_WINDOW_SIZE_TOKENS,  # was 3
    step_size: int = DEFAULT_STEP_SIZE_TOKENS,       # was 1
    threshold: float = DEFAULT_SPLITTING_THRESHOLD,  # was 0.5
    min_segment_length: int = DEFAULT_MIN_SEGMENT_LENGTH_WORDS,  # was 2
) -> list[dict]:
```

---

### Phase 2: Update CLI Commands (Use Central Constants)

**File:** `phentrieve/cli/text_commands.py`

Add imports at top:

```python
from phentrieve.config import (
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_MODEL,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)
```

Update `interactive()` command defaults (lines 45-150):

```python
    strategy: ... = DEFAULT_CHUNKING_STRATEGY,  # was "sliding_window_punct_conj_cleaned"
    window_size: ... = DEFAULT_WINDOW_SIZE_TOKENS,  # was 3
    step_size: ... = DEFAULT_STEP_SIZE_TOKENS,  # was 1
    split_threshold: ... = DEFAULT_SPLITTING_THRESHOLD,  # was 0.5
    min_segment_length: ... = DEFAULT_MIN_SEGMENT_LENGTH_WORDS,  # was 2
    chunk_retrieval_threshold: ... = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,  # was 0.3 → 0.7
    aggregated_term_confidence: ... = DEFAULT_MIN_CONFIDENCE_AGGREGATED,  # was 0.35 → 0.75
```

Update `process_text_for_hpo_command()` defaults (lines 255-387):

```python
    strategy: ... = DEFAULT_CHUNKING_STRATEGY,
    window_size: ... = DEFAULT_WINDOW_SIZE_TOKENS,
    step_size: ... = DEFAULT_STEP_SIZE_TOKENS,
    split_threshold: ... = DEFAULT_SPLITTING_THRESHOLD,
    min_segment_length: ... = DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    chunk_retrieval_threshold: ... = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    aggregated_term_confidence: ... = DEFAULT_MIN_CONFIDENCE_AGGREGATED,
```

Update `chunk_text_command()` defaults (lines 830-862):

```python
    strategy: ... = DEFAULT_CHUNKING_STRATEGY,  # was "sliding_window" → now consistent!
    window_size: ... = DEFAULT_WINDOW_SIZE_TOKENS,
    step_size: ... = DEFAULT_STEP_SIZE_TOKENS,
    split_threshold: ... = DEFAULT_SPLITTING_THRESHOLD,
    min_segment_length: ... = DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
```

---

### Phase 3: Update API Schema (Use Central Constants)

**File:** `api/schemas/text_processing_schemas.py`

Update imports:

```python
from phentrieve.config import (
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)
```

Update `TextProcessingRequest` fields:

```python
    chunking_strategy: str = Field(
        default=DEFAULT_CHUNKING_STRATEGY,  # was hardcoded string
        ...
    )
    window_size: Optional[int] = Field(
        default=DEFAULT_WINDOW_SIZE_TOKENS,  # was 2 → 3
        ...
    )
    step_size: Optional[int] = Field(
        default=DEFAULT_STEP_SIZE_TOKENS,  # was 1 (unchanged)
        ...
    )
    split_threshold: Optional[float] = Field(
        default=DEFAULT_SPLITTING_THRESHOLD,  # was 0.3 → 0.5
        ...
    )
    min_segment_length: Optional[int] = Field(
        default=DEFAULT_MIN_SEGMENT_LENGTH_WORDS,  # was 1 → 2
        ...
    )
    chunk_retrieval_threshold: Optional[float] = Field(
        default=DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,  # was 0.3 → 0.7
        ...
    )
    aggregated_term_confidence: Optional[float] = Field(
        default=DEFAULT_MIN_CONFIDENCE_AGGREGATED,  # was 0.35 → 0.75
        json_schema_extra={"example": 0.75},
        ...
    )
```

---

### Phase 4: Update API Router (Use Central Constants)

**File:** `api/routers/text_processing_router.py`

Update imports:

```python
from phentrieve.config import (
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_MODEL,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)
```

Update `_get_chunking_config_for_api()` (lines 56-60):

```python
    ws = request.window_size if request.window_size is not None else DEFAULT_WINDOW_SIZE_TOKENS
    ss = request.step_size if request.step_size is not None else DEFAULT_STEP_SIZE_TOKENS
    th = request.split_threshold if request.split_threshold is not None else DEFAULT_SPLITTING_THRESHOLD
    msl = request.min_segment_length if request.min_segment_length is not None else DEFAULT_MIN_SEGMENT_LENGTH_WORDS
```

Update `process_text_extract_hpo()` (lines 340-342):

```python
    chunk_retrieval_threshold=request.chunk_retrieval_threshold or DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    min_confidence_for_aggregated=request.aggregated_term_confidence or DEFAULT_MIN_CONFIDENCE_AGGREGATED,
```

---

### Phase 5: Update Frontend (Use Consistent Values)

**File:** `frontend/src/components/QueryInterface.vue`

Update `data()` return values (lines 890-901):

```javascript
      chunkingStrategy: 'sliding_window_punct_conj_cleaned',  // unchanged
      windowSize: 3,  // was 2 → 3
      stepSize: 1,    // unchanged
      splitThreshold: 0.5,  // was 0.25 → 0.5
      minSegmentLength: 2,  // was 1 → 2
      chunkRetrievalThreshold: 0.7,  // was 0.5 → 0.7
      aggregatedTermConfidence: 0.75,  // was 0.4 → 0.75
```

---

### Phase 6: Update Extraction Benchmark (Use Central Constants)

**File:** `phentrieve/benchmark/extraction_benchmark.py`

Update imports:

```python
from phentrieve.config import (
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)
```

Update `ExtractionConfig` (lines 53-54):

```python
    chunk_retrieval_threshold: float = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
    min_confidence_for_aggregated: float = DEFAULT_MIN_CONFIDENCE_AGGREGATED
```

Update `_lazy_init()` chunking pipeline (lines 89-102):

```python
        self._pipeline = TextProcessingPipeline(
            language=self.config.language,
            chunking_pipeline_config=[
                {"type": "paragraph"},
                {"type": "sentence"},
                {"type": "conjunction"},
                {"type": "fine_grained_punctuation"},
                {
                    "type": "sliding_window",
                    "config": {
                        "window_size_tokens": DEFAULT_WINDOW_SIZE_TOKENS,
                        "step_size_tokens": DEFAULT_STEP_SIZE_TOKENS,
                        "splitting_threshold": DEFAULT_SPLITTING_THRESHOLD,
                        "min_split_segment_length_words": DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
                    },
                },
                {"type": "final_chunk_cleaner"},
            ],
            ...
        )
```

---

## File Change Summary

| File | Changes | Lines |
|------|---------|-------|
| `phentrieve/config.py` | Add centralized constants, update functions | ~40 |
| `phentrieve/cli/utils.py` | Import + use constants | ~10 |
| `phentrieve/cli/text_commands.py` | Import + use constants in 3 commands | ~30 |
| `api/schemas/text_processing_schemas.py` | Import + use constants | ~15 |
| `api/routers/text_processing_router.py` | Import + use constants | ~10 |
| `frontend/src/components/QueryInterface.vue` | Update 6 default values | 6 |
| `phentrieve/benchmark/extraction_benchmark.py` | Import + use constants | ~20 |

**Total:** ~131 lines across 7 files

---

## Before/After Comparison

### After Implementation (All Unified)

| Parameter | config.py | CLI | API | Frontend |
|-----------|-----------|-----|-----|----------|
| `window_size` | 3 | 3 | 3 | 3 |
| `step_size` | 1 | 1 | 1 | 1 |
| `split_threshold` | 0.5 | 0.5 | 0.5 | 0.5 |
| `min_segment_length` | 2 | 2 | 2 | 2 |
| `chunk_retrieval_threshold` | 0.7 | 0.7 | 0.7 | 0.7 |
| `min_confidence_aggregated` | 0.75 | 0.75 | 0.75 | 0.75 |
| `default_strategy` | `punct_conj_cleaned` | `punct_conj_cleaned` | `punct_conj_cleaned` | `punct_conj_cleaned` |

---

## Validation Checklist

### Pre-Implementation
- [ ] Document current behavior for regression testing

### Post-Implementation

1. **Python checks:**
   ```bash
   make check && make typecheck-fast && make test
   ```

2. **Frontend checks:**
   ```bash
   make frontend-lint && make frontend-test
   ```

3. **Congruence test (CLI vs API should produce identical results):**
   ```bash
   # CLI
   phentrieve text process "Patient has microcephaly and seizures" -o json_lines

   # API
   curl -X POST http://localhost:8734/api/v1/text/process \
     -H "Content-Type: application/json" \
     -d '{"text_content": "Patient has microcephaly and seizures"}'
   ```

   **Expected:** Same HPO terms, same scores (within floating point tolerance)

4. **Benchmark validation:**
   ```bash
   phentrieve benchmark extraction run tests/data/en/phenobert --dataset GeneReviews
   ```
   Expected: F1 ~0.33

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Higher thresholds (0.3→0.7) | May miss low-confidence matches | Validated by benchmarks; configurable |
| Strategy change for CLI `chunk` | Existing scripts may see different output | Document in changelog |
| Frontend defaults change | Users see different results | UI shows values; adjustable |

---

## Not Included (Explicitly Excluded)

- `top_term_per_chunk: True` - Per user request, keeping as `False`

---

## Sources

- [Unstructured: Chunking for RAG Best Practices](https://unstructured.io/blog/chunking-for-rag-best-practices)
- [Databricks: Ultimate Guide to Chunking Strategies](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- [IBM: Chunking Strategies for RAG](https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai)
- [Firecrawl: Best Chunking Strategies 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)

---

## References

- `plan/05-analysis/EXTRACTION-BENCHMARK-ACTION-PLAN.md` - Benchmark validation
- `plan/05-analysis/EXTRACTION-BENCHMARK-DEEP-ANALYSIS.md` - FP/FN analysis
