# Multi-Vector Duplicate Results Investigation Report

**Date:** 2025-12-10
**Status:** Bug Identified - Fix Required
**Priority:** High

## Executive Summary

Investigation confirms that duplicate HPO term results appear in multi-vector mode due to the API endpoint not properly using the `query_multi_vector()` method. The aggregation and deduplication logic exists but is being bypassed.

## Problem Statement

When using the multi-vector index (enabled by default with `DEFAULT_MULTI_VECTOR=True`), the same HPO term appears multiple times in search results with different similarity scores.

### Observed Behavior

**Query: "Kleinwuchs" (German for "short stature")**
| Rank | HPO ID | Label | Score |
|------|--------|-------|-------|
| 1 | HP:0004322 | Short stature | 0.98 |
| 2 | HP:0004322 | Short stature | 0.97 |
| 3 | HP:0004322 | Short stature | 0.97 |
| 7 | HP:0003508 | Proportionate short stature | 0.93 |
| 9 | HP:0003508 | Proportionate short stature | 0.93 |
| 10 | HP:0003508 | Proportionate short stature | 0.92 |

**Query: "Nierenzysten" (German for "renal cysts")**
| Rank | HPO ID | Label | Score |
|------|--------|-------|-------|
| 1-3, 7 | HP:0000107 | Renal cyst | 1.00, 0.99, 0.98, 0.93 |
| 4, 5, 10 | HP:0012581 | Simple renal cyst | 0.94, 0.94, 0.93 |
| 8, 9 | HP:0005562 | Multiple renal cysts | 0.93, 0.93 |

### Expected Behavior

Each HPO term should appear only once with its highest (aggregated) score.

## Root Cause Analysis

### Architecture Overview

Multi-vector indexing creates separate embeddings for each component of an HPO term:
- **Label**: Primary term name (e.g., "Short stature")
- **Synonyms**: Alternative names (can have multiple)
- **Definition**: Full definition text

This results in ~60,385 documents for ~19,393 HPO terms (~3.1 docs per term on average).

### The Bug

The API endpoint at `api/routers/query_router.py:217` calls:

```python
query_results_dict = await execute_hpo_retrieval_for_api(
    text=request.text,
    language=language_to_use,
    retriever=retriever,
    num_results=request.num_results,
    # ... other params
    # NOTE: multi_vector params are NOT passed!
)
```

Inside `phentrieve/retrieval/api_helpers.py:106`, the function uses:

```python
query_results = retriever.query(  # <-- WRONG METHOD
    text=segment_to_process,
    n_results=rerank_count if enable_reranker else num_results,
    include_similarities=True,
)
```

**This calls `retriever.query()` instead of `retriever.query_multi_vector()`!**

### What Should Happen

The `query_multi_vector()` method in `dense_retriever.py:472-542`:
1. Queries the multi-vector index with multiplied n_results
2. Calls `aggregate_multi_vector_results()` which:
   - Groups results by HPO ID (`group_results_by_hpo_id()`)
   - Aggregates component scores using the specified strategy
   - Returns deduplicated results sorted by aggregated score

## Existing Deduplication Infrastructure

### Aggregation Module (`phentrieve/retrieval/aggregation.py`)

Fully implemented aggregation strategies:

| Strategy | Description |
|----------|-------------|
| `label_only` | Use only label score |
| `label_synonyms_max` | **Default** - Maximum of label or synonyms |
| `label_synonyms_min` | Minimum (conservative approach) |
| `all_weighted` | Weighted average with configurable weights |
| `all_max` | Maximum across all components |
| `all_min` | Minimum across all components |
| `custom` | User-defined formula via safe AST evaluation |

### Key Functions

1. **`group_results_by_hpo_id(results)`** - Groups ChromaDB results by HPO ID
2. **`aggregate_scores(...)`** - Applies aggregation strategy to component scores
3. **`aggregate_multi_vector_results(results, strategy, ...)`** - Full pipeline

### Query Orchestrator (`phentrieve/retrieval/query_orchestrator.py`)

Contains `_execute_multi_vector_query()` helper at lines 96-129 that properly calls `query_multi_vector()` and converts results. This is used by CLI but not API.

### API Schema (`api/schemas/query_schemas.py`)

The schema already has all required parameters:
- `multi_vector: bool = Field(default=DEFAULT_MULTI_VECTOR)`
- `aggregation_strategy: Literal[...]`
- `component_weights: Optional[dict[str, float]]`
- `custom_formula: Optional[str]`

**But these parameters are never passed through to the retrieval function!**

## Test Coverage Gap

There's an E2E test at `tests/e2e/test_api_e2e.py:419-442` that checks for unique HPO IDs:

```python
def test_query_returns_unique_hpo_terms(self, api_query_endpoint: str):
    """Verify query results contain unique HPO IDs (no duplicates)."""
    # ...
    unique_hpo_ids = set(hpo_ids)
    assert len(hpo_ids) == len(unique_hpo_ids), (
        f"Results should not contain duplicate HPO IDs, got: {hpo_ids}"
    )
```

This test may be passing with single-vector indexes but would fail with multi-vector.

## Recommended Fix

### Option 1: Modify `execute_hpo_retrieval_for_api()` (Recommended)

Add multi-vector support to the API helper function:

```python
async def execute_hpo_retrieval_for_api(
    text: str,
    language: str,
    retriever: DenseRetriever,
    num_results: int,
    similarity_threshold: float,
    enable_reranker: bool,
    cross_encoder: Optional[CrossEncoder],
    rerank_count: int,
    include_details: bool = False,
    detect_query_assertion: bool = True,
    query_assertion_language: Optional[str] = None,
    query_assertion_preference: str = "dependency",
    debug: bool = False,
    # NEW: Multi-vector parameters
    multi_vector: bool = False,
    aggregation_strategy: str = "label_synonyms_max",
    component_weights: Optional[dict[str, float]] = None,
    custom_formula: Optional[str] = None,
) -> dict[str, Any]:
    # ...

    # Detect index type and use appropriate query method
    index_type = retriever.detect_index_type()

    if multi_vector and index_type == "multi_vector":
        # Use multi-vector query with aggregation
        query_results = retriever.query_multi_vector(
            text=segment_to_process,
            n_results=rerank_count if enable_reranker else num_results,
            aggregation_strategy=aggregation_strategy,
            component_weights=component_weights,
            custom_formula=custom_formula,
        )
        # Convert to standard format for downstream processing
        # ...
    else:
        # Original single-vector query
        query_results = retriever.query(
            text=segment_to_process,
            n_results=rerank_count if enable_reranker else num_results,
            include_similarities=True,
        )
```

### Option 2: Pass Parameters Through Router

Update `api/routers/query_router.py` to pass multi-vector params:

```python
query_results_dict = await execute_hpo_retrieval_for_api(
    # ... existing params
    multi_vector=request.multi_vector,
    aggregation_strategy=request.aggregation_strategy,
    component_weights=request.component_weights,
    custom_formula=request.custom_formula,
)
```

## Implementation Checklist

1. [ ] Add multi-vector parameters to `execute_hpo_retrieval_for_api()` function signature
2. [ ] Detect index type using `retriever.detect_index_type()`
3. [ ] Conditionally call `query_multi_vector()` vs `query()`
4. [ ] Convert multi-vector results format for downstream processing
5. [ ] Update router to pass multi-vector params from request
6. [ ] Update E2E tests to cover multi-vector mode
7. [ ] Verify deduplication works with reranking enabled
8. [ ] Test component_scores in response when multi_vector=True

## Files to Modify

| File | Changes |
|------|---------|
| `phentrieve/retrieval/api_helpers.py` | Add multi_vector params, conditional query method |
| `api/routers/query_router.py` | Pass multi_vector params to helper function |
| `tests/e2e/test_api_e2e.py` | Add multi-vector specific tests |

## Impact Assessment

- **User Impact:** High - Duplicate results confuse users and waste result slots
- **Fix Complexity:** Medium - Infrastructure exists, just needs wiring
- **Risk:** Low - Changes are additive, backward compatible
- **Testing:** Required - E2E tests should verify deduplication

## References

- Issue #136: Multi-vector embedding design
- PR #138: Multi-vector implementation (merged)
- `phentrieve/retrieval/aggregation.py`: Aggregation strategies
- `phentrieve/retrieval/dense_retriever.py:472-542`: `query_multi_vector()` method
