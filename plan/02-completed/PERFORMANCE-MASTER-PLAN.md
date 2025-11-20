# Performance Optimization Master Plan

**Status:** Active
**Date:** 2025-11-18
**Priority:** Critical (P0)
**Estimated Effort:** 1 week (immediate wins) + 2-3 weeks (long-term)
**Principles:** KISS, Profile First, Fix Critical Bugs Before Optimizing

---

## Executive Summary

This plan addresses **critical production issues** where the frontend times out on real documents, then provides a data-driven roadmap for long-term performance optimization.

### Current State

**Test Case 1:** `tests/data/de/phentrieve/annotations/clinical_case_001.json` (125 chars, 4 annotations)
- Status: ❌ "Relatively slow" (~10s, should be instant)
- Root Cause: Model loading on every request

**Test Case 2:** `tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK1379.json` (1588 chars, 22 annotations)
- Status: ❌ Frontend timeout: "Es konnte keine Verbindung zum Server hergestellt werden"
- Root Cause: 60+ seconds processing → Frontend timeout at ~30-60s

### Goals

1. **Week 1:** Fix critical bugs (timeouts, model caching) → All test cases work
2. **Week 2:** Profile with real data → Identify actual bottlenecks (data-driven!)
3. **Week 3-4:** Implement validated optimizations → 5-10x overall speedup
4. **Week 5+:** Future enhancements based on profiling data

---

## Table of Contents

1. [Phase 0: Immediate Wins (Days 1-2)](#phase-0-immediate-wins-days-1-2)
2. [Phase 1: Data-Driven Profiling (Day 3)](#phase-1-data-driven-profiling-day-3)
3. [Phase 2: Core Optimizations (Week 2)](#phase-2-core-optimizations-week-2)
4. [Phase 3: Infrastructure Improvements (Week 3-4)](#phase-3-infrastructure-improvements-week-3-4)
5. [Phase 4: Future Enhancements (Week 5+)](#phase-4-future-enhancements-week-5)
6. [Validation & Testing Strategy](#validation--testing-strategy)
7. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Phase 0: Immediate Wins (Days 1-2)

**Goal:** Fix critical production bugs with minimal code changes (< 50 lines total!)

### Quick Win 1: Model Caching (2 hours) ⚡

**Problem:** API loads models on EVERY request (5-10s overhead)

**Root Cause Analysis:**
```python
# File: api/routers/text_processing_router.py:237-253
# ❌ CRITICAL BUG: Bypasses existing caching!
retrieval_sbert_model = await run_in_threadpool(
    load_embedding_model,  # ← Loads fresh model every time!
    model_name=retrieval_model_name_to_load,
)
```

**Solution:** Use existing dependency injection (already implemented in `api/dependencies.py`!)

```python
# File: api/routers/text_processing_router.py

# BEFORE (❌ Lines 237-253 - 5-10s per request):
retrieval_sbert_model = await run_in_threadpool(
    load_embedding_model,
    model_name=retrieval_model_name_to_load,
    trust_remote_code=request.trust_remote_code or False,
)
sbert_for_chunking = retrieval_sbert_model
if sbert_for_chunking_name_to_load != retrieval_model_name_to_load:
    sbert_for_chunking = await run_in_threadpool(
        load_embedding_model,
        model_name=sbert_for_chunking_name_to_load,
        trust_remote_code=request.trust_remote_code or False,
    )
retriever = await run_in_threadpool(
    DenseRetriever.from_model_name,
    model=retrieval_sbert_model,
    model_name=retrieval_model_name_to_load,
    min_similarity=request.chunk_retrieval_threshold or 0.3,
)

# AFTER (✅ Use existing cached dependencies - ~0.1s):
from api.dependencies import (
    get_sbert_model_dependency,
    get_dense_retriever_dependency,
    get_cross_encoder_dependency,
)

# Get cached retrieval model
retrieval_sbert_model = await get_sbert_model_dependency(
    model_name_requested=retrieval_model_name_to_load,
    device_override=None,  # Use default
    trust_remote_code=request.trust_remote_code or False,
)

# Get cached chunking model (or reuse retrieval model)
if sbert_for_chunking_name_to_load != retrieval_model_name_to_load:
    sbert_for_chunking = await get_sbert_model_dependency(
        model_name_requested=sbert_for_chunking_name_to_load,
        device_override=None,
        trust_remote_code=request.trust_remote_code or False,
    )
else:
    sbert_for_chunking = retrieval_sbert_model  # Reuse!

# Get cached retriever
retriever = await get_dense_retriever_dependency(
    sbert_model_name_for_retriever=retrieval_model_name_to_load
)

# Get cached cross-encoder (if enabled)
cross_enc = None
if request.enable_reranker:
    reranker_to_load = request.reranker_model_name
    if request.reranker_mode == "monolingual" and actual_language != "en":
        reranker_to_load = request.monolingual_reranker_model_name
    if reranker_to_load:
        cross_enc = await get_cross_encoder_dependency(
            reranker_model_name=reranker_to_load,
            device_override=None,
        )
```

**Impact:**
- **Before:** 5-10s model loading + processing
- **After:** ~0.1s model retrieval + processing
- **Speedup:** 50-100x for model loading overhead
- **Code Changes:** ~15 lines (replacing existing lines)

**Validation:**
```bash
# Test with real small file (should be instant after fix)
time phentrieve text process tests/data/de/phentrieve/annotations/clinical_case_001.json

# Expected: <2s total (was ~10s before)
```

**Deliverables:**
- [ ] Update `api/routers/text_processing_router.py` (lines 237-280)
- [ ] Test with `clinical_case_001.json` - verify <2s response
- [ ] Test with API endpoint - verify model not reloaded on 2nd request
- [ ] Commit: "fix: Use cached model dependencies in text processing endpoint"

---

### Quick Win 2: API Timeout Protection (2 hours) ⚡

**Problem:** No timeout → Frontend waits forever → Shows "Verbindung verloren"

**Solution:** Add graceful timeout with adaptive thresholds

```python
# File: api/routers/text_processing_router.py

import asyncio
from fastapi import status

@router.post("/process", response_model=TextProcessingResponseAPI)
async def process_text_extract_hpo(request: TextProcessingRequest):
    """
    Process clinical text to extract HPO terms.

    Includes adaptive timeout based on text length to prevent frontend disconnects.
    """
    logger.info(
        f"API: Received request to process text. "
        f"Language: {request.language}, Strategy: {request.chunking_strategy}"
    )

    # Calculate adaptive timeout based on text length
    text_length = len(request.text_content)
    if text_length < 500:
        timeout_seconds = 30
    elif text_length < 2000:
        timeout_seconds = 60
    elif text_length < 5000:
        timeout_seconds = 120
    else:
        timeout_seconds = 180

    logger.info(
        f"API: Processing {text_length} chars with {timeout_seconds}s timeout"
    )

    try:
        # Wrap processing with timeout
        return await asyncio.wait_for(
            _process_text_internal(request),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.error(
            f"API: Request timed out after {timeout_seconds}s "
            f"(text length: {text_length} chars)"
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"Text processing timed out after {timeout_seconds} seconds. "
                f"Text length: {text_length} characters. "
                f"Please try: (1) reducing text length, "
                f"(2) using 'simple' chunking strategy, or "
                f"(3) disabling reranker."
            )
        )


async def _process_text_internal(request: TextProcessingRequest):
    """
    Internal processing function.

    This is the existing process_text_extract_hpo logic,
    now wrapped by timeout handler above.
    """
    # Move ALL existing code from process_text_extract_hpo here
    # (lines 206-435 from current implementation)
    try:
        # ... existing implementation ...
        pass
    except HTTPException:
        raise
    except ValueError as ve:
        # ... existing error handlers ...
        pass
```

**Frontend Update (for better UX):**

```javascript
// File: frontend/src/services/api.js (or equivalent)

async function processText(textContent, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000); // 180s max

  try {
    const response = await fetch('/api/v1/text/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text_content: textContent,
        ...options
      }),
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (response.status === 504) {
      const error = await response.json();
      throw new Error(
        'Die Textverarbeitung dauerte zu lange. ' +
        error.detail || 'Bitte kürzen Sie den Text oder verwenden Sie eine einfachere Strategie.'
      );
    }

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();

  } catch (error) {
    clearTimeout(timeout);
    if (error.name === 'AbortError') {
      throw new Error('Die Anfrage wurde abgebrochen (Timeout).');
    }
    throw error;
  }
}
```

**Impact:**
- **Before:** Frontend shows "Es konnte keine Verbindung zum Server hergestellt werden"
- **After:** Clear error message with actionable suggestions
- **UX:** Users understand what went wrong and how to fix it

**Validation:**
```bash
# Test timeout with long processing (before other optimizations)
# Should return 504 with clear message instead of hanging

curl -X POST http://localhost:8734/api/v1/text/process \
  -H "Content-Type: application/json" \
  -d @tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK1379.json \
  -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"
```

**Deliverables:**
- [ ] Add timeout wrapper to `process_text_extract_hpo()`
- [ ] Move existing logic to `_process_text_internal()`
- [ ] Update frontend error handling
- [ ] Test with `GeneReviews_NBK1379.json` - verify graceful timeout
- [ ] Commit: "fix: Add adaptive timeout to text processing endpoint"

---

### Quick Win 3: Batch ChromaDB Queries (4 hours) ⚡

**Problem:** Sequential queries kill performance for multi-chunk documents

**Root Cause:**
```python
# File: phentrieve/text_processing/hpo_extraction_orchestrator.py
# ❌ Sequential queries (1-2s each × 20-30 chunks = 20-60s!)
for chunk_idx, chunk_text in enumerate(text_chunks):
    results = retriever.query(chunk_text, n_results=10)  # ← Separate query!
    # Process results...
```

**Solution:** ChromaDB supports batch queries natively!

```python
# File: phentrieve/retrieval/dense_retriever.py

class DenseRetriever:
    """Dense retrieval using sentence embeddings and ChromaDB."""

    def query_batch(
        self,
        query_texts: list[str],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include_distances: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Query ChromaDB for multiple texts at once (batched for performance).

        Args:
            query_texts: List of query texts to search for
            n_results: Number of results to return per query
            where: Optional metadata filter
            include_distances: Whether to include similarity distances

        Returns:
            List of results (one dict per query text) with format:
            [
                {
                    "ids": [...],
                    "distances": [...],
                    "metadatas": [...],
                    "documents": [...]
                },
                ...
            ]
        """
        if not query_texts:
            return []

        logger.info(
            f"Batch querying ChromaDB: {len(query_texts)} queries, "
            f"{n_results} results each"
        )

        from phentrieve.profiling import TimingContext

        with TimingContext("ChromaDB batch query", log_on_exit=True):
            # ChromaDB's native batch query API
            raw_results = self.collection.query(
                query_texts=query_texts,  # ✅ Batch query!
                n_results=n_results,
                where=where,
                include=["metadatas", "distances", "documents"] if include_distances
                        else ["metadatas", "documents"],
            )

        # Parse results into per-query format
        parsed_results = []
        for i in range(len(query_texts)):
            parsed_results.append({
                "ids": raw_results["ids"][i],
                "distances": raw_results["distances"][i] if "distances" in raw_results else None,
                "metadatas": raw_results["metadatas"][i],
                "documents": raw_results["documents"][i] if "documents" in raw_results else None,
            })

        logger.info(
            f"Batch query returned {sum(len(r['ids']) for r in parsed_results)} total results"
        )

        return parsed_results

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include_distances: bool = True,
    ) -> dict[str, Any]:
        """
        Query ChromaDB for a single text.

        This is a convenience wrapper around query_batch() for single queries.
        For multiple queries, use query_batch() directly for better performance.
        """
        results = self.query_batch(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include_distances=include_distances,
        )
        return results[0] if results else {"ids": [], "distances": [], "metadatas": [], "documents": []}
```

**Update Orchestrator:**

```python
# File: phentrieve/text_processing/hpo_extraction_orchestrator.py

def orchestrate_hpo_extraction(
    text_chunks: list[str],
    assertion_statuses: list[str | None],
    retriever: DenseRetriever,
    cross_encoder: CrossEncoder | None = None,
    language: str = "en",
    chunk_retrieval_threshold: float = 0.3,
    num_results_per_chunk: int = 10,
    reranker_mode: str = "cross-lingual",
    translation_dir_path: Path | None = None,
    min_confidence_for_aggregated: float = 0.35,
    top_term_per_chunk: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Orchestrate HPO term extraction from text chunks.

    Uses batched retrieval for improved performance.
    """
    logger.info(f"Orchestrating HPO extraction for {len(text_chunks)} chunks")

    # ✅ BATCH QUERY (instead of sequential loop)
    logger.info("Performing batch retrieval from ChromaDB...")
    batch_retrieval_results = retriever.query_batch(
        query_texts=text_chunks,
        n_results=num_results_per_chunk,
    )

    # Process each chunk with pre-fetched results
    detailed_chunk_results: list[dict[str, Any]] = []

    for chunk_idx, (chunk_text, assertion_status, retrieval_result) in enumerate(
        zip(text_chunks, assertion_statuses, batch_retrieval_results)
    ):
        logger.debug(f"Processing chunk {chunk_idx + 1}/{len(text_chunks)}")

        # Filter by similarity threshold
        matches = []
        for hpo_id, distance, metadata in zip(
            retrieval_result["ids"],
            retrieval_result["distances"] or [],
            retrieval_result["metadatas"],
        ):
            # Convert distance to similarity (ChromaDB returns L2 distance)
            similarity = 1.0 - (distance / 2.0) if distance is not None else 0.0

            if similarity >= chunk_retrieval_threshold:
                matches.append({
                    "id": hpo_id,
                    "name": metadata.get("label", ""),
                    "score": similarity,
                    "metadata": metadata,
                })

        # Re-rank if cross-encoder provided
        if cross_encoder and matches:
            matches = _rerank_matches(
                chunk_text=chunk_text,
                matches=matches,
                cross_encoder=cross_encoder,
                reranker_mode=reranker_mode,
                language=language,
                translation_dir_path=translation_dir_path,
            )

        # Store detailed results
        detailed_chunk_results.append({
            "chunk_idx": chunk_idx,
            "chunk_text": chunk_text,
            "assertion_status": assertion_status,
            "matches": matches,
        })

    # Aggregate results (existing logic)
    logger.info("Aggregating HPO terms across chunks...")
    aggregated_terms = _aggregate_hpo_terms(
        detailed_chunk_results=detailed_chunk_results,
        min_confidence=min_confidence_for_aggregated,
        top_term_per_chunk=top_term_per_chunk,
    )

    logger.info(
        f"Orchestration complete: {len(aggregated_terms)} aggregated terms from "
        f"{len(detailed_chunk_results)} chunks"
    )

    return aggregated_terms, detailed_chunk_results
```

**Impact:**
- **Before:** 20-30 sequential queries × 1-2s = 20-60s
- **After:** 1 batch query × 2-3s = 2-3s
- **Speedup:** 10-20x for multi-chunk documents
- **Code Changes:** ~30 lines (new method + orchestrator update)

**Validation:**
```bash
# Test with medium-sized document
time phentrieve text process tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK1379.json

# Expected: <10s (was 60+ seconds, timing out)
```

**Deliverables:**
- [ ] Add `query_batch()` method to `DenseRetriever`
- [ ] Update `query()` to use `query_batch()` internally
- [ ] Update `orchestrate_hpo_extraction()` to use batch queries
- [ ] Unit tests for `query_batch()`
- [ ] Integration test with `GeneReviews_NBK1379.json`
- [ ] Commit: "perf: Add batch query support to DenseRetriever"

---

### Phase 0 Summary

**Total Time:** 1-2 days
**Total Code Changes:** ~60 lines
**Expected Impact:**

| Test Case | Before | After Phase 0 | Speedup |
|-----------|--------|---------------|---------|
| Small text (125 chars) | ~10s | <2s | 5x |
| Medium text (1588 chars) | 65s (timeout!) | <10s | 7x |
| API response time | ~10s | <1s | 10x |

**Success Criteria:**
- ✅ All test files process successfully (no timeouts)
- ✅ Frontend shows clear errors if timeout still occurs
- ✅ Model loading happens once per server startup
- ✅ CI/CD passes

---

## Phase 1: Data-Driven Profiling (Day 3)

**Goal:** Use REAL data to identify ACTUAL bottlenecks (no assumptions!)

### Profiling Strategy: KISS Approach

**Use existing Python profiling tools** (no custom infrastructure!)

```bash
#!/bin/bash
# profile_real_data.sh - Simple profiling script

PROFILE_DIR="results/profiling/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROFILE_DIR"

echo "=== Profiling with Real Data ==="
echo "Output directory: $PROFILE_DIR"
echo ""

# Test 1: Small text
echo "Test 1: Small text (clinical_case_001.json)"
python -m cProfile -o "$PROFILE_DIR/small_text.prof" \
  -m phentrieve text process \
  tests/data/de/phentrieve/annotations/clinical_case_001.json \
  > "$PROFILE_DIR/small_text_output.txt" 2>&1

time phentrieve text process \
  tests/data/de/phentrieve/annotations/clinical_case_001.json \
  > /dev/null 2>&1

# Test 2: Medium text
echo ""
echo "Test 2: Medium text (GeneReviews_NBK1379.json)"
python -m cProfile -o "$PROFILE_DIR/medium_text.prof" \
  -m phentrieve text process \
  tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK1379.json \
  > "$PROFILE_DIR/medium_text_output.txt" 2>&1

time phentrieve text process \
  tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK1379.json \
  > /dev/null 2>&1

# Test 3: Find more test files and profile largest one
echo ""
echo "Test 3: Finding largest test file..."
LARGEST_FILE=$(find tests/data -name "*.json" -type f -exec wc -c {} + | sort -rn | head -1 | awk '{print $2}')
echo "Largest file: $LARGEST_FILE"

if [ -n "$LARGEST_FILE" ]; then
  python -m cProfile -o "$PROFILE_DIR/large_text.prof" \
    -m phentrieve text process "$LARGEST_FILE" \
    > "$PROFILE_DIR/large_text_output.txt" 2>&1

  time phentrieve text process "$LARGEST_FILE" > /dev/null 2>&1
fi

# Analyze profiles
echo ""
echo "=== Generating Analysis Reports ==="

for prof_file in "$PROFILE_DIR"/*.prof; do
  base_name=$(basename "$prof_file" .prof)
  echo "Analyzing: $base_name"

  python -m pstats "$prof_file" > "$PROFILE_DIR/${base_name}_analysis.txt" << EOF
sort cumtime
stats 30
quit
EOF
done

echo ""
echo "=== Profiling Complete ==="
echo "Results saved to: $PROFILE_DIR"
echo ""
echo "Review top functions by cumulative time:"
echo "  cat $PROFILE_DIR/*_analysis.txt"
```

### Analysis Checklist

```bash
# Run profiling
bash scripts/profile_real_data.sh

# Analyze results manually
cat results/profiling/*/medium_text_analysis.txt

# Look for:
# 1. Functions with high cumtime (> 10% of total)
# 2. Unexpected bottlenecks (e.g., ChromaDB still slow after batching?)
# 3. Memory issues (use memory_profiler if needed)

# Document findings
vim plan/PROFILING-RESULTS.md
```

### Documentation Template

**File:** `plan/PROFILING-RESULTS.md`

```markdown
# Profiling Results

**Date:** 2025-01-XX
**Branch:** main (after Phase 0 fixes)

## Test Environment

- **Hardware:** [CPU, RAM]
- **Python Version:** 3.11
- **ChromaDB Version:** [version]
- **Test Files:**
  - Small: clinical_case_001.json (125 chars)
  - Medium: GeneReviews_NBK1379.json (1588 chars)
  - Large: [file] ([size] chars)

## Results

### Small Text (clinical_case_001.json)

**Total Time:** 1.23s

**Top Functions by Cumulative Time:**

| Function | Cumtime (s) | % Total | Notes |
|----------|-------------|---------|-------|
| text_pipeline.process | 0.82 | 67% | Main processing |
| ChromaDB query (cached) | 0.31 | 25% | Single chunk |
| Assertion detection | 0.08 | 7% | spaCy processing |

**Analysis:** ✅ Performance acceptable. No further optimization needed.

### Medium Text (GeneReviews_NBK1379.json)

**Total Time:** 8.45s

**Top Functions by Cumulative Time:**

| Function | Cumtime (s) | % Total | Notes |
|----------|-------------|---------|-------|
| ChromaDB batch query | 2.10 | 25% | 28 chunks batched |
| Embedding generation | 3.85 | 46% | Sliding window chunking |
| Cross-encoder reranking | 2.12 | 25% | Optional, if enabled |
| Other | 0.38 | 4% | Assertion, aggregation |

**Analysis:**
- ⚠️ Embedding generation is bottleneck (46%)
- Investigate: Is parallelization needed?
- Measure: How many embeddings generated?

### Large Text ([file])

**Total Time:** [time]

[Similar analysis...]

## Prioritized Optimization Opportunities

Based on measured data, prioritize:

1. **[Issue 1]:** [Description] - [% of total time] - **Priority: High/Medium/Low**
2. **[Issue 2]:** [Description] - [% of total time] - **Priority: High/Medium/Low**

## Recommendations

- ✅ **Implement:** [Optimization X] - Expected impact: [speedup]
- ⏸️ **Defer:** [Optimization Y] - Not a bottleneck (<5% of time)
- ❌ **Skip:** [Optimization Z] - Premature optimization
```

### Deliverables

- [ ] Create `scripts/profile_real_data.sh`
- [ ] Run profiling on all test files
- [ ] Document findings in `plan/PROFILING-RESULTS.md`
- [ ] Identify top 3 bottlenecks with measured impact
- [ ] Create prioritized optimization backlog

---

## Phase 2: Core Optimizations (Week 2)

**Goal:** Implement ONLY optimizations validated by profiling data

**Note:** These optimizations are CONDITIONAL - only implement if profiling shows they're needed!

### Optimization A: Parallel Embedding Generation

**Condition:** ONLY if profiling shows embedding generation > 30% of total time

**Problem:** Large documents generate 100+ embeddings sequentially

**Solution:**

```python
# File: phentrieve/text_processing/chunkers.py

class SlidingWindowSemanticSplitter(TextChunker):
    """..."""

    def __init__(
        self,
        language: str = "en",
        model: "SentenceTransformer | None" = None,
        # ... existing params ...
        # NEW: Parallel embedding parameters
        use_parallel_embeddings: bool = False,
        parallel_batch_size: int = 32,
        parallel_workers: int | None = None,
        parallel_threshold: int = 100,
        **kwargs,
    ):
        """
        Initialize sliding window splitter.

        Args:
            use_parallel_embeddings: Enable parallel embedding generation
            parallel_batch_size: Batch size for encoding
            parallel_workers: Number of workers (None = auto)
            parallel_threshold: Use parallel if embeddings > threshold
        """
        super().__init__(language=language, **kwargs)
        # ... existing initialization ...

        self.use_parallel_embeddings = use_parallel_embeddings
        self.parallel_batch_size = parallel_batch_size
        self.parallel_workers = parallel_workers
        self.parallel_threshold = parallel_threshold

    def _generate_embeddings(self, window_texts: list[str]) -> np.ndarray:
        """
        Generate embeddings with optional parallelization.

        Uses sentence-transformers' built-in parallelization when beneficial.
        """
        # Decide: parallel or sequential?
        use_parallel = (
            self.use_parallel_embeddings and
            len(window_texts) >= self.parallel_threshold
        )

        if use_parallel:
            from multiprocessing import cpu_count
            workers = self.parallel_workers or max(1, cpu_count() - 1)

            logger.debug(
                f"Parallel embedding: {len(window_texts)} texts, "
                f"{workers} workers"
            )

            embeddings = self.model.encode_multi_process(
                window_texts,
                pool_size=workers,
                batch_size=self.parallel_batch_size,
                show_progress_bar=False,
            )
        else:
            logger.debug(f"Sequential embedding: {len(window_texts)} texts")
            embeddings = self.model.encode(
                window_texts,
                batch_size=self.parallel_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        return embeddings

    def _split_one_segment_by_sliding_window(
        self, current_text_segment: str
    ) -> list[str]:
        """..."""
        # ... existing code up to embedding generation ...

        # REPLACE: window_embeddings = self.model.encode(window_texts, ...)
        # WITH:
        window_embeddings = self._generate_embeddings(window_texts)

        # ... rest of method unchanged ...
```

**Configuration:**

```yaml
# phentrieve.yaml
chunking:
  performance:
    parallel_embeddings: true
    parallel_batch_size: 32
    parallel_threshold: 100  # Use parallel if > 100 embeddings
```

**Expected Impact:** 2-3x speedup for embedding generation (if it's a bottleneck)

**Validation:**
```bash
# Profile before/after
python -m cProfile -o before_parallel.prof -m phentrieve text process large_file.json
# Enable parallel in config
python -m cProfile -o after_parallel.prof -m phentrieve text process large_file.json

# Compare
python scripts/compare_profiles.py before_parallel.prof after_parallel.prof
```

### Optimization B: Streaming Response (UX Improvement)

**Goal:** Provide progress updates to frontend (prevents user frustration)

**Implementation:**

```python
# File: api/routers/text_processing_router.py

from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json

@router.post("/process-stream")
async def process_text_with_progress(request: TextProcessingRequest):
    """
    Process text with Server-Sent Events (SSE) progress updates.

    Frontend can display real-time progress instead of waiting blindly.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events during processing."""
        try:
            # Event 1: Started
            yield _sse_event({
                'status': 'started',
                'progress': 0,
                'message': 'Initiating text processing...'
            })

            # Event 2: Language detection
            yield _sse_event({
                'status': 'language_detection',
                'progress': 10,
                'message': 'Detecting language...'
            })

            actual_language = request.language or "en"
            # ... language detection logic ...

            # Event 3: Model loading (should be instant due to caching!)
            yield _sse_event({
                'status': 'loading_models',
                'progress': 20,
                'message': 'Loading models (cached)...'
            })

            # Get cached models (Phase 0 fix!)
            retrieval_sbert_model = await get_sbert_model_dependency(...)
            retriever = await get_dense_retriever_dependency(...)

            # Event 4: Text chunking
            yield _sse_event({
                'status': 'chunking',
                'progress': 40,
                'message': 'Analyzing text structure...'
            })

            text_pipeline = TextProcessingPipeline(...)
            processed_chunks_list = await run_in_threadpool(
                text_pipeline.process,
                request.text_content
            )
            num_chunks = len(processed_chunks_list)

            # Event 5: HPO extraction
            yield _sse_event({
                'status': 'extracting_hpo',
                'progress': 60,
                'message': f'Extracting HPO terms from {num_chunks} chunks...'
            })

            # ... extraction logic ...

            # Event 6: Aggregation
            yield _sse_event({
                'status': 'aggregating',
                'progress': 90,
                'message': 'Aggregating results...'
            })

            # ... aggregation logic ...

            # Event 7: Complete
            result = TextProcessingResponseAPI(...)
            yield _sse_event({
                'status': 'completed',
                'progress': 100,
                'data': result.dict()
            })

        except asyncio.TimeoutError:
            yield _sse_event({
                'status': 'error',
                'error': 'Processing timed out',
                'message': 'Text is too large. Please reduce length.'
            })
        except Exception as e:
            yield _sse_event({
                'status': 'error',
                'error': str(e),
                'message': 'An error occurred during processing.'
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


def _sse_event(data: dict) -> str:
    """Format data as Server-Sent Event."""
    return f"data: {json.dumps(data)}\n\n"
```

**Frontend Update:**

```javascript
// File: frontend/src/services/textProcessingApi.js

export async function processTextWithProgress(textContent, onProgress, options = {}) {
  const url = '/api/v1/text/process-stream';

  return new Promise((resolve, reject) => {
    const eventSource = new EventSource(url, {
      method: 'POST',
      body: JSON.stringify({
        text_content: textContent,
        ...options
      })
    });

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.status) {
        case 'started':
        case 'language_detection':
        case 'loading_models':
        case 'chunking':
        case 'extracting_hpo':
        case 'aggregating':
          // Update progress bar
          onProgress({
            progress: data.progress,
            message: data.message,
            status: data.status
          });
          break;

        case 'completed':
          eventSource.close();
          resolve(data.data);
          break;

        case 'error':
          eventSource.close();
          reject(new Error(data.message || data.error));
          break;
      }
    };

    eventSource.onerror = (error) => {
      eventSource.close();
      reject(new Error('Verbindung zum Server verloren'));
    };
  });
}

// Vue component usage:
export default {
  data() {
    return {
      processing: false,
      progress: 0,
      statusMessage: '',
    };
  },
  methods: {
    async processText() {
      this.processing = true;

      try {
        const result = await processTextWithProgress(
          this.inputText,
          ({ progress, message }) => {
            this.progress = progress;
            this.statusMessage = message;
          }
        );

        this.showResults(result);
      } catch (error) {
        this.showError(error.message);
      } finally {
        this.processing = false;
      }
    }
  }
};
```

**Expected Impact:**
- Better UX (users see progress)
- Reduced perceived wait time
- Clear feedback during long operations

### Optimization C: Memory-Efficient Data Loading

**Condition:** ONLY if processing very large corpora (>1000 documents)

**Implementation:**

```python
# File: phentrieve/utils/lazy_loading.py (NEW)

from typing import Iterator
from pathlib import Path
import json

def iter_jsonl_documents(
    file_path: Path,
    batch_size: int = 100,
) -> Iterator[list[dict]]:
    """
    Iterate over JSONL file in batches (memory-efficient).

    For processing large document collections without loading
    everything into memory.

    Args:
        file_path: Path to JSONL file
        batch_size: Number of documents per batch

    Yields:
        Batches of document dictionaries
    """
    batch = []

    with open(file_path) as f:
        for line in f:
            try:
                doc = json.loads(line)
                batch.append(doc)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON: {e}")

    if batch:
        yield batch


# Usage in batch processing:
from phentrieve.utils.lazy_loading import iter_jsonl_documents

for batch in iter_jsonl_documents("large_corpus.jsonl", batch_size=100):
    # Process batch...
    results = process_documents_batch(batch)
```

**Expected Impact:** Process unlimited corpus size with constant memory usage

---

## Phase 3: Infrastructure Improvements (Week 3-4)

**Goal:** Build maintainable infrastructure for ongoing optimization

### Tool 1: Simple Performance Regression Tests

```python
# File: tests_new/performance/test_performance_regression.py

import pytest
import time
from pathlib import Path

# Performance baseline (updated after each optimization)
PERFORMANCE_BASELINES = {
    "small_text": 2.0,   # clinical_case_001.json should be < 2s
    "medium_text": 10.0,  # GeneReviews_NBK1379.json should be < 10s
    "api_cached": 1.0,    # API with cached models should be < 1s
}


def test_small_text_performance():
    """Ensure small texts process quickly."""
    from phentrieve.text_processing import process_text

    test_file = Path("tests/data/de/phentrieve/annotations/clinical_case_001.json")

    # Read test file
    import json
    with open(test_file) as f:
        data = json.load(f)

    # Measure processing time
    start = time.time()
    result = process_text(
        text=data["full_text"],
        language=data["language"],
    )
    elapsed = time.time() - start

    # Assert performance baseline
    assert elapsed < PERFORMANCE_BASELINES["small_text"], (
        f"Small text processing too slow: {elapsed:.2f}s "
        f"(baseline: {PERFORMANCE_BASELINES['small_text']}s)"
    )


def test_medium_text_performance():
    """Ensure medium texts process within timeout."""
    from phentrieve.text_processing import process_text

    test_file = Path("tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK1379.json")

    import json
    with open(test_file) as f:
        data = json.load(f)

    start = time.time()
    result = process_text(
        text=data["full_text"],
        language=data["language"],
    )
    elapsed = time.time() - start

    assert elapsed < PERFORMANCE_BASELINES["medium_text"], (
        f"Medium text processing too slow: {elapsed:.2f}s "
        f"(baseline: {PERFORMANCE_BASELINES['medium_text']}s)"
    )


@pytest.mark.asyncio
async def test_api_model_caching():
    """Ensure API uses cached models (not reloading)."""
    from api.dependencies import get_sbert_model_dependency
    from phentrieve.config import DEFAULT_MODEL

    # First call (may load model)
    start1 = time.time()
    model1 = await get_sbert_model_dependency(
        model_name_requested=DEFAULT_MODEL,
    )
    elapsed1 = time.time() - start1

    # Second call (should be cached!)
    start2 = time.time()
    model2 = await get_sbert_model_dependency(
        model_name_requested=DEFAULT_MODEL,
    )
    elapsed2 = time.time() - start2

    # Verify caching
    assert model1 is model2, "Models should be same instance (cached)"
    assert elapsed2 < PERFORMANCE_BASELINES["api_cached"], (
        f"Cached model retrieval too slow: {elapsed2:.2f}s "
        f"(should be <{PERFORMANCE_BASELINES['api_cached']}s)"
    )
```

**Run in CI/CD:**

```yaml
# .github/workflows/ci.yml

  performance-regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --all-extras

      - name: Run performance regression tests
        run: |
          pytest tests_new/performance/ -v --tb=short
```

### Tool 2: Profiling Comparison Script

```python
#!/usr/bin/env python3
# scripts/compare_profiles.py

import pstats
import sys
from pathlib import Path

def compare_profiles(profile1_path: str, profile2_path: str):
    """Compare two cProfile outputs."""
    print("=" * 80)
    print("Profile Comparison")
    print("=" * 80)

    # Load profiles
    stats1 = pstats.Stats(profile1_path)
    stats2 = pstats.Stats(profile2_path)

    # Get total times
    total1 = sum(stat[2] for stat in stats1.stats.values())
    total2 = sum(stat[2] for stat in stats2.stats.values())

    # Calculate improvement
    speedup = total1 / total2 if total2 > 0 else 0
    improvement_pct = ((total1 - total2) / total1 * 100) if total1 > 0 else 0

    print(f"\nProfile 1: {profile1_path}")
    print(f"  Total time: {total1:.4f}s")
    print(f"\nProfile 2: {profile2_path}")
    print(f"  Total time: {total2:.4f}s")
    print(f"\n{'SPEEDUP' if speedup > 1 else 'SLOWDOWN'}: {speedup:.2f}x")
    print(f"Improvement: {improvement_pct:+.1f}%")
    print("\n" + "=" * 80)

    # Show top functions
    print("\nTop 10 Functions by Cumulative Time (Profile 1):")
    print("-" * 80)
    stats1.sort_stats('cumulative')
    stats1.print_stats(10)

    print("\nTop 10 Functions by Cumulative Time (Profile 2):")
    print("-" * 80)
    stats2.sort_stats('cumulative')
    stats2.print_stats(10)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compare_profiles.py <profile1.prof> <profile2.prof>")
        sys.exit(1)

    compare_profiles(sys.argv[1], sys.argv[2])
```

**Usage:**
```bash
python scripts/compare_profiles.py \
  results/profiling/before.prof \
  results/profiling/after.prof
```

---

## Phase 4: Future Enhancements (Week 5+)

**These are DEFERRED until Phase 0-2 are complete and validated**

### Advanced Chunking Optimizations

From `CHUNKING-OPTIMIZATION-PLAN.md`:
- Adaptive window sizing
- Medical abbreviation handling
- Annotation-based chunking

**Decision Point:** Only implement if:
1. Current chunking quality is insufficient (F1 < 0.65)
2. Profiling shows chunking is a bottleneck (> 20% of time)
3. Phase 0-2 optimizations are complete

### Benchmarking Framework

From `CHUNKING-OPTIMIZATION-PLAN.md`:
- Systematic strategy comparison
- Span-based metrics
- Automated reporting

**Decision Point:** Only implement if:
1. Need to compare multiple chunking strategies systematically
2. Have sufficient annotated test data (>50 documents)
3. Phase 0-2 optimizations are complete

---

## Validation & Testing Strategy

### Test Files (Real Data)

```bash
# Organize test files by size
tests/data/
├── small/       # <500 chars, should be <2s
│   └── clinical_case_001.json (125 chars)
├── medium/      # 500-2000 chars, should be <10s
│   └── GeneReviews_NBK1379.json (1588 chars)
└── large/       # >2000 chars, should be <60s
    └── [find largest test file]
```

### Validation Checklist

**After Phase 0:**
- [ ] Small texts process in <2s
- [ ] Medium texts process in <10s (no timeout!)
- [ ] Frontend shows clear error messages
- [ ] Models loaded only once per server start
- [ ] All existing tests pass

**After Phase 1:**
- [ ] Profiling results documented
- [ ] Top 3 bottlenecks identified
- [ ] Optimization priorities set (data-driven!)

**After Phase 2:**
- [ ] Implemented optimizations show measured improvement
- [ ] Performance regression tests pass
- [ ] No regressions in functionality

### Performance Benchmarks

```bash
#!/bin/bash
# scripts/validate_performance.sh

set -e

echo "=== Performance Validation Suite ==="
echo ""

# Test 1: Small text
echo "Test 1: Small text (clinical_case_001.json)"
time phentrieve text process \
  tests/data/de/phentrieve/annotations/clinical_case_001.json \
  > /tmp/test1_output.json
echo "✅ Expected: <2s"
echo ""

# Test 2: Medium text
echo "Test 2: Medium text (GeneReviews_NBK1379.json)"
time phentrieve text process \
  tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK1379.json \
  > /tmp/test2_output.json
echo "✅ Expected: <10s"
echo ""

# Test 3: API endpoint (should use cached models)
echo "Test 3: API endpoint with cached models"
curl -X POST http://localhost:8734/api/v1/text/process \
  -H "Content-Type: application/json" \
  -d '{"text_content": "Patient presents with ataxia and hypotonia."}' \
  -w "\nTime: %{time_total}s\n" \
  -o /tmp/test3_output.json
echo "✅ Expected: <1s"
echo ""

echo "=== All Tests Passed! ==="
```

---

## Anti-Patterns to Avoid

### ❌ What NOT to Do

1. **Premature Optimization**
   - ❌ Building parallelization before profiling
   - ✅ Fix critical bugs first, profile, THEN optimize

2. **Over-Engineering**
   - ❌ Custom profiling infrastructure when `cProfile` works
   - ✅ Use existing tools (KISS principle)

3. **Synthetic Workloads**
   - ❌ Creating test data instead of using real files
   - ✅ Use actual problem files: `clinical_case_001.json`, `GeneReviews_NBK1379.json`

4. **Assumption-Based Development**
   - ❌ "I think X is slow, let's optimize X"
   - ✅ "Profiling shows X is 45% of time, let's optimize X"

5. **Infrastructure Before Value**
   - ❌ Building benchmarking framework before fixing bugs
   - ✅ Fix production-breaking bugs first

6. **Complexity Creep**
   - ❌ "Let's build a custom ModelPool singleton with thread locks"
   - ✅ "Let's use the existing dependency injection that's already there"

### ✅ Best Practices Applied

1. **Fix Critical Bugs First** - Timeout is production-breaking
2. **Use Existing Code** - `api/dependencies.py` already has caching!
3. **Test with Real Data** - Use actual problematic files
4. **KISS Principle** - Simplest solution that works
5. **Measure Everything** - Before/after timing with real cases
6. **Incremental Delivery** - Each phase delivers value independently
7. **Data-Driven Decisions** - Profile before optimizing

---

## Success Metrics

### Phase 0 Success Criteria

| Metric | Before | Target | Status |
|--------|--------|--------|--------|
| Small text (125 chars) | ~10s | <2s | ⏳ |
| Medium text (1588 chars) | 65s (timeout!) | <10s | ⏳ |
| API cached response | ~10s | <1s | ⏳ |
| Frontend timeout errors | ❌ Confusing | ✅ Clear | ⏳ |

### Overall Success Criteria

| Metric | Baseline | Phase 0 | Phase 2 | Phase 3 |
|--------|----------|---------|---------|---------|
| Small text | 10s | <2s | <1s | <0.5s |
| Medium text | 65s (timeout!) | <10s | <5s | <3s |
| Large text | N/A (timeout!) | <60s | <30s | <15s |
| User satisfaction | ❌ | ✅ | ✅✅ | ✅✅✅ |

---

## Timeline Summary

```
Phase 0: Days 1-2   ████░░░░░░░░░░░░░░  Critical Fixes (Model caching, timeouts, batching)
Phase 1: Day 3      ░░░░████░░░░░░░░░░  Profiling (Real data, identify bottlenecks)
Phase 2: Week 2     ░░░░░░░░████████░░  Core Optimizations (Data-driven)
Phase 3: Week 3-4   ░░░░░░░░░░░░░░████  Infrastructure (Monitoring, tests)
Phase 4: Week 5+    ░░░░░░░░░░░░░░░░░░  Future (Deferred until needed)

Total: 1 week (critical fixes + profiling) + 2-3 weeks (optimizations)
```

---

## Appendix A: Quick Reference

### Commands

```bash
# Profile CLI with real data
python -m cProfile -o output.prof -m phentrieve text process input.json
python -m pstats output.prof

# Simple timing
time phentrieve text process input.json

# API testing
curl -X POST http://localhost:8734/api/v1/text/process \
  -H "Content-Type: application/json" \
  -d @input.json \
  -w "\nTime: %{time_total}s\n"

# Batch profiling
bash scripts/profile_real_data.sh

# Performance validation
bash scripts/validate_performance.sh

# Compare profiles
python scripts/compare_profiles.py before.prof after.prof
```

### Test Files

```
# Small (<500 chars)
tests/data/de/phentrieve/annotations/clinical_case_001.json

# Medium (500-2000 chars)
tests/data/en/phenobert/GeneReviews/annotations/GeneReviews_NBK1379.json

# Find more
find tests/data -name "*.json" -type f
```

---

## Appendix B: Related Plans

This master plan consolidates and supersedes:

1. ❌ ~~`TECHNICAL-OPTIMIZATIONS-PLAN.md`~~ - Over-engineered, premature
2. ❌ ~~`CRITICAL-PERFORMANCE-FIXES.md`~~ - Merged into Phase 0
3. ⏸️ `CHUNKING-OPTIMIZATION-PLAN.md` - Deferred to Phase 4 (algorithm improvements)

**Active Plans:**
- ✅ `PERFORMANCE-MASTER-PLAN.md` (THIS FILE) - Single source of truth

---

**END OF MASTER PLAN**

## Next Steps

1. **Review this plan** with team/stakeholders
2. **Start Phase 0, Day 1:** Model caching fix (2 hours)
3. **Continue Phase 0:** Timeout + batching (1-2 days)
4. **Profile with real data** (Day 3)
5. **Iterate based on data** (Weeks 2-4)
