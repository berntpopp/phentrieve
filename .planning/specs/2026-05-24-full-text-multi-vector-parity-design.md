# Full-Text Multi-Vector Parity Design

## Goal

Make standard full-text HPO extraction use HPO-level multi-vector aggregation
when it is connected to a multi-vector index, matching the aggregation semantics
already used by direct query.

## Source Analysis

This design implements Candidate A from
`.planning/analysis/2026-05-23-phentrieve-rag-prompting-literature-report.md`.
The report identifies a correctness gap: direct query uses
`DenseRetriever.query_multi_vector()` to aggregate label, synonym, and
definition component hits by HPO ID, but standard full-text extraction calls raw
`DenseRetriever.query_batch()` and can therefore treat multi-vector component
rows as independent term matches.

## Scope

In scope:

- Add batch multi-vector retrieval for lists of chunks.
- Use that retrieval path automatically in standard full-text extraction when
  the retriever is connected to a multi-vector index.
- Preserve current single-vector behavior.
- Preserve `precomputed_query_results` as the authoritative bypass path.
- Keep the public standard full-text API response shape unchanged.
- Add focused unit tests for retrieval shape, aggregation, fallback, and
  orchestrator routing.
- Keep adaptive rechunking on the same retrieval-mode normalized candidate path
  when it retrieves child chunks.

Out of scope:

- LLM backend changes.
- New CLI, API, or frontend options.
- Hybrid lexical/dense retrieval.
- Optional HPO-specific reranking.
- Evidence-aware document-level confidence formula changes.
- Prompt changes or mapping-payload enrichment.

## Target Architecture

Add a first-class multi-vector batch retrieval API to
`phentrieve/retrieval/dense_retriever.py` named
`DenseRetriever.query_batch_multi_vector(...)`.

The method will mirror `query_multi_vector(...)` semantics for a list of input
texts:

1. Return `[]` immediately for empty input.
2. Request enough raw component hits per chunk by using
   `n_results * MULTI_VECTOR_RESULT_MULTIPLIER`.
3. Query through the existing `query_batch(...)` path with that expanded
   `n_results` value so chunks are encoded and queried together, and so the
   effective over-fetch behavior remains equivalent to
   `query_multi_vector(...)`.
4. For each chunk, call `aggregate_multi_vector_results(...)` with the same
   aggregation strategy, component weights, custom formula, and minimum
   similarity semantics as direct query.
5. Truncate each aggregated chunk result to `n_results`.
6. Convert each aggregated list back to the Chroma-style result shape consumed
   by `process_chunk_matches(...)`.

Add a small retrieval-selection helper named `query_chunk_candidates(...)` in
`phentrieve/retrieval/utils.py`, so the initial full-text retrieval path and
adaptive child-chunk retrieval use the same logic. Retrieval selection will use
this order:

1. If `precomputed_query_results` is provided, validate its length and use it
   unchanged. This branch remains inside `orchestrate_hpo_extraction(...)`.
2. Otherwise, if guarded index-type detection returns `"multi_vector"` and the
   retriever exposes a callable `query_batch_multi_vector(...)`, call the new
   retriever method.
3. Otherwise, call the existing `retriever.query_batch(...)`.

This keeps the standard backend's downstream processing stable while replacing
multi-vector component-level rows with aggregated HPO-level rows before chunk
match processing.

## Components

### Dense Retriever

`DenseRetriever.query_batch_multi_vector(...)` will own the retrieval-mode
specific work:

- detect and warn when called on a non-multi-vector index, matching the current
  defensive behavior of `query_multi_vector(...)`;
- batch query raw component vectors;
- aggregate each chunk independently by HPO ID;
- convert each chunk into Chroma-style output;
- include `similarities` so existing chunk threshold filtering works unchanged;
- return one empty Chroma-style result per input text if querying or aggregation
  fails.

The existing `query_multi_vector(...)` remains behaviorally unchanged for this
milestone. The implementation may extract a private shared helper only if the
focused tests prove the single-query output is unchanged.

### Retrieval Utilities

`phentrieve/retrieval/utils.py` already provides
`convert_multi_vector_to_chromadb_format(...)`. Extend it with an
`include_similarities: bool = False` keyword. The default must preserve existing
callers. When `include_similarities=True`, add `similarities` from each
aggregated result's `similarity` value so `process_chunk_matches(...)` does not
see every converted multi-vector hit as score `0.0`.

Add the shared retrieval-selection helper here as well. It should catch missing
or failing `detect_index_type()` defensively, log the fallback, and use
`query_batch(...)` unless it can positively confirm multi-vector retrieval.

### Full-Text Orchestrator

`orchestrate_hpo_extraction(...)` remains the public boundary for standard
full-text extraction. Its new responsibility is limited to selecting the
correct batch retrieval method through the shared retrieval-selection helper
before calling the existing helpers:

- `process_chunk_matches(...)`
- `load_term_details(...)`
- `build_evidence_map(...)`
- `aggregate_and_rank(...)`

No response adaptation logic belongs in the orchestrator.

### Adaptive Rechunker

`phentrieve/retrieval/adaptive_rechunker.py` currently retrieves accepted child
chunks with `retriever.query_batch(...)`. That call must move to the same shared
retrieval-selection helper used by the orchestrator. Otherwise, an adaptive run
would start with aggregated multi-vector parent results but reintroduce raw
component-level child results after subdivision.

The existing cost-model invariant remains: adaptive rechunking may perform at
most one retrieval call per recursion level. The retrieval method may be
`query_batch(...)` or `query_batch_multi_vector(...)` depending on index type,
but the number of retrieval calls must stay bounded by `max_depth`.

### Standard Backend

`run_standard_backend(...)` already initializes the retriever with
`multi_vector=DEFAULT_MULTI_VECTOR`. No new public argument is required for this
milestone. The backend should benefit from the orchestrator's automatic
retrieval selection without changing its request or response contract.

## Data Flow

The standard full-text flow remains:

1. `TextProcessingPipeline.process(...)` produces chunks and assertion status.
2. `orchestrate_hpo_extraction(...)` retrieves per-chunk HPO candidates.
3. `process_chunk_matches(...)` applies chunk thresholding and assertion
   propagation.
4. `build_evidence_map(...)` groups evidence by HPO ID.
5. `aggregate_and_rank(...)` produces document-level HPO terms.
6. `adapt_standard_response(...)` exposes the stable API shape.

For single-vector indexes, step 2 still returns raw ChromaDB query rows.

For multi-vector indexes, step 2 returns Chroma-style rows that already
represent aggregated HPO terms for each chunk. Downstream code sees the same
shape, but duplicate component hits for one HPO ID no longer appear as separate
chunk matches.

When adaptive rechunking is enabled, child chunks follow the same rule: a
multi-vector retriever returns aggregated HPO-level child candidates, while a
single-vector retriever returns raw ChromaDB child candidates.

`OrchestrationResult.raw_query_results` should be understood as retrieval-mode
normalized per-chunk results: raw Chroma hits for single-vector indexes and
aggregated Chroma-style HPO hits for multi-vector indexes.

## Error Handling

- Empty text lists return empty result lists without querying.
- If index type detection fails, log a warning and use `query_batch(...)`.
- If index type detection is unavailable, use `query_batch(...)` without
  treating dynamic mock attributes as proof of multi-vector support.
- If `query_batch_multi_vector(...)` is called on a non-multi-vector index, log
  a warning and still execute using the same aggregation path, matching
  `query_multi_vector(...)`.
- If collection querying, result conversion, or aggregation raises an exception,
  log the error and return one empty Chroma-style result per input text.
- If a supplied aggregation strategy is invalid, log the error and return empty
  per-text results for that batch rather than failing the whole full-text
  request.
- If `precomputed_query_results` length differs from `text_chunks`, keep the
  existing `ValueError`.

## Compatibility

Single-vector behavior must remain unchanged:

- `orchestrate_hpo_extraction(...)` calls `retriever.query_batch(...)`.
- Chunk result ordering and filtering stay as currently characterized.
- Document-level aggregation still uses average score for confidence and
  ranking.

Multi-vector behavior changes intentionally:

- Per-chunk results are aggregated by HPO ID before chunk matching.
- The similarity used by chunk filtering is the aggregated multi-vector score.
- Component metadata such as `component_scores`, `matched_component`, and
  `matched_text` may be preserved in metadata for debugging and future
  enrichment, but no downstream consumer should depend on it for this milestone.

Adaptive rechunking remains enabled through its current configuration, but its
quality assessment receives retrieval-mode normalized results. Tests must
characterize the routing and result shape because adaptive rechunking reads
`raw_query_results` scores directly.

Adaptive child retrieval must also receive retrieval-mode normalized results.
Existing call-count tests should continue to assert the number of retrieval
calls, while new tests assert the correct retrieval method for multi-vector
indexes.

## Testing Strategy

Add or update focused unit tests before implementation.

Retrieval tests under `tests/unit/retrieval/`:

- `query_batch_multi_vector([])` returns `[]` and does not query the collection.
- Two component hits for one HPO ID in one chunk aggregate to one HPO result.
- Two chunks are aggregated independently and preserve input order.
- Converted results include `ids`, `metadatas`, `documents`, `distances`, and
  `similarities` in the same nested-list style as `query_batch(...)`.
- Query or aggregation failure returns one empty result per input text.
- `convert_multi_vector_to_chromadb_format(..., include_similarities=True)`
  includes similarities while the default call shape remains backward
  compatible.

Orchestrator tests under `tests/unit/text_processing/`:

- A multi-vector retriever uses `query_batch_multi_vector(...)`.
- A single-vector retriever uses `query_batch(...)`.
- `precomputed_query_results` bypasses both retrieval methods.
- Standard full-text chunk processing no longer emits duplicate matches for a
  single HPO ID when the retriever is multi-vector and raw component hits
  include label and synonym rows for that HPO ID.

Adaptive rechunking tests under `tests/unit/retrieval/adaptive_rechunker/`:

- A multi-vector retriever uses `query_batch_multi_vector(...)` for child
  chunks.
- A single-vector retriever keeps using `query_batch(...)` for child chunks.
- Existing `max_depth` call-count invariants remain true with either retrieval
  method.

Recommended focused commands:

```bash
uv run pytest tests/unit/retrieval/test_dense_retriever_real.py tests/unit/retrieval/test_retrieval_utils.py -n 0
uv run pytest tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py tests/unit/text_processing/test_orchestrate_with_precomputed.py -n 0
uv run pytest tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py -n 0
uv run pytest tests/unit/text_processing/test_full_text_service.py -n 0
```

Required final checks before claiming implementation completion:

```bash
make check
make typecheck-fast
make test
```

## Benchmark Gate

The implementation should record comparable before/after evidence for standard
full-text behavior. At minimum, the implementation plan should include a
focused local fixture or benchmark command that demonstrates the duplicate
component-hit bug is fixed.

The implementation plan should list these broader benchmark commands as
optional follow-up verification when the required data and runtime budget are
available:

- GeneReviews full-text strict, soft, and partial F1.
- Runtime per document.
- Failed document count.
- 570 German direct retrieval benchmark metrics, which must not regress because
  direct query semantics should stay unchanged.

## Success Criteria

- Standard full-text extraction uses HPO-level multi-vector aggregation when
  connected to a multi-vector index.
- Direct query and standard full-text use equivalent multi-vector aggregation
  semantics for the same strategy.
- Single-vector standard full-text behavior remains unchanged.
- `precomputed_query_results` still bypasses retrieval.
- The public standard backend response shape does not change.
- Focused unit tests cover retrieval aggregation, orchestrator routing, and the
  duplicate component-hit regression.
- Required repository checks pass during implementation.
