# Spec: Adaptive Re-Chunking for Poor-Quality Retrieval

- **Date**: 2026-04-25
- **Status**: Draft (awaiting user review)
- **Issues addressed**: #148 (adaptive re-chunking)
- **Companion spec**: `2026-04-25-cli-profiles-default-resolution-spec.md`
- **Depends on**: Spec A's profile system for full configurability (CLI flags also work standalone).

## Goal

Add an opt-in mechanism that detects per-chunk retrieval quality and, when poor, subdivides the chunk into smaller sentence-bounded sub-chunks, re-queries them, and merges results. Improves recall on multi-concept clinical sentences without affecting users who don't enable it.

## Non-goals

- LLM-as-judge confidence in the trigger loop (expensive, latency-hostile, deferred indefinitely).
- Ontology-coherence as an active trigger signal in v1 (reserved YAML knob shipped inert; real implementation deferred — see Future work).
- Embedding cache for sub-chunk amortization across runs (deferred — see Future work).
- Adaptive rechunking on `phentrieve text interactive` (deferred — interactive mode favors low-latency feedback loops).
- Token-level sliding-window subdivision. Sentence-bounded subdivision is the only strategy in v1.

## Architecture

A new module `phentrieve/retrieval/adaptive_rechunker.py` owns the feature. It plugs in at one well-defined seam: inside `run_standard_backend()` in `phentrieve/text_processing/full_text_service.py`, between the existing `orchestrate_hpo_extraction()` call and `adapt_standard_response()`.

### Required upstream change to `orchestrate_hpo_extraction`

The current `orchestrate_hpo_extraction` (`hpo_extraction_orchestrator.py:62-67`) calls `retriever.query_batch(text_chunks, ...)` internally, then **filters matches by `chunk_retrieval_threshold` before storing them in `chunk_results`** (line 89). This means `chunk_results` contains only matches above threshold — by construction, `top_1`/`top_2`/margin cannot be computed from `chunk_results` for chunks where retrieval was genuinely poor (the matches that would tell us so were filtered out).

Two changes are required:

1. **`orchestrate_hpo_extraction` returns a `dataclass` instead of a tuple** so we can add fields without breaking unpack-on-assignment at every call site. Specifically, return a `frozen` dataclass `OrchestrationResult` with `aggregated_results`, `chunk_results`, and `raw_query_results` fields. The dataclass implements `__iter__` / `__getitem__` returning `(aggregated_results, chunk_results)` for backward compat with **existing tuple-unpacking call sites that we explicitly choose to leave alone**, and exposes `raw_query_results` as an attribute for the new caller (`run_adaptive_rechunking`).

   **Existing 2-tuple-unpacking call sites that must be audited and confirmed compatible** (verified against the tree as of 2026-04-25):
   - `phentrieve/text_processing/full_text_service.py:599` — wrapper around the new seam; this one we'll update directly to use named-field access (`.aggregated_results`, `.chunk_results`, `.raw_query_results`).
   - `phentrieve/cli/text_interactive.py:540` — leave as 2-tuple unpack via dataclass `__iter__`.
   - `phentrieve/benchmark/extraction_benchmark.py:182` — leave as 2-tuple unpack.
   - `phentrieve/evaluation/full_text_runner.py:112` — leave as 2-tuple unpack.
   - `phentrieve/llm/provider.py:1772` — leave as 2-tuple unpack.

   The dataclass approach (`__iter__` returning the legacy 2-tuple) means none of these need to change, while the new field is reachable for new callers. Test coverage in B5 explicitly asserts both unpack styles work.

2. **`orchestrate_hpo_extraction` accepts an optional `precomputed_query_results: list[dict] | None = None` parameter**. When provided, it skips the `query_batch` call and uses the supplied raw results directly. This lets the rechunker re-aggregate over a curated mix of original-chunk results and child-chunk results without re-querying chunks it already has scores for.

The dataclass return type is the right call even setting aside adaptive rechunking — it's a small, principled change that future-proofs the function against further field additions. Spec A's CHANGELOG explicitly mentions this contract change so any third-party caller is notified.

### Run-time flow

```python
# full_text_service.py:run_standard_backend — modified
processed_chunks = text_pipeline.process(text, include_positions=include_positions)
text_chunks = [c["text"] for c in processed_chunks]
assertion_statuses = [_normalize_status(c.get("status")) for c in processed_chunks]

result = orchestrate_hpo_extraction(
    text_chunks=text_chunks, retriever=retriever, ...,
    assertion_statuses=assertion_statuses,
    ...
)
aggregated_results = result.aggregated_results
chunk_results = result.chunk_results
raw_query_results = result.raw_query_results

adaptive_meta: dict[str, Any] | None = None
adaptive_cfg = kwargs.pop("adaptive_rechunking", None)
if adaptive_cfg and adaptive_cfg.enabled:
    rechunk_result = run_adaptive_rechunking(
        processed_chunks=processed_chunks,
        chunk_results=chunk_results,
        raw_query_results=raw_query_results,        # <- raw scores, not filtered
        retriever=retriever,
        language=language,
        config=adaptive_cfg,
        num_results_per_chunk=...,
        chunk_retrieval_threshold=...,
        min_confidence_for_aggregated=...,
        include_details=...,
    )
    processed_chunks = rechunk_result.processed_chunks
    aggregated_results = rechunk_result.aggregated_results
    chunk_results = rechunk_result.chunk_results
    adaptive_meta = rechunk_result.meta   # dict with trigger_count, etc.

return adapt_standard_response(
    processed_chunks,
    (aggregated_results, chunk_results),
    extra_meta={"adaptive_rechunking": adaptive_meta} if adaptive_meta else None,
)
```

`run_adaptive_rechunking` returns a dataclass `AdaptiveRechunkingResult` with four fields: `processed_chunks`, `aggregated_results`, `chunk_results`, `meta` (the stats dict). `adapt_standard_response` gains a new optional `extra_meta: dict[str, Any] | None = None` parameter that, when non-None, is merged into the `meta` block of the response. Without this, `meta.adaptive_rechunking` cannot reach the API output.

Why this seam: at this point we have `processed_chunks` (positions / assertion / details), `chunk_results` (above-threshold matches per chunk), AND `raw_query_results` (full top-K from `query_batch`, including matches below `chunk_retrieval_threshold` that drive the trigger). The downstream `_adapt_processed_chunks` glues `processed_chunks` and `chunk_results` by index, so we must update both lists in lockstep when we add sub-chunks.

### The orchestration loop

1. **Detect** poor chunks by reading `raw_query_results[i]` (not `chunk_results[i]`!) and computing `top_1`, `top_2`, margin from the unfiltered top-K. Apply the score-and-margin gate.
2. **Subdivide** each poor chunk's text via direct `SentenceChunker(language).chunk([parent_text])` calls. Drop sub-chunks shorter than `min_chunk_chars`. If only one sub-chunk falls out (parent was a single sentence), skip — no useful subdivision.
3. **Batch-query the children** in a single `retriever.query_batch(child_texts, ...)` call. Capture `child_raw_results` (the full top-K, same shape as `raw_query_results`).
4. **Apply score-improvement gate** per parent: `best(child_top_1) >= parent_top_1 + score_improvement_gate`. Reverted parents drop their children from the new flat list and keep their original entry.
5. **Build a new combined raw_results dict**: kept original `raw_query_results` for non-subdivided / reverted chunks, plus accepted `child_raw_results` for surviving children. Indices renumbered to a flat sequence.
6. **Re-aggregate WITHOUT re-querying** by calling `orchestrate_hpo_extraction(text_chunks=new_flat_text_chunks, retriever=retriever, precomputed_query_results=combined_raw_results, ...)`. The function skips `query_batch` because `precomputed_query_results` is provided, and runs only its filtering + aggregation logic over the supplied raw data.
7. **Recurse** on children that flag as poor against the *new* `raw_query_results` returned by step 6, up to `max_depth=2`. Each recursion level performs **exactly one** new `query_batch` call (for the new children at that level).

### Cost model (corrected) — split invariants

There are two separable costs: **`query_batch` RPC count** (network round trips to ChromaDB) and **encoding work** (SentenceTransformer.encode on chunk text). They scale differently and must not be conflated.

**`query_batch` call count** — bounded tightly:
- 1 call for the initial extraction (always, regardless of `enabled`).
- 1 additional call per recursion level when chunks flag as poor at that level.
- At `max_depth=2`: worst case **3 calls** (initial + depth-1 children + depth-2 grandchildren).

This is the hard invariant. Made possible by `precomputed_query_results` — without it, every recursion level would re-query the whole flat list and the bound would be 5 calls.

**Encoding work** — bounded loosely:
- Each `query_batch` call encodes `len(input_texts)` strings via SentenceTransformer.
- The initial pass encodes `N` original chunks.
- Each recursion level encodes the children added at that level, which can fan out: `K` poor parents × `~3` sentence-window children = `~3K` strings encoded at depth 1.
- At `max_depth=2` worst case: roughly `N + 3·N + 3·(3·N) = 13N` strings encoded if every chunk recurses.

So **encoding work can exceed the query-call multiplier**. A 10-chunk document where every chunk flags through both levels could see ~13× encoding work despite only 3 `query_batch` calls.

In practice fan-out is limited by `min_chunk_chars`, the score-improvement gate (children that don't help are reverted before further recursion), and the rarity of a chunk that genuinely benefits from depth-2 recursion. Empirical typical: 1.2–1.5× wall time. Empirical worst case at `max_depth=2`: hard to predict tightly because it depends on encoder batch-throughput and chunk-size distribution.

The user-facing user-guide quotes the typical 1.2–1.5× range. The performance regression test (in Tests below) splits these into two assertions:

1. **Hard invariant**: `retriever.query_batch.call_count == 3` for a fixture where every chunk flags at every level. This is a contract guarantee.
2. **Wall-time sanity bound**: total wall time ≤ 5× the no-adaptive baseline. Loose bound, smoke-test only — designed to catch *regressions* (e.g. accidentally re-running the full pipeline) rather than to specify performance.

## Configuration

```python
@dataclass(frozen=True)
class AdaptiveRechunkingConfig:
    enabled: bool = False
    quality_threshold: float = 0.55
    margin_threshold: float = 0.03
    use_ontology_coherence: bool = False    # future, inert in v1
    max_depth: int = 2
    min_chunk_chars: int = 30
    max_sentences_per_subchunk: int = 3
    overlap_sentences: int = 1
    score_improvement_gate: float = 0.05
```

Built from CLI flags / `Profile.adaptive_rechunking` (Spec A) / YAML `extraction.adaptive_rechunking` block, in Spec A's precedence stack. The corresponding Pydantic block on `Profile`:

```python
class AdaptiveRechunkingProfileBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool | None = None
    quality_threshold: float | None = None
    margin_threshold: float | None = None
    use_ontology_coherence: bool | None = None
    max_depth: int | None = None
    min_chunk_chars: int | None = None
    max_sentences_per_subchunk: int | None = None
    overlap_sentences: int | None = None
    score_improvement_gate: float | None = None
```

YAML structure (additive under existing `extraction:` section introduced by Spec A):

```yaml
extraction:
  adaptive_rechunking:
    enabled: false                  # opt-in
    quality_threshold: 0.55         # encoder-calibrated for BioLORD
    margin_threshold: 0.03
    use_ontology_coherence: false   # reserved, inert in v1
    max_depth: 2
    min_chunk_chars: 30
    max_sentences_per_subchunk: 3
    overlap_sentences: 1
    score_improvement_gate: 0.05
```

Default thresholds are calibrated for BioLORD-class biomedical encoders. Users on other models should retune (documented in `docs/user-guide/adaptive-rechunking.md`).

## Quality assessment

Reads from **raw query results** (`raw_query_results[chunk_idx]`), not from the threshold-filtered `chunk_results`. The raw results contain the full top-K from `query_batch` (today `n_results * 3`, see `dense_retriever.py:300`), so `top_1` and `top_2` are present even when retrieval was so poor that nothing exceeded `chunk_retrieval_threshold`.

```python
@dataclass(frozen=True)
class ChunkQualitySignals:
    chunk_idx: int
    top_1: float | None
    top_2: float | None
    margin: float | None
    n_matches_above_threshold: int   # informational, not a trigger condition
    is_poor: bool
    reason: str   # "low_score" | "low_margin" | "no_matches" | "ok"

def assess_chunk_quality(
    raw_query_result: dict,            # one entry from raw_query_results
    chunk_retrieval_threshold: float,  # only used to populate n_matches_above_threshold
    config: AdaptiveRechunkingConfig,
) -> ChunkQualitySignals:
    """Reads similarities[0] from a single chunk's raw query_batch output."""
```

Trigger semantics:

```
is_poor = top_1 < quality_threshold
          AND (margin < margin_threshold OR top_2 is None)
```

Both score AND margin must be problematic for the chunk to flag. If `top_1 ≥ quality_threshold` we trust the result regardless of margin (a correct top match doesn't need to beat the runner-up by much in HPO). If margin is large but `top_1` is mediocre, we likely have a clear-but-weak match — also don't subdivide. The conjunction keeps the trigger precise, trading a slightly higher false-negative rate for a much lower false-positive rate (subdivision costs latency).

`top_2 is None` (only one raw result returned, which is rare since `query_batch` requests `n_results * 3`) is treated like `low margin` — the parent gets subdivided. Truly empty raw results (zero matches at all) flag as poor with `reason="no_matches"`.

## Sub-chunking strategy

Sentence-boundary subdivision via the existing `SentenceChunker`. Algorithm:

1. Instantiate `SentenceChunker(language=parent_language)`.
2. Call `chunker.chunk([parent_text])` to get sentence strings.
3. Group sentences into sub-chunks of `max_sentences_per_subchunk` (default 3) using a sliding window with `overlap_sentences` (default 1). 3-sentence windows step by 2.
4. Drop sub-chunks shorter than `min_chunk_chars` and sub-chunks identical to the parent.
5. Compute spans by linear search inside `parent_text` (using existing `find_span_in_text` utility), then add `parent.start_char` for document-absolute positions.
6. Inherit `assertion_status` and `assertion_details` from the parent. Re-detection is not done in v1 (subdividing a NEGATED parent would risk flipping children to AFFIRMED for non-negated tail clauses).
7. Append `"adaptive_rechunker_depth_<N>"` to `source_indices.processing_stages` for traceability.

Reference signature:

```python
def subdivide_parent_chunk(
    parent_chunk: dict,
    language: str,
    config: AdaptiveRechunkingConfig,
    depth: int,
) -> list[dict]:
    """Returns sub-chunks in the same dict shape as TextProcessingPipeline output.
    Sub-chunks inherit parent's assertion. Returns [] if no useful subdivision possible."""
```

Why sentence-boundary, not sliding-window over tokens: clinical phenotype phrases live within single sentences. Splitting mid-sentence risks breaking phrases like "severe intellectual disability" across boundaries. The existing `SlidingWindowSemanticSplitter` is used by the upstream chunking pipeline but applies negation-aware merging (`chunkers.py:1103-1240`) that would partially undo sentence-level subdivision. Direct `SentenceChunker` is simpler and aligned with how clinicians write findings.

## Score-improvement gate

```python
for parent_idx, children in parent_to_children.items():
    parent_top_1 = parent_top_1_by_idx[parent_idx]   # from initial raw_query_results
    best_child_top_1 = max(child_top_1_by_idx[c] for c in children)  # from child raw results
    if best_child_top_1 < parent_top_1 + config.score_improvement_gate:
        revert_to_parent(parent_idx, children)
```

Operates on raw `top_1` values from `raw_query_results` (initial pass) and `child_raw_results` (this recursion level's child query). No re-aggregation runs before the gate decision. After the gate, the rechunker calls `orchestrate_hpo_extraction(..., precomputed_query_results=combined_raw_results)` once to re-aggregate over the curated mix — this skips retrieval because the raw results are supplied. **One real `query_batch` call per recursion level**, period.

## Aggregator behavior

The aggregator (`hpo_extraction_orchestrator.py`) is **not modified by this spec**. Adaptive rechunking works correctly with the existing `avg_score` filter and `(-avg_score, -count)` sort because:

- **Replaced parents are removed from the flat list**, so a "weak parent + strong child average to mediocre" scenario doesn't arise — only the surviving children's evidence contributes to the aggregator's per-term avg.
- **Children that don't improve enough are reverted**, so their weak evidence doesn't enter the aggregation either.
- **Multiple children that legitimately match the same HPO term** with similar strength produce a higher aggregate count, which the existing sort key already privileges as a tiebreaker.

Earlier drafts of this spec proposed two aggregator changes (filter on `max_score`, add `max_score` sort tiebreaker). They are **out of scope for this spec** because they alter behavior for users who never enable adaptive rechunking, violating the opt-in invariant. The argument for those changes (HPO-extraction quality benefits from max-score-aware filtering even without adaptive rechunking) is real but deserves its own spec and CHANGELOG cycle. Tracked in Future work.

## API parity

`api/schemas/text_processing_schemas.py:TextProcessingRequest` gains an optional `adaptive_rechunking` field shaped as `AdaptiveRechunkingProfileBlock | None`. `api/routers/text_processing_router.py` passes it through to `run_full_text_service`. Without this, CLI and API diverge — the same bug class #171 fixed.

The response schema gains an optional `meta.adaptive_rechunking` block:

```python
"meta": {
    "extraction_backend": "standard",
    "adaptive_rechunking": {
        "enabled": True,
        "trigger_count": 3,
        "subdivided_count": 2,
        "reverted_count": 1,
        "max_depth_reached": 1,
        "extra_chunks_added": 4,
    },
}
```

Block omitted entirely when `enabled=False`.

## CLI surface

Flags on `phentrieve text process` only:
- `--adaptive-rechunking / --no-adaptive-rechunking` (boolean enable)
- `--adaptive-rechunking-quality-threshold FLOAT`
- `--adaptive-rechunking-margin-threshold FLOAT`
- `--adaptive-rechunking-max-depth INT`

Full knob set reachable via YAML / profile only. Keeps CLI surface tight.

Not added to `phentrieve text interactive` in v1: interactive mode is meant to be fast and iterative; adaptive rechunking adds unpredictable latency.

## Performance posture

Adaptive rechunking only fires for users who opt in. The cost has two separable components — `query_batch` RPCs (network round trips to ChromaDB) and encoder work (SentenceTransformer.encode on chunk text) — that scale differently and must not be conflated.

**`query_batch` call count — hard invariant:**

| Scenario | Calls (`max_depth=2`) |
|---|---|
| Adaptive disabled | 1 |
| Enabled, no chunks flag | 1 |
| Enabled, chunks flag at depth 1 only | 2 |
| Enabled, chunks flag at depth 1 AND surviving children flag at depth 2 | 3 |

Tightly bounded because of `precomputed_query_results` — without it every recursion level would re-query already-known parents.

**Encoder work — loose bound:**

Each `query_batch` call encodes its full input list. A 10-chunk document where every chunk flags through depth 2 and produces 3 sentence-window children per parent encodes roughly `10 + 30 + 90 = 130` strings — about 13× the original 10. So **encoder work can substantially exceed the query-call multiplier**. In practice this is bounded by `min_chunk_chars`, the score-improvement gate (children that don't help are reverted before further recursion), and the rarity of chunks that genuinely benefit from depth-2 recursion.

**User-facing claims:**
- Typical wall time: 1.2–1.5× (only some chunks flag at depth 1, very few at depth 2). Documented in user-guide.
- Worst-case wall time: hard to bound tightly; depends on encoder batch throughput and chunk-size distribution. We do not promise a specific multiplier in docs.

**Test posture (full detail in Tests below):**
- Hard test: `query_batch.call_count == 3` for the worst-case fixture.
- Smoke test: wall time ≤ 5× — designed to catch a regression that re-runs the pipeline, not to specify performance.

## Observability

`logging.getLogger("phentrieve.adaptive_rechunker")` namespace.
- INFO: per-iteration `(trigger_count, subdivided_count, reverted_count, depth)`.
- DEBUG: full `ChunkQualitySignals` per chunk and per-parent gate decisions.

`meta.adaptive_rechunking` in the standard response is the user-facing observability surface (above).

`--show-resolved-config` (Spec A) prints the resolved `AdaptiveRechunkingConfig` source-tagged when adaptive is enabled.

`phentrieve.adaptive_rechunker.dump_quality_report(chunk_results, config) -> str` helper produces a human-readable per-chunk quality report. Library-only in v1; not exposed via CLI.

No metrics or telemetry.

## Tests

Coverage target: 100% for new modules. No drop on touched files.

| Layer | Test file | What it covers |
|---|---|---|
| Unit | `tests/unit/retrieval/adaptive_rechunker/test_quality_assessment.py` | All branches of `assess_chunk_quality`, **fed from raw `query_batch` output** (not filtered `chunk_results`): empty raw → poor/no_matches; single raw match below threshold → poor/low_score; single raw match above → ok (top_2=None special-case); two raw matches above with low margin → ok (score-AND-margin conjunction); two raw matches with low score AND low margin → poor/low_margin; encoder-default thresholds |
| Unit | `tests/unit/retrieval/adaptive_rechunker/test_raw_score_access.py` | **Critical invariant test**: a chunk where retrieval was poor (all matches below `chunk_retrieval_threshold`) produces empty `chunk_results.matches` but populated `raw_query_results[i].similarities`. Assert `assess_chunk_quality` reads `top_1`/`top_2` from raw and correctly flags as poor. Without this test, regression to filtered-input could silently break the trigger. |
| Unit | `tests/unit/retrieval/adaptive_rechunker/test_subdivide_parent_chunk.py` | Multi-sentence parent → multiple sub-chunks with correct spans; single-sentence parent → []; pysbd unavailable / empty fallback → []; sub-chunk shorter than `min_chunk_chars` → dropped; assertion-status inheritance preserved; `source_indices.processing_stages` carries depth tag |
| Unit | `tests/unit/retrieval/adaptive_rechunker/test_score_improvement_gate.py` | All children below gate → revert; one child above gate → keep; multiple parents, mixed outcomes → correct per-parent decisions |
| Unit | `tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py` | `enabled=False` → no-op; no poor chunks → no-op; one poor chunk → two children that improve → final list has children, parent removed; recursion to depth 2; recursion respects `max_depth`; **exactly one `query_batch` mock call per recursion level** (asserts the cost-model invariant) |
| Unit | `tests/unit/text_processing/test_orchestrate_with_precomputed.py` | New parameter `precomputed_query_results` on `orchestrate_hpo_extraction`: when provided, `retriever.query_batch` is not called; aggregation runs over supplied raw data; output shape identical to non-precomputed path; backward-compatibility — omitting the parameter behaves exactly as before. Also: the new `OrchestrationResult` dataclass return type works under **both** access patterns: legacy `aggregated, chunks = orchestrate_hpo_extraction(...)` (via `__iter__` returning the 2-tuple) AND new `result = orchestrate_hpo_extraction(...); result.raw_query_results` (attribute access). Both paths covered with explicit assertions. |
| Unit | `tests/unit/text_processing/test_orchestration_result_legacy_unpack.py` | Verifies each known legacy call site's unpack pattern still works: import each module listed in the Architecture section, monkey-patch `orchestrate_hpo_extraction` to return an `OrchestrationResult`, and confirm the call site's tuple unpacking succeeds. Pinpoints any consumer that broke if someone later changes `__iter__` semantics. |
| Unit | `tests/unit/text_processing/test_adapt_standard_response_extra_meta.py` | New `extra_meta` parameter on `adapt_standard_response`: with `None`, response shape unchanged (existing tests pass). With a dict, keys merge into `meta` block. Existing meta keys (`extraction_backend`, `num_processed_chunks`, etc.) are preserved; `extra_meta` keys are additive. |
| Unit | `tests/unit/retrieval/adaptive_rechunker/test_adaptive_config.py` | Pydantic dataclass defaults; YAML round-trip via `Profile.adaptive_rechunking` block; CLI flags map to dataclass fields; precedence (CLI > profile > YAML > defaults) |
| Integration | `tests/integration/test_adaptive_rechunking_e2e.py` | Real `SentenceChunker`, real `DenseRetriever` against test fixture index, synthetic clinical text known to flag as poor; assert sub-chunking happened, `meta.adaptive_rechunking.subdivided_count > 0`, final aggregated terms include at least one HPO term not present without adaptive rechunking; verify `start_char`/`end_char` of sub-chunks fall within parent span |
| Integration | `tests/integration/test_adaptive_rechunking_api.py` | Request to `/api/text/process` with `adaptive_rechunking.enabled=true` payload returns `meta.adaptive_rechunking` block; CLI and API produce identical aggregated terms for the same input + config |
| Integration | `tests/integration/test_adaptive_rechunking_benchmark.py` | Run a small fixture from `tests/data/benchmarks/german/tiny_v1.json` with adaptive on vs. off; assert ontology-aware metrics (the `9402a57` MRR-with-LCA-credit) improve or are at least not worse |
| Invariant | `tests/integration/test_adaptive_rechunking_call_count.py` | **Hard contract**: 10-chunk fixture, all chunks flag at depth 1 + all surviving children flag at depth 2 (`max_depth=2`). Mocked retriever asserts `query_batch.call_count == 3` exactly. This is the cost-model invariant — failing this test means the rechunker is re-querying chunks it shouldn't. |
| Performance | `tests/integration/test_adaptive_rechunking_perf.py` (smoke) | Same fixture as the call-count test. Asserts wall-time multiplier ≤ 5× the no-adaptive baseline. **Loose bound, smoke-test only** — designed to catch egregious regressions (e.g. accidentally re-running the full pipeline) rather than to specify performance. The 5× bound accounts for child fan-out on encoding work, which scales with sub-chunk count not query-call count. |

Cross-spec test:
- `tests/integration/test_specA_specB_interaction.py` — a profile setting `adaptive_rechunking.enabled=true` plus `quality_threshold=0.6` cleanly applies; explicit `--adaptive-rechunking-quality-threshold 0.5` overrides; `--show-resolved-config` shows both correctly source-tagged.

Fixtures:
- `tests/fixtures/adaptive_rechunking/synthetic_multi_finding.txt` — clinical text built once empirically to trigger the feature against the test ChromaDB index.
- `tests/fixtures/adaptive_rechunking/expected_terms_with_adaptive.json` — reference HPO terms expected to surface only with adaptive enabled.

## Documentation

Existing files updated (paths verified against the repo as of 2026-04-25):

- `README.md` — one-line teaser under "Features": "Optional adaptive re-chunking improves recall on multi-concept clinical sentences (`--adaptive-rechunking`)." Link to `docs/user-guide/adaptive-rechunking.md`.
- `docs/user-guide/cli-usage.md` — add the four new CLI flags under `phentrieve text process`. One-paragraph cross-reference to the adaptive-rechunking page.
- `docs/user-guide/api-usage.md` — document the new optional `adaptive_rechunking` field in `TextProcessingRequest` and the `meta.adaptive_rechunking` response block.
- `docs/user-guide/text-processing-guide.md` — short cross-reference paragraph at the end of the chunking-strategy section pointing at the new adaptive-rechunking page.
- `docs/user-guide/index.md` — add link to the new `adaptive-rechunking.md` page.
- `phentrieve.yaml.template` — add commented `extraction.adaptive_rechunking:` block with all knobs at defaults and per-knob comments.
- `CHANGELOG.md` — one entry: adaptive re-chunking feature addition. (The aggregator behavior changes considered in earlier drafts are out of scope and tracked in Future work.)
- Function and helper docstrings.

New files:
- `docs/user-guide/adaptive-rechunking.md` — canonical reference: what the feature does, when to enable it, how it works, full YAML schema, three worked examples, encoder-calibration warning, cost envelope.
- `tests/fixtures/adaptive_rechunking/synthetic_multi_finding.txt`.
- `tests/fixtures/adaptive_rechunking/expected_terms_with_adaptive.json`.

Not created:
- Architecture doc beyond this spec.
- Tuning guide beyond the user-guide section.
- Standalone benchmark numbers doc.

Documentation discipline: every YAML snippet in `docs/user-guide/adaptive-rechunking.md` is loaded by `tests/integration/test_documented_yaml.py` (shared with Spec A) and asserted to parse + resolve cleanly.

## Frontend integration

The frontend (`frontend/`) calls the API and must not break when the new optional `adaptive_rechunking` request field or the new `meta.adaptive_rechunking` response block appear. The frontend does not appear to have a generic metadata renderer today, so no work goes into "surfacing" the new metadata — it is simply parsed and discarded.

**Frontend changes for v1 (strictly minimal):**

1. **Payload pass-through** in `frontend/src/services/PhentrieveService.js`: if a caller (today no caller does this — the field is YAGNI in the UI) supplies `adaptive_rechunking`, forward it on the request body unchanged. Default behavior unchanged.
2. **Graceful response handling**: the existing response parser must not error on unknown `meta.*` keys. Verify this is already the case (most JSON parsers ignore unknown keys); if not, add a minimal fix.

That is the entire v1 scope. **No type/interface update is required** unless a typed response schema exists today and explicitly forbids extra fields — which the codebase does not currently have. **No UI surface** for the toggle or per-knob controls. **No metadata panel** for displaying trigger/subdivided counts.

**Why this is the right minimum:** the feature is opt-in, niche (multi-concept clinical paragraphs), and reachable by users via the CLI or by direct API calls. Frontend exposure follows user demand, not API surface area.

A frontend test (`frontend/src/test/services/PhentrieveService.test.js`, extended) confirms (a) a request with `adaptive_rechunking.enabled=true` is forwarded as expected, and (b) a response with `meta.adaptive_rechunking` parses without error and does not break the existing rendering path. Both are smoke tests, not integration tests.

## Migration and rollout

- v0 ships with `enabled=False` everywhere. Users opt in explicitly.
- **No aggregator behavior change.** Earlier drafts proposed switching the confidence filter from `avg_score` to `max_score`; that's been moved to Future work since it would affect users who never enable adaptive rechunking, breaking the opt-in invariant.
- README adds a one-paragraph note pointing at `docs/user-guide/adaptive-rechunking.md`.
- No data migration. No YAML migration — new sections additive.
- The new third tuple element of `orchestrate_hpo_extraction` and the new `precomputed_query_results` parameter are additive: existing callers ignoring the third element and not passing the parameter behave identically to today.
- Benchmark integration is a release gate: before merging, run the ontology-aware benchmark with adaptive on vs. off; include the diff in PR description.

## Future work

- **Embedding cache (LRU, content-hashed) in `DenseRetriever`**: amortizes sub-chunk encoding across runs. Defer until benchmarks show re-encoding dominates. → GitHub issue.
- **Ontology-coherence trigger**: real implementation of the inert YAML knob. Needs HPO graph in retrieval path, threshold tuning, benchmark validation. → GitHub issue.
- **Adaptive rechunking on `text interactive`**: deferred from v1 because of unpredictable latency. Worth revisiting once we know typical recursion behavior. → GitHub issue.
- **`phentrieve config calibrate-thresholds` subcommand**: runs current model against a fixture set, suggests `quality_threshold` / `margin_threshold` values. → GitHub issue (shared with Spec A).
- **Frontend UI exposure** of the `--adaptive-rechunking` toggle and a status panel for `meta.adaptive_rechunking`. Deferred — see Frontend integration section. → GitHub issue.
- **Score-improvement gate via re-aggregation**: v1 uses raw `top_1` comparison. Using re-aggregated `avg_score` would be more accurate but doubles `orchestrate_hpo_extraction` calls per recursion level. → Note here.
- **Adaptive rechunking benchmark integration in CI**: run ontology-aware metrics with adaptive on/off as part of the default benchmark suite, not just as a release-gate manual check. → GitHub issue.
- **Multi-encoder retrieval ensemble** as a quality signal: cheaper than ontology coherence, more discriminating than score+margin alone. Future research. → Note here.
- **Aggregator confidence filter on `max_score` (with `(-avg_score, -max_score, -count)` sort tiebreaker)**: deferred from this spec because it changes behavior for users who never enable adaptive rechunking. The argument that HPO extraction in general benefits from max-score-aware filtering is independently sound. Belongs in its own spec with its own CHANGELOG cycle and benchmarks. → GitHub issue.

## Open questions

None at spec-write time. All architecture and trigger / sub-chunking / aggregation decisions are locked.
