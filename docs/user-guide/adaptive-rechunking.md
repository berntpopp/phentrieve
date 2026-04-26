# Adaptive Re-Chunking

Optional retrieval-quality-driven sub-chunking. When a chunk's retrieval is
poor (low top-1 similarity AND low margin between top-1 and top-2), Phentrieve
subdivides the chunk into sentence-bounded sub-chunks, re-queries each, and
merges the results. Improves recall on multi-concept clinical sentences
without affecting users who don't enable it.

The feature is opt-in. Default behavior is unchanged.

## When to Enable

Enable adaptive re-chunking when:

- Inputs include long multi-finding paragraphs that combine several phenotypes
  into one sentence.
- You want to maximize recall and accept up to ~1.5x retrieval cost on average.
- You're using a BioLORD-class biomedical encoder (default thresholds are
  calibrated for this family).

Skip it when:

- Latency matters more than recall (e.g. interactive triage).
- Inputs are typically short single-finding sentences. The trigger rarely
  fires and the overhead is wasted.
- You're using a non-biomedical encoder without retuning the thresholds.
- You're running `phentrieve text interactive`. Adaptive rechunking is not
  wired into interactive mode in v1.

## Quick Start

CLI:

```bash
phentrieve text process note.txt --adaptive-rechunking
```

YAML (`phentrieve.yaml`):

```yaml
extraction:
  adaptive_rechunking:
    enabled: true
    quality_threshold: 0.55
    max_depth: 2
```

Profile (combine with [Configuration Profiles](./configuration-profiles.md)):

```yaml
profiles:
  high_recall:
    command: text process
    adaptive_rechunking:
      enabled: true
      quality_threshold: 0.55
```

```bash
phentrieve text process note.txt --profile high_recall
```

## How It Works

The orchestration loop runs after the standard extraction completes and
before the response is adapted. It has four steps per recursion level:

1. **Detect**: For each chunk, read the unfiltered top-K from `query_batch`
   and compute `top_1`, `top_2`, and `margin = top_1 - top_2`. A chunk flags
   as poor when:

   ```
   top_1 < quality_threshold
       AND (margin < margin_threshold OR top_2 is None)
   ```

   Both conditions must hold. A confident top-1 (above the quality threshold)
   trusts the result regardless of margin; a clear-but-weak match (low score
   but large margin) is also trusted.

2. **Subdivide**: Each flagged chunk is re-chunked at sentence boundaries via
   `SentenceChunker`, grouped into sliding windows of
   `max_sentences_per_subchunk` sentences with `overlap_sentences` overlap.
   Sub-chunks shorter than `min_chunk_chars` are dropped. If only one
   sub-chunk falls out, the parent is kept (no useful subdivision).

3. **Query and gate**: All accepted child chunks are queried in a single
   batched `query_batch` call. For each parent, the score-improvement gate
   compares the parent's `top_1` to the best child's `top_1`. Subdivisions
   that don't lift the top-1 by `score_improvement_gate` are reverted - the
   parent stays in the flat list and its children are dropped.

4. **Re-aggregate**: A combined raw-results dict is built (kept original
   results for non-subdivided / reverted parents, plus accepted child
   results) and `orchestrate_hpo_extraction` is called with
   `precomputed_query_results=combined_raw_results`. This re-runs filtering
   and aggregation only - no new `query_batch` call.

Steps 1-4 then recurse on surviving children up to `max_depth` levels.

## Configuration

| Knob | Default | Notes |
|---|---|---|
| `enabled` | `false` | Opt-in switch. |
| `quality_threshold` | `0.55` | top-1 similarity floor. Encoder-calibrated for BioLORD. |
| `margin_threshold` | `0.03` | Minimum top-1 minus top-2 distance. |
| `max_depth` | `2` | Recursion cap (depth-1 = subdivide once, depth-2 = subdivide children). |
| `min_chunk_chars` | `30` | Sub-chunks shorter than this are dropped. |
| `max_sentences_per_subchunk` | `3` | Sliding-window size in sentences. |
| `overlap_sentences` | `1` | Sentence-level overlap between windows. |
| `score_improvement_gate` | `0.05` | Children below this top-1 lift over the parent are reverted. |
| `use_ontology_coherence` | `false` | Reserved, inert in v1. See Future Work in the spec. |

The full YAML schema (under the `extraction:` section):

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

The CLI exposes the four most-tuned knobs:

- `--adaptive-rechunking` / `--no-adaptive-rechunking`
- `--adaptive-rechunking-quality-threshold FLOAT`
- `--adaptive-rechunking-margin-threshold FLOAT`
- `--adaptive-rechunking-max-depth INT`

Other knobs are reachable only via YAML or a profile. Resolution order is
CLI > profile > YAML > defaults; use `--show-resolved-config` to inspect the
final values.

## Encoder Calibration Warning

The default `quality_threshold` (0.55) and `margin_threshold` (0.03) are
calibrated for BioLORD-class biomedical encoders. If you switch to a
different `retrieval_model`, the score distribution will differ - sentence-
transformers/LaBSE, for example, produces markedly lower cosine similarities
on the same input - and the trigger will mis-fire (either over-firing on
every chunk or never firing). Retune both thresholds for your encoder before
trusting the feature.

A future `phentrieve config calibrate-thresholds` subcommand will help with
this.

## Cost Envelope

Two costs scale differently and shouldn't be conflated:

**`query_batch` RPC count - hard invariant:**

- Adaptive disabled: 1 call.
- Enabled, no chunks flag: 1 call.
- Enabled, chunks flag at depth 1 only: 2 calls.
- Enabled, chunks flag at depth 1 AND surviving children flag at depth 2: 3 calls.

The general bound at depth `N` is `1 + N` calls.

**Encoder work - loose bound:** each `query_batch` call encodes its full
input list. A 10-chunk document where every chunk flags through depth 2
with three sentence-window children per parent encodes roughly
`10 + 30 + 90 = 130` strings - about 13x the original 10. So encoder
workload can substantially exceed the query-call multiplier.

In practice fan-out is bounded by `min_chunk_chars`, the score-improvement
gate, and the rarity of chunks that genuinely benefit from depth-2
recursion. Empirical typical wall-time multiplier: **1.2x to 1.5x**. Worst
case is hard to bound tightly because it depends on encoder batch
throughput and chunk-size distribution; we don't promise a specific
multiplier.

## Worked Examples

### Aggressive recall

Lower the quality threshold so more chunks flag, and keep depth at 2:

```yaml
extraction:
  adaptive_rechunking:
    enabled: true
    quality_threshold: 0.6
    margin_threshold: 0.05
    max_depth: 2
```

### Conservative cost

Cap recursion at 1 and require a larger improvement before keeping
sub-chunks:

```yaml
extraction:
  adaptive_rechunking:
    enabled: true
    max_depth: 1
    score_improvement_gate: 0.1
```

### Cross-language (German) via profile

```yaml
profiles:
  german_recall:
    command: text process
    language: de
    adaptive_rechunking:
      enabled: true
      quality_threshold: 0.5
```

```bash
phentrieve text process note.txt --profile german_recall
```

## Response Metadata

When enabled and triggered, the response `meta` block carries a summary:

```json
{
  "meta": {
    "extraction_backend": "standard",
    "adaptive_rechunking": {
      "enabled": true,
      "trigger_count": 3,
      "subdivided_count": 2,
      "reverted_count": 1,
      "max_depth_reached": 1,
      "extra_chunks_added": 4
    }
  }
}
```

The block is omitted entirely when adaptive rechunking is disabled.

## Related Notes

- The interactive command (`phentrieve text interactive`) does not currently
  accept the adaptive rechunking flags. The feature is wired into
  `phentrieve text process` and the API only.
- The CLI surface intentionally exposes only four knobs. The full set is
  reachable via YAML / profiles for tuning experiments.
- The aggregator (`hpo_extraction_orchestrator`) is unchanged by this
  feature. Adaptive rechunking only changes the chunk list and the raw
  query results that feed the existing aggregator.
- See [Configuration Profiles](./configuration-profiles.md) for how to
  bundle adaptive-rechunking knobs into named profiles.
- See [CLI Usage](./cli-usage.md) for the `text process` command surface.
- See [API Usage](./api-usage.md) for the request/response schema.
