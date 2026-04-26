# Ontology-Aware Benchmark Metrics Spec

Date: 2026-04-24
Status: consolidated implementation input

This document consolidates the local code investigation, Janpaul bench-fork
findings, literature review, implementation plan, and GeneReviews smoke test for
ontology-aware full-text HPO extraction benchmarking.

## Goal

Build benchmark metrics for full-text HPO extraction that give partial credit
when a predicted HPO term is ontologically close to the gold term.

Example:

```text
gold:      HP:0001249 Intellectual disability
predicted: HP:0001256 Mild intellectual disability
```

The current strict exact metric counts this as one false positive and one false
negative. The new metric should score it as a strong but imperfect match.

## Non-Goals

- Do not change runtime full-text extraction ranking or retrieval behavior.
- Do not replace strict exact precision/recall/F1.
- Do not hide the benchmark effect behind a single opaque score.
- Do not use LLM judging for this metric. The metric should be deterministic and
  based on HPO structure plus semantic similarity.

## Existing Code And Prior Art

### Current strict extraction metrics

`phentrieve/evaluation/extraction_metrics.py` evaluates each document as exact
sets of `(hpo_id, assertion)` tuples:

```text
tp = predicted_set & gold_set
fp = predicted_set - gold_set
fn = gold_set - predicted_set
```

This remains the strict baseline.

### Existing single-term ontology similarity

`phentrieve/evaluation/metrics.py` already has single-term and retrieval-style
ontology similarity:

- `calculate_semantic_similarity(expected_term, retrieved_term, formula)`
- `calculate_max_similarity(expected_terms, retrieved_terms, top_k, formula)`
- `average_max_similarity(expected_terms, retrieved_terms, top_k, formula)`
- `calculate_test_case_max_ont_sim(expected_ids, retrieved_ids, formula)`

This is useful term-term infrastructure, but not sufficient for full-document
extraction scoring because it does not convert multiple predicted and gold terms
into document-level TP/FP/FN.

Important limitation: the current "Resnik" implementation is a depth/LCA proxy,
not true information-content Resnik. It can undervalue close clinical
parent/child pairs and sometimes logs invalid path-length warnings. V1 can use
it as a fallback only, but relation-aware parent/child credit should dominate.

### Existing relaxed semantic PR/F1

`phentrieve/evaluation/semantic_metrics.py` includes thresholded semantic PR/F1.
It exact-matches first, then greedily gives full credit when similarity exceeds a
threshold. This is close, but not the desired final metric because:

- a pair just below threshold gets zero;
- a pair just above threshold gets full credit;
- greedy matching can miss the best one-to-one document assignment.

Keep it for compatibility, but do not make it the headline ontology-aware
benchmark metric.

### Janpaul bench-fork findings

`../phentrieve-bench` branches reviewed:

- `feature/llm-annotation-system`
- `feature/agentic-judge-mode`

Findings:

- The fork has `MaxOntSim@K` style retrieval metrics.
- It has thresholded semantic PR/F1 similar to this repo.
- It has a hierarchical matching direction with exact, ancestor, descendant,
  sibling, cousin, and unrelated classes.
- It does not appear to have a finished, robust full-document soft PR/F1 metric
  with optimal one-to-one matching and auditable fractional TP/FP/FN.

Interpretation: use the fork as design support, not as a drop-in solution.

## Metric Set To Build

### 1. Strict exact PR/F1

Keep current exact metrics unchanged.

Names:

- `strict_precision`
- `strict_recall`
- `strict_f1`

Strict metrics stay as the comparability baseline and should remain the default
top-level output where existing callers expect `micro`, `macro`, and `weighted`.

### 2. Soft ontology PR/F1

This is the new headline ontology-aware benchmark metric.

For each document and assertion bucket, compute pairwise credit:

```text
credit(predicted_hpo_id, gold_hpo_id) in [0, 1]
```

Then solve maximum-weight one-to-one matching. A predicted term can credit at
most one gold term, and a gold term can be credited by at most one prediction.

Definitions:

```text
soft_tp = sum(matched_pair_credits)
soft_fp = number_of_predictions - soft_tp
soft_fn = number_of_gold_terms - soft_tp

soft_precision = soft_tp / number_of_predictions
soft_recall    = soft_tp / number_of_gold_terms
soft_f1        = harmonic_mean(soft_precision, soft_recall)
```

For the intellectual-disability example, if the pair credit is `0.90`:

```text
soft_tp = 0.90
soft_fp = 0.10
soft_fn = 0.10
soft_f1 = 0.90
```

### 3. Partial best-match diagnostics

Also report directional best-match metrics:

```text
partial_precision = mean over predictions of max credit(prediction, any gold)
partial_recall    = mean over gold terms of max credit(any prediction, gold)
partial_f1        = harmonic_mean(partial_precision, partial_recall)
```

These are diagnostics, not the headline metric. They can be optimistic because
they are not one-to-one and can reuse the same nearby term.

### 4. Match breakdown and examples

Every detailed benchmark report should expose:

- exact match count and credit;
- descendant match count and credit;
- ancestor match count and credit;
- sibling match count and credit;
- cousin match count and credit;
- semantic fallback match count and credit;
- unrelated count if included in diagnostics;
- matched pair details for inspection.

Matched pair details should include:

- predicted HPO ID and label;
- gold HPO ID and label;
- assertion bucket;
- match kind;
- credit;
- relationship distance when relevant;
- fallback semantic similarity when relevant.

## Pair Credit Function

Use relation-aware credit for V1:

```text
if pred == gold:
    credit = 1.0
elif pred is descendant of gold:
    credit = max(0.75, 0.95 - 0.05 * (distance - 1))
elif pred is ancestor of gold:
    credit = max(0.50, 0.85 - 0.08 * (distance - 1))
elif pred and gold are siblings:
    credit = max(term_similarity, 0.65)
elif pred and gold are cousins:
    credit = max(term_similarity, 0.45)
else:
    credit = term_similarity if term_similarity >= 0.30 else 0.0
```

Rationale:

- More-specific predicted child terms are often acceptable near misses, so
  descendant credit starts high.
- More-general predicted parent terms lose specificity, so ancestor credit starts
  lower.
- Sibling/cousin credit should be visible and calibratable because these are the
  most likely places to over-credit.
- Exact matches must be locked before soft matching so non-exact similarity
  cannot displace a true exact match.

## Assertion Handling

Only compare predicted and gold terms with the same assertion bucket.

Examples:

```text
("HP:0001249", "PRESENT") matches only PRESENT gold terms.
("HP:0001249", "ABSENT") must not receive credit for PRESENT gold terms.
```

Use the same assertion normalization already used by benchmark data loading.

## Matching Algorithm

For each document:

1. Normalize predictions and gold annotations to unique `(hpo_id, assertion)`
   tuples.
2. Group predictions and gold terms by assertion.
3. Lock exact tuple matches first.
4. Remove locked exact matches from the unmatched prediction/gold pools.
5. Build a pair-credit matrix for each remaining assertion bucket.
6. Run maximum-weight bipartite matching.
7. Drop zero-credit pairs.
8. Sum matched credits into `soft_tp`.
9. Compute `soft_fp`, `soft_fn`, precision, recall, and F1.
10. Compute partial best-match diagnostics separately.
11. Emit detailed matched-pair records when detailed output is enabled.

Use `networkx.algorithms.matching.max_weight_matching` if practical. The repo
already depends on `networkx`; avoiding a new optimization dependency is the
simplest V1.

## Output Shape

Keep backward compatibility:

- Existing top-level `micro`, `macro`, and `weighted` values remain strict exact
  metrics.
- Add ontology-aware metrics in explicit new fields.

Recommended nested output:

```json
{
  "strict": {
    "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
  },
  "soft": {
    "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
  },
  "partial": {
    "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
  },
  "match_breakdown": {
    "exact": {"count": 0, "credit": 0.0},
    "descendant": {"count": 0, "credit": 0.0},
    "ancestor": {"count": 0, "credit": 0.0},
    "sibling": {"count": 0, "credit": 0.0},
    "cousin": {"count": 0, "credit": 0.0},
    "semantic": {"count": 0, "credit": 0.0}
  }
}
```

For CLI summaries, add flat convenience fields:

- `soft_micro_precision`
- `soft_micro_recall`
- `soft_micro_f1`
- `partial_micro_precision`
- `partial_micro_recall`
- `partial_micro_f1`

## Proposed Files

Create:

- `phentrieve/evaluation/ontology_credit.py`
  - relationship classification;
  - pair-credit dataclasses and config;
  - `calculate_pair_credit()`.

- `phentrieve/evaluation/ontology_matching.py`
  - document-level soft matching;
  - partial best-match metrics;
  - matched-pair records.

- `tests/unit/evaluation/test_ontology_credit.py`
  - pair-credit unit tests with mocked graph data and one real-HPO integration
    case when data is available.

- `tests/unit/evaluation/test_ontology_matching.py`
  - exact, partial, assertion-mismatch, one-to-one, overprediction, and
    underprediction cases.

Modify:

- `phentrieve/evaluation/extraction_metrics.py`
  - corpus aggregation for ontology-aware metrics.

- `phentrieve/benchmark/extraction_benchmark.py`
  - include ontology metrics in benchmark results.

- `phentrieve/benchmark/extraction_cli.py`
  - add toggles/configuration if needed.

- `phentrieve/benchmark/extraction_reporter.py`
  - render soft/partial metrics and match breakdown.

- `phentrieve/evaluation/semantic_metrics.py`
  - clarify that existing thresholded semantic PR/F1 is compatibility behavior,
    not the preferred ontology-aware metric.

## Benchmark Command Integration

V1 should expose ontology-aware scoring as an explicit benchmark option on both
full-text benchmark command surfaces:

```bash
phentrieve benchmark extraction run ... --ontology-aware-metrics
phentrieve benchmark llm ... --ontology-aware-metrics
```

Use opt-in default behavior for V1:

- default: ontology-aware metrics disabled;
- `--ontology-aware-metrics`: calculate and persist soft/partial metrics;
- `--no-ontology-aware-metrics`: explicit off switch for scripts and parity.

Reasoning: strict benchmark outputs should stay stable unless the caller asks
for the new scoring. After calibration, we can reconsider making the option
default-on.

The option must not change extraction behavior, prompts, retrieval, LLM calls, or
strict exact metrics. It only adds extra evaluation over already-produced
predicted/gold HPO tuples.

### Extraction Benchmark Command

Target command:

```bash
phentrieve benchmark extraction run tests/data/en/phenobert \
  --dataset GeneReviews \
  --ontology-aware-metrics
```

Implementation surface:

- `phentrieve/benchmark/extraction_cli.py`
  - add Typer option;
  - pass the value into `ExtractionConfig`.
- `phentrieve/benchmark/extraction_benchmark.py`
  - add config fields;
  - calculate ontology metrics after strict metrics;
  - include ontology output in `extraction_results.json`;
  - include flat ontology summary fields in `extraction_summary.json`.
- `phentrieve/benchmark/extraction_reporter.py`
  - display ontology-aware metrics when result files contain them.

### LLM Benchmark Command

Target command:

```bash
phentrieve benchmark llm \
  --test-file tests/data/en/phenobert \
  --dataset GeneReviews \
  --llm-model gemini-3.1-flash-lite-preview \
  --llm-provider gemini \
  --ontology-aware-metrics
```

Implementation surface:

- `phentrieve/benchmark/llm_cli.py`
  - add Typer option;
  - pass through `run_llm_benchmark_cli()`;
  - include the option in checkpoint compatibility state;
  - write ontology metrics to artifact metrics JSON.
- `phentrieve/benchmark/llm_benchmark.py`
  - accept the option in `run_llm_benchmark()`;
  - compute ontology metrics for both assertion-aware and id-only result sets
    when enabled;
  - include ontology settings and metrics in the benchmark payload.

### Suggested Option Names

Use explicit names to avoid confusing the feature with runtime ontology
similarity:

```text
--ontology-aware-metrics / --no-ontology-aware-metrics
--ontology-semantic-floor FLOAT
--ontology-similarity-formula [hybrid|simple_resnik_like]
```

The semantic floor and formula options are advanced configuration. They can be
added in the same implementation if the plumbing stays small; otherwise ship the
boolean option first with documented defaults.

Suggested fields:

```text
ontology_aware_metrics: bool = False
ontology_semantic_floor: float = 0.30
ontology_similarity_formula: str = "hybrid"
```

## Tests And Acceptance Criteria

Required behavior:

- Strict exact metrics remain unchanged.
- Exact matches receive `1.0` credit.
- `HP:0001256` Mild intellectual disability predicted for `HP:0001249`
  Intellectual disability receives high descendant credit.
- More-specific predicted child terms score higher than more-general predicted
  parent terms at the same distance.
- Assertion mismatches receive zero credit.
- One prediction cannot credit multiple gold terms in soft PR/F1.
- Extra unrelated predictions lower soft precision.
- Missing related predictions lower soft recall.
- Partial best-match metrics are reported separately and clearly labeled as
  diagnostic.
- Detailed output exposes the actual matched pairs and credits.

Verification commands:

```bash
uv run pytest tests/unit/evaluation/test_ontology_credit.py -q
uv run pytest tests/unit/evaluation/test_ontology_matching.py -q
uv run pytest tests/unit/test_extraction_metrics.py -q
make check
make typecheck-fast
make test
```

## Smoke Test Evidence

### Current saved Gemini 3.1 Flash Lite artifact

Artifact:

```text
results/llm/g31_flash_lite_preview_full10/
```

Strict saved metrics:

```text
precision = 0.829060
recall    = 0.818565
f1        = 0.823779
tp/fp/fn  = 194 / 40 / 43
```

Ontology-aware smoke on the same predictions:

| Metric | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Strict exact | 0.829060 | 0.818565 | 0.823779 |
| Soft one-to-one ontology | 0.874583 | 0.863513 | 0.869013 |
| Partial best-match diagnostic | 0.938051 | 0.932750 | 0.935393 |

Matched-pair breakdown:

| Kind | Matched pairs | Total credit |
| --- | ---: | ---: |
| exact | 194 | 194.000 |
| descendant | 4 | 3.600 |
| ancestor | 4 | 3.320 |
| sibling | 3 | 1.950 |
| cousin | 2 | 0.914 |
| semantic | 2 | 0.868 |

### Live rerun with `.env` loaded

Artifact:

```text
results/llm/rerun-g31-flash-lite-20260424/
```

Strict rerun metrics:

```text
precision = 0.819742
recall    = 0.805907
f1        = 0.812766
tp/fp/fn  = 191 / 42 / 46
pred/gold = 233 / 237
tokens    = 117875
cost      = $0.05051125
wall time = 76.69s
```

Ontology-aware smoke on the rerun predictions:

| Metric | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Strict exact | 0.819742 | 0.805907 | 0.812766 |
| Soft one-to-one ontology | 0.880022 | 0.865169 | 0.872532 |
| Partial best-match diagnostic | 0.943259 | 0.928945 | 0.936047 |

Matched-pair breakdown:

| Kind | Matched pairs | Total credit |
| --- | ---: | ---: |
| exact | 191 | 191.000 |
| descendant | 8 | 7.200 |
| ancestor | 3 | 2.550 |
| sibling | 4 | 2.600 |
| cousin | 2 | 0.914 |
| semantic | 2 | 0.781 |

Interpretation: current strict F1 is already around `0.81-0.82`, and
ontology-aware soft F1 raises the benchmark to about `0.87` by crediting real
near misses instead of treating them as fully wrong.

## Literature Basis

The evaluation direction is consistent with ontology annotation evaluation and
HPO phenotype-profile similarity practice:

- keep strict exact metrics for comparability;
- add partial precision/recall style metrics;
- use term-term ontology similarity and best-match aggregation;
- prefer one-to-one matching for extraction-style PR/F1;
- expose matched terms and credits for auditability.

References:

- HPO overview: https://human-phenotype-ontology.github.io/about.html
- HPO annotations: https://obophenotype.github.io/human-phenotype-ontology/annotations/introduction/
- Partial precision/recall for ontology annotations:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6301375/
- HPO phenotype-profile semantic similarity:
  https://link.springer.com/article/10.1186/s12911-022-01770-4
- Biomedical ontology semantic similarity review:
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000443
- HPO imprecision/noise and set similarity:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5998886/
- PyHPO similarity algorithms:
  https://pyhpo.readthedocs.io/en/stable/similarity.html
- Monarch Semsimian:
  https://monarch-initiative.github.io/monarch-documentation/Repositories/semsimian/

## Calibration Risks

- Sibling and cousin minimum credits may be too generous. They must be reviewed
  using matched-pair reports across more corpora.
- The current legacy semantic similarity fallback is not strong enough for final
  scientific claims. V1 can still ship relation-aware soft matching, but the
  fallback similarity should be upgraded to true IC/MICA Resnik or Lin.
- HPO is a DAG, not a tree. Relationship and distance logic must use ancestor
  sets and depths carefully and should not assume a single parent.
- Partial best-match scores can look very high. They must be labeled as
  diagnostic, not headline extraction F1.

## V1 Completion Criteria

V1 is complete when:

- strict benchmark outputs remain backward-compatible;
- new `soft` and `partial` metric blocks are emitted for full-text extraction
  benchmarks;
- detailed output contains auditable matched-pair records;
- unit tests cover exact, parent/child, sibling/cousin, semantic fallback,
  assertion mismatch, overprediction, underprediction, and one-to-one behavior;
- GeneReviews smoke can be reproduced from saved predictions without an ad hoc
  script;
- docs/comments clearly distinguish strict exact, soft ontology, and partial
  best-match metrics.

## Post-V1 Work

- Replace the current depth/LCA "Resnik" proxy with true IC/MICA similarity.
- Add Lin similarity for normalized IC comparison.
- Calibrate relation-aware, Resnik, Lin, and thresholded relaxed PR/F1 on the
  same benchmark corpora.
- Consider Semsimian or OAK as a validation oracle for pairwise HPO similarity
  if dependency policy allows it.
