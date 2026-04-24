# LLM Lean V1 Routing And Phase-1 Concurrency Spec

**Date:** 2026-04-17
**Branch target:** `feat/llm-full-text-lean-v1`
**Scope:** Internal-only optimization of the current CLI and benchmark LLM path.

---

## 1. Goal

Reduce the grounded shared-chunk regression on `feat/llm-full-text-lean-v1`
without changing the public CLI or benchmark interface.

This spec covers exactly two optimizations:

1. Confidence-gated routing into Phase 2B so the mapping LLM is used only for
   ambiguous phrases.
2. Bounded concurrency for grouped Phase-1 extraction so multi-group notes do
   not pay strictly serial remote latency.

Success is measured against the current shared-chunk benchmark result:

- [results/llm/pr216_grounded_chunkrefactor.json](/home/bernt-popp/development/phentrieve/results/llm/pr216_grounded_chunkrefactor.json)
  - wall clock `1051.52s`
  - API calls `60`
  - total tokens `91,892`
  - assertion-aware micro F1 `0.7631`

The immediate objective is to materially reduce wall-clock and API calls while
maintaining or improving benchmark quality.

---

## 2. Non-goals

This work does **not** include:

- public CLI flag changes
- prompt-template redesign
- adaptive neighbor-context expansion
- token-preflight heuristics
- judge-mode or additional LLM phases
- rollback or replacement of the shared-chunk architecture

Those remain follow-up work.

---

## 3. Current problem

The shared-chunk internal refactor is stable but operationally worse than the
best grounded baseline:

| Variant | Wall clock | API calls | Total tokens | Micro F1 |
|---|---:|---:|---:|---:|
| Grounded fixed baseline | `744.10s` | `49` | `88,595` | `0.7682` |
| Shared-chunk refactor | `1051.52s` | `60` | `91,892` | `0.7631` |

The main regression signature is:

- too many remote calls, especially into Phase 2B mapping
- grouped Phase-1 extraction still running serially
- too much LLM work for phrases that are already locally resolvable

This means the next highest-ROI work is routing and concurrency, not another
large architectural refactor.

---

## 4. Design overview

### 4.1 Confidence-gated routing

Add an internal routing stage between candidate retrieval/local matching and the
Phase-2B mapping prompt.

For each extracted phrase, the pipeline will classify the phrase into one of
three outcomes:

1. **Accept locally**
   - emit a phenotype without calling the mapping LLM
2. **Defer to Phase 2B**
   - retain current mapping behavior
3. **Skip because no viable candidates**
   - preserve current no-candidate skip behavior

The routing decision will be based on:

- local match strategy confidence
- top retrieval score
- optional score margin against the second-best candidate
- language-specific routing policy

The routing stage must remain conservative. When uncertain, it defers to the
mapping LLM rather than over-accepting local matches.

### 4.2 Phase-1 grouped-call concurrency

When `extraction_groups` are present, Phase-1 calls will execute with bounded
parallelism rather than strict serial ordering.

The concurrency design must preserve:

- all current group-level traces
- deterministic merged output order
- existing failure accounting and partial-failure semantics
- current single-group behavior

The concurrency limit is internal-only for now and should default to a low,
provider-safe value.

---

## 5. Routing design

### 5.1 Confidence sources

Routing confidence will be derived from existing data already available in the
pipeline:

- local matcher outcome from `_try_local_match()`
- top candidate `score`
- score gap between top and runner-up candidate when available

The routing layer should distinguish between stronger and weaker local matches.
At minimum:

- exact token-set and normalized exact matches are considered strong
- substring and fuzzy matches are weaker and require stronger retrieval support

### 5.2 Language policy

Routing must be language-aware and conservative outside calibrated languages.

Initial policy:

- `en`
  - allow score-gated local acceptance for strong local matches
  - allow high-score fuzzy/substring acceptance only if thresholds are met
- `de`
  - allow exact and normalized-exact local acceptance
  - defer substring/fuzzy cases until German thresholds are calibrated
- unknown languages
  - use safe fallback behavior close to the current pipeline

The policy should be implemented as explicit configuration in code, not as
scattered conditionals.

### 5.3 Observability

The routing stage must emit benchmark-visible counts so its effect is measurable.

Required counts:

- `phase2b_local_accept_count`
- `phase2b_deferred_count`
- `phase2b_no_candidate_skip_count`

Optional but useful:

- per-language routing policy name
- strong-match vs weak-match acceptance counts

These counts should appear in the LLM benchmark meta/observability path without
breaking current consumers.

---

## 6. Phase-1 concurrency design

### 6.1 Execution model

Grouped extraction will use bounded concurrency over the existing per-group
Phase-1 call path.

The implementation may use:

- `asyncio.to_thread`
- a thread pool
- a small executor abstraction

The design should favor minimal change to provider code and keep the current
provider interface intact if possible.

### 6.2 Ordering and merge semantics

Parallel execution must not create nondeterministic final outputs.

Required behavior:

- groups may complete in any order internally
- the pipeline merges group results in stable extraction-group order
- existing `_deduplicate_phase1_extractions()` semantics stay intact

### 6.3 Failures

If some groups fail and others succeed:

- current partial-failure accounting must remain correct
- successful groups still contribute results
- all-group failure still raises the existing phase error

No group failure should silently disappear because of concurrent execution.

---

## 7. Files expected to change

Primary code:

- `phentrieve/llm/pipeline.py`
- `phentrieve/llm/types.py` if a typed routing decision/result is needed
- `phentrieve/text_processing/full_text_service.py`
- `phentrieve/benchmark/llm_benchmark.py`

Tests:

- `tests/unit/llm/test_pipeline.py`
- `tests/unit/text_processing/test_full_text_service.py`
- `tests/unit/test_llm_benchmark.py`
- `tests/integration/test_benchmark_workflow.py` if observability shape changes

No public CLI or API schema change is required.

---

## 8. Testing requirements

### 8.1 Unit tests

Add unit coverage for:

- high-confidence English local acceptance
- conservative German routing
- fallback behavior for unknown languages
- ambiguous phrases correctly deferred to Phase 2B
- no-candidate skip path counts
- parallel Phase-1 grouped execution preserving stable merge order
- partial group failure accounting under concurrency

### 8.2 Verification

Required repo verification after implementation:

- `make check`
- `make typecheck-fast`
- `make test`

Required benchmark verification:

- rerun the 10-doc GeneReviews grounded benchmark on the branch
- compare against:
  - `pr216_grounded_benchmark_fix1.json`
  - `pr216_grounded_chunkrefactor.json`

---

## 9. Acceptance criteria

The work is successful when all of the following are true:

1. The grounded shared-chunk path remains functionally correct.
2. No benchmark documents fail.
3. API calls decrease from `60`.
4. Wall-clock decreases from `1051.52s`.
5. Assertion-aware micro F1 does not drop below `0.7631`.
6. Repo verification remains green.

Stretch target:

- recover toward the grounded fixed baseline:
  - API calls near `49`
  - wall-clock below `800s`

---

## 10. Rollout guidance

Land this work on `feat/llm-full-text-lean-v1`, not on a new feature branch.

This branch remains the maturation branch until the CLI path is stable enough
to merge into `main`. The benchmark comparison story depends on keeping the
optimization sequence on one branch with stable artifact lineage.
