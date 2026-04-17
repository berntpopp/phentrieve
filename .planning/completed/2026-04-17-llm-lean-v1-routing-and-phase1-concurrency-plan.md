# LLM Lean V1 Routing And Phase-1 Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce unnecessary Phase-2B LLM calls and serial Phase-1 grouped latency on `feat/llm-full-text-lean-v1` without changing the public CLI or benchmark interface.

**Architecture:** Keep the current two-phase shared-chunk pipeline intact, add an internal confidence-gated routing layer after candidate retrieval/local matching, and execute grouped Phase-1 extraction with bounded concurrency while preserving deterministic merge/failure semantics.

**Tech Stack:** Python 3.11, Ruff, mypy, pytest, Gemini-backed LLM provider, existing `TwoPhaseLLMPipeline`, `.planning/` specs and benchmark artifacts.

**Status:** Completed on 2026-04-17. Final rerun artifact: `results/llm/pr216_grounded_routing_phase1_parallel_fix4.json`.

---

## File map

- `phentrieve/llm/pipeline.py`
  - add routing helpers and bounded Phase-1 concurrency
  - preserve current result types and benchmark-visible metadata
- `phentrieve/llm/types.py`
  - add typed routing metadata only if it simplifies the pipeline
- `phentrieve/text_processing/full_text_service.py`
  - expose new observability counters without changing public response shape
- `phentrieve/benchmark/llm_benchmark.py`
  - record routing counters in benchmark output
- `tests/unit/llm/test_pipeline.py`
  - primary TDD surface for routing and Phase-1 concurrency behavior
- `tests/unit/text_processing/test_full_text_service.py`
  - verify surfaced observability
- `tests/unit/test_llm_benchmark.py`
  - verify benchmark observability carries the new counters
- `tests/integration/test_benchmark_workflow.py`
  - update only if metric shape changes require it

---

### Task 1: Add routing-decision unit tests

**Files:**
- Modify: `tests/unit/llm/test_pipeline.py`
- Modify: `phentrieve/llm/pipeline.py`

- [x] **Step 1: Write failing tests for English high-confidence local acceptance**

Add tests near the existing local-match / Phase-2B tests:

```python
def test_two_phase_pipeline_routes_high_confidence_english_match_locally() -> None:
    pipeline = TwoPhaseLLMPipeline(
        provider=FakeProvider(),
        tool_executor=FakeToolExecutor(),
        prompt_registry=FakePromptRegistry(),
    )
    item = {
        "phrase": "scoliosis",
        "category": "Abnormal",
        "candidates": [
            {"hpo_id": "HP:0002650", "term_name": "Scoliosis", "score": 0.97},
            {"hpo_id": "HP:0000001", "term_name": "All", "score": 0.21},
        ],
        "grounded_context": {"chunk_ids": [1], "primary_chunk_text": "scoliosis", "neighbor_chunk_texts": []},
        "chunk_ids": [1],
        "evidence_text": "scoliosis",
    }

    resolved, unresolved, counts = pipeline._route_phase2_candidates(
        phrase_candidates=[item],
        language="en",
    )

    assert [term.term_id for term in resolved] == ["HP:0002650"]
    assert unresolved == []
    assert counts["phase2b_local_accept_count"] == 1
    assert counts["phase2b_deferred_count"] == 0
```

- [x] **Step 2: Write failing tests for conservative German and ambiguous fallback**

Add:

```python
def test_two_phase_pipeline_keeps_german_substring_match_deferred() -> None:
    pipeline = TwoPhaseLLMPipeline(
        provider=FakeProvider(),
        tool_executor=FakeToolExecutor(),
        prompt_registry=FakePromptRegistry(),
    )
    item = {
        "phrase": "deutliche skoliose der wirbelsaeule",
        "category": "Abnormal",
        "candidates": [
            {"hpo_id": "HP:0002650", "term_name": "Skoliose", "score": 0.89},
            {"hpo_id": "HP:0001627", "term_name": "Myokarditis", "score": 0.62},
        ],
        "grounded_context": {"chunk_ids": [1], "primary_chunk_text": "deutliche skoliose der wirbelsaeule", "neighbor_chunk_texts": []},
        "chunk_ids": [1],
        "evidence_text": "deutliche skoliose der wirbelsaeule",
    }

    resolved, unresolved, counts = pipeline._route_phase2_candidates(
        phrase_candidates=[item],
        language="de",
    )

    assert resolved == []
    assert len(unresolved) == 1
    assert counts["phase2b_local_accept_count"] == 0
    assert counts["phase2b_deferred_count"] == 1
```

Also add a no-candidate skip test.

- [x] **Step 3: Run the targeted tests and confirm failure**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "routes_high_confidence_english_match_locally or keeps_german_substring_match_deferred or no_candidate_skip" -v --no-cov
```

Expected: failing with missing `_route_phase2_candidates` or incorrect counts.

- [x] **Step 4: Commit the red tests**

```bash
git add tests/unit/llm/test_pipeline.py
git commit -m "test: add phase2 routing expectations"
```

---

### Task 2: Implement confidence-gated routing

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `tests/unit/llm/test_pipeline.py`

- [x] **Step 1: Add minimal routing helpers**

Implement focused helpers inside `TwoPhaseLLMPipeline`:

```python
def _route_phase2_candidates(
    self,
    *,
    phrase_candidates: list[dict[str, Any]],
    language: str,
) -> tuple[list[LLMPhenotype], list[dict[str, Any]], dict[str, int]]:
    resolved: list[LLMPhenotype] = []
    unresolved: list[dict[str, Any]] = []
    counts = {
        "phase2b_local_accept_count": 0,
        "phase2b_deferred_count": 0,
        "phase2b_no_candidate_skip_count": 0,
    }
    for item in phrase_candidates:
        candidates = item.get("candidates", [])
        if not candidates:
            counts["phase2b_no_candidate_skip_count"] += 1
            continue
        decision = self._decide_phase2_routing(item=item, language=language)
        if decision == "local":
            local_match = self._try_local_match(item["phrase"], candidates)
            if local_match is not None:
                resolved.append(self._phenotype_from_candidate(item=item, candidate=local_match))
                counts["phase2b_local_accept_count"] += 1
                continue
        unresolved.append(item)
        counts["phase2b_deferred_count"] += 1
    return resolved, unresolved, counts
```

- [x] **Step 2: Add a conservative routing policy helper**

Implement a minimal internal decision helper:

```python
def _decide_phase2_routing(self, *, item: dict[str, Any], language: str) -> str:
    local_match = self._try_local_match(str(item["phrase"]), list(item.get("candidates", [])))
    if local_match is None:
        return "defer"

    candidates = list(item.get("candidates", []))
    top_score = float(candidates[0].get("score", 0.0) or 0.0) if candidates else 0.0
    second_score = float(candidates[1].get("score", 0.0) or 0.0) if len(candidates) > 1 else 0.0
    margin = top_score - second_score
    normalized_language = (language or "").strip().lower()
    phrase_clean = _clean_text(str(item["phrase"]))
    term_clean = _clean_text(str(local_match["term_name"]))

    if normalized_language == "en":
        if phrase_clean == term_clean and top_score >= 0.85:
            return "local"
        if top_score >= 0.94 and margin >= 0.08:
            return "local"
        return "defer"

    if normalized_language == "de":
        if phrase_clean == term_clean and top_score >= 0.85:
            return "local"
        return "defer"

    return "defer"
```

- [x] **Step 3: Wire routing into the existing Phase-2 flow**

Replace the current unconditional local-match-then-map split with the routing
helper in `run()` or the existing Phase-2 path:

```python
resolved_local, unresolved, routing_counts = self._route_phase2_candidates(
    phrase_candidates=phrase_candidates,
    language=language,
)
```

Then merge `routing_counts` into `meta.phase_counts`.

- [x] **Step 4: Run the targeted tests and ensure they pass**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "routes_high_confidence_english_match_locally or keeps_german_substring_match_deferred or no_candidate_skip" -v --no-cov
```

Expected: PASS.

- [x] **Step 5: Run the broader pipeline unit slice**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -v --no-cov
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add phentrieve/llm/pipeline.py tests/unit/llm/test_pipeline.py
git commit -m "feat: add confidence-gated phase2 routing"
```

---

### Task 3: Surface routing observability through service and benchmark paths

**Files:**
- Modify: `phentrieve/text_processing/full_text_service.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `tests/unit/text_processing/test_full_text_service.py`
- Modify: `tests/unit/test_llm_benchmark.py`

- [x] **Step 1: Write failing observability tests**

Add tests asserting the new counters are surfaced:

```python
def test_run_llm_backend_surfaces_phase2_routing_counts(mocker) -> None:
    mocker.patch(
        "phentrieve.text_processing.full_text_service.run_llm_pipeline",
        return_value={
            "aggregated_hpo_terms": [],
            "meta": {
                "phase_counts": {
                    "phase2b_local_accept_count": 3,
                    "phase2b_deferred_count": 2,
                    "phase2b_no_candidate_skip_count": 1,
                }
            },
        },
    )

    result = run_llm_backend(...)

    assert result["observability"]["phase2b_local_accept_count"] == 3
    assert result["observability"]["phase2b_deferred_count"] == 2
    assert result["observability"]["phase2b_no_candidate_skip_count"] == 1
```

Add an equivalent benchmark test for exported observability.

- [x] **Step 2: Run the focused observability tests and confirm failure**

Run:

```bash
uv run pytest -n 0 tests/unit/text_processing/test_full_text_service.py -k "phase2_routing_counts" -v --no-cov
uv run pytest -n 0 tests/unit/test_llm_benchmark.py -k "phase2_routing_counts" -v --no-cov
```

Expected: FAIL on missing counters.

- [x] **Step 3: Implement the observability plumb-through**

Update service and benchmark observability extraction:

```python
"phase2b_local_accept_count": int(phase_counts.get("phase2b_local_accept_count", 0) or 0),
"phase2b_deferred_count": int(phase_counts.get("phase2b_deferred_count", 0) or 0),
"phase2b_no_candidate_skip_count": int(phase_counts.get("phase2b_no_candidate_skip_count", 0) or 0),
```

- [x] **Step 4: Run the focused tests and then the full affected slices**

Run:

```bash
uv run pytest -n 0 tests/unit/text_processing/test_full_text_service.py tests/unit/test_llm_benchmark.py -v --no-cov
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add phentrieve/text_processing/full_text_service.py phentrieve/benchmark/llm_benchmark.py tests/unit/text_processing/test_full_text_service.py tests/unit/test_llm_benchmark.py
git commit -m "feat: expose phase2 routing observability"
```

---

### Task 4: Add Phase-1 concurrency tests

**Files:**
- Modify: `tests/unit/llm/test_pipeline.py`
- Modify: `phentrieve/llm/pipeline.py`

- [x] **Step 1: Write failing tests for stable merge order under concurrent groups**

Add a test that simulates out-of-order group completion but requires stable
group-order merge semantics:

```python
def test_two_phase_pipeline_grouped_phase1_keeps_stable_merge_order(mocker) -> None:
    pipeline = TwoPhaseLLMPipeline(
        provider=FakeProvider(),
        tool_executor=FakeToolExecutor(),
        prompt_registry=FakePromptRegistry(),
    )
    calls: list[int] = []

    def fake_run(*, extraction_group, **kwargs):
        calls.append(extraction_group["group_index"])
        if extraction_group["group_index"] == 1:
            time.sleep(0.01)
        return (
            [{"phrase": f"group-{extraction_group['group_index']}", "category": "Abnormal", "chunk_ids": [extraction_group["group_index"] + 1], "evidence_text": "x"}],
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            1,
        )

    mocker.patch.object(pipeline, "_run_phase1_group", side_effect=fake_run)

    extracted, _, _, _, _ = pipeline._extract_phase1_grouped(...)

    assert [item["phrase"] for item in extracted] == ["group-0", "group-1"]
```

- [x] **Step 2: Add a failing test for partial failure accounting**

Add:

```python
def test_two_phase_pipeline_grouped_phase1_tracks_partial_failures_under_concurrency(mocker) -> None:
    ...
    assert trace[0]["status"] == "completed"
    assert trace[1]["status"] == "failed"
```

- [x] **Step 3: Run the targeted tests and confirm failure**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "stable_merge_order or partial_failures_under_concurrency" -v --no-cov
```

Expected: FAIL because grouped extraction is still serial or helper is missing.

- [x] **Step 4: Commit the red tests**

```bash
git add tests/unit/llm/test_pipeline.py
git commit -m "test: add grouped phase1 concurrency expectations"
```

---

### Task 5: Implement bounded Phase-1 concurrency

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `tests/unit/llm/test_pipeline.py`

- [x] **Step 1: Extract a per-group helper if needed**

Refactor the existing grouped Phase-1 body into a per-group helper that returns
indexed results:

```python
def _run_phase1_group(
    self,
    *,
    extraction_group: dict[str, Any],
    text: str,
    extraction_prompt,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    ...
```

- [x] **Step 2: Add bounded concurrency around the group helper**

Implement a small bounded-concurrency execution path:

```python
max_workers = min(len(extraction_groups), 2)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(
            self._run_phase1_group,
            extraction_group=group,
            text=text,
            extraction_prompt=extraction_prompt,
        ): group["group_index"]
        for group in extraction_groups
    }
    indexed_results = []
    for future in as_completed(futures):
        group_index = futures[future]
        indexed_results.append((group_index, future.result()))
indexed_results.sort(key=lambda item: item[0])
```

Preserve current trace semantics and accumulate usage/request counters exactly as
before.

- [x] **Step 3: Keep all-group failure behavior identical**

Make sure the existing error remains:

```python
if extraction_groups and not any_group_succeeded:
    raise LLMPipelinePhaseError(
        "phase1",
        "Structured extraction failed for all extraction groups",
    )
```

- [x] **Step 4: Run the targeted tests**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "stable_merge_order or partial_failures_under_concurrency" -v --no-cov
```

Expected: PASS.

- [x] **Step 5: Run the full pipeline slice**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -v --no-cov
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add phentrieve/llm/pipeline.py tests/unit/llm/test_pipeline.py
git commit -m "feat: parallelize grouped phase1 extraction"
```

---

### Task 6: Run repo verification and benchmark comparison

**Files:**
- Modify: none required unless verification fails
- Reference: `results/llm/pr216_grounded_benchmark_fix1.json`
- Reference: `results/llm/pr216_grounded_chunkrefactor.json`

- [x] **Step 1: Run repo verification**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected: all commands succeed.

- [x] **Step 2: Rerun the 10-doc grounded benchmark**

Run:

```bash
uv run phentrieve benchmark llm \
  --test-file tests/data/en/phenobert \
  --dataset GeneReviews \
  --llm-model gemini-2.5-flash \
  --llm-mode two_phase \
  --llm-internal-mode whole_document_grounded \
  --debug \
  --output results/llm/pr216_grounded_routing_phase1_parallel.json
```

Expected: benchmark completes with `10/10` documents.

- [x] **Step 3: Compare against the two baselines**

Run:

```bash
python - <<'PY'
import json
for name, path in [
    ("grounded_fix1", "results/llm/pr216_grounded_benchmark_fix1.json"),
    ("grounded_chunkrefactor", "results/llm/pr216_grounded_chunkrefactor.json"),
    ("routing_phase1_parallel", "results/llm/pr216_grounded_routing_phase1_parallel.json"),
]:
    with open(path) as f:
        data = json.load(f)
    metrics = data["metrics"]["assertion_aware"]
    timing = data["timing_breakdown"]
    usage = data["token_usage"]
    print(
        name,
        timing["wall_clock_seconds"],
        usage["api_calls"],
        usage["total_tokens"],
        metrics["micro"]["f1"],
    )
PY
```

Expected:

- API calls lower than `60`
- wall clock lower than `1051.52`
- micro F1 not below `0.7631`

- [x] **Step 4: Commit benchmark-output references if desired, then final code commit**

If code changed during verification fixes:

```bash
git add <updated files>
git commit -m "fix: finalize routing and phase1 concurrency verification"
```

If no code changed, do not create a no-op commit.

---

## Self-review

Spec coverage:

- Confidence-gated routing: Tasks 1 to 3
- Phase-1 grouped-call concurrency: Tasks 4 to 5
- Verification and benchmark comparison: Task 6

Placeholder scan:

- No `TODO` / `TBD`
- Every task contains concrete files, commands, and code targets

Type consistency:

- Routing helpers consistently use `phase2b_local_accept_count`,
  `phase2b_deferred_count`, and `phase2b_no_candidate_skip_count`
- Grouped Phase-1 helper and merged result order are referenced consistently
