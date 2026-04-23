# Shared Chunk Pipeline Internal Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current grounded whole-document LLM preprocessing path with an internal shared-chunk pipeline that preserves provenance, improves multilingual robustness, and reduces latency without adding a new user-visible mode.

**Architecture:** Keep the public `llm_internal_mode` surface unchanged, but internally route `whole_document_grounded` through the existing `TextProcessingPipeline`, group adjacent chunks into token-bounded extraction groups, run phase-1 extraction per group, then aggregate into the existing retrieval and mapping stages with stronger observability and duplicate suppression. Preserve `whole_document_legacy` as the control path for benchmark comparison.

**Tech Stack:** Python 3.10+, Ruff, mypy, pytest, Gemini API (`count_tokens`, structured outputs, implicit caching constraints), existing `TextProcessingPipeline`, `BioLORD` embeddings, benchmark harness in `phentrieve/benchmark/`.

---

## Why This Plan

### Verified repo-local evidence

- Current fixed grounded mode is stable but still slower than legacy:
  - fixed grounded: `744.10s`, micro F1 `0.7682`, macro F1 `0.7780`
  - legacy: `668.23s`, micro F1 `0.7629`, macro F1 `0.7918`
- Remaining latency is dominated by remote Gemini time, especially `phase1` and
  `phase2b_llm`.
- The hard failure on `GeneReviews_NBK1277` was fixed by removing prompt
  duplication and raising grounded phase-1 output headroom, but that did not
  remove the architectural whole-document bottleneck.

### External best-practice constraints

- Gemini structured outputs are intended for extraction/classification tasks and
  should use explicit schemas and application-side validation.
  Source: Google Gemini structured outputs docs,
  https://ai.google.dev/gemini-api/docs/structured-output
- Gemini token counting should use `count_tokens` before inference rather than
  character heuristics.
  Source: Google token guide,
  https://ai.google.dev/gemini-api/docs/tokens
- Gemini implicit caching works best when large shared prompt content is kept at
  the beginning and request-specific content is appended later.
  Source: Google caching docs,
  https://ai.google.dev/gemini-api/docs/caching/
- Gemini 2.5 Flash supports structured outputs, caching, batch API, and up to
  `65,536` output tokens.
  Source: Google models docs,
  https://ai.google.dev/gemini-api/docs/models
- Batch APIs are the right fit for large offline evaluations, but they are a
  separate concern from fixing the synchronous architecture.
  Sources:
  - Google rate limits / batch docs,
    https://ai.google.dev/gemini-api/docs/rate-limits
  - OpenAI Batch API guide,
    https://developers.openai.com/api/docs/guides/batch

## Scope And Non-Goals

### In scope

- internal preprocessing refactor only
- no new CLI or API mode names
- shared chunk pipeline reused by both live service and benchmark paths
- group-level phase-1 extraction with failure isolation
- provenance-preserving aggregation into the existing two-phase pipeline
- latency work focused on duplicate suppression and smaller LLM mapping payloads
- stronger benchmark observability

### Out of scope

- replacing the provider
- switching to an asynchronous provider path
- adding a user-visible chunking mode for the LLM backend
- introducing arbitrary phenotype caps
- moving offline benchmark execution to Batch API in the same PR

## File Map

### Create

- `phentrieve/llm/preprocessing.py`
  - internal shared preprocessing module for grounded chunks and extraction
    groups
- `tests/unit/llm/test_preprocessing.py`
  - unit coverage for chunk grouping, token budgeting, overlap, and provenance

### Modify

- `phentrieve/text_processing/full_text_service.py`
  - replace direct `_build_grounded_chunks()` usage with shared internal
    preprocessing entrypoint
- `phentrieve/benchmark/llm_benchmark.py`
  - benchmark path uses the same preprocessing contract as live service and
    records group-level observability
- `phentrieve/llm/pipeline.py`
  - phase 1 consumes extraction groups, aggregates group-level outputs, and
    isolates failures
- `phentrieve/llm/types.py`
  - add typed structures for grounded chunks / extraction groups if keeping them
    out of raw `dict[str, Any]`
- `phentrieve/llm/provider.py`
  - optional support for reusing count-token or client helpers if needed by the
    new preprocessing path
- `tests/unit/text_processing/test_full_text_service.py`
  - update expectations from "single grounded chunk list" to shared preprocessing
    output
- `tests/unit/llm/test_pipeline.py`
  - add per-group phase-1, aggregation, and failure-isolation tests
- `tests/integration/llm/test_grounded_pipeline_integration.py`
  - extend provenance expectations to grouped chunk extraction
- `tests/integration/test_benchmark_workflow.py`
  - benchmark artifact and trace expectations for the new observability fields

## Design Decisions To Lock In

1. Keep `whole_document_grounded` as the public/internal mode string.
   Internally, it stops meaning "single whole-note phase-1 request" and instead
   means "shared chunk pipeline with grouped grounded extraction".

2. Add an internal preprocessing layer with two outputs:
   - `grounded_chunks`: canonical chunk/span provenance from
     `TextProcessingPipeline`
   - `extraction_groups`: adjacent chunk bundles sized by real token count

3. Run phase-1 extraction once per extraction group, not once per document.
   Aggregation happens before retrieval/mapping.

4. Treat phase-1 failures as partial document failures with explicit trace
   entries, never as silent empty extraction.

5. Preserve legacy mode as the benchmark control path until the new grounded
   path consistently outperforms it on the target metrics.

## Success Criteria

- No benchmark document fails in grounded mode on the 10-doc GeneReviews slice.
- Grounded mode remains at or above the current fixed grounded micro F1 minus
  `0.01`.
- Wall-clock time improves materially from `744.10s`; target band is
  `<= 700s`.
- Benchmark artifacts expose enough per-document detail to explain regressions
  without scraping debug logs.
- `make check`, `make typecheck-fresh`, and `make test` pass.

## Task 1: Add Shared Preprocessing Types And Group Builder

**Files:**
- Create: `phentrieve/llm/preprocessing.py`
- Modify: `phentrieve/llm/types.py`
- Test: `tests/unit/llm/test_preprocessing.py`

- [ ] **Step 1: Write failing unit tests for grouped preprocessing**

Add tests covering:

```python
def test_build_extraction_groups_preserves_chunk_ids_and_positions() -> None:
    ...


def test_build_extraction_groups_respects_token_budget() -> None:
    ...


def test_build_extraction_groups_keeps_adjacent_context_overlap_small() -> None:
    ...
```

Assertions must verify:
- groups are ordered
- chunk ids are preserved exactly
- group text is derived from canonical chunk text, not regenerated sentences
- no chunk appears outside its allowed neighbor-overlap window

- [ ] **Step 2: Run the new unit tests and verify failure**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_preprocessing.py -v --no-cov
```

Expected: failure because `phentrieve.llm.preprocessing` and the tested entry
points do not exist yet.

- [ ] **Step 3: Add minimal preprocessing module and types**

Create typed structures along these lines:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundedChunk:
    chunk_id: int
    text: str
    start_char: int | None
    end_char: int | None
    status: str


@dataclass(frozen=True)
class ExtractionGroup:
    group_id: int
    chunk_ids: list[int]
    text: str
    estimated_prompt_tokens: int
```

And implement these entry points:

```python
def build_grounded_chunks_from_text_pipeline(...) -> list[GroundedChunk]:
    ...


def build_extraction_groups(
    *,
    grounded_chunks: list[GroundedChunk],
    provider,
    system_prompt: str,
    max_prompt_tokens: int,
    neighbor_overlap: int = 1,
) -> list[ExtractionGroup]:
    ...
```

Use the provider `count_tokens()` API for final budgeting decisions instead of
relying on `chars // 4`.

- [ ] **Step 4: Run the preprocessing unit tests and make them pass**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_preprocessing.py -v --no-cov
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/preprocessing.py phentrieve/llm/types.py tests/unit/llm/test_preprocessing.py
git commit -m "feat: add llm shared preprocessing groups"
```

## Task 2: Route Live Service Through Shared Preprocessing

**Files:**
- Modify: `phentrieve/text_processing/full_text_service.py`
- Test: `tests/unit/text_processing/test_full_text_service.py`

- [ ] **Step 1: Write failing service tests for shared preprocessing**

Add tests for:

```python
def test_run_llm_backend_uses_shared_preprocessing_for_grounded_mode(mocker) -> None:
    ...


def test_run_llm_backend_logs_group_preflight_details(mocker, caplog) -> None:
    ...
```

Assertions must verify:
- grounded mode calls the new preprocessing entrypoint once
- pipeline `run()` receives both canonical chunks and extraction groups
- preflight logging includes chunk count, group count, and token counts

- [ ] **Step 2: Run those tests and verify failure**

Run:

```bash
uv run pytest -n 0 tests/unit/text_processing/test_full_text_service.py -k "shared_preprocessing or group_preflight" -v --no-cov
```

Expected: failure because `run_llm_backend()` still constructs only a flat
`grounded_chunks` list.

- [ ] **Step 3: Implement service-side integration**

Refactor `run_llm_backend()` so grounded mode does this:

```python
preprocessed = preprocess_grounded_document(
    text=text,
    language=language,
    provider=provider,
    extraction_prompt=extraction_prompt,
    retrieval_model_name=retrieval_model_name,
    ...
)
```

and passes:

```python
pipeline.run(
    text=text,
    grounded_chunks=preprocessed.grounded_chunks,
    extraction_groups=preprocessed.extraction_groups,
    config=config,
)
```

Do not change the public function signature or mode names.

- [ ] **Step 4: Run the focused service tests**

Run:

```bash
uv run pytest -n 0 tests/unit/text_processing/test_full_text_service.py -k "shared_preprocessing or group_preflight" -v --no-cov
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add phentrieve/text_processing/full_text_service.py tests/unit/text_processing/test_full_text_service.py
git commit -m "refactor: route llm backend through shared preprocessing"
```

## Task 3: Teach `TwoPhaseLLMPipeline` To Consume Extraction Groups

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Test: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests for grouped phase-1 extraction**

Add tests for:

```python
def test_phase1_runs_once_per_extraction_group() -> None:
    ...


def test_grouped_phase1_aggregates_mentions_before_retrieval() -> None:
    ...


def test_grouped_phase1_keeps_group_chunk_ids_in_trace() -> None:
    ...
```

Assertions must verify:
- provider structured calls match the number of extraction groups
- extracted mentions are merged into one actionable list before phase 2A
- trace output retains `group_id` and source `chunk_ids`

- [ ] **Step 2: Run the focused pipeline tests and verify failure**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "extraction_group or grouped_phase1" -v --no-cov
```

Expected: failure because `TwoPhaseLLMPipeline.run()` has no group-aware phase-1
path.

- [ ] **Step 3: Implement grouped phase-1 execution**

Adjust the internal pipeline contract to support:

```python
def run(
    *,
    text: str,
    grounded_chunks: list[dict[str, Any]] | None = None,
    extraction_groups: list[dict[str, Any]] | None = None,
    config: LLMPipelineConfig,
) -> LLMExtractionResult:
    ...
```

Implementation rules:
- if `extraction_groups` is present, iterate groups in order
- render each phase-1 prompt from the group-local chunk index only
- collect per-group token usage and timings
- aggregate extracted mentions into one document-level list before retrieval

- [ ] **Step 4: Run the focused grouped-phase tests**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "extraction_group or grouped_phase1" -v --no-cov
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/pipeline.py tests/unit/llm/test_pipeline.py
git commit -m "feat: add grouped phase1 extraction path"
```

## Task 4: Add Failure Isolation And Partial-Failure Tracing

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Test: `tests/unit/llm/test_pipeline.py`
- Test: `tests/integration/test_benchmark_workflow.py`

- [ ] **Step 1: Write failing tests for per-group failure isolation**

Add coverage for:

```python
def test_group_failure_is_recorded_without_zeroing_document() -> None:
    ...


def test_benchmark_trace_persists_group_failures() -> None:
    ...
```

Assertions must verify:
- one failed extraction group does not erase successful groups
- trace includes failed group ids, error type, and surviving extracted mentions
- benchmark artifacts record partial failure counts

- [ ] **Step 2: Run the focused failure tests and verify failure**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "group_failure" tests/integration/test_benchmark_workflow.py -k "group_failures" -v --no-cov
```

Expected: failure because phase-1 failures are not yet represented at group
granularity.

- [ ] **Step 3: Implement failure isolation**

Rules:
- benchmark mode records partial failures explicitly in trace/meta
- live mode logs and continues when at least one group succeeds
- full-document failure is raised only when all extraction groups fail

Add trace fields such as:

```python
trace["phase1"]["groups"] = [
    {
        "group_id": 1,
        "chunk_ids": [1, 2, 3],
        "status": "completed",
        "extracted_count": 7,
    },
    {
        "group_id": 2,
        "chunk_ids": [4, 5],
        "status": "failed",
        "error": "Structured extraction failed",
    },
]
```

- [ ] **Step 4: Run the focused failure tests**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "group_failure" tests/integration/test_benchmark_workflow.py -k "group_failures" -v --no-cov
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/pipeline.py phentrieve/benchmark/llm_benchmark.py tests/unit/llm/test_pipeline.py tests/integration/test_benchmark_workflow.py
git commit -m "feat: isolate grouped phase1 failures"
```

## Task 5: Suppress Duplicate Work Before Retrieval And Mapping

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Test: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write failing tests for duplicate phrase suppression**

Add tests for:

```python
def test_grouped_mentions_deduplicate_before_retrieval() -> None:
    ...


def test_unresolved_mapping_batches_skip_duplicate_phrase_candidate_sets() -> None:
    ...
```

Assertions must verify:
- repeated extracted phrases from overlapping groups do not trigger duplicate
  retrieval queries
- repeated unresolved mapping items with the same phrase/category/candidate set
  are mapped once and fan out to all matching mentions

- [ ] **Step 2: Run the focused duplicate-suppression tests and verify failure**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "deduplicate_before_retrieval or skip_duplicate_phrase_candidate_sets" -v --no-cov
```

Expected: failure because grouped extraction will initially multiply repeated
mentions.

- [ ] **Step 3: Implement duplicate suppression**

Use stable keys such as:

```python
dedupe_key = (
    normalized_phrase,
    normalized_category,
    tuple(sorted(candidate_ids)),
)
```

Apply suppression in two places:
- before phase 2A retrieval
- before phase 2B LLM mapping

Fan out the mapped result back to all originating mentions so provenance is not
lost.

- [ ] **Step 4: Run the focused duplicate-suppression tests**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "deduplicate_before_retrieval or skip_duplicate_phrase_candidate_sets" -v --no-cov
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/pipeline.py tests/unit/llm/test_pipeline.py
git commit -m "perf: suppress duplicate llm retrieval and mapping work"
```

## Task 6: Compact Mapping Payloads And Preserve Cache-Friendly Prefixes

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `phentrieve/llm/prompts/loader.py` or template files if required
- Test: `tests/unit/llm/test_pipeline.py`
- Test: `tests/unit/llm/test_prompts.py`

- [ ] **Step 1: Write failing tests for compact mapping payload construction**

Add tests for:

```python
def test_mapping_prompt_uses_compact_grounded_context() -> None:
    ...


def test_mapping_prompt_prefix_stays_stable_before_variable_context() -> None:
    ...
```

Assertions must verify:
- mapping payload includes only the minimal grounded context needed for
  disambiguation
- repeated static instructions stay at the beginning of the prompt
- variable phrase/candidate content is appended later for better implicit cache
  eligibility

- [ ] **Step 2: Run the focused prompt tests and verify failure**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "compact_grounded_context" tests/unit/llm/test_prompts.py -k "stable_before_variable_context" -v --no-cov
```

Expected: failure because mapping payloads still reflect the current, more
verbose structure.

- [ ] **Step 3: Implement compact mapping payloads**

Use only the fields that materially help disambiguation:

```python
{
    "primary_chunk_text": ...,
    "neighbor_chunk_text": ...,
    "phrase": ...,
    "category": ...,
    "candidates": [...],
}
```

Do not repeat full document text or nonessential metadata in phase 2B prompts.

- [ ] **Step 4: Run the focused prompt tests**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_pipeline.py -k "compact_grounded_context" tests/unit/llm/test_prompts.py -k "stable_before_variable_context" -v --no-cov
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/pipeline.py phentrieve/llm/prompts/loader.py phentrieve/llm/prompts/templates/two_phase/en.yaml tests/unit/llm/test_pipeline.py tests/unit/llm/test_prompts.py
git commit -m "perf: compact llm mapping payloads"
```

## Task 7: Unify Benchmark And Live Observability

**Files:**
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `phentrieve/text_processing/full_text_service.py`
- Test: `tests/integration/test_benchmark_workflow.py`

- [ ] **Step 1: Write failing benchmark observability tests**

Add tests for:

```python
def test_benchmark_artifact_persists_group_counts_and_phase1_group_timings(tmp_path, monkeypatch) -> None:
    ...


def test_benchmark_record_includes_partial_failure_counts(tmp_path, monkeypatch) -> None:
    ...
```

Assertions must verify persisted metrics for:
- grounded chunk count
- extraction group count
- per-group phase-1 timings
- partial failure count
- deduplicated retrieval / mapping counts

- [ ] **Step 2: Run the focused benchmark tests and verify failure**

Run:

```bash
uv run pytest -n 0 tests/integration/test_benchmark_workflow.py -k "group_counts or partial_failure_counts" -v --no-cov
```

Expected: failure because current benchmark artifacts only expose document-level
phase timings.

- [ ] **Step 3: Implement benchmark observability**

Persist additional fields into prediction records and traces:

```python
{
    "observability": {
        "grounded_chunks": ...,
        "extraction_groups": ...,
        "failed_groups": ...,
        "deduplicated_phase1_mentions": ...,
        "deduplicated_unresolved_mappings": ...,
    }
}
```

- [ ] **Step 4: Run the focused benchmark observability tests**

Run:

```bash
uv run pytest -n 0 tests/integration/test_benchmark_workflow.py -k "group_counts or partial_failure_counts" -v --no-cov
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add phentrieve/benchmark/llm_benchmark.py phentrieve/text_processing/full_text_service.py tests/integration/test_benchmark_workflow.py
git commit -m "feat: add grouped llm benchmark observability"
```

## Task 8: Multilingual And End-To-End Verification

**Files:**
- Modify: `tests/integration/llm/test_grounded_pipeline_integration.py`
- Modify: `tests/unit/cli/test_text_commands.py` if call signatures changed

- [ ] **Step 1: Add failing multilingual integration tests**

Add or extend tests that prove:

```python
def test_grounded_llm_pipeline_grouped_path_preserves_english_provenance() -> None:
    ...


def test_grounded_llm_pipeline_grouped_path_preserves_german_provenance() -> None:
    ...
```

Assertions must verify:
- grouped extraction still resolves correct source chunk ids
- neighbor overlap does not corrupt German provenance
- final grounded context remains chunk-derived, not sentence-reconstructed

- [ ] **Step 2: Run the multilingual integration tests and verify failure**

Run:

```bash
uv run pytest -n 0 tests/integration/llm/test_grounded_pipeline_integration.py -v --no-cov
```

Expected: failure until grouped-path provenance is wired through all phases.

- [ ] **Step 3: Make integration tests pass**

Touch whichever of these are needed:

```bash
phentrieve/llm/pipeline.py
phentrieve/llm/preprocessing.py
tests/integration/llm/test_grounded_pipeline_integration.py
tests/unit/cli/test_text_commands.py
```

Ensure no public mode strings or CLI options change.

- [ ] **Step 4: Run focused verification**

Run:

```bash
uv run pytest -n 0 tests/unit/llm/test_preprocessing.py tests/unit/llm/test_pipeline.py tests/unit/llm/test_prompts.py tests/unit/text_processing/test_full_text_service.py tests/integration/llm/test_grounded_pipeline_integration.py tests/integration/test_benchmark_workflow.py --no-cov
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/integration/llm/test_grounded_pipeline_integration.py tests/unit/cli/test_text_commands.py
git commit -m "test: cover grouped grounded llm path"
```

## Task 9: Full Verification And Benchmark Comparison

**Files:**
- No new code expected unless verification reveals a regression

- [ ] **Step 1: Run repo verification**

Run:

```bash
make check
make typecheck-fresh
make test
```

Expected:
- Ruff clean
- mypy clean
- pytest passes with coverage gate

- [ ] **Step 2: Run the fixed benchmark control and refactor benchmark**

Run:

```bash
bash -lc 'set -a && source .env && set +a && uv run phentrieve benchmark llm --test-file tests/data/en/phenobert --dataset GeneReviews --llm-model gemini-2.5-flash --llm-mode two_phase --llm-internal-mode whole_document_legacy --debug --output-path results/llm/pr216_legacy_control_postrefactor.json'
```

Run:

```bash
bash -lc 'set -a && source .env && set +a && uv run phentrieve benchmark llm --test-file tests/data/en/phenobert --dataset GeneReviews --llm-model gemini-2.5-flash --llm-mode two_phase --llm-internal-mode whole_document_grounded --debug --output-path results/llm/pr216_grounded_chunkrefactor.json'
```

Expected:
- both complete without document failure
- grounded mode remains within the target quality band
- grounded wall-clock improves over `744.10s`

- [ ] **Step 3: Compare metrics and record outcome**

Compare against:

```text
results/llm/pr216_grounded_benchmark_fix1.json
results/llm/llm_benchmark_20260416T_full_genereviews_phrasefix.json
results/llm/pr216_grounded_chunkrefactor.json
results/llm/pr216_legacy_control_postrefactor.json
```

Record:
- micro / macro / weighted metrics
- wall-clock and avg/case
- prompt / completion / total tokens
- API calls
- worst three document timings
- whether `phase2b_llm` remains dominant

- [ ] **Step 4: Commit final verification or follow-up fix**

If no code changed during verification:

```bash
git commit --allow-empty -m "chore: verify grouped llm refactor"
```

If verification required code changes, commit them with a specific fix message.

## Plan Self-Review

### Spec coverage

- shared chunk pipeline as internal source of truth: Tasks 1-3
- failure isolation and observability: Tasks 4 and 7
- latency-first duplicate suppression and compact prompts: Tasks 5 and 6
- multilingual provenance validation: Task 8
- benchmark and repo verification: Task 9

### Placeholder scan

- no `TODO` / `TBD`
- every task names exact files
- every verification step includes commands
- every performance claim is anchored to an existing artifact or a measurable
  target

### Type consistency

- `GroundedChunk`, `ExtractionGroup`, and `preprocess_grounded_document(...)`
  are defined before they are consumed
- `pipeline.run(..., extraction_groups=...)` is introduced before benchmark and
  service tasks rely on it

## Notes For The Implementer

- Do not add a new public `llm_internal_mode` value for this refactor.
- Do not remove `whole_document_legacy`; it is required as the benchmark
  control.
- Do not reintroduce sentence reconstruction or full-note duplication into
  grounded prompts.
- Prefer one conservative overlap parameter and real token counts over clever
  heuristics.
- If chunk grouping increases duplicate mentions, fix it in aggregation rather
  than by lowering extraction recall.
