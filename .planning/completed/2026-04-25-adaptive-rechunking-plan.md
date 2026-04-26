# Adaptive Re-Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--adaptive-rechunking` mechanism that detects per-chunk retrieval quality and, when poor, subdivides the chunk into sentence-bounded sub-chunks, re-queries them, and merges results. Resolves issue #148.

**Architecture:** New module `phentrieve/retrieval/adaptive_rechunker.py` plugs into `run_standard_backend()` between `orchestrate_hpo_extraction()` and `adapt_standard_response()`. Two upstream changes: `orchestrate_hpo_extraction` returns an `OrchestrationResult` dataclass (with `__iter__` for backward-compat unpacking) including `raw_query_results` as a third field; it also accepts an optional `precomputed_query_results` parameter so re-aggregation skips the retrieval call. Trigger reads from raw query output (full top-K, including matches below `chunk_retrieval_threshold`), uses score-AND-margin conjunction. Sub-chunks are sentence-window groupings via the existing `SentenceChunker`. Cost model: 1 + N `query_batch` calls at depth N. Aggregator unchanged — the `max_score` filter switch from earlier drafts is moved to Future work.

**Tech Stack:** Python 3.10+, pydantic v2 (config), Typer (CLI flags), existing `SentenceChunker` for subdivision, ChromaDB-backed `DenseRetriever`, pytest, FastAPI (API parity), Vue 3 (minimal frontend pass-through).

---

## Spec Reference

Implementation strictly follows `.planning/specs/2026-04-25-adaptive-rechunking-spec.md`. When in doubt, the spec is canonical. **Plan A (`2026-04-25-cli-profiles-default-resolution-plan.md`) should land first** — Plan B's profile integration depends on it. The CLI flags in Plan B work standalone if Plan A hasn't landed yet.

## Command Contract

```bash
# Disabled (default) — no behavior change.
phentrieve text process note.txt

# Enabled via flag, all knobs at defaults.
phentrieve text process note.txt --adaptive-rechunking

# Tune thresholds.
phentrieve text process note.txt --adaptive-rechunking \
  --adaptive-rechunking-quality-threshold 0.5 \
  --adaptive-rechunking-margin-threshold 0.05 \
  --adaptive-rechunking-max-depth 1

# Via profile (after Plan A lands).
phentrieve text process note.txt --profile high_recall_with_adaptive
```

YAML configuration:

```yaml
extraction:
  adaptive_rechunking:
    enabled: false
    quality_threshold: 0.55
    margin_threshold: 0.03
    max_depth: 2
    min_chunk_chars: 30
    max_sentences_per_subchunk: 3
    overlap_sentences: 1
    score_improvement_gate: 0.05
```

API:

```json
POST /api/text/process
{
  "text": "...",
  "adaptive_rechunking": {
    "enabled": true,
    "quality_threshold": 0.5
  }
}
```

Response includes `meta.adaptive_rechunking` block when enabled.

## File Structure

Create:

- `phentrieve/retrieval/adaptive_rechunker.py` — `AdaptiveRechunkingConfig`, `ChunkQualitySignals`, `AdaptiveRechunkingResult`, `assess_chunk_quality`, `subdivide_parent_chunk`, `apply_score_improvement_gate`, `run_adaptive_rechunking`, `dump_quality_report`.
- `phentrieve/text_processing/orchestration_result.py` — the new `OrchestrationResult` dataclass with `__iter__` for legacy 2-tuple unpacking.
- `tests/unit/retrieval/adaptive_rechunker/__init__.py`
- `tests/unit/retrieval/adaptive_rechunker/test_quality_assessment.py`
- `tests/unit/retrieval/adaptive_rechunker/test_raw_score_access.py`
- `tests/unit/retrieval/adaptive_rechunker/test_subdivide_parent_chunk.py`
- `tests/unit/retrieval/adaptive_rechunker/test_score_improvement_gate.py`
- `tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py`
- `tests/unit/retrieval/adaptive_rechunker/test_adaptive_config.py`
- `tests/unit/text_processing/test_orchestration_result_legacy_unpack.py`
- `tests/unit/text_processing/test_orchestrate_with_precomputed.py`
- `tests/unit/text_processing/test_adapt_standard_response_extra_meta.py`
- `tests/integration/test_adaptive_rechunking_e2e.py`
- `tests/integration/test_adaptive_rechunking_api.py`
- `tests/integration/test_adaptive_rechunking_benchmark.py`
- `tests/integration/test_adaptive_rechunking_call_count.py`
- `tests/integration/test_adaptive_rechunking_perf.py`
- `tests/integration/test_specA_specB_interaction.py`
- `tests/fixtures/adaptive_rechunking/synthetic_multi_finding.txt`
- `tests/fixtures/adaptive_rechunking/expected_terms_with_adaptive.json`
- `docs/user-guide/adaptive-rechunking.md`

Modify:

- `phentrieve/text_processing/hpo_extraction_orchestrator.py` — return `OrchestrationResult` dataclass; accept optional `precomputed_query_results` parameter.
- `phentrieve/text_processing/full_text_service.py` — `run_standard_backend()` consumes the new dataclass; calls `run_adaptive_rechunking` when `adaptive_rechunking.enabled`; threads `extra_meta` through `adapt_standard_response`. `adapt_standard_response` gains `extra_meta: dict | None = None` parameter that merges into `meta`.
- `phentrieve/cli/text_commands.py` — add four CLI flags and YAML/profile resolution into `AdaptiveRechunkingConfig`; pass to `run_full_text_service`.
- `api/schemas/text_processing_schemas.py` — `TextProcessingRequest.adaptive_rechunking: AdaptiveRechunkingProfileBlock | None`.
- `api/routers/text_processing_router.py` — pass `adaptive_rechunking` through to `run_full_text_service`.
- `phentrieve.yaml` and `phentrieve.yaml.template` — add commented `extraction.adaptive_rechunking:` block.
- `frontend/src/services/PhentrieveService.js` — payload pass-through for `adaptive_rechunking`.
- `frontend/src/test/services/PhentrieveService.test.js` — extend with pass-through smoke test.
- `docs/user-guide/cli-usage.md` — document the four new CLI flags.
- `docs/user-guide/api-usage.md` — document the new request field and response meta block.
- `docs/user-guide/text-processing-guide.md` — cross-reference the new page.
- `docs/user-guide/index.md` — link to adaptive-rechunking.md.
- `README.md` — one-line teaser.
- `CHANGELOG.md` — feature-addition entry.

---

## Phase 1: OrchestrationResult dataclass (foundation)

This phase changes `orchestrate_hpo_extraction`'s return type without breaking any existing call site.

### Task 1: Create `OrchestrationResult` dataclass

**Files:**
- Create: `phentrieve/text_processing/orchestration_result.py`
- Create: `tests/unit/text_processing/test_orchestration_result_legacy_unpack.py`

- [ ] **Step 1: Write failing test for legacy unpack and attribute access**

Create `tests/unit/text_processing/test_orchestration_result_legacy_unpack.py`:

```python
"""Tests that OrchestrationResult supports both legacy 2-tuple unpacking
and modern attribute access. Legacy unpacking is what keeps existing call
sites working."""

import pytest


class TestOrchestrationResult:
    def test_legacy_2_tuple_unpack_works(self):
        from phentrieve.text_processing.orchestration_result import OrchestrationResult

        result = OrchestrationResult(
            aggregated_results=[{"id": "HP:0001"}],
            chunk_results=[{"chunk_idx": 0}],
            raw_query_results=[{"similarities": [[0.9]]}],
        )
        # The legacy call sites do this:
        agg, chunks = result
        assert agg == [{"id": "HP:0001"}]
        assert chunks == [{"chunk_idx": 0}]

    def test_attribute_access_works(self):
        from phentrieve.text_processing.orchestration_result import OrchestrationResult

        result = OrchestrationResult(
            aggregated_results=[],
            chunk_results=[],
            raw_query_results=[{"similarities": [[]]}],
        )
        assert result.aggregated_results == []
        assert result.chunk_results == []
        assert result.raw_query_results == [{"similarities": [[]]}]

    def test_indexing_returns_legacy_2_tuple_elements(self):
        from phentrieve.text_processing.orchestration_result import OrchestrationResult

        result = OrchestrationResult(
            aggregated_results=[1, 2],
            chunk_results=[3, 4],
            raw_query_results=[5, 6],
        )
        # Some call sites may index instead of unpack.
        assert result[0] == [1, 2]
        assert result[1] == [3, 4]

    def test_iteration_yields_2_tuple(self):
        from phentrieve.text_processing.orchestration_result import OrchestrationResult

        result = OrchestrationResult([1], [2], [3])
        assert list(result) == [[1], [2]]

    def test_immutable(self):
        from phentrieve.text_processing.orchestration_result import OrchestrationResult

        result = OrchestrationResult([], [], [])
        with pytest.raises(AttributeError):
            result.aggregated_results = []  # frozen=True
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/text_processing/test_orchestration_result_legacy_unpack.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create the dataclass**

Create `phentrieve/text_processing/orchestration_result.py`:

```python
"""Return type for orchestrate_hpo_extraction.

Implements __iter__ and __getitem__ to preserve legacy 2-tuple unpacking
(aggregated_results, chunk_results) while exposing raw_query_results as a
new field for callers that need access to unfiltered retrieval scores
(specifically the adaptive rechunker).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class OrchestrationResult:
    """Return value of orchestrate_hpo_extraction.

    Backward compatibility: iteration and indexing yield the legacy 2-tuple
    `(aggregated_results, chunk_results)`. Attribute access exposes
    `raw_query_results` for new callers.
    """

    aggregated_results: list[dict[str, Any]]
    chunk_results: list[dict[str, Any]]
    raw_query_results: list[dict[str, Any]] = field(default_factory=list)

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        """Yield (aggregated_results, chunk_results) for legacy 2-tuple unpack."""
        yield self.aggregated_results
        yield self.chunk_results

    def __getitem__(self, idx: int) -> list[dict[str, Any]]:
        """Index 0 -> aggregated_results, 1 -> chunk_results. For legacy callers."""
        if idx == 0:
            return self.aggregated_results
        if idx == 1:
            return self.chunk_results
        raise IndexError(f"OrchestrationResult index {idx} out of range (0..1)")

    def __len__(self) -> int:
        return 2  # iteration yields 2 elements
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/text_processing/test_orchestration_result_legacy_unpack.py -v`
Expected: PASS, five tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/text_processing/orchestration_result.py tests/unit/text_processing/test_orchestration_result_legacy_unpack.py
git commit -m "feat(text-processing): add OrchestrationResult dataclass

Frozen dataclass with __iter__ and __getitem__ that yield the legacy
2-tuple (aggregated_results, chunk_results). raw_query_results is exposed
as a new attribute for callers that need unfiltered retrieval scores
(adaptive rechunker, Plan B). Existing call sites continue to work
unchanged via legacy unpacking."
```

### Task 2: Refactor `orchestrate_hpo_extraction` to return the dataclass

**Files:**
- Modify: `phentrieve/text_processing/hpo_extraction_orchestrator.py`
- Create: `tests/unit/text_processing/test_orchestrate_with_precomputed.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/text_processing/test_orchestrate_with_precomputed.py`:

```python
"""Tests for the new precomputed_query_results parameter and
OrchestrationResult return type on orchestrate_hpo_extraction."""

from unittest.mock import MagicMock

import pytest

from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)
from phentrieve.text_processing.orchestration_result import OrchestrationResult


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    # Standard query_batch return shape.
    r.query_batch.return_value = [
        {
            "ids": [["HP:0001250"]],
            "metadatas": [[{"id": "HP:0001250", "label": "Seizure"}]],
            "similarities": [[0.85]],
            "distances": [[0.15]],
            "documents": [[""]],
        }
    ]
    return r


def test_returns_orchestration_result(mock_retriever):
    result = orchestrate_hpo_extraction(
        text_chunks=["seizures"], retriever=mock_retriever, num_results_per_chunk=1
    )
    assert isinstance(result, OrchestrationResult)
    assert result.aggregated_results
    assert result.chunk_results
    assert result.raw_query_results  # populated from query_batch


def test_legacy_unpack_still_works(mock_retriever):
    aggregated, chunks = orchestrate_hpo_extraction(
        text_chunks=["seizures"], retriever=mock_retriever, num_results_per_chunk=1
    )
    assert isinstance(aggregated, list)
    assert isinstance(chunks, list)


def test_precomputed_skips_retrieval(mock_retriever):
    raw = [
        {
            "ids": [["HP:0001"]],
            "metadatas": [[{"id": "HP:0001", "label": "Foo"}]],
            "similarities": [[0.9]],
            "distances": [[0.1]],
            "documents": [[""]],
        }
    ]
    result = orchestrate_hpo_extraction(
        text_chunks=["any text"],
        retriever=mock_retriever,
        num_results_per_chunk=1,
        precomputed_query_results=raw,
    )
    # query_batch should NOT have been called.
    assert mock_retriever.query_batch.call_count == 0
    assert result.raw_query_results == raw
    assert any(t["id"] == "HP:0001" for t in result.aggregated_results)


def test_precomputed_value_drives_aggregation(mock_retriever):
    raw = [
        {
            "ids": [["HP:0002"]],
            "metadatas": [[{"id": "HP:0002", "label": "Bar"}]],
            "similarities": [[0.95]],
            "distances": [[0.05]],
            "documents": [[""]],
        }
    ]
    # mock_retriever would have returned HP:0001250 (from fixture) if called,
    # but precomputed should drive aggregation entirely.
    result = orchestrate_hpo_extraction(
        text_chunks=["unused"],
        retriever=mock_retriever,
        num_results_per_chunk=1,
        precomputed_query_results=raw,
    )
    ids = [t["id"] for t in result.aggregated_results]
    assert "HP:0002" in ids
    assert "HP:0001250" not in ids  # precomputed wins
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/text_processing/test_orchestrate_with_precomputed.py -v`
Expected: FAIL — no `precomputed_query_results` parameter, return type is tuple.

- [ ] **Step 3: Modify `hpo_extraction_orchestrator.py`**

Edit `phentrieve/text_processing/hpo_extraction_orchestrator.py`:

1. Add import at the top:

```python
from phentrieve.text_processing.orchestration_result import OrchestrationResult
```

2. Change the signature of `orchestrate_hpo_extraction` to accept the new parameter and return the new type:

```python
def orchestrate_hpo_extraction(
    text_chunks: list[str],
    retriever: DenseRetriever,
    num_results_per_chunk: int = 10,
    chunk_retrieval_threshold: float = 0.3,
    language: str = "en",
    top_term_per_chunk: bool = False,
    min_confidence_for_aggregated: float = 0.0,
    assertion_statuses: list[str | None] | None = None,
    include_details: bool = False,
    precomputed_query_results: list[dict] | None = None,   # NEW
) -> OrchestrationResult:                                   # was tuple[list, list]
```

3. Update the function body to use the precomputed results when supplied:

Replace the existing block at line 60-67:

```python
# OPTIMIZATION: Query all chunks at once using batch API (10-20x faster!)
logger.info(f"Batch querying {len(text_chunks)} chunks at once")
all_query_results = retriever.query_batch(
    texts=text_chunks,
    n_results=num_results_per_chunk,
    include_similarities=True,
)
```

with:

```python
# Use precomputed results if supplied (e.g., from adaptive rechunker re-aggregation pass).
if precomputed_query_results is not None:
    if len(precomputed_query_results) != len(text_chunks):
        raise ValueError(
            f"precomputed_query_results length ({len(precomputed_query_results)}) "
            f"does not match text_chunks length ({len(text_chunks)})"
        )
    all_query_results = precomputed_query_results
    logger.info(
        "Using %d precomputed query results (skipping retrieval)",
        len(all_query_results),
    )
else:
    logger.info(f"Batch querying {len(text_chunks)} chunks at once")
    all_query_results = retriever.query_batch(
        texts=text_chunks,
        n_results=num_results_per_chunk,
        include_similarities=True,
    )
```

4. Change the return statement at line 297 from:

```python
return (aggregated_results_list, chunk_results)
```

to:

```python
return OrchestrationResult(
    aggregated_results=aggregated_results_list,
    chunk_results=chunk_results,
    raw_query_results=all_query_results,
)
```

5. Update the type annotation on the function:

```python
) -> OrchestrationResult:
    """..."""
```

(Update the docstring to mention the new return type and the precomputed parameter.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/text_processing/test_orchestrate_with_precomputed.py tests/unit/text_processing/test_orchestration_result_legacy_unpack.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run existing orchestrator tests to confirm no regressions**

Run: `uv run pytest tests/unit/text_processing/ -v`
Expected: PASS, including pre-existing tests.

- [ ] **Step 6: Verify all known legacy call sites compile**

Run: `uv run python -c "from phentrieve.text_processing.full_text_service import run_standard_backend; from phentrieve.cli.text_interactive import interactive_text_mode; from phentrieve.benchmark.extraction_benchmark import *; from phentrieve.evaluation.full_text_runner import *; from phentrieve.llm.provider import *; print('all imports ok')"`
Expected: `all imports ok`.

- [ ] **Step 7: Run the broader test suite**

Run: `uv run pytest tests/ -x --ignore=tests/integration -q`
Expected: PASS — every existing 2-tuple unpacker still works via `__iter__`.

- [ ] **Step 8: Commit**

```bash
git add phentrieve/text_processing/hpo_extraction_orchestrator.py tests/unit/text_processing/test_orchestrate_with_precomputed.py
git commit -m "feat(text-processing): orchestrate_hpo_extraction returns OrchestrationResult

Two changes, both additive:
1. Return type changes from tuple[list, list] to OrchestrationResult
   dataclass. Legacy 2-tuple unpacking still works via __iter__.
   raw_query_results is exposed as a third attribute for new callers
   (adaptive rechunker).
2. New optional parameter precomputed_query_results: when provided,
   skips retriever.query_batch and aggregates over the supplied raw
   data. Used by the adaptive rechunker for re-aggregation passes
   without re-querying."
```

### Task 3: Add `adapt_standard_response.extra_meta` parameter

**Files:**
- Modify: `phentrieve/text_processing/full_text_service.py`
- Create: `tests/unit/text_processing/test_adapt_standard_response_extra_meta.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/text_processing/test_adapt_standard_response_extra_meta.py`:

```python
"""Tests for adapt_standard_response.extra_meta merging."""

from phentrieve.text_processing.full_text_service import adapt_standard_response


def test_extra_meta_none_preserves_existing_shape():
    response = adapt_standard_response(
        pipeline_result=[],
        extraction_result=([], []),
        extra_meta=None,
    )
    assert "meta" in response
    assert response["meta"]["extraction_backend"] == "standard"
    # No new keys introduced.
    assert "adaptive_rechunking" not in response["meta"]


def test_extra_meta_dict_merges_into_meta():
    extra = {"adaptive_rechunking": {"enabled": True, "trigger_count": 3}}
    response = adapt_standard_response(
        pipeline_result=[],
        extraction_result=([], []),
        extra_meta=extra,
    )
    assert response["meta"]["adaptive_rechunking"] == {
        "enabled": True,
        "trigger_count": 3,
    }
    # Existing meta keys preserved.
    assert response["meta"]["extraction_backend"] == "standard"


def test_extra_meta_does_not_overwrite_extraction_backend():
    """If extra_meta accidentally includes extraction_backend, the canonical
    value from the response wins (we set extraction_backend last)."""
    extra = {"extraction_backend": "wrong"}
    response = adapt_standard_response(
        pipeline_result=[],
        extraction_result=([], []),
        extra_meta=extra,
    )
    assert response["meta"]["extraction_backend"] == "standard"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/text_processing/test_adapt_standard_response_extra_meta.py -v`
Expected: FAIL — `adapt_standard_response` doesn't accept `extra_meta`.

- [ ] **Step 3: Modify `adapt_standard_response`**

Edit `phentrieve/text_processing/full_text_service.py`. The current `adapt_standard_response` is around lines 490-533. Change its signature and body:

```python
def adapt_standard_response(
    pipeline_result: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    extraction_result: tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    | Mapping[str, Any]
    | None,
    extra_meta: dict[str, Any] | None = None,   # NEW
) -> StableBackendResponse:
    """Convert pipeline and extraction outputs into the stable response shape.

    extra_meta: optional dict merged into the response's meta block. Used by
    adaptive rechunking to surface its meta.adaptive_rechunking summary.
    The caller's keys are merged BEFORE the response builds its own
    canonical keys (extraction_backend, num_processed_chunks,
    num_aggregated_hpo_terms), so canonical keys always win.
    """
    if isinstance(pipeline_result, Mapping):
        processed_chunks = _coerce_list(
            pipeline_result.get("processed_chunks") or pipeline_result.get("chunks")
        )
    else:
        processed_chunks = _coerce_list(pipeline_result)

    if isinstance(extraction_result, Mapping):
        aggregated_results = _coerce_list(
            extraction_result.get("aggregated_hpo_terms")
            or extraction_result.get("aggregated_results")
        )
        chunk_results = _coerce_list(
            extraction_result.get("chunk_results")
            or extraction_result.get("detailed_chunk_results")
        )
    elif extraction_result is None:
        aggregated_results = []
        chunk_results = []
    else:
        aggregated_results = list(extraction_result[0])
        chunk_results = list(extraction_result[1])

    adapted_chunks = _adapt_processed_chunks(processed_chunks, chunk_results)
    adapted_terms = _adapt_aggregated_terms(aggregated_results)

    # Build meta starting from extra_meta so canonical keys override.
    meta_block: dict[str, Any] = dict(extra_meta or {})
    meta_block["num_processed_chunks"] = len(adapted_chunks)
    meta_block["num_aggregated_hpo_terms"] = len(adapted_terms)

    return adapt_full_text_response(
        {
            "meta": meta_block,
            "processed_chunks": adapted_chunks,
            "aggregated_hpo_terms": adapted_terms,
        },
        extraction_backend="standard",
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/text_processing/test_adapt_standard_response_extra_meta.py -v`
Expected: PASS, three tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/text_processing/full_text_service.py tests/unit/text_processing/test_adapt_standard_response_extra_meta.py
git commit -m "feat(text-processing): adapt_standard_response.extra_meta parameter

New optional dict parameter merges into the response's meta block.
Canonical keys (extraction_backend, num_processed_chunks,
num_aggregated_hpo_terms) always win on conflict. Used by adaptive
rechunking to surface its meta.adaptive_rechunking summary in the
final response."
```

---

## Phase 2: AdaptiveRechunkingConfig

### Task 4: Define `AdaptiveRechunkingConfig` and the Profile block

**Files:**
- Create: `phentrieve/retrieval/adaptive_rechunker.py` (skeleton with config only)
- Create: `tests/unit/retrieval/adaptive_rechunker/__init__.py`
- Create: `tests/unit/retrieval/adaptive_rechunker/test_adaptive_config.py`

- [ ] **Step 1: Create empty `__init__.py`**

Run: `mkdir -p tests/unit/retrieval/adaptive_rechunker && touch tests/unit/retrieval/adaptive_rechunker/__init__.py`

- [ ] **Step 2: Write failing test**

Create `tests/unit/retrieval/adaptive_rechunker/test_adaptive_config.py`:

```python
"""Tests for AdaptiveRechunkingConfig and AdaptiveRechunkingProfileBlock."""

import pytest


def test_default_config_disabled():
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig

    cfg = AdaptiveRechunkingConfig()
    assert cfg.enabled is False  # opt-in
    assert cfg.quality_threshold == 0.55
    assert cfg.margin_threshold == 0.03
    assert cfg.use_ontology_coherence is False
    assert cfg.max_depth == 2
    assert cfg.min_chunk_chars == 30
    assert cfg.max_sentences_per_subchunk == 3
    assert cfg.overlap_sentences == 1
    assert cfg.score_improvement_gate == 0.05


def test_config_is_frozen():
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig

    cfg = AdaptiveRechunkingConfig(enabled=True)
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.enabled = False  # type: ignore[misc]


def test_profile_block_pydantic_extra_forbid():
    from pydantic import ValidationError

    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingProfileBlock

    with pytest.raises(ValidationError):
        AdaptiveRechunkingProfileBlock(unknown_knob=123)


def test_profile_block_optional_fields():
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingProfileBlock

    block = AdaptiveRechunkingProfileBlock(enabled=True)
    assert block.enabled is True
    assert block.quality_threshold is None  # optional
    assert block.max_depth is None


def test_profile_block_all_fields():
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingProfileBlock

    block = AdaptiveRechunkingProfileBlock(
        enabled=True,
        quality_threshold=0.6,
        margin_threshold=0.02,
        max_depth=1,
        min_chunk_chars=40,
        max_sentences_per_subchunk=2,
        overlap_sentences=0,
        score_improvement_gate=0.1,
        use_ontology_coherence=False,
    )
    assert block.quality_threshold == 0.6
    assert block.max_depth == 1
```

- [ ] **Step 3: Run tests to verify failure**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_adaptive_config.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 4: Create `phentrieve/retrieval/adaptive_rechunker.py` with the config**

```python
"""Adaptive re-chunking for poor-quality retrieval results.

Implements Spec B (.planning/specs/2026-04-25-adaptive-rechunking-spec.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class AdaptiveRechunkingConfig:
    """End-to-end configuration carried through the pipeline."""

    enabled: bool = False
    quality_threshold: float = 0.55
    margin_threshold: float = 0.03
    use_ontology_coherence: bool = False  # reserved, inert in v1
    max_depth: int = 2
    min_chunk_chars: int = 30
    max_sentences_per_subchunk: int = 3
    overlap_sentences: int = 1
    score_improvement_gate: float = 0.05


class AdaptiveRechunkingProfileBlock(BaseModel):
    """Pydantic block on Profile.adaptive_rechunking. extra='forbid' so YAML
    typos error at load time. Plan A's Profile schema imports this type.
    """

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


def adaptive_config_from_profile_block(
    block: AdaptiveRechunkingProfileBlock | None,
    yaml_block: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AdaptiveRechunkingConfig:
    """Resolve the config from CLI > profile > YAML > built-in defaults."""
    defaults = AdaptiveRechunkingConfig()
    fields = {f for f in defaults.__dataclass_fields__}

    resolved: dict[str, Any] = {}
    for name in fields:
        cli_value = (cli_overrides or {}).get(name)
        if cli_value is not None:
            resolved[name] = cli_value
            continue
        profile_value = getattr(block, name, None) if block is not None else None
        if profile_value is not None:
            resolved[name] = profile_value
            continue
        yaml_value = (yaml_block or {}).get(name)
        if yaml_value is not None:
            resolved[name] = yaml_value
            continue
        resolved[name] = getattr(defaults, name)
    return AdaptiveRechunkingConfig(**resolved)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_adaptive_config.py -v`
Expected: PASS, five tests.

- [ ] **Step 6: Wire `AdaptiveRechunkingProfileBlock` into Plan A's Profile**

If Plan A has landed, edit `phentrieve/profiles.py`:

```python
# Replace the placeholder line:
# AdaptiveRechunkingProfileBlock = dict[str, Any]
from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingProfileBlock
```

If Plan A has not landed yet, leave the Plan A file as-is (the placeholder dict shape works for now); revisit when Plan A lands.

- [ ] **Step 7: Commit**

```bash
git add phentrieve/retrieval/adaptive_rechunker.py tests/unit/retrieval/adaptive_rechunker/__init__.py tests/unit/retrieval/adaptive_rechunker/test_adaptive_config.py phentrieve/profiles.py
git commit -m "feat(adaptive): AdaptiveRechunkingConfig + AdaptiveRechunkingProfileBlock

Frozen dataclass for runtime config; pydantic block for profile schema.
Resolution helper merges CLI > profile > YAML > defaults. Wires into
Plan A's Profile schema (no-op if Plan A hasn't landed)."
```

---

## Phase 3: Quality assessment

### Task 5: `ChunkQualitySignals` and `assess_chunk_quality`

**Files:**
- Modify: `phentrieve/retrieval/adaptive_rechunker.py`
- Create: `tests/unit/retrieval/adaptive_rechunker/test_quality_assessment.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/retrieval/adaptive_rechunker/test_quality_assessment.py`:

```python
"""Tests for assess_chunk_quality.

The trigger conjunction:
    is_poor = top_1 < quality_threshold AND (margin < margin_threshold OR top_2 is None)
"""

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    ChunkQualitySignals,
    assess_chunk_quality,
)


def make_raw_result(similarities: list[float]) -> dict:
    """Helper to build a raw query_batch output entry."""
    return {
        "ids": [[f"HP:{i:07d}" for i in range(len(similarities))]],
        "metadatas": [[{"id": f"HP:{i:07d}"} for i in range(len(similarities))]],
        "similarities": [similarities],
        "distances": [[1 - s for s in similarities]],
        "documents": [[""] * len(similarities)],
    }


@pytest.fixture
def config():
    return AdaptiveRechunkingConfig(quality_threshold=0.55, margin_threshold=0.03)


class TestAssessChunkQuality:
    def test_empty_raw_results_is_poor_no_matches(self, config):
        raw = make_raw_result([])
        sig = assess_chunk_quality(raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config)
        assert sig.is_poor is True
        assert sig.reason == "no_matches"
        assert sig.top_1 is None

    def test_single_match_above_threshold_is_ok(self, config):
        raw = make_raw_result([0.9])
        sig = assess_chunk_quality(raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config)
        # top_2 is None — but top_1 above quality_threshold means we trust it.
        assert sig.is_poor is False
        assert sig.reason == "ok"

    def test_single_match_below_threshold_is_poor_low_score(self, config):
        raw = make_raw_result([0.4])
        sig = assess_chunk_quality(raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config)
        # top_1 < 0.55 AND top_2 is None → poor.
        assert sig.is_poor is True
        assert sig.reason == "low_score"

    def test_two_matches_above_threshold_with_low_margin_ok(self, config):
        # top_1=0.9, top_2=0.89, margin=0.01 < 0.03.
        # But top_1 ≥ quality_threshold so we trust the result regardless.
        raw = make_raw_result([0.9, 0.89])
        sig = assess_chunk_quality(raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config)
        assert sig.is_poor is False
        assert sig.reason == "ok"

    def test_two_matches_low_score_high_margin_ok(self, config):
        # top_1=0.5, top_2=0.1, margin=0.4 ≥ 0.03.
        # Conjunction: top_1 < 0.55 AND (margin < 0.03 OR top_2 is None).
        # margin is high → second condition false → not poor.
        raw = make_raw_result([0.5, 0.1])
        sig = assess_chunk_quality(raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config)
        assert sig.is_poor is False
        assert sig.reason == "ok"

    def test_two_matches_low_score_low_margin_is_poor_low_margin(self, config):
        # top_1=0.5, top_2=0.49, margin=0.01 < 0.03. Both bad → poor.
        raw = make_raw_result([0.5, 0.49])
        sig = assess_chunk_quality(raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config)
        assert sig.is_poor is True
        assert sig.reason == "low_margin"

    def test_signals_record_top_1_top_2_margin(self, config):
        raw = make_raw_result([0.4, 0.38, 0.35])
        sig = assess_chunk_quality(raw, chunk_idx=2, chunk_retrieval_threshold=0.7, config=config)
        assert sig.chunk_idx == 2
        assert sig.top_1 == 0.4
        assert sig.top_2 == 0.38
        assert sig.margin == pytest.approx(0.02)

    def test_n_matches_above_threshold_informational(self, config):
        # chunk_retrieval_threshold filters chunk_results, but
        # assess_chunk_quality reads raw — n_matches_above_threshold is informational.
        raw = make_raw_result([0.9, 0.85, 0.5, 0.4])  # 2 above 0.7
        sig = assess_chunk_quality(raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config)
        assert sig.n_matches_above_threshold == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_quality_assessment.py -v`
Expected: FAIL.

- [ ] **Step 3: Add the helper to `phentrieve/retrieval/adaptive_rechunker.py`**

Append:

```python
@dataclass(frozen=True)
class ChunkQualitySignals:
    """Quality assessment of one chunk's retrieval result."""

    chunk_idx: int
    top_1: float | None
    top_2: float | None
    margin: float | None
    n_matches_above_threshold: int
    is_poor: bool
    reason: str  # "low_score" | "low_margin" | "no_matches" | "ok"


def assess_chunk_quality(
    raw_query_result: dict,
    chunk_idx: int,
    chunk_retrieval_threshold: float,
    config: AdaptiveRechunkingConfig,
) -> ChunkQualitySignals:
    """Read top-K from raw query_batch output, decide if the chunk is poor.

    Reads from raw_query_result["similarities"][0] (the unfiltered list of
    top-K similarity scores from query_batch). Crucially, this is NOT the
    threshold-filtered chunk_results — see Spec B Architecture for why.
    """
    similarities = (
        raw_query_result.get("similarities", [[]])[0]
        if raw_query_result.get("similarities")
        else []
    )

    if not similarities:
        return ChunkQualitySignals(
            chunk_idx=chunk_idx,
            top_1=None,
            top_2=None,
            margin=None,
            n_matches_above_threshold=0,
            is_poor=True,
            reason="no_matches",
        )

    top_1 = similarities[0]
    top_2 = similarities[1] if len(similarities) > 1 else None
    margin = (top_1 - top_2) if top_2 is not None else None
    n_above = sum(1 for s in similarities if s >= chunk_retrieval_threshold)

    score_low = top_1 < config.quality_threshold
    margin_low_or_unknown = (
        top_2 is None or margin is not None and margin < config.margin_threshold
    )

    is_poor = score_low and margin_low_or_unknown
    if not is_poor:
        reason = "ok"
    elif top_2 is None:
        reason = "low_score"  # only one match, score too low
    elif margin is not None and margin >= config.margin_threshold:
        # Score is low but margin is fine — shouldn't reach here given is_poor logic.
        reason = "ok"
    else:
        reason = "low_margin"

    return ChunkQualitySignals(
        chunk_idx=chunk_idx,
        top_1=top_1,
        top_2=top_2,
        margin=margin,
        n_matches_above_threshold=n_above,
        is_poor=is_poor,
        reason=reason,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_quality_assessment.py -v`
Expected: PASS, eight tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/retrieval/adaptive_rechunker.py tests/unit/retrieval/adaptive_rechunker/test_quality_assessment.py
git commit -m "feat(adaptive): assess_chunk_quality reads raw query_batch top-K

Score-AND-margin conjunction trigger: top_1 < quality_threshold AND
(margin < margin_threshold OR top_2 is None). Reads from raw query
output, not the threshold-filtered chunk_results, so chunks where
retrieval was genuinely poor (everything below chunk_retrieval_threshold)
are still detectable."
```

### Task 6: Raw-score-access invariant test

**Files:**
- Create: `tests/unit/retrieval/adaptive_rechunker/test_raw_score_access.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/retrieval/adaptive_rechunker/test_raw_score_access.py`:

```python
"""Critical invariant: assess_chunk_quality reads from raw query_batch
output, NOT from threshold-filtered chunk_results. Without this test, a
regression to filtered input could silently break the trigger for the
exact case adaptive rechunking is designed to handle."""

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    assess_chunk_quality,
)


class TestRawScoreAccess:
    def test_genuinely_poor_chunk_detectable(self):
        """Simulates a chunk where every retrieval result was below
        chunk_retrieval_threshold. orchestrate_hpo_extraction would put
        an empty `matches` list into chunk_results — but raw_query_results
        still has the (low) similarities."""
        config = AdaptiveRechunkingConfig(quality_threshold=0.55, margin_threshold=0.03)

        # All similarities below the standard chunk_retrieval_threshold of 0.7.
        # The corresponding chunk_results entry would have matches=[].
        raw = {
            "ids": [["HP:0001", "HP:0002"]],
            "metadatas": [[{"id": "HP:0001"}, {"id": "HP:0002"}]],
            "similarities": [[0.4, 0.39]],  # Both below 0.55, low margin.
            "distances": [[0.6, 0.61]],
            "documents": [["", ""]],
        }

        sig = assess_chunk_quality(
            raw_query_result=raw,
            chunk_idx=0,
            chunk_retrieval_threshold=0.7,  # all matches below this!
            config=config,
        )

        # The trigger MUST fire — both score and margin are bad.
        assert sig.is_poor is True
        assert sig.reason == "low_margin"
        # And n_matches_above_threshold reports the chunk_results filter result.
        assert sig.n_matches_above_threshold == 0  # nothing above 0.7

    def test_top_2_present_in_raw_even_if_filtered_out_of_chunk_results(self):
        """When chunk_retrieval_threshold = 0.7 and similarities = [0.6, 0.5],
        chunk_results[0]['matches'] is empty. But raw still has both scores.
        """
        config = AdaptiveRechunkingConfig(quality_threshold=0.55, margin_threshold=0.03)
        raw = {
            "ids": [["HP:0001", "HP:0002"]],
            "metadatas": [[{"id": "HP:0001"}, {"id": "HP:0002"}]],
            "similarities": [[0.6, 0.5]],
            "distances": [[0.4, 0.5]],
            "documents": [["", ""]],
        }
        sig = assess_chunk_quality(
            raw, chunk_idx=0, chunk_retrieval_threshold=0.7, config=config
        )
        # top_1=0.6 ≥ quality_threshold 0.55, so we trust it.
        assert sig.is_poor is False
        assert sig.top_2 == 0.5
        assert sig.margin == pytest.approx(0.1)
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_raw_score_access.py -v`
Expected: PASS, two tests.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/retrieval/adaptive_rechunker/test_raw_score_access.py
git commit -m "test(adaptive): raw-score-access invariant for assess_chunk_quality

Pins the contract that the trigger reads from raw query output rather
than threshold-filtered chunk_results. A regression that swapped the
two inputs would silently miss the exact case adaptive rechunking is
designed to handle: chunks where retrieval was genuinely poor."
```

---

## Phase 4: Sub-chunking

### Task 7: `subdivide_parent_chunk`

**Files:**
- Modify: `phentrieve/retrieval/adaptive_rechunker.py`
- Create: `tests/unit/retrieval/adaptive_rechunker/test_subdivide_parent_chunk.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/retrieval/adaptive_rechunker/test_subdivide_parent_chunk.py`:

```python
"""Tests for subdivide_parent_chunk."""

from unittest.mock import patch

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    subdivide_parent_chunk,
)


@pytest.fixture
def config():
    return AdaptiveRechunkingConfig(
        min_chunk_chars=20, max_sentences_per_subchunk=3, overlap_sentences=1
    )


def make_parent(text: str, start_char: int = 0) -> dict:
    return {
        "text": text,
        "status": "AFFIRMED",
        "assertion_details": {"trigger": "default"},
        "source_indices": {"processing_stages": ["paragraph", "sentence"]},
        "start_char": start_char,
        "end_char": start_char + len(text),
    }


class TestSubdivideParentChunk:
    def test_multi_sentence_parent_produces_subchunks(self, config):
        text = (
            "Patient has severe intellectual disability. "
            "He shows recurrent seizures since age 3. "
            "Brain MRI revealed cortical atrophy. "
            "Family history is unremarkable."
        )
        parent = make_parent(text)
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=1
        )
        assert len(children) >= 1
        # Every child preserves assertion_status.
        for child in children:
            assert child["status"] == "AFFIRMED"

    def test_single_sentence_parent_returns_empty(self, config):
        parent = make_parent("Patient has severe intellectual disability.")
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=1
        )
        # Either 0 (no useful split) or 1 (a single child equal to the parent —
        # filtered out).
        assert children == []

    def test_subchunks_track_depth_in_processing_stages(self, config):
        text = "First sentence here. Second sentence here. Third sentence."
        parent = make_parent(text)
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=2
        )
        for child in children:
            stages = child.get("source_indices", {}).get("processing_stages", [])
            assert any("adaptive_rechunker_depth_2" in s for s in stages)

    def test_subchunks_offset_by_parent_start_char(self, config):
        text = "First sentence. Second sentence. Third sentence."
        parent = make_parent(text, start_char=100)
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=1
        )
        for child in children:
            assert child["start_char"] >= 100
            assert child["end_char"] <= 100 + len(text)

    def test_short_subchunks_below_min_chunk_chars_dropped(self):
        config = AdaptiveRechunkingConfig(
            min_chunk_chars=80, max_sentences_per_subchunk=1, overlap_sentences=0
        )
        text = "Short. Tiny. Brief sentence. " + "x" * 100 + "."
        parent = make_parent(text)
        children = subdivide_parent_chunk(
            parent_chunk=parent, language="en", config=config, depth=1
        )
        # Sentences shorter than min_chunk_chars (80) should be dropped.
        for child in children:
            assert len(child["text"]) >= 80
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_subdivide_parent_chunk.py -v`
Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Implement `subdivide_parent_chunk`**

Append to `phentrieve/retrieval/adaptive_rechunker.py`:

```python
def subdivide_parent_chunk(
    parent_chunk: dict[str, Any],
    language: str,
    config: AdaptiveRechunkingConfig,
    depth: int,
) -> list[dict[str, Any]]:
    """Generate sentence-bounded sub-chunks from a parent chunk.

    Returns sub-chunks in the same dict shape as TextProcessingPipeline output.
    Sub-chunks inherit parent's assertion_status / assertion_details.
    Returns [] if no useful subdivision is possible.
    """
    # Lazy import — SentenceChunker pulls in pysbd which is heavy.
    from phentrieve.text_processing.chunkers import SentenceChunker

    parent_text = parent_chunk.get("text", "")
    if not parent_text:
        return []

    chunker = SentenceChunker(language=language)
    sentences = chunker.chunk([parent_text])
    if len(sentences) <= 1:
        return []  # Single sentence — no subdivision possible.

    # Group sentences into sub-chunks with overlap.
    window = max(1, config.max_sentences_per_subchunk)
    overlap = max(0, min(config.overlap_sentences, window - 1))
    step = window - overlap
    if step <= 0:
        step = 1

    parent_start = parent_chunk.get("start_char", 0)
    parent_status = parent_chunk.get("status")
    parent_details = parent_chunk.get("assertion_details")
    parent_stages = list(
        parent_chunk.get("source_indices", {}).get("processing_stages", [])
    )

    children: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for i in range(0, len(sentences), step):
        group = sentences[i : i + window]
        sub_text = " ".join(group).strip()
        if not sub_text:
            continue
        if sub_text == parent_text.strip():
            continue  # Identical to parent — no useful subdivision.
        if sub_text in seen_texts:
            continue
        seen_texts.add(sub_text)
        if len(sub_text) < config.min_chunk_chars:
            continue

        # Locate within the parent text to compute spans.
        idx = parent_text.find(sub_text)
        if idx < 0:
            # Fallback: try first sentence of group.
            idx = parent_text.find(group[0]) if group else -1
            if idx < 0:
                idx = 0
        start_char = parent_start + idx
        end_char = start_char + len(sub_text)

        children.append(
            {
                "text": sub_text,
                "status": parent_status,
                "assertion_details": parent_details,
                "source_indices": {
                    "processing_stages": parent_stages + [f"adaptive_rechunker_depth_{depth}"],
                },
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    return children
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_subdivide_parent_chunk.py -v`
Expected: PASS, five tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/retrieval/adaptive_rechunker.py tests/unit/retrieval/adaptive_rechunker/test_subdivide_parent_chunk.py
git commit -m "feat(adaptive): subdivide_parent_chunk via SentenceChunker

Sentence-bounded subdivision with overlap. Sub-chunks inherit parent
assertion status (no re-detection in v1), offset start_char from parent,
drop sub-chunks shorter than min_chunk_chars or identical to parent.
Tags processing_stages with adaptive_rechunker_depth_<N> for traceability."
```

---

## Phase 5: Score-improvement gate

### Task 8: `apply_score_improvement_gate`

**Files:**
- Modify: `phentrieve/retrieval/adaptive_rechunker.py`
- Create: `tests/unit/retrieval/adaptive_rechunker/test_score_improvement_gate.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/retrieval/adaptive_rechunker/test_score_improvement_gate.py`:

```python
"""Tests for apply_score_improvement_gate."""

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    apply_score_improvement_gate,
)


def test_all_children_below_gate_revert():
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    parent_top_1 = {0: 0.5}  # parent_idx 0 had top_1 = 0.5
    child_top_1 = {1: 0.51, 2: 0.49}  # neither beats parent + 0.05 = 0.55

    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: [1, 2]},
        parent_top_1=parent_top_1,
        child_top_1=child_top_1,
        config=config,
    )
    assert revert == {0}
    assert keep == set()


def test_one_child_above_gate_keep():
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    parent_top_1 = {0: 0.5}
    child_top_1 = {1: 0.4, 2: 0.7}  # child 2 beats 0.5 + 0.05 = 0.55

    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: [1, 2]},
        parent_top_1=parent_top_1,
        child_top_1=child_top_1,
        config=config,
    )
    assert revert == set()
    assert keep == {0}


def test_multiple_parents_mixed_outcome():
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    parent_top_1 = {0: 0.5, 1: 0.6}
    child_top_1 = {2: 0.45, 3: 0.46, 4: 0.7, 5: 0.71}
    # Parent 0 (children 2,3): max(0.46) < 0.55 → revert.
    # Parent 1 (children 4,5): max(0.71) > 0.65 → keep.
    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: [2, 3], 1: [4, 5]},
        parent_top_1=parent_top_1,
        child_top_1=child_top_1,
        config=config,
    )
    assert revert == {0}
    assert keep == {1}


def test_parent_with_no_children_skipped():
    config = AdaptiveRechunkingConfig(score_improvement_gate=0.05)
    revert, keep = apply_score_improvement_gate(
        parent_to_children={0: []},
        parent_top_1={0: 0.5},
        child_top_1={},
        config=config,
    )
    # No children → no decision, parent stays as-is.
    assert revert == set()
    assert keep == set()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_score_improvement_gate.py -v`
Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Implement `apply_score_improvement_gate`**

Append to `phentrieve/retrieval/adaptive_rechunker.py`:

```python
def apply_score_improvement_gate(
    parent_to_children: dict[int, list[int]],
    parent_top_1: dict[int, float],
    child_top_1: dict[int, float],
    config: AdaptiveRechunkingConfig,
) -> tuple[set[int], set[int]]:
    """Decide per-parent whether subdivision improved retrieval enough to keep.

    Returns:
        (revert_parents, keep_parents) — indices of parents where children
        should be reverted (sub-chunks dropped, parent restored) vs kept
        (sub-chunks replace parent in the final flat list).

    A parent is kept iff at least one of its children's top_1 is at least
    parent_top_1 + score_improvement_gate. Otherwise reverted.
    """
    revert: set[int] = set()
    keep: set[int] = set()
    for parent_idx, children in parent_to_children.items():
        if not children:
            continue
        parent_t1 = parent_top_1.get(parent_idx)
        if parent_t1 is None:
            continue
        child_scores = [child_top_1.get(c) for c in children]
        child_scores = [s for s in child_scores if s is not None]
        if not child_scores:
            revert.add(parent_idx)
            continue
        best = max(child_scores)
        if best >= parent_t1 + config.score_improvement_gate:
            keep.add(parent_idx)
        else:
            revert.add(parent_idx)
    return revert, keep
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_score_improvement_gate.py -v`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/retrieval/adaptive_rechunker.py tests/unit/retrieval/adaptive_rechunker/test_score_improvement_gate.py
git commit -m "feat(adaptive): apply_score_improvement_gate

Per-parent gate: keep iff best child top_1 >= parent top_1 + gate.
Operates on raw top_1 values without re-aggregation, keeping the
recursion loop to one query_batch call per level."
```

---

## Phase 6: Top-level `run_adaptive_rechunking`

### Task 9: AdaptiveRechunkingResult dataclass and the orchestration function

**Files:**
- Modify: `phentrieve/retrieval/adaptive_rechunker.py`
- Create: `tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py`:

```python
"""Tests for the top-level run_adaptive_rechunking orchestration."""

from unittest.mock import MagicMock

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    AdaptiveRechunkingResult,
    run_adaptive_rechunking,
)


def make_raw(similarities: list[float]) -> dict:
    return {
        "ids": [[f"HP:{i:07d}" for i in range(len(similarities))]],
        "metadatas": [[{"id": f"HP:{i:07d}", "label": "x"} for i in range(len(similarities))]],
        "similarities": [similarities],
        "distances": [[1 - s for s in similarities]],
        "documents": [[""] * len(similarities)],
    }


@pytest.fixture
def mock_retriever():
    return MagicMock()


@pytest.fixture
def basic_inputs():
    """One good chunk + one poor chunk for happy-path tests."""
    processed_chunks = [
        {"text": "Good chunk.", "status": "AFFIRMED", "start_char": 0, "end_char": 11,
         "source_indices": {"processing_stages": ["sentence"]}},
        {"text": "Poor multi-finding sentence one. Poor multi-finding sentence two. Poor multi-finding sentence three.",
         "status": "AFFIRMED", "start_char": 12, "end_char": 113,
         "source_indices": {"processing_stages": ["sentence"]}},
    ]
    chunk_results = [
        {"chunk_idx": 0, "chunk_text": processed_chunks[0]["text"], "matches": [{"id": "HP:0001", "name": "Good", "score": 0.9}]},
        {"chunk_idx": 1, "chunk_text": processed_chunks[1]["text"], "matches": []},  # below threshold
    ]
    raw_query_results = [
        make_raw([0.9, 0.8]),    # good chunk
        make_raw([0.4, 0.39]),   # poor chunk: low score AND low margin
    ]
    return processed_chunks, chunk_results, raw_query_results


class TestRunAdaptiveRechunking:
    def test_disabled_returns_inputs_unchanged(self, mock_retriever, basic_inputs):
        chunks, results, raw = basic_inputs
        config = AdaptiveRechunkingConfig(enabled=False)
        out = run_adaptive_rechunking(
            processed_chunks=chunks,
            chunk_results=results,
            raw_query_results=raw,
            retriever=mock_retriever,
            language="en",
            config=config,
            num_results_per_chunk=10,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )
        assert isinstance(out, AdaptiveRechunkingResult)
        assert out.processed_chunks == chunks
        assert out.chunk_results == results
        assert out.meta["enabled"] is False
        # No retrieval calls were made.
        assert mock_retriever.query_batch.call_count == 0

    def test_no_poor_chunks_no_op(self, mock_retriever):
        """All chunks are fine — no subdivision, no extra query_batch calls."""
        processed = [
            {"text": "Good chunk one.", "status": "AFFIRMED", "start_char": 0, "end_char": 15,
             "source_indices": {"processing_stages": ["sentence"]}},
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": [{"id": "HP:0001", "name": "x", "score": 0.9}]}]
        raw = [make_raw([0.95, 0.85])]
        config = AdaptiveRechunkingConfig(enabled=True)

        out = run_adaptive_rechunking(
            processed_chunks=processed, chunk_results=results, raw_query_results=raw,
            retriever=mock_retriever, language="en", config=config,
            num_results_per_chunk=10, chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0, include_details=False,
        )
        assert mock_retriever.query_batch.call_count == 0
        assert out.meta["trigger_count"] == 0
        assert out.meta["subdivided_count"] == 0

    def test_one_poor_chunk_subdivided_with_improving_children(self, mock_retriever):
        """One poor chunk subdivided into children that improve via the gate."""
        processed = [
            {"text": "First sentence here. Second sentence here. Third sentence here.",
             "status": "AFFIRMED", "start_char": 0, "end_char": 64,
             "source_indices": {"processing_stages": ["sentence"]}},
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]  # poor: low score, low margin

        # Mock retriever returns improved scores for children.
        # We expect 1 query_batch call (the children) at depth 1.
        mock_retriever.query_batch.return_value = [
            make_raw([0.9, 0.5]),  # child 1: top_1 = 0.9 > 0.4 + 0.05 = 0.45 → improves!
        ]

        config = AdaptiveRechunkingConfig(
            enabled=True,
            quality_threshold=0.55,
            margin_threshold=0.03,
            score_improvement_gate=0.05,
            max_depth=1,
            max_sentences_per_subchunk=3,
            overlap_sentences=0,
            min_chunk_chars=10,
        )

        out = run_adaptive_rechunking(
            processed_chunks=processed, chunk_results=results, raw_query_results=raw,
            retriever=mock_retriever, language="en", config=config,
            num_results_per_chunk=10, chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0, include_details=False,
        )

        # Exactly one query_batch call for the children at this single recursion level.
        assert mock_retriever.query_batch.call_count == 1
        assert out.meta["enabled"] is True
        assert out.meta["trigger_count"] == 1
        assert out.meta["subdivided_count"] == 1
        assert out.meta["reverted_count"] == 0

    def test_call_count_invariant_at_max_depth_2(self, mock_retriever):
        """The hard cost-model contract: 1 call per recursion level, max_depth+1 levels total."""
        # Initial pass already happened by the caller — we check only what
        # run_adaptive_rechunking itself triggers.
        processed = [
            {"text": "Sentence one. Sentence two. Sentence three.",
             "status": "AFFIRMED", "start_char": 0, "end_char": 44,
             "source_indices": {"processing_stages": ["sentence"]}},
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]

        # Children always also "poor" so recursion continues.
        # Children improve enough to pass the gate, then still flag at depth 2.
        depth_1_children = make_raw([0.6, 0.59])  # improves over 0.4 by 0.2 → keeps;
        depth_2_grandchildren = make_raw([0.9, 0.5])  # improves further

        mock_retriever.query_batch.side_effect = [
            [depth_1_children],      # depth 1 query
            [depth_2_grandchildren], # depth 2 query
        ]

        config = AdaptiveRechunkingConfig(
            enabled=True, quality_threshold=0.7,  # higher to keep flagging
            margin_threshold=0.03, score_improvement_gate=0.05,
            max_depth=2, max_sentences_per_subchunk=2,
            overlap_sentences=0, min_chunk_chars=5,
        )

        run_adaptive_rechunking(
            processed_chunks=processed, chunk_results=results, raw_query_results=raw,
            retriever=mock_retriever, language="en", config=config,
            num_results_per_chunk=10, chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0, include_details=False,
        )

        # AT MOST 2 query_batch calls (depth 1 + depth 2). Hard invariant.
        assert mock_retriever.query_batch.call_count <= 2

    def test_recursion_respects_max_depth(self, mock_retriever):
        """max_depth=1 means: subdivide once, do not recurse further."""
        processed = [
            {"text": "Sentence one. Sentence two. Sentence three.",
             "status": "AFFIRMED", "start_char": 0, "end_char": 44,
             "source_indices": {"processing_stages": ["sentence"]}},
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]
        # Children also flag as poor — at max_depth=1 we should not recurse.
        mock_retriever.query_batch.return_value = [make_raw([0.6, 0.59])]

        config = AdaptiveRechunkingConfig(
            enabled=True, max_depth=1,
            quality_threshold=0.7, margin_threshold=0.03,
            score_improvement_gate=0.05, max_sentences_per_subchunk=2,
            overlap_sentences=0, min_chunk_chars=5,
        )

        out = run_adaptive_rechunking(
            processed_chunks=processed, chunk_results=results, raw_query_results=raw,
            retriever=mock_retriever, language="en", config=config,
            num_results_per_chunk=10, chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0, include_details=False,
        )

        assert mock_retriever.query_batch.call_count == 1
        assert out.meta["max_depth_reached"] == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py -v`
Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Implement `AdaptiveRechunkingResult` and `run_adaptive_rechunking`**

Append to `phentrieve/retrieval/adaptive_rechunker.py`:

```python
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveRechunkingResult:
    """Return value of run_adaptive_rechunking."""

    processed_chunks: list[dict[str, Any]]
    aggregated_results: list[dict[str, Any]]
    chunk_results: list[dict[str, Any]]
    meta: dict[str, Any]


def run_adaptive_rechunking(
    processed_chunks: list[dict[str, Any]],
    chunk_results: list[dict[str, Any]],
    raw_query_results: list[dict[str, Any]],
    retriever: Any,
    language: str,
    config: AdaptiveRechunkingConfig,
    num_results_per_chunk: int,
    chunk_retrieval_threshold: float,
    min_confidence_for_aggregated: float,
    include_details: bool,
    assertion_statuses: list[str | None] | None = None,
) -> AdaptiveRechunkingResult:
    """Top-level adaptive-rechunking orchestration.

    Cost contract: at most max_depth additional retriever.query_batch calls
    (one per recursion level where chunks flag as poor). Re-aggregation
    uses orchestrate_hpo_extraction's precomputed_query_results parameter
    so existing chunk results are not re-queried.
    """
    meta: dict[str, Any] = {
        "enabled": config.enabled,
        "trigger_count": 0,
        "subdivided_count": 0,
        "reverted_count": 0,
        "max_depth_reached": 0,
        "extra_chunks_added": 0,
    }

    if not config.enabled:
        # Run aggregation once over the supplied raw results (no retrieval).
        return _no_op_result(processed_chunks, chunk_results, meta)

    # Detect poor chunks at depth 0 (the initial pass).
    quality = [
        assess_chunk_quality(raw, idx, chunk_retrieval_threshold, config)
        for idx, raw in enumerate(raw_query_results)
    ]
    poor_indices = [s.chunk_idx for s in quality if s.is_poor]
    meta["trigger_count"] = len(poor_indices)

    if not poor_indices:
        return AdaptiveRechunkingResult(
            processed_chunks=list(processed_chunks),
            aggregated_results=_aggregate(
                processed_chunks, raw_query_results, retriever,
                num_results_per_chunk, chunk_retrieval_threshold,
                language, min_confidence_for_aggregated, include_details,
                assertion_statuses,
            ).aggregated_results,
            chunk_results=list(chunk_results),
            meta=meta,
        )

    # Recursion loop.
    current_chunks = list(processed_chunks)
    current_raw = list(raw_query_results)
    current_assertions = list(assertion_statuses or [c.get("status") for c in current_chunks])

    for depth in range(1, config.max_depth + 1):
        # Identify chunks still flagged as poor at this depth.
        depth_quality = [
            assess_chunk_quality(raw, idx, chunk_retrieval_threshold, config)
            for idx, raw in enumerate(current_raw)
        ]
        depth_poor = [s.chunk_idx for s in depth_quality if s.is_poor]
        if not depth_poor:
            break

        # Subdivide each poor parent.
        parent_to_children: dict[int, list[int]] = {}
        new_chunks: list[dict[str, Any]] = []
        new_assertions: list[str | None] = []
        new_raw: list[dict[str, Any]] = []
        children_texts: list[str] = []
        # Track parent → child slot in new flat list (we'll fill child raw later).
        child_slots: dict[int, list[int]] = {}

        for idx, parent in enumerate(current_chunks):
            if idx not in depth_poor:
                # Keep as-is.
                slot = len(new_chunks)
                new_chunks.append(parent)
                new_assertions.append(current_assertions[idx])
                new_raw.append(current_raw[idx])
                continue
            children = subdivide_parent_chunk(parent, language, config, depth)
            if not children:
                # Subdivision wasn't possible — keep parent.
                slot = len(new_chunks)
                new_chunks.append(parent)
                new_assertions.append(current_assertions[idx])
                new_raw.append(current_raw[idx])
                continue
            # Reserve slots for children; raw filled after batch query below.
            child_slot_indices: list[int] = []
            for child in children:
                child_slot_indices.append(len(new_chunks))
                new_chunks.append(child)
                new_assertions.append(child.get("status"))
                new_raw.append({"similarities": [[]], "ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]]})  # placeholder
                children_texts.append(child["text"])
            child_slots[idx] = child_slot_indices

        if not children_texts:
            break  # Nothing to query.

        # ONE query_batch call per recursion level (the cost-model invariant).
        child_raw = retriever.query_batch(
            texts=children_texts,
            n_results=num_results_per_chunk,
            include_similarities=True,
        )
        meta["max_depth_reached"] = depth

        # Fill in placeholders.
        ci = 0
        for parent_idx, slot_indices in child_slots.items():
            for slot in slot_indices:
                new_raw[slot] = child_raw[ci]
                ci += 1

        # Apply score-improvement gate.
        parent_top_1 = {
            idx: (current_raw[idx].get("similarities", [[]])[0] or [None])[0]
            for idx in child_slots
        }
        child_top_1: dict[int, float] = {}
        for slot_indices in child_slots.values():
            for slot in slot_indices:
                sims = new_raw[slot].get("similarities", [[]])[0]
                if sims:
                    child_top_1[slot] = sims[0]

        revert, keep = apply_score_improvement_gate(
            child_slots, parent_top_1, child_top_1, config
        )

        if revert:
            # Rebuild without reverted children.
            keep_chunks: list[dict[str, Any]] = []
            keep_assertions: list[str | None] = []
            keep_raw: list[dict[str, Any]] = []
            for idx, parent in enumerate(current_chunks):
                if idx in revert:
                    # Restore parent.
                    keep_chunks.append(parent)
                    keep_assertions.append(current_assertions[idx])
                    keep_raw.append(current_raw[idx])
                elif idx in keep:
                    # Append children.
                    for slot in child_slots[idx]:
                        keep_chunks.append(new_chunks[slot])
                        keep_assertions.append(new_assertions[slot])
                        keep_raw.append(new_raw[slot])
                else:
                    # Untouched (was not flagged).
                    keep_chunks.append(parent)
                    keep_assertions.append(current_assertions[idx])
                    keep_raw.append(current_raw[idx])
            new_chunks, new_assertions, new_raw = keep_chunks, keep_assertions, keep_raw

        meta["subdivided_count"] += len(keep)
        meta["reverted_count"] += len(revert)

        current_chunks = new_chunks
        current_assertions = new_assertions
        current_raw = new_raw

        if not keep:
            break  # All subdivisions reverted — no point recursing.

    meta["extra_chunks_added"] = len(current_chunks) - len(processed_chunks)

    # Final re-aggregation using precomputed_query_results — no retrieval call.
    final = _aggregate(
        current_chunks, current_raw, retriever,
        num_results_per_chunk, chunk_retrieval_threshold,
        language, min_confidence_for_aggregated, include_details,
        current_assertions,
    )

    return AdaptiveRechunkingResult(
        processed_chunks=current_chunks,
        aggregated_results=final.aggregated_results,
        chunk_results=final.chunk_results,
        meta=meta,
    )


def _no_op_result(processed_chunks, chunk_results, meta):
    return AdaptiveRechunkingResult(
        processed_chunks=list(processed_chunks),
        aggregated_results=[],   # caller already has these from the initial extraction
        chunk_results=list(chunk_results),
        meta=meta,
    )


def _aggregate(
    chunks, raw, retriever, num_results_per_chunk, chunk_retrieval_threshold,
    language, min_confidence_for_aggregated, include_details, assertion_statuses,
):
    """Re-aggregate via orchestrate_hpo_extraction with precomputed raw results."""
    from phentrieve.text_processing.hpo_extraction_orchestrator import (
        orchestrate_hpo_extraction,
    )
    text_chunks = [c.get("text", "") for c in chunks]
    return orchestrate_hpo_extraction(
        text_chunks=text_chunks,
        retriever=retriever,
        num_results_per_chunk=num_results_per_chunk,
        chunk_retrieval_threshold=chunk_retrieval_threshold,
        language=language,
        min_confidence_for_aggregated=min_confidence_for_aggregated,
        assertion_statuses=assertion_statuses,
        include_details=include_details,
        precomputed_query_results=raw,  # KEY: skips query_batch!
    )


def dump_quality_report(
    raw_query_results: list[dict[str, Any]],
    chunk_retrieval_threshold: float,
    config: AdaptiveRechunkingConfig,
) -> str:
    """Library-only helper for users tuning thresholds. Returns a per-chunk
    quality report as a human-readable string."""
    lines = ["chunk_idx  is_poor  reason       top_1  top_2  margin"]
    for idx, raw in enumerate(raw_query_results):
        s = assess_chunk_quality(raw, idx, chunk_retrieval_threshold, config)
        t1 = f"{s.top_1:.3f}" if s.top_1 is not None else "  -  "
        t2 = f"{s.top_2:.3f}" if s.top_2 is not None else "  -  "
        m = f"{s.margin:.3f}" if s.margin is not None else "  -  "
        lines.append(f"{idx:>9}  {s.is_poor!s:<6}  {s.reason:<11}  {t1}  {t2}  {m}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py -v`
Expected: PASS, five tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/retrieval/adaptive_rechunker.py tests/unit/retrieval/adaptive_rechunker/test_run_adaptive_rechunking.py
git commit -m "feat(adaptive): run_adaptive_rechunking orchestration

Loop: detect poor chunks via raw scores, subdivide into sentence-window
children, batch-query children once per depth, apply score-improvement
gate, rebuild flat list with kept-children + reverted-parents, re-aggregate
via orchestrate_hpo_extraction(precomputed_query_results=...) — skipping
the retrieval call. Cost invariant: 1 query_batch per recursion level."
```

---

## Phase 7: Wire into `run_standard_backend`

### Task 10: Modify `full_text_service.py:run_standard_backend`

**Files:**
- Modify: `phentrieve/text_processing/full_text_service.py`
- Test: integration coverage in Task 16

- [ ] **Step 1: Update `run_standard_backend` to consume the new dataclass**

Find `run_standard_backend` at lines 536-617 and change the orchestration block:

Replace:

```python
processed_chunks = text_pipeline.process(text, include_positions=include_positions)
if not processed_chunks:
    return adapt_standard_response([], ([], []))

text_chunks = [chunk["text"] for chunk in processed_chunks]
assertion_statuses: list[str | None] = [
    _normalize_status(chunk.get("status")) for chunk in processed_chunks
]

aggregated_results, chunk_results = orchestrate_hpo_extraction(
    text_chunks=text_chunks,
    retriever=retriever,
    num_results_per_chunk=kwargs.pop("num_results_per_chunk", 10),
    chunk_retrieval_threshold=kwargs.pop(
        "chunk_retrieval_threshold", DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
    ),
    language=language,
    top_term_per_chunk=kwargs.pop("top_term_per_chunk", False),
    min_confidence_for_aggregated=kwargs.pop(
        "min_confidence_for_aggregated", DEFAULT_MIN_CONFIDENCE_AGGREGATED
    ),
    assertion_statuses=assertion_statuses,
    include_details=kwargs.pop("include_details", False),
)

return adapt_standard_response(
    processed_chunks, (aggregated_results, chunk_results)
)
```

with:

```python
processed_chunks = text_pipeline.process(text, include_positions=include_positions)
if not processed_chunks:
    return adapt_standard_response([], ([], []))

text_chunks = [chunk["text"] for chunk in processed_chunks]
assertion_statuses: list[str | None] = [
    _normalize_status(chunk.get("status")) for chunk in processed_chunks
]

# Pop kwargs once so the values are reusable across initial + adaptive paths.
num_results_per_chunk = kwargs.pop("num_results_per_chunk", 10)
chunk_retrieval_threshold = kwargs.pop(
    "chunk_retrieval_threshold", DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
)
top_term_per_chunk = kwargs.pop("top_term_per_chunk", False)
min_confidence_for_aggregated = kwargs.pop(
    "min_confidence_for_aggregated", DEFAULT_MIN_CONFIDENCE_AGGREGATED
)
include_details = kwargs.pop("include_details", False)

orchestration = orchestrate_hpo_extraction(
    text_chunks=text_chunks,
    retriever=retriever,
    num_results_per_chunk=num_results_per_chunk,
    chunk_retrieval_threshold=chunk_retrieval_threshold,
    language=language,
    top_term_per_chunk=top_term_per_chunk,
    min_confidence_for_aggregated=min_confidence_for_aggregated,
    assertion_statuses=assertion_statuses,
    include_details=include_details,
)
aggregated_results = orchestration.aggregated_results
chunk_results = orchestration.chunk_results
raw_query_results = orchestration.raw_query_results

# Adaptive rechunking — opt-in.
adaptive_meta: dict[str, Any] | None = None
adaptive_cfg = kwargs.pop("adaptive_rechunking", None)
if adaptive_cfg is not None and getattr(adaptive_cfg, "enabled", False):
    from phentrieve.retrieval.adaptive_rechunker import run_adaptive_rechunking

    rechunk = run_adaptive_rechunking(
        processed_chunks=processed_chunks,
        chunk_results=chunk_results,
        raw_query_results=raw_query_results,
        retriever=retriever,
        language=language,
        config=adaptive_cfg,
        num_results_per_chunk=num_results_per_chunk,
        chunk_retrieval_threshold=chunk_retrieval_threshold,
        min_confidence_for_aggregated=min_confidence_for_aggregated,
        include_details=include_details,
        assertion_statuses=assertion_statuses,
    )
    processed_chunks = rechunk.processed_chunks
    aggregated_results = rechunk.aggregated_results
    chunk_results = rechunk.chunk_results
    adaptive_meta = rechunk.meta

return adapt_standard_response(
    processed_chunks,
    (aggregated_results, chunk_results),
    extra_meta={"adaptive_rechunking": adaptive_meta} if adaptive_meta else None,
)
```

- [ ] **Step 2: Run a quick smoke import**

Run: `uv run python -c "from phentrieve.text_processing.full_text_service import run_standard_backend; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Run existing full_text_service tests**

Run: `uv run pytest tests/ -x -q -k "full_text_service or text_processing_router" --ignore=tests/integration`
Expected: PASS — new third tuple element + extra_meta parameter are backward-compat.

- [ ] **Step 4: Commit**

```bash
git add phentrieve/text_processing/full_text_service.py
git commit -m "feat(adaptive): wire run_adaptive_rechunking into run_standard_backend

The seam: between orchestrate_hpo_extraction (initial pass) and
adapt_standard_response (output building). Threads adaptive_meta
through extra_meta so meta.adaptive_rechunking surfaces in the response.
No behavior change when adaptive_rechunking config is absent or disabled."
```

---

## Phase 8: CLI flags

### Task 11: Add four flags to `phentrieve text process`

**Files:**
- Modify: `phentrieve/cli/text_commands.py`
- Test: extend `tests/unit/cli/test_text_commands.py`

- [ ] **Step 1: Append the flag tests**

Append to `tests/unit/cli/test_text_commands.py`:

```python
class TestProcessAdaptiveRechunkingFlags:
    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_disabled_by_default(self, mock_run, tmp_path, monkeypatch):
        from phentrieve.__main__ import app
        from phentrieve.config import _load_yaml_config

        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()
        mock_run.return_value = {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

        (tmp_path / "in.txt").write_text("text")
        runner = CliRunner()
        result = runner.invoke(app, ["text", "process", str(tmp_path / "in.txt")])
        assert result.exit_code == 0
        cfg = mock_run.call_args.kwargs.get("adaptive_rechunking")
        # When the flag isn't passed, the config is built with enabled=False
        # (or absent — both acceptable).
        if cfg is not None:
            assert getattr(cfg, "enabled", False) is False

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_enabled_via_flag(self, mock_run, tmp_path, monkeypatch):
        from phentrieve.__main__ import app
        from phentrieve.config import _load_yaml_config

        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()
        mock_run.return_value = {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

        (tmp_path / "in.txt").write_text("text")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["text", "process", str(tmp_path / "in.txt"), "--adaptive-rechunking"],
        )
        assert result.exit_code == 0
        cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
        assert cfg.enabled is True

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_threshold_flags(self, mock_run, tmp_path, monkeypatch):
        from phentrieve.__main__ import app
        from phentrieve.config import _load_yaml_config

        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()
        mock_run.return_value = {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

        (tmp_path / "in.txt").write_text("text")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "text", "process", str(tmp_path / "in.txt"),
                "--adaptive-rechunking",
                "--adaptive-rechunking-quality-threshold", "0.5",
                "--adaptive-rechunking-margin-threshold", "0.05",
                "--adaptive-rechunking-max-depth", "1",
            ],
        )
        assert result.exit_code == 0
        cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
        assert cfg.enabled is True
        assert cfg.quality_threshold == 0.5
        assert cfg.margin_threshold == 0.05
        assert cfg.max_depth == 1
```

- [ ] **Step 2: Add flags to `process_text_for_hpo_command`**

In `phentrieve/cli/text_commands.py`, add the four new options to the function signature (placement near the end of the parameter list):

```python
adaptive_rechunking: Annotated[
    bool,
    typer.Option(
        "--adaptive-rechunking/--no-adaptive-rechunking",
        help="Enable adaptive re-chunking (opt-in feature, see issue #148).",
    ),
] = False,
adaptive_rechunking_quality_threshold: Annotated[
    float | None,
    typer.Option(
        "--adaptive-rechunking-quality-threshold",
        help="top-1 similarity below which a chunk flags as poor.",
    ),
] = None,
adaptive_rechunking_margin_threshold: Annotated[
    float | None,
    typer.Option(
        "--adaptive-rechunking-margin-threshold",
        help="top_1 - top_2 below which a chunk flags as poor (with low score).",
    ),
] = None,
adaptive_rechunking_max_depth: Annotated[
    int | None,
    typer.Option(
        "--adaptive-rechunking-max-depth",
        help="Recursion depth cap (default 2).",
    ),
] = None,
```

Add this import at the top of `phentrieve/cli/text_commands.py`:

```python
from phentrieve.config import _load_yaml_config
from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    adaptive_config_from_profile_block,
)
```

In the function body of `process_text_for_hpo_command`, build the config and pass it via `run_full_text_service`:

```python
# Resolve config from CLI > profile > YAML > defaults.
# (Profile-block path layers in via Plan A's apply_profile_callback once Plan A lands.)
adaptive_yaml = (
    _load_yaml_config().get("extraction", {}).get("adaptive_rechunking", {})
)
cli_overrides: dict[str, Any] = {"enabled": adaptive_rechunking}
if adaptive_rechunking_quality_threshold is not None:
    cli_overrides["quality_threshold"] = adaptive_rechunking_quality_threshold
if adaptive_rechunking_margin_threshold is not None:
    cli_overrides["margin_threshold"] = adaptive_rechunking_margin_threshold
if adaptive_rechunking_max_depth is not None:
    cli_overrides["max_depth"] = adaptive_rechunking_max_depth

adaptive_config = adaptive_config_from_profile_block(
    block=None,
    yaml_block=adaptive_yaml,
    cli_overrides=cli_overrides,
)
```

Pass `adaptive_rechunking=adaptive_config` to `run_full_text_service(...)`.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/cli/test_text_commands.py::TestProcessAdaptiveRechunkingFlags -v`
Expected: PASS, three tests.

- [ ] **Step 4: Commit**

```bash
git add phentrieve/cli/text_commands.py tests/unit/cli/test_text_commands.py
git commit -m "feat(cli): add --adaptive-rechunking* flags to text process

Four new flags: --adaptive-rechunking (boolean enable),
--adaptive-rechunking-quality-threshold, --adaptive-rechunking-margin-threshold,
--adaptive-rechunking-max-depth. Resolves config from CLI > YAML >
defaults; profile resolution layers in via Plan A's apply_profile_callback."
```

---

## Phase 9: API parity

### Task 12: Add `adaptive_rechunking` to API request schema

**Files:**
- Modify: `api/schemas/text_processing_schemas.py`
- Modify: `api/routers/text_processing_router.py`
- Create: `tests/integration/test_adaptive_rechunking_api.py`

- [ ] **Step 1: Add field to `TextProcessingRequest`**

In `api/schemas/text_processing_schemas.py`, add the new optional field (around the existing fields):

```python
from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingProfileBlock


class TextProcessingRequest(BaseModel):
    # ... existing fields ...
    adaptive_rechunking: AdaptiveRechunkingProfileBlock | None = Field(
        default=None,
        description="Optional adaptive re-chunking configuration.",
    )
```

- [ ] **Step 2: Pass through in the router**

In `api/routers/text_processing_router.py`, find where `run_full_text_service` is called and add the field:

```python
adaptive_config = None
if request.adaptive_rechunking is not None:
    from phentrieve.retrieval.adaptive_rechunker import (
        adaptive_config_from_profile_block,
    )
    adaptive_config = adaptive_config_from_profile_block(
        block=request.adaptive_rechunking,
        yaml_block=None,
        cli_overrides=None,
    )

response = run_full_text_service(
    text=request.text,
    extraction_backend=request.extraction_backend,
    # ... existing kwargs ...
    adaptive_rechunking=adaptive_config,
)
```

- [ ] **Step 3: Write API integration test**

Create `tests/integration/test_adaptive_rechunking_api.py`:

```python
"""API parity: meta.adaptive_rechunking surfaces in the response."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app  # adjust if app lives elsewhere
    return TestClient(app)


class TestAdaptiveRechunkingAPI:
    @patch("api.routers.text_processing_router.run_full_text_service")
    def test_request_with_adaptive_rechunking_passes_through(self, mock_run, client):
        mock_run.return_value = {
            "meta": {"adaptive_rechunking": {"enabled": True, "trigger_count": 1}},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        response = client.post(
            "/api/text/process",
            json={
                "text": "Patient with seizures.",
                "adaptive_rechunking": {"enabled": True, "quality_threshold": 0.5},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["adaptive_rechunking"]["enabled"] is True

    @patch("api.routers.text_processing_router.run_full_text_service")
    def test_request_without_adaptive_no_meta_block(self, mock_run, client):
        mock_run.return_value = {
            "meta": {},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
        response = client.post("/api/text/process", json={"text": "Patient."})
        assert response.status_code == 200
        body = response.json()
        assert "adaptive_rechunking" not in body.get("meta", {})
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/integration/test_adaptive_rechunking_api.py -v`
Expected: PASS, two tests.

- [ ] **Step 5: Commit**

```bash
git add api/schemas/text_processing_schemas.py api/routers/text_processing_router.py tests/integration/test_adaptive_rechunking_api.py
git commit -m "feat(api): adaptive_rechunking request field and meta response

TextProcessingRequest gains optional adaptive_rechunking block;
text_processing_router passes it through. Response includes
meta.adaptive_rechunking when enabled. Without the field, response
shape is unchanged."
```

---

## Phase 10: Frontend integration

### Task 13: Frontend payload pass-through and response handling

**Files:**
- Modify: `frontend/src/services/PhentrieveService.js`
- Modify: `frontend/src/test/services/PhentrieveService.test.js`

- [ ] **Step 1: Append payload-pass-through test**

Append to `frontend/src/test/services/PhentrieveService.test.js`:

```js
describe('adaptive_rechunking pass-through', () => {
  it('forwards adaptive_rechunking when supplied', async () => {
    const fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        meta: { adaptive_rechunking: { enabled: true, trigger_count: 2 } },
        processed_chunks: [],
        aggregated_hpo_terms: [],
      }),
    });
    global.fetch = fetchMock;

    const service = new PhentrieveService();
    await service.processText({
      text: 'Patient.',
      adaptive_rechunking: { enabled: true, quality_threshold: 0.5 },
    });

    const payload = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(payload.adaptive_rechunking).toEqual({
      enabled: true,
      quality_threshold: 0.5,
    });
  });

  it('parses meta.adaptive_rechunking response without error', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        meta: { adaptive_rechunking: { enabled: true, trigger_count: 1 } },
        processed_chunks: [],
        aggregated_hpo_terms: [],
      }),
    });
    const service = new PhentrieveService();
    const result = await service.processText({ text: 'Patient.' });
    expect(result.meta.adaptive_rechunking.enabled).toBe(true);
  });
});
```

- [ ] **Step 2: Update `PhentrieveService.js`**

In `frontend/src/services/PhentrieveService.js`, find the `processText` method (or its equivalent that POSTs to `/api/text/process`). If it accepts a payload object, ensure `adaptive_rechunking` is preserved:

```js
async processText(payload) {
  const body = JSON.stringify({
    text: payload.text,
    extraction_backend: payload.extraction_backend,
    // ... other existing fields ...
    adaptive_rechunking: payload.adaptive_rechunking,  // NEW: pass-through
  });
  const response = await fetch(`${this.baseUrl}/api/text/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return await response.json();  // includes meta.adaptive_rechunking when enabled
}
```

(If the existing implementation already uses object spread, it already passes `adaptive_rechunking` through transparently. Verify by inspection.)

- [ ] **Step 3: Run frontend tests**

Run: `make frontend-test-ci`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/PhentrieveService.js frontend/src/test/services/PhentrieveService.test.js
git commit -m "feat(frontend): pass-through adaptive_rechunking in PhentrieveService

Minimal v1 frontend change: payload pass-through + graceful response
parsing of meta.adaptive_rechunking. No UI surface for the toggle or
metadata panel — those are deferred (Spec B Future work)."
```

---

## Phase 11: YAML and documentation

### Task 14: YAML template + user-guide

**Files:**
- Modify: `phentrieve.yaml.template`
- Modify: `phentrieve.yaml`
- Create: `docs/user-guide/adaptive-rechunking.md`
- Modify: `docs/user-guide/cli-usage.md`
- Modify: `docs/user-guide/api-usage.md`
- Modify: `docs/user-guide/text-processing-guide.md`
- Modify: `docs/user-guide/index.md`
- Modify: `README.md`

- [ ] **Step 1: Update phentrieve.yaml.template**

Append:

```yaml
#   adaptive_rechunking:
#     enabled: false                 # opt-in
#     quality_threshold: 0.55        # encoder-calibrated for BioLORD
#     margin_threshold: 0.03
#     use_ontology_coherence: false  # reserved, inert in v1
#     max_depth: 2
#     min_chunk_chars: 30
#     max_sentences_per_subchunk: 3
#     overlap_sentences: 1
#     score_improvement_gate: 0.05
```

(Place under the existing `extraction:` section comment block.)

- [ ] **Step 2: Create docs/user-guide/adaptive-rechunking.md**

```markdown
# Adaptive Re-Chunking

Optional retrieval-quality-driven sub-chunking. When a chunk's retrieval is poor
(low top-1 similarity AND low margin), Phentrieve subdivides the chunk into
sentence-bounded sub-chunks, re-queries each, and merges the results. Improves
recall on multi-concept clinical sentences without affecting users who don't
enable it.

## When to Enable

Enable adaptive re-chunking when:
- Your inputs include long multi-finding paragraphs.
- You want to maximize recall and accept up to ~1.5× retrieval cost.

Skip it when:
- Latency matters more than recall (e.g. interactive mode).
- Your inputs are typically short single-finding sentences.

## Quick Start

```bash
phentrieve text process note.txt --adaptive-rechunking
```

Or in `phentrieve.yaml`:

```yaml
extraction:
  adaptive_rechunking:
    enabled: true
    quality_threshold: 0.55
    max_depth: 2
```

## How It Works

1. Initial extraction runs as usual.
2. For each chunk, the trigger evaluates `top_1 < quality_threshold AND
   (margin < margin_threshold OR top_2 is None)`.
3. Flagged chunks are subdivided at sentence boundaries.
4. Sub-chunks are queried in a single batch.
5. The score-improvement gate compares each parent's `top_1` to the best child's
   `top_1`; subdivisions that don't improve by `score_improvement_gate` are reverted.
6. Re-aggregation uses the precomputed query results — no re-queries.
7. Up to `max_depth` recursion levels.

## Configuration

| Knob | Default | Notes |
|---|---|---|
| `enabled` | `false` | Opt-in. |
| `quality_threshold` | `0.55` | top-1 similarity floor. Encoder-calibrated for BioLORD. |
| `margin_threshold` | `0.03` | top-1 minus top-2 floor. |
| `max_depth` | `2` | Recursion cap. |
| `min_chunk_chars` | `30` | Sub-chunks shorter than this are dropped. |
| `max_sentences_per_subchunk` | `3` | Window size for sentence grouping. |
| `overlap_sentences` | `1` | Sentence-level overlap between sub-chunk windows. |
| `score_improvement_gate` | `0.05` | Subdivisions that don't lift top-1 by this much are reverted. |
| `use_ontology_coherence` | `false` | Reserved, inert in v1. |

## Encoder Calibration Warning

Default thresholds are calibrated for BioLORD-class biomedical encoders. If you
switch to a different `retrieval_model`, the score distribution will differ and
you should retune `quality_threshold` and `margin_threshold`. A future
`phentrieve config calibrate-thresholds` subcommand will help with this.

## Cost Envelope

- Typical: 1.2–1.5× retrieval cost.
- Worst case at `max_depth=2` with all chunks recursing: encoder workload can
  reach ~13× the original since each recursion level encodes more chunks. The
  number of `query_batch` RPC calls is hard-bounded at `1 + max_depth` (worst
  case 3 calls at `max_depth=2`).

## Examples

Aggressive recall:

```yaml
extraction:
  adaptive_rechunking:
    enabled: true
    quality_threshold: 0.6
    max_depth: 2
```

Conservative cost:

```yaml
extraction:
  adaptive_rechunking:
    enabled: true
    max_depth: 1
    score_improvement_gate: 0.1   # require larger improvements
```

Cross-language (German):

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
```

- [ ] **Step 3: Update other docs**

In `docs/user-guide/cli-usage.md` under the `text process` option table, add:

```markdown
| `--adaptive-rechunking` / `--no-adaptive-rechunking` | flag | Enable adaptive re-chunking. See [Adaptive Re-Chunking](./adaptive-rechunking.md). |
| `--adaptive-rechunking-quality-threshold` | float | Override the quality threshold (default 0.55). |
| `--adaptive-rechunking-margin-threshold` | float | Override the margin threshold (default 0.03). |
| `--adaptive-rechunking-max-depth` | int | Override the recursion cap (default 2). |
```

In `docs/user-guide/api-usage.md`, add:

```markdown
### Adaptive re-chunking

Pass `adaptive_rechunking` in the request body to enable adaptive re-chunking:

\`\`\`json
{
  "text": "...",
  "adaptive_rechunking": { "enabled": true, "quality_threshold": 0.5 }
}
\`\`\`

The response includes `meta.adaptive_rechunking` with trigger/subdivided/reverted
counts when the feature is enabled. See [Adaptive Re-Chunking](./adaptive-rechunking.md).
```

In `docs/user-guide/text-processing-guide.md`, add a one-paragraph cross-reference at the end of the chunking-strategy section:

```markdown
### Adaptive re-chunking (opt-in)

When initial retrieval on a chunk is poor, Phentrieve can optionally subdivide
the chunk into sentence-window sub-chunks and re-query. See
[Adaptive Re-Chunking](./adaptive-rechunking.md).
```

In `docs/user-guide/index.md`, add the link.

In `README.md`, under the existing "Features" or similar section, add a one-liner:

```markdown
- Optional adaptive re-chunking improves recall on multi-concept clinical
  sentences (`--adaptive-rechunking`). See
  [docs/user-guide/adaptive-rechunking.md](docs/user-guide/adaptive-rechunking.md).
```

- [ ] **Step 4: Verify YAML snippets parse**

The shared `tests/integration/test_documented_yaml.py` (created by Plan A Task 22) will pick up the new YAML snippets. If Plan A hasn't landed yet, create the file as a stub here:

```python
# (Use the same content as Plan A Task 22.)
```

Run: `uv run pytest tests/integration/test_documented_yaml.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add phentrieve.yaml.template docs/user-guide/adaptive-rechunking.md docs/user-guide/cli-usage.md docs/user-guide/api-usage.md docs/user-guide/text-processing-guide.md docs/user-guide/index.md README.md
git commit -m "docs(adaptive): user guide for adaptive re-chunking

Canonical reference page covering how it works, configuration knobs,
encoder calibration warning, cost envelope, and three worked examples.
Cross-references in cli-usage.md, api-usage.md, text-processing-guide.md,
index.md, README.md, phentrieve.yaml.template."
```

---

## Phase 12: Integration tests

### Task 15: End-to-end test with real fixtures

**Files:**
- Create: `tests/fixtures/adaptive_rechunking/synthetic_multi_finding.txt`
- Create: `tests/fixtures/adaptive_rechunking/expected_terms_with_adaptive.json`
- Create: `tests/integration/test_adaptive_rechunking_e2e.py`

- [ ] **Step 1: Create the fixture text**

Create `tests/fixtures/adaptive_rechunking/synthetic_multi_finding.txt`:

```text
The patient has severe global developmental delay, frequent generalized tonic-clonic seizures since age 2, and bilateral sensorineural hearing loss. Brain MRI revealed cortical atrophy and bilateral periventricular nodular heterotopia. The mother and maternal grandmother are reported to have similar findings.
```

Create a placeholder `expected_terms_with_adaptive.json`:

```json
{
  "_note": "Populated empirically by running the e2e test against the test ChromaDB index. The test asserts at least one HPO term not present in the no-adaptive baseline.",
  "minimum_extra_terms": 1
}
```

- [ ] **Step 2: Write the e2e test**

Create `tests/integration/test_adaptive_rechunking_e2e.py`:

```python
"""End-to-end test: adaptive rechunking improves recall on a synthetic
multi-finding fixture against a real (or mocked) ChromaDB index."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_TXT = REPO_ROOT / "tests" / "fixtures" / "adaptive_rechunking" / "synthetic_multi_finding.txt"


@pytest.mark.integration
def test_adaptive_rechunking_finds_more_terms(tmp_path):
    """With adaptive on, the aggregated terms include at least one HPO term
    not surfaced by the no-adaptive baseline."""
    pytest.importorskip("chromadb")

    from phentrieve.text_processing.full_text_service import run_standard_backend
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig

    text = FIXTURE_TXT.read_text()

    # No-adaptive baseline.
    baseline = run_standard_backend(text=text, language="en")
    baseline_ids = {t["id"] for t in baseline["aggregated_hpo_terms"]}

    # Adaptive on with permissive threshold to ensure trigger fires.
    config = AdaptiveRechunkingConfig(
        enabled=True, quality_threshold=0.7, margin_threshold=0.1,
        max_depth=2, min_chunk_chars=20,
    )
    adaptive = run_standard_backend(
        text=text, language="en", adaptive_rechunking=config
    )
    adaptive_ids = {t["id"] for t in adaptive["aggregated_hpo_terms"]}

    extra = adaptive_ids - baseline_ids
    # At least one term gained.
    assert len(extra) >= 1, (
        f"Adaptive rechunking did not surface any new terms. "
        f"Baseline: {baseline_ids}. Adaptive: {adaptive_ids}."
    )

    # Meta block populated.
    assert "adaptive_rechunking" in adaptive["meta"]
    assert adaptive["meta"]["adaptive_rechunking"]["enabled"] is True
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/integration/test_adaptive_rechunking_e2e.py -v`
Expected: PASS (or SKIP if no ChromaDB index is built locally).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/adaptive_rechunking/ tests/integration/test_adaptive_rechunking_e2e.py
git commit -m "test(integration): adaptive rechunking finds more terms on multi-finding fixture

Synthetic clinical text with multiple co-occurring phenotypes that
typically trigger the quality gate. Asserts at least one HPO term
appears in the adaptive run that was missed by the no-adaptive baseline."
```

### Task 16: Call-count invariant + perf smoke

**Files:**
- Create: `tests/integration/test_adaptive_rechunking_call_count.py`
- Create: `tests/integration/test_adaptive_rechunking_perf.py`

- [ ] **Step 1: Write the call-count test**

Create `tests/integration/test_adaptive_rechunking_call_count.py`:

```python
"""HARD CONTRACT: at most max_depth additional retriever.query_batch calls."""

from unittest.mock import MagicMock

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    run_adaptive_rechunking,
)


def make_raw(similarities: list[float]) -> dict:
    return {
        "ids": [[f"HP:{i:07d}" for i in range(len(similarities))]],
        "metadatas": [[{"id": f"HP:{i:07d}", "label": "x"} for i in range(len(similarities))]],
        "similarities": [similarities],
        "distances": [[1 - s for s in similarities]],
        "documents": [[""] * len(similarities)],
    }


def test_at_most_max_depth_query_batch_calls():
    """Worst case: every chunk flags at every level. Hard cap = max_depth calls."""
    retriever = MagicMock()
    # Simulate every level still flagging poor.
    retriever.query_batch.side_effect = [
        # depth 1
        [make_raw([0.5, 0.49]) for _ in range(3)],  # 3 children, all still poor
        # depth 2
        [make_raw([0.55, 0.54]) for _ in range(3)],  # grandchildren — improve to keep
    ]

    processed = [{
        "text": "First sentence. Second sentence. Third sentence.",
        "status": "AFFIRMED", "start_char": 0, "end_char": 49,
        "source_indices": {"processing_stages": ["sentence"]},
    }]
    chunk_results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
    raw = [make_raw([0.4, 0.39])]

    config = AdaptiveRechunkingConfig(
        enabled=True, max_depth=2,
        quality_threshold=0.6, margin_threshold=0.03,
        score_improvement_gate=0.05,
        max_sentences_per_subchunk=1, overlap_sentences=0,
        min_chunk_chars=5,
    )

    run_adaptive_rechunking(
        processed_chunks=processed, chunk_results=chunk_results, raw_query_results=raw,
        retriever=retriever, language="en", config=config,
        num_results_per_chunk=10, chunk_retrieval_threshold=0.7,
        min_confidence_for_aggregated=0.0, include_details=False,
    )

    # The hard contract.
    assert retriever.query_batch.call_count <= config.max_depth, (
        f"query_batch was called {retriever.query_batch.call_count} times; "
        f"max_depth={config.max_depth} should be the cap. "
        f"This fails if precomputed_query_results regression re-queries parents."
    )
```

- [ ] **Step 2: Write the perf smoke**

Create `tests/integration/test_adaptive_rechunking_perf.py`:

```python
"""Perf smoke test: wall time for a fully-flagging fixture document is
within a loose 5x bound. Designed to catch egregious regressions
(accidentally re-running the full pipeline), not to specify performance."""

import time

import pytest


@pytest.mark.integration
def test_wall_time_within_loose_bound():
    """5x is loose because encoder fan-out scales with sub-chunk count, not
    query-call count. The hard invariant is the call_count test."""
    pytest.importorskip("chromadb")

    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig
    from phentrieve.text_processing.full_text_service import run_standard_backend

    text = (
        "Patient has seizures. Patient has hearing loss. Patient has developmental delay. "
        "Patient has microcephaly. Patient has hypotonia. Patient has ataxia. "
        "Patient has spasticity. Patient has scoliosis. Patient has dysmorphic features. "
        "Patient has cognitive impairment."
    )

    t0 = time.perf_counter()
    run_standard_backend(text=text, language="en")
    baseline = time.perf_counter() - t0

    config = AdaptiveRechunkingConfig(
        enabled=True, quality_threshold=0.95,  # very aggressive — everything flags
        margin_threshold=0.5,
        max_depth=2, max_sentences_per_subchunk=2, overlap_sentences=0,
        min_chunk_chars=10,
    )
    t0 = time.perf_counter()
    run_standard_backend(text=text, language="en", adaptive_rechunking=config)
    adaptive = time.perf_counter() - t0

    multiplier = adaptive / baseline if baseline > 0 else 1.0
    assert multiplier <= 5.0, (
        f"Adaptive rechunking is {multiplier:.2f}× slower than baseline; "
        f"loose bound is 5×. Encoder fan-out can legitimately exceed query-call "
        f"multiplier, but exceeding 5× usually means a regression."
    )
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/integration/test_adaptive_rechunking_call_count.py tests/integration/test_adaptive_rechunking_perf.py -v`
Expected: PASS (perf may SKIP without ChromaDB).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_adaptive_rechunking_call_count.py tests/integration/test_adaptive_rechunking_perf.py
git commit -m "test(adaptive): cost-model invariant + perf smoke

Hard contract test: query_batch.call_count <= max_depth even when every
chunk flags at every level. The precomputed_query_results parameter is
what makes this bound tight; a regression that re-queries parents would
fail this. Plus a loose 5x perf bound that catches egregious regressions
without specifying performance."
```

### Task 17: Benchmark integration test

**Files:**
- Create: `tests/integration/test_adaptive_rechunking_benchmark.py`

- [ ] **Step 1: Write the benchmark test**

Create `tests/integration/test_adaptive_rechunking_benchmark.py`:

```python
"""Benchmark integration: adaptive rechunking does not regress ontology-aware
metrics. Uses the small German fixture from tests/data/benchmarks/german/tiny_v1.json.

The ontology-aware metrics (commit 9402a57) are the validation gate: adaptive
rechunking should improve or at least maintain MRR-with-LCA-credit on real data."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "data" / "benchmarks" / "german" / "tiny_v1.json"


@pytest.mark.integration
@pytest.mark.slow
def test_adaptive_does_not_regress_ontology_metric():
    pytest.importorskip("chromadb")
    if not FIXTURE.exists():
        pytest.skip("German tiny benchmark fixture missing")

    cases = json.loads(FIXTURE.read_text())
    # Run baseline + adaptive on the first few cases (smoke).
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig
    from phentrieve.text_processing.full_text_service import run_standard_backend

    baseline_terms_count = 0
    adaptive_terms_count = 0
    for case in cases[:3]:  # Keep test runtime bounded.
        text = case.get("text", "")
        if not text:
            continue
        baseline = run_standard_backend(text=text, language="de")
        adaptive_cfg = AdaptiveRechunkingConfig(enabled=True, max_depth=1)
        adaptive = run_standard_backend(text=text, language="de", adaptive_rechunking=adaptive_cfg)

        baseline_terms_count += len(baseline["aggregated_hpo_terms"])
        adaptive_terms_count += len(adaptive["aggregated_hpo_terms"])

    # Adaptive should not produce *fewer* terms on average. Loose check.
    assert adaptive_terms_count >= baseline_terms_count - 1, (
        f"Adaptive surfaced fewer terms ({adaptive_terms_count}) than baseline "
        f"({baseline_terms_count}); investigate before merging."
    )
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/integration/test_adaptive_rechunking_benchmark.py -v`
Expected: PASS or SKIP.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_adaptive_rechunking_benchmark.py
git commit -m "test(adaptive): benchmark does-not-regress check

Smoke test against the German tiny benchmark fixture. Adaptive on vs off
should not produce fewer aggregated terms. Full ontology-aware metric
validation is the release-gate manual run quoted in the spec; this test
is the CI tripwire."
```

### Task 18: Cross-spec interaction test

**Files:**
- Create: `tests/integration/test_specA_specB_interaction.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/test_specA_specB_interaction.py`:

```python
"""Cross-spec test: a profile setting adaptive_rechunking is properly
applied; explicit CLI flags override profile values."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config


@pytest.fixture(autouse=True)
def _yaml(tmp_path, monkeypatch):
    (tmp_path / "phentrieve.yaml").write_text(
        "profiles:\n"
        "  german_recall:\n"
        "    command: text process\n"
        "    language: de\n"
        "    adaptive_rechunking:\n"
        "      enabled: true\n"
        "      quality_threshold: 0.6\n"
    )
    monkeypatch.chdir(tmp_path)
    _load_yaml_config.cache_clear()
    yield
    _load_yaml_config.cache_clear()


@pytest.fixture
def app():
    from phentrieve.__main__ import app as a
    return a


@patch("phentrieve.cli.text_commands.run_full_text_service")
def test_profile_provides_adaptive_config(mock_run, app, tmp_path):
    mock_run.return_value = {
        "meta": {}, "processed_chunks": [], "aggregated_hpo_terms": [],
    }
    (tmp_path / "in.txt").write_text("Patient.")
    runner = CliRunner()
    result = runner.invoke(
        app, ["text", "process", str(tmp_path / "in.txt"), "--profile", "german_recall"]
    )
    assert result.exit_code == 0, result.output
    cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
    assert cfg.enabled is True
    assert cfg.quality_threshold == 0.6


@patch("phentrieve.cli.text_commands.run_full_text_service")
def test_explicit_flag_overrides_profile_adaptive(mock_run, app, tmp_path):
    mock_run.return_value = {
        "meta": {}, "processed_chunks": [], "aggregated_hpo_terms": [],
    }
    (tmp_path / "in.txt").write_text("Patient.")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["text", "process", str(tmp_path / "in.txt"),
         "--profile", "german_recall",
         "--adaptive-rechunking-quality-threshold", "0.5"],
    )
    assert result.exit_code == 0
    cfg = mock_run.call_args.kwargs["adaptive_rechunking"]
    assert cfg.quality_threshold == 0.5  # CLI wins
    assert cfg.enabled is True            # from profile
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/integration/test_specA_specB_interaction.py -v`
Expected: PASS, two tests.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_specA_specB_interaction.py
git commit -m "test(integration): Spec A profile + Spec B adaptive_rechunking interaction

Verifies a profile's adaptive_rechunking block is propagated correctly
and that explicit --adaptive-rechunking-* flags override profile values."
```

---

## Phase 13: CHANGELOG and final integration

### Task 19: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add entry**

Prepend to `CHANGELOG.md` (under appropriate version heading):

```markdown
### Added

- **Adaptive re-chunking** (issue #148): an opt-in mechanism that detects
  per-chunk retrieval quality and, when poor, subdivides the chunk into
  sentence-bounded sub-chunks, re-queries them, and merges results. Enable
  via `phentrieve text process FILE --adaptive-rechunking` or in
  `phentrieve.yaml` under `extraction.adaptive_rechunking:`. See
  [docs/user-guide/adaptive-rechunking.md](docs/user-guide/adaptive-rechunking.md).
- **`OrchestrationResult` dataclass**: `orchestrate_hpo_extraction` now returns
  this dataclass instead of a plain tuple. Backward-compatible — legacy
  2-tuple unpacking via `__iter__` still works for all existing call sites.
  New attribute `raw_query_results` exposes the unfiltered top-K from
  `query_batch` for callers that need scores below `chunk_retrieval_threshold`.
- **`adapt_standard_response.extra_meta` parameter**: optional dict merged
  into the response's `meta` block. Used by adaptive rechunking to surface
  `meta.adaptive_rechunking` summaries in the API response.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): adaptive re-chunking + OrchestrationResult"
```

### Task 20: Final cleanup and PR readiness

- [ ] **Step 1: Run all required checks**

Run: `make check && make typecheck-fast && make test`
Expected: PASS.

Run: `make frontend-test-ci && make frontend-build-ci`
Expected: PASS.

Run: `make ci-local && make precommit`
Expected: PASS.

- [ ] **Step 2: Run lint-fix**

Run: `make lint-fix`

- [ ] **Step 3: Final commit (if lint-fix changed things)**

```bash
git add -u
git diff --cached --quiet || git commit -m "chore: ruff format pass"
```

- [ ] **Step 4: Verify no spec drift**

Open `.planning/specs/2026-04-25-adaptive-rechunking-spec.md` and `.planning/specs/2026-04-25-cli-profiles-default-resolution-spec.md`. For each section, verify the implementation matches.

---

## Self-Review Checklist

After completing all tasks:

- [ ] Architecture: `OrchestrationResult` returned (Task 1, 2). `precomputed_query_results` parameter (Task 2). `adapt_standard_response.extra_meta` (Task 3). Seam in `run_standard_backend` (Task 10). All four match Spec B's Architecture section.
- [ ] Configuration: `AdaptiveRechunkingConfig` (Task 4). `AdaptiveRechunkingProfileBlock` (Task 4). YAML schema documented (Task 14).
- [ ] Quality assessment: `assess_chunk_quality` (Task 5). Raw-score-access invariant test (Task 6).
- [ ] Sub-chunking: `subdivide_parent_chunk` (Task 7).
- [ ] Score-improvement gate: `apply_score_improvement_gate` (Task 8).
- [ ] Top-level orchestration: `run_adaptive_rechunking` + `AdaptiveRechunkingResult` (Task 9). Cost invariant: 1 call per recursion level (Task 9 + Task 16).
- [ ] CLI flags: four flags on `text process` (Task 11).
- [ ] API parity: request schema + router pass-through + response meta (Task 12).
- [ ] Frontend: payload pass-through + graceful response (Task 13).
- [ ] Documentation: user-guide page + cross-references (Task 14).
- [ ] Tests: every helper has a unit test; e2e + benchmark + perf + call-count + cross-spec integration tests (Tasks 15-18).
- [ ] CHANGELOG (Task 19).
- [ ] No "TBD"/"TODO" placeholders.
- [ ] Type signatures consistent: `OrchestrationResult`, `AdaptiveRechunkingConfig`, `AdaptiveRechunkingResult`, `ChunkQualitySignals`, `AdaptiveRechunkingProfileBlock`, `assess_chunk_quality`, `subdivide_parent_chunk`, `apply_score_improvement_gate`, `run_adaptive_rechunking`, `dump_quality_report`.

If issues: fix inline.
