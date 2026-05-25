# LLM Evidence Validation And Enriched Mapping Implementation Plan

> Status: Archived on 2026-05-25. PR #261 was closed as superseded after a
> same-model, same-document focused A/B against current `main` showed a strict-ID
> mapping regression. See
> `.planning/analysis/2026-05-25-llm-evidence-validation-enriched-mapping-pr-regression.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate phase-1 LLM evidence before phase-2 mapping, keep phase 1 source-faithful, and enrich phase-2 mapping payloads with compact candidate context.

**Architecture:** First extract phase-2A retrieval orchestration from the oversized `pipeline.py` into `phentrieve/llm/phase2a.py` without behavior change. Then add a pure evidence validator before actionable filtering, move phrase rewrite behavior into `prepare_retrieval_queries(...)`, enrich compact mapping payloads through the existing details helper, and persist validation/token observability through existing trace and benchmark metadata fields.

**Tech Stack:** Python 3.11 typing, Pydantic LLM result models, existing LLM prompt YAML loader, existing HPO details enrichment helper, pytest, Ruff, mypy.

---

## Sources Read

- Spec: `.planning/specs/2026-05-25-llm-evidence-validation-enriched-mapping-design.md`
- Analysis: `.planning/analysis/2026-05-23-phentrieve-rag-prompting-literature-report.md`
- Core files:
  - `phentrieve/llm/pipeline.py`
  - `phentrieve/llm/pipeline_phase1.py`
  - `phentrieve/llm/pipeline_phase2.py`
  - `phentrieve/llm/pipeline_trace.py`
  - `phentrieve/llm/types.py`
  - `phentrieve/retrieval/details_enrichment.py`
  - `phentrieve/benchmark/llm_benchmark.py`
- Prompt files:
  - `phentrieve/llm/prompts/templates/two_phase/en.yaml`
  - `phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml`
  - `phentrieve/llm/prompts/templates/two_phase/en_mapping_batch.yaml`
- Existing tests:
  - `tests/unit/llm/test_pipeline.py`
  - `tests/unit/llm/test_two_phase.py`
  - `tests/unit/llm/test_prompts.py`
  - `tests/unit/llm/test_pipeline_characterization.py`
  - `tests/unit/test_llm_benchmark.py`
  - `tests/integration/llm/test_grounded_pipeline_integration.py`

## File Structure

- Create: `phentrieve/llm/phase2a.py`
  - Own phase-2A retrieval orchestration currently embedded in `TwoPhaseLLMPipeline._retrieve_candidates(...)`.
- Create: `phentrieve/llm/evidence_validation.py`
  - Pure evidence validation, repair, downgrade, and trace-summary helpers.
- Modify: `phentrieve/llm/pipeline.py`
  - Delegate phase-2A retrieval to `phase2a.retrieve_candidates(...)`.
  - Run evidence validation after phase-1 expansion/deduplication and before actionable filtering.
  - Add validation trace and phase counts.
  - Add `phase2_mapping_prompt_tokens_per_request`.
- Modify: `phentrieve/llm/pipeline_phase1.py`
  - Stop phase-1 post-processing from expanding abbreviations or source-unfaithful shared-head rewrites.
- Modify: `phentrieve/llm/pipeline_phase2.py`
  - Add conservative query variants.
  - Add candidate definition and matched-synonym payload enrichment.
- Modify: `phentrieve/llm/prompts/templates/two_phase/en.yaml`
  - Make phase-1 prompt and examples source-faithful.
- Modify: `phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml`
  - Teach enriched payload fields and specificity.
- Modify: `phentrieve/llm/prompts/templates/two_phase/en_mapping_batch.yaml`
  - Same mapping prompt updates for batch mode.
- Modify: `tests/unit/llm/test_phase2a.py`
  - New focused phase-2A extraction coverage.
- Modify: `tests/unit/llm/test_evidence_validation.py`
  - New pure validator coverage.
- Modify: `tests/unit/llm/test_pipeline.py`
  - Pipeline integration coverage for validation and mapping-token metrics.
- Modify: `tests/unit/llm/test_two_phase.py`
  - Source-faithful expansion and query variant coverage.
- Modify: `tests/unit/llm/test_prompts.py`
  - Prompt version and example coverage.
- Modify: `tests/unit/test_llm_benchmark.py`
  - Benchmark trace/observability coverage if the pipeline-level test does not already prove the field flows through.

## Task 1: Extract Phase-2A Retrieval Orchestration

**Files:**
- Create: `phentrieve/llm/phase2a.py`
- Create: `tests/unit/llm/test_phase2a.py`
- Modify: `phentrieve/llm/pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/llm/test_phase2a.py`:

```python
from __future__ import annotations

import pytest

from phentrieve.llm.phase2a import retrieve_candidates

pytestmark = pytest.mark.unit


class FakeToolExecutor:
    def __init__(self, batch_results: list[dict[str, object]]) -> None:
        self.batch_results = list(batch_results)
        self.queries: list[dict[str, object]] = []

    def query_batch_hpo_terms(self, *, phrases, language, n_results):
        self.queries.append(
            {
                "phrases": list(phrases),
                "language": language,
                "n_results": n_results,
            }
        )
        return list(self.batch_results)


def test_retrieve_candidates_merges_query_variants_and_preserves_grounding() -> None:
    tool_executor = FakeToolExecutor(
        [
            {
                "phrase": "serum creatinine 11.2 mg/dL",
                "candidates": [
                    {
                        "hpo_id": "HP:0003259",
                        "term_name": "Elevated serum creatinine",
                        "score": 0.70,
                    }
                ],
            },
            {
                "phrase": "serum creatinine 11.2",
                "candidates": [
                    {
                        "hpo_id": "HP:0003259",
                        "term_name": "Elevated serum creatinine",
                        "score": 0.91,
                    }
                ],
            },
        ]
    )

    results = retrieve_candidates(
        actionable=[
            {
                "phrase": "serum creatinine 11.2 mg/dL",
                "category": "Abnormal",
                "chunk_ids": [2],
                "evidence_text": "serum creatinine 11.2 mg/dL",
                "start_char": 9,
                "end_char": 36,
            }
        ],
        grounded_chunks=[
            {"chunk_id": 1, "text": "Prior history."},
            {"chunk_id": 2, "text": "Labs show serum creatinine 11.2 mg/dL."},
            {"chunk_id": 3, "text": "Follow-up was arranged."},
        ],
        language="en",
        tool_executor=tool_executor,
        n_results_per_phrase=50,
        max_unique_candidates=10,
        min_unique_candidates=3,
        similarity_threshold=0.60,
    )

    assert tool_executor.queries == [
        {
            "phrases": ["serum creatinine 11.2 mg/dL", "serum creatinine 11.2"],
            "language": "en",
            "n_results": 50,
        }
    ]
    assert results == [
        {
            "phrase": "serum creatinine 11.2 mg/dL",
            "category": "Abnormal",
            "candidates": [
                {
                    "hpo_id": "HP:0003259",
                    "term_name": "Elevated serum creatinine",
                    "score": 0.91,
                    "retrieval_query": "serum creatinine 11.2",
                }
            ],
            "chunk_ids": [2],
            "evidence_text": "serum creatinine 11.2 mg/dL",
            "start_char": 9,
            "end_char": 36,
            "grounded_context": {
                "chunk_ids": [2],
                "primary_chunk_text": "Labs show serum creatinine 11.2 mg/dL.",
                "neighbor_chunk_texts": ["Prior history.", "Follow-up was arranged."],
            },
        }
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/unit/llm/test_phase2a.py -n 0 -v
```

Expected: fail with `ModuleNotFoundError: No module named 'phentrieve.llm.phase2a'`.

- [ ] **Step 3: Write the minimal implementation**

Create `phentrieve/llm/phase2a.py`:

```python
from __future__ import annotations

import logging
from typing import Any

from phentrieve.llm.pipeline_phase2 import (
    build_grounded_context,
    downstream_dedupe_key,
    extract_first_result_list,
    hybrid_select_candidates,
    prepare_retrieval_queries,
)

logger = logging.getLogger(__name__)


def retrieve_candidates(
    *,
    actionable: list[dict[str, Any]],
    grounded_chunks: list[dict[str, Any]],
    language: str,
    tool_executor: Any,
    n_results_per_phrase: int,
    max_unique_candidates: int,
    min_unique_candidates: int,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    unique_actionable: list[dict[str, Any]] = []
    actionable_groups: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for item in actionable:
        dedupe_key = downstream_dedupe_key(item, include_candidate_ids=False)
        actionable_groups.setdefault(dedupe_key, []).append(item)
        if len(actionable_groups[dedupe_key]) == 1:
            unique_actionable.append(item)

    expanded_queries: list[str] = []
    expanded_query_keys: list[tuple[str, str, tuple[str, ...]]] = []
    for item in unique_actionable:
        dedupe_key = downstream_dedupe_key(item, include_candidate_ids=False)
        query_variants = prepare_retrieval_queries(str(item["phrase"]))
        if not query_variants:
            query_variants = [str(item["phrase"])]
        for query in query_variants:
            expanded_queries.append(query)
            expanded_query_keys.append(dedupe_key)

    batched_variant_results = tool_executor.query_batch_hpo_terms(
        phrases=expanded_queries,
        language=language,
        n_results=n_results_per_phrase,
    )

    shared_results: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    grouped_variant_results: dict[
        tuple[str, str, tuple[str, ...]], list[tuple[str, dict[str, Any]]]
    ] = {}
    for index, dedupe_key in enumerate(expanded_query_keys):
        query = expanded_queries[index]
        batch_result = (
            batched_variant_results[index]
            if index < len(batched_variant_results)
            else {}
        )
        grouped_variant_results.setdefault(dedupe_key, []).append(
            (query, dict(batch_result) if isinstance(batch_result, dict) else {})
        )

    for item in unique_actionable:
        phrase = str(item["phrase"])
        category = str(item["category"])
        dedupe_key = downstream_dedupe_key(item, include_candidate_ids=False)
        merged_candidates: dict[str, dict[str, Any]] = {}
        for query, batch_result in grouped_variant_results.get(dedupe_key, []):
            if "candidates" in batch_result:
                candidates = [
                    {**dict(candidate), "retrieval_query": query}
                    for candidate in batch_result.get("candidates", [])
                    if isinstance(candidate, dict)
                ]
            else:
                metadatas = extract_first_result_list(batch_result, "metadatas")
                similarities = extract_first_result_list(batch_result, "similarities")
                candidates = [
                    {**candidate, "retrieval_query": query}
                    for candidate in hybrid_select_candidates(
                        phrase=query,
                        metadatas=metadatas,
                        similarities=similarities,
                        max_unique_candidates=max_unique_candidates,
                        min_unique_candidates=min_unique_candidates,
                        similarity_threshold=similarity_threshold,
                    )
                ]

            for candidate in candidates:
                hpo_id = str(candidate.get("hpo_id", "")).strip()
                if not hpo_id:
                    continue
                existing = merged_candidates.get(hpo_id)
                if existing is None or float(candidate.get("score", 0.0) or 0.0) > float(
                    existing.get("score", 0.0) or 0.0
                ):
                    merged_candidates[hpo_id] = candidate

        merged = sorted(
            merged_candidates.values(),
            key=lambda candidate: float(candidate.get("score", 0.0) or 0.0),
            reverse=True,
        )[:n_results_per_phrase]
        shared_results[dedupe_key] = {
            "phrase": phrase,
            "category": category,
            "candidates": merged,
        }
        logger.debug(
            "Phase 2A candidate retrieval: phrase=%r candidates=%d",
            phrase,
            len(merged),
        )

    results: list[dict[str, Any]] = []
    for item in actionable:
        dedupe_key = downstream_dedupe_key(item, include_candidate_ids=False)
        shared_result = dict(shared_results.get(dedupe_key, {}))
        shared_result.setdefault("phrase", str(item["phrase"]))
        shared_result.setdefault("category", str(item["category"]))
        shared_result["chunk_ids"] = list(item.get("chunk_ids", []))
        shared_result["evidence_text"] = item.get("evidence_text")
        shared_result["start_char"] = item.get("start_char")
        shared_result["end_char"] = item.get("end_char")
        shared_result["grounded_context"] = build_grounded_context(
            item=item,
            grounded_chunks=grounded_chunks,
        )
        shared_result["candidates"] = list(shared_result.get("candidates", []))
        results.append(shared_result)
    return results
```

The imported `hybrid_select_candidates(...)` name is an existing dense-only
helper in this codebase; this task must not add hybrid lexical/dense retrieval
or any new retrieval channel.

Modify `phentrieve/llm/pipeline.py`:

```python
from phentrieve.llm.phase2a import retrieve_candidates as _retrieve_phase2a_candidates
```

Replace the body of `TwoPhaseLLMPipeline._retrieve_candidates(...)` with:

```python
        return _retrieve_phase2a_candidates(
            actionable=actionable,
            grounded_chunks=grounded_chunks,
            language=language,
            tool_executor=self.tool_executor,
            n_results_per_phrase=self.n_results_per_phrase,
            max_unique_candidates=self.max_unique_candidates,
            min_unique_candidates=self.min_unique_candidates,
            similarity_threshold=self.similarity_threshold,
        )
```

Leave `_extract_first_result_list(...)`, `_build_grounded_context(...)`, and
`_hybrid_select_candidates(...)` wrappers in place until the full test suite
confirms no callers depend on removal.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/unit/llm/test_phase2a.py tests/unit/llm/test_pipeline.py::test_two_phase_pipeline_maps_phrase_via_retrieved_candidates -n 0 -v
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/llm/phase2a.py phentrieve/llm/pipeline.py tests/unit/llm/test_phase2a.py
git commit -m "refactor: extract llm phase2a retrieval"
```

## Task 2: Add Pure Phase-1 Evidence Validator

**Files:**
- Create: `phentrieve/llm/evidence_validation.py`
- Create: `tests/unit/llm/test_evidence_validation.py`

- [ ] **Step 1: Write the failing validator tests**

Create `tests/unit/llm/test_evidence_validation.py`:

```python
from __future__ import annotations

import pytest

from phentrieve.llm.evidence_validation import validate_phase1_evidence

pytestmark = pytest.mark.unit


def test_validate_phase1_evidence_drops_unknown_chunk_id() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [99],
                "evidence_text": "recurrent seizures",
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept == []
    assert report.dropped == [
        {
            "phrase": "recurrent seizures",
            "reason": "unknown_chunk_id",
            "chunk_ids": [99],
        }
    ]


def test_validate_phase1_evidence_drops_empty_chunk_ids() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [],
                "evidence_text": "recurrent seizures",
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept == []
    assert report.dropped == [
        {
            "phrase": "recurrent seizures",
            "reason": "empty_chunk_ids",
            "chunk_ids": [],
        }
    ]


def test_validate_phase1_evidence_repairs_missing_evidence_from_phrase() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": None,
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept[0]["evidence_text"] == "recurrent seizures"
    assert report.repairs == [
        {"phrase": "recurrent seizures", "kind": "evidence_text_repair"}
    ]


def test_validate_phase1_evidence_repairs_offsets_from_exact_evidence() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "recurrent seizures",
                "start_char": 999,
                "end_char": 1009,
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept[0]["start_char"] == 12
    assert report.kept[0]["end_char"] == 30
    assert {"phrase": "recurrent seizures", "kind": "offset_repair"} in report.repairs


def test_validate_phase1_evidence_accepts_document_absolute_offsets_as_local() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "recurrent seizures",
                "start_char": 112,
                "end_char": 130,
            }
        ],
        grounded_chunks=[
            {
                "chunk_id": 1,
                "text": "Patient had recurrent seizures.",
                "start_char": 100,
                "end_char": 132,
            }
        ],
    )

    assert report.kept[0]["start_char"] == 12
    assert report.kept[0]["end_char"] == 30
    assert {"phrase": "recurrent seizures", "kind": "offset_coordinate_repair"} in report.repairs


def test_validate_phase1_evidence_downgrades_multichunk_offsets() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizures",
                "category": "Abnormal",
                "chunk_ids": [1, 2],
                "evidence_text": "Patient had recurrent seizures",
                "start_char": 0,
                "end_char": 30,
            }
        ],
        grounded_chunks=[
            {"chunk_id": 1, "text": "Patient had recurrent"},
            {"chunk_id": 2, "text": "seizures."},
        ],
    )

    assert report.kept[0]["start_char"] is None
    assert report.kept[0]["end_char"] is None
    assert report.repairs == [
        {"phrase": "recurrent seizures", "kind": "multi_chunk_offset_downgrade"}
    ]


def test_validate_phase1_evidence_downgrades_fuzzy_evidence_to_chunk_level() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "recurrent seizure",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "recurrent seizure",
                "start_char": 12,
                "end_char": 29,
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
        fuzzy_threshold=80.0,
    )

    assert report.kept[0]["start_char"] is None
    assert report.kept[0]["end_char"] is None
    assert report.kept[0]["evidence_text"] == "recurrent seizure"
    assert report.repairs == [
        {"phrase": "recurrent seizure", "kind": "fuzzy_evidence_downgrade"}
    ]


def test_validate_phase1_evidence_drops_ungrounded_evidence() -> None:
    report = validate_phase1_evidence(
        extracted=[
            {
                "phrase": "invented ataxia",
                "category": "Abnormal",
                "chunk_ids": [1],
                "evidence_text": "invented ataxia",
            }
        ],
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
    )

    assert report.kept == []
    assert report.dropped == [
        {
            "phrase": "invented ataxia",
            "reason": "evidence_not_grounded",
            "chunk_ids": [1],
        }
    ]


def test_validate_phase1_evidence_skips_ungrounded_legacy_inputs() -> None:
    original = [
        {
            "phrase": "recurrent seizures",
            "category": "Abnormal",
            "chunk_ids": [],
            "evidence_text": None,
        }
    ]

    report = validate_phase1_evidence(extracted=original, grounded_chunks=[])

    assert report.status == "skipped_no_grounded_chunks"
    assert report.kept == original
    assert report.dropped == []
    assert report.repairs == []
    assert report.kept is not original
```

- [ ] **Step 2: Run the validator tests and verify failure**

Run:

```bash
uv run pytest tests/unit/llm/test_evidence_validation.py -n 0 -v
```

Expected: fail with `ModuleNotFoundError: No module named 'phentrieve.llm.evidence_validation'`.

- [ ] **Step 3: Write the minimal validator implementation**

Create `phentrieve/llm/evidence_validation.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

FUZZY_MATCH_RATIO_THRESHOLD = 90.0


@dataclass
class EvidenceValidationReport:
    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    repairs: list[dict[str, Any]]
    status: str = "validated"


def validate_phase1_evidence(
    *,
    extracted: list[dict[str, Any]],
    grounded_chunks: list[dict[str, Any]],
    fuzzy_threshold: float = FUZZY_MATCH_RATIO_THRESHOLD,
) -> EvidenceValidationReport:
    if not grounded_chunks:
        return EvidenceValidationReport(
            kept=[dict(item) for item in extracted],
            dropped=[],
            repairs=[],
            status="skipped_no_grounded_chunks",
        )

    chunks_by_id = {int(chunk["chunk_id"]): chunk for chunk in grounded_chunks}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    for raw_item in extracted:
        item = dict(raw_item)
        phrase = str(item.get("phrase", "") or "").strip()
        chunk_ids = _coerce_chunk_ids(item.get("chunk_ids"))
        item["chunk_ids"] = chunk_ids

        if not chunk_ids:
            dropped.append(
                {"phrase": phrase, "reason": "empty_chunk_ids", "chunk_ids": chunk_ids}
            )
            continue
        if any(chunk_id not in chunks_by_id for chunk_id in chunk_ids):
            dropped.append(
                {"phrase": phrase, "reason": "unknown_chunk_id", "chunk_ids": chunk_ids}
            )
            continue

        referenced_chunks = [chunks_by_id[chunk_id] for chunk_id in chunk_ids]
        haystack = " ".join(
            str(chunk.get("text", "") or "") for chunk in referenced_chunks
        )
        evidence = str(item.get("evidence_text") or "").strip()
        if not evidence and phrase and _find_case_insensitive(phrase, haystack):
            item["evidence_text"] = phrase
            evidence = phrase
            repairs.append({"phrase": phrase, "kind": "evidence_text_repair"})

        if not evidence:
            dropped.append({"phrase": phrase, "reason": "empty_evidence", "chunk_ids": chunk_ids})
            continue

        exact_span = _find_case_insensitive(evidence, haystack)
        if exact_span is not None:
            repaired = _repair_offsets(
                item=item,
                evidence=evidence,
                haystack=haystack,
                exact_span=exact_span,
                referenced_chunks=referenced_chunks,
            )
            if repaired["kind"]:
                repairs.append({"phrase": phrase, "kind": repaired["kind"]})
            kept.append(repaired["item"])
            continue

        ratio = _best_window_ratio(evidence, haystack)
        if ratio >= fuzzy_threshold:
            downgraded = {**item, "start_char": None, "end_char": None}
            repairs.append({"phrase": phrase, "kind": "fuzzy_evidence_downgrade"})
            kept.append(downgraded)
            continue

        dropped.append(
            {"phrase": phrase, "reason": "evidence_not_grounded", "chunk_ids": chunk_ids}
        )

    return EvidenceValidationReport(kept=kept, dropped=dropped, repairs=repairs)


def validation_report_summary(report: EvidenceValidationReport) -> dict[str, Any]:
    downgraded_count = sum(
        1
        for repair in report.repairs
        if str(repair.get("kind", "")).endswith("_downgrade")
    )
    return {
        "status": report.status,
        "kept_count": len(report.kept),
        "dropped_count": len(report.dropped),
        "repair_count": len(report.repairs),
        "downgraded_count": downgraded_count,
        "dropped": list(report.dropped),
        "repairs": list(report.repairs),
    }


def _coerce_chunk_ids(value: Any) -> list[int]:
    ids: list[int] = []
    for chunk_id in value or []:
        try:
            ids.append(int(chunk_id))
        except (TypeError, ValueError):
            continue
    return ids


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _find_case_insensitive(needle: str, haystack: str) -> tuple[int, int] | None:
    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)", re.IGNORECASE)
    match = pattern.search(haystack)
    if match is None:
        return None
    return match.start(), match.end()


def _best_window_ratio(needle: str, haystack: str) -> float:
    needle_tokens = _normalize_text(needle).split()
    haystack_tokens = _normalize_text(haystack).split()
    if not needle_tokens or not haystack_tokens:
        return 0.0
    window_size = max(1, len(needle_tokens))
    best = 0.0
    for size in {window_size - 1, window_size, window_size + 1}:
        if size < 1:
            continue
        for start in range(0, max(len(haystack_tokens) - size + 1, 1)):
            window = " ".join(haystack_tokens[start : start + size])
            best = max(best, SequenceMatcher(None, _normalize_text(needle), window).ratio() * 100.0)
    return best


def _repair_offsets(
    *,
    item: dict[str, Any],
    evidence: str,
    haystack: str,
    exact_span: tuple[int, int] | None,
    referenced_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(referenced_chunks) != 1:
        return {
            "item": {**item, "start_char": None, "end_char": None},
            "kind": "multi_chunk_offset_downgrade",
        }

    chunk = referenced_chunks[0]
    chunk_text = str(chunk.get("text", "") or "")
    start = item.get("start_char")
    end = item.get("end_char")

    if isinstance(start, int) and isinstance(end, int):
        if _span_matches_evidence(chunk_text, start, end, evidence):
            return {"item": item, "kind": ""}
        chunk_start = chunk.get("start_char")
        if isinstance(chunk_start, int):
            local_start = start - chunk_start
            local_end = end - chunk_start
            if _span_matches_evidence(chunk_text, local_start, local_end, evidence):
                return {
                    "item": {**item, "start_char": local_start, "end_char": local_end},
                    "kind": "offset_coordinate_repair",
                }

    if exact_span is not None:
        repaired = {**item, "start_char": exact_span[0], "end_char": exact_span[1]}
        return {"item": repaired, "kind": "offset_repair"}
    downgraded = {**item, "start_char": None, "end_char": None}
    return {"item": downgraded, "kind": "offset_downgrade"}


def _span_matches_evidence(text: str, start: int, end: int, evidence: str) -> bool:
    if not (0 <= start < end <= len(text)):
        return False
    if text[start:end].lower() != evidence.lower():
        return False
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return (not before.isalnum() and before != "_") and (
        not after.isalnum() and after != "_"
    )
```

After the tests pass, run Ruff formatting if line length fails:

```bash
uv run ruff format phentrieve/llm/evidence_validation.py tests/unit/llm/test_evidence_validation.py
```

- [ ] **Step 4: Run the validator tests and verify pass**

Run:

```bash
uv run pytest tests/unit/llm/test_evidence_validation.py -n 0 -v
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/llm/evidence_validation.py tests/unit/llm/test_evidence_validation.py
git commit -m "feat: validate llm phase1 evidence"
```

## Task 3: Wire Evidence Validation Into The Two-Phase Pipeline

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write the failing pipeline test**

Add this test to `tests/unit/llm/test_pipeline.py` near the existing phase-1 trace tests:

```python
def test_two_phase_pipeline_drops_ungrounded_phase1_records_before_retrieval() -> None:
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "recurrent seizures",
                            "Abnormal",
                            chunk_ids=[1],
                            evidence_text="recurrent seizures",
                        ),
                        grounded_phenotype(
                            "invented ataxia",
                            "Abnormal",
                            chunk_ids=[99],
                            evidence_text="invented ataxia",
                        ),
                    ]
                }
            }
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "recurrent seizures",
                "candidates": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Recurrent seizures",
                        "score": 0.95,
                    }
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=tool_executor)

    result = pipeline.run(
        text="Patient had recurrent seizures.",
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert tool_executor.queries == [
        {
            "phrases": ["recurrent seizures"],
            "language": "en",
            "n_results": 50,
        }
    ]
    assert result.meta.phase_counts["extracted_phrases"] == 2
    assert result.meta.phase_counts["phase1_validated_phrases"] == 1
    assert result.meta.phase_counts["phase1_evidence_kept"] == 1
    assert result.meta.phase_counts["phase1_evidence_dropped"] == 1
    assert result.meta.trace["phase1_evidence_validation"] == {
        "status": "validated",
        "kept_count": 1,
        "dropped_count": 1,
        "repair_count": 0,
        "downgraded_count": 0,
        "dropped": [
            {
                "phrase": "invented ataxia",
                "reason": "unknown_chunk_id",
                "chunk_ids": [99],
            }
        ],
        "repairs": [],
    }
    assert [term.term_id for term in result.terms] == ["HP:0001250"]
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
uv run pytest tests/unit/llm/test_pipeline.py::test_two_phase_pipeline_drops_ungrounded_phase1_records_before_retrieval -n 0 -v
```

Expected: fail because the invalid phase-1 record is still sent to retrieval or because `phase1_evidence_validation` is missing.

- [ ] **Step 3: Wire validation into `pipeline.py`**

Add imports:

```python
from phentrieve.llm.evidence_validation import (
    validate_phase1_evidence,
    validation_report_summary,
)
```

After phase-1 expansion and deduplication, replace:

```python
        extracted = self._deduplicate_phase1_extractions(
            _expand_combined_phase1_extractions(extracted)
        )
        actionable = [
```

with:

```python
        extracted = self._deduplicate_phase1_extractions(
            _expand_combined_phase1_extractions(extracted)
        )
        phase1_extracted_count = len(extracted)
        evidence_validation_report = validate_phase1_evidence(
            extracted=extracted,
            grounded_chunks=grounded_chunks,
        )
        extracted = evidence_validation_report.kept
        evidence_validation_trace = validation_report_summary(evidence_validation_report)
        actionable = [
```

Replace the existing `"extracted_phrases": len(extracted)` entry and add the
new validation counts where `phase_counts` is initialized:

```python
            "extracted_phrases": phase1_extracted_count,
            "phase1_validated_phrases": len(extracted),
            "phase1_evidence_kept": len(evidence_validation_report.kept),
            "phase1_evidence_dropped": len(evidence_validation_report.dropped),
            "phase1_evidence_repaired": len(evidence_validation_report.repairs),
            "phase1_evidence_downgraded": int(
                evidence_validation_trace.get("downgraded_count", 0) or 0
            ),
```

Add trace field when `trace` is initialized:

```python
            "phase1_evidence_validation": evidence_validation_trace,
```

Keep `build_phase1_trace(...)` fed with the validated `extracted` list so the
phase-1 extracted trace matches what reaches phase 2.
Do not change the existing `extracted_phrases` metric to post-validation
semantics; it remains the post-expansion, post-dedup, pre-validation count for
benchmark comparability. Use `phase1_validated_phrases` for the post-validation
count.

- [ ] **Step 4: Run focused pipeline tests and verify pass**

Run:

```bash
uv run pytest tests/unit/llm/test_pipeline.py::test_two_phase_pipeline_drops_ungrounded_phase1_records_before_retrieval tests/unit/llm/test_pipeline.py::test_two_phase_pipeline_maps_phrase_via_retrieved_candidates -n 0 -v
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/llm/pipeline.py tests/unit/llm/test_pipeline.py
git commit -m "feat: gate llm mapping on validated evidence"
```

## Task 4: Make Phase 1 Source-Faithful

**Files:**
- Modify: `phentrieve/llm/prompts/templates/two_phase/en.yaml`
- Modify: `phentrieve/llm/pipeline_phase1.py`
- Modify: `tests/unit/llm/test_prompts.py`
- Modify: `tests/unit/llm/test_two_phase.py`

- [ ] **Step 1: Write failing source-faithful tests**

Add imports to `tests/unit/llm/test_prompts.py`:

```python
import json
```

Add this test:

```python
def test_phase1_few_shot_examples_are_source_faithful() -> None:
    template = loader.get_prompt(AnnotationMode.TWO_PHASE, "en")

    for example in template.few_shot_examples:
        source_text = example["input"].split("---", 2)[1]
        output = json.loads(example["output"])
        for item in output["phenotypes"]:
            assert item["phrase"] in item["evidence_text"]
            assert item["evidence_text"] in source_text
```

Replace `test_phase1_prompt_mentions_normalized_clinical_phrase_for_lab_or_event_evidence`
in `tests/unit/llm/test_two_phase.py` with:

```python
def test_phase1_prompt_says_phase2_handles_retrieval_variants():
    prompt = get_prompt(AnnotationMode.TWO_PHASE, "en")
    system = prompt.render_system_prompt()

    assert "Phase 2 will compute retrieval variants" in system
    assert "faithful extraction" in system
    assert "short normalized clinical phrase" not in system
```

Replace `test_expand_combined_phase1_extractions_expands_common_phenotype_abbreviations`
with:

```python
def test_expand_combined_phase1_extractions_keeps_abbreviations_source_faithful():
    expanded = expand_combined_phase1_extractions(
        [
            {
                "phrase": "XLID",
                "category": "abnormal",
                "chunk_ids": [29],
                "evidence_text": "XLID",
            }
        ]
    )

    assert expanded == [
        {
            "phrase": "XLID",
            "category": "abnormal",
            "chunk_ids": [29],
            "evidence_text": "XLID",
        }
    ]
```

Replace `test_expand_combined_phase1_extractions_splits_slash_and_shared_head_mentions`
with:

```python
def test_expand_combined_phase1_extractions_splits_only_source_substring_mentions():
    expanded = expand_combined_phase1_extractions(
        [
            {
                "phrase": "hypertonia/spasticity of the extremities",
                "category": "abnormal",
                "chunk_ids": [15],
                "evidence_text": "hypertonia/spasticity of the extremities",
            },
            {
                "phrase": "pontine cerebellar hypoplasia",
                "category": "abnormal",
                "chunk_ids": [44, 45],
                "evidence_text": "pontine cerebellar hypoplasia",
            },
        ]
    )

    phrases = [item["phrase"] for item in expanded]

    assert phrases == [
        "hypertonia",
        "spasticity of the extremities",
        "pontine cerebellar hypoplasia",
    ]
    assert all(item["category"] == "abnormal" for item in expanded)
    assert expanded[0]["chunk_ids"] == [15]
    assert expanded[2]["chunk_ids"] == [44, 45]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/unit/llm/test_prompts.py::test_phase1_few_shot_examples_are_source_faithful tests/unit/llm/test_two_phase.py::test_phase1_prompt_says_phase2_handles_retrieval_variants tests/unit/llm/test_two_phase.py::test_expand_combined_phase1_extractions_keeps_abbreviations_source_faithful tests/unit/llm/test_two_phase.py::test_expand_combined_phase1_extractions_splits_only_source_substring_mentions -n 0 -v
```

Expected: fail because the current prompt example rewrites lactate dehydrogenase, the prompt still mentions normalized clinical phrases, and phase-1 post-processing still expands `XLID` and shared-head phrases.

- [ ] **Step 3: Update prompt and phase-1 expansion**

In `phentrieve/llm/prompts/templates/two_phase/en.yaml`:

- change `version: "v3.0.0"` to `version: "v3.1.0"`;
- replace the extraction rule `If the note clearly describes an abnormality indirectly...` with:

```yaml
  - Every phrase must be a verbatim substring of the evidence_text
  - Every evidence_text must be a verbatim substring of the referenced chunk
  - Phase 2 will compute retrieval variants such as canonicalized noun phrases and abbreviation expansion; your job is faithful extraction, not rewriting
```

- update the blood-test example using the existing block-scalar string shape:

```yaml
    output: |
      {
        "phenotypes": [
          {"phrase": "lactate dehydrogenase was markedly elevated", "category": "Abnormal", "chunk_ids": [1], "evidence_text": "lactate dehydrogenase was markedly elevated"},
          {"phrase": "urine output remained low", "category": "Abnormal", "chunk_ids": [1], "evidence_text": "urine output remained low"}
        ]
      }
```

In `phentrieve/llm/pipeline_phase1.py`, remove abbreviation and shared-head rewriting from `split_combined_phase1_phrase(...)`:

```python
def split_combined_phase1_phrase(phrase: str) -> list[str]:
    """Split clear combined phenotype mentions into standalone source phrases."""
    return _split_slash_combined_phrase(phrase)
```

And remove the abbreviation branch from `expand_combined_phase1_extractions(...)` so the loop starts:

```python
    for item in extracted:
        phrase = str(item.get("phrase", "")).strip()
        split_phrases = split_combined_phase1_phrase(phrase)
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/unit/llm/test_prompts.py::test_phase1_few_shot_examples_are_source_faithful tests/unit/llm/test_two_phase.py::test_phase1_prompt_says_phase2_handles_retrieval_variants tests/unit/llm/test_two_phase.py::test_expand_combined_phase1_extractions_keeps_abbreviations_source_faithful tests/unit/llm/test_two_phase.py::test_expand_combined_phase1_extractions_splits_only_source_substring_mentions -n 0 -v
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/llm/prompts/templates/two_phase/en.yaml phentrieve/llm/pipeline_phase1.py tests/unit/llm/test_prompts.py tests/unit/llm/test_two_phase.py
git commit -m "feat: keep llm phase1 source faithful"
```

## Task 5: Move Rewrite Behavior Into Retrieval-Query Preparation

**Files:**
- Modify: `phentrieve/llm/pipeline_phase2.py`
- Modify: `tests/unit/llm/test_two_phase.py`

- [ ] **Step 1: Write failing query-variant tests**

Add these tests to `tests/unit/llm/test_two_phase.py` near the existing
`prepare_retrieval_queries(...)` tests:

```python
def test_prepare_retrieval_queries_expands_known_abbreviations_after_original():
    queries = prepare_retrieval_queries("XLID")

    assert queries[0] == "XLID"
    assert "X-linked intellectual disability" in queries


def test_prepare_retrieval_queries_adds_conservative_lab_canonical_variant():
    queries = prepare_retrieval_queries("lactate dehydrogenase was markedly elevated")

    assert queries[0] == "lactate dehydrogenase was markedly elevated"
    assert "elevated lactate dehydrogenase" in queries


def test_prepare_retrieval_queries_adds_conservative_low_output_variant():
    queries = prepare_retrieval_queries("urine output remained low")

    assert queries[0] == "urine output remained low"
    assert "low urine output" in queries
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/unit/llm/test_two_phase.py::test_prepare_retrieval_queries_expands_known_abbreviations_after_original tests/unit/llm/test_two_phase.py::test_prepare_retrieval_queries_adds_conservative_lab_canonical_variant tests/unit/llm/test_two_phase.py::test_prepare_retrieval_queries_adds_conservative_low_output_variant tests/unit/llm/test_two_phase.py::test_prepare_retrieval_queries_does_not_invent_hand_written_paraphrases -n 0 -v
```

Expected: fail for the new abbreviation and canonical variant assertions.

- [ ] **Step 3: Implement conservative query variants**

In `phentrieve/llm/pipeline_phase2.py`, import the existing abbreviation map:

```python
from phentrieve.llm.pipeline_phase1 import PHENOTYPE_ABBREVIATIONS
```

Add helpers above `prepare_retrieval_queries(...)`:

```python
TRAILING_STATE_PATTERN = re.compile(
    r"^(?P<noun>[a-z0-9][a-z0-9\s\-/]+?)\s+"
    r"(?P<verb>was|were|is|are|remained)\s+"
    r"(?:(?P<intensity>markedly|severely|mildly)\s+)?"
    r"(?P<state>elevated|increased|reduced|low)$",
    re.IGNORECASE,
)
CANONICAL_STATES = {
    "elevated": "elevated",
    "increased": "increased",
    "reduced": "reduced",
    "low": "low",
}


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def conservative_canonical_phrase_variant(phrase: str) -> str | None:
    match = TRAILING_STATE_PATTERN.match(phrase.strip())
    if not match:
        return None
    noun = " ".join(match.group("noun").split())
    state = CANONICAL_STATES.get(match.group("state").lower())
    if not noun or state is None:
        return None
    return f"{state} {noun}"
```

Replace `prepare_retrieval_queries(...)` with:

```python
def prepare_retrieval_queries(phrase: str) -> list[str]:
    original = " ".join(str(phrase or "").split()).strip()
    if not original:
        return []

    variants = [original]
    stripped_units = " ".join(UNIT_TOKEN_PATTERN.sub(" ", original).split())
    if stripped_units and stripped_units != original:
        variants.append(stripped_units)

    canonical = conservative_canonical_phrase_variant(original)
    if canonical and canonical != original:
        variants.append(canonical)

    expanded_abbreviation = PHENOTYPE_ABBREVIATIONS.get(original.lower())
    if expanded_abbreviation:
        variants.append(expanded_abbreviation)

    return _dedupe_preserving_order(variants)
```

If `re` is not already imported in `pipeline_phase2.py`, add:

```python
import re
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/unit/llm/test_two_phase.py::test_prepare_retrieval_queries_expands_known_abbreviations_after_original tests/unit/llm/test_two_phase.py::test_prepare_retrieval_queries_adds_conservative_lab_canonical_variant tests/unit/llm/test_two_phase.py::test_prepare_retrieval_queries_adds_conservative_low_output_variant tests/unit/llm/test_two_phase.py::test_prepare_retrieval_queries_does_not_invent_hand_written_paraphrases tests/unit/llm/test_phase2a.py -n 0 -v
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/llm/pipeline_phase2.py tests/unit/llm/test_two_phase.py
git commit -m "feat: prepare llm retrieval query variants"
```

## Task 6: Enrich Compact Mapping Payloads

**Files:**
- Modify: `phentrieve/llm/pipeline_phase2.py`
- Modify: `tests/unit/llm/test_two_phase.py`

- [ ] **Step 1: Write failing payload enrichment tests**

Add these tests to `tests/unit/llm/test_two_phase.py`:

```python
def test_compact_mapping_item_includes_truncated_definition_and_matched_synonym(monkeypatch):
    from phentrieve.llm import pipeline_phase2

    def fake_enrich_results_with_details(results, data_dir_override=None):
        assert results == [{"hpo_id": "HP:0002359", "label": "Frequent falls"}]
        return [
            {
                "hpo_id": "HP:0002359",
                "label": "Frequent falls",
                "definition": (
                    "Increased frequency of falls relative to peers and expected "
                    "developmental stage with repeated loss of balance during gait."
                ),
                "synonyms": ["Frequent falls", "Repeated falls"],
            }
        ]

    monkeypatch.setattr(
        pipeline_phase2,
        "enrich_results_with_details",
        fake_enrich_results_with_details,
    )

    payload = pipeline_phase2.compact_mapping_item(
        {
            "phrase": "frequent falls",
            "category": "abnormal",
            "grounded_context": {
                "primary_chunk_text": "The child has frequent falls while walking.",
                "neighbor_chunk_texts": [],
            },
            "candidates": [
                {
                    "hpo_id": "HP:0002359",
                    "term_name": "Frequent falls",
                    "score": 0.91,
                    "matched_text": "Frequent falls",
                    "matched_component": "synonym",
                }
            ],
        },
        definition_char_limit=64,
    )

    candidate = payload["candidates"][0]
    assert candidate["definition"] == "Increased frequency of falls relative to peers and expected..."
    assert candidate["matched_synonym"] == "Frequent falls"
    assert candidate["matched_text"] == "Frequent falls"
    assert candidate["matched_component"] == "synonym"


def test_compact_mapping_item_continues_without_enrichment_on_database_error(monkeypatch):
    from phentrieve.llm import pipeline_phase2

    def fake_enrich_results_with_details(results, data_dir_override=None):
        raise RuntimeError("database locked")

    monkeypatch.setattr(
        pipeline_phase2,
        "enrich_results_with_details",
        fake_enrich_results_with_details,
    )

    payload = pipeline_phase2.compact_mapping_item(
        {
            "phrase": "frequent falls",
            "category": "abnormal",
            "grounded_context": {"primary_chunk_text": "frequent falls"},
            "candidates": [
                {
                    "hpo_id": "HP:0002359",
                    "term_name": "Frequent falls",
                    "score": 0.91,
                }
            ],
        }
    )

    assert payload["candidates"] == [
        {
            "id": "HP:0002359",
            "term": "Frequent falls",
            "retrieval_score": 0.91,
        }
    ]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/unit/llm/test_two_phase.py::test_compact_mapping_item_includes_truncated_definition_and_matched_synonym tests/unit/llm/test_two_phase.py::test_compact_mapping_item_continues_without_enrichment_on_database_error -n 0 -v
```

Expected: fail because `compact_mapping_item(...)` has no `definition_char_limit` argument and does not enrich candidates.

- [ ] **Step 3: Implement compact payload enrichment**

In `phentrieve/llm/pipeline_phase2.py`, add these imports and logger only if
they are not already present:

```python
import logging

from phentrieve.retrieval.details_enrichment import enrich_results_with_details

logger = logging.getLogger(__name__)
```

Add helpers above `compact_mapping_item(...)`:

```python
def truncate_definition(text: str, *, char_limit: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= char_limit:
        return stripped
    truncated = stripped[:char_limit].rsplit(" ", 1)[0].rstrip()
    return f"{truncated}..." if truncated else f"{stripped[:char_limit]}..."


def candidate_details_by_id(
    candidates: list[dict[str, Any]],
    *,
    data_dir_override: str | None = None,
) -> dict[str, dict[str, Any]]:
    if not candidates:
        return {}
    try:
        enriched = enrich_results_with_details(
            [
                {
                    "hpo_id": str(candidate.get("hpo_id", "")),
                    "label": str(candidate.get("term_name", "")),
                }
                for candidate in candidates
                if str(candidate.get("hpo_id", "")).strip()
            ],
            data_dir_override=data_dir_override,
        )
    except Exception:
        logger.exception("Failed to enrich LLM mapping candidates with HPO details")
        return {}
    return {str(row.get("hpo_id", "")): dict(row) for row in enriched}


def matched_synonym(candidate: dict[str, Any], details: dict[str, Any]) -> str | None:
    matched_text = str(candidate.get("matched_text") or "").strip()
    if not matched_text:
        return None
    if str(candidate.get("matched_component") or "").strip().lower() == "synonym":
        return matched_text
    synonyms = details.get("synonyms")
    if not isinstance(synonyms, list):
        return None
    for synonym in synonyms:
        synonym_text = str(synonym).strip()
        if synonym_text.lower() == matched_text.lower():
            return synonym_text
    return None
```

Change the `compact_mapping_item(...)` signature:

```python
def compact_mapping_item(
    item: dict[str, Any],
    *,
    item_id: str | None = None,
    enrich_candidates: bool = True,
    definition_char_limit: int = 240,
    data_dir_override: str | None = None,
) -> dict[str, Any]:
```

Before the candidate loop, add:

```python
    details_lookup = (
        candidate_details_by_id(
            list(item["candidates"]),
            data_dir_override=data_dir_override,
        )
        if enrich_candidates
        else {}
    )
```

Inside the candidate loop, after the base `compact_candidate` is created, add:

```python
        details = details_lookup.get(str(candidate.get("hpo_id", "")), {})
        definition = details.get("definition")
        if isinstance(definition, str) and definition.strip():
            compact_candidate["definition"] = truncate_definition(
                definition,
                char_limit=definition_char_limit,
            )
        synonym = matched_synonym(candidate, details)
        if synonym:
            compact_candidate["matched_synonym"] = synonym
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/unit/llm/test_two_phase.py::test_compact_mapping_item_includes_truncated_definition_and_matched_synonym tests/unit/llm/test_two_phase.py::test_compact_mapping_item_continues_without_enrichment_on_database_error tests/unit/llm/test_two_phase.py::test_resolve_with_mapping_prompt_normalizes_phrase_before_llm_call -n 0 -v
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/llm/pipeline_phase2.py tests/unit/llm/test_two_phase.py
git commit -m "feat: enrich llm mapping payloads"
```

## Task 7: Update Mapping Prompts For Enriched Payloads

**Files:**
- Modify: `phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml`
- Modify: `phentrieve/llm/prompts/templates/two_phase/en_mapping_batch.yaml`
- Modify: `tests/unit/llm/test_prompts.py`

- [ ] **Step 1: Write failing prompt tests**

Update `tests/unit/llm/test_prompts.py`:

```python
def test_get_mapping_prompt_loads_packaged_template() -> None:
    template = loader.get_mapping_prompt("fr")

    assert template.language == "fr"
    assert template.version == "v4.2.0"
    assert "You map clinical phenotype phrases to HPO terms." in template.system_prompt
    assert template.source_path.endswith("two_phase/en_mapping.yaml")
```

Update `test_get_batch_mapping_prompt_uses_shared_english_template_with_requested_language`:

```python
def test_get_batch_mapping_prompt_uses_shared_english_template_with_requested_language() -> None:
    template = loader.get_batch_mapping_prompt("de")

    assert template.language == "de"
    assert template.version == "v4.2.0"
    assert template.source_path.endswith("two_phase/en_mapping_batch.yaml")
```

Add:

```python
def test_mapping_prompts_describe_enriched_candidate_fields() -> None:
    for template in [loader.get_mapping_prompt("en"), loader.get_batch_mapping_prompt("en")]:
        system = template.system_prompt

        assert "optional definition" in system
        assert "optional matched_synonym" in system
        assert "optional matched_component" in system
        assert "optional matched_text" in system
        assert "most specific term supported by the evidence" in system
        assert "retrieval_score as a hint" in system


def test_mapping_prompt_example_prefers_specific_frequent_falls_candidate() -> None:
    template = loader.get_mapping_prompt("en")
    example = template.few_shot_examples[0]

    assert '"id": "HP:0002359"' in example["input"]
    assert '"matched_synonym": "Frequent falls"' in example["input"]
    assert example["output"] == '{"phrase": "frequent falls", "hpo_id": "HP:0002359"}'
```

- [ ] **Step 2: Run prompt tests and verify failure**

Run:

```bash
uv run pytest tests/unit/llm/test_prompts.py::test_get_mapping_prompt_loads_packaged_template tests/unit/llm/test_prompts.py::test_get_batch_mapping_prompt_uses_shared_english_template_with_requested_language tests/unit/llm/test_prompts.py::test_mapping_prompts_describe_enriched_candidate_fields tests/unit/llm/test_prompts.py::test_mapping_prompt_example_prefers_specific_frequent_falls_candidate -n 0 -v
```

Expected: fail because prompt versions are still `v4.1.0` and examples do not include enriched fields or `HP:0002359`.

- [ ] **Step 3: Update mapping prompt YAML**

In both mapping prompt files:

- change `version: "v4.1.0"` to `version: "v4.2.0"`;
- replace the candidate field bullet with:

```yaml
  - candidates with id, term, retrieval_score, optional definition, optional matched_synonym, optional matched_component, optional matched_text
```

- add rules:

```yaml
  - Use the grounded context, candidate definitions, and matched synonyms to disambiguate.
  - Treat retrieval_score as a hint, not the only decision rule.
  - Prefer the most specific term supported by the evidence.
```

Keep the existing abstention and candidate-list rules.

In `en_mapping.yaml`, replace the few-shot example using block-scalar strings:

```yaml
few_shot_examples:
  - input: |
      Map the following JSON payload to the best HPO candidate.
      Return JSON only.

      Payload:
      {"primary_chunk_text": "The child has frequent falls while walking.", "neighbor_chunk_texts": [], "phrase": "frequent falls", "category": "abnormal", "candidates": [{"id": "HP:0002355", "term": "Difficulty walking", "retrieval_score": 0.93, "definition": "Reduced ability to walk (ambulate).", "matched_synonym": "Walking difficulty"}, {"id": "HP:0002359", "term": "Frequent falls", "retrieval_score": 0.91, "definition": "Increased frequency of falls relative to peers.", "matched_synonym": "Frequent falls"}]}
    output: '{"phrase": "frequent falls", "hpo_id": "HP:0002359"}'
```

In `en_mapping_batch.yaml`, update the item-2 frequent falls candidate list to
include `HP:0002359` with definition and matched synonym, and update item-2
output to `HP:0002359`, preserving the existing `input: |` block-scalar string
and single-line JSON `output` string.

- [ ] **Step 4: Run prompt tests and verify pass**

Run:

```bash
uv run pytest tests/unit/llm/test_prompts.py::test_get_mapping_prompt_loads_packaged_template tests/unit/llm/test_prompts.py::test_get_batch_mapping_prompt_uses_shared_english_template_with_requested_language tests/unit/llm/test_prompts.py::test_mapping_prompts_describe_enriched_candidate_fields tests/unit/llm/test_prompts.py::test_mapping_prompt_example_prefers_specific_frequent_falls_candidate tests/unit/llm/test_prompts.py::test_mapping_prompts_keep_single_compact_example -n 0 -v
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml phentrieve/llm/prompts/templates/two_phase/en_mapping_batch.yaml tests/unit/llm/test_prompts.py
git commit -m "feat: teach mapping prompts enriched candidates"
```

## Task 8: Add Mapping Prompt Token Metric

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `tests/unit/llm/test_pipeline.py`
- Modify: `tests/unit/test_llm_benchmark.py`

- [ ] **Step 1: Write the failing metric tests**

Add this test to `tests/unit/llm/test_pipeline.py`:

```python
def test_phase2_mapping_prompt_tokens_per_request_is_recorded() -> None:
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        grounded_phenotype(
                            "frequent falls",
                            "Abnormal",
                            chunk_ids=[1],
                            evidence_text="frequent falls",
                        )
                    ]
                }
            },
            {
                "parsed": {
                    "phrase": "frequent falls",
                    "hpo_id": "HP:0002359",
                },
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 6,
                    "total_tokens": 48,
                },
                "request_count": 1,
            },
        ]
    )
    tool_executor = FakeToolExecutor(
        batch_results=[
            {
                "phrase": "frequent falls",
                "candidates": [
                    {
                        "hpo_id": "HP:0002355",
                        "term_name": "Difficulty walking",
                        "score": 0.81,
                    },
                    {
                        "hpo_id": "HP:0002359",
                        "term_name": "Frequent falls",
                        "score": 0.79,
                    },
                ],
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider,
        tool_executor=tool_executor,
        mapping_batch_size=1,
    )

    result = pipeline.run(
        text="The child has frequent falls while walking.",
        grounded_chunks=[{"chunk_id": 1, "text": "The child has frequent falls while walking."}],
        config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
    )

    assert result.meta.phase_counts["phase2_mapping_prompt_tokens_per_request"] == 42
```

Add this test to `tests/unit/test_llm_benchmark.py`:

```python
def test_run_llm_benchmark_observability_includes_mapping_prompt_token_metric(monkeypatch):
    def fake_load_benchmark_data(test_path: Path, dataset: str):
        return {
            "metadata": {"dataset_name": "phenobert_GeneReviews"},
            "documents": [
                {
                    "id": "doc-1",
                    "text": "Clinical text",
                    "gold_hpo_terms": [],
                    "source_dataset": "GeneReviews",
                }
            ],
        }

    class _FakePipeline:
        def __init__(self, provider):
            self.provider = provider

        def warmup(self, *, language: str) -> None:
            return None

        def run(self, *, text, grounded_chunks, config):
            from phentrieve.llm.types import LLMExtractionResult, LLMMeta

            return LLMExtractionResult(
                terms=[],
                meta=LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                    phase_counts={
                        "phase2_mapping_prompt_tokens_per_request": 42,
                        "phase1_evidence_kept": 1,
                        "phase1_evidence_dropped": 0,
                    },
                    trace={
                        "phase1_evidence_validation": {
                            "status": "validated",
                            "kept_count": 1,
                            "dropped_count": 0,
                            "repair_count": 0,
                            "downgraded_count": 0,
                            "dropped": [],
                            "repairs": [],
                        }
                    },
                ),
            )

    monkeypatch.setattr(llm_benchmark, "load_benchmark_data", fake_load_benchmark_data)
    monkeypatch.setattr(llm_benchmark, "get_llm_provider", lambda llm_model: object())
    monkeypatch.setattr(llm_benchmark, "TwoPhaseLLMPipeline", _FakePipeline)

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        dataset="GeneReviews",
    )

    record = result["prediction_records"][0]
    assert record["metadata"]["observability"]["phase2_mapping_prompt_tokens_per_request"] == 42
    assert record["trace"]["phase1_evidence_validation"]["status"] == "validated"
```

- [ ] **Step 2: Run metric tests and verify failure**

Run:

```bash
uv run pytest tests/unit/llm/test_pipeline.py::test_phase2_mapping_prompt_tokens_per_request_is_recorded tests/unit/test_llm_benchmark.py::test_run_llm_benchmark_observability_includes_mapping_prompt_token_metric -n 0 -v
```

Expected: pipeline test fails because the phase count is missing. The benchmark test may already pass through `**phase_counts`; keep it as a regression guard.

- [ ] **Step 3: Implement the metric**

In `phentrieve/llm/pipeline.py`, initialize the metric in `phase_counts`:

```python
            "phase2_mapping_prompt_tokens_per_request": 0,
```

After phase-2B LLM mapping returns and after `phase_request_counts["phase2b_llm_requests"] = mapping_request_count`, add:

```python
                if mapping_request_count > 0:
                    phase_counts["phase2_mapping_prompt_tokens_per_request"] = round(
                        mapping_prompt_tokens / mapping_request_count
                    )
```

No benchmark code change is required if the benchmark test passes, because
`_build_observability_counts(...)` already expands `phase_counts`.

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/unit/llm/test_pipeline.py::test_phase2_mapping_prompt_tokens_per_request_is_recorded tests/unit/test_llm_benchmark.py::test_run_llm_benchmark_observability_includes_mapping_prompt_token_metric -n 0 -v
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add phentrieve/llm/pipeline.py tests/unit/llm/test_pipeline.py tests/unit/test_llm_benchmark.py
git commit -m "feat: record llm mapping prompt token metric"
```

## Focused Regression Sweep

After Task 8, run the LLM-focused regression sweep:

```bash
uv run pytest tests/unit/llm tests/unit/test_llm_benchmark.py tests/integration/llm -n 0 -v
```

Expected: pass. If it fails, write a focused failing test in the affected
existing test file, verify that single test fails, implement the minimal fix,
rerun the focused test and this sweep, then commit the fix with the affected
files only.

## Final Verification

Run required repo checks:

```bash
make check
make typecheck-fast
make test
```

Expected: all pass.

If frontend files were not changed, frontend CI commands are not required for
this scope.

## A/B Benchmark Gates

Run the before benchmark before implementation if a comparable artifact does
not already exist for the base commit. Run the after benchmark after Task 8 and
the Focused Regression Sweep.

GeneReviews smoke command:

```bash
uv run phentrieve benchmark llm \
  --test-file tests/data/en/phenobert \
  --dataset GeneReviews \
  --doc-id GeneReviews_NBK1277 \
  --doc-id GeneReviews_NBK148668 \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash \
  --llm-seed 1 \
  --ontology-aware-metrics \
  --capture-phase1-debug \
  --output-path .planning/analysis/llm-evidence-validation-gene-reviews-after.json
```

CSC smoke command:

```bash
uv run phentrieve benchmark llm \
  --test-file tests/data/en/raghpo_paper \
  --dataset CSC \
  --doc-id CSC_1 \
  --doc-id CSC_2 \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash \
  --llm-seed 1 \
  --ontology-aware-metrics \
  --capture-phase1-debug \
  --output-path .planning/analysis/llm-evidence-validation-csc-after.json
```

Compare before/after token growth when both artifacts exist:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

before = json.loads(Path(".planning/analysis/llm-evidence-validation-gene-reviews-before.json").read_text())
after = json.loads(Path(".planning/analysis/llm-evidence-validation-gene-reviews-after.json").read_text())

def metric(payload):
    values = []
    for record in payload["prediction_records"]:
        value = record["metadata"]["observability"].get("phase2_mapping_prompt_tokens_per_request")
        if value:
            values.append(float(value))
    return sum(values) / len(values) if values else 0.0

before_value = metric(before)
after_value = metric(after)
growth = 0.0 if before_value == 0 else (after_value - before_value) / before_value
print({"before": before_value, "after": after_value, "growth": growth})
raise SystemExit(0 if growth <= 0.25 else 1)
PY
```

Expected: exit code 0 and growth less than or equal to `0.25`.

Skip benchmark only when one of these is true:

- no configured provider credentials and no reachable local provider;
- expected runtime is not acceptable for the current execution window;
- required HPO data or ontology graph artifacts are missing;
- provider quota or rate limit blocks the run.

When a benchmark is skipped, record the skipped command and exact reason in the
implementation summary or a follow-up `.planning/analysis/...` note.

## Plan Self-Review

Spec coverage:

- Phase-2A extraction prerequisite: Task 1.
- Evidence validator, word-boundary exact matching, document-absolute offset
  handling, multi-chunk offset downgrades, and trace persistence: Tasks 2 and 3.
- Source-faithful phase 1: Task 4.
- Retrieval-query rewrite behavior: Task 5.
- Enriched mapping payloads: Task 6.
- Prompt updates: Task 7.
- Token-growth observability and benchmark gates: Task 8 and A/B Benchmark Gates.
- Focused tests and final required checks: Tasks 1-8, Focused Regression Sweep,
  and Final Verification.

Placeholder scan:

- No `TBD`, `TODO`, `FIXME`, or open-ended implementation steps are present.

Scope check:

- The plan does not add reranking, cross-encoder reranking, LLM-as-judge
  reranking, generic reranking, hybrid lexical/dense retrieval, or a new
  retrieval subsystem.
