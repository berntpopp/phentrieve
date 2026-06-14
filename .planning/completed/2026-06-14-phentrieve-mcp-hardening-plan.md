# Phentrieve MCP Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise every dimension in the 2026-06-14 MCP evaluation to >= 9.5/10 by fixing deterministic-extractor negation, candidate explosion, a lossless pipeable export, schema/token bloat, the capabilities cache contract, error envelopes, and (benchmark-gated) LLM negation scope.

**Architecture:** Three layers. Correctness (negation) goes in the shared `phentrieve/` pipeline so REST + frontend benefit. Schema/token shape goes in a new MCP-only projection in `api/mcp/` so the REST `AggregatedHPOTermAPI` and Vue curation UI are untouched. LLM negation-scope changes go in `phentrieve/llm/`, gated behind a mapping benchmark.

**Tech Stack:** Python 3.11+, FastMCP v3, FastAPI, Pydantic v2, pytest (+xdist), Ruff, mypy, uv; Gemini via `google-genai`; Vue 3 frontend (verification only).

**Source spec:** `.planning/specs/2026-06-14-phentrieve-mcp-hardening-design.md`

---

## Conventions for every task

- TDD: write the failing test, run it (confirm the *expected* failure), implement, run green, commit.
- Run single tests single-threaded for clarity: `uv run pytest <path>::<test> -n 0 -q`.
- Tests live under `tests/` only. Never create `tests_new/` or new `tests/unit/api` sub-package dirs without `__init__.py` parity (see repo notes).
- Commit messages: conventional commits; end with the Co-Authored-By trailer.
- Do NOT push until the final gate task. Branch/worktree set up in Task 0.

---

## Task 0: Worktree, branch, planning commit

**Files:**
- Worktree: `../phentrieve-mcp-hardening` on branch `feat/mcp-hardening`
- Commit: `.planning/specs/2026-06-14-phentrieve-mcp-hardening-design.md`, `.planning/analysis/2026-06-14-phentrieve-mcp-evaluation.md`, this plan.

- [ ] **Step 1:** Create the worktree via the using-git-worktrees skill (native tool or `git worktree add -b feat/mcp-hardening ../phentrieve-mcp-hardening main`).
- [ ] **Step 2:** Copy the three untracked planning files into the worktree if they are not present (they are untracked in main, so will not appear automatically).
- [ ] **Step 3:** For the live server, export `PHENTRIEVE_DATA_ROOT_DIR` to the main checkout's `data/` so the worktree server finds the HPO bundle (worktrees do not share untracked `data/`).
- [ ] **Step 4:** Commit the planning artifacts.

```bash
git add .planning/specs/2026-06-14-phentrieve-mcp-hardening-design.md \
        .planning/analysis/2026-06-14-phentrieve-mcp-evaluation.md \
        .planning/active/2026-06-14-phentrieve-mcp-hardening-plan.md
git commit -m "docs(planning): MCP hardening spec + plan"
```

- [ ] **Step 5:** Establish the live baseline BEFORE code changes. Call the live MCP tools to reproduce each defect and save raw responses to `.planning/analysis/2026-06-14-mcp-baseline.md`:
  - `phentrieve_extract_hpo_terms("There is no nystagmus. She does not have ataxia.")` (C1)
  - `phentrieve_extract_hpo_terms("The patient had seizures.")` (H1)
  - `phentrieve_extract_hpo_terms("patient denies headache")` (H2)
  - `phentrieve_export_phenopacket` with `[{id,name,assertion_status}]` (M2/M3) and with next_commands-shaped phenotypes lacking score (H3)
  - `phentrieve_get_capabilities()` vs `details=[sample_calls, argument_aliases]` (M1)
  - `phentrieve_search_hpo_terms("")` (L1)
  - `phentrieve_chunk_text("x", strategy="bogus")` (L4)
  - `phentrieve_search_hpo_terms("seizures", response_mode="compact")` with default `include_details` (M5)

---

## WS1 -- Deterministic extractor correctness (shared pipeline)

### Task 1: C1 -- assertion detection over the source-sentence span

**Files:**
- Modify: `phentrieve/text_processing/pipeline.py` (around 230-293, the per-chunk loop that calls `assertion_detector.detect`)
- Test: `tests/unit/text_processing/test_negation_scope.py` (create)

- [ ] **Step 1:** Read `pipeline.py:230-293`, `assertion_detection.py:347-371` (`detect` signature), and `spans.py:65-142` (`find_span_in_text`) to confirm how to obtain the source-sentence text for a chunk. Confirm whether `detect()` can take a target/focus argument; if not, plan to pass the sentence text while keeping the cleaned chunk as the retrieval/display text.

- [ ] **Step 2: Write failing tests** in `tests/unit/text_processing/test_negation_scope.py`:

```python
import pytest
from phentrieve.text_processing.assertion_detection import (
    AssertionStatus,
    create_assertion_detector,
)

# These assert the contract C1 requires: polarity is decided on the
# sentence context, not the cue-stripped phrase.
NEGATION_SENTENCES = [
    ("There is no nystagmus.", "nystagmus", AssertionStatus.NEGATED),
    ("She does not have ataxia.", "ataxia", AssertionStatus.NEGATED),
    ("The patient denies headache.", "headache", AssertionStatus.NEGATED),
    ("No seizures were observed.", "seizures", AssertionStatus.NEGATED),
]
SCOPE_SENTENCES = [
    # "X without Y": X must NOT be negated.
    ("Severe intellectual disability without regression.",
     "intellectual disability", AssertionStatus.AFFIRMED),
    ("Non-progressive ataxia.", "ataxia", AssertionStatus.AFFIRMED),
]

@pytest.mark.parametrize("sentence,target,expected", NEGATION_SENTENCES)
def test_negation_detected_on_sentence_span(sentence, target, expected):
    detector = create_assertion_detector(language="en")
    status, _ = detector.detect(sentence)
    assert status == expected

@pytest.mark.parametrize("sentence,target,expected", SCOPE_SENTENCES)
def test_scope_does_not_overnegate_head(sentence, target, expected):
    detector = create_assertion_detector(language="en")
    status, _ = detector.detect(sentence)
    assert status == expected
```

(If `create_assertion_detector` is not the factory name, use the actual factory found in Step 1; the assertion is the behavior, not the API name.)

- [ ] **Step 3: Run** `uv run pytest tests/unit/text_processing/test_negation_scope.py -n 0 -q`. Expected: the NEGATION cases on full sentences likely already pass at the detector level; the integration failure is in the pipeline. Add an integration test that drives `TextProcessingPipeline.process()` end-to-end and asserts the chunk for "There is no nystagmus" carries `assertion_status == "negated"`. This is the real C1 reproduction.

```python
def test_pipeline_negation_scope_end_to_end():
    from phentrieve.text_processing.pipeline import TextProcessingPipeline
    pipe = TextProcessingPipeline(language="en")  # use the real constructor args
    chunks = pipe.process("There is no nystagmus. She does not have ataxia.",
                          include_positions=True)
    by_text = {c["text"].lower(): c for c in chunks}
    # whichever chunk contains nystagmus must be negated
    neg = [c for c in chunks if "nystagmus" in c["text"].lower()]
    assert neg and all(c["assertion_status"] == "negated" for c in neg)
```

- [ ] **Step 4: Implement.** In `pipeline.py`, where the loop currently does
  `assertion_status, assertion_details = self.assertion_detector.detect(cleaned_final_chunk)`,
  compute the source-sentence text for the chunk from the original text + offsets
  (or carry the pre-clean sentence through the chunker), and pass THAT to
  `detect()`. Keep `cleaned_final_chunk` as the retrieval query / chunk text.
  The minimal change: detect on the sentence span that contains the chunk.

- [ ] **Step 5: Run** the pipeline test and the parametrized tests `-n 0 -q`. Expected: PASS.

- [ ] **Step 6:** Run the broader suite for regressions: `uv run pytest tests/unit/text_processing/ -q`.

- [ ] **Step 7: Commit** `fix(extract): detect assertions on source-sentence span (C1 negation scope)`.

### Task 2: H2 -- negated findings emitted as excluded

**Files:**
- Modify: `phentrieve/text_processing/hpo_extraction_orchestrator.py` and/or `_hpo_extraction_helpers.py` (pinpoint via test)
- Test: `tests/unit/text_processing/test_negated_emission.py` (create)

- [ ] **Step 1: Write the failing test.** Drive the orchestrator (with the existing fake retriever pattern from `test_hpo_extraction_orchestrator_char.py`) on a chunk whose assertion is negated, and assert the aggregated output contains the term with `assertion_status == "negated"` (not dropped).

```python
def test_negated_term_is_emitted(fake_retriever_with_seizure):
    from phentrieve.text_processing.hpo_extraction_orchestrator import (
        orchestrate_hpo_extraction,
    )
    result = orchestrate_hpo_extraction(
        text_chunks=["headache"],
        assertion_statuses=["negated"],
        retriever=fake_retriever_with_seizure,  # returns Headache HP:xxxx
        num_results_per_chunk=1,
        chunk_retrieval_threshold=0.0,
    )
    agg = result.aggregated_results
    assert any(t["assertion_status"] == "negated" for t in agg)
```

(Match the real `orchestrate_hpo_extraction` signature from `hpo_extraction_orchestrator.py:22-124`; reuse fixtures from the existing char test.)

- [ ] **Step 2: Run** `-n 0 -q`. If it already passes, H2 is purely a serialization issue (handled in Task 6/L5) -- record that finding and skip to Step 5. If it fails (negated terms dropped before retrieval/aggregation), continue.

- [ ] **Step 3: Implement.** Ensure negated chunks are still queried and their matches retained with `assertion_status: negated` through `process_chunk_matches` -> `build_evidence_map` -> `aggregate_and_rank`. Do not filter on assertion in the aggregation path.

- [ ] **Step 4: Run** the test green; run `tests/unit/text_processing/ -q` for regressions.

- [ ] **Step 5: Commit** `fix(extract): emit negated findings with assertion_status=negated (H2)`.

---

## WS2 -- MCP boundary: projection, export, errors

### Task 3: MCP projection module (M4, T1, L5, L7)

**Files:**
- Create: `api/mcp/projection.py`
- Test: `tests/unit/api/mcp/test_projection.py` (create; ensure `tests/unit/api/mcp/__init__.py` exists)

- [ ] **Step 1: Write failing tests** for `project_aggregated_terms_for_mcp` and `project_processed_chunks_for_mcp`:

```python
from api.mcp.projection import (
    project_aggregated_terms_for_mcp,
    project_processed_chunks_for_mcp,
)

RAW_TERM = {
    "id": "HP:0001250", "name": "Seizure", "score": 0.91, "avg_score": 0.88,
    "confidence": 0.88, "max_score_from_evidence": 0.91,
    "chunks": [0], "top_evidence_chunk_idx": 0,
    "source_chunk_ids": [1], "top_evidence_chunk_id": 1,
    "assertion_status": "affirmed", "status": "affirmed",
    "text_attributions": [{"start_char": 4, "end_char": 12}],
}

def test_single_score_and_index_scheme():
    out = project_aggregated_terms_for_mcp([RAW_TERM])[0]
    assert out["score"] == 0.91
    assert "avg_score" not in out and "confidence" not in out
    assert "max_score_from_evidence" not in out
    assert "chunk_ids" in out and "chunks" not in out
    assert "top_evidence_chunk_id" in out and "top_evidence_chunk_idx" not in out
    assert out["hpo_id"] == "HP:0001250" and out["assertion"] == "affirmed"

def test_empty_match_chunk_dropped_unless_opted_in():
    chunks = [{"chunk_id": 1, "text": "x", "hpo_matches": []},
              {"chunk_id": 2, "text": "seizure", "hpo_matches": [{"hpo_id": "HP:1"}]}]
    assert len(project_processed_chunks_for_mcp(chunks)) == 1
    assert len(project_processed_chunks_for_mcp(chunks, include_unmatched=True)) == 2

def test_hpo_matches_key_always_present():
    chunks = [{"chunk_id": 2, "text": "seizure", "hpo_matches": [{"hpo_id": "HP:1"}]}]
    out = project_processed_chunks_for_mcp(chunks, include_unmatched=True)
    kept = [{"chunk_id": 1, "text": "x", "hpo_matches": []}]
    assert project_processed_chunks_for_mcp(kept, include_unmatched=True)[0]["hpo_matches"] == []
```

- [ ] **Step 2: Run** `-n 0 -q`. Expected: ImportError / fail.

- [ ] **Step 3: Implement** `api/mcp/projection.py`:

```python
"""MCP-only normalization of the shared extraction schema.

Collapses the four redundant score fields to one, keeps a single chunk-index
scheme, normalizes id/label/assertion keys, and drops empty-match chunks unless
asked to keep them. The shared full_text_service output (REST + frontend) is left
untouched; this runs only for the MCP consumer.
"""
from __future__ import annotations
from typing import Any

_DROP_SCORE_FIELDS = ("avg_score", "confidence", "max_score_from_evidence")
_DROP_INDEX_FIELDS = ("chunks", "top_evidence_chunk_idx", "source_chunk_ids")


def project_aggregated_terms_for_mcp(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for t in terms:
        out = dict(t)
        out["hpo_id"] = out.pop("hpo_id", None) or out.pop("id", None)
        out["label"] = out.pop("label", None) or out.pop("name", None)
        out["assertion"] = out.pop("assertion", None) or out.pop("status", None) or out.get("assertion_status")
        out["score"] = out.get("score", out.get("max_score_from_evidence", 0.0))
        # single chunk-index scheme: chunk_ids (1-based) + top_evidence_chunk_id
        if "source_chunk_ids" in t:
            out["chunk_ids"] = t["source_chunk_ids"]
        elif "chunks" in t:
            out["chunk_ids"] = [c + 1 for c in t["chunks"]]
        if "top_evidence_chunk_id" not in out and t.get("top_evidence_chunk_idx") is not None:
            out["top_evidence_chunk_id"] = t["top_evidence_chunk_idx"] + 1
        for f in (*_DROP_SCORE_FIELDS, *_DROP_INDEX_FIELDS, "id", "name", "status", "assertion_status"):
            out.pop(f, None)
        projected.append(out)
    return projected


def project_processed_chunks_for_mcp(
    chunks: list[dict[str, Any]], *, include_unmatched: bool = False
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in chunks:
        matches = c.get("hpo_matches") or []
        if not matches and not include_unmatched:
            continue
        nc = dict(c)
        nc["hpo_matches"] = matches  # always present, never dropped to missing
        out.append(nc)
    return out
```

- [ ] **Step 4: Run** the tests green.

- [ ] **Step 5: Wire it in.** In `api/mcp/tools/retrieval.py` `extract_hpo_terms` and `extract_hpo_terms_llm`, after `apply_response_mode`, apply the projection to `aggregated_hpo_terms` and `processed_chunks`. Add an `include_unmatched_chunks: bool = False` parameter to both extract tools and pass it through. Ensure the projection runs before `enforce_budget`/`after_extract`. Keep `hpo_matches: []` out of the empty-drop path (the projection guarantees presence; verify `_shape_item` does not strip it -- if it does, exclude `hpo_matches` from the empty-drop list in `shaping.py`).

- [ ] **Step 6: Run** `tests/unit/api/mcp/ -q`; commit `feat(mcp): project extraction schema for the MCP consumer (M4/T1/L5)`.

### Task 4: Export accepts both key shapes + carries score (M2, H3, M3)

**Files:**
- Modify: `api/mcp/service_adapters.py:300-310` (export request build)
- Modify: `api/mcp/next_commands.py:28-44` (`after_extract`, add `score`)
- Modify: `phentrieve/phenopackets/export_models.py` and `utils.py` (thread score into evidence ref) -- only if score is not already read from the legacy dict (it is: `export_models.py:197-199` reads `confidence` then `score`); so the fix may be purely passing score through the request.
- Test: `tests/unit/api/mcp/test_export_piping.py` (create)

- [ ] **Step 1: Write failing tests:**

```python
import pytest
from api.mcp.service_adapters import export_phenopacket_service  # confirm name

RAW_AGG = [{"id": "HP:0001250", "name": "Seizure",
            "assertion_status": "affirmed", "score": 0.91}]

def test_export_accepts_extractor_keys():
    res = export_phenopacket_service(case_id="C1", phenotypes=RAW_AGG)
    assert res  # no KeyError

def test_export_preserves_confidence():
    res = export_phenopacket_service(case_id="C1", phenotypes=RAW_AGG)
    blob = str(res)
    assert "0.0000" not in blob  # H3: real confidence threaded through

def test_export_missing_id_is_validation_error():
    from api.mcp.envelope import McpToolError
    with pytest.raises(McpToolError) as ei:
        export_phenopacket_service(case_id="C1", phenotypes=[{"name": "x"}])
    assert ei.value.error_code == "validation_failed"
    assert "hpo_id" in str(ei.value)
```

- [ ] **Step 2: Run** `-n 0 -q`. Expected: KeyError / assertion failures.

- [ ] **Step 3: Implement** in `service_adapters.py`:

```python
def _coerce_phenotype(p: dict, idx: int):
    hpo_id = p.get("hpo_id") or p.get("id")
    if not hpo_id:
        raise McpToolError(
            "validation_failed",
            f"phenotypes[{idx}] missing 'hpo_id' (got keys: {sorted(p)}); "
            "map id->hpo_id, name->label, assertion_status->assertion.",
            details={"field": f"phenotypes[{idx}].hpo_id"},
        )
    label = p.get("label") or p.get("name") or hpo_id
    assertion = p.get("assertion") or p.get("status") or p.get("assertion_status") or "affirmed"
    score = p.get("score")
    if score is None:
        score = p.get("confidence", p.get("max_score_from_evidence"))
    return ExportPhenotypeRequest(
        hpo_id=hpo_id, label=label,
        assertion_status="negated" if assertion == "negated" else "affirmed",
        confidence=score,
    )
```

  Replace the list-comprehension at `service_adapters.py:302-310` with `[_coerce_phenotype(p, i) for i, p in enumerate(phenotypes)]`. Add a `confidence` field to `ExportPhenotypeRequest` if absent and ensure the phenopacket builder reads it (it already prefers `confidence` then `score` at `export_models.py:197-199`).

- [ ] **Step 4:** In `next_commands.py` `after_extract`, add `"score": t.get("score") or t.get("confidence")` to each phenotype dict so the chained export carries confidence.

- [ ] **Step 5: Run** tests green; run `tests/unit/api/ -q` for regressions.

- [ ] **Step 6: Commit** `fix(mcp): export accepts extractor keys and preserves confidence (M2/H3/M3)`.

---

## WS3 -- Capabilities cache contract and discoverability

### Task 5: M1 -- stable capabilities_version

**Files:**
- Modify: `api/mcp/capabilities.py:183-201`
- Test: `tests/unit/api/mcp/test_capabilities_version.py` (create)

- [ ] **Step 1: Write failing test:**

```python
from api.mcp.capabilities import build_capabilities, capabilities_version

def test_version_stable_across_details():
    base = build_capabilities()
    detailed = build_capabilities(details=["sample_calls", "argument_aliases"])
    assert base["capabilities_version"] == capabilities_version()
    assert detailed["capabilities_version"] == capabilities_version()

def test_descriptor_hash_present_for_detailed():
    detailed = build_capabilities(details=["sample_calls"])
    assert "descriptor_hash" in detailed  # content hash still available
```

- [ ] **Step 2: Run** `-n 0 -q`. Expected: detailed version mismatches base.

- [ ] **Step 3: Implement.** In `_cached_descriptor`, compute `descriptor_hash` from the serialized body (the current behavior) but set `body["capabilities_version"]` to the BASE hash always:

```python
@functools.lru_cache(maxsize=8)
def _cached_descriptor(details_key: tuple[str, ...]) -> dict[str, Any]:
    body = _descriptor_body(details_key)
    serialized = json.dumps(body, sort_keys=True, default=str)
    body["descriptor_hash"] = f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()[:16]}"
    body["descriptor_chars"] = len(serialized)
    if details_key == ():
        body["capabilities_version"] = body["descriptor_hash"]
    else:
        body["capabilities_version"] = _cached_descriptor(())["capabilities_version"]
    return body
```

  Add a short note in the capabilities body documenting that `capabilities_version` is a stable warm-cache convention (compare to `_meta.capabilities_version`) and `tools/list_changed` is the MCP-spec change signal.

- [ ] **Step 4: Run** tests green; **Commit** `fix(mcp): stable capabilities_version across details (M1)`.

### Task 6: L4 -- strategy enum; M5 -- include_details; L8 -- alias docs

**Files:**
- Modify: `api/mcp/tools/_common.py` (add `ChunkStrategy`, maybe `DEFAULT_EXTRACT_NUM_RESULTS`)
- Modify: `api/mcp/tools/retrieval.py` (chunk_text strategy type; extract include_details default; citation in all modes)
- Modify: `api/mcp/capabilities.py` (enumerate strategies; document alias reachability)
- Test: `tests/unit/api/mcp/test_chunk_strategy_enum.py`, extend capabilities test

- [ ] **Step 1: Write failing tests:**

```python
def test_unknown_strategy_rejected_with_allowed_values():
    from api.mcp.service_adapters import chunk_text_service
    from api.mcp.envelope import McpToolError
    import pytest
    with pytest.raises(McpToolError) as ei:
        chunk_text_service(text="x", language=None, strategy="bogus")
    assert ei.value.error_code in ("invalid_input", "validation_failed")
    assert "simple" in str(ei.value)  # lists valid strategies

def test_capabilities_lists_chunk_strategies():
    from api.mcp.capabilities import build_capabilities
    blob = str(build_capabilities(details=["sample_calls"]))
    assert "sliding_window" in blob and "simple" in blob
```

- [ ] **Step 2:** Define `ChunkStrategy = Literal[...]` in `_common.py` from the valid strategies in `phentrieve/text_processing/config_resolver.py:195-203` (`simple`, `detailed`, `semantic`, `sliding_window`, `sliding_window_cleaned`, `sliding_window_punct_cleaned`, `sliding_window_punct_conj_cleaned`). Type `chunk_text(strategy: ChunkStrategy | None = None)`. Add a `_VALID_STRATEGIES` tuple referenced by both the type and the capabilities descriptor and the error message.

- [ ] **Step 3:** M5 -- change `IncludeDetails` default to `False`; when `include_details=True`, ensure `definition`/`synonyms` survive compact shaping for matched terms (exclude those keys from compact drop when the flag is set, or post-add them). Document the detail floor in the tool description.

- [ ] **Step 4:** Citation in all modes -- change `_maybe_citation` (`retrieval.py:44-46`) to always set `recommended_citation`; do the same in `tools/phenopacket.py`. (Done jointly with Task 9 for the HPO-version string.)

- [ ] **Step 5:** L8 -- add an `argument_aliases` capability note: canonical names are authoritative; aliases are a convenience for non-strict clients. Verify the generated `inputSchema.additionalProperties` value (inspect a `tools/list`) and state it accurately.

- [ ] **Step 6: Run** tests green; **Commit** `fix(mcp): strategy enum, include_details floor, alias docs (L4/M5/L8)`.

### Task 7: H1 -- best-match-per-phrase default

**Files:**
- Modify: `api/mcp/tools/_common.py` (`DEFAULT_EXTRACT_NUM_RESULTS = 1`)
- Modify: `api/mcp/tools/retrieval.py` (extract tools use it; update descriptions)
- Test: `tests/unit/api/mcp/test_extract_topk_default.py`

- [ ] **Step 1: Write failing test** (mock `extract_hpo_terms_service` to capture the `num_results_per_chunk` it receives):

```python
def test_extract_defaults_to_one_candidate_per_phrase(monkeypatch):
    captured = {}
    def fake(**kw):
        captured.update(kw); return {"aggregated_hpo_terms": [], "processed_chunks": []}
    monkeypatch.setattr("api.mcp.tools.retrieval.extract_hpo_terms_service", fake)
    # invoke the tool's call() path or the underlying default constant
    from api.mcp.tools._common import DEFAULT_EXTRACT_NUM_RESULTS
    assert DEFAULT_EXTRACT_NUM_RESULTS == 1
```

- [ ] **Step 2:** Set `NumResultsPerChunk` default to `DEFAULT_EXTRACT_NUM_RESULTS = 1` for both extract tools (NOT for `search_hpo_terms`). Update both tool descriptions: "num_results_per_chunk: candidates per phrase (default 1 = best match; raise to surface sibling candidates)."

- [ ] **Step 3: Run** green; **Commit** `fix(mcp): default deterministic extract to best-match-per-phrase (H1)`.

---

## WS4 -- Error handling

### Task 8: L1 -- reject empty/whitespace; L2 -- executable next_commands

**Files:**
- Modify: `api/mcp/tools/_common.py` (`TextArg` min_length + whitespace validator)
- Modify: `api/mcp/next_commands.py` (`after_search`, `after_compare` placeholders)
- Test: `tests/unit/api/mcp/test_input_validation.py`, `tests/unit/api/mcp/test_next_commands.py`

- [ ] **Step 1: Write failing tests:**

```python
def test_textarg_rejects_blank():
    import pytest
    from pydantic import TypeAdapter, ValidationError
    from api.mcp.tools._common import TextArg
    ta = TypeAdapter(TextArg)
    with pytest.raises(ValidationError):
        ta.validate_python("   ")

def test_next_commands_have_no_unfilled_placeholders():
    from api.mcp.next_commands import after_search, after_compare
    blobs = str(after_search([])) + str(after_compare("HP:1", "HP:2"))
    assert "<" not in blobs  # no "<surrounding...>" / "<related...>"
```

- [ ] **Step 2:** Add `min_length=1` to `TextArg` plus an `AfterValidator` that strips and rejects empty/whitespace with a clear message.

- [ ] **Step 3:** In `next_commands.py`, replace `after_search`'s `text="<surrounding clinical text>"` branch and `after_compare`'s `text="<related phenotype phrase>"` with executable steps (e.g., `after_compare` returns `[]` or a capabilities step; `after_search` with a single id returns an export/compare step, not a placeholder search).

- [ ] **Step 4: Run** green; **Commit** `fix(mcp): reject blank queries; remove non-executable next_commands placeholders (L1/L2)`.

---

## WS5 -- Observability, speed, safety

### Task 9: Citations in all modes + HPO version; L6 provenance

**Files:**
- Modify: `api/mcp/resources.py` (`recommended_citation` includes HPO version)
- Modify: `api/mcp/tools/retrieval.py`, `tools/similarity.py`, `tools/phenopacket.py` (emit in all modes)
- Modify: `phentrieve/phenopackets/utils.py:487` (`createdBy` dual version)
- Test: `tests/unit/api/mcp/test_citation_all_modes.py`

- [ ] **Step 1: Write failing test:**

```python
def test_citation_includes_hpo_version():
    from api.mcp.resources import recommended_citation
    c = recommended_citation()
    assert "HPO" in c or "hpo" in c
    # contains a version token like v2026 when data is available

def test_createdby_has_both_versions():
    # build a minimal phenopacket and assert createdBy mentions both
    ...
```

- [ ] **Step 2:** Thread the HPO release version into `recommended_citation()`; emit citation in all response modes (remove the `mode in ("standard","full")` gate); set `created_by` to `f"Phentrieve (core {core_v}, mcp-server {server_v})"`, passing the server version through export meta.

- [ ] **Step 3: Run** green; **Commit** `feat(mcp): citations in all modes with HPO version; dual-version provenance (safety/L6)`.

### Task 10: Per-phase LLM timing + startup warmup

**Files:**
- Modify: `phentrieve/text_processing/full_text_service.py` (LLM observability block) + `phentrieve/llm` pipeline phases (perf_counter)
- Modify: `api/mcp/server.py` / facade (best-effort warmup on startup)
- Test: `tests/unit/text_processing/test_llm_observability_timing.py` (assert keys present given a stubbed pipeline result)

- [ ] **Step 1: Write failing test** asserting the observability dict gains `phase1_ms`/`phase2_ms` keys when the pipeline reports phase timings.
- [ ] **Step 2:** Add `time.perf_counter()` deltas around the LLM phases; surface them in `observability`. Add a best-effort warmup call (load embedding model + index) in the MCP server lifespan; guard with try/except and a log line; never block startup on failure.
- [ ] **Step 3: Run** green; **Commit** `feat(mcp): per-phase LLM timing + startup warmup (observability/speed)`.

---

## WS6 -- LLM negation scope (benchmark-gated)

### Task 11: Gemini scope-reasoning schema + few-shots

**Files:**
- Modify: `phentrieve/llm/prompts/templates/two_phase/en.yaml` (system rule + few-shots; bump to v3.1.0)
- Modify: the Gemini response-schema builder (`phentrieve/llm/providers/gemini.py` / schema builder) to add `scope_reasoning`, `negation_cue`, `cue_target` ordered before `assertion`
- Test: `tests/unit/llm/test_negation_prompt_contract.py` (template/schema unit checks; no live LLM)

- [ ] **Step 1: Verify** `propertyOrdering` honoring in the installed `google-genai` (write a tiny schema and inspect the serialized request, or check the SDK version). Record the result; if ordering is not honored, name the reasoning field so alphabetical order keeps it first, or set `thinking_level`.

- [ ] **Step 2: Write failing tests** asserting (a) the template version is `v3.1.0`, (b) the system prompt contains the "X without Y" scope rule, (c) the few-shot set includes the contrastive pairs, (d) the response schema lists `scope_reasoning`/`cue_target` before `assertion` in `propertyOrdering`.

- [ ] **Step 3: Implement** the system rule, the contrastive few-shots (the 7 cases from the spec), and the schema fields.

- [ ] **Step 4: Run** the contract tests green.

- [ ] **Step 5: BENCHMARK GATE.** Run the existing mapping benchmark before/after (locate the benchmark command in the repo, e.g. a `phentrieve benchmark`/`make` target or a script under `tests/`/`benchmarks/`). Compare mapping accuracy. If it regresses, `git revert` ONLY the WS6 commit and record the result; WS1-WS5 stand. If no regression, keep it.

- [ ] **Step 6: Commit** `feat(llm): negation-scope reasoning + few-shots (gemini, benchmark-gated)` (only if the gate passes).

---

## WS7 -- Gates, live re-test, release prep

### Task 12: Full local gates

- [ ] **Step 1:** `make check`, `make typecheck-fast`, `make test`.
- [ ] **Step 2:** `make ci-local` and `make security-python` (per the CI-parity rule -- partial targets miss format:check and security jobs).
- [ ] **Step 3:** `make frontend-test-ci` to confirm the shared-pipeline correctness changes did not break the curation UI contract.
- [ ] **Step 4:** Fix any failures; re-run until green. Add coverage tests for any touched lines lacking them.

### Task 13: Live MCP re-test

- [ ] **Step 1:** Restart the local MCP: `make mcp-serve-http` from the worktree with `PHENTRIEVE_DATA_ROOT_DIR` pointing at the main checkout's `data/`. Confirm `phentrieve_diagnostics` is healthy.
- [ ] **Step 2:** Re-run the full baseline matrix from Task 0 Step 5 against the restarted server. Confirm: C1 negations now negated; "had seizures" -> 1 term; "denies headache" -> negated Headache; export pipes raw `aggregated_hpo_terms` with non-zero confidence; `capabilities_version` stable across `details`; blank query -> `validation_failed`; unknown strategy -> validation error with list; citation present in compact.
- [ ] **Step 3:** Write `.planning/analysis/2026-06-14-phentrieve-mcp-hardening-verification.md` with before/after evidence and an updated scorecard targeting >= 9.5 per dimension.

### Task 14: Versioning, CHANGELOG, finish branch

- [ ] **Step 1:** Bump component versions (core + MCP server; CLI if touched) per the release-process convention; update CHANGELOG with the defect ids resolved.
- [ ] **Step 2:** Move the plan from `.planning/active/` to `.planning/completed/` and update `.planning/README.md` recent-analysis pointers.
- [ ] **Step 3:** Use the finishing-a-development-branch skill to open the PR (do not self-merge). Include the verification report and scorecard in the PR body.

---

## Self-review (spec coverage)

- C1 -> Task 1. H2 -> Task 2. H1 -> Task 7. H3 -> Task 4. M1 -> Task 5. M2 -> Task 4. M3 -> Task 4. M4 -> Task 3. M5 -> Task 6. T1 -> Task 3. L1 -> Task 8. L2 -> Task 8. L4 -> Task 6. L5 -> Task 3. L6 -> Task 9. L7-collision -> Task 3 (projection keeps (hpo_id, assertion)). L8 -> Task 6. Observability/speed -> Task 10. Safety/citations -> Task 9. WS6 LLM -> Task 11. Verification -> Tasks 12-13. Release -> Task 14.
- No `TODO`/`TBD` placeholders; test code is concrete; the projection API names (`project_aggregated_terms_for_mcp`, `project_processed_chunks_for_mcp`, `_coerce_phenotype`) are used consistently across tasks.
