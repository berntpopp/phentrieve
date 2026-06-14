# Phentrieve MCP Assessment Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the assessment defects PR #288 left open or punted -- deterministic over-negation (D1), co-finding loss + chunk hygiene (D2/D3), payload shape (D4/D5/D7/D13), low-confidence signalling (D6), and polish (D8/D9/D11) -- to reach >= 9.5/10 on every dimension the assessment scored.

**Architecture:** Three layers. Correctness (D1/D2/D3) goes in the shared `phentrieve/` pipeline so REST + the Vue frontend benefit. Output shape + signalling (D4/D5/D6/D7/D11/D13) goes at the `api/mcp/` boundary so the REST schema and curation UI are untouched. Startup warmup (D9) goes in the MCP server lifespan. LLM-D1 is deferred (regressed the Gemini benchmark; documented limitation).

**Tech Stack:** Python 3.11+, FastMCP v3, FastAPI, Pydantic v2, pytest (+xdist), Ruff, mypy, uv. Vue 3 frontend (verification only).

**Source spec:** `.planning/specs/2026-06-14-phentrieve-mcp-assessment-remediation-design.md`

---

## Conventions for every task

- TDD: write the failing test, run it (confirm the *expected* failure), implement, run green, commit.
- Single-threaded runs for clarity: `uv run pytest <path>::<test> -n 0 -q`.
- Tests live under `tests/` only. Never create `tests_new/` or a `tests/unit/api/mcp/` package dir without an `__init__.py`.
- Conventional commits; end every commit body with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Do NOT push until the final gate task. Worktree + branch already exist (`worktree-mcp-remediation`).

## File map (what changes, and why)

| File | Responsibility | Tasks |
|---|---|---|
| `phentrieve/text_processing/assertion_detection.py` | surface negated/normal scope **spans** (offsets), not just text | T1 |
| `phentrieve/text_processing/pipeline.py` | translate scope spans into the chunk-text frame; attach to chunk | T1 |
| `phentrieve/text_processing/_hpo_extraction_helpers.py` | per-match span-overlap assertion (replace blanket chunk status) | T1 |
| `phentrieve/text_processing/chunkers.py` | split multi-concept spans (D2); drop degenerate chunks (D3) | T2, T3 |
| `api/mcp/tools/phenopacket.py` | add native `phenopacket` object (additive) | T4 |
| `api/mcp/tools/similarity.py` + `api/mcp/service_adapters.py` | compare returns MICA/IC/labels/path; honour response_mode | T5 |
| `api/mcp/projection.py` | guarantee `text_attributions` key; drop null padding | T6 |
| `api/mcp/tools/retrieval.py` | `processed_chunks` opt-in; confidence band/flag wiring | T6, T8 |
| `api/mcp/service_adapters.py` (export) | mark client-supplied provenance | T7 |
| `api/mcp/shaping.py` / new `confidence.py` | confidence-band helper | T8 |
| `api/mcp/capabilities.py` | cache-key contract note; band + strategy docs | T8, T9 |
| `api/mcp/server.py` / `api/mcp/facade.py` | best-effort startup warmup | T10 |

---

## WS-A -- Deterministic correctness (shared pipeline)

### Task 1: D1 -- span-level negation (CRITICAL)

**Files:**
- Modify: `phentrieve/text_processing/assertion_detection.py` (`_detect_with_context_rules` ~443-599; `KeywordAssertionDetector.detect` 347-371; `CombinedAssertionDetector` 1098-1214)
- Modify: `phentrieve/text_processing/pipeline.py` (~247-294, the C1 region)
- Modify: `phentrieve/text_processing/_hpo_extraction_helpers.py` (`process_chunk_matches` 23-92, `build_evidence_map` 143-176, `aggregate_and_rank` 179-240)
- Test: `tests/unit/text_processing/test_negation_span_overlap.py` (create)

- [ ] **Step 1: Read** the three call sites above plus `text_attribution.py:18-112` (attribution spans are chunk-relative `start_char`/`end_char`). Confirm exactly where `_detect_with_context_rules` computes `cue_index`/`scope_start`/`scope_end` so they can be returned as spans.

- [ ] **Step 2: Write the failing detector-span test:**

```python
# tests/unit/text_processing/test_negation_span_overlap.py
from phentrieve.text_processing.assertion_detection import create_assertion_detector

def test_detector_returns_negated_scope_spans_for_without():
    det = create_assertion_detector(language="en")
    status, details = det.detect("severe intellectual disability without regression")
    spans = details.get("negated_scope_spans") or []
    assert spans, "detector must expose negated scope spans"
    text = "severe intellectual disability without regression"
    # every negated span must cover 'regression', none may cover 'intellectual'
    assert all(text.find("regression") < e and s <= text.find("regression") + len("regression")
               for s, e in spans)
    assert all(not (s <= text.find("intellectual") < e) for s, e in spans)
```

(If `create_assertion_detector` is not the factory, use the real one found in Step 1; the contract is the behaviour.)

- [ ] **Step 3: Run** `uv run pytest tests/unit/text_processing/test_negation_span_overlap.py -n 0 -q`. Expected: FAIL (`negated_scope_spans` absent).

- [ ] **Step 4: Implement scope spans.** In `_extract_scope` (or its caller `_detect_with_context_rules`), additionally return the `(scope_start, scope_end)` char span(s) it already computes. Thread them up so `KeywordAssertionDetector.detect` adds `assertion_details["negated_scope_spans"]` (and `"normal_scope_spans"`) as lists of `(start, end)` int tuples in the **detection-text frame**, alongside the existing `keyword_negated_scopes` text. `CombinedAssertionDetector` merges/forwards these in `combined_details`.

- [ ] **Step 5: Run** Step 2's test green.

- [ ] **Step 6: Write the failing pipeline+orchestrator integration test:**

```python
# append to the same file
from phentrieve.text_processing.hpo_extraction_orchestrator import orchestrate_hpo_extraction

class _FakeRetriever:
    """Returns a fixed HPO match per query text containing a keyword."""
    def __init__(self, mapping): self.mapping = mapping
    def query(self, texts, n_results, **_):
        out = []
        for t in texts:
            hit = next(((hid, name) for kw, (hid, name) in self.mapping.items()
                        if kw in t.lower()), None)
            if hit:
                out.append({"metadatas": [[{"id": hit[0], "label": hit[1]}]],
                            "similarities": [[0.95]]})
            else:
                out.append({"metadatas": [[]], "similarities": [[]]})
        return out

def test_without_does_not_negate_the_head_concept():
    # one chunk holding both concepts; ID must stay affirmed, regression negated
    chunks = ["severe intellectual disability without regression"]
    retr = _FakeRetriever({
        "intellectual disability": ("HP:0010864", "Severe intellectual disability"),
    })
    result = orchestrate_hpo_extraction(
        text_chunks=chunks,
        all_query_results=retr.query(chunks, n_results=5),
        num_results_per_chunk=3,
        chunk_retrieval_threshold=0.0,
        top_term_per_chunk=False,
        assertion_statuses=["negated"],            # chunk-level says negated (the bug input)
        chunk_negated_spans=[[(31, 41)]],          # 'regression' span (chunk-relative)
        chunk_attribution_text=chunks,             # detect attributions against chunk text
    )
    agg = result["aggregated_hpo_terms"] if isinstance(result, dict) else result.aggregated_results
    id_term = next(t for t in agg if t["id"] == "HP:0010864")
    assert id_term["assertion_status"] == "affirmed"
```

(Match the real `orchestrate_hpo_extraction` signature from `hpo_extraction_orchestrator.py:22-124`; reuse fixtures from `tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py`. The new `chunk_negated_spans` arg name is defined here and MUST be used identically in implementation.)

- [ ] **Step 7: Run** `-n 0 -q`. Expected: FAIL (ID marked negated).

- [ ] **Step 8: Implement frame translation (pipeline).** In `pipeline.py` after `detect(assertion_input)`, read `assertion_details["negated_scope_spans"]` and translate each `(s, e)` from the `assertion_input` frame to the chunk-text frame:
  - If `assertion_input` was the restored context: `chunk_s = assertion_context_start + s - start_char`, same for `e`; clamp to `[0, len(cleaned_final_chunk)]` (a leading cue in the prefix clamps to 0 -> still scopes the chunk = C1 preserved).
  - If `assertion_input` was the cleaned chunk: spans are already chunk-relative.
  Attach `chunk["negated_spans"]` (and `normal_spans`) to the chunk record carried into extraction.

- [ ] **Step 9: Implement per-match overlap.** Replace the blanket propagation in `process_chunk_matches` (`_hpo_extraction_helpers.py:66-71`). Compute each match's status in `build_evidence_map` where `get_text_attributions` runs: a match is `negated` iff any of its attribution spans overlaps any `chunk["negated_spans"]` entry (`a_s < n_e and n_s < a_e`); `normal` on normal-span overlap; else `affirmed`. When no attribution span is found, fall back to the chunk-level `assertion_status` (preserves C1 for cue-stripped single-concept chunks). `aggregate_and_rank` votes on the per-match status.

- [ ] **Step 10: Run** Step 6 test green; run the carried-over C1 cases to prove no regression:

```python
import pytest
from phentrieve.text_processing.pipeline import TextProcessingPipeline

@pytest.mark.parametrize("text,kw", [
    ("There is no nystagmus.", "nystagmus"),
    ("She does not have ataxia.", "ataxia"),
    ("The patient denies headache.", "headache"),
])
def test_c1_negation_still_detected(text, kw):
    pipe = TextProcessingPipeline(language="en")
    chunks = pipe.process(text, include_positions=True)
    neg = [c for c in chunks if kw in c["text"].lower()]
    assert neg and all(c.get("assertion_status") == "negated" for c in neg)
```

- [ ] **Step 11: Run** `uv run pytest tests/unit/text_processing/ -n 0 -q` for regressions.

- [ ] **Step 12: Commit** `fix(extract): negate only matches overlapping a negated scope span (D1)`.

### Task 2: D2 -- split multi-concept chunks

**Files:**
- Modify: `phentrieve/text_processing/chunkers.py` (`ConjunctionChunker` ~623-839, or the punctuation/transition splitter feeding the default `sliding_window_punct_conj_cleaned`)
- Test: `tests/unit/text_processing/test_multi_concept_split.py` (create)

- [ ] **Step 1: Read** `chunkers.py:623-839` and `phentrieve/config.py:294-315` (the default strategy chain) to find where intra-clause splitting happens.

- [ ] **Step 2: Write the failing test:**

```python
# tests/unit/text_processing/test_multi_concept_split.py
from phentrieve.text_processing.pipeline import TextProcessingPipeline

def test_progressing_to_splits_into_two_concepts():
    pipe = TextProcessingPipeline(language="en")
    text = "floppy infant with initial hypotonia progressing to hypertonia of the extremities"
    chunks = [c["text"].lower() for c in pipe.process(text)]
    assert any("hypotonia" in c and "hypertonia" not in c for c in chunks)
    assert any("hypertonia" in c and "hypotonia" not in c for c in chunks)
```

- [ ] **Step 3: Run** `-n 0 -q`. Expected: FAIL (single chunk holds both).

- [ ] **Step 4: Implement.** Add transition markers ("progressing to", "with", "without", "->" / "→") as split points in the conjunction/transition splitter for the default strategy, so each clause carrying a distinct concept becomes its own chunk. Keep existing conjunction behaviour; add a `_TRANSITION_MARKERS` tuple referenced by the splitter.

- [ ] **Step 5: Run** Step 2 test green; run `tests/unit/text_processing/ -n 0 -q` for regressions (some existing chunk-count assertions may need updating -- update them to the new, more correct segmentation, not the old).

- [ ] **Step 6: Commit** `fix(chunking): split multi-concept spans so co-findings surface (D2)`.

### Task 3: D3 -- drop degenerate chunks

**Files:**
- Modify: `phentrieve/text_processing/chunkers.py` (`FinalChunkCleaner` ~86-400)
- Test: `tests/unit/text_processing/test_degenerate_chunk_drop.py` (create)

- [ ] **Step 1: Write the failing test:**

```python
# tests/unit/text_processing/test_degenerate_chunk_drop.py
from phentrieve.text_processing.chunkers import FinalChunkCleaner

def test_function_word_only_chunks_dropped():
    cleaner = FinalChunkCleaner(language="en")   # use the real constructor args
    out = cleaner.chunk(["due", "the patient", "seizures", "walk"])
    texts = [c.strip().lower() for c in out]
    assert "due" not in texts
    assert "seizures" in texts            # real phenotype kept
```

(Confirm `FinalChunkCleaner`'s real method name/signature in Step 0; adapt the call, keep the assertion.)

- [ ] **Step 2: Run** `-n 0 -q`. Expected: FAIL ("due" retained).

- [ ] **Step 3: Implement.** In `FinalChunkCleaner`, after cleaning, drop a chunk whose remaining tokens are all stopwords/function words, or that is a single token shorter than a small min length and not an HPO-significant word. Use the existing stopword resources if present; otherwise a minimal English function-word set guarded by language.

- [ ] **Step 4: Run** green; run `tests/unit/text_processing/ -n 0 -q`.

- [ ] **Step 5: Commit** `fix(chunking): drop degenerate function-word-only chunks (D3)`.

---

## WS-B -- Output shape & schema (MCP boundary)

### Task 4: D4 -- native phenopacket object (additive)

**Files:**
- Modify: `api/mcp/tools/phenopacket.py` (~58-94)
- Test: `tests/unit/api/mcp/test_phenopacket_native_object.py` (create; ensure `tests/unit/api/mcp/__init__.py` exists)

- [ ] **Step 1: Write the failing test:**

```python
# tests/unit/api/mcp/test_phenopacket_native_object.py
import json
from api.mcp.service_adapters import export_phenopacket_service

def test_export_returns_native_phenopacket_object():
    res = export_phenopacket_service(
        case_id="C1",
        phenotypes=[{"hpo_id": "HP:0001250", "label": "Seizure", "assertion": "affirmed"}],
    )
    assert isinstance(res.get("phenopacket"), dict)           # native object present
    assert res["phenopacket"].get("id")
    # string kept for back-compat and must be equivalent
    assert json.loads(res["phenopacket_json"]) == res["phenopacket"]
```

- [ ] **Step 2: Run** `-n 0 -q`. Expected: FAIL (no `phenopacket` key).

- [ ] **Step 3: Implement.** In the export service/tool, after the bundle is built, set `result["phenopacket"] = json.loads(result["phenopacket_json"])`. Keep `phenopacket_json`. Update the tool description to state both fields and that `phenopacket` is the canonical machine-readable object.

- [ ] **Step 4: Run** green; run `tests/unit/api/mcp/ -n 0 -q`.

- [ ] **Step 5: Commit** `feat(mcp): export returns native phenopacket object, string kept for compat (D4)`.

### Task 5: D5 -- compare honours response_mode (MICA/IC/labels/path)

**Files:**
- Modify: `api/mcp/service_adapters.py` (`compare_hpo_terms_service` ~254-275)
- Modify: `api/mcp/tools/similarity.py` (~37-67) + `api/mcp/schemas.py` `COMPARE_SCHEMA`
- Reuse: `phentrieve/evaluation/metrics.py` `find_lowest_common_ancestor` (returns `(lca, lca_depth)`), `load_hpo_graph_data` (depths)
- Test: `tests/unit/api/mcp/test_compare_response_mode.py` (create)

- [ ] **Step 1: Write the failing test:**

```python
# tests/unit/api/mcp/test_compare_response_mode.py
from api.mcp.service_adapters import compare_hpo_terms_service

def test_standard_mode_returns_mica_and_ic():
    res = compare_hpo_terms_service(term1_id="HP:0000787", term2_id="HP:0004724",
                                    formula="hybrid", response_mode="standard")
    assert "mica" in res and res["mica"].get("hpo_id")
    assert "lca_depth" in res and "term1_depth" in res and "term2_depth" in res

def test_compact_mode_stays_lean():
    res = compare_hpo_terms_service(term1_id="HP:0000787", term2_id="HP:0004724",
                                    formula="hybrid", response_mode="compact")
    assert "mica" not in res
    assert set(res) >= {"term1_id", "term2_id", "formula_used", "similarity_score"}
```

(Confirm the real `compare_hpo_terms_service` signature; if `response_mode` is applied in the tool layer not the service, drive the test through the tool's call path instead. Keep the field assertions.)

- [ ] **Step 2: Run** `-n 0 -q`. Expected: FAIL.

- [ ] **Step 3: Implement.** Compute the enriched block once: `lca, lca_depth = find_lowest_common_ancestor(t1, t2)`; `depths = load_hpo_graph_data()[1]`; `term1_depth`, `term2_depth`; `mica = {"hpo_id": lca, "label": <HPODatabase label lookup>}`; `term*_ic = depth/max_depth`; `path_length = (d1 - lca_depth) + (d2 - lca_depth)`. Return these only when the resolved mode is `standard`/`full`; keep the 4-field payload at `minimal`/`compact`. Wire `apply_response_mode` (or an explicit mode gate) so the keys are actually dropped at compact. Add the enriched keys to `COMPARE_SCHEMA`.

- [ ] **Step 4: Run** green; run `tests/unit/api/mcp/ -n 0 -q`.

- [ ] **Step 5: Commit** `feat(mcp): compare returns MICA/IC/labels/path and honours response_mode (D5)`.

### Task 6: D7 residual trim + D13 attribution consistency

**Files:**
- Modify: `api/mcp/projection.py` (`project_aggregated_terms_for_mcp` 44-73)
- Modify: `api/mcp/tools/retrieval.py` (extract tools: `processed_chunks` opt-in already via `include_unmatched_chunks`; confirm `full` mode does not duplicate)
- Test: `tests/unit/api/mcp/test_projection_attribution.py` (create)

- [ ] **Step 1: Write the failing test:**

```python
# tests/unit/api/mcp/test_projection_attribution.py
from api.mcp.projection import project_aggregated_terms_for_mcp

def test_text_attributions_key_always_present():
    out = project_aggregated_terms_for_mcp([
        {"id": "HP:1", "name": "A", "score": 0.9, "text_attributions": [{"start_char": 0, "end_char": 1}]},
        {"id": "HP:2", "name": "B", "score": 0.8},   # no attributions
    ])
    assert all("text_attributions" in t for t in out)
    assert out[1]["text_attributions"] == []

def test_null_padding_dropped():
    out = project_aggregated_terms_for_mcp([
        {"id": "HP:1", "name": "A", "score": 0.9, "start_char": None, "end_char": None},
    ])[0]
    assert "start_char" not in out and "end_char" not in out
```

- [ ] **Step 2: Run** `-n 0 -q`. Expected: FAIL.

- [ ] **Step 3: Implement.** In `project_aggregated_terms_for_mcp`: `out.setdefault("text_attributions", [])`; and drop keys whose value is `None` (after normalization) for the known optional set (`start_char`, `end_char`, plus any `*_count` that is 0 by default and only meaningful in `full`). Confirm in `retrieval.py` that `processed_chunks` is only emitted with matched evidence unless `include_unmatched_chunks=True` (the projection already drops empty-match chunks -- assert the extract tool passes `include_unmatched` through and does not also echo aggregated data at `full`).

- [ ] **Step 4: Run** green; run `tests/unit/api/mcp/ -n 0 -q`.

- [ ] **Step 5: Commit** `fix(mcp): consistent text_attributions + drop null padding in projection (D7/D13)`.

### Task 7: D11 -- honest client-supplied provenance

**Files:**
- Modify: `api/mcp/service_adapters.py` (`export_phenopacket_service` ~313-355, `_coerce_export_phenotype`)
- Test: `tests/unit/api/mcp/test_export_provenance.py` (create)

- [ ] **Step 1: Write the failing test:**

```python
# tests/unit/api/mcp/test_export_provenance.py
import json
from api.mcp.service_adapters import export_phenopacket_service

def test_client_supplied_phenotype_marked_provenance():
    res = export_phenopacket_service(
        case_id="C1",
        phenotypes=[{"hpo_id": "HP:0001250", "label": "Seizure", "assertion": "affirmed"}],
    )
    blob = json.dumps(res)
    assert "legacy_dict" not in blob
    assert "client_supplied" in blob or "unknown" in blob
```

- [ ] **Step 2: Run** `-n 0 -q`. Expected: FAIL (`legacy_dict` present).

- [ ] **Step 3: Implement.** When a phenotype is supplied directly (no extractor `source_mode`/`match_method`), set `match_method = "client_supplied"` and `source_mode = "unknown"` (or the schema's nearest fields) instead of defaulting to `legacy_dict`/`chunk`. Locate the default in the export models / `_coerce_export_phenotype`.

- [ ] **Step 4: Run** green; run `tests/unit/api/mcp/ -n 0 -q` and `tests/unit/api/ -n 0 -q`.

- [ ] **Step 5: Commit** `fix(mcp): mark client-supplied phenotype provenance instead of legacy_dict (D11)`.

---

## WS-C -- Confidence & discoverability (MCP boundary)

### Task 8: D6 -- low-confidence signal

**Files:**
- Create: `api/mcp/confidence.py` (band helper)
- Modify: `api/mcp/tools/retrieval.py` (search: annotate results + top-level flag; extract: annotate chunk matches)
- Modify: `api/mcp/capabilities.py` (document band semantics)
- Test: `tests/unit/api/mcp/test_confidence_band.py` (create)

- [ ] **Step 1: Write the failing test:**

```python
# tests/unit/api/mcp/test_confidence_band.py
from api.mcp.confidence import confidence_band, annotate_search_confidence

def test_band_thresholds():
    assert confidence_band(0.85) == "high"
    assert confidence_band(0.5) == "moderate"
    assert confidence_band(0.2) == "low"

def test_no_high_confidence_flag_when_top_below_floor():
    payload = {"results": [{"hpo_id": "HP:1", "similarity": 0.59},
                           {"hpo_id": "HP:2", "similarity": 0.48}]}
    out = annotate_search_confidence(payload, score_key="similarity")
    assert out["no_high_confidence_match"] is True
    assert out["results"][0]["confidence_band"] == "low"

def test_no_flag_when_strong_top_hit():
    payload = {"results": [{"hpo_id": "HP:1", "similarity": 0.82}]}
    out = annotate_search_confidence(payload, score_key="similarity")
    assert out.get("no_high_confidence_match") in (False, None)
    assert out["results"][0]["confidence_band"] == "high"
```

- [ ] **Step 2: Run** `-n 0 -q`. Expected: ImportError.

- [ ] **Step 3: Implement** `api/mcp/confidence.py`:

```python
"""Low-confidence signalling for retrieval results (assessment D6)."""
from __future__ import annotations
from typing import Any

HIGH_FLOOR = 0.7
MODERATE_FLOOR = 0.4

def confidence_band(score: float) -> str:
    if score >= HIGH_FLOOR:
        return "high"
    if score >= MODERATE_FLOOR:
        return "moderate"
    return "low"

def annotate_search_confidence(payload: dict[str, Any], *, score_key: str = "similarity") -> dict[str, Any]:
    results = payload.get("results") or []
    top = max((r.get(score_key, 0.0) for r in results), default=0.0)
    for r in results:
        r["confidence_band"] = confidence_band(r.get(score_key, 0.0))
    out = dict(payload)
    out["no_high_confidence_match"] = bool(results) and top < HIGH_FLOOR
    return out
```

- [ ] **Step 4: Run** green.

- [ ] **Step 5: Wire in.** In `search_hpo_terms` (`retrieval.py`), after shaping, call `annotate_search_confidence` (use the real score key from a search result item; inspect `execute_hpo_retrieval_for_api` output -- likely `similarity` or `score`). For extract, annotate each chunk match's `confidence_band`. Do NOT change the default `similarity_threshold`.

- [ ] **Step 6:** Document the band semantics + floor in the search tool description and capabilities body.

- [ ] **Step 7: Run** `tests/unit/api/mcp/ -n 0 -q`; **Commit** `feat(mcp): low-confidence band + no_high_confidence_match signal (D6)`.

### Task 9: D8 -- cache-key contract note

**Files:**
- Modify: `api/mcp/capabilities.py` (descriptor body)
- Test: extend `tests/unit/api/mcp/test_capabilities_version.py` (exists from PR #288) or create `test_capabilities_contract.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_capabilities_documents_cache_key_contract():
    from api.mcp.capabilities import build_capabilities
    blob = str(build_capabilities())
    assert "capabilities_version" in blob
    assert "descriptor_hash" in blob
    # a one-line contract is present (canonical key vs per-descriptor hash)
    assert "warm" in blob.lower() or "canonical" in blob.lower()
```

- [ ] **Step 2: Run** `-n 0 -q`. Expected: FAIL.

- [ ] **Step 3: Implement.** Add a `cache_contract` note string to the capabilities body: "`capabilities_version` is the canonical warm-cache key (echoed in every `_meta`, stable across `details`); `descriptor_hash` is the per-descriptor content hash of the expanded view; `tools/list_changed` is the MCP change signal."

- [ ] **Step 4: Run** green; **Commit** `docs(mcp): document capabilities cache-key contract (D8)`.

---

## WS-D -- Speed & polish

### Task 10: D9 -- best-effort startup warmup

**Files:**
- Modify: `api/mcp/server.py` and/or `api/mcp/facade.py` (lifespan)
- Test: `tests/unit/api/mcp/test_startup_warmup.py` (create)

- [ ] **Step 1: Read** `api/mcp/server.py` / `facade.py` to find the FastMCP lifespan/startup hook and the embedding-model + index loader (`get_dense_retriever_dependency`).

- [ ] **Step 2: Write the failing test** (warmup is best-effort and never raises):

```python
# tests/unit/api/mcp/test_startup_warmup.py
import asyncio
from api.mcp import facade  # adjust to where warmup() lives

def test_warmup_is_best_effort(monkeypatch):
    called = {"n": 0}
    async def boom(*a, **k):
        called["n"] += 1
        raise RuntimeError("model load failed")
    monkeypatch.setattr("api.mcp.facade._warmup_retriever", boom, raising=False)
    # warmup() must swallow the error and still return
    asyncio.get_event_loop().run_until_complete(facade.warmup())
    assert called["n"] >= 1
```

(Adjust names to the real warmup entry point defined in Step 3; the contract is "best-effort, never raises".)

- [ ] **Step 3: Implement.** Add an async `warmup()` that loads the embedding model + vector index via the existing dependency, wrapped in try/except with a single `logger.info`/`logger.warning`. Call it from the MCP server lifespan startup (non-blocking; never fail startup). Document the warmup in the latency profile / capabilities.

- [ ] **Step 4: Run** green; **Commit** `feat(mcp): best-effort startup warmup for embedding model + index (D9)`.

---

## WS-E -- Gates, live re-test, release

### Task 11: Full local + frontend gates

- [ ] **Step 1:** `make check`; `make typecheck-fast`; `make test`.
- [ ] **Step 2:** `make ci-local` and `make security-python` (CI-parity rule: partial targets miss format:check + security).
- [ ] **Step 3:** `make frontend-test-ci` (shared-pipeline correctness changes must not break the curation UI contract).
- [ ] **Step 4:** Fix failures; re-run until green. Add coverage tests for any touched lines lacking them.

### Task 12: Live MCP re-test + verification report

- [ ] **Step 1:** Start the worktree MCP: `make mcp-serve-http` with `PHENTRIEVE_DATA_ROOT_DIR` pointing at the symlinked `data/`. Confirm `phentrieve_diagnostics` healthy and warm.
- [ ] **Step 2:** Re-run the assessment matrix and capture raw responses: D1 fixtures (ID affirmed / regression negated; "does not walk or speak"; "respiratory arrest under propofol sedation"); D2 (hypotonia + hypertonia both); D4 (native object); D5 (MICA/IC at standard, lean at compact); D6 (gibberish -> `no_high_confidence_match`); D7/D13 (no triplication/null padding, attributions on every term); D8 (contract note); D9 (diagnostics fast); D11 (client_supplied). Plus C1 regression (no nystagmus / denies headache stay negated).
- [ ] **Step 3:** Write `.planning/analysis/2026-06-14-phentrieve-mcp-assessment-remediation-verification.md` with before/after evidence and an updated scorecard targeting >= 9.5 per dimension.

### Task 13: Versioning, CHANGELOG, finish branch

- [ ] **Step 1:** Bump component versions (core + MCP server; CLI if touched) per the release-process convention; update CHANGELOG `[Unreleased]` with the defect ids resolved (D1, D2, D3, D4, D5, D6, D7, D8, D9, D11, D13).
- [ ] **Step 2:** Move this plan `.planning/active/` -> `.planning/completed/`; update `.planning/README.md` pointers.
- [ ] **Step 3:** Use the finishing-a-development-branch skill to open the PR (do not self-merge). Include the verification report + scorecard in the PR body; note LLM-D1 deferred.

---

## Self-review (spec coverage)

- D1 -> T1. D2 -> T2. D3 -> T3. D4 -> T4. D5 -> T5. D7 -> T6. D13 -> T6. D11 -> T7. D6 -> T8. D8 -> T9. D9 -> T10. D12 -> verify in T12 (already fixed). LLM-D1 -> deferred (documented).
- No `TODO`/`TBD`; test code is concrete; new identifiers (`negated_scope_spans`, `chunk_negated_spans`, `negated_spans`, `confidence_band`, `annotate_search_confidence`, `warmup`) are used consistently across tasks and match the spec.
- Gates (T11) match the CI-parity memory; blast-radius frontend check included.
