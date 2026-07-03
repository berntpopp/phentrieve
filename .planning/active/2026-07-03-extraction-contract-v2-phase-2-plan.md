# Extraction Contract v2 -- Phase 2 Implementation Plan (LLM contract v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Rev 3** -- rev 1 -> rev 2 fixed a Task-2 validation bug, the vocabulary strategy, the `_retrieve_candidates` axis-threading, the B3 emit point, the `capabilities_version`/MCP-projection mechanisms, and the frontend/prompt paths (self-review + Codex round 1). Rev 2 -> rev 3 (Codex round 2) made the compat-`category` concrete (D3 helper + conflict tests), added a non-breaking `excluded` flag for extract-status consistency, threaded family through the real `adapt_full_text_response`, added family-mapping cost accounting, required generated exclusions to inherit valid chunk refs (or they drop at the service boundary), de-brittled the capabilities test, and fixed the Vitest watch-mode command. See "Review deltas" at the end.

**Goal:** Make the LLM extraction output fully correct and consistent across MCP/REST/Vue -- assertion becomes load-bearing, family-history surfaces as its own list instead of being silently dropped, and "X without Y" becomes a machine-actionable excluded term -- all guarded by one canonical export vocabulary so `absent` can never silently export as present.

**Architecture:** Four atomic, individually-gated behavior changes in dependency order, all converging on ONE explicit pipeline data-flow (see "Data-flow contract" below). **B0** adds a shared canonical **export** vocabulary helper (`affirmed / negated / normal / uncertain`) applied at the export boundaries (MCP/REST/Vue), replacing ad-hoc `== "negated"` checks. **B1** makes the model's own `experiencer` + `assertion` load-bearing by threading them through the entire parse -> Phase-1 dedup -> `_retrieve_candidates` -> mapping payload -> `phenotype_from_candidate` chain (the legacy `category` becomes a derived compat field). **B2** partitions family-experiencer phrases before the actionable filter, resolves them through the same retrieval+mapping path, and emits a dedicated `family_history_findings` list kept out of the proband phenopacket. **B3** maps a resolved finding's `negated_qualifier` phrase to an excluded HPO term (confidence-gated, string fallback), generated after proband resolution and before final dedup. New fields are threaded through the shared service + REST/MCP data schemas here; the Vue *display* is Phase 3.

**Tech Stack:** Python 3.11+, `uv`, Ruff, mypy, pytest (+ pytest-xdist); Pydantic v2; FastAPI; FastMCP; Vue 3 + Vitest (B0 export mirror only).

**Source spec:** `.planning/specs/2026-07-03-extraction-contract-v2-and-finalization-design.md` (section 7 = B0-B3 + carrier surfaces; decisions locked in section 3). Anchor line numbers below are verified against `feat/extraction-contract-v2` HEAD.

## Scope boundary (read first)

- **This plan (Phase 2):** B0 canonical export vocabulary + boundary hardening (MCP export + Vue *export* coercion), B1 assertion load-bearing, B2 family-history list, B3 qualifier->excluded, and threading the new fields through the shared service + REST/MCP **data** schemas.
- **Deferred to Phase 3:** the Vue **display** components (family-history section, "excluded" chips), i18n locale keys for those, closing issue #289, the final verification doc, and the coordinated CLI/API/Frontend release. B0's Vue *export* mirror IS here (coercion, not display).
- **Environment gate (spec 9.1):** Phase 2's acceptance is three-part (spec 6.4): (1) present-only no-regression on the deterministic corpora, (2) new golden edge cases pass in strict, (3) the **LLM mapping benchmark does not regress**. (1)+(2) run in this checkout via the Phase 1 machinery. (3) needs the **Docker stack + Gemini key** (not this checkout). Task 13 makes running (3) an explicit, owned step. Do NOT claim Phase 2 "gated" without (3); if it cannot run here, land the code with (1)+(2) green and STOP for the operator.

## Global Constraints

- `uv` only; never `pip`. Modern typing (`list[str]`, `str | None`, `dict[str, X]`); mypy targets Python 3.11. Ruff. ASCII only. All tests under `tests/`; never `tests_new/` or new `tests/unit/api/` sub-package dirs.
- Gates per task: `make check`, `make typecheck-fast`, `make test`. Before push: `make ci-local` + `make security-python`. Task 3 (frontend) is covered by `make ci-local`.
- Atomic commits (one behavior = one commit); coverage-improving tests on all touched code; TDD per fix.
- **B0's `canonicalize_assertion`/`is_excluded` are for the EXPORT boundaries only** (MCP/REST/Vue). The LLM pipeline keeps its internal `present/negated/uncertain` vocabulary; do NOT push `affirmed` into the pipeline (see Data-flow contract, decision D2).
- **No proband leakage:** a `family_history` experiencer term must never appear in the subject's `PhenotypicFeature`s (regression test, Task 8).
- **Determinism:** the Phase 1 identity guarantee still holds; the deterministic extraction baselines must not regress under `present-only` (Task 13).

## Vocabulary reality (verified -- THREE vocabularies)

| Layer | Values | Source of truth |
|---|---|---|
| **LLM wire** (what the model emits) | `present / absent / uncertain` | `phentrieve/llm/types.py:102-107,131-136` (`Literal[...]`) |
| **LLM pipeline internal** | `present / negated / uncertain` (+ legacy `family_history`/`other` conflated) | `phentrieve/llm/config.py:50-52` (`PRESENT_ASSERTION="present"` ...); `pipeline_phase2.py:27-33` (`CATEGORY_TO_ASSERTION`) |
| **Export / boundary canonical** | `affirmed / negated / normal / uncertain` | `phentrieve/text_processing/assertion_detection.py:28-36` |

Two helpers already exist and must not be conflated:
- `phentrieve/llm/utils.py:76-88` `parse_assertion(str) -> AssertionStatus` maps `absent/negated/excluded/no/denied -> NEGATED` and targets the **llm/types** `AssertionStatus` enum whose values are `present/negated/uncertain` -- i.e. **exactly the pipeline vocabulary**. It is currently DEAD (only its own unit test calls it). B1 revives it as the wire->pipeline mapper.
- B0's new `canonicalize_assertion` targets the **export canonical** vocabulary and is used only at export boundaries.

## Data-flow contract (READ BEFORE B1-B3)

Locked decisions that every B1-B3 task references:

- **D1 -- one explicit final flow** in `TwoPhaseLLMPipeline.run()` (assembly at `pipeline.py:496-517`):
  1. **Parse** (Phase-1): each item carries `phrase, category, negated_qualifier, evidence/span` AND (new, B1) `experiencer`, `assertion` (raw wire value).
  2. **Partition:** `family_items = [it for it if experiencer=="family_history"]`; `proband_items = the rest`.
  3. **Proband resolve:** actionable filter -> `_retrieve_candidates(proband_items)` (`pipeline.py:1117-1123`) -> `_route_phase2_candidates` (`:1380-1434`) / `_resolve_with_mapping_prompt` (`:1482-1566`) -> `resolved_proband: list[LLMPhenotype]`.
  4. **Family resolve (B2):** `_retrieve_candidates(family_items)` -> same resolution path -> `resolved_family: list[LLMPhenotype]` (kept separate; never merged into proband).
  5. **Qualifier exclusions (B3):** for each resolved proband finding with `negated_qualifier="Y"`, resolve `Y` via one retrieval call (floor = `self.similarity_threshold`); if mapped, build a generated excluded `LLMPhenotype`; else keep today's metadata string.
  6. **Assemble:** `terms = self._deduplicate_terms(resolved_proband + qualifier_exclusions)`; `family_history_findings = self._deduplicate_terms(resolved_family)`.
- **D2 -- B0 is export-only.** Inside the pipeline, assertion stays in the `present/negated/uncertain` vocabulary. B1 makes the model's assertion load-bearing by mapping the wire value via `parse_assertion(...).value` (`absent->negated`, `present/uncertain` unchanged) -- NOT via `canonicalize_assertion`. This deviates from spec B0's "one helper reused by the LLM->pipeline mapping"; the fuller unification (retire `PRESENT_ASSERTION="present"` -> `"affirmed"` across `CATEGORY_TO_ASSERTION`, dedup keys, every `==` comparison, and dozens of tests) is out of proportion to the bug being fixed (which lives at the export boundary) and is recorded as a follow-up. **If you want the full unification instead, stop and re-plan -- it is a materially larger change.**
- **D3 -- compat `category` via an explicit helper.** Where category-dependent logic remains (actionability `pipeline.py:340-344`; Phase-1 dedup key `pipeline_phase1.py:209-217`; mapping payload `pipeline_phase2.py:102-110`), the model's axes must win when they disagree with the legacy enum. Add `derive_category_from_axes(experiencer: str | None, assertion: str | None, fallback_category: str) -> str` (in `pipeline_phase2.py`) and STORE its result as the phenotype's `category` (Task 5), so partition/dedup/mapping see a category consistent with the model's axes while existing category consumers keep working. Partition itself keys on `experiencer` directly (Task 6), so a `family_history` experiencer with a legacy `Abnormal` category still routes to family.

## File structure

| File | Responsibility | Tasks |
|---|---|---|
| `phentrieve/assertion_vocab.py` (new) | Shared export `canonicalize_assertion()` + `is_excluded()` + constants | 1 |
| `api/mcp/service_adapters.py` | MCP export coercion (2-valued via `is_excluded`); experiencer guard | 2, 8 |
| `frontend/src/composables/usePhenotypeCollection.js` | Vue export coercion (B0 mirror) | 3 |
| `phentrieve/llm/pipeline.py` | Parse axes; partition; family resolve; qualifier exclusions; assembly | 4, 6, 7, 10 |
| `phentrieve/llm/pipeline_phase1.py` | Phase-1 dedup carries axes; actionable filter (compat category) | 4 |
| `phentrieve/llm/pipeline_phase2.py` | Mapping payload carries axes; `phenotype_from_candidate` consumes model assertion | 4, 5 |
| `phentrieve/llm/utils.py` | Revive `parse_assertion` as the wire->pipeline mapper | 5 |
| `phentrieve/llm/types.py` | `LLMExtractionResult.family_history_findings`; excluded-finding fields | 7, 10 |
| `phentrieve/text_processing/full_text_service.py` | Shared service: carry experiencer + family + excluded (`_adapt_llm_aggregated_terms:434-455`; `adapt_shared_service_response_to_api`) | 9, 11 |
| `api/schemas/text_processing_schemas.py` | REST response schema fields | 9, 11 |
| `api/services/text_processing_execution.py` | REST response builder (stop dropping fields) | 9, 11 |
| `api/mcp/schemas.py` + `api/mcp/capabilities.py` | MCP output schema + descriptor (rolls `capabilities_version`) | 9 |
| `api/mcp/projection.py`, `api/mcp/shaping.py` | MCP per-term projection + budget for the family list | 9, 11 |

---

## B0 -- Canonical export vocabulary (foundation; land first)

### Task 1: Add the shared `canonicalize_assertion()` helper

**Files:** Create `phentrieve/assertion_vocab.py`; Test `tests/unit/test_assertion_vocab.py`.

**Interfaces:** Produces `canonicalize_assertion(raw: str | None) -> str` in `{affirmed, negated, normal, uncertain}`; `is_excluded(raw: str | None) -> bool`; constants `AFFIRMED/NEGATED/NORMAL/UNCERTAIN`. Consumed by Tasks 2, 8, 9, 11 (export boundaries only).

- [ ] **Step 1: Write the failing test** -- create `tests/unit/test_assertion_vocab.py`:

```python
import pytest

from phentrieve.assertion_vocab import (
    AFFIRMED, NEGATED, NORMAL, UNCERTAIN, canonicalize_assertion, is_excluded,
)


@pytest.mark.parametrize("raw,expected", [
    ("present", AFFIRMED), ("affirmed", AFFIRMED), ("abnormal", AFFIRMED),
    ("absent", NEGATED), ("negated", NEGATED), ("excluded", NEGATED), ("no", NEGATED),
    ("normal", NORMAL), ("uncertain", UNCERTAIN), ("suspected", UNCERTAIN),
    ("  ABSENT  ", NEGATED), (None, AFFIRMED), ("nonsense", AFFIRMED),
])
def test_canonicalize(raw, expected):
    assert canonicalize_assertion(raw) == expected


@pytest.mark.parametrize("raw", ["absent", "negated", "excluded", "NO"])
def test_is_excluded_true(raw):
    assert is_excluded(raw) is True


@pytest.mark.parametrize("raw", ["present", "affirmed", "normal", "uncertain", None])
def test_is_excluded_false(raw):
    assert is_excluded(raw) is False
```

- [ ] **Step 2: Run it and confirm it fails** -- `uv run pytest tests/unit/test_assertion_vocab.py -n 0` -> ModuleNotFoundError.

- [ ] **Step 3: Implement** -- create `phentrieve/assertion_vocab.py`:

```python
"""Canonical EXPORT assertion vocabulary (affirmed / negated / normal / uncertain).

Used only at the export boundaries (MCP/REST/Vue) so an LLM ``assertion="absent"``
can never silently export as an affirmed (present) feature. The LLM pipeline keeps
its own present/negated/uncertain vocabulary -- do not use this inside the pipeline.
"""

from __future__ import annotations

AFFIRMED = "affirmed"
NEGATED = "negated"
NORMAL = "normal"
UNCERTAIN = "uncertain"

_NEGATED = {"negated", "negative", "absent", "excluded", "no", "denied"}
_UNCERTAIN = {"uncertain", "possible", "suspected", "probable"}
_NORMAL = {"normal"}


def canonicalize_assertion(raw: str | None) -> str:
    normalized = str(raw).strip().lower() if raw is not None else ""
    if normalized in _NEGATED:
        return NEGATED
    if normalized in _UNCERTAIN:
        return UNCERTAIN
    if normalized in _NORMAL:
        return NORMAL
    return AFFIRMED


def is_excluded(raw: str | None) -> bool:
    """True iff ``raw`` denotes a ruled-out finding (``excluded: true``)."""
    return canonicalize_assertion(raw) == NEGATED
```

- [ ] **Step 4: Run the test** -- `uv run pytest tests/unit/test_assertion_vocab.py -n 0` -> passed.
- [ ] **Step 5: Commit** -- `git add phentrieve/assertion_vocab.py tests/unit/test_assertion_vocab.py && git commit -m "feat(llm): add canonical export assertion vocabulary helper (B0 foundation)"`

### Task 2: Route MCP phenopacket export through the 2-valued `is_excluded`

**Files:** Modify `api/mcp/service_adapters.py` (`_coerce_export_phenotype`, line 397); Test `tests/unit/mcp_server/test_mcp_service_adapters.py`.

**Interfaces:** Consumes `is_excluded` (Task 1). Produces: `assertion_status` set to `"negated" if is_excluded(assertion) else "affirmed"` -- so `absent -> negated -> excluded: true` (today `absent` silently becomes `affirmed`).

> **Fix vs rev 1:** `ExportPhenotypeRequest.assertion_status` is `Literal["affirmed","negated"]` with `extra="forbid"` (`api/schemas/phenopacket_schemas.py:150`). Passing the 4-valued `canonicalize_assertion(...)` would raise a Pydantic ValidationError for `normal`/`uncertain`. Use the 2-valued `is_excluded`.

- [ ] **Step 1: Write the failing test** -- add to `tests/unit/mcp_server/test_mcp_service_adapters.py` (reuse the file's existing `ExportPhenotypeRequest` import):

```python
def test_absent_assertion_exports_as_negated_not_affirmed():
    out = _coerce_export_phenotype(ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": "absent"}, 0)
    assert out is not None and out.assertion_status == "negated"


def test_normal_and_uncertain_do_not_crash_and_are_not_excluded():
    for a in ("normal", "uncertain"):
        out = _coerce_export_phenotype(ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": a}, 0)
        assert out.assertion_status == "affirmed"  # not excluded, and no ValidationError


def test_present_assertion_affirmed():
    out = _coerce_export_phenotype(ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": "present"}, 0)
    assert out.assertion_status == "affirmed"
```

- [ ] **Step 2: Run it and confirm it fails** -- `uv run pytest tests/unit/mcp_server/test_mcp_service_adapters.py -k assertion -n 0` -> FAIL (`absent` gives `affirmed` today).
- [ ] **Step 3: Implement** -- add `from phentrieve.assertion_vocab import is_excluded` at the top; change line 397 to:

```python
        assertion_status="negated" if is_excluded(assertion) else "affirmed",
```

(The family-history early-return at line 383 stays until Task 8 replaces it with the experiencer guard.)

- [ ] **Step 4: Run tests + gates** -- `uv run pytest tests/unit/mcp_server/test_mcp_service_adapters.py -n 0 && make typecheck-fast` -> passed; clean.
- [ ] **Step 5: Commit** -- `git add api/mcp/service_adapters.py tests/unit/mcp_server/test_mcp_service_adapters.py && git commit -m "fix(mcp): export absent as excluded via is_excluded (B0)"`

### Task 3: Mirror the excluded rule in the Vue export composable

**Files:** Modify `frontend/src/composables/usePhenotypeCollection.js` (line 148); Test **`frontend/src/test/composables/usePhenotypeCollection.test.js`** (extend the existing file, ~lines 77-107 -- there is NO `__tests__` dir).

**Interfaces:** Export sets `excluded` true for `negated` AND `absent` (today only literal `negated`).

- [ ] **Step 1: Write the failing test** -- in `frontend/src/test/composables/usePhenotypeCollection.test.js`, add cases: a collected phenotype with `assertion_status: 'absent'` exports `excluded: true`; `'negated'` -> true; `'present'` -> false. Follow the existing test's setup in that file.
- [ ] **Step 2: Run it and confirm it fails** -- `cd frontend && npm run test:run -- usePhenotypeCollection` -> FAIL for `'absent'`.
- [ ] **Step 3: Implement** -- near the top of `usePhenotypeCollection.js` add:

```js
const EXCLUDED_ASSERTIONS = new Set(['negated', 'absent', 'excluded', 'no', 'denied'])
const isExcluded = (a) => EXCLUDED_ASSERTIONS.has(String(a ?? '').trim().toLowerCase())
```

and change line 148 `excluded: cp.assertion_status === 'negated',` -> `excluded: isExcluded(cp.assertion_status),`.

- [ ] **Step 4: Run test + gate** -- `cd frontend && npm run test:run -- usePhenotypeCollection` then `make frontend-test-ci` -> passed.
- [ ] **Step 5: Commit** -- `git add frontend/src/composables/usePhenotypeCollection.js frontend/src/test/composables/usePhenotypeCollection.test.js && git commit -m "fix(frontend): treat absent as excluded in phenopacket export (B0 mirror)"`

---

## B1 -- Assertion load-bearing (thread BOTH axes through the whole chain)

### Task 4: Carry `experiencer` + `assertion` through parse -> dedup -> retrieve -> mapping payload

**Files:** Modify `phentrieve/llm/pipeline.py` (parse ~582-593; `_retrieve_candidates` result rebuild ~1215-1241); `phentrieve/llm/pipeline_phase1.py` (dedup ~209-217); `phentrieve/llm/pipeline_phase2.py` (mapping payload ~102-110). Test `tests/unit/llm/test_axis_threading.py`.

**Interfaces:** Produces: every item dict emerging from `_retrieve_candidates` carries the model's `experiencer` and `assertion` (raw wire values), consumed by Task 5.

> **Critical (Codex):** adding axes only at the parse is insufficient -- `_retrieve_candidates` (`pipeline.py:1215-1219`) REBUILDS the candidate dicts (`phrase/category/candidates`) and reattaches only `negated_qualifier` + context (`:1230-1241`), dropping the axes again. Thread them through all four sites.

- [ ] **Step 1: Write the failing test** -- create `tests/unit/llm/test_axis_threading.py`. Monkeypatch the retriever/LLM so no model or Gemini is needed (mirror existing `tests/unit/llm/` pipeline-test stubs -- stub BOTH the retriever and the phase-2 LLM mapping call). Drive `_retrieve_candidates` (or the smallest function that rebuilds the candidate dicts) with a parsed item carrying `experiencer="family_history"`, `assertion="absent"`, and assert those two keys survive on the rebuilt candidate dict.
- [ ] **Step 2: Run it and confirm it fails** -- `uv run pytest tests/unit/llm/test_axis_threading.py -n 0` -> FAIL (axes dropped in rebuild).
- [ ] **Step 3: Implement** -- at each site, carry the two keys:
  - Parse (~582-593): add `"experiencer": getattr(phenotype, "experiencer", None)` and `"assertion": getattr(phenotype, "assertion", None)`.
  - Phase-1 dedup (`pipeline_phase1.py:209-217`): ensure the retained deduped item keeps `experiencer`/`assertion`; add `experiencer` to the dedup key so a proband vs family mention of the same phrase does not collapse.
  - `_retrieve_candidates` rebuild (`pipeline.py:1215-1241`): copy `experiencer` and `assertion` onto the rebuilt result dict alongside `negated_qualifier`.
  - Mapping payload (`pipeline_phase2.py:102-110`): include `experiencer`/`assertion` so `phenotype_from_candidate` (Task 5) can read them and the LLM mapping context has them.
- [ ] **Step 4: Run test + full llm suite + gates** -- `uv run pytest tests/unit/llm/ -n 0 && make typecheck-fast` -> passed.
- [ ] **Step 5: Commit** -- `git add phentrieve/llm/pipeline.py phentrieve/llm/pipeline_phase1.py phentrieve/llm/pipeline_phase2.py tests/unit/llm/test_axis_threading.py && git commit -m "feat(llm): thread model experiencer+assertion through parse->retrieve->map (B1)"`

### Task 5: Consume the model's `assertion` in Phase-2; `category` becomes derived-compat

**Files:** Modify `phentrieve/llm/pipeline_phase2.py` (`phenotype_from_candidate` ~255-287; keep dedup key at :342); `phentrieve/llm/utils.py` (revive `parse_assertion` usage). Test `tests/unit/llm/test_phase2_assertion_source.py`.

**Interfaces:** Consumes the candidate dict with `"assertion"`/`"experiencer"` (Task 4); `parse_assertion` (`llm/utils.py:76-88`, wire->pipeline vocab). Produces: `derive_category_from_axes(experiencer, assertion, fallback_category) -> str` (new, D3); `phenotype_from_candidate` sets `assertion = parse_assertion(item["assertion"]).value` (pipeline vocab `present/negated/uncertain`) when the model supplied it, else the category fallback; `experiencer` from the model with `experiencer_for_category` fallback; and STORES `category = derive_category_from_axes(...)` so the stored category reflects the axes, not the raw legacy value. Dedup key `(term_id, experiencer, assertion)` unchanged (`:342`).

> **Note (D2):** use `parse_assertion` (-> `present/negated/uncertain`), NOT B0's `canonicalize_assertion` (-> `affirmed/...`). Injecting `affirmed` here would break `== PRESENT_ASSERTION` comparisons and the dedup key.

- [ ] **Step 1: Write the failing test** -- create `tests/unit/llm/test_phase2_assertion_source.py`: (a) an `item` whose `category` maps to `present` via `CATEGORY_TO_ASSERTION` but whose model `assertion="absent"` -> resulting phenotype `assertion == "negated"` (model wins). (b) no `assertion` key -> category-derived fallback (backward compatible). (c) **conflict/compat-category:** `item` with `assertion="absent"`, `category="Abnormal"` -> phenotype `assertion == "negated"` AND the stored `category` is the axis-derived compat value (NOT the raw `"Abnormal"`), proving `derive_category_from_axes` is stored, not bypassed.
- [ ] **Step 2: Run it and confirm it fails** -- `uv run pytest tests/unit/llm/test_phase2_assertion_source.py -n 0` -> FAIL (assertion is always category-derived today).
- [ ] **Step 3: Implement** -- in `phenotype_from_candidate` (~273-277) replace the category-only derivation with model-first / category-fallback, using `parse_assertion`:

```python
        assertion=(
            parse_assertion(item["assertion"]).value
            if item.get("assertion") is not None
            else CATEGORY_TO_ASSERTION.get(normalize_category(str(item.get("category", ""))), PRESENT_ASSERTION)
        ),
        experiencer=(
            item["experiencer"] if item.get("experiencer")
            else experiencer_for_category(str(item.get("category", "")))
        ),
```

Add `from phentrieve.llm.utils import parse_assertion`. Add `derive_category_from_axes(experiencer, assertion, fallback_category)` (D3) -- map `experiencer=="family_history" -> "family_history"`, `experiencer=="other" -> "other"`, else project the pipeline `assertion` back onto the category enum (`negated -> "normal"`, `uncertain -> "suspected"`, `present -> "abnormal"`), defaulting to `fallback_category` when axes are absent -- and set the phenotype's stored `category` to its result. Confirm the dedup key at `:342` is unchanged.

- [ ] **Step 4: Run test + llm suite + gates** -- `uv run pytest tests/unit/llm/ -n 0 && make typecheck-fast` -> passed.
- [ ] **Step 5: Commit** -- `git add phentrieve/llm/pipeline_phase2.py tests/unit/llm/test_phase2_assertion_source.py && git commit -m "feat(llm): consume model assertion (via parse_assertion); category now derived (B1)"`

---

## B2 -- Family history -> separate list

### Task 6: Partition + resolve family-experiencer phrases through the real resolution path

**Files:** Modify `phentrieve/llm/pipeline.py` (partition before actionable filter ~337-344; family resolve reusing `_retrieve_candidates:1117-1123` + `_route_phase2_candidates:1380-1434` / `_resolve_with_mapping_prompt:1482-1566`). Test `tests/unit/llm/test_family_resolve.py`.

**Interfaces:** Produces a local `resolved_family: list[LLMPhenotype]` in `run()` (attached to the result in Task 7). Proband path unchanged except that family items are removed from it.

> **Important (Codex):** proband terms are resolved by retrieval + a phase-2 LLM mapping/local-match pass, not raw retrieval. Family items MUST go through the same path (`_retrieve_candidates` -> `_route_phase2_candidates`/`_resolve_with_mapping_prompt`) to become `LLMPhenotype`s. Tests stub BOTH retrieval and the LLM mapping.
>
> **Important (Codex round 2) -- accounting:** the proband path accumulates token totals, request counts, timings, `phase_counts`, and trace once (`pipeline.py:459` area). A second family mapping pass that does not feed the same accumulators will UNDERREPORT LLM calls/cost. Route the family pass through the same accumulation (prefer a shared resolver helper that both proband and family call, accumulating into the shared meta counters) and give it distinguishable trace/timing keys.

- [ ] **Step 1: Write the failing test** -- create `tests/unit/llm/test_family_resolve.py` with stubbed retrieval + stubbed phase-2 mapping. Feed a parsed set with one `experiencer="family_history"` phrase ("long QT syndrome") and one proband phrase; assert `run()` yields a resolved family phenotype for the family phrase and that it is NOT in the proband `resolved_terms`. **Accounting assertion:** when the stubbed family mapping records an LLM call, `result.meta` `request_count`/`phase_request_counts` reflect BOTH the proband and family mapping passes (family calls are not silently uncounted).
- [ ] **Step 2: Run it and confirm it fails** -- `uv run pytest tests/unit/llm/test_family_resolve.py -n 0` -> FAIL (family dropped by actionable filter before retrieval).
- [ ] **Step 3: Implement** -- per D1 steps 2 + 4: before the actionable filter, partition `family_items` (experiencer == family_history) out of `proband_items`; run `family_items` through `_retrieve_candidates` + the same resolution routing used for proband; keep the result in `resolved_family`. Do not merge into `resolved_terms`.
- [ ] **Step 4: Run test + llm suite + gates** -- `uv run pytest tests/unit/llm/ -n 0 && make typecheck-fast` -> passed.
- [ ] **Step 5: Commit** -- `git add phentrieve/llm/pipeline.py tests/unit/llm/test_family_resolve.py && git commit -m "feat(llm): partition + resolve family-experiencer phrases via the real mapping path (B2)"`

### Task 7: Emit `family_history_findings` on `LLMExtractionResult`

**Files:** Modify `phentrieve/llm/types.py` (`LLMExtractionResult` :199-201); `phentrieve/llm/pipeline.py` (assembly :496-517). Test `tests/unit/llm/test_llm_result_family_field.py`.

**Interfaces:** Produces `LLMExtractionResult.family_history_findings: list[LLMPhenotype] = Field(default_factory=list)`. Assembly (per D1 step 6): `family_history_findings=self._deduplicate_terms(resolved_family)`.

- [ ] **Step 1: Write the failing test** -- assert `LLMExtractionResult` accepts + defaults `family_history_findings=[]`; and (with the Task-6 stubs) that a family phrase lands in `result.family_history_findings` and NOT `result.terms`.
- [ ] **Step 2: Run it and confirm it fails** -- `uv run pytest tests/unit/llm/test_llm_result_family_field.py -n 0` -> FAIL (no field).
- [ ] **Step 3: Implement** -- add the field to `types.py:199-201`; in `pipeline.py:496-517` pass `family_history_findings=self._deduplicate_terms(resolved_family)`.
- [ ] **Step 4: Run test + gates** -- `uv run pytest tests/unit/llm/ -n 0 && make typecheck-fast` -> passed.
- [ ] **Step 5: Commit** -- `git add phentrieve/llm/types.py phentrieve/llm/pipeline.py tests/unit/llm/test_llm_result_family_field.py && git commit -m "feat(llm): emit family_history_findings, kept out of proband terms (B2)"`

### Task 8: Switch the phenopacket export guard to experiencer-based

**Files:** Modify `api/mcp/service_adapters.py` (guard :380-384). Test `tests/unit/mcp_server/test_mcp_service_adapters.py`.

**Interfaces:** Drops a phenotype iff `experiencer == "family_history"` (was `assertion == "family_history"`, which B1 no longer produces). No family term in subject `PhenotypicFeature`s.

- [ ] **Step 1: Write the failing test** -- a dict with `experiencer="family_history"` returns `None` from `_coerce_export_phenotype`; integration-style: `export_phenopacket_service` over mixed proband+family yields zero family terms in subject features.
- [ ] **Step 2: Run it and confirm it fails** -- `uv run pytest tests/unit/mcp_server/test_mcp_service_adapters.py -k family -n 0` -> FAIL (guard keys on assertion).
- [ ] **Step 3: Implement** -- replace the guard at `:381-384` with:

```python
    experiencer = str(p.get("experiencer") or "").strip().lower()
    if experiencer == "family_history" or str(assertion).strip().lower() == "family_history":
        return None
```

(keep the legacy assertion clause as an OR fallback).

- [ ] **Step 4: Run tests + gates** -- `uv run pytest tests/unit/mcp_server/ -n 0 && make typecheck-fast` -> passed.
- [ ] **Step 5: Commit** -- `git add api/mcp/service_adapters.py tests/unit/mcp_server/test_mcp_service_adapters.py && git commit -m "fix(mcp): guard phenopacket export by experiencer, not assertion (B2)"`

### Task 9: Thread `family_history_findings` + `experiencer` through service -> REST -> MCP

**Files:** Modify `phentrieve/text_processing/full_text_service.py` (`_adapt_llm_aggregated_terms` :434-455 -- carry `experiencer` + derive `excluded`; **`adapt_full_text_response` :146-172 -- add `family_history_findings` to the returned dict (today it returns ONLY meta/processed_chunks/aggregated_hpo_terms)**; `adapt_shared_service_response_to_api`); `api/schemas/text_processing_schemas.py` (`AggregatedHPOTermAPI` :189-225 add `experiencer` + `excluded: bool`; `TextProcessingResponseAPI` :228-263 add `family_history_findings`); `api/services/text_processing_execution.py` (builder :141-166 -- copy all); `api/mcp/schemas.py` (`EXTRACT_SCHEMA` :43-47 add `family_history_findings`); `api/mcp/capabilities.py` (descriptor :107-118 -- document the new output fields so `capabilities_version` rolls); `api/mcp/projection.py` (:69 `dict(term)` copies all keys -- verify `experiencer`/`excluded` survive; project the family list); `api/mcp/shaping.py` + `tools/retrieval.py` (:171-172,:248-249 -- a second `enforce_budget(list_field="family_history_findings")`). Test `tests/unit/api/test_family_findings_surface.py`, `tests/unit/mcp_server/test_mcp_family_projection.py`.

**Interfaces:** `family_history_findings` on REST + MCP extract envelope; `experiencer` preserved per term; a derived `excluded: bool` (from `is_excluded(status)`) on every surfaced term; the family list budgeted independently.

> **Fixes vs rev 1:** (a) the term `experiencer` is dropped at `_adapt_llm_aggregated_terms` (`full_text_service.py:434-455`), NOT at MCP projection (`projection.py:69` does `dict(term)` and its drop list `:35-47` excludes `experiencer`); fix the earlier layer. (b) `capabilities_version` hashes `capabilities.py`'s descriptor (`:207-223`, `:107-118`), NOT `EXTRACT_SCHEMA`; update the descriptor.
>
> **Fixes vs rev 2 (Codex round 2):**
> - **F2 (extract-status consistency):** the LLM service emits `status=term.assertion` (pipeline vocab `present/negated/uncertain`) at `full_text_service.py:439`. B0 only hardens the *phenopacket export*; the *extract* surface still reports raw pipeline status. Rather than change the `status` value (breaking), add a **derived `excluded: bool`** via `is_excluded(status)` on `AggregatedHPOTermAPI` + the MCP projection, so consumers get the actionable excluded signal consistently without a breaking vocabulary change. (Full status canonicalization to `affirmed/...` on the extract surface is a deliberate Phase-3 API change.)
> - **F4 (real drop point):** `adapt_full_text_response` (`:146-172`) returns only `meta/processed_chunks/aggregated_hpo_terms` -- so the field must be added THERE and tested through `run_llm_backend()` + `adapt_shared_service_response_to_api()`, not only via `TextProcessingResponseAPI.model_validate()`.
> - **F6 (de-brittle the capabilities test):** do NOT assert against a hard-coded old hash. Assert (i) the descriptor text documents `family_history_findings`, (ii) `EXTRACT_SCHEMA` exposes it, and (iii) the existing capabilities hash-stability tests still pass.

- [ ] **Step 1: Write the failing tests** -- `test_family_findings_surface.py`: drive `adapt_full_text_response()` / `adapt_shared_service_response_to_api()` (the REAL adapter chain) with a service response containing family findings + a `negated` term, and assert the returned dict/`TextProcessingResponseAPI` carries `family_history_findings`, per-term `experiencer`, and a derived `excluded: true` on the negated term (`excluded: false` on a present term) -- NOT just a bare `model_validate`. `test_mcp_family_projection.py`: the MCP extract envelope includes `family_history_findings`, each term keeps `experiencer`/`excluded`, the capabilities descriptor text documents `family_history_findings`, `EXTRACT_SCHEMA` exposes it, and the existing capabilities hash-stability tests still pass (do NOT hard-code an old hash).
- [ ] **Step 2: Run them and confirm they fail** -- `uv run pytest tests/unit/api/test_family_findings_surface.py tests/unit/mcp_server/test_mcp_family_projection.py -n 0` -> FAIL.
- [ ] **Step 3: Implement** per the file list above (carry `experiencer` from `_adapt_llm_aggregated_terms` onward; add schema fields; copy in the builder; add to `EXTRACT_SCHEMA`; roll the descriptor in `capabilities.py`; add the second `enforce_budget`).
- [ ] **Step 4: Run tests + gates** -- `uv run pytest tests/unit/api/ tests/unit/mcp_server/ -n 0 && make check && make typecheck-fast` -> passed; clean.
- [ ] **Step 5: Commit** -- `git add phentrieve/text_processing/full_text_service.py api/schemas/text_processing_schemas.py api/services/text_processing_execution.py api/mcp/schemas.py api/mcp/capabilities.py api/mcp/projection.py api/mcp/shaping.py tests/unit/api/test_family_findings_surface.py tests/unit/mcp_server/test_mcp_family_projection.py && git commit -m "feat(api): surface family_history_findings + experiencer through service, REST, MCP (B2)"`

---

## B3 -- "X without Y" -> excluded term

### Task 10: Generate a qualifier-derived excluded finding after proband resolution

**Files:** Modify `phentrieve/llm/pipeline.py` (generate exclusions in `run()` AFTER proband resolve and BEFORE `_deduplicate_terms` at :496-497; use `self.similarity_threshold` :148-153 as the floor); `phentrieve/llm/types.py` (add `qualifier_surface_text`, `match_method` to the phenotype/`LLMPhenotype` shape). Test `tests/unit/llm/test_qualifier_excluded_term.py`.

**Interfaces:** Produces, when `Y` maps above the floor, a generated finding: `hpo_id`, `label`, `assertion="negated"` (pipeline vocab), `qualifier_surface_text="Y"`, `match_method="negated_qualifier_derived"`, `confidence >= floor`, AND -- critically -- the **source finding's `source_chunk_ids` + evidence records** (see F5 below). Below floor -> keep today's metadata string, no generated term.

> **Fixes vs rev 1:** (a) emit point is the resolve flow (before `_deduplicate_terms` :496-497), NOT line 1234 (which only copies metadata and lacks the resolved primary finding/evidence). (b) the floor is `self.similarity_threshold`, NOT the nonexistent `min_confidence_for_aggregated`.
>
> **Fix vs rev 2 (Codex round 2, F5):** `_adapt_llm_aggregated_terms` DROPS any term whose evidence references no valid chunk -- `if raw_source_chunk_ids and not source_chunk_ids: continue` (`full_text_service.py:412`). A generated excluded term with empty/invalid chunk ids would silently vanish at the service boundary. The generated term MUST inherit the source proband finding's `source_chunk_ids` and at least one evidence record with a valid chunk reference (the "X without Y" span lives in the same chunk as X). Task 11 adds a service-level survival test.

- [ ] **Step 1: Write the failing test** -- `tests/unit/llm/test_qualifier_excluded_term.py`, stubbed retrieval. Case A (maps above floor): a resolved proband finding with `negated_qualifier="fever"` yields a generated finding, `assertion == "negated"`, `match_method == "negated_qualifier_derived"`, `qualifier_surface_text == "fever"`. Case B (below floor/unmappable): no generated term; `negated_qualifier` metadata string retained.
- [ ] **Step 2: Run it and confirm it fails** -- `uv run pytest tests/unit/llm/test_qualifier_excluded_term.py -n 0` -> FAIL.
- [ ] **Step 3: Implement** -- add the fields to `types.py`; in `run()` after `resolved_terms` is built (before `:497`), for each finding with a truthy `negated_qualifier`, run one retrieval on the qualifier phrase; if top confidence `>= self.similarity_threshold`, build the generated excluded `LLMPhenotype` and collect it into `qualifier_exclusions`; assemble `terms=self._deduplicate_terms(resolved_terms + qualifier_exclusions)`.
- [ ] **Step 4: Run test + llm suite + gates** -- `uv run pytest tests/unit/llm/ -n 0 && make typecheck-fast` -> passed.
- [ ] **Step 5: Commit** -- `git add phentrieve/llm/pipeline.py phentrieve/llm/types.py tests/unit/llm/test_qualifier_excluded_term.py && git commit -m "feat(llm): map negated_qualifier to a generated excluded finding, gated by similarity_threshold (B3)"`

### Task 11: Thread the excluded-finding shape through service -> REST -> MCP

**Files:** Modify `phentrieve/text_processing/full_text_service.py`; `api/schemas/text_processing_schemas.py` + `api/services/text_processing_execution.py` (:141-166 add `negated_qualifier`/`qualifier_surface_text`/`match_method`/evidence to `AggregatedHPOTermAPI`, stop dropping); `api/mcp/projection.py`. Test `tests/unit/api/test_qualifier_excluded_surface.py`.

**Interfaces:** `qualifier_surface_text`, `match_method`, evidence present on REST + MCP for a `negated_qualifier_derived` finding.

- [ ] **Step 1: Write the failing test** -- (a) a `negated_qualifier_derived` finding round-trips through `AggregatedHPOTermAPI` with the fields populated; the response builder copies them. (b) **service-survival (F5):** run a generated excluded term (carrying the source finding's `source_chunk_ids` + one valid evidence record) through `_adapt_llm_aggregated_terms()` and assert it SURVIVES (is not dropped by the `:412` `continue`), and a variant with no valid chunk ref is the one that would drop -- proving the inherited chunk ids are what keep it alive.
- [ ] **Step 2: Run it and confirm it fails** -- `uv run pytest tests/unit/api/test_qualifier_excluded_surface.py -n 0` -> FAIL.
- [ ] **Step 3: Implement** -- add fields to `AggregatedHPOTermAPI`; copy in the builder; carry from `_adapt_llm_aggregated_terms`; project in MCP.
- [ ] **Step 4: Run tests + gates** -- `uv run pytest tests/unit/api/ tests/unit/mcp_server/ -n 0 && make check && make typecheck-fast` -> passed; clean.
- [ ] **Step 5: Commit** -- `git add phentrieve/text_processing/full_text_service.py api/schemas/text_processing_schemas.py api/services/text_processing_execution.py api/mcp/projection.py tests/unit/api/test_qualifier_excluded_surface.py && git commit -m "feat(api): surface negated_qualifier-derived excluded findings through REST + MCP (B3)"`

---

## Guard + close

### Task 12: i18n negation-scope guard test

**Files:** Test `tests/unit/llm/test_two_phase_prompt_negation_guard.py`.

**Interfaces:** Asserts every **extraction** `two_phase` prompt template contains the negation-scope block + a `negated_qualifier` few-shot.

> **Fix vs rev 1:** the real files are `en.yaml`, `en_mapping.yaml`, `en_mapping_batch.yaml`. The mapping prompts do NOT contain the negation block, so `_mapping*` (rev 1) is the wrong exclusion. Positively select extraction templates: files matching `two_phase/*.yaml` and NOT `*_mapping.yaml` / `*_mapping_batch.yaml`.

- [ ] **Step 1: Write the test** -- glob `phentrieve/llm/prompts/templates/two_phase/*.yaml`, drop `*_mapping.yaml` and `*_mapping_batch.yaml`, and assert each remaining template's text contains the negation-scope rule (present in `en.yaml:27-31`) and a `negated_qualifier` few-shot (`en.yaml:119-137`). Today only `en.yaml` qualifies, so it passes now and guards future locales.
- [ ] **Step 2: Run it** -- `uv run pytest tests/unit/llm/test_two_phase_prompt_negation_guard.py -n 0` -> PASS.
- [ ] **Step 3: Commit** -- `git add tests/unit/llm/test_two_phase_prompt_negation_guard.py && git commit -m "test(llm): assert extraction two_phase prompts keep negation-scope + qualifier few-shots"`

### Task 13: Full-gate + partial acceptance (LLM benchmark gate owned, not silent)

- [ ] **Step 1: Repo trio** -- `make check && make typecheck-fast && make test` -> all green.
- [ ] **Step 2: CI-parity + security** -- `make ci-local && make security-python` -> EXIT 0.
- [ ] **Step 3: Deterministic present-only no-regression (runs here)** -- regenerate the tiny extraction summary and assert vs the committed Phase 1 baseline:

```bash
PHENTRIEVE_DATA_ROOT_DIR=<repo>/data uv run phentrieve benchmark extraction run tests/data/extraction/tiny_extraction_test.json \
  --model FremyCompany/BioLORD-2023-M --scoring-mode present-only --output-dir results/_p2_present --no-bootstrap-ci
uv run phentrieve benchmark extraction assert-no-regression \
  --baseline tests/data/benchmarks/baselines/tiny_present_only_summary.json \
  --candidate results/_p2_present/extraction_summary.json
```
Expected: "No regression", exit 0 (B0-B3 change the LLM path, not the deterministic extractor).

- [ ] **Step 4: Golden edge cases + all new tests pass in strict** -- `uv run pytest tests/unit/llm tests/unit/mcp_server tests/unit/api tests/unit/benchmark -n 0` -> passed.
- [ ] **Step 5: LLM mapping benchmark gate -- OWNED, Docker/Gemini (spec 9.1)** -- cannot run in a bare checkout. Either (a) run it in Docker/Gemini and confirm no regression vs the committed LLM baseline, or (b) STOP and hand off: land B0-B3 with Steps 1-4 green and record in the PR that the LLM-benchmark gate is pending an operator run before merge to `main`.

---

## Follow-on (Phase 3, not this plan)

- Vue **display**: `FullTextResponseReceipt.vue`, `AggregatedTermsView.vue`, `PhenotypeCollectionPanel.vue`, `ResultsDisplay.vue` -- a "Family history" section + an "excluded" chip; i18n locale keys (`make frontend-i18n-check`); component tests.
- Close #289; final verification doc; CHANGELOG; coordinated version bump; release.
- **Deferred from Phase 2 (D2):** fully unify the pipeline onto the canonical `affirmed/...` vocabulary (retire `PRESENT_ASSERTION="present"`), if desired -- a large, separately-gated change.

## Review deltas (rev 1 -> rev 2)

Applied from self-review + Codex review (all verified against real code):
- **[Critical] Task 2** used the 4-valued `canonicalize_assertion` against a 2-valued `Literal["affirmed","negated"]` export field -> ValidationError. Now `is_excluded` (2-valued). (`phenopacket_schemas.py:150`)
- **[Critical] Vocabulary strategy.** Three vocabularies, not two. B0 is export-only; B1 uses the revived `parse_assertion` (wire->pipeline) internally, not `canonicalize_assertion`. New "Data-flow contract" D1-D3. (`llm/config.py:50-52`)
- **[Critical] B1 axes re-dropped** in `_retrieve_candidates` rebuild (`pipeline.py:1215-1241`) and category still load-bearing -> Task 4 now threads axes through parse + Phase-1 dedup + retrieve rebuild + mapping payload; Task 5 derives compat category.
- **[Important] B2** family resolution must use the real `_retrieve_candidates` + `_route_phase2_candidates`/`_resolve_with_mapping_prompt` path (not raw retrieval); explicit storage boundary (`resolved_family`) and assembly point (`:496-517`).
- **[Important] B3** emit point moved to before `_deduplicate_terms` (`:496-497`); floor is `self.similarity_threshold` (`min_confidence_for_aggregated` does not exist on the LLM path).
- **[Important] Task 9** `capabilities_version` rolls from `capabilities.py` descriptor, not `EXTRACT_SCHEMA`; the term `experiencer` drop point is `_adapt_llm_aggregated_terms`, not MCP projection.
- **[Minor] Task 12** glob exclusion corrected (`*_mapping.yaml`/`*_mapping_batch.yaml`); **Task 3** frontend test path corrected (`frontend/src/test/composables/usePhenotypeCollection.test.js`).
- **[Confirmed]** experiencer literal is exactly `family_history` (`types.py:102`); no normalization task needed.

### Review deltas (rev 2 -> rev 3, Codex round 2 -- all verified against real code)

- **[Critical] D3/Task 5** stated "category becomes derived compat" but the snippet didn't implement it (category still built from the legacy value at `pipeline_phase2.py:273`, and actionability still routes on category at `pipeline.py:340`). Added an explicit `derive_category_from_axes()` helper, required storing its result as the phenotype `category`, and added an axes-vs-category conflict test.
- **[High] Task 9 extract-status (F2).** B0 only hardened phenopacket export; the extract surface still emits `status=term.assertion` (`full_text_service.py:439`). Added a non-breaking derived `excluded: bool` (via `is_excluded`) on REST + MCP terms instead of a breaking status-vocabulary change (full canonicalization deferred to Phase 3).
- **[High] Task 6 accounting (F3).** A second family mapping pass would underreport tokens/requests/timings/trace (`pipeline.py:459`). Required shared accumulation + a test asserting family mapping increments `request_count`/`phase_request_counts`.
- **[Medium] Task 9 real drop point (F4).** `adapt_full_text_response` (`:146-172`) returns only `meta/processed_chunks/aggregated_hpo_terms`; family must be added there and tested through `run_llm_backend()`/`adapt_shared_service_response_to_api()`, not just `model_validate`.
- **[Medium] Task 10/11 evidence survival (F5).** `_adapt_llm_aggregated_terms` drops terms without a valid chunk ref (`:412` `continue`); generated exclusions must inherit the source finding's `source_chunk_ids` + evidence. Added a service-level survival test.
- **[Medium] Task 9 capabilities test (F6).** De-brittled: assert descriptor documents the field + `EXTRACT_SCHEMA` exposes it + existing hash-stability tests pass, instead of comparing to a hard-coded old hash.
- **[Minor] Task 3 (F7).** `npm run test` is Vitest watch-mode; use `npm run test:run` (verified `frontend/package.json:14-17`).
