# Extraction Contract v2 -- Phase 2 Implementation Plan (LLM contract v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LLM extraction output fully correct and consistent across MCP/REST/Vue -- assertion becomes load-bearing, family-history surfaces as its own list instead of being silently dropped, and "X without Y" becomes a machine-actionable excluded term -- all built on one shared canonical-assertion vocabulary so `absent` can never silently export as present.

**Architecture:** Four atomic, individually-gated behavior changes in dependency order. **B0** lands first: a single shared `canonicalize_assertion()` helper (canonical vocabulary `affirmed / negated / normal / uncertain`) reused at every boundary, replacing ad-hoc `== "negated"` string checks. **B1** carries the model's own `experiencer` + `assertion` end-to-end instead of re-deriving polarity from the legacy `category` enum. **B2** collects family-experiencer phrases before the actionable filter, maps them through the same retrieval, and emits a dedicated `family_history_findings` list kept out of the proband phenopacket. **B3** maps the `negated_qualifier` phrase to an HPO id and emits it as an excluded finding (confidence-gated, string fallback). The new fields are threaded through the shared service + REST/MCP schemas in this phase; the Vue *display* is Phase 3.

**Tech Stack:** Python 3.11+, `uv`, Ruff, mypy, pytest (+ pytest-xdist); Pydantic v2 (LLM/REST schemas); FastAPI; FastMCP; Vue 3 + Vitest (B0 export mirror only).

**Source spec:** `.planning/specs/2026-07-03-extraction-contract-v2-and-finalization-design.md` (section 7 = B0-B3 + carrier surfaces; decisions locked in section 3). Anchor line numbers below are verified against branch `feat/extraction-contract-v2` HEAD.

## Scope boundary (read first)

- **This plan (Phase 2):** B0 canonical vocabulary + boundary hardening (MCP export + Vue *export* coercion), B1 assertion load-bearing, B2 family-history list, B3 qualifier->excluded, and threading the new fields through the shared service + REST/MCP **data** schemas.
- **Deferred to Phase 3:** the Vue **display** components (family-history section, "excluded" chips on ruled-out terms), i18n locale keys for those, closing issue #289, the final verification doc, and the coordinated CLI/API/Frontend release. B0's Vue *export composable* mirror IS in this plan (it is coercion, not display, and B0's correctness spans that boundary).
- **Environment gate (called out, not assumed -- spec 9.1):** Phase 2's full acceptance gate is three-part (spec 6.4): (1) present-only no-regression on the deterministic corpora, (2) new golden edge cases pass in strict, (3) the **LLM mapping benchmark does not regress**. Gates (1) and (2) run in this checkout with the Phase 1 machinery. Gate (3) needs the **Docker stack + Gemini key**, which live in the container, not this checkout -- Task 13 makes running it an explicit, owned step, not a silent assumption. Do NOT claim Phase 2 "gated" without (3); if (3) cannot run here, land the code with (1)+(2) green and STOP for the operator to run (3) in the Docker/Gemini environment.

## Global Constraints

- Dependency management: `uv` only; never `pip`.
- Typing: modern (`list[str]`, `str | None`, `dict[str, X]`); mypy targets Python 3.11.
- Format/lint: Ruff. All tests under `tests/`; never create `tests_new/` or new `tests/unit/api/` sub-package dirs. ASCII only.
- Required gates before claiming a task done: `make check`, `make typecheck-fast`, `make test`. Before any push: `make ci-local` + `make security-python`. Frontend task (Task 3) also runs `make frontend-test-ci` + `make frontend-i18n-check` is NOT needed (no locale keys here).
- `make test` runs under `pytest-xdist`; for single-file debugging use `uv run pytest ... -n 0`.
- Atomic commits (one behavior/deliverable = one commit); coverage-improving tests on all touched code. TDD per fix.
- **B0 is the foundation and lands first.** Every boundary that converts an assertion to an excluded/affirmed feature MUST route through `canonicalize_assertion()` -- no new ad-hoc `== "negated"` string checks.
- **Canonical vocabulary = `affirmed / negated / normal / uncertain`** (mirrors `phentrieve/text_processing/assertion_detection.py:28-36`). `absent` (LLM wire value) and `negated` both canonicalize to `negated` and both render `excluded: true`.
- **No proband leakage:** a `family_history` experiencer term must never appear in the subject's `PhenotypicFeature`s. Guarded by a regression test (Task 8).
- **Determinism / no-regression:** the Phase 1 identity guarantee still holds; the deterministic extraction benchmark's committed baselines must not regress under `present-only` (Task 13).

## Vocabulary reality (verified -- three vocabularies exist today)

1. **LLM wire schema literal** `Literal["present", "absent", "uncertain"]` -- `phentrieve/llm/types.py:105-107` (`LLMExtractedPhenotype.assertion`) and `:134-136` (`LLMGroundedExtractedPhenotype.assertion`). This is what the model emits.
2. **`llm/types.py` `AssertionStatus`** StrEnum `present / negated / uncertain` (`types.py:24-28`) -- barely used; default `assertion` field at `:59`.
3. **Canonical pipeline/export `AssertionStatus`** enum `affirmed / negated / normal / uncertain` -- `phentrieve/text_processing/assertion_detection.py:28-36`. **This is the canonical boundary vocabulary B0 standardizes on.**

A dead `parse_assertion()` (`phentrieve/llm/utils.py:76-88`) already maps `absent/negated/excluded/no/denied -> AssertionStatus.NEGATED` but targets vocabulary (2) and is never called. B0 supersedes it with a canonical helper targeting vocabulary (3).

## File structure

| File | Responsibility | Tasks |
|---|---|---|
| `phentrieve/assertion_vocab.py` (new) | Shared `canonicalize_assertion()` + `is_excluded()` + canonical constants | 1 |
| `api/mcp/service_adapters.py` | MCP phenopacket export coercion; family drop; experiencer guard | 2, 8 |
| `frontend/src/composables/usePhenotypeCollection.js` | Vue phenopacket export coercion (B0 mirror) | 3 |
| `phentrieve/llm/pipeline.py` | Phase-1 parse (carry axes); actionable filter; family collect+map; qualifier merge | 4, 6, 10 |
| `phentrieve/llm/pipeline_phase2.py` | Consume model assertion/experiencer; category->derived-compat | 5 |
| `phentrieve/llm/types.py` | `LLMExtractionResult.family_history_findings`; excluded-finding shape | 7, 10 |
| `phentrieve/text_processing/full_text_service.py` | Shared service payload: carry family + excluded + experiencer | 9, 11 |
| `api/schemas/text_processing_schemas.py` | REST response schema fields | 9, 11 |
| `api/services/text_processing_execution.py` | REST response builder (stop dropping fields) | 9, 11 |
| `api/mcp/schemas.py`, `api/mcp/projection.py`, `api/mcp/shaping.py` | MCP output schema + per-term projection + budget for the new list | 9, 11 |
| `tests/...` | unit/integration coverage per task | all |

---

## B0 -- Canonical boundary vocabulary (foundation; land first)

### Task 1: Add the shared `canonicalize_assertion()` helper

**Files:**
- Create: `phentrieve/assertion_vocab.py`
- Test: `tests/unit/test_assertion_vocab.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `canonicalize_assertion(raw: str | None) -> str` returning one of `"affirmed" | "negated" | "normal" | "uncertain"`; `is_excluded(raw: str | None) -> bool`; module constants `AFFIRMED="affirmed"`, `NEGATED="negated"`, `NORMAL="normal"`, `UNCERTAIN="uncertain"`. Consumed by Tasks 2, 8, 9, 10, 11.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_assertion_vocab.py`:

```python
import pytest

from phentrieve.assertion_vocab import (
    AFFIRMED,
    NEGATED,
    NORMAL,
    UNCERTAIN,
    canonicalize_assertion,
    is_excluded,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("present", AFFIRMED),
        ("affirmed", AFFIRMED),
        ("abnormal", AFFIRMED),
        ("absent", NEGATED),      # LLM wire value MUST become negated (not affirmed)
        ("negated", NEGATED),
        ("excluded", NEGATED),
        ("no", NEGATED),
        ("normal", NORMAL),
        ("uncertain", UNCERTAIN),
        ("suspected", UNCERTAIN),
        ("  ABSENT  ", NEGATED),  # case/space-insensitive
        (None, AFFIRMED),          # missing -> default present/affirmed
        ("nonsense", AFFIRMED),    # unknown -> default affirmed
    ],
)
def test_canonicalize(raw, expected):
    assert canonicalize_assertion(raw) == expected


@pytest.mark.parametrize("raw", ["absent", "negated", "excluded", "NO"])
def test_is_excluded_true(raw):
    assert is_excluded(raw) is True


@pytest.mark.parametrize("raw", ["present", "affirmed", "normal", "uncertain", None])
def test_is_excluded_false(raw):
    assert is_excluded(raw) is False
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/test_assertion_vocab.py -n 0`
Expected: FAIL (ModuleNotFoundError: `phentrieve.assertion_vocab`).

- [ ] **Step 3: Implement the helper**

Create `phentrieve/assertion_vocab.py`:

```python
"""Canonical assertion vocabulary shared across every boundary.

The pipeline, LLM wire schema, and legacy category enum use different words for
the same polarity. This module is the single place that normalizes them to the
canonical set ``affirmed / negated / normal / uncertain`` so that, in
particular, an LLM ``assertion="absent"`` can never silently export as an
affirmed (present) feature. Mirror this mapping in the Vue export composable.
"""

from __future__ import annotations

AFFIRMED = "affirmed"
NEGATED = "negated"
NORMAL = "normal"
UNCERTAIN = "uncertain"

# Every synonym that must be treated as an excluded (ruled-out) finding.
_NEGATED = {"negated", "negative", "absent", "excluded", "no", "denied"}
_UNCERTAIN = {"uncertain", "possible", "suspected", "probable"}
_NORMAL = {"normal"}


def canonicalize_assertion(raw: str | None) -> str:
    """Map any assertion synonym onto the canonical vocabulary.

    Unknown / missing values default to ``affirmed`` (present), matching the
    LLM schema default -- but callers that must fail safe should gate on
    :func:`is_excluded` for the excluded decision, which never defaults to True.
    """
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

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/unit/test_assertion_vocab.py -n 0`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/assertion_vocab.py tests/unit/test_assertion_vocab.py
git commit -m "feat(llm): add canonical assertion vocabulary helper (B0 foundation)"
```

### Task 2: Route the MCP phenopacket export through `canonicalize_assertion`

**Files:**
- Modify: `api/mcp/service_adapters.py` (`_coerce_export_phenotype`, line 380-402)
- Test: `tests/unit/mcp_server/test_mcp_service_adapters.py` (add cases)

**Interfaces:**
- Consumes: `canonicalize_assertion`, `is_excluded` (Task 1).
- Produces: `_coerce_export_phenotype` now sets `assertion_status` via the canonical helper, so `absent` -> `negated` -> `excluded: true` (today `absent` silently becomes `affirmed`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/mcp_server/test_mcp_service_adapters.py` (import the module's `_coerce_export_phenotype` and the export request class it uses -- match the existing test file's imports):

```python
from phentrieve.assertion_vocab import NEGATED
# ... existing imports; reuse the ExportPhenotypeRequest class already imported there


def test_absent_assertion_exports_as_excluded_not_present():
    # The core B0 defect: an LLM 'absent' must not become an affirmed feature.
    out = _coerce_export_phenotype(ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": "absent"}, 0)
    assert out is not None
    assert out.assertion_status == NEGATED


def test_negated_assertion_still_excluded():
    out = _coerce_export_phenotype(ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": "negated"}, 0)
    assert out.assertion_status == NEGATED


def test_present_assertion_affirmed():
    out = _coerce_export_phenotype(ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": "present"}, 0)
    assert out.assertion_status == "affirmed"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/mcp_server/test_mcp_service_adapters.py -k absent_assertion -n 0`
Expected: FAIL (`assertion_status == "affirmed"` today for `absent`).

- [ ] **Step 3: Implement**

In `api/mcp/service_adapters.py`, add `from phentrieve.assertion_vocab import canonicalize_assertion` at the top, and change line 397 from:

```python
        assertion_status="negated" if assertion == "negated" else "affirmed",
```

to:

```python
        assertion_status=canonicalize_assertion(assertion),
```

(Note: the family-history early-return at line 383 stays for now; Task 8 replaces it with an experiencer-based guard once B1 splits the axes.)

- [ ] **Step 4: Run the tests + gates**

Run: `uv run pytest tests/unit/mcp_server/test_mcp_service_adapters.py -n 0 && make typecheck-fast`
Expected: passed; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add api/mcp/service_adapters.py tests/unit/mcp_server/test_mcp_service_adapters.py
git commit -m "fix(mcp): canonicalize assertion at phenopacket export so absent->excluded (B0)"
```

### Task 3: Mirror the canonical excluded rule in the Vue export composable

**Files:**
- Modify: `frontend/src/composables/usePhenotypeCollection.js` (line 148)
- Test: `frontend/src/composables/__tests__/usePhenotypeCollection.spec.js` (create or extend; match existing frontend test layout)

**Interfaces:**
- Consumes: nothing (small local mirror of the canonical rule -- JS has no import of the Python helper).
- Produces: export sets `excluded` true for `negated` AND `absent` (today only literal `negated`).

- [ ] **Step 1: Write the failing test**

Create/extend `frontend/src/composables/__tests__/usePhenotypeCollection.spec.js` asserting that a collected phenotype with `assertion_status: 'absent'` exports with `excluded: true`, and `'negated'` likewise, and `'present'` with `excluded: false`. (Follow the existing composable-test pattern in `frontend/src/**/__tests__`; drive the same export function that contains line 148.)

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd frontend && npm run test -- usePhenotypeCollection`
Expected: FAIL for the `'absent'` case (today only `=== 'negated'`).

- [ ] **Step 3: Implement**

Add a small local helper near the top of `usePhenotypeCollection.js`:

```js
const EXCLUDED_ASSERTIONS = new Set(['negated', 'absent', 'excluded', 'no', 'denied'])
const isExcluded = (a) => EXCLUDED_ASSERTIONS.has(String(a ?? '').trim().toLowerCase())
```

and change line 148 from `excluded: cp.assertion_status === 'negated',` to `excluded: isExcluded(cp.assertion_status),`.

- [ ] **Step 4: Run the test + frontend gate**

Run: `cd frontend && npm run test -- usePhenotypeCollection` then `make frontend-test-ci`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/composables/usePhenotypeCollection.js frontend/src/composables/__tests__/usePhenotypeCollection.spec.js
git commit -m "fix(frontend): treat absent as excluded in phenopacket export (B0 mirror)"
```

---

## B1 -- Assertion load-bearing (carry BOTH axes end-to-end)

### Task 4: Carry the model's `experiencer` + `assertion` through the Phase-1 parse

**Files:**
- Modify: `phentrieve/llm/pipeline.py` (the Phase-1 parse dict, ~line 582-594)
- Test: `tests/unit/llm/test_phase1_parse_axes.py`

**Interfaces:**
- Consumes: the Phase-1 response phenotype objects (already carry `.experiencer` and `.assertion`).
- Produces: each `parsed` dict now includes `"experiencer"` and `"assertion"` keys (canonicalized assertion), consumed by Task 5.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/llm/test_phase1_parse_axes.py`. Drive the real Phase-1 parse helper (find the function that builds the `parsed` dict around `pipeline.py:582-594`; it is the smallest unit that maps a response phenotype -> dict). If that dict-building is inline in a larger function, extract it into a named helper `_parse_phase1_phenotype(phenotype) -> dict` in Step 3 first, then test the helper. Assert that given a phenotype with `experiencer="family_history"` and `assertion="absent"`, the returned dict has `parsed["experiencer"] == "family_history"` and `parsed["assertion"] == "absent"` (raw wire value preserved here; canonicalization happens at consumption in Task 5 and at export in B0).

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/llm/test_phase1_parse_axes.py -n 0`
Expected: FAIL (KeyError / missing keys -- the dict drops both axes today).

- [ ] **Step 3: Implement**

In `pipeline.py`, in the parsed-dict construction (~582-594), add two keys alongside the existing `phrase`/`category`/`negated_qualifier`:

```python
        "experiencer": getattr(phenotype, "experiencer", None),
        "assertion": getattr(phenotype, "assertion", None),
```

Do NOT remove `category` yet (Task 5 makes it derived-compat). If the dict is built inline, extract `_parse_phase1_phenotype` per Step 1 so the test has a unit to target.

- [ ] **Step 4: Run the test + gates**

Run: `uv run pytest tests/unit/llm/test_phase1_parse_axes.py -n 0 && make typecheck-fast`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/pipeline.py tests/unit/llm/test_phase1_parse_axes.py
git commit -m "feat(llm): carry model experiencer + assertion through phase-1 parse (B1)"
```

### Task 5: Consume the model's `assertion` in Phase-2; make `category` derived-compat

**Files:**
- Modify: `phentrieve/llm/pipeline_phase2.py` (`phenotype_from_candidate`, ~line 255-287; keep dedup key at :342)
- Test: `tests/unit/llm/test_phase2_assertion_source.py`

**Interfaces:**
- Consumes: the parsed dict with `"assertion"` / `"experiencer"` (Task 4); `canonicalize_assertion` (Task 1).
- Produces: `phenotype_from_candidate` sets `assertion` from the model's own value (canonicalized) when present, falling back to `CATEGORY_TO_ASSERTION[category]` only when the model omitted it; `experiencer` from the model's value with `experiencer_for_category` as fallback. Dedup key `(term_id, experiencer, assertion)` unchanged (`:342`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/llm/test_phase2_assertion_source.py`: build a candidate `item` dict whose `category` would map to `affirmed` via `CATEGORY_TO_ASSERTION` but whose model-emitted `assertion="absent"`. Call `phenotype_from_candidate(item, ...)` and assert the resulting phenotype's `assertion` canonicalizes to `negated` (the model wins, not the category). A second case: `item` with no `assertion` key falls back to the category-derived value (backward compatible).

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/llm/test_phase2_assertion_source.py -n 0`
Expected: FAIL (today assertion is always `CATEGORY_TO_ASSERTION[category]`, ignoring the model's `absent`).

- [ ] **Step 3: Implement**

In `pipeline_phase2.py` `phenotype_from_candidate` (~273-277), replace the category-only derivation:

```python
        assertion=CATEGORY_TO_ASSERTION.get(normalize_category(str(item.get("category", ""))), PRESENT_ASSERTION),
        experiencer=experiencer_for_category(str(item.get("category", ""))),
```

with model-first, category-fallback (using `canonicalize_assertion`):

```python
        assertion=(
            canonicalize_assertion(item["assertion"])
            if item.get("assertion") is not None
            else CATEGORY_TO_ASSERTION.get(normalize_category(str(item.get("category", ""))), PRESENT_ASSERTION)
        ),
        experiencer=(
            item["experiencer"]
            if item.get("experiencer")
            else experiencer_for_category(str(item.get("category", "")))
        ),
```

Add `from phentrieve.assertion_vocab import canonicalize_assertion`. Keep `category` on the phenotype as a derived/compat field. Confirm the dedup key at `:342` still reads `(term.term_id, term.experiencer, term.assertion)`.

- [ ] **Step 4: Run the test + gates**

Run: `uv run pytest tests/unit/llm/test_phase2_assertion_source.py tests/unit/llm/ -n 0 && make typecheck-fast`
Expected: passed (existing phase-2 tests still green).

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/pipeline_phase2.py tests/unit/llm/test_phase2_assertion_source.py
git commit -m "feat(llm): consume model assertion/experiencer as polarity (category now derived) (B1)"
```

---

## B2 -- Family history -> separate list

### Task 6: Collect + map family-experiencer phrases before the actionable filter

**Files:**
- Modify: `phentrieve/llm/pipeline.py` (actionable filter ~340-344; retrieval gate ~399-406)
- Test: `tests/unit/llm/test_family_collect_and_map.py`

**Interfaces:**
- Consumes: the parsed items with `"experiencer"` (Task 4); the existing retrieval used for proband terms (`_retrieve_candidates`).
- Produces: a `family_candidates` list resolved to HPO ids through the same retrieval path, available to Task 7's emit -- collected from items whose experiencer is `family_history` BEFORE the `ACTIONABLE_CATEGORIES` filter drops them.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/llm/test_family_collect_and_map.py`. Monkeypatch the retrieval so no model/index is needed (mirror the pattern in existing `tests/unit/llm/` pipeline tests -- stub the retriever/`_retrieve_candidates` to return a deterministic mapping). Feed a parsed set containing one `experiencer="family_history"` phrase ("long QT syndrome") and one proband phrase. Assert the pipeline produces a family mapping for "long QT syndrome" and that the proband path does NOT contain it.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/llm/test_family_collect_and_map.py -n 0`
Expected: FAIL (family phrase is dropped by the actionable filter before retrieval; no family mapping exists).

- [ ] **Step 3: Implement**

In `pipeline.py`, immediately BEFORE the actionable filter at ~340-344, partition the parsed items:

```python
        family_items = [it for it in extracted if str(it.get("experiencer") or "").strip().lower() == "family_history"]
```

Run these `family_items` through the same retrieval/mapping used for proband terms (a parallel resolution pass -- reuse the `_retrieve_candidates` call the proband path uses, applied to `family_items`; do NOT reuse the actionable-filtered set, which drops them). Keep the mapped family results in a local (e.g. `family_findings`) for Task 7 to attach to the result. The existing actionable filter and proband retrieval stay unchanged for the proband path.

Note: `experiencer_for_category` (`pipeline_phase1.py:54-66`) still projects legacy category onto `family_history` for pre-B1 inputs; the partition above keys on the (now model-supplied) experiencer, so both sources are covered.

- [ ] **Step 4: Run the test + gates**

Run: `uv run pytest tests/unit/llm/test_family_collect_and_map.py tests/unit/llm/ -n 0 && make typecheck-fast`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/pipeline.py tests/unit/llm/test_family_collect_and_map.py
git commit -m "feat(llm): collect + map family-experiencer phrases before the actionable filter (B2)"
```

### Task 7: Emit `family_history_findings` on `LLMExtractionResult`; keep out of proband

**Files:**
- Modify: `phentrieve/llm/types.py` (`LLMExtractionResult`, :199-201)
- Modify: `phentrieve/llm/pipeline.py` (result construction -- attach the family findings from Task 6)
- Test: `tests/unit/llm/test_llm_result_family_field.py`

**Interfaces:**
- Consumes: the mapped family findings (Task 6).
- Produces: `LLMExtractionResult.family_history_findings: list[LLMPhenotype]` (default empty). Proband `terms` excludes family experiencers.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/llm/test_llm_result_family_field.py`: assert `LLMExtractionResult` accepts a `family_history_findings` list and defaults it to `[]`; and (end-to-end with the monkeypatched retrieval from Task 6) assert a family phrase lands in `result.family_history_findings` and NOT in `result.terms`.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/llm/test_llm_result_family_field.py -n 0`
Expected: FAIL (`LLMExtractionResult` has no `family_history_findings`).

- [ ] **Step 3: Implement**

In `types.py:199-201`, add the field:

```python
    family_history_findings: list[LLMPhenotype] = Field(default_factory=list)
```

In `pipeline.py` result construction, pass the Task-6 family findings into it and ensure the proband `terms` are the non-family set.

- [ ] **Step 4: Run the test + gates**

Run: `uv run pytest tests/unit/llm/test_llm_result_family_field.py tests/unit/llm/ -n 0 && make typecheck-fast`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/types.py phentrieve/llm/pipeline.py tests/unit/llm/test_llm_result_family_field.py
git commit -m "feat(llm): emit family_history_findings list, kept out of proband terms (B2)"
```

### Task 8: Switch the phenopacket export guard to experiencer-based

**Files:**
- Modify: `api/mcp/service_adapters.py` (`_coerce_export_phenotype` guard, :380-384; `export_phenopacket_service`, :406-432)
- Test: `tests/unit/mcp_server/test_mcp_service_adapters.py` (add regression case)

**Interfaces:**
- Consumes: phenotype dicts now carrying `experiencer` (B1/B2).
- Produces: the export drops a phenotype iff `experiencer == "family_history"` (was `assertion == "family_history"`, which no longer matches once experiencer is its own axis). No family term reaches the subject's `PhenotypicFeature`s.

- [ ] **Step 1: Write the failing test**

Add a regression test: a phenotype dict with `experiencer="family_history"` (and any assertion) returns `None` from `_coerce_export_phenotype` (i.e. excluded from the subject packet); and an integration-style assertion that `export_phenopacket_service` given a mix of proband + family phenotypes yields zero family terms in the subject `PhenotypicFeature`s.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/mcp_server/test_mcp_service_adapters.py -k family -n 0`
Expected: FAIL (guard keys on `assertion == "family_history"`, which B1 no longer produces -- family would leak into the packet).

- [ ] **Step 3: Implement**

Replace the guard at `:381-384`:

```python
    if str(assertion).strip().lower() == "family_history":
        return None
```

with an experiencer-based guard:

```python
    experiencer = str(p.get("experiencer") or "").strip().lower()
    # A family-history mention is not a proband phenotypic feature (LLM-1/B2).
    if experiencer == "family_history":
        return None
```

(Read `experiencer` from the phenotype dict; keep the legacy `assertion == "family_history"` check as an OR-clause fallback for pre-B1 callers if any remain.)

- [ ] **Step 4: Run the test + gates**

Run: `uv run pytest tests/unit/mcp_server/ -n 0 && make typecheck-fast`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add api/mcp/service_adapters.py tests/unit/mcp_server/test_mcp_service_adapters.py
git commit -m "fix(mcp): guard phenopacket export by experiencer, not assertion (B2)"
```

### Task 9: Thread `family_history_findings` through service + REST + MCP

**Files:**
- Modify: `phentrieve/text_processing/full_text_service.py` (`_adapt_llm_aggregated_terms` ~355-465; surface family list + experiencer)
- Modify: `api/schemas/text_processing_schemas.py` (`TextProcessingResponseAPI` :228-263; add `family_history_findings`)
- Modify: `api/services/text_processing_execution.py` (response builder ~141-166; pass the family list through)
- Modify: `api/mcp/schemas.py` (add `family_history_findings` to `EXTRACT_SCHEMA` :43-47; roll `capabilities_version`), `api/mcp/projection.py` (preserve `experiencer`; project the family list :63-104), `api/mcp/shaping.py` (a second `enforce_budget` call for the new list at the retrieval tool call sites)
- Test: `tests/unit/api/test_family_findings_surface.py`, `tests/unit/mcp_server/test_mcp_family_projection.py`

**Interfaces:**
- Consumes: `LLMExtractionResult.family_history_findings` (Task 7).
- Produces: `family_history_findings` present on the REST `TextProcessingResponseAPI` and the MCP extract envelope; `experiencer` preserved on each projected term; the family list budgeted independently (not silently trimmed with, nor exempted from, the aggregated-terms budget).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/api/test_family_findings_surface.py`: build a `TextProcessingResponseAPI` with a `family_history_findings` entry and assert it round-trips (schema accepts + serializes the field). Create `tests/unit/mcp_server/test_mcp_family_projection.py`: assert `project_aggregated_terms_for_mcp` preserves `experiencer` on a term, and that the MCP extract envelope includes a `family_history_findings` key.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/unit/api/test_family_findings_surface.py tests/unit/mcp_server/test_mcp_family_projection.py -n 0`
Expected: FAIL (no `family_history_findings` field; `experiencer` dropped in projection).

- [ ] **Step 3: Implement**

- `full_text_service.py`: in `_adapt_llm_aggregated_terms`, carry `experiencer` into each adapted term dict, and surface the LLM result's `family_history_findings` on the service payload (adapt them with the same term-shaping used for proband terms).
- `text_processing_schemas.py`: add `family_history_findings: list[AggregatedHPOTermAPI] = Field(default_factory=list)` to `TextProcessingResponseAPI`; add an `experiencer: str | None = None` to `AggregatedHPOTermAPI` (:189-225).
- `text_processing_execution.py`: copy `family_history_findings` (and per-term `experiencer`) into the response builder (~141-166) -- stop dropping them.
- `api/mcp/schemas.py`: add `family_history_findings=_ARR` to `EXTRACT_SCHEMA`; bump the MCP `capabilities_version` (its cache-key contract).
- `api/mcp/projection.py`: preserve `experiencer` on each projected term (~72-74 area); add a projection for the family list.
- `api/mcp/shaping.py` / the two `tools/retrieval.py` call sites (`:171-172`, `:248-249`): add a second `enforce_budget(..., list_field="family_history_findings")` so the new list is budgeted deliberately.

- [ ] **Step 4: Run the tests + gates**

Run: `uv run pytest tests/unit/api/ tests/unit/mcp_server/ -n 0 && make check && make typecheck-fast`
Expected: passed; clean.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/text_processing/full_text_service.py api/schemas/text_processing_schemas.py api/services/text_processing_execution.py api/mcp/schemas.py api/mcp/projection.py api/mcp/shaping.py tests/unit/api/test_family_findings_surface.py tests/unit/mcp_server/test_mcp_family_projection.py
git commit -m "feat(api): surface family_history_findings + experiencer through service, REST, and MCP (B2)"
```

---

## B3 -- "X without Y" -> excluded term

### Task 10: Map the `negated_qualifier` phrase to a generated excluded finding

**Files:**
- Modify: `phentrieve/llm/pipeline.py` (qualifier merge ~1234; add a qualifier-resolution pass)
- Modify: `phentrieve/llm/types.py` (excluded-finding fields on the phenotype model)
- Test: `tests/unit/llm/test_qualifier_excluded_term.py`

**Interfaces:**
- Consumes: findings carrying `negated_qualifier="Y"`; the same retrieval used for proband terms; a confidence floor.
- Produces: when `Y` maps above the floor, a generated finding with `hpo_id`, `label`, `assertion="negated"` (canonical, `excluded: true`), `qualifier_surface_text="Y"`, `evidence`/`attribution_span`, `match_method="negated_qualifier_derived"`, `confidence >= floor`. Below the floor -> fall back to today's `negated_qualifier` metadata string (no generated term).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/llm/test_qualifier_excluded_term.py` with monkeypatched retrieval. Case A (maps above floor): a finding with `negated_qualifier="fever"` yields a generated finding with `assertion` canonicalizing to `negated`, `match_method == "negated_qualifier_derived"`, and `qualifier_surface_text == "fever"`. Case B (below floor / unmappable): the finding falls back to the `negated_qualifier` metadata string and produces NO generated term.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/llm/test_qualifier_excluded_term.py -n 0`
Expected: FAIL (no generated excluded term today; qualifier is a string only).

- [ ] **Step 3: Implement**

- Add the excluded-finding fields to the phenotype model in `types.py` (`qualifier_surface_text: str | None`, `match_method: str | None`, `attribution_span`/evidence already present or add).
- In `pipeline.py` near the qualifier merge (~1234), for each finding with a truthy `negated_qualifier`, run ONE retrieval call on the qualifier phrase; if the top result's confidence `>= floor` (reuse the pipeline's existing `min_confidence_for_aggregated`/retrieval-confidence config; do not invent a new knob if one exists), emit a generated finding with `assertion=canonicalize_assertion("absent")` (`= negated`), `match_method="negated_qualifier_derived"`, `qualifier_surface_text=<the Y span>`, and the evidence/attribution context. Otherwise keep the existing metadata-string behavior.

- [ ] **Step 4: Run the test + gates**

Run: `uv run pytest tests/unit/llm/test_qualifier_excluded_term.py tests/unit/llm/ -n 0 && make typecheck-fast`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/pipeline.py phentrieve/llm/types.py tests/unit/llm/test_qualifier_excluded_term.py
git commit -m "feat(llm): map negated_qualifier to a generated excluded finding (confidence-gated) (B3)"
```

### Task 11: Thread the excluded-finding shape through service + REST + MCP

**Files:**
- Modify: `phentrieve/text_processing/full_text_service.py` (carry the excluded-finding fields)
- Modify: `api/schemas/text_processing_schemas.py` + `api/services/text_processing_execution.py` (:141-166 -- add `negated_qualifier` / `qualifier_surface_text` / `match_method` / evidence to `AggregatedHPOTermAPI`, stop dropping them)
- Modify: `api/mcp/projection.py` (project the new fields)
- Test: `tests/unit/api/test_qualifier_excluded_surface.py`

**Interfaces:**
- Consumes: the generated excluded findings (Task 10).
- Produces: `qualifier_surface_text`, `match_method`, and evidence/provenance present on the REST + MCP outputs for a `negated_qualifier_derived` finding (today `text_processing_execution.py:141-166` drops all of them).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/api/test_qualifier_excluded_surface.py`: assert a `negated_qualifier_derived` finding round-trips through `AggregatedHPOTermAPI` with `match_method`, `qualifier_surface_text`, and evidence populated, and that the response builder copies them.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/api/test_qualifier_excluded_surface.py -n 0`
Expected: FAIL (fields absent / dropped).

- [ ] **Step 3: Implement**

Add the fields to `AggregatedHPOTermAPI` (`text_processing_schemas.py:189-225`), copy them in the response builder (`text_processing_execution.py:141-166`), carry them in `full_text_service` adapted terms, and project them in `api/mcp/projection.py`.

- [ ] **Step 4: Run the test + gates**

Run: `uv run pytest tests/unit/api/ tests/unit/mcp_server/ -n 0 && make check && make typecheck-fast`
Expected: passed; clean.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/text_processing/full_text_service.py api/schemas/text_processing_schemas.py api/services/text_processing_execution.py api/mcp/projection.py tests/unit/api/test_qualifier_excluded_surface.py
git commit -m "feat(api): surface negated_qualifier-derived excluded findings through REST + MCP (B3)"
```

---

## Guard + close

### Task 12: i18n negation-scope guard test

**Files:**
- Test: `tests/unit/llm/test_two_phase_prompt_negation_guard.py`

**Interfaces:**
- Consumes: `phentrieve/llm/prompts/templates/two_phase/*.yaml`.
- Produces: a test asserting every `two_phase/*.yaml` prompt template contains the negation-scope block and a `negated_qualifier` few-shot, so a future localized template cannot silently regress.

- [ ] **Step 1: Write the test**

Create `tests/unit/llm/test_two_phase_prompt_negation_guard.py`: glob `phentrieve/llm/prompts/templates/two_phase/*.yaml` (exclude the `_mapping*` files if they are a different prompt stage), load each, and assert its text contains the negation-scope rule and at least one `negated_qualifier` few-shot. Today only `en.yaml` exists (negation-scope at :29-30, qualifier few-shots :119-137), so the test passes now and guards future locales.

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/unit/llm/test_two_phase_prompt_negation_guard.py -n 0`
Expected: PASS (guards against future drift).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/llm/test_two_phase_prompt_negation_guard.py
git commit -m "test(llm): assert two_phase prompts keep the negation-scope block + qualifier few-shots"
```

### Task 13: Full-gate + partial acceptance (LLM benchmark gate owned, not silent)

- [ ] **Step 1: Repo-required trio**

Run: `make check && make typecheck-fast && make test`
Expected: all green.

- [ ] **Step 2: CI-parity + security before any push**

Run: `make ci-local && make security-python`
Expected: EXIT 0. Frontend (Task 3) is covered by `make ci-local`.

- [ ] **Step 3: Deterministic no-regression gate (present-only) -- runs here**

Regenerate the tiny extraction summaries and assert against the committed Phase 1 baselines:

```bash
PHENTRIEVE_DATA_ROOT_DIR=<repo>/data uv run phentrieve benchmark extraction run tests/data/extraction/tiny_extraction_test.json \
  --model FremyCompany/BioLORD-2023-M --scoring-mode present-only --output-dir results/_p2_present --no-bootstrap-ci
uv run phentrieve benchmark extraction assert-no-regression \
  --baseline tests/data/benchmarks/baselines/tiny_present_only_summary.json \
  --candidate results/_p2_present/extraction_summary.json
```
Expected: "No regression", exit 0. (B0-B3 change the LLM path, not the deterministic extractor, so this should be unaffected -- but the gate proves it.)

- [ ] **Step 4: Golden edge cases pass in strict**

Run the assertion golden fixture + the new B0-B3 unit/integration tests: `uv run pytest tests/unit/llm tests/unit/mcp_server tests/unit/api tests/unit/benchmark -n 0`
Expected: all passed.

- [ ] **Step 5: LLM mapping benchmark gate -- OWNED, Docker/Gemini (spec 9.1)**

This gate CANNOT run in a bare checkout (needs the Docker stack + Gemini key). Do NOT skip silently. Either:
  (a) run it in the Docker/Gemini environment and confirm no regression vs the committed LLM baseline, or
  (b) if unavailable, STOP and hand off: land B0-B3 with Steps 1-4 green, and record in the PR/verification that the LLM-benchmark gate is pending an operator run in Docker/Gemini before merge to `main`.

---

## Follow-on (Phase 3, not this plan)

- Vue **display**: `FullTextResponseReceipt.vue`, `AggregatedTermsView.vue`, `PhenotypeCollectionPanel.vue`, `ResultsDisplay.vue` -- a "Family history" section + an "excluded" chip on ruled-out terms; i18n locale keys (`make frontend-i18n-check`); component tests.
- Close #289 (`Closes #289` on that PR); final verification doc; CHANGELOG; coordinated CLI/API/Frontend version bump; release per the repo release process.

## Self-review notes (author)

- **Spec coverage:** B0 (spec 7 "B0") = Tasks 1-3; B1 = Tasks 4-5; B2 = Tasks 6-9; B3 = Tasks 10-11; i18n guard = Task 12; carrier surfaces (spec 7 "Carrier surfaces") = Tasks 9 + 11; gate machinery (spec 6.4/9.1) = Task 13. Vue display + close-out explicitly deferred to Phase 3 (scope boundary).
- **B1 partly pre-existing:** the dedup key `(term_id, experiencer, assertion)` and the `experiencer` plumbing through `LLMPhenotype`/dedup already exist but are *category-derived*; Tasks 4-5 make the model's own axes load-bearing. Not duplicate work.
- **Type consistency:** `canonicalize_assertion(raw) -> str` (canonical vocab) and `is_excluded(raw) -> bool` are used identically in Tasks 1/2/5/10; `family_history_findings: list[LLMPhenotype]` in Tasks 7/9; the excluded-finding fields (`qualifier_surface_text`, `match_method`) in Tasks 10/11.
- **Honest gaps for the implementer:** Tasks 6, 10 specify interfaces + tests + exact anchors but the impl adapts to the real `_retrieve_candidates` / qualifier-merge functions (read them first). Task 3 follows the existing frontend test layout. Task 13 Step 5 is the environment-gated LLM benchmark -- owned, not silent.
