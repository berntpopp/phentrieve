# Extraction Contract v2 -- Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the `.planning/` tree and make the extraction benchmark's no-regression gate real, deterministic, and un-foolable by a better polarity standard (opt-in `present-only` scoring), so the Phase 2 LLM behavior changes can be gated safely.

**Architecture:** Phase 0 is documentation/file hygiene only (no application code). Phase 1 adds a pure `normalize_for_scoring()` projection to the extraction scorer, seeds the bootstrap for determinism, threads a `--scoring-mode {strict, present-only}` option end-to-end (default `strict` = byte-for-byte-equal metrics), adds an assertion-labelled golden fixture, and ships an `assert-no-regression` CLI command that exits non-zero on a real regression.

**Tech Stack:** Python 3.11+, `uv`, Ruff, mypy, pytest (+ pytest-xdist), Typer CLI, dataclasses.

**Source spec:** `.planning/specs/2026-07-03-extraction-contract-v2-and-finalization-design.md` (Phases 0 and 1; sections 5, 6).

## Scope boundary (read first)

- **This plan:** Phase 0 (cleanup) + Phase 1 **extraction** benchmark safeguard.
- **Deferred to the Phase 2 plan:** threading `present-only` through the **LLM** benchmark
  (`llm_benchmark.py`) and committing LLM baselines. Rationale: that projection is coupled
  to the LLM benchmark's checkpoint/CLI/reporter flow and its gate only runs under
  Docker/Gemini, so it ships with the LLM behavior changes it gates (spec 6.5 / 9.1).
- **Also Phase 2+:** the LLM-behaviour golden cases (family-history routing, "X without Y"
  -> excluded Y). Phase 1's golden fixture only carries proband present/absent cases that the
  **deterministic** extractor is scored on.

## Global Constraints

- Dependency management: `uv` only; never `pip`.
- Typing: modern (`list[str]`, `str | None`); mypy targets Python 3.11.
- Format/lint: Ruff. All tests under `tests/`; never create `tests_new/` or new
  `tests/unit/api/` sub-package dirs. ASCII only.
- Required gates before claiming a task done: `make check`, `make typecheck-fast`,
  `make test`. Before any push: `make ci-local` + `make security-python`.
- `make test` runs under `pytest-xdist`; for single-file debugging use `uv run pytest ... -n 0`.
- Atomic commits (one deliverable = one commit); coverage-improving tests on all touched code.
- New CLI/scoring behavior must default to today's behavior (`scoring_mode="strict"`), so no
  committed benchmark number moves unless a caller opts in.

---

## Phase 0 -- Close-out & planning cleanup (no application code)

### Task 1: Create the documented-but-missing planning directories

**Files:**
- Create: `.planning/active/.gitkeep`, `.planning/drafts/.gitkeep`

**Interfaces:**
- Consumes: nothing.
- Produces: the `active/` and `drafts/` directories the README layout already documents.

- [ ] **Step 1: Check current state**

Run: `ls -d .planning/active .planning/drafts 2>&1`
Expected: `active/` already exists (this plan lives in it); `drafts/` is missing. The next
step is idempotent, so create both regardless.

- [ ] **Step 2: Create both with a keep file**

```bash
mkdir -p .planning/active .planning/drafts
: > .planning/active/.gitkeep
: > .planning/drafts/.gitkeep
```

- [ ] **Step 3: Verify**

Run: `ls -a .planning/active .planning/drafts`
Expected: each lists `.gitkeep`.

- [ ] **Step 4: Commit**

```bash
git add .planning/active/.gitkeep .planning/drafts/.gitkeep
git commit -m "chore(planning): create documented active/ and drafts/ dirs"
```

### Task 2: Quarantine pre-convention legacy files by rule + manifest

**Files:**
- Create: `.planning/archived/pre-convention/` (moved files land here) + `MANIFEST.md`
- Move: every top-level `*.md` in `.planning/completed/` and `.planning/archived/` whose
  basename does NOT match `YYYY-MM-DD-*` (64 files as of 2026-07-03), plus
  `.planning/archived/unified-output-format/`.

**Interfaces:**
- Consumes: nothing.
- Produces: a clean dated convention in `completed/`+`archived/`; an auditable manifest.

- [ ] **Step 1: Generate the selection list (dry run) and count it**

```bash
mkdir -p .planning/archived/pre-convention
find .planning/completed .planning/archived -maxdepth 1 -type f -name '*.md' \
  | grep -E '/[A-Z0-9_-]+\.md$' | grep -vE '/[0-9]{4}-[0-9]{2}-[0-9]{2}-' | sort > /tmp/legacy_list.txt
wc -l /tmp/legacy_list.txt
```
Expected: `64 /tmp/legacy_list.txt`. If the count differs, STOP and re-read the rule before moving anything.

- [ ] **Step 2: Write the manifest**

```bash
{
  echo "# Pre-convention planning files (quarantined 2026-07-03)"
  echo
  echo "Moved here by rule: top-level *.md in completed/ + archived/ not matching YYYY-MM-DD-*."
  echo "Git history is preserved (\`git log --follow\`). No deletions."
  echo
  sed 's#^#- #' /tmp/legacy_list.txt
  echo "- .planning/archived/unified-output-format/ (stray directory)"
} > .planning/archived/pre-convention/MANIFEST.md
```

- [ ] **Step 3: Move the files with `git mv`**

```bash
while IFS= read -r f; do git mv "$f" .planning/archived/pre-convention/; done < /tmp/legacy_list.txt
git mv .planning/archived/unified-output-format .planning/archived/pre-convention/unified-output-format
```

- [ ] **Step 4: Verify no pre-convention files remain at the top level**

Run: `find .planning/completed .planning/archived -maxdepth 1 -type f -name '*.md' | grep -E '/[A-Z0-9_-]+\.md$' | grep -vE '/[0-9]{4}-[0-9]{2}-[0-9]{2}-' | wc -l`
Expected: `0`.

- [ ] **Step 5: Commit**

```bash
git add -A .planning
git commit -m "chore(planning): quarantine 64 pre-convention files into archived/pre-convention"
```

### Task 3: Backfill PR #291's missing spec / plan / verification artifacts

**Files:**
- Create: `.planning/specs/2026-06-14-mcp-stabilization-design.md`
- Create: `.planning/completed/2026-06-14-mcp-stabilization-plan.md`
- Create: `.planning/analysis/2026-06-14-mcp-stabilization-verification.md`

**Interfaces:**
- Consumes: `.planning/analysis/2026-06-14-mcp-stabilization-plan.md` (source analysis);
  PR #291 body (`gh pr view 291 --json body`); the 2026-07-03 deep re-verification results.
- Produces: parity with the hardening (#288) and remediation (#290) efforts.

- [ ] **Step 1: Write the retro design spec**

Create `.planning/specs/2026-06-14-mcp-stabilization-design.md` with: a metadata block
(Date 2026-06-14, retroactive), a one-paragraph "why" pointing to the analysis doc, and the
14-finding table (B1, B2, LLM-1, LLM-2, R1, R2, B3, D4, D3, D1, D2, R3, Q1, B4) with each
finding's one-line fix -- copied from the analysis doc's section 6 roadmap. Mark
`Status: SHIPPED in PR #291 (v0.24.0)`.

- [ ] **Step 2: Write the completed execution plan**

Create `.planning/completed/2026-06-14-mcp-stabilization-plan.md` from `gh pr view 291`'s body
(each finding -> commit hash mapping is in the PR description). One row per finding:
`ID | commit | one-line`.

- [ ] **Step 3: Write the verification record**

Create `.planning/analysis/2026-06-14-mcp-stabilization-verification.md` recording the
2026-07-03 deep re-verification: all 14 findings verified against current code, all shipped
in v0.24.0/0.24.1, plus the residuals that seeded this v2 effort (advisory axes,
family-history dropped, qualifier metadata-only) and the open-#289 note.

- [ ] **Step 4: Verify the three files exist and are non-empty**

Run: `wc -l .planning/specs/2026-06-14-mcp-stabilization-design.md .planning/completed/2026-06-14-mcp-stabilization-plan.md .planning/analysis/2026-06-14-mcp-stabilization-verification.md`
Expected: three files, each > 20 lines.

- [ ] **Step 5: Commit**

```bash
git add .planning/specs/2026-06-14-mcp-stabilization-design.md .planning/completed/2026-06-14-mcp-stabilization-plan.md .planning/analysis/2026-06-14-mcp-stabilization-verification.md
git commit -m "docs(planning): backfill PR #291 stabilization spec/plan/verification"
```

### Task 4: Refresh the planning index (README + STATUS)

**Files:**
- Modify: `.planning/README.md` (Recent Analysis, Recently Completed, Current Active Work,
  Current Specs sections)
- Modify: `.planning/STATUS.md`

**Interfaces:**
- Consumes: Tasks 1-3 outputs.
- Produces: an index that matches the real tree.

- [ ] **Step 1: Add the stabilization entries + this active effort to README**

Edit `.planning/README.md`: under "Recent Analysis" add the stabilization verification;
under "Recently Completed" add the stabilization plan (PR #291); change
"Current Active Work: None" to reference
`.planning/active/2026-07-03-extraction-contract-v2-phase-0-1-plan.md`; note the
`archived/pre-convention/` quarantine in the "Archived And Superseded" section.

- [ ] **Step 2: Reconcile STATUS.md**

Edit `.planning/STATUS.md` so it reflects: stabilization shipped (v0.24.1); extraction
contract v2 active (this plan).

- [ ] **Step 3: Verify the README no longer claims "Active: None" and mentions stabilization**

Run: `grep -n "extraction-contract-v2\|stabilization" .planning/README.md`
Expected: at least two matches; no remaining "Current Active Work" set to "None".

- [ ] **Step 4: Commit**

```bash
git add .planning/README.md .planning/STATUS.md
git commit -m "docs(planning): refresh index for stabilization + contract-v2 active work"
```

---

## Phase 1 -- Extraction benchmark safeguard + gate

### Task 5: Add the `normalize_for_scoring` projection helper

**Files:**
- Modify: `phentrieve/evaluation/extraction_metrics.py` (add helper near top, after imports)
- Test: `tests/unit/evaluation/test_normalize_for_scoring.py`

**Interfaces:**
- Consumes: `ExtractionResult` (`phentrieve/evaluation/_extraction_types.py:6-12`,
  `dataclass(doc_id: str, predicted: list[tuple[str,str]], gold: list[tuple[str,str]])`).
- Produces: `normalize_for_scoring(results: list[ExtractionResult], mode: str = "strict") -> list[ExtractionResult]`
  and `PROBAND_PRESENT: frozenset[str]`. Used by Task 7 (benchmark wiring) and Task 9 (assert cmd).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/evaluation/test_normalize_for_scoring.py`:

```python
import pytest

from phentrieve.evaluation._extraction_types import ExtractionResult
from phentrieve.evaluation.extraction_metrics import normalize_for_scoring


def _r():
    return [
        ExtractionResult(
            doc_id="d1",
            predicted=[("HP:1", "PRESENT"), ("HP:2", "ABSENT"), ("HP:3", "FAMILY_HISTORY")],
            gold=[("HP:1", "PRESENT"), ("HP:2", "ABSENT")],
        )
    ]


def test_strict_mode_is_identity():
    results = _r()
    assert normalize_for_scoring(results, "strict") is results


def test_present_only_drops_non_present_from_pred_and_gold():
    out = normalize_for_scoring(_r(), "present-only")
    assert out[0].predicted == [("HP:1", "PRESENT")]
    assert out[0].gold == [("HP:1", "PRESENT")]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        normalize_for_scoring(_r(), "bogus")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/evaluation/test_normalize_for_scoring.py -n 0`
Expected: FAIL (ImportError: cannot import name `normalize_for_scoring`).

- [ ] **Step 3: Implement the helper**

In `phentrieve/evaluation/extraction_metrics.py`, after the existing
`from phentrieve.evaluation._extraction_types import ExtractionResult` import, add:

```python
PROBAND_PRESENT: frozenset[str] = frozenset({"PRESENT"})


def normalize_for_scoring(
    results: list[ExtractionResult], mode: str = "strict"
) -> list[ExtractionResult]:
    """Project scored results for a given scoring mode.

    ``strict`` (default) is the identity -- byte-identical scored inputs, so committed
    benchmark metrics are reproduced exactly. ``present-only`` filters both predicted and
    gold to PRESENT terms and re-stamps, collapsing the assertion-strict tuple comparison
    into an id-level proband-present comparison (fair against polarity-blind gold).
    """
    if mode == "strict":
        return results
    if mode != "present-only":
        raise ValueError(f"Unknown scoring mode: {mode!r} (expected strict|present-only)")
    return [
        ExtractionResult(
            doc_id=r.doc_id,
            predicted=[(hid, "PRESENT") for hid, a in r.predicted if a in PROBAND_PRESENT],
            gold=[(hid, "PRESENT") for hid, a in r.gold if a in PROBAND_PRESENT],
        )
        for r in results
    ]
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/unit/evaluation/test_normalize_for_scoring.py -n 0`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/evaluation/extraction_metrics.py tests/unit/evaluation/test_normalize_for_scoring.py
git commit -m "feat(benchmark): add normalize_for_scoring present-only projection helper"
```

### Task 6: Seed the bootstrap for deterministic CIs

**Files:**
- Modify: `phentrieve/evaluation/extraction_metrics.py`
  (`CorpusExtractionMetrics.bootstrap_confidence_intervals`, ~line 419-469)
- Modify: `phentrieve/benchmark/extraction_benchmark.py` (`ExtractionConfig` ~line 51-70;
  the bootstrap call ~line 313-316)
- Test: `tests/unit/evaluation/test_bootstrap_seed.py`

**Interfaces:**
- Consumes: `CorpusExtractionMetrics(averaging).bootstrap_confidence_intervals(results, n_bootstrap, confidence_level)`.
- Produces: same method with an added keyword `seed: int | None = None`; a new
  `ExtractionConfig.bootstrap_seed: int | None = 12345`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/evaluation/test_bootstrap_seed.py`:

```python
from phentrieve.evaluation._extraction_types import ExtractionResult
from phentrieve.evaluation.extraction_metrics import CorpusExtractionMetrics


def _results():
    return [
        ExtractionResult(f"d{i}", [(f"HP:{i}", "PRESENT")], [(f"HP:{i}", "PRESENT")])
        for i in range(10)
    ] + [ExtractionResult("dx", [("HP:99", "PRESENT")], [("HP:1", "PRESENT")])]


def test_seeded_bootstrap_is_deterministic():
    m = CorpusExtractionMetrics()
    a = m.bootstrap_confidence_intervals(_results(), n_bootstrap=200, seed=7)
    b = m.bootstrap_confidence_intervals(_results(), n_bootstrap=200, seed=7)
    assert a == b
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/evaluation/test_bootstrap_seed.py -n 0`
Expected: FAIL (TypeError: unexpected keyword argument `seed`).

- [ ] **Step 3: Add the `seed` parameter and use a local RNG**

In `bootstrap_confidence_intervals`, change the signature to add `seed: int | None = None`,
and replace the module-level `random.choices(...)` call with a local RNG:

```python
    def bootstrap_confidence_intervals(
        self,
        results: list[ExtractionResult],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        ...
        rng = random.Random(seed)
        for _ in range(n_bootstrap):
            sample = rng.choices(results, k=len(results))  # noqa: S311
            metrics = self._compute_micro(sample)
            ...
```

- [ ] **Step 4: Thread a seed from the benchmark config**

In `phentrieve/benchmark/extraction_benchmark.py`, add to `ExtractionConfig`:

```python
    bootstrap_seed: int | None = 12345
```

and update the bootstrap call in `run_benchmark` (~line 314):

```python
            ci = evaluator.bootstrap_confidence_intervals(
                results, n_bootstrap=config.bootstrap_samples, seed=config.bootstrap_seed
            )
```

- [ ] **Step 5: Run the test + typecheck**

Run: `uv run pytest tests/unit/evaluation/test_bootstrap_seed.py -n 0 && make typecheck-fast`
Expected: 1 passed; mypy clean.

- [ ] **Step 6: Commit**

```bash
git add phentrieve/evaluation/extraction_metrics.py phentrieve/benchmark/extraction_benchmark.py tests/unit/evaluation/test_bootstrap_seed.py
git commit -m "feat(benchmark): seed bootstrap CIs for deterministic no-regression gating"
```

### Task 7: Wire `scoring_mode` through config, choke point, and CLI

**Files:**
- Modify: `phentrieve/benchmark/extraction_benchmark.py` (`ExtractionConfig`; `run_benchmark`
  after the doc loop, before metrics ~line 296)
- Modify: `phentrieve/benchmark/extraction_cli.py` (`run` command options + config build)
- Test: `tests/unit/benchmark/test_scoring_mode_wiring.py`

**Interfaces:**
- Consumes: `normalize_for_scoring` (Task 5); `ExtractionBenchmark.run_benchmark`.
- Produces: `ExtractionConfig.scoring_mode: str = "strict"`; CLI `--scoring-mode`.

- [ ] **Step 1: Write the failing test (monkeypatched extractor, no model needed)**

Create `tests/unit/benchmark/test_scoring_mode_wiring.py`:

```python
import json
from pathlib import Path

from phentrieve.benchmark.extraction_benchmark import ExtractionBenchmark, ExtractionConfig


def _fixture(tmp_path: Path) -> Path:
    payload = {
        "documents": [
            {
                "id": "d1",
                "text": "no fever",
                "gold_hpo_terms": [{"id": "HP:0001945", "assertion": "ABSENT"}],
            }
        ]
    }
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(payload))
    return p


def test_present_only_drops_absent_from_metrics(tmp_path, monkeypatch):
    # Extractor predicts the same term as ABSENT -> strict TP, present-only drops both sides.
    cfg = ExtractionConfig(scoring_mode="present-only", bootstrap_ci=False)
    bench = ExtractionBenchmark("BAAI/bge-m3", config=cfg)
    monkeypatch.setattr(
        bench.extractor, "extract", lambda text: [("HP:0001945", "ABSENT")]
    )
    metrics = bench.run_benchmark(_fixture(tmp_path), tmp_path / "out")
    # present-only removed the only (absent) pair from both sides -> no tp/fp/fn -> zeros
    assert metrics.micro["f1"] == 0.0


def test_strict_scores_the_absent_tuple(tmp_path, monkeypatch):
    cfg = ExtractionConfig(scoring_mode="strict", bootstrap_ci=False)
    bench = ExtractionBenchmark("BAAI/bge-m3", config=cfg)
    monkeypatch.setattr(
        bench.extractor, "extract", lambda text: [("HP:0001945", "ABSENT")]
    )
    metrics = bench.run_benchmark(_fixture(tmp_path), tmp_path / "out2")
    assert metrics.micro["f1"] == 1.0
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/benchmark/test_scoring_mode_wiring.py -n 0`
Expected: FAIL (`ExtractionConfig` has no `scoring_mode`, or metrics score the absent tuple in present-only mode).

- [ ] **Step 3: Add the config field**

In `ExtractionConfig` (extraction_benchmark.py), add:

```python
    scoring_mode: str = "strict"  # strict | present-only
```

- [ ] **Step 4: Apply the projection at the choke point**

In `run_benchmark`, immediately after the `for idx, doc in ...` loop finishes populating
`results` and BEFORE `evaluator = CorpusExtractionMetrics(...)` (~line 296), insert:

```python
        from phentrieve.evaluation.extraction_metrics import normalize_for_scoring

        results = normalize_for_scoring(results, config.scoring_mode)
```

(One reassignment; metrics, ontology-aware metrics, and bootstrap all consume `results`, so
this single call covers every downstream path.)

- [ ] **Step 5: Add the CLI option**

In `extraction_cli.py` `run`, add an option next to `averaging`:

```python
    scoring_mode: str = typer.Option(
        "strict",
        "--scoring-mode",
        help="Scoring mode: strict (assertion-aware, default) or present-only "
        "(proband-present id-level; for legacy polarity-blind corpora).",
    ),
```

and add `scoring_mode=scoring_mode,` to the `ExtractionConfig(...)` construction.

- [ ] **Step 6: Run the tests + gates**

Run: `uv run pytest tests/unit/benchmark/test_scoring_mode_wiring.py -n 0 && make check && make typecheck-fast`
Expected: 2 passed; ruff + mypy clean.

- [ ] **Step 7: Commit**

```bash
git add phentrieve/benchmark/extraction_benchmark.py phentrieve/benchmark/extraction_cli.py tests/unit/benchmark/test_scoring_mode_wiring.py
git commit -m "feat(benchmark): thread scoring_mode (strict|present-only) through config + CLI"
```

### Task 8: Add the assertion-labelled golden fixture + loader test

**Files:**
- Create: `tests/data/benchmarks/en/assertion_edge_cases.json`
- Test: `tests/unit/benchmark/test_assertion_golden_fixture.py`

**Interfaces:**
- Consumes: `parse_gold_terms` (`phentrieve/benchmark/data_loader.py:188-201`, reads
  `gold_hpo_terms[].assertion`) and `load_benchmark_data`.
- Produces: a document-payload fixture with proband PRESENT/ABSENT gold (deterministic-path
  cases only; family/qualifier cases are Phase 2).

- [ ] **Step 1: Create the fixture (document-payload format, `assertion` field NOT `assertion_status`)**

Create `tests/data/benchmarks/en/assertion_edge_cases.json`:

```json
{
  "metadata": {"dataset_name": "assertion_edge_cases", "source": "handwritten"},
  "documents": [
    {
      "id": "no_nystagmus",
      "text": "There is no nystagmus.",
      "gold_hpo_terms": [{"id": "HP:0000639", "assertion": "ABSENT"}]
    },
    {
      "id": "id_without_regression",
      "text": "Severe intellectual disability without regression of milestones.",
      "gold_hpo_terms": [{"id": "HP:0010864", "assertion": "PRESENT"}]
    },
    {
      "id": "plain_present",
      "text": "The patient has seizures.",
      "gold_hpo_terms": [{"id": "HP:0001250", "assertion": "PRESENT"}]
    }
  ]
}
```

- [ ] **Step 2: Write the failing loader test**

Create `tests/unit/benchmark/test_assertion_golden_fixture.py`:

```python
from pathlib import Path

from phentrieve.benchmark.data_loader import load_benchmark_data, parse_gold_terms

FIXTURE = Path("tests/data/benchmarks/en/assertion_edge_cases.json")


def test_fixture_parses_assertion_field_not_assertion_status():
    data = load_benchmark_data(FIXTURE, dataset="all")
    docs = {d["id"]: d for d in data["documents"]}
    gold = dict(parse_gold_terms(docs["no_nystagmus"]["gold_hpo_terms"]))
    assert gold == {"HP:0000639": "ABSENT"}  # proves .assertion is read, default not applied


def test_present_case_reads_present():
    data = load_benchmark_data(FIXTURE, dataset="all")
    docs = {d["id"]: d for d in data["documents"]}
    gold = dict(parse_gold_terms(docs["plain_present"]["gold_hpo_terms"]))
    assert gold == {"HP:0001250": "PRESENT"}
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/unit/benchmark/test_assertion_golden_fixture.py -n 0`
Expected: PASS (the fixture + existing loader already satisfy it -- this test guards the
`assertion`-not-`assertion_status` contract against future drift).

- [ ] **Step 4: Commit**

```bash
git add tests/data/benchmarks/en/assertion_edge_cases.json tests/unit/benchmark/test_assertion_golden_fixture.py
git commit -m "test(benchmark): add assertion-labelled golden fixture + loader contract test"
```

### Task 9: Add the `assert-no-regression` CLI command (exit non-zero on regression)

**Files:**
- Modify: `phentrieve/benchmark/extraction_cli.py` (new `assert_no_regression` command)
- Test: `tests/unit/benchmark/test_assert_no_regression.py`

**Interfaces:**
- Consumes: two `extraction_summary.json` files (baseline + candidate) -- each has
  `micro_f1`, `micro_precision`, `micro_recall` keys (see `extraction_benchmark._save_results`
  ~line 710-722).
- Produces: CLI `phentrieve benchmark extraction assert-no-regression --baseline <f> --candidate <f> [--tolerance 0.0]`;
  a pure `_regressions(baseline: dict, candidate: dict, tolerance: float) -> list[str]` helper.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/benchmark/test_assert_no_regression.py`:

```python
from phentrieve.benchmark.extraction_cli import _regressions


def test_no_regression_when_candidate_equal_or_better():
    base = {"micro_f1": 0.80, "micro_precision": 0.80, "micro_recall": 0.80}
    cand = {"micro_f1": 0.80, "micro_precision": 0.82, "micro_recall": 0.80}
    assert _regressions(base, cand, tolerance=0.0) == []


def test_regression_detected_on_f1_drop():
    base = {"micro_f1": 0.80, "micro_precision": 0.80, "micro_recall": 0.80}
    cand = {"micro_f1": 0.78, "micro_precision": 0.80, "micro_recall": 0.80}
    out = _regressions(base, cand, tolerance=0.0)
    assert any("micro_f1" in r for r in out)


def test_tolerance_absorbs_small_noise():
    base = {"micro_f1": 0.80, "micro_precision": 0.80, "micro_recall": 0.80}
    cand = {"micro_f1": 0.795, "micro_precision": 0.80, "micro_recall": 0.80}
    assert _regressions(base, cand, tolerance=0.01) == []
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/unit/benchmark/test_assert_no_regression.py -n 0`
Expected: FAIL (ImportError: cannot import name `_regressions`).

- [ ] **Step 3: Implement the helper + command**

In `phentrieve/benchmark/extraction_cli.py` add:

```python
_GATED_METRICS = ("micro_f1", "micro_precision", "micro_recall")


def _regressions(
    baseline: dict[str, Any], candidate: dict[str, Any], tolerance: float
) -> list[str]:
    """Return a human-readable line per metric that dropped beyond tolerance."""
    out: list[str] = []
    for key in _GATED_METRICS:
        base = float(baseline.get(key, 0.0))
        cand = float(candidate.get(key, 0.0))
        if cand < base - tolerance:
            out.append(f"{key}: {cand:.4f} < baseline {base:.4f} (tol {tolerance})")
    return out


@app.command(name="assert-no-regression")
def assert_no_regression(
    baseline: Path = typer.Option(..., help="Baseline extraction_summary.json"),
    candidate: Path = typer.Option(..., help="Candidate extraction_summary.json"),
    tolerance: float = typer.Option(0.0, help="Allowed absolute drop per metric"),
):
    """Exit non-zero if the candidate regresses any gated metric vs the baseline."""
    base = json.loads(baseline.read_text())
    cand = json.loads(candidate.read_text())
    regressions = _regressions(base, cand, tolerance)
    if regressions:
        console.print("[red]REGRESSION[/red]")
        for line in regressions:
            console.print(f"  {line}")
        raise typer.Exit(1)
    console.print("[green]No regression[/green]")
```

- [ ] **Step 4: Run the test + gates**

Run: `uv run pytest tests/unit/benchmark/test_assert_no_regression.py -n 0 && make check && make typecheck-fast`
Expected: 3 passed; clean.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/benchmark/extraction_cli.py tests/unit/benchmark/test_assert_no_regression.py
git commit -m "feat(benchmark): add assert-no-regression CLI gate (exits non-zero on drop)"
```

### Task 10: Generate + commit the extraction baselines (execution-time)

**Files:**
- Create: `tests/data/benchmarks/baselines/tiny_strict_summary.json`
- Create: `tests/data/benchmarks/baselines/tiny_present_only_summary.json`
- Test: `tests/unit/benchmark/test_baselines_committed.py`

**Interfaces:**
- Consumes: the `run` CLI (Task 7) + `assert-no-regression` (Task 9). Requires the HPO index
  + embedding model available in the environment (data bundle symlinked, per repo notes).
- Produces: committed baseline fixtures + a self-consistency test.

- [ ] **Step 1: Generate strict + present-only baselines on the small in-repo fixture**

```bash
mkdir -p tests/data/benchmarks/baselines results/_tmp_strict results/_tmp_present
uv run phentrieve benchmark extraction run tests/data/extraction/tiny_extraction_test.json \
  --scoring-mode strict --output-dir results/_tmp_strict --no-bootstrap-ci
uv run phentrieve benchmark extraction run tests/data/extraction/tiny_extraction_test.json \
  --scoring-mode present-only --output-dir results/_tmp_present --no-bootstrap-ci
cp results/_tmp_strict/extraction_summary.json tests/data/benchmarks/baselines/tiny_strict_summary.json
cp results/_tmp_present/extraction_summary.json tests/data/benchmarks/baselines/tiny_present_only_summary.json
```

If the environment lacks the HPO index/model, STOP and run this task where the data bundle is
available (see spec 9.1); do not fabricate baseline numbers.

- [ ] **Step 2: Write a self-consistency test (a baseline never regresses itself)**

Create `tests/unit/benchmark/test_baselines_committed.py`:

```python
import json
from pathlib import Path

from phentrieve.benchmark.extraction_cli import _regressions

BASE = Path("tests/data/benchmarks/baselines")


def test_baseline_files_exist_and_have_metrics():
    for name in ("tiny_strict_summary.json", "tiny_present_only_summary.json"):
        data = json.loads((BASE / name).read_text())
        assert "micro_f1" in data


def test_baseline_does_not_regress_itself():
    data = json.loads((BASE / "tiny_present_only_summary.json").read_text())
    assert _regressions(data, data, tolerance=0.0) == []
```

- [ ] **Step 3: Run the test + confirm the assert command is green against itself**

```bash
uv run pytest tests/unit/benchmark/test_baselines_committed.py -n 0
uv run phentrieve benchmark extraction assert-no-regression \
  --baseline tests/data/benchmarks/baselines/tiny_present_only_summary.json \
  --candidate tests/data/benchmarks/baselines/tiny_present_only_summary.json
```
Expected: tests pass; command prints "No regression" and exits 0.

- [ ] **Step 4: Commit**

```bash
git add tests/data/benchmarks/baselines tests/unit/benchmark/test_baselines_committed.py
git commit -m "test(benchmark): commit tiny strict+present-only baselines as the regression fence"
```

### Task 11: Full-gate check and Phase 1 close

- [ ] **Step 1: Run the repo-required trio**

Run: `make check && make typecheck-fast && make test`
Expected: all green (new tests included).

- [ ] **Step 2: Run the CI-parity gate before any push**

Run: `make ci-local && make security-python`
Expected: EXIT 0.

- [ ] **Step 3: Confirm the default path is unchanged (strict == today)**

Run: `uv run pytest tests/unit/evaluation/test_normalize_for_scoring.py::test_strict_mode_is_identity -n 0`
Expected: PASS (identity guarantee -> committed benchmark metrics unchanged for default callers).

---

## Follow-on (next plan, not this one)

- **Phase 2 plan:** B0 canonical vocabulary (`canonicalize_assertion()` shared helper) ->
  B1 assertion load-bearing (carry experiencer+assertion end-to-end) -> B2 family-history
  list (collect->map->emit->exclude; experiencer-based phenopacket guard) -> B3 qualifier
  -> excluded term; plus threading `present-only` through the LLM benchmark and committing
  LLM baselines; each change gated by §6.4 (present-only no-regression via the Task 9 command
  + golden cases + LLM benchmark via Docker/Gemini).
- **Phase 3 plan:** REST schema + `FullTextResponseReceipt.vue` / `AggregatedTermsView.vue` /
  `PhenotypeCollectionPanel.vue` display + i18n; close #289; verification doc; release.

## Self-review notes (author)

- Spec coverage: Phase 0 (spec 5) = Tasks 1-4. Phase 1 (spec 6.2/6.3/6.5) = Tasks 5-11.
  The LLM-benchmark present-only thread (spec 6.5 bullet 4) is explicitly deferred to Phase 2
  with rationale (scope boundary above).
- Type consistency: `normalize_for_scoring(results, mode) -> list[ExtractionResult]` and
  `_regressions(baseline, candidate, tolerance) -> list[str]` are used with identical
  signatures in Tasks 5/7/9/10.
- Defaults: `scoring_mode="strict"` and `normalize_for_scoring(..., "strict") is results`
  guarantee no committed number moves for existing callers (Task 11 Step 3).
