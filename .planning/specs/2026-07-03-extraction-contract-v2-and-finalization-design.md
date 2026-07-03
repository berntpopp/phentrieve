# Phentrieve -- LLM Extraction Contract v2 + Stabilization Finalization Design

- Date: 2026-07-03
- Author: Claude (Opus 4.8), acting as senior full-stack engineer
- Source review: `.planning/analysis/2026-06-14-mcp-stabilization-plan.md` (deep re-verification, 2026-07-03)
- Prior work: PR #291 (MCP stabilization, all 14 findings shipped in v0.24.0/0.24.1)
- Tracks issue: #289 (LLM assertion polarity / "X without Y" / family-history)
- Status: APPROVED to plan + execute (decisions locked in section 3)

## 1. Why this exists

A deep re-verification of the MCP stabilization plan (four code-grounded passes over
current `main`) found that **all 14 findings shipped** and are test-pinned -- including
LLM-1/LLM-2, which the two 06-14 verification docs had recorded as "deferred". The plan
is fully executed. What remains are **residual behaviors** the shipped fixes left on the
table, plus **planning-hygiene debt**:

- The LLM `experiencer`/`assertion` axes exist on the schema but are **advisory**:
  polarity is re-derived from the legacy `category` enum
  (`phentrieve/llm/pipeline_phase2.py:273-279`), so the model's own `assertion` is not
  load-bearing.
- Family-history mentions are **silently dropped** (removed from actionable at
  `phentrieve/llm/pipeline.py:340-343`, and from export at
  `api/mcp/service_adapters.py:381-384`) -- pedigree information vanishes with no trace.
- `negated_qualifier` ("X without Y") is captured as a **metadata string only**
  (`phentrieve/text_processing/full_text_service.py:457-463`); Y is never mapped, so it
  is not a machine-actionable exclusion.
- Issue #289 is still **open** despite being resolved by PR #291 (no `Closes #289`
  keyword); PR #291 has **no spec / completed-plan / verification** artifacts in
  `.planning/` (only its analysis doc, committed 3 weeks late); the `.planning/README.md`
  index omits it; and `completed/` + `archived/` carry ~40 pre-convention ALL-CAPS files.

This design finalizes all of the above as a single, phased effort.

## 2. Goal

Make the LLM extraction output **fully correct and consistent across all three consumers**
(MCP, REST, Vue), gated by a **benchmark regime that cannot be fooled into penalizing a
better standard**, and leave the `.planning/` tree clean and truthful.

## 3. Decisions (locked)

1. **Assertion becomes load-bearing.** Consume the model-emitted `assertion` as the
   polarity signal; retire the `category`-derived polarity. `category` remains only as a
   derived/compat field.
2. **Family history -> its own list.** Carry `experiencer` through aggregation into a
   dedicated `family_history_findings` list (mapped HPO terms), kept **out** of the
   proband phenopacket.
3. **"X without Y" -> excluded term.** Retrieve the `negated_qualifier` phrase, map Y to
   an HPO id, and emit it as an **excluded** (`assertion=absent`) finding ->
   `excluded: true` PhenotypicFeature in the phenopacket. Confidence-gated; falls back to
   today's metadata string when Y does not map cleanly.
4. **Surface everywhere now.** The two new outputs (family-history list + excluded terms)
   are wired through the shared service, REST schemas, **and** the Vue UI in this effort.
5. **Benchmarks: proper + no regression.** Current corpora are good but polarity-blind and
   do not cover these edge cases. We must not regress on them, and must not be penalized
   for a better standard. Safeguard = a `present-only` scoring mode + a new assertion-
   labeled golden mini-corpus for the edge cases (a full deterministic-corpus re-annotation
   is **out of scope** this round).
6. **Quarantine, don't delete, the legacy planning tree.**
7. **One spec, phased plan.** Both tracks live in this document; the implementation plan
   phases them.

## 4. Scope & decomposition

| Phase | Deliverable | Risk | Ships |
|---|---|---|:--:|
| **0** | Close-out & planning cleanup | none (docs) | first, independent |
| **1** | Benchmark safeguard + baselines + golden set | low (test infra) | before Phase 2 |
| **2** | LLM contract v2 (B1 assertion, B2 family, B3 qualifier) | **high** | gated by Phase 1 |
| **3** | Consumer surface (REST + Vue) + close-out | medium | last |

Phase 1 **must precede** Phase 2: we cannot gate the behavior changes without the
safeguard and baselines in place first.

## 5. Phase 0 -- Close-out & planning cleanup (no behavior change)

- **Backfill PR #291 artifacts** so it matches the hardening/remediation efforts:
  - `specs/2026-06-14-mcp-stabilization-design.md` (retroactive, from the analysis doc)
  - `completed/2026-06-14-mcp-stabilization-plan.md` (the 14-finding execution record)
  - `analysis/2026-06-14-mcp-stabilization-verification.md` (the 2026-07-03 deep re-verification)
- **Refresh the index:** update `.planning/README.md` (add stabilization + #291; correct
  "Current Active Work: None"); reconcile top-level `STATUS.md`.
- **Quarantine legacy files:** move the pre-convention ALL-CAPS files in `completed/` and
  `archived/`, plus the stray `archived/unified-output-format/`, into
  `.planning/archived/pre-convention/` with a one-line index README. Git history is
  preserved; the dated convention becomes clean. No deletions.
- **Issue #289** stays open as the tracking anchor; it is closed at the end of Phase 3
  (with a `Closes #289` on that PR).

Gate: `make check`. This phase touches no application code.

## 6. Phase 1 -- Benchmark safeguard (the crux)

### 6.1 Verified problem (2026-07-03)

The extraction scorer is **already assertion-strict**: it compares full
`(hpo_id, assertion)` tuples in every metric family, not term-ID sets.

- `phentrieve/evaluation/_extraction_types.py:7-12` -- `predicted`/`gold` are
  `list[tuple[str, str]]` = `(hpo_id, assertion)`.
- `phentrieve/evaluation/extraction_metrics.py:317-332` (`_compute_micro`) and `:66-79`
  (`_doc_metrics`) do `set(predicted) & set(gold)` on tuples.
- `phentrieve/evaluation/ontology_matching.py:55,142-162,213-217,226-246` -- exact and
  partial matches are grouped/keyed by `annotation[1]` (assertion); cross-assertion pairs
  never match.

The gold is **de-facto 100% PRESENT**: across every `tests/data/**/annotations/*.json`,
5215 items are `affirmed`, exactly **1** is `negated`, and **0** are `family_history` or
`uncertain`. Predicted tuples, however, already carry `ABSENT`/`UNCERTAIN` because
assertion detection is on by default (`extraction_benchmark.py:62,179,194-200`;
`extraction_cli.py:42-44`).

Consequence (both confirmed against the code path):
- **"no family history of X"** reclassified as family -> `(X,PRESENT)` in gold vs
  `(X,FAMILY_HISTORY)` in pred -> **+1 FN and +1 FP** (double penalty), when X is a gold
  term (`extraction_metrics.py:328-329`).
- **"X without Y"** -> `(Y,ABSENT)` predicted vs `(Y,PRESENT)`-or-absent gold -> FP (and
  FN if gold listed Y as present). This **already** penalizes correct negation today.

There is **no** present-only / assertion-filtered scoring mode; the only adjacent lever,
`--include-assertions/--no-include-assertions` (`extraction_cli.py:42-44`), disables
detection upstream and cripples the model. The LLM benchmark scores via
`phentrieve/evaluation/assertion_metrics.py` (joint-F1 `:26-43`, stratified filters
`:71-101`) -- same tuple/assertion basis, so it needs the same safeguard.

### 6.2 Safeguard: `--scoring-mode {strict, present-only}` (default `strict`)

Pure helper in `phentrieve/evaluation/extraction_metrics.py`:

```python
PROBAND_PRESENT = {"PRESENT"}

def normalize_for_scoring(results, mode="strict"):
    if mode == "strict":
        return results  # identity -> byte-identical reproduction of current numbers
    return [
        ExtractionResult(
            doc_id=r.doc_id,
            predicted=[(hid, "PRESENT") for hid, a in r.predicted if a in PROBAND_PRESENT],
            gold=[(hid, "PRESENT") for hid, a in r.gold if a in PROBAND_PRESENT],
        )
        for r in results
    ]
```

- Applied at the single choke point `extraction_benchmark.py:257` (before both
  `CorpusExtractionMetrics.calculate_*` and `calculate_ontology_aware_metrics`), so strict
  and ontology-aware paths are both covered by one call.
- `present-only` filters to `PRESENT` and re-stamps, collapsing the strict tuple
  comparison into an **id-level proband-present** comparison: absent/family predictions
  can no longer become false positives, and a polarity-blind (all-PRESENT) gold compares
  fairly against the improved model.
- Threaded via `scoring_mode: str = "strict"` on `ExtractionConfig` +
  `--scoring-mode` option in `extraction_cli.py`. Mirror the same normalization into the
  LLM benchmark scoring path (`assertion_metrics.py`).

**Why legacy numbers are provably reproduced:** with the default `strict`,
`normalize_for_scoring` returns the input list unchanged, so identical `ExtractionResult`s
flow into identical, untouched metric code. A no-regression unit test asserts
`normalize_for_scoring(results, "strict") is results` and that strict output equals a
stored baseline.

### 6.3 New golden mini-corpus (assertion + experiencer labelled)

`tests/data/benchmarks/en/assertion_edge_cases.json` (gold items carry proper
`assertion_status` incl. `negated`/`family_history`):

- "severe intellectual disability without regression" -> ID **present**, regression **excluded**
- "seizures without fever" -> seizures **present**, fever **excluded**
- "no family history of long QT syndrome" -> long QT in `family_history_findings`, **not** proband
- "no family history of deafness" -> deafness in family list, not proband
- C1 preserve: "There is no nystagmus" -> nystagmus **negated** (must stay correct)

### 6.4 The no-regression gate (precise)

Baselines (strict + present-only, both benchmarks) are committed **before** any Phase-2
change. A Phase-2 change is accepted iff **all** hold:

1. **present-only** precision/recall/F1 do **not** drop on existing corpora (fair
   apples-to-apples) -- concretely, the point-estimate F1 stays at or above the committed
   baseline's 95% bootstrap-CI lower bound (`ExtractionConfig.bootstrap_ci` is already on),
   so noise is not mistaken for regression, AND
2. new golden edge cases **pass in strict mode** (proves the new behavior is correct), AND
3. the **LLM mapping benchmark** does not regress (run via the Docker stack / Gemini key --
   see 9.1), retiring PR #291's one unchecked acceptance box.

Strict-mode deltas on legacy corpora are **reported for transparency but not gated** --
they move by design (that is the false penalty this safeguard neutralizes).

Honest caveat: present-only cannot cure a **recall** penalty where the gold itself
mislabels a genuinely family/negated finding as present. The corpora contain ~1 such item
total, so it is negligible; the real cure is richer gold, which the new golden set begins.

## 7. Phase 2 -- LLM extraction contract v2

Three atomic, individually benchmark-gated changes.

### B1 -- assertion load-bearing
- Now: `phentrieve/llm/pipeline_phase2.py:273-279` sets
  `assertion = CATEGORY_TO_ASSERTION[category]`, ignoring the model's `assertion`.
- Target: consume the model's `assertion` field directly as polarity;
  `category` becomes a derived compat field. Dedup key stays
  `(term_id, experiencer, assertion)` (`pipeline_phase2.py:339-349`).

### B2 -- family history -> separate list
- Now: `family_history` excluded from `ACTIONABLE_CATEGORIES`
  (`pipeline_phase1.py:17`), filtered out at `pipeline.py:340-343`, and dropped from
  export at `service_adapters.py:381-384` -- silently lost.
- Target: carry `experiencer` through aggregation into a dedicated
  `family_history_findings: [{hpo_id, label, assertion, evidence...}]`. Proband findings
  and the phenopacket remain proband-only. `_coerce_export_phenotype`
  (`service_adapters.py:363-402`) continues to exclude family terms from proband
  `PhenotypicFeature`s.

### B3 -- "X without Y" -> excluded term
- Now: `negated_qualifier` surfaces as a string only
  (`full_text_service.py:457-463`; consumed at `pipeline.py:1234`).
- Target: when a finding carries `negated_qualifier="Y"`, run one retrieval call on the
  qualifier phrase, map Y to an HPO id, and emit it as an **excluded**
  (`assertion=absent`) finding -> `excluded: true` PhenotypicFeature. Guardrails: a
  retrieval-confidence floor (no garbage exclusions); fall back to the metadata string
  when Y does not map above the floor.

### Shared output shape (all consumers)
`{ proband_findings[], family_history_findings[], ... }`, each finding carrying
`assertion in {present, absent, uncertain}`. MCP `capabilities_version` rolls (its own
cache-key contract).

### i18n guard
The negation-scope rules live only in `phentrieve/llm/prompts/templates/two_phase/en.yaml`
(v3.1.0); the loader falls back to `en`, so this is not a live bug. Phase 2 adds a **test**
asserting any `two_phase/*.yaml` includes the negation-scope block + qualifier few-shots,
so a future localized template cannot silently regress.

## 8. Phase 3 -- Consumer surface (REST + Vue) + close-out

- **REST:** extend `api/schemas/text_processing_schemas.py` (+ `query_schemas.py` as
  needed) and the response builders in `api/services/text_processing_execution.py` with
  `family_history_findings` and excluded findings; bump the API version.
- **Vue:** `AggregatedTermsView.vue`, `PhenotypeCollectionPanel.vue`, `ResultsDisplay.vue`
  -- a "Family history" section + an "excluded" chip on ruled-out terms; add i18n locale
  keys (`make frontend-i18n-check`); component tests.
- **Close-out:** `Closes #289`; write the final verification doc; CHANGELOG; coordinated
  CLI/API/Frontend version bump; release per the repo release process.

## 9. Testing, sequencing & environment

### 9.1 Environment dependency (called out, not assumed)
The LLM-benchmark gate needs the Docker stack and the Gemini key, which lives in the
container, not this checkout. Phase-2 acceptance therefore runs against the Docker MCP;
the plan will make this an explicit, owned step (not a silent assumption).

### 9.2 Gates per phase
- All phases: `make check`, `make typecheck-fast`, `make test`; before push `make ci-local`
  + `make security-python`.
- Phase 3 adds `make frontend-test-ci` + `make frontend-build-ci` + `make frontend-i18n-check`.
- Coverage-improving tests on all touched code; TDD per fix; atomic commits (one finding =
  one commit), matching PR #291 discipline.

### 9.3 Ordering
Phase 0 (independent, anytime) -> Phase 1 (safeguard + baselines) -> Phase 2 (B1 -> B2 ->
B3, each gated) -> Phase 3 (surface + close-out).

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM prompt/behavior change regresses mapping benchmark (PR #261 precedent) | Atomic per-behavior commits; present-only + new-golden gate; revert the single offending commit |
| Strict-mode legacy numbers "look worse" and alarm reviewers | Report strict deltas but gate on present-only; document the false-penalty mechanic in the results |
| Qualifier retrieval emits a wrong excluded term | Confidence floor + string fallback; golden case asserts the mapping |
| Output-contract change ripples to REST/Vue unexpectedly | Shared shape defined once (section 7); frontend CI as blast-radius check; capabilities_version roll signals warm clients |
| Family list leaks into proband phenopacket | `_coerce_export_phenotype` guard retained + golden/export test |

## 11. Out of scope (this round)
- Full re-annotation of the deterministic corpora with assertion/experiencer labels
  (mini-corpus only).
- Making the deterministic retrieval detector experiencer-aware (it has no family state,
  `assertion_detection.py:33-36`); family logic stays in the LLM path.
- GA4GH pedigree/`Family` message modelling (family findings surface in results, not as a
  formal pedigree).

## 12. File index (verified sites)

| Area | Primary sites |
|---|---|
| Scorer core | `evaluation/extraction_metrics.py:66-79,317-332`; `evaluation/_extraction_types.py:7-12` |
| Ontology scorer | `evaluation/ontology_matching.py:55,142-162,213-217,226-246` |
| Gold loading / projection | `benchmark/data_loader.py:12-18,41-54,188-201,204-242` |
| Benchmark runner / choke point | `benchmark/extraction_benchmark.py:62,116-117,179,194-200,257,277-280,344-346` |
| Benchmark CLI | `benchmark/extraction_cli.py:39-44,116-133` |
| LLM benchmark scoring | `evaluation/assertion_metrics.py:26-43,71-101` |
| Assertion (B1) | `llm/pipeline_phase2.py:273-279,339-349` |
| Family history (B2) | `llm/pipeline_phase1.py:17`; `llm/pipeline.py:340-343`; `api/mcp/service_adapters.py:363-402,381-384` |
| Qualifier (B3) | `text_processing/full_text_service.py:457-463`; `llm/pipeline.py:1234` |
| Schema axes | `llm/types.py:95-118`; prompt `llm/prompts/templates/two_phase/en.yaml` (v3.1.0) |
| REST surface | `api/schemas/text_processing_schemas.py`; `api/services/text_processing_execution.py` |
| Vue surface | `frontend/src/components/{AggregatedTermsView,PhenotypeCollectionPanel,ResultsDisplay}.vue` |
