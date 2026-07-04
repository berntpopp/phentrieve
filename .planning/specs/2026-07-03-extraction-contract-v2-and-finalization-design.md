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
- Issue #289 is still **open**, and correctly so: PR #291 reduced the acute symptoms
  (removed the deterministic-detector override on the LLM path; stopped the contradictory
  present+negated pair on one id) but did **not** deliver the full contract -- load-bearing
  assertion, family surfaced instead of dropped, qualifier mapped. That is this v2 work, so
  #289 stays open until Phase 3 closes it. Separately, PR #291 has **no spec /
  completed-plan / verification** artifacts in `.planning/` (only its analysis doc,
  committed 3 weeks late); the `.planning/README.md` index omits it; and `completed/` +
  `archived/` carry ~64 pre-convention ALL-CAPS files.

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
   an HPO id, and emit it as an **excluded** finding (LLM `absent` -> canonical `negated`,
   per B0) -> `excluded: true` PhenotypicFeature in the phenopacket. Confidence-gated;
   falls back to today's metadata string when Y does not map cleanly.
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
| **2** | LLM contract v2 (B0 vocabulary, B1 assertion, B2 family, B3 qualifier) | **high** | gated by Phase 1 |
| **3** | Consumer surface (REST + Vue) + close-out | medium | last |

Phase 1 **must precede** Phase 2: we cannot gate the behavior changes without the
safeguard and baselines in place first.

## 5. Phase 0 -- Close-out & planning cleanup (no behavior change)

- **Backfill PR #291 artifacts** so it matches the hardening/remediation efforts:
  - `specs/2026-06-14-mcp-stabilization-design.md` (retroactive, from the analysis doc)
  - `completed/2026-06-14-mcp-stabilization-plan.md` (the 14-finding execution record)
  - `analysis/2026-06-14-mcp-stabilization-verification.md` (the 2026-07-03 deep re-verification)
- **Refresh the index:** update `.planning/README.md` (add stabilization + #291; correct
  "Current Active Work: None"); reconcile top-level `STATUS.md`. The README layout documents
  `active/` **and** `drafts/`, but **neither directory exists** -- create both (`.gitkeep`)
  so the documented tree is real (this effort's plan lands in `active/`).
- **Quarantine legacy files by rule, not by count.** Selection rule: every top-level
  `*.md` in `completed/` and `archived/` whose basename does **not** match the
  `YYYY-MM-DD-*` convention (the pre-convention ALL-CAPS files -- **64** of them as of
  2026-07-03), plus the stray `archived/unified-output-format/`. Move them into
  `.planning/archived/pre-convention/` and write a generated `MANIFEST.md` listing what
  moved (so the selection is auditable and reproducible). Git history is preserved; no
  deletions.
- **Issue #289** stays open as the tracking anchor; it is closed at the end of Phase 3
  (with a `Closes #289` on that PR).

Gate: `make check`, `make typecheck-fast`, `make test` (the repo-required trio; trivially
green for docs, but run for parity). This phase touches no application code.

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

There is **no** present-only / proband-filtered scoring mode; the only adjacent lever,
`--include-assertions/--no-include-assertions` (`extraction_cli.py:42-44`), disables
detection upstream and cripples the model. The **LLM** benchmark scores with the same
`CorpusExtractionMetrics` and **already computes two projections** side by side --
`assertion_results` (strict tuple) and `id_only_results` (assertion stripped) at
`phentrieve/benchmark/llm_benchmark.py:695-696`. Note `id_only` is **not** the same as
present-only: it ignores polarity entirely, so a correctly-predicted `(Y,ABSENT)` exclusion
still counts as a bare-id `Y` prediction. `present-only` (filter to PRESENT, then compare
ids) is the fairer proband view, and it slots into the LLM benchmark's existing
dual-projection pattern as a **third** projection rather than a new code path.

### 6.2 Safeguard: `--scoring-mode {strict, present-only}` (default `strict`)

Pure helper in `phentrieve/evaluation/extraction_metrics.py`:

```python
PROBAND_PRESENT = {"PRESENT"}

def normalize_for_scoring(results, mode="strict"):
    if mode == "strict":
        return results  # identity -> identical metric values (see note below)
    return [
        ExtractionResult(
            doc_id=r.doc_id,
            predicted=[(hid, "PRESENT") for hid, a in r.predicted if a in PROBAND_PRESENT],
            gold=[(hid, "PRESENT") for hid, a in r.gold if a in PROBAND_PRESENT],
        )
        for r in results
    ]
```

- Applied at the single choke point **after the `results` list is fully built and before
  the metric calculators run** -- `extraction_benchmark.py:~296` (the loop populates
  `results` through ~294; `calculate_all_metrics` / `calculate_ontology_aware_metrics` run
  at ~297-298). It must **not** be applied at line 257, which is only the empty-list
  declaration. One call covers both strict and ontology-aware paths.
- `present-only` filters to `PRESENT` and re-stamps, collapsing the strict tuple
  comparison into an **id-level proband-present** comparison: absent/family predictions
  can no longer become false positives, and a polarity-blind (all-PRESENT) gold compares
  fairly against the improved model.
- **Scope:** present-only is **only** for legacy polarity-blind corpora (the directory
  annotation sets). Assertion-labeled fixtures -- the new golden mini-corpus **and** the
  existing `tests/data/extraction/tiny_extraction_test.json`, which already carries ABSENT
  gold (`data_loader.py:126` accepts it) -- always run **strict**, since present-only would
  hide exactly the absent-handling those fixtures exist to verify.
- Threaded via `scoring_mode: str = "strict"` on `ExtractionConfig` +
  `--scoring-mode` option in `extraction_cli.py`. In the **LLM** benchmark, add
  `present-only` as a third projection next to the existing `assertion_results` /
  `id_only_results` at `llm_benchmark.py:695-696` (same `CorpusExtractionMetrics`, no new
  scoring code).

**Why legacy numbers are provably reproduced:** with the default `strict`,
`normalize_for_scoring` returns the input list unchanged, so identical `ExtractionResult`s
flow into identical, untouched metric code -- the computed **metric values** are identical
(not the whole results file, which carries a fresh `datetime.now()` timestamp at
`extraction_benchmark.py:674`). A no-regression unit test asserts
`normalize_for_scoring(results, "strict") is results` and that the strict **metrics** equal
a stored baseline.

### 6.3 New golden mini-corpus (assertion + experiencer labelled)

**Format:** the corpus is a **document-payload** JSON (`{"documents": [{"id", "text",
"gold_hpo_terms": [...]}]}`), because `parse_gold_terms` reads `gold_hpo_terms[].assertion`
(`data_loader.py:194`) -- **not** `assertion_status` (that field is only read by the
phenobert directory path, `:214`). Each gold item is `{"id", "assertion"}` where
`assertion in {PRESENT, ABSENT}` (the internal enum, `ASSERTION_STATUS_MAP`).

**Proband present/absent cases** (fit the `(id, assertion)` tuple scorer, run in `strict`):
- "severe intellectual disability without regression" -> ID **PRESENT**, regression **ABSENT** (excluded)
- "seizures without fever" -> seizures **PRESENT**, fever **ABSENT** (excluded)
- C1 preserve: "There is no nystagmus" -> nystagmus **ABSENT** (must stay correct)

**Family-history cases are NOT encoded as an assertion.** Experiencer is a separate axis and
the `(id, assertion)` benchmark cannot express it cleanly (jamming `FAMILY_HISTORY` into the
assertion slot is exactly the smell the review flagged). Extending the scorer to a
`(id, experiencer, assertion)` triple is out of scope for a mini round. Instead, family-history
routing is verified by **dedicated unit / integration tests**, not the corpus benchmark:
- "no family history of long QT syndrome" -> long QT appears in `family_history_findings`,
  **absent** from proband findings **and** from the phenopacket.
- "no family history of deafness" -> deafness in the family list, not proband.

### 6.4 The no-regression gate (precise)

Baselines (strict + present-only, both benchmarks) are committed **before** any Phase-2
change. A Phase-2 change is accepted iff **all** hold:

1. **present-only** precision/recall/F1 do **not** drop below the committed baseline on
   existing corpora (fair apples-to-apples) -- compared by the deterministic machinery in
   §6.5 (point-estimate gating by default; seeded-CI lower bound optional/advisory), AND
2. new golden edge cases **pass in strict mode** (proves the new behavior is correct), AND
3. the **LLM mapping benchmark** does not regress (run via the Docker stack / Gemini key --
   see 9.1), retiring PR #291's one unchecked acceptance box.

Strict-mode deltas on legacy corpora are **reported for transparency but not gated** --
they move by design (that is the false penalty this safeguard neutralizes).

Honest caveat: present-only cannot cure a **recall** penalty where the gold itself
mislabels a genuinely family/negated finding as present. The corpora contain ~1 such item
total, so it is negligible; the real cure is richer gold, which the new golden set begins.

### 6.5 Gate machinery (the gate is a criterion; Phase 1 must make it executable)

Section 6.4 is the *criterion*; today nothing enforces it -- `extraction_cli.py`'s
`_test_significance` only **reports** CI overlap (`:335`), the bootstrap is **unseeded**
(`extraction_metrics.py:442`), and the LLM benchmark's `calculate_all_metrics`
(`llm_benchmark.py:695-696`) returns **empty** `confidence_intervals` (bootstrap never runs
on that path). Phase 1 therefore owns these concrete tasks:

- **Committed baseline files:** run both benchmarks (strict + present-only) and commit the
  metric JSONs as the regression fence (a fixture, not a transient run).
- **Deterministic comparison:** either seed the bootstrap (`random.Random(seed)` in
  `bootstrap_confidence_intervals`) so the CI lower bound is stable, **or** gate on the
  deterministic **point-estimate** F1/precision/recall (which are already reproducible) and
  keep CIs as advisory. Point-estimate gating is the simpler, seed-free default; the LLM
  path (empty CIs) uses point-estimate gating regardless.
- **An assert command:** a `phentrieve benchmark assert-no-regression --baseline <f> --candidate <f> --mode present-only`
  (or a test) that **exits non-zero** on regression -- turning the report into a gate.
- **Thread present-only through the LLM benchmark's full surface:** not just beside
  `assertion_results`/`id_only_results` at `llm_benchmark.py:695`, but through the
  checkpoint/failure serialization and the CLI/reporter, so a resumed or failed run still
  emits the third projection.

## 7. Phase 2 -- LLM extraction contract v2

Four atomic, individually benchmark-gated changes (B0 lands first as the foundation).

### B0 -- canonical boundary vocabulary (do this first)

There are **two** vocabularies today and they must not be conflated (this is a real bug
risk): the LLM schema axis is `present / absent / uncertain` (`llm/types.py`), while the
pipeline/export/benchmark boundary uses `affirmed / negated / normal` (and the benchmark's
`PRESENT / ABSENT / FAMILY_HISTORY`). Critically, the MCP export coerces **only** the literal
`"negated"` to an excluded feature -- `api/mcp/service_adapters.py:397`
(`assertion_status="negated" if assertion == "negated" else "affirmed"`) -- and the Vue
phenopacket export does the same (`frontend/src/composables/usePhenotypeCollection.js:148`).
So an LLM `assertion="absent"` reaching either boundary unchanged would **silently export as
present**.

Decision: define the **canonical internal status vocabulary** = `affirmed / negated / normal /
uncertain`, and implement it as **one shared canonicalization helper** (a single Python
function -- e.g. `canonicalize_assertion()` -- plus its mirror in the Vue export composable),
reused by the LLM->pipeline mapping, the REST adapter, the MCP export/projection, and the Vue
phenopacket export -- **not** ad-hoc string checks scattered per boundary. It maps
`present->affirmed`, `absent->negated`, `uncertain->uncertain`, and treats `negated` **and**
`absent` as excluded (defensive). Tests assert `absent`/`negated` -> `excluded: true` at every
boundary that consumes it (MCP `service_adapters.py`, Vue `usePhenotypeCollection.js`).

### B1 -- assertion load-bearing (carry BOTH axes end-to-end)
- Now: Phase-1 parsing keeps only `phrase` / `category` / `negated_qualifier` and
  **discards `experiencer` and `assertion`** (`phentrieve/llm/pipeline.py:582-588`); Phase 2
  then re-derives `assertion = CATEGORY_TO_ASSERTION[category]`
  (`pipeline_phase2.py:273-279`), so the model's own axes are never used.
- Target: carry `experiencer` and `assertion` (canonicalized per B0) **through the entire
  chain** -- Phase-1 parse (`pipeline.py:582`), the Phase-1 dedup/expand, the actionable
  filter, retrieval keys, fallback traces, the mapping payloads, and the final `LLMPhenotype`
  -- and consume the model's `assertion` as polarity. `category` becomes a derived compat
  field. Dedup key stays `(term_id, experiencer, assertion)` (`pipeline_phase2.py:339-349`).

### B2 -- family history -> separate list (needs a real parallel mapping path)
- Now: `family_history` is not in `ACTIONABLE_CATEGORIES` (`pipeline_phase1.py:17`) and the
  actionable filter runs **before** retrieval (`pipeline.py:340-343`), so family phrases are
  dropped **before** they are ever mapped to an HPO id; they are also dropped from export
  (`service_adapters.py:381-384`). And no result schema carries a family list --
  `LLMExtractionResult` exposes only `terms` (`llm/types.py:199`) and the REST
  `TextProcessingResponse` only `aggregated_hpo_terms` (`api/schemas/text_processing_schemas.py`).
- Target (explicit path): **collect** family-experiencer phrases before the actionable
  filter -> **retrieve/map** them through the same retrieval used for proband terms (a
  parallel resolution pass, not a reuse of the dropped set) -> **emit** a dedicated
  `family_history_findings: [{hpo_id, label, assertion, experiencer, evidence...}]` on the
  result contract (new field on `LLMExtractionResult` and the REST/MCP response schemas) ->
  **exclude** from the proband phenopacket.
- **Fix the phenopacket guard (breaking otherwise):** today `_coerce_export_phenotype`
  drops family only when `assertion == "family_history"` (`service_adapters.py:380-384`).
  Once experiencer is a separate axis (B1), that condition no longer matches, so the guard
  must switch to `experiencer == "family_history"` **or** export only `proband_findings`.
  Guarded by a regression test asserting no family term reaches the subject's
  `PhenotypicFeature`s.

### B3 -- "X without Y" -> excluded term (with a defined output shape)
- Now: `negated_qualifier` surfaces as a string only (`full_text_service.py:457-463`;
  consumed at `pipeline.py:1234`), and the REST adaptation drops evidence/qualifier context
  entirely when it builds `AggregatedHPOTermAPI` (`api/services/text_processing_execution.py:135-149`).
- Target: when a finding carries `negated_qualifier="Y"`, run one retrieval call on the
  qualifier phrase and, if it maps above a confidence floor, emit a **generated excluded
  finding** with this explicit shape (which REST/MCP must stop dropping):
  - `hpo_id`, `label`
  - `assertion` = canonical `negated` (per B0) -> `excluded: true` in the phenopacket
  - `qualifier_surface_text` (the literal "Y" span from the source)
  - `evidence` context + `attribution_span` (so the exclusion is auditable, not bare)
  - `provenance` flag, e.g. `match_method="negated_qualifier_derived"` (distinguishes it
    from a directly-extracted negation)
  - `confidence` (>= floor)
- Guardrails: the retrieval-confidence floor prevents garbage exclusions; when Y does not
  map above the floor, **fall back** to today's `negated_qualifier` metadata string (no
  generated term). Golden/unit tests assert both the mapped and the fallback path.

### Carrier surfaces (thread the new fields end-to-end; each is a known drop point)
The new fields (`experiencer`, canonical `assertion`, `family_history_findings`, the B3
excluded-term shape) must be carried through **every** layer -- these are the specific sites
that currently drop them:
- `LLMExtractionResult` -- add `family_history_findings` (today only `terms`, `llm/types.py:199`).
- Shared service payload (`full_text_service`) -- surface family + excluded findings, not the
  metadata string only (`:457-463`).
- REST schema `text_processing_schemas.py` + adapter -- the adapter drops qualifier/provenance
  today (`text_processing_execution.py:135-149`); add the fields.
- MCP `api/mcp/schemas.py` (output schema) + `projection.py` (per-term projection) +
  `shaping.py` budgeting (`enforce_budget` must know the new list so it is not silently
  trimmed or, conversely, exempted incorrectly).
- Vue display + export composables (see Phase 3).

### Shared output shape (all consumers)
`{ proband_findings[], family_history_findings[], ... }`, each finding carrying the
**canonical** status (`affirmed / negated / normal / uncertain` per B0; `negated` renders as
`excluded: true`). MCP `capabilities_version` rolls (its own cache-key contract).

### i18n guard
The negation-scope rules live only in `phentrieve/llm/prompts/templates/two_phase/en.yaml`
(v3.1.0); the loader falls back to `en`, so this is not a live bug. Phase 2 adds a **test**
asserting any `two_phase/*.yaml` includes the negation-scope block + qualifier few-shots,
so a future localized template cannot silently regress.

## 8. Phase 3 -- Consumer surface (REST + Vue) + close-out

- **REST:** extend `api/schemas/text_processing_schemas.py` (+ `query_schemas.py` as
  needed) and the response builders in `api/services/text_processing_execution.py` with
  `family_history_findings` and excluded findings; bump the API version.
- **Vue:** the full-text extraction results render through `FullTextResponseReceipt.vue`
  (mounted from `QueryInterface.vue:128`), **not** only `ResultsDisplay.vue` -- so the edit
  set is `FullTextResponseReceipt.vue`, `AggregatedTermsView.vue`, `PhenotypeCollectionPanel.vue`,
  `ResultsDisplay.vue`, plus `useUserNoteAnnotations.js` (annotation carrier) and the
  `usePhenotypeCollection.js` export path (B0). Add a "Family history" section + an
  "excluded" chip on ruled-out terms; i18n locale keys (`make frontend-i18n-check`); component
  tests.
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
Phase 0 (independent, anytime) -> Phase 1 (safeguard + baselines) -> Phase 2 (B0 -> B1 ->
B2 -> B3, each gated) -> Phase 3 (surface + close-out).

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM prompt/behavior change regresses mapping benchmark (PR #261 precedent) | Atomic per-behavior commits; present-only + new-golden gate; revert the single offending commit |
| Strict-mode legacy numbers "look worse" and alarm reviewers | Report strict deltas but gate on present-only; document the false-penalty mechanic in the results |
| Qualifier retrieval emits a wrong excluded term | Confidence floor + string fallback; golden case asserts the mapping |
| Output-contract change ripples to REST/Vue unexpectedly | Shared shape defined once (section 7); frontend CI as blast-radius check; capabilities_version roll signals warm clients |
| Family list leaks into proband phenopacket (guard silently breaks when experiencer splits from assertion) | Switch `_coerce_export_phenotype` guard from `assertion=="family_history"` to experiencer-based (B2); regression test asserts no family term in subject `PhenotypicFeature`s |
| No-regression gate is report-only today (CI overlap, unseeded bootstrap, empty LLM CIs) | Phase 1 adds committed baselines, deterministic point-estimate gating, and an exit-non-zero assert command (§6.5) |
| Vocabulary conflation: LLM `absent` exports as present (only `"negated"` is excluded at `service_adapters.py:397` / Vue) | B0 canonicalizes at the boundary; both export paths harden to treat `absent`+`negated` as excluded; test asserts `excluded: true` |

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
| Benchmark runner / choke point | `benchmark/extraction_benchmark.py:62,116-117,179,194-200`; normalize after build loop (~261-294) before metrics (~296-298), **not** at decl `:257` |
| Benchmark CLI | `benchmark/extraction_cli.py:39-44,116-133` |
| Gate machinery (report-only today) | `extraction_cli.py:324-338` (`_test_significance` reports CI overlap, does not gate); `extraction_metrics.py:442` (unseeded bootstrap) |
| ABSENT fixture (stays strict) | `tests/data/extraction/tiny_extraction_test.json`; loader `benchmark/data_loader.py:126` |
| LLM benchmark scoring (dual projection) | `benchmark/llm_benchmark.py:695-696` (`assertion_results` + `id_only_results`) |
| Vocabulary boundary (B0) | `llm/types.py:24` (LLM default `present`); `api/mcp/service_adapters.py:397`; `frontend/src/composables/usePhenotypeCollection.js:145,148` |
| Assertion (B1) | `llm/pipeline.py:582-588` (Phase-1 discards axes); `llm/pipeline_phase2.py:273-279,339-349` |
| Family history (B2) | `llm/pipeline_phase1.py:17`; `llm/pipeline.py:340-343`; `llm/types.py:199` (no family field); `api/mcp/service_adapters.py:363-402,381-384` |
| Qualifier (B3) | `text_processing/full_text_service.py:457-463`; `llm/pipeline.py:1234`; `api/services/text_processing_execution.py:135-149` (drops fields) |
| Schema axes | `llm/types.py:95-118`; prompt `llm/prompts/templates/two_phase/en.yaml` (v3.1.0) |
| REST surface | `api/schemas/text_processing_schemas.py`; `api/services/text_processing_execution.py` |
| Vue surface | `frontend/src/components/{FullTextResponseReceipt,AggregatedTermsView,PhenotypeCollectionPanel,ResultsDisplay}.vue` (receipt mounted at `QueryInterface.vue:128`); `composables/{useUserNoteAnnotations,usePhenotypeCollection}.js` |
