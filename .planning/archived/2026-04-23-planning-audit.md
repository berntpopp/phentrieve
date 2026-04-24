# Planning Audit — 2026-04-23

**Scope:** `.planning/active/` and `.planning/analysis/` checked against the
current `main` worktree and reachable git history.

## Outcome

### Moved from `active/` to `completed/`

- `2026-04-16-llm-cli-grounded-whole-note-implementation-plan.md`
  - Grounded extraction types, grounded context, CLI mode wiring, and
    integration coverage are present in `phentrieve/llm/types.py`,
    `phentrieve/llm/pipeline.py`, `phentrieve/text_processing/full_text_service.py`,
    `phentrieve/cli/text_commands.py`, and
    `tests/integration/llm/test_grounded_pipeline_integration.py`.
  - Supporting history includes commits `aebe9c8`, `66e938d`, `c8743a4`,
    `e45fa6c`, and merge `46bd5d6`.

- `2026-04-16-llm-shared-chunk-pipeline-internal-refactor-plan.md`
  - Shared grouped preprocessing landed in `phentrieve/llm/preprocessing.py`
    and is consumed by the pipeline, full-text service, and benchmark path.
  - Supporting history includes commits `7fff42c`, `13b118b`, `a2d0f1a`, and
    current references to `build_extraction_groups(...)` in runtime and tests.

- `2026-04-18-llm-benchmark-configurable-cost-and-energy-implementation-plan.md`
- `2026-04-18-openai-structured-output-provider-implementation-plan.md`
- `2026-04-19-multi-provider-llm-quality-stabilization-implementation-plan.md`
  - Current code contains `BenchmarkAccountingConfig`,
    `TokenPricingConfig`, `EnergyAccountingConfig`,
    `estimated_token_cost`, `phentrieve/benchmark/energy.py`,
    OpenAI/Anthropic/Ollama providers, structured retries, and phase-1
    fallback coverage.
  - These changes were merged in commit `5a5b0b2`.

- `2026-04-19-phenopacket-v2-and-annotation-sidecar-implementation-plan.md`
  - Normalized export models, annotation sidecar schema/helpers, CLI plumbing,
    and API bundle export are present in `phentrieve/phenopackets/*`,
    `phentrieve/cli/text_commands.py`, `api/routers/phenopacket_router.py`,
    and related tests.
  - Supporting history includes `84ad279`, `ac84d45`, `be0950b`, `cf094c1`,
    and merge `da9aa32`.

- `2026-04-22-full-text-query-ui-correction-implementation-plan.md`
  - The plan already stated implementation on branch
    `feat/unified-full-text-workspace`; current `main` contains the merged UI,
    backend, and tests from PR `#229`.
  - Supporting history includes `8881e0d`, `176373a`, and merge `60e28fc`.

- `CI-SPEEDUP-PLAN-2026-04-10.md`
  - The speedup work is reflected in git history and in
    `CODE-QUALITY-REVIEW-2026-04-09.md`, which records executed tasks and
    before/after timings.
  - Supporting history includes `0872b6b`, `10586b2`, `20c0e80`, `0937e5b`,
    and related CI/perf commits.

- `HPO-EXTRACTION-IMPLEMENTATION-PLAN.md`
  - The extraction benchmark feature exists in the current codebase under
    `phentrieve/evaluation/` and `phentrieve/benchmark/extraction_benchmark.py`
    and was merged earlier via PR `#131`.
  - Supporting history includes `2eb5b86`, `6ece285`, `d4ce115`, `3b07348`,
    and merge `aeea2ea`.

### Moved from `active/` to `archived/`

- `unified-output-format/README.md`
- `unified-output-format/UNIFIED-OUTPUT-FORMAT-PHENOPACKETS.md`
  - These documents still described an approval-stage design with obsolete
    assumptions and old planning paths. The core Phenopacket capability was
    implemented long ago in PR `#113` (`61ba259`), and later sidecar work
    superseded the original architecture.

### Moved from `analysis/` to `archived/`

- `POST-REFACTORING-CLEANUP-ASSESSMENT.md`
  - The primary inconsistency it highlighted (CLI/API default divergence) is no
    longer present in the current codebase, and the note reads as a resolved
    action list rather than active analysis.

- `PERFORMANCE-REVIEW-ANTIPATTERNS.md`
  - This note critiques a superseded frontend performance plan and no longer
    maps to an active planning thread or current implementation state.

## Remaining Active Item

- `2026-04-17-ontology-embedding-fidelity-implementation-plan.md`
  - **Kept active.** There is reachable git history on
    `feat/ontology-embedding-fidelity` (`a2faa84`, `07ca297`, `f527eff`,
    `3205644`, `47479d0`), but the current `main` worktree does **not**
    contain `phentrieve/analysis/*` or `scripts/analyze_embedding_ontology.py`.
    This is active branch work, not completed `main` work.

## Analysis Notes Retained

The remaining files under `.planning/analysis/` were left in place because they
still function as useful review or verification history, even where some
findings have since been addressed.
