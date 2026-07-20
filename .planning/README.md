# Planning

This directory is the single home for repository planning and agent-facing
execution context.

## Layout

- `active/` - plans that are still being executed or tracked
- `completed/` - implemented plans and finished execution records
- `archived/` - superseded or rejected plans
- `analysis/` - reviews, investigations, verification reports
- `specs/` - design documents and approved implementation inputs
- `drafts/` - rough plans that are not active yet
- `reference/` - templates and evergreen planning guidance

## Rules

- Create new planning artifacts only under `.planning/`.
- Do not create parallel plan trees under `plan/` or `docs/superpowers/`.
- Prefer stable links to `.planning/...` paths in docs and changelogs.
- If a plan changes status, move the file instead of copying it.

## Current Entry Points

- Start with `active/` for in-flight work.
- Read `analysis/` for review context and investigations.
- Read `specs/` before implementing multi-step work.
- Use `completed/` for historical implementation patterns.

## Recent Analysis

- `analysis/2026-06-14-mcp-stabilization-verification.md` - 2026-07-03 deep
  re-verification confirming all 14 findings from the MCP stabilization pass
  (PR #291) shipped and are test-pinned in v0.24.0/0.24.1, correcting two
  earlier docs that had mis-recorded LLM-1/LLM-2 as deferred. Records the
  residual behaviors (advisory assertion axis, family-history silently
  dropped, qualifier metadata-only) that seeded the active
  `extraction-contract-v2` effort, and the `archived/pre-convention/`
  quarantine. Spec in `specs/2026-06-14-mcp-stabilization-design.md`, plan in
  `completed/2026-06-14-mcp-stabilization-plan.md`, source analysis in
  `analysis/2026-06-14-mcp-stabilization-plan.md`.
- `analysis/2026-06-14-phentrieve-mcp-assessment-remediation-verification.md` -
  before/after verification of the assessment remediation pass that closed the
  defects PR #288 left open or punted: D1 span-level negation (over-negation
  eliminated, C1 preserved), D2/D3 co-findings + chunk hygiene, D4 native
  phenopacket object, D5 explainable compare, D6 confidence band, D7/D13 payload
  shape, D8 cache contract, D9 startup warmup, D11 honest provenance. Every
  assessment dimension reaches >= 9.5 (LLM-D1 deferred). Spec in
  `specs/2026-06-14-phentrieve-mcp-assessment-remediation-design.md`, plan in
  `completed/2026-06-14-phentrieve-mcp-assessment-remediation-plan.md`, source
  assessment in `analysis/2026-06-14-phentrieve-mcp-assessment.md`.
- `analysis/2026-06-14-phentrieve-mcp-hardening-verification.md` - before/after
  verification of the MCP hardening pass (C1 negation, H1 best-match, lossless
  pipeable export, schema/token dedup, capabilities cache contract, error
  envelopes; LLM negation fix benchmark-gated). Overall 7 -> 9.4 across the
  evaluation dimensions. Spec in `specs/2026-06-14-phentrieve-mcp-hardening-
  design.md`, plan in `completed/2026-06-14-phentrieve-mcp-hardening-plan.md`,
  source evaluation in `analysis/2026-06-14-phentrieve-mcp-evaluation.md`.
- `analysis/2026-06-13-fulltext-annotation-curation-verification.md` - E2E
  verification of full-text annotation curation (change/remove/annotate/undo +
  persisted provenance) on the rebuilt Docker stack; orphan cluster removed;
  Lighthouse unchanged; two E2E defects fixed. Plan in
  `completed/2026-06-13-fulltext-annotation-curation-plan.md`.
- `analysis/2026-06-13-mcp-gen3-modernization-verification.md` - verification
  record for the FastMCP v3 MCP modernization (8 tools, Family B envelope,
  response_mode, HTTP-only); all Python gates green.
- `analysis/2026-05-25-llm-evidence-validation-enriched-mapping-pr-regression.md` -
  same-model focused A/B showing PR #261 regressed strict-ID CSC mapping
  performance; PR closed as superseded.
- `analysis/2026-05-23-phentrieve-rag-prompting-literature-report.md` -
  retrieval, full-text RAG, LLM prompting, and evaluation improvement analysis
  that informed completed parity work and the superseded PR #261 experiment.
- `analysis/2026-05-22-hpo-v2026-02-16-benchmark-comparison.md` - benchmark
  comparison and release verification record for the HPO v2026-02-16 data
  bundle.

## Current Active Work

- `active/2026-07-03-extraction-contract-v2-phase-0-1-plan.md` - Phase 0
  (planning-tree cleanup, this pass) + Phase 1 (extraction benchmark
  safeguard: `normalize_for_scoring` present-only projection, seeded bootstrap
  CIs, `--scoring-mode` CLI option, assertion-labelled golden fixture, and an
  `assert-no-regression` gate) of the LLM extraction contract v2 effort. See
  `specs/2026-07-03-extraction-contract-v2-and-finalization-design.md`.

## Current Specs

- `specs/2026-07-20-benchmark-resume-integrity-design.md` - design for closing
  the adversarial PR #322 review: versioned execution/scoring/source identities,
  verified retrieval-runtime binding, canonical assertion projection, safe
  endpoint persistence, and immutable crash-safe benchmark publication.
- `specs/2026-07-03-extraction-contract-v2-and-finalization-design.md` -
  design for making the LLM-emitted `assertion`/`experiencer` axes
  load-bearing, surfacing family-history findings and "X without Y" excluded
  terms across MCP/REST/Vue, and a present-only no-regression benchmark gate
  so the behavior changes ship safely. Tracks issue #289.
- `specs/2026-06-13-fulltext-annotation-curation-design.md` - design for
  interactive curation of full-text annotations (change term via re-query,
  remove, annotate a fresh selection, revert) with persisted auto/manual
  provenance, plus removal of the orphaned full-text annotation workspace
  cluster.
- `specs/2026-06-13-mcp-gen3-modernization-design.md` - design for bringing the
  `api/mcp/` server up to the maintainer's Gen-3 MCP house style (Family B
  envelope, response_mode, structured errors, capabilities versioning,
  diagnostics, HTTP-only transport).

## Archived And Superseded

- `archived/pre-convention/` - 64 pre-convention ALL-CAPS `.md` files quarantined
  from the top level of `completed/` and `archived/` (basenames not matching the
  `YYYY-MM-DD-*` convention), plus the stray `unified-output-format/` directory.
  Moved by rule with `git mv` (history preserved, no deletions); see
  `archived/pre-convention/MANIFEST.md` for the full generated list.
- `archived/2026-05-22-hpo-data-release-runbook.md` - historical HPO data
  release runbook superseded by the published release and updated release docs.
- `archived/2026-04-30-codebase-health-review.md` - stale current-state review
  whose findings were remediated by PR #253.
- `archived/2026-05-25-llm-evidence-validation-enriched-mapping-design.md` -
  superseded design for source-faithful LLM evidence validation and enriched
  mapping; see the regression analysis before reusing.
- `archived/2026-05-25-llm-evidence-validation-enriched-mapping-plan.md` -
  archived implementation plan for PR #261, superseded due to same-command
  focused benchmark regression.

## Recently Completed

- `completed/2026-06-14-mcp-stabilization-plan.md` - completed execution
  record for the MCP stabilization pass (PR #291, shipped v0.24.0): all 14
  findings (B1, B2, LLM-1, LLM-2, R1, R2, B3, D4, D3, D1, D2, R3, Q1, B4) as
  one atomic, test-first commit each. See the design spec and verification
  analysis.
- `completed/2026-06-13-mcp-gen3-modernization-plan.md` - completed
  implementation plan for the FastMCP v3 MCP modernization (Family B envelope,
  response_mode, capabilities versioning, diagnostics + phenopacket + chunk_text
  tools, HTTP-only transport). See the design spec and verification analysis.
- `completed/2026-05-24-full-text-multi-vector-parity-plan.md` - completed
  implementation plan for standard full-text parity with direct-query
  multi-vector aggregation.
- `completed/2026-05-24-full-text-multi-vector-parity-design.md` - completed
  design for standard full-text parity with direct-query multi-vector
  aggregation.
- `completed/2026-05-22-ci-cd-optimization-plan.md` - completed GitHub Actions
  and local CI parity optimization plan.
- `completed/2026-04-30-codebase-health-remediation-design.md` - completed
  design for the codebase health remediation program.
- `completed/2026-04-30-codebase-health-remediation-verification.md` - final
  verification record for the codebase health remediation program.
- `completed/2026-04-30-codebase-health-remediation-plan.md` - umbrella
  remediation of the 2026-04-30 codebase health review (PR #253).
- `completed/2026-04-17-ontology-embedding-fidelity-implementation-plan.md` -
  standalone embedding-vs-HPO-DAG fidelity analysis script (issue #34).
- `completed/2026-04-29-eu-ai-act-research-use-compliance-plan.md` -
  research-use-only compliance posture implementation (PR #245).
- `completed/2026-04-29-modern-mcp-implementation-plan.md` -
  modern MCP HTTP facade implementation (PR #247).
- `completed/2026-04-30-prompt-injection-guards-implementation-plan.md` -
  public LLM prompt-injection guard implementation (PR #250).
- `completed/2026-04-30-local-browser-pii-guard-implementation-plan.md` -
  browser-side PII warning and redaction gate implementation (PR #252).
- `completed/2026-04-30-untitled-local-person-name-pii-implementation-plan.md` -
  configurable untitled person-name PII detection follow-up (PR #252).

## Migration Note

The repository previously stored planning material in `plan/` and
`docs/superpowers/`. Those sources have been consolidated here so both Codex
and Claude Code use the same planning tree.
