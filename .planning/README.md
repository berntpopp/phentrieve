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

- None.

## Current Specs

- None.

## Archived And Superseded

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
