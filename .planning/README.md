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

- `analysis/2026-04-29-eu-ai-act-research-use-review.md` - EU AI Act
  research-use posture review for Phentrieve documentation, UI, API, deployment,
  and open-source/public-service positioning.

## Current Active Work

- `active/2026-04-30-local-browser-pii-guard-implementation-plan.md` -
  implementation plan for GitHub issue #249, covering local browser-side PII
  detection and redaction before Query and Full Text submissions.

## Recently Completed

- `specs/2026-04-17-ontology-embedding-fidelity-design.md` - canonical design
  input for the completed ontology-embedding fidelity work.
- `completed/2026-04-17-ontology-embedding-fidelity-implementation-plan.md` -
  standalone embedding-vs-HPO-DAG fidelity analysis script (issue #34).
- `completed/2026-04-29-eu-ai-act-research-use-compliance-plan.md` -
  research-use-only compliance posture implementation (PR #245).
- `completed/2026-04-29-modern-mcp-implementation-plan.md` -
  modern MCP HTTP facade implementation (PR #247).
- `completed/2026-04-30-prompt-injection-guards-implementation-plan.md` -
  public LLM prompt-injection guard implementation (PR #250).

## Migration Note

The repository previously stored planning material in `plan/` and
`docs/superpowers/`. Those sources have been consolidated here so both Codex
and Claude Code use the same planning tree.
