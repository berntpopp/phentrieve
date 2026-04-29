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

- `active/2026-04-17-ontology-embedding-fidelity-implementation-plan.md` -
  plan for the standalone embedding-vs-HPO-DAG fidelity analysis script
  (issue #34). Branch `feat/ontology-embedding-fidelity` (worktree).
- `specs/2026-04-17-ontology-embedding-fidelity-design.md` - canonical design
  input for the ontology-embedding fidelity work above.
- `active/2026-04-29-eu-ai-act-research-use-compliance-plan.md` -
  implementation plan for aligning Phentrieve's docs, UI, API, CLI, MCP,
  deployment defaults, and tests with a research-use-only EU AI Act posture.

## Migration Note

The repository previously stored planning material in `plan/` and
`docs/superpowers/`. Those sources have been consolidated here so both Codex
and Claude Code use the same planning tree.
