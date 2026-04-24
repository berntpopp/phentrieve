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

## Current Active Work

- `active/2026-04-24-ontology-aware-benchmark-command-option-plan.md` - plan
  for adding opt-in ontology-aware soft/partial metrics to benchmark commands.
- `specs/2026-04-24-ontology-aware-benchmark-metrics-spec.md` - consolidated
  design input for ontology-aware full-text HPO benchmark metrics.

## Migration Note

The repository previously stored planning material in `plan/` and
`docs/superpowers/`. Those sources have been consolidated here so both Codex
and Claude Code use the same planning tree.
