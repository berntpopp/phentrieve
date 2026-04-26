# Execution Handoff: CLI Profiles + Adaptive Re-Chunking

> **For the next Claude session.** Specs and plans were drafted in a prior session and are committed. This document tells you how to execute them with max parallelization, landing in one branch and one PR.

## What you're building

Two specs, two plans, one PR:

- **Spec/Plan A** (`.planning/specs/2026-04-25-cli-profiles-default-resolution-spec.md` + `.planning/active/2026-04-25-cli-profiles-default-resolution-plan.md`) — CLI profiles + `#171` interactive defaults fix. Closes issues #28 and #171.
- **Spec/Plan B** (`.planning/specs/2026-04-25-adaptive-rechunking-spec.md` + `.planning/active/2026-04-25-adaptive-rechunking-plan.md`) — Adaptive re-chunking. Closes issue #148.

Both specs and plans are TDD-style and self-contained. Read each spec fully before its plan; the spec is canonical when the plan and spec disagree.

Open companion GitHub issues spawned during planning:
- #234 — `phentrieve config calibrate-thresholds` subcommand (future work)
- #235 — Aggregator `max_score` confidence filter (future work)

These are tracked separately and **not** part of this PR.

## Strategy

### Branch and PR

- One feature branch: `feat/cli-profiles-and-adaptive-rechunking`.
- Use `git worktree add` to create the working tree at `.worktrees/profiles-adaptive` (per the user's saved memory: always use worktrees for parallel work).
- Final deliverable: one PR from that branch to `main` covering both plans.
- Commit message style follows the existing `[codex] ...` and `feat(scope): ...` conventions visible in `git log --oneline`. Each task in the plans already has a draft commit message — use it.

### Plan ordering

- **Plan A first, Plan B second.** Plan B's profile-block integration (`Profile.adaptive_rechunking`) depends on Plan A's `Profile` schema. Plan B is implementable without Plan A, but the cross-spec test `test_specA_specB_interaction.py` requires both.
- **Exception**: Plan B Phase 1 (`OrchestrationResult` dataclass refactor) is independent of Plan A and can land first if a worktree budget allows. Doing so unblocks anyone else touching `orchestrate_hpo_extraction`.

### Parallelization within a plan

Use the **superpowers:dispatching-parallel-agents** skill. Each subagent works in its own ephemeral worktree (via the `Agent` tool's `isolation: "worktree"` parameter) on a focused chunk of work, then commits to the shared feature branch.

**Plan A independent task groups** (run in parallel after Phase 1-3 land sequentially):
- Phase 4 (`text interactive` wiring)
- Phase 5 (`text process` wiring)
- Phase 6 (`query` wiring)
- Phase 8 (`phentrieve config` subcommand)
- Phase 10 (Frontend parity — `defaults.js` fix + parity test) — fully independent, can run from the start
- Phase 11 (Documentation rewrite of `configuration-profiles.md`) — independent of code phases

**Plan A sequential**:
- Phase 1 → 2 → 3 (foundation, can't be parallelized)
- Phase 7 (root `--profile`) depends on 4-6
- Phase 9 (`--show-resolved-config`) depends on 3
- Phase 12 (CHANGELOG + integration tests) at the end

**Plan B independent task groups**:
- Phase 1 (OrchestrationResult dataclass) — can run in parallel with Plan A
- Phase 4 (sub-chunking) and Phase 5 (score-improvement gate) — both depend on Phase 2 (config); independent of each other
- Phase 8 (CLI flags), Phase 9 (API), Phase 10 (frontend pass-through) — all independent of each other once Phase 2 lands

**Plan B sequential**:
- Phase 1 → 2 → (3, 4, 5 parallel) → 6 → 7
- Phase 11 (docs), Phase 12 (integration tests), Phase 13 (CHANGELOG) at the end

### File-overlap conflict zones

These files are touched by both plans. Sequence touches to avoid conflicts:

- `phentrieve/cli/text_commands.py` — Plan A Phase 5 adds `--profile`; Plan B Phase 8 adds `--adaptive-rechunking-*` flags. Run Plan B Phase 8 only after Plan A Phase 5 has merged into the feature branch.
- `phentrieve.yaml` and `phentrieve.yaml.template` — both plans append sections. Sequence Plan A's YAML edits before Plan B's.
- `CHANGELOG.md` — append-only; conflicts unlikely if both plans append at the same heading. Have a single subagent reconcile if a conflict arises.
- `docs/user-guide/index.md` — both add a link. Trivial conflict; resolve in either subagent.

### Subagent prompt template

When dispatching a subagent for a task group:

```
You're executing tasks N..M of Plan A (or B). Read these files in full:
- .planning/specs/2026-04-25-<spec>.md
- .planning/active/2026-04-25-<plan>.md

Execute tasks N..M in TDD order: failing test → minimal impl → passing
test → commit. Each task has its own draft commit message in the plan;
use it. Do not skip any test step.

Constraints:
- Stay within tasks N..M. Do not touch files outside the task's File
  Structure entries.
- Do not modify the spec or plan documents.
- Run `make check && make typecheck-fast && make test` before final commit.
- For frontend tasks: also `make frontend-test-ci`.
- Use uv for Python; do not use pip.
- Follow project AGENTS.md conventions.

Return a summary of: tasks completed, commit SHAs, any deviations from
the plan and why, any spec questions to escalate.
```

Use the **superpowers:executing-plans** skill (or **superpowers:subagent-driven-development** if you need finer-grained per-task agents) — they encode the right discipline.

## Pre-flight checklist before dispatching the first subagent

1. **Create worktree and branch**: `git worktree add .worktrees/profiles-adaptive -b feat/cli-profiles-and-adaptive-rechunking main`. Subsequent agents work inside that worktree.
2. **Verify clean baseline**: from inside the worktree, `make check && make typecheck-fast && make test` should pass on `main` before any task starts. If it doesn't, fix the baseline first.
3. **Confirm dependency landing**: tasks in later phases assume earlier phases have merged. Don't dispatch later phases until earlier ones report completion and the test suite is green.

## Acceptance gates before opening the PR

- [ ] All tasks in both plans checked off.
- [ ] `make check && make typecheck-fast && make test` passes.
- [ ] `make frontend-test-ci && make frontend-build-ci` passes.
- [ ] `make ci-local && make precommit` passes.
- [ ] Spec self-review checklists at the bottom of each plan are run.
- [ ] CHANGELOG.md has the expected entries (Plan A: 3 entries; Plan B: 1 entry).
- [ ] Both `tests/integration/test_documented_yaml.py` and `tests/integration/test_specA_specB_interaction.py` pass.
- [ ] PR description references issues #28, #171, #148 and links to both spec files.

## What NOT to do

- Do not re-design from the spec. The specs went through multiple review rounds; trust them.
- Do not split the work across two PRs. The user wants one branch, one PR.
- Do not open additional GitHub issues from the future-work register — #234 and #235 are the only two that should exist.
- Do not change behavior beyond what the spec documents. If you find a bug in the spec during execution, escalate to the user instead of patching silently.

## Kickoff prompt for the new session

Paste this into the new session as your first message:

```
Execute the two plans at .planning/active/2026-04-25-cli-profiles-default-resolution-plan.md
and .planning/active/2026-04-25-adaptive-rechunking-plan.md. Read
.planning/active/EXECUTION-HANDOFF.md first — it has the parallelization
strategy, worktree setup, file-overlap zones, subagent prompt template,
and the acceptance gates. Land everything on one branch
(feat/cli-profiles-and-adaptive-rechunking) and open one PR at the end.
Use superpowers (subagent-driven-development for per-task agents,
dispatching-parallel-agents for the independent task groups identified
in the handoff doc).
```
