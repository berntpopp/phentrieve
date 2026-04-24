# Maintainability Quick Wins — Design Spec

**Date**: 2026-04-15
**Goal**: Raise maintainability quickly by improving codebase truthfulness, safety, and boundary clarity without starting the larger hotspot refactors yet.
**Approach**: Small, parallel workstreams focused on API model-loading safety, CI and typecheck truthfulness, test taxonomy cleanup, and docs alignment.
**Timeline**: 1-2 weeks
**Regression policy**: Zero behavior drift beyond the explicitly intended API hardening and CI enforcement changes.

---

## Context

The maintainability review in `.planning/analysis/CODEBASE-MAINTAINABILITY-REVIEW-2026-04-15.md` identified two distinct classes of work:

1. Large structural refactors with high eventual payoff but higher coupling and execution risk.
2. Small trust-restoring fixes that improve maintainability immediately and can be executed in parallel.

This spec covers only the second class. The objective is to make the repository easier to change safely before taking on the larger decomposition work in `query_orchestrator.py`, `assertion_detection.py`, `hpo_parser.py`, and `frontend/src/components/QueryInterface.vue`.

## Problem Statement

Maintainability is currently reduced by four practical failures:

- The API accepts request-level model-loading knobs that are too permissive for a server boundary.
- CI does not fully enforce the same standards developers rely on locally.
- Test metadata and categorization are not fully trustworthy, which weakens selective runs and review confidence.
- Developer docs no longer match the actual `uv`-based workflow and current CI/test commands.

These issues create friction disproportionate to their implementation size. They also reduce confidence in future refactoring work because the feedback loop is weaker than it should be.

## Goals

- Remove user-controlled remote model code execution from the text-processing API boundary.
- Make CI authoritative for mypy and catch pytest collection/marker drift early.
- Restore trust in test classification with a minimal, high-signal cleanup pass.
- Align docs with the current repository workflow and command surface.

## Non-Goals

- Full configuration redesign in `phentrieve/config.py`
- Splitting backend orchestration modules
- Splitting frontend mega-components
- Full reorganization of the `tests/` tree
- New product features or UX changes

## Recommended Workstreams

### Workstream A: API Safety Boundary

The text-processing request schema currently exposes `trust_remote_code`, and the router passes it through directly to model loading. That is the wrong boundary for a server process. Model trust policy should be server-owned, not caller-owned.

Design decisions:

- Remove `trust_remote_code` from the external request schema.
- Replace free-form caller model selection with a server-side allowlist for retrieval and semantic chunking models.
- Reject unsupported model names with a clear 4xx response rather than silently attempting arbitrary loads.
- Replace `x or default` numeric fallback patterns in the text-processing router with explicit `is not None` handling so valid `0.0` values are preserved.

Expected result: safer runtime behavior, clearer config ownership, and fewer hidden request-path surprises.

### Workstream B: CI Truthfulness

The repository already expects `make check`, `make typecheck-fast`, and `make test` to matter. CI should reflect that by failing when type checks fail and by catching pytest collection problems before the main test phase.

Design decisions:

- Remove the unused `type: ignore` in `api/dependencies.py`.
- Make the existing mypy step blocking in `.github/workflows/ci.yml`.
- Add a pytest collection-only smoke step using strict markers so marker drift fails early and cheaply.

Expected result: CI becomes a dependable signal instead of a partial advisory system.

### Workstream C: Test Taxonomy Truthfulness

The immediate maintainability issue is not the entire test tree layout. It is that marker usage and practical test meaning do not fully match.

Design decisions:

- Fix the undeclared `benchmark` marker mismatch in the benchmark integration test path.
- Remove obviously contradictory marker combinations where a test is labeled both `unit` and `integration`.
- Defer large-scale directory moves and broad taxonomy cleanup to a later plan once the quick wins are merged.

Expected result: test intent becomes easier to understand and selective test execution becomes more trustworthy.

### Workstream D: Docs as Source of Truth

Several docs pages still describe obsolete commands, manual virtualenv activation, or outdated CI file names. This causes repeated confusion and unnecessary support overhead.

Design decisions:

- Update the development and installation docs to use the current `uv` and `make` workflow.
- Remove instructions that depend on `pip` or `venv` as the primary path, while keeping a short fallback note only if still intentionally supported.
- Correct CI file references and test command examples to match the current repo.
- Prefer links into `.planning/` and current commands over historical paths.

Expected result: onboarding and contribution docs match the actual toolchain and reduce avoidable developer friction.

## Parallel Execution Model

These workstreams are intentionally separable:

- Workstream A touches `api/` runtime boundary files.
- Workstream B touches CI and one typecheck cleanup file.
- Workstream C touches pytest config and specific test modules.
- Workstream D touches docs and `README`-adjacent workflow guidance.

The only coordination point is between Workstreams B and C around pytest marker/config changes. That overlap is small and should be handled by sequencing the final merge of the pytest-related edits.

## Architecture and Boundary Rationale

This tranche improves maintainability by making boundaries more explicit:

- Server-owned policy belongs in server config, not request schemas.
- CI-owned quality gates belong in CI as hard requirements, not advisory output.
- Test labels should describe reality, not aspiration.
- Docs should reflect the actual workflow surface, not historical residue.

These are high-leverage changes because they improve the feedback loop around the entire codebase, including future refactors.

## Testing Strategy

- Add or update focused API tests covering rejected model names and correct handling of explicit `0.0` threshold values.
- Run collection-only pytest checks to validate markers before running the full suite.
- Run targeted test commands for changed files first, then the repo-standard verification commands.
- Confirm docs examples by checking that referenced commands exist in `Makefile`, `pyproject.toml`, or current workflows.

## Success Criteria

- No request field allows caller-controlled `trust_remote_code`.
- Unsupported text-processing model names are rejected explicitly.
- CI fails on mypy errors and on pytest collection/marker errors.
- Benchmark-related test metadata is declared and non-contradictory.
- The main install/test/dev docs match the current repository workflow.

## Risks and Mitigations

- Risk: existing clients may rely on arbitrary model names.
  Mitigation: document the allowlist and return explicit validation errors.

- Risk: making mypy blocking could expose unrelated latent failures.
  Mitigation: land the one-line cleanup first and verify locally before changing CI semantics.

- Risk: test taxonomy cleanup could sprawl.
  Mitigation: limit this tranche to obvious marker truthfulness fixes only.

- Risk: docs drift can reappear.
  Mitigation: align docs against `Makefile`, `pyproject.toml`, and workflow files rather than duplicating separate command variants.

## Follow-On Plans

After these quick wins, the next maintainability plan should target the first structural refactor tranche:

- `phentrieve/retrieval/query_orchestrator.py`
- `phentrieve/text_processing/assertion_detection.py`
- `phentrieve/data_processing/hpo_parser.py`
- `frontend/src/components/QueryInterface.vue`

Those refactors will have better odds of success once the codebase feedback loop is trustworthy again.
