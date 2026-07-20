# Benchmark Resume Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR #322 checkpoint reuse complete and non-destructive, then publish and document the final identity contract.

**Architecture:** Preserve the existing Result Store and CLI. Defer LLM artifact cleanup until after the preserved checkpoint passes both fingerprint and full-configuration validation. Extend dataset identity with explicit schema/projection semantics, expose the existing HPO expectation through Typer, and keep manifest-v1 aliases while publishing v2 only once.

**Tech Stack:** Python 3.11, Typer, pytest, Ruff, mypy, Markdown.

## Global Constraints

- Do not change retrieval or extraction overwrite behavior.
- Do not add a destructive restart flag.
- Do not mention downstream repository workflows.
- Every behavior fix follows a red-green test cycle.

---

### Task 1: Restore complete checkpoint validation

**Files:**
- Modify: `tests/unit/test_llm_benchmark.py`
- Modify: `phentrieve/benchmark/llm_cli.py`

**Interfaces:**
- Consumes: `_build_checkpoint_identity(...) -> dict[str, Any]`
- Produces: `_load_checkpoint_payload(..., current_run=..., execution_fingerprint=..., scoring_fingerprint=...)`

- [ ] Add a test whose checkpoint fingerprints match but whose
  `capture_phase1_debug` differs from `current_run`; assert a configuration
  mismatch.
- [ ] Run the focused test and confirm it fails because `current_run` is ignored
  when fingerprints are supplied.
- [ ] Make `_load_checkpoint_payload` validate fingerprints and then validate
  `current_run` independently; include new-run-id/removal remediation in every
  mismatch error.
- [ ] Pass `checkpoint_identity` from `run_llm_benchmark_cli` and rerun the
  focused checkpoint tests.

### Task 2: Validate before overwrite cleanup

**Files:**
- Modify: `tests/unit/test_llm_benchmark.py`
- Modify: `tests/unit/benchmark/test_result_store.py`
- Modify: `phentrieve/benchmark/result_store.py`
- Modify: `phentrieve/benchmark/llm_cli.py`

**Interfaces:**
- Produces: `create_run_layout(..., reset_existing: bool = True)`
- Produces: `reset_run_artifacts(run_dir: Path) -> None`

- [ ] Add an LLM CLI test that writes a checkpoint plus summary, triggers a
  fingerprint mismatch with `--overwrite`, and asserts the summary is unchanged.
- [ ] Run it and confirm the existing eager `_reset_run_dir` deletes the summary.
- [ ] Allow the LLM caller to defer reset, validate the checkpoint, then call
  `reset_run_artifacts`; keep the existing default for retrieval/extraction.
- [ ] Add/adjust Result Store tests for explicit deferred reset and rerun both
  focused suites.

### Task 3: Complete identity and HPO contracts

**Files:**
- Modify: `tests/unit/benchmark/test_run_identity.py`
- Modify: `tests/unit/cli/test_benchmark_commands.py`
- Modify: `phentrieve/benchmark/run_identity.py`
- Modify: `phentrieve/benchmark/llm_cli.py`
- Delete: `tests/fixtures/benchmark_identity/dataset_identity.json`
- Delete: `tests/fixtures/benchmark_identity/two_phase_identity.json`

**Interfaces:**
- Produces: `DatasetIdentity.schema_version`
- Produces: `DatasetIdentity.projection_sha256`
- Produces: `--evaluation-hpo-version TEXT`

- [ ] Add tests proving projection mapping changes the scoring fingerprint and
  the dataset identity carries `phentrieve-dataset-identity/v1`.
- [ ] Add a CLI test proving `--evaluation-hpo-version` reaches
  `run_llm_benchmark_cli`.
- [ ] Implement the minimal fields/forwarding and pass the runtime projection
  mapping from `DATASET_ASSERTION_PROJECTION`.
- [ ] Remove the two unused stale JSON examples and rerun identity/CLI tests.

### Task 4: Remove redundant publication and document the contract

**Files:**
- Modify: `phentrieve/benchmark/llm_cli.py`
- Modify: `docs/user-guide/benchmarking-guide.md`
- Modify: `CHANGELOG.md`

- [ ] Remove the partial-run `write_manifest` call, its LLM-only helper, and the
  now-unused import; retain the schema-v2 partial-manifest test.
- [ ] Document prerequisites, HPO assertion, identities/fingerprints, schema
  ownership, aliases, checkpoint migration, and remediation.
- [ ] Add concise `Unreleased` Added/Changed entries.
- [ ] Run `git diff --check` and the focused benchmark suites.

### Task 5: Verify and publish

**Files:** all files changed above.

- [ ] Run `make check`.
- [ ] Run `make typecheck-fast`.
- [ ] Run `make test` and record exact counts.
- [ ] Confirm a clean intended diff and no downstream-repository references.
- [ ] Commit the focused follow-up, push `feat/benchmark-identity-v2-main`, and
  recheck PR #322 CI.

### Task 6: Close merge-readiness edge cases

**Files:**
- Modify: `phentrieve/benchmark/result_store.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `phentrieve/benchmark/llm_cli.py`
- Modify: `tests/unit/test_llm_benchmark.py`

- [ ] Reject overwrite for a pre-existing run whose checkpoint is missing or is
  not a JSON object, without deleting any artifact.
- [ ] Share a versioned assertion-projection descriptor between runtime scoring
  and dataset identity, including the normalized-passthrough fallback.
- [ ] Bind producer version/commit provenance to complete checkpoint
  configuration validation.
- [ ] Run the focused identity, Result Store, CLI, prompt, and LLM suites.
