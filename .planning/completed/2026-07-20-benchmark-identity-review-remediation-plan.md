# Benchmark Identity Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every actionable finding from the adversarial review of PR #322 with versioned, runtime-faithful benchmark identities and crash-safe artifact publication.

**Architecture:** Canonical assertion, endpoint, producer, and retrieval contracts are constructed once and threaded into runtime and persistence. LLM artifacts are written into immutable generations and become visible by one atomic manifest replacement, while existing fixed-root layouts remain readable through compatibility aliases.

**Tech Stack:** Python 3.11 typing, dataclasses, pathlib, hashlib, FastAPI-era Pydantic, Ruff, mypy, pytest/xdist, uv.

## Global Constraints

- Keep all tests under `tests/` and all planning artifacts under `.planning/`.
- Use `uv`; do not use `pip`.
- Use modern Python typing and ASCII unless an existing file requires otherwise.
- Old identity checkpoints are rejected before any filesystem mutation.
- Existing result manifests remain readable; old checkpoints are not silently migrated.
- Raw provider exception strings and endpoint credentials must never be persisted.
- Complete with `make check`, `make typecheck-fast`, `make test`, `make ci-python-quality`, and Python 3.12/3.13 compatibility gates.

---

### Task 1: Canonical Assertion And Projection Contract

**Files:**
- Modify: `phentrieve/benchmark/data_loader.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `phentrieve/benchmark/run_identity.py`
- Test: `tests/unit/benchmark/test_run_identity.py`
- Test: `tests/unit/test_llm_benchmark.py`

**Interfaces:**
- Produces: `normalize_benchmark_assertion(value: object | None, *, reject_unknown: bool) -> str`
- Produces: `effective_dataset_assertion_projection(dataset: str, source_dataset: str | None = None) -> dict[str, str | None] | None`
- Produces: `describe_dataset_assertion_projection(dataset: str) -> dict[str, object]` using schema `phentrieve-assertion-projection/v2`

- [x] **Step 1: Add failing normalization and aggregate-projection tests**

Cover directory and JSON annotations for `absent`, uppercase/whitespace variants, tuple/list terms, missing assertions, unknown explicit gold assertions, and `dataset="all"` source-specific CSC/GSC versus passthrough behavior. Assert that changing the effective normalization table changes `projection_sha256` and `scoring_sha256`.

- [x] **Step 2: Run the focused tests and confirm the current behavior fails**

Run: `uv run pytest tests/unit/benchmark/test_run_identity.py tests/unit/test_llm_benchmark.py -n 0 -k "assertion or projection or absent"`

Expected: failures showing `absent` becomes `PRESENT`, JSON assertions remain raw, and the `all` descriptor is plain passthrough.

- [x] **Step 3: Implement the canonical normalizer and shared projection**

Use the following normalization contract in `data_loader.py`:

```python
CANONICAL_ASSERTION_MAP: dict[str, str] = {
    "present": "PRESENT",
    "affirmed": "PRESENT",
    "normal": "PRESENT",
    "absent": "ABSENT",
    "negated": "ABSENT",
    "uncertain": "UNCERTAIN",
    "family_history": "FAMILY_HISTORY",
}

def normalize_benchmark_assertion(
    value: object | None, *, reject_unknown: bool
) -> str:
    if value is None or not str(value).strip():
        return DEFAULT_SIMPLE_ASSERTION
    normalized = str(value).strip().casefold()
    if normalized in CANONICAL_ASSERTION_MAP:
        return CANONICAL_ASSERTION_MAP[normalized]
    if reject_unknown:
        raise ValueError(f"Unknown benchmark assertion: {value!r}")
    return DEFAULT_SIMPLE_ASSERTION
```

Normalize every gold form in `parse_gold_terms()` and directory conversion. In runtime projection, select `source_dataset` when present, apply normalization once, and then apply the selected source mapping. Remove the `projection=None`/`positive_hpo_present_v1` identity fallback and require the v2 descriptor.

- [x] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/benchmark/test_run_identity.py tests/unit/test_llm_benchmark.py -n 0 -k "assertion or projection or absent"`

Expected: all selected tests pass.

- [x] **Step 5: Commit**

```text
git add phentrieve/benchmark/data_loader.py phentrieve/benchmark/llm_benchmark.py phentrieve/benchmark/run_identity.py tests/unit/benchmark/test_run_identity.py tests/unit/test_llm_benchmark.py
git commit -m "fix: canonicalize benchmark assertion contracts"
```

### Task 2: Canonical Provider Endpoint And Safe Persisted Errors

**Files:**
- Modify: `phentrieve/llm/providers/resolver.py`
- Modify: `phentrieve/llm/providers/openai.py`
- Modify: `phentrieve/llm/providers/ollama.py`
- Modify: `phentrieve/llm/providers/anthropic.py`
- Modify: `phentrieve/benchmark/run_identity.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Test: `tests/unit/llm/test_provider.py`
- Test: `tests/unit/benchmark/test_run_identity.py`
- Test: `tests/unit/test_llm_benchmark.py`

**Interfaces:**
- Produces: `canonicalize_llm_base_url(value: str | None) -> str | None`
- Extends: `sanitize_behavioral_base_url(value, *, secrets=())` with opaque
  path/query identities and known-secret redaction
- Produces: `_public_pipeline_error(exc: LLMPipelinePhaseError) -> tuple[str, str]`

- [x] **Step 1: Add failing URL and secret-canary tests**

Test blank/whitespace, root and nested trailing slash, IPv6, query, fragment,
userinfo, environment-derived URLs, Gemini rejection, sensitive query keys,
percent-encoded known secrets, and provider exceptions containing URL/header
canaries. Serialize nested persisted payloads and assert no canary remains.

- [x] **Step 2: Run focused tests and confirm failures**

Run: `uv run pytest tests/unit/llm/test_provider.py tests/unit/benchmark/test_run_identity.py tests/unit/test_llm_benchmark.py -n 0 -k "base_url or endpoint or credential or error_message"`

Expected: trailing-slash mismatch and secret persistence failures.

- [x] **Step 3: Implement one canonical endpoint path**

`canonicalize_llm_base_url()` must trim, parse, reject fragments, require a hostname, preserve non-secret query behavior, and remove trailing path slashes without applying `rstrip()` to the whole raw URL. Provider constructors store the supplied canonical value unchanged. Endpoint identity removes userinfo and redacts sensitive query values plus exact known secrets before display or hashing.

Persist failures as stable pairs such as `("provider_error", "Provider request failed")`; keep `str(exc)` only in logger exception context and exception chaining.

- [x] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all selected tests pass.

- [x] **Step 5: Commit**

```text
git add phentrieve/llm/providers phentrieve/benchmark/run_identity.py phentrieve/benchmark/llm_benchmark.py tests/unit/llm/test_provider.py tests/unit/benchmark/test_run_identity.py tests/unit/test_llm_benchmark.py
git commit -m "fix: canonicalize persisted provider identity"
```

### Task 3: Versioned Producer And Scoring Fingerprints

**Files:**
- Modify: `phentrieve/benchmark/run_identity.py`
- Modify: `phentrieve/benchmark/llm_cli.py`
- Test: `tests/unit/benchmark/test_run_identity.py`
- Test: `tests/unit/test_llm_benchmark.py`

**Interfaces:**
- Produces: `build_producer_identity(package_root: Path | None = None) -> dict[str, str | None]`
- Changes: `build_run_fingerprints(dataset, prompt, model, asset, *, scoring, producer_source_sha256) -> RunFingerprints`
- Adds schema: `phentrieve-run-fingerprint/v2` and `phentrieve-producer-source/v1`

- [x] **Step 1: Add failing producer and scoring tests**

Assert source edits change both fingerprints, pycache/docs do not, LF/CRLF normalize equally, full 40/64 hexadecimal commits resolve, invalid lengths fail closed, and checkout/package inventories with identical runtime files hash equally. Assert ontology enabled/floor/formula/HPO and assertion-algorithm descriptors change scoring identity.

- [x] **Step 2: Run focused tests and confirm failures**

Run: `uv run pytest tests/unit/benchmark/test_run_identity.py tests/unit/test_llm_benchmark.py -n 0 -k "producer or fingerprint or ontology"`

- [x] **Step 3: Implement deterministic source inventory and v2 fingerprints**

Hash sorted runtime package files with POSIX-relative paths; normalize CRLF/CR to LF for text and exclude `__pycache__`, `.pyc`, distribution metadata, docs, and tests. Include runtime-relevant untracked files. Keep commit/dirty fields descriptive and use `source_sha256` for compatibility. Put `source_sha256` into execution and scoring payloads and include a versioned exact scoring descriptor.

- [x] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all selected tests pass.

- [x] **Step 5: Commit**

```text
git add phentrieve/benchmark/run_identity.py phentrieve/benchmark/llm_cli.py tests/unit/benchmark/test_run_identity.py tests/unit/test_llm_benchmark.py
git commit -m "fix: bind fingerprints to producer and scoring contracts"
```

### Task 4: Verified Retrieval Content And Runtime Configuration

**Files:**
- Modify: `phentrieve/data_processing/bundle_manifest.py`
- Modify: `phentrieve/data_processing/bundle_packager.py`
- Modify: `phentrieve/benchmark/run_identity.py`
- Modify: `phentrieve/llm/tools.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `phentrieve/benchmark/llm_cli.py`
- Test: `tests/unit/data_processing/test_bundle_manifest.py`
- Test: `tests/unit/data_processing/test_bundle_packager.py`
- Test: `tests/unit/benchmark/test_run_identity.py`
- Test: `tests/unit/test_llm_benchmark.py`

**Interfaces:**
- Produces: `verify_bundle_inventory(manifest: BundleManifest, data_dir: Path) -> str`
- Extends: `RetrievalAssetIdentity` with model revision, trust/code revision, vector mode, and verified content digest
- Extends: `ToolExecutor.__init__()` with explicit model metadata and `index_dir: Path`

- [x] **Step 1: Add failing manifest and runtime-binding tests**

Cover empty/incomplete checksums, missing DB/indexes, tampering, absolute/parent traversal, escaping symlinks, POSIX/Windows separator equivalence, and exact propagation of manifest model/revision/trust/code revision/vector mode/index path into `load_embedding_model()` and `DenseRetriever.from_model_name()`.

- [x] **Step 2: Run focused tests and confirm failures**

Run: `uv run pytest tests/unit/data_processing/test_bundle_manifest.py tests/unit/data_processing/test_bundle_packager.py tests/unit/benchmark/test_run_identity.py tests/unit/test_llm_benchmark.py -n 0 -k "checksum or inventory or retrieval or multi_vector"`

- [x] **Step 3: Implement fail-closed verification and runtime threading**

Require checksum entries for `hpo_data.db` and `indexes/`, validate every key below `data_dir`, reject symlink escape, and calculate a canonical content digest from verified role/path/checksum triples. Pass the exact manifest model metadata and `data_dir / "indexes"` into `ToolExecutor`; remove runtime dependence on floating defaults for benchmark execution.

- [x] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all selected tests pass.

- [x] **Step 5: Commit**

```text
git add phentrieve/data_processing phentrieve/benchmark/run_identity.py phentrieve/llm/tools.py phentrieve/benchmark/llm_benchmark.py phentrieve/benchmark/llm_cli.py tests/unit/data_processing tests/unit/benchmark/test_run_identity.py tests/unit/test_llm_benchmark.py
git commit -m "fix: bind benchmark runtime to verified retrieval content"
```

### Task 5: Zero-Mutation Layout Validation And Robust Discovery

**Files:**
- Modify: `phentrieve/benchmark/result_store.py`
- Modify: `phentrieve/benchmark/llm_cli.py`
- Test: `tests/unit/benchmark/test_result_store.py`
- Test: `tests/unit/test_llm_benchmark.py`

**Interfaces:**
- Produces: `plan_run_layout(...) -> RunLayout` without filesystem writes
- Produces: `validate_run_boundary(layout: RunLayout) -> None`
- Produces: `materialize_run_layout(layout: RunLayout) -> None`

- [x] **Step 1: Add failing zero-mutation, malformed-manifest, symlink, and Windows reparse-point tests**

Snapshot the complete tree before missing/non-object/pre-v2/mismatched checkpoint rejection. Add malformed manifest roots and `artifacts` values alongside one valid manifest. Add outside sentinels behind final-component and parent-component links; use a Windows-gated junction test.

- [x] **Step 2: Run focused tests and confirm failures**

Run: `uv run pytest tests/unit/benchmark/test_result_store.py tests/unit/test_llm_benchmark.py -n 0 -k "manifest or checkpoint or symlink or junction or mutation"`

- [x] **Step 3: Separate planning from materialization and harden discovery**

Compute paths without `mkdir`, validate checkpoint and boundary first, then materialize. Reject symlink/reparse components and revalidate before mutation. Require parsed manifest and artifact inventory to be mappings before `.get()`/`.values()` and skip malformed files independently.

- [x] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all selected tests pass.

- [x] **Step 5: Commit**

```text
git add phentrieve/benchmark/result_store.py phentrieve/benchmark/llm_cli.py tests/unit/benchmark/test_result_store.py tests/unit/test_llm_benchmark.py
git commit -m "fix: validate benchmark storage before mutation"
```

### Task 6: Immutable Generation Publication

**Files:**
- Modify: `phentrieve/benchmark/result_store.py`
- Modify: `phentrieve/benchmark/llm_cli.py`
- Test: `tests/unit/benchmark/test_result_store.py`
- Test: `tests/unit/test_llm_benchmark.py`
- Test: `tests/integration/test_benchmark_workflow.py`

**Interfaces:**
- Changes: `publish_manifest_v2()` copies a complete immutable generation,
  hashes it, and atomically replaces only the root manifest
- Produces: `active_checkpoint_path(layout: RunLayout) -> Path`

- [x] **Step 1: Add failing failure-injection and compatibility tests**

Start from a complete run, reject invalid generation inputs before manifest
replacement, and assert old manifest bytes and referenced checksums remain valid.
Test successful switches, bounded rollback retention, legacy fixed-root checkpoint
fallback, generation checkpoint resume, relative result roots, and compatibility
working copies.

- [x] **Step 2: Run focused tests and confirm failures**

Run: `uv run pytest tests/unit/benchmark/test_result_store.py tests/unit/test_llm_benchmark.py tests/integration/test_benchmark_workflow.py -n 0 -k "generation or overwrite or checkpoint or manifest"`

- [x] **Step 3: Implement manifest-as-commit publication**

Copy all authoritative artifacts below `.generations/<unique-id>/`; build and
hash the inventory there; write a unique temporary manifest beside root
`manifest.json`; atomically replace the root manifest only after the generation
is complete. Resolve resume checkpoints through the active manifest, fall back
to root checkpoint for old layouts, treat root files as non-authoritative
working copies, and retain only the active plus one rollback generation.

- [x] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all selected tests pass.

- [x] **Step 5: Commit**

```text
git add phentrieve/benchmark/result_store.py phentrieve/benchmark/llm_cli.py tests/unit/benchmark/test_result_store.py tests/unit/test_llm_benchmark.py tests/integration/test_benchmark_workflow.py
git commit -m "fix: publish benchmark runs as immutable generations"
```

### Task 7: Migration Documentation And Full Verification

**Files:**
- Modify: `docs/user-guide/benchmarking-guide.md`
- Modify: `CHANGELOG.md`
- Modify: `.planning/active/2026-07-20-benchmark-identity-review-remediation-plan.md`

**Interfaces:**
- Consumes all prior task interfaces.

- [x] **Step 1: Document versioned identity, endpoint, retrieval, assertion, and generation behavior**

State that credentials are excluded, arbitrary path credentials are unsupported, old checkpoints require a new run ID or deliberate removal, manifests are authoritative, aliases are compatibility-only, and unknown gold assertions are rejected.

- [x] **Step 2: Run focused remediation suites serially**

Run: `uv run pytest tests/unit/benchmark/test_run_identity.py tests/unit/benchmark/test_result_store.py tests/unit/test_llm_benchmark.py tests/unit/llm/test_provider.py tests/unit/llm/test_pipeline.py tests/unit/data_processing/test_bundle_manifest.py tests/unit/data_processing/test_bundle_packager.py tests/integration/test_benchmark_workflow.py -n 0`

Expected: all pass.

- [x] **Step 3: Run required repository gates**

Run each command and record its result:

```text
make check
make typecheck-fast
make test
make ci-python-quality
make ci-python-compat PYTHON=3.12
make ci-python-compat PYTHON=3.13
```

- [x] **Step 4: Move this plan to completed and update `.planning/README.md`**

Move the plan to `.planning/completed/2026-07-20-benchmark-identity-review-remediation-plan.md`, mark every checkbox complete, and link it from the planning index.

- [x] **Step 5: Commit**

```text
git add CHANGELOG.md docs/user-guide/benchmarking-guide.md .planning/README.md .planning/completed/2026-07-20-benchmark-identity-review-remediation-plan.md
git commit -m "docs: complete benchmark identity review remediation"
```

### Task 8: Final Agentic Review Remediation

- [x] **Step 1: Close storage and publication findings**

Support relative output roots, detect Windows reparse points, use unique manifest
temporary files, and retain only the active and previous immutable generations.

- [x] **Step 2: Close identity and persistence findings**

Require immutable retrieval revisions, make checkout and wheel producer hashes
equivalent, use the effective per-document assertion projection, and persist only
opaque endpoint components plus stable group error descriptors.

- [x] **Step 3: Add regression coverage and align documentation**

Cover relative roots, malformed manifests, real Windows junctions, wheel source
hashes, endpoint secret canaries, and bounded generation retention. Document the
single-writer publication contract and non-authoritative root working copies.

- [x] **Step 4: Re-run specialized reviewers and all repository gates**

Repeat the focused suite, request verification of every final-review finding,
then run every command listed in Task 7 Step 3 against the resulting code.
