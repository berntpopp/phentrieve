# HPO Data Release Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the verified HPO `v2026-06-23` minimal, single-vector, and multivector bundle matrix for all nine supported retrieval models from a dedicated data-release repository, while separating this lifecycle from Phentrieve software releases.

**Architecture:** Pin data builds to immutable HPO bytes, a Phentrieve source commit, lockfile, and model revisions. Enforce index completeness/provenance in the library before a bundle can be made. The software repository supplies a reusable build and verification workflow; `berntpopp/phentrieve-data` owns data release specifications, draft assets, immutable publication, and release history.

**Tech Stack:** Python 3.13, uv, PyTorch CUDA 13, SentenceTransformers, ChromaDB, GitHub Actions reusable workflows, GitHub Releases, GitHub CLI, pytest, Ruff, mypy.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `phentrieve/data_processing/release_contract.py` | Immutable release spec, matrix, upstream digest, and validation types. |
| `phentrieve/data_processing/hpo_parser.py` | Hash-verified input preparation and correct release-date metadata. |
| `phentrieve/indexing/chromadb_orchestrator.py` | Resolve pinned ontology provenance and pass it into index construction. |
| `phentrieve/indexing/chromadb_indexer.py` | Persist accurate collection metadata and fail incomplete index builds. |
| `phentrieve/data_processing/bundle_manifest.py` | Versioned provenance fields for a published bundle. |
| `phentrieve/data_processing/bundle_packager.py` | Reject incompatible/incomplete collections and package the validated manifest. |
| `phentrieve/data_processing/bundle_downloader.py` | Configurable data repository with legacy override support. |
| `scripts/verify_data_release.py` | Full 19-bundle release verifier and retrieval smoke checker. |
| `.github/workflows/build-data-bundles.yml` | Reusable, SHA-pinned builder without software-release trigger. |
| `docker-compose.yml`, `.env.docker.template`, Docker docs | Dedicated data-repository URL and migration guidance. |
| `berntpopp/phentrieve-data` | Metadata-only data repository, release specs, and caller workflow. |

## Task 1: Define The Release Contract

**Files:**
- Create: `phentrieve/data_processing/release_contract.py`
- Create: `tests/unit/data_processing/test_release_contract.py`
- Modify: `phentrieve/data_processing/__init__.py`

- [ ] **Step 1: Write failing contract tests**

Write a fixture for `hpo-v2026-06-23-r1` and assert the nine-model matrix, the exact source hash, and counts:

```python
assert spec.hpo_version == "v2026-06-23"
assert len(spec.models) == 9
assert spec.expected_document_count("single_vector") == 19836
assert spec.expected_document_count("multi_vector") == 63428
assert spec.hpo_sha256 == (
    "3b646565695329aa399e937883c68d5d424d0df5eaab2f22baa0e08d44fdbe87"
)
```

Also assert construction rejects duplicate slugs, an invalid 64-character SHA-256, a non-40-character source SHA, and unknown index modes.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -n 0 tests/unit/data_processing/test_release_contract.py -v
```

Expected: import failure because `release_contract` does not exist.

- [ ] **Step 3: Implement the smallest immutable contract**

Implement frozen `ModelReleaseSpec` and `DataReleaseSpec` dataclasses. Define `DATA_RELEASE_MODELS` once with the nine names/slugs and expose `expected_document_count(index_type)`. Validate tag pattern, digests, full commit SHA, unique slugs, model revisions, and the two expected count values.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: all contract tests pass.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/data_processing/release_contract.py   phentrieve/data_processing/__init__.py   tests/unit/data_processing/test_release_contract.py
git commit -m "feat(data): define immutable release contract"
```

## Task 2: Verify HPO Source Bytes

**Files:**
- Modify: `phentrieve/data_processing/hpo_parser.py`
- Modify: `phentrieve/data_processing/hpo_database.py`
- Modify: `tests/unit/data_processing/test_hpo_parser.py`

- [ ] **Step 1: Write failing source-provenance tests**

Add a local JSON fixture and assert a matching expected SHA-256 is accepted while a mismatch raises `ValueError` before parsing. Add a preparation assertion that `v2026-06-23` stores `hpo_release_date=2026-06-23`, `hpo_download_date` separately, and the verified source digest.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -n 0 tests/unit/data_processing/test_hpo_parser.py -k "sha256 or release_date" -v
```

Expected: failure because digest verification and release-date metadata do not exist.

- [ ] **Step 3: Implement verified preparation**

Add a streaming SHA-256 helper. Thread an optional expected digest through the download/preparation API; in release mode it is required and validates both downloaded and existing `hp.json`. Store separate `hpo_release_date`, `hpo_download_date`, and `hpo_source_sha256` metadata values.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2; then:

```bash
git add phentrieve/data_processing/hpo_parser.py   phentrieve/data_processing/hpo_database.py   tests/unit/data_processing/test_hpo_parser.py
git commit -m "fix(data): verify HPO source provenance"
```

## Task 3: Reject Incomplete Or Unpinned Indexes

**Files:**
- Modify: `phentrieve/indexing/chromadb_orchestrator.py`
- Modify: `phentrieve/indexing/chromadb_indexer.py`
- Create: `tests/unit/indexing/test_chromadb_indexer.py`
- Modify: `tests/integration/test_vector_store_stability.py`

- [ ] **Step 1: Write failing index-safety tests**

Use a fake embedding model and temporary Chroma directory. Assert a successful build writes the supplied HPO version, model, index type, dimension, and expected document count. Make one `encode` call raise and assert the build returns `False`. Simulate a persisted-count mismatch and assert `False`.

```python
assert collection.metadata["hpo_version"] == "v2026-06-23"
assert collection.metadata["expected_document_count"] == len(documents)
assert build_chromadb_index(..., hpo_version="v2026-06-23") is False
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -n 0 tests/unit/indexing/test_chromadb_indexer.py -v
```

Expected: failure because the indexer writes `latest` and continues after batch errors.

- [ ] **Step 3: Implement strict completion semantics**

Make the orchestrator read `hpo_version` from the selected SQLite database and pass it to `build_chromadb_index`. Replace the hard-coded metadata literal. Return `False` immediately on any encode/add error and, after all batches, require `collection.count() == len(documents)` before success.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
uv run pytest -n 0 tests/unit/indexing/test_chromadb_indexer.py -v
uv run pytest -n 0 tests/integration/test_vector_store_stability.py -k reproducible -v
git add phentrieve/indexing/chromadb_orchestrator.py   phentrieve/indexing/chromadb_indexer.py   tests/unit/indexing/test_chromadb_indexer.py   tests/integration/test_vector_store_stability.py
git commit -m "fix(index): reject incomplete pinned builds"
```

## Task 4: Enforce Bundle Provenance

**Files:**
- Modify: `phentrieve/data_processing/bundle_manifest.py`
- Modify: `phentrieve/data_processing/bundle_packager.py`
- Modify: `tests/unit/data_processing/test_bundle_manifest.py`
- Modify: `tests/unit/data_processing/test_bundle_packager.py`

- [ ] **Step 1: Write failing package tests**

Create temporary collections that have `hpo_version=latest`, a version different from the database, a mismatched model/index type, and a short document count. Assert `create_bundle` raises `ValueError` for each. Add a manifest round-trip test for release date, HPO source SHA, source commit, lockfile SHA, and model revision.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -n 0 tests/unit/data_processing/test_bundle_manifest.py   tests/unit/data_processing/test_bundle_packager.py -v
```

Expected: new tests fail because the packager only checks collection presence.

- [ ] **Step 3: Implement contract validation**

Extend the manifest with backward-compatible provenance fields. Resolve the named collection and reject metadata not matching the database HPO version, model, vector mode, and expected count. Set `hpo_release_date` from its correctly named database value, not the download timestamp.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2; then:

```bash
git add phentrieve/data_processing/bundle_manifest.py   phentrieve/data_processing/bundle_packager.py   tests/unit/data_processing/test_bundle_manifest.py   tests/unit/data_processing/test_bundle_packager.py
git commit -m "fix(bundle): enforce release provenance contract"
```

## Task 5: Verify A Complete Data Release

**Files:**
- Create: `scripts/verify_data_release.py`
- Create: `tests/unit/scripts/test_verify_data_release.py`
- Modify: `scripts/README.md`

- [ ] **Step 1: Write failing verifier tests**

Create miniature minimal/single/multivector archives. Assert the verifier rejects missing expected archives, bad `SHA256SUMS`, wrong term/document counts, and incorrect metadata; assert it accepts a complete contract without a network call.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -n 0 tests/unit/scripts/test_verify_data_release.py -v
```

Expected: failure because the verifier does not exist.

- [ ] **Step 3: Implement verifier**

Implement a CLI accepting `--spec`, `--bundle-dir`, and `--smoke-test`. It checks expected filenames, checksums, extracted bundle checksums, manifest provenance, Chroma collection count/metadata, and one retrieval query for each non-minimal archive. Document the command in `scripts/README.md`.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2; then:

```bash
git add scripts/verify_data_release.py tests/unit/scripts/test_verify_data_release.py scripts/README.md
git commit -m "feat(data): verify complete HPO bundle releases"
```

## Task 6: Separate Software And Data Distribution

**Files:**
- Modify: `phentrieve/config.py`
- Modify: `phentrieve/data_processing/bundle_downloader.py`
- Modify: `tests/unit/data_processing/test_bundle_downloader.py`
- Modify: `.github/workflows/build-data-bundles.yml`
- Modify: `.github/workflows/docker-publish.yml`
- Modify: `docker-compose.yml`
- Modify: `.env.docker.template`
- Modify: `docs/DOCKER-DEPLOYMENT.md`

- [ ] **Step 1: Write failing downloader tests**

Assert the downloader reads `PHENTRIEVE_DATA_RELEASE_REPOSITORY`, defaults to `berntpopp/phentrieve-data`, builds GitHub API URLs from that value, and permits `berntpopp/phentrieve` as an explicit legacy override.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -n 0 tests/unit/data_processing/test_bundle_downloader.py -v
```

Expected: new tests fail because the repository is hard-coded.

- [ ] **Step 3: Implement configuration and workflow split**

Replace the downloader constant with config-backed repository resolution. Convert the build workflow to `workflow_call` plus `workflow_dispatch`; remove `release: created` so a software release cannot generate data. Make inputs pin HPO/source/model values. Change Docker and Compose to use data-repository bundle URLs and document legacy assets.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
uv run pytest -n 0 tests/unit/data_processing/test_bundle_downloader.py -v
uv run python - <<'PY'
from pathlib import Path
import yaml
for path in Path(".github/workflows").glob("*.yml"):
    yaml.safe_load(path.read_text())
print("workflow yaml ok")
PY
git add phentrieve/config.py phentrieve/data_processing/bundle_downloader.py   tests/unit/data_processing/test_bundle_downloader.py   .github/workflows/build-data-bundles.yml .github/workflows/docker-publish.yml   docker-compose.yml .env.docker.template docs/DOCKER-DEPLOYMENT.md
git commit -m "feat(data): separate bundle distribution repository"
```

## Task 7: Create The Dedicated Data Repository

**Repository:** `berntpopp/phentrieve-data`

- [ ] **Step 1: Create the empty public repository**

Run:

```bash
gh repo create berntpopp/phentrieve-data --public --description   "Versioned, verified HPO vector-index bundles for Phentrieve" --clone
```

- [ ] **Step 2: Add only metadata and automation**

Add `README.md`, `LICENSE`, `ATTRIBUTION.md`, `.gitignore`, `releases/hpo-v2026-06-23-r1.json`, and `.github/workflows/release-data.yml`. The README distinguishes code/data release lifecycles and explains download/verification. The attribution file records HPO source/release/license link. Ignore `dist/`, `data/`, indexes, model caches, and `.runs/`.

- [ ] **Step 3: Add pinned caller workflow**

Make the data repository's manual workflow call:

```yaml
uses: berntpopp/phentrieve/.github/workflows/build-data-bundles.yml@<full-sha>
```

Use caller-owned `contents: write` to create only a draft release. Pin the reuse reference to a full commit SHA, not a moving branch/tag.

- [ ] **Step 4: Validate and commit**

Run:

```bash
git check-ignore -v dist data .runs
git ls-files | rg "(hpo_data\.db|chroma\.sqlite3|data_level0\.bin)" && exit 1 || true
git add README.md LICENSE ATTRIBUTION.md .gitignore releases .github/workflows
git commit -m "chore: initialize immutable HPO data releases"
git push -u origin main
```

## Task 8: Build And Verify The RTX Matrix

**Inputs:** `hpo-v2026-06-23-r1`, the committed release specification, RTX 5090, and the exact tagged Phentrieve source commit.

- [ ] **Step 1: Preflight immutable inputs**

Verify source commit, `uv.lock`, GPU/CUDA availability, and source bytes:

```bash
echo "3b646565695329aa399e937883c68d5d424d0df5eaab2f22baa0e08d44fdbe87  hp.json" | sha256sum --check
```

- [ ] **Step 2: Prepare and verify the database**

Prepare using the pre-verified file. Require 19,836 active terms, 577 obsolete terms filtered, HPO release date `2026-06-23`, and stored source digest. Create and validate the minimal archive before using the GPU.

- [ ] **Step 3: Build nine single-vector bundles serially**

Start each model with an empty index directory; run `phentrieve index build --recreate --batch-size 256`; package only after strict checks; verify immediately; clear the index. On CUDA OOM, remove the incomplete index and retry only that model at batch size 128, recording it in the manifest.

- [ ] **Step 4: Build nine multivector bundles serially**

Repeat with `--multi-vector`. Require 63,428 vectors, explicit pinned HPO version, correct model ID, and `multi_vector` metadata in every archive.

- [ ] **Step 5: Run the complete verifier**

Generate `SHA256SUMS` and run:

```bash
uv run python scripts/verify_data_release.py   --spec releases/hpo-v2026-06-23-r1.json   --bundle-dir dist/hpo-v2026-06-23-r1 --smoke-test
```

Expected: exactly 19 archives, valid checksums/manifests/counts, and one retrieval result per model/mode.

## Task 9: Publish And Switch Consumers

**Files:**
- Create: data release assets `release-manifest.json`, `verification-report.json`
- Modify: `CHANGELOG.md`
- Modify: data repository `README.md`

- [ ] **Step 1: Run final quality gates**

Run:

```bash
make check
make typecheck-fast
make test
make ci-local
```

Run the fixed BioLORD retrieval benchmark on both old and new data releases and save inputs, commands, metric delta, and verdict in `verification-report.json`. Investigate a material regression before publication.

- [ ] **Step 2: Create and inspect a draft release**

Create `hpo-v2026-06-23-r1` as a draft in the data repository. Upload 19 archives, `SHA256SUMS`, `release-manifest.json`, and `verification-report.json`. Confirm the expected 22 assets through the GitHub API and compare each remote digest with local SHA-256.

- [ ] **Step 3: Publish immutable release**

Enable immutable releases in the data repository. Publish the reviewed draft, then run `gh release verify` and `gh release verify-asset` for every archive. Update the data README with tag and release URL.

- [ ] **Step 4: Release software patch and external default**

Create the normal Phentrieve software patch release containing Tasks 1-6. Update Docker image builds only after the data release is remotely verifiable. Verify a built image extracts the external BioLORD multivector bundle and reports the pinned HPO version.

- [ ] **Step 5: Record release and commit**

```bash
git add CHANGELOG.md   .planning/specs/2026-07-15-hpo-data-release-separation-design.md   .planning/active/2026-07-15-hpo-data-release-separation-plan.md
git commit -m "docs: record HPO v2026-06-23 data release"
```

## Final Acceptance Criteria

- The release is built from an exact source commit, HPO asset digest, lockfile, and model revisions.
- The data repository publishes 22 immutable assets: 19 archives, `SHA256SUMS`, `release-manifest.json`, and `verification-report.json`.
- Each collection is complete, explicitly versioned, checksum-verified, and smoke queried after extraction.
- Phentrieve software releases no longer trigger data builds; downloader/Docker defaults target the dedicated data repository.
- Legacy `data-v2026-02-16` stays downloadable from the software repository.
