# HPO v2026-09-01-r1 Data Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute, verify, and publish the verified `hpo-v2026-09-01-r1` data release across all 8 retrieval models (17 bundles) using the local NVIDIA GeForce RTX 5090, then integrate the updated bundle into Phentrieve.

**Architecture:** Pinned immutable HPO bytes (`v2026-09-01`), exact source commit (`278a666`, v0.28.2), and lockfile SHA-256. Generated and packaged locally on CUDA, verified with the smoke-test retrieval harness, and published to `berntpopp/phentrieve-data`.

**Tech Stack:** Python 3.13, PyTorch CUDA 13, SentenceTransformers 6.0.1, ChromaDB, NVIDIA RTX 5090 (32GB VRAM), GitHub CLI (`gh`).

---

## Task 1: Commit Data Release Specification in `phentrieve-data`

**Files:**
- Create: `/home/bernt-popp/development/phentrieve-data/releases/hpo-v2026-09-01-r1.json`
- Modify: `/home/bernt-popp/development/phentrieve-data/.github/workflows/release-data.yml`

- [x] **Step 1: Write `releases/hpo-v2026-09-01-r1.json`**
  Write the JSON release specification with:
  - `active_terms`: 19894
  - `multivector_documents`: 63586
  - `hpo_release_date`: "2026-09-01"
  - `hpo_sha256`: "a7b3a012e7b4007a35a7cf8da35f2373b54cc16907f80d13d62470d31f833501"
  - `hpo_source_url`: "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-09-01/hp.json"
  - `hpo_version`: "v2026-09-01"
  - `release_tag`: "hpo-v2026-09-01-r1"
  - `phentrieve_version`: "0.28.2"
  - `source_commit`: "278a6665bd99fcc631d0da62d16ca1a23fbee836"
  - `lockfile_sha256`: "8f4fed04319294a90bac14febbeda2039d36a41afa52a8fadfac679f83ee8149"
  - 8 models with pinned revisions.

- [x] **Step 2: Update workflow dispatch options in `phentrieve-data`**
  In `/home/bernt-popp/development/phentrieve-data/.github/workflows/release-data.yml`, add `hpo-v2026-09-01-r1` as an option and update the source commit pin.

- [x] **Step 3: Commit and push branch in `phentrieve-data`**
  Create branch `release/hpo-v2026-09-01-r1` in `phentrieve-data`, commit changes, push, and merge or open PR.

---

## Task 2: Build Matrix Locally on RTX 5090

**Files:**
- Output directory: `dist/hpo-v2026-09-01-r1`
- Temporary data: `.runs/hpo-v2026-09-01-r1/data`

- [x] **Step 1: Execute `build_data_release.py` on CUDA**
  Run:
  ```bash
  uv run python scripts/build_data_release.py \
    --spec /home/bernt-popp/development/phentrieve-data/releases/hpo-v2026-09-01-r1.json \
    --data-dir .runs/hpo-v2026-09-01-r1/data \
    --output-dir dist/hpo-v2026-09-01-r1 \
    --batch-size 256 \
    --device cuda
  ```
  Verify all 17 archives, `SHA256SUMS`, `release-manifest.json`, and `verification-report.json` are created.

- [x] **Step 2: Run verification and smoke-tests**
  Run:
  ```bash
  uv run python scripts/verify_data_release.py \
    --spec /home/bernt-popp/development/phentrieve-data/releases/hpo-v2026-09-01-r1.json \
    --bundle-dir dist/hpo-v2026-09-01-r1 \
    --smoke-test
  ```
  Ensure all 16 vector collections answer retrieval queries with 0 errors.

---

## Task 3: Publish Data Release in `berntpopp/phentrieve-data`

- [x] **Step 1: Create draft GitHub release**
  ```bash
  gh release create hpo-v2026-09-01-r1 \
    --repo berntpopp/phentrieve-data \
    --draft \
    --title "HPO v2026-09-01 data release (r1)" \
    --notes "Verified HPO vector-index data release for HPO v2026-09-01. Built with Phentrieve 0.28.2 on NVIDIA RTX 5090 (CUDA 13, batch size 256)." \
    dist/hpo-v2026-09-01-r1/*.tar.gz \
    dist/hpo-v2026-09-01-r1/SHA256SUMS \
    dist/hpo-v2026-09-01-r1/release-manifest.json \
    dist/hpo-v2026-09-01-r1/verification-report.json
  ```

- [x] **Step 2: Verify uploaded asset checksums via GitHub API**
  Compare API remote digests against local `SHA256SUMS` to verify transfer integrity across all 20 files.

- [x] **Step 3: Publish the release (undraft)**
  Mark the release as published and immutable.

---

## Task 4: Integrate New Release into `berntpopp/phentrieve`

**Files:**
- Modify: `api/Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `.env.docker.template`
- Modify: `docs/DOCKER-DEPLOYMENT.md`
- Modify: `scripts/README.md`
- Modify: `tests/unit/test_docker_data_release_policy.py`

- [x] **Step 1: Update bundle URLs to `hpo-v2026-09-01-r1`**
  Update default BioLORD multivector bundle URL:
  `https://github.com/berntpopp/phentrieve-data/releases/download/hpo-v2026-09-01-r1/phentrieve-data-v2026-09-01-biolord-multivec.tar.gz`

- [x] **Step 2: Update policy tests**
  Update `DEFAULT_BUNDLE_URL` in `tests/unit/test_docker_data_release_policy.py`.

- [x] **Step 3: Verify all test gates**
  Run `make check`, `make typecheck-fast`, and `uv run pytest tests/unit/test_docker_data_release_policy.py`.

- [x] **Step 4: Commit and push integration PR**
  Branch: `chore/update-hpo-data-release-v2026-09-01`
  Open PR on `berntpopp/phentrieve`.
