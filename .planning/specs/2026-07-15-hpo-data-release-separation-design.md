# HPO Data Release Separation Design

- Date: 2026-07-15
- Status: approved for implementation
- Scope: HPO `v2026-06-23`, all supported retrieval-model indexes, and separation of Phentrieve software releases from published data releases.

## Goal

Publish a verified, immutable HPO `v2026-06-23` data release containing the minimal database plus single-vector and multivector indexes for every supported RAG retrieval model. Future software releases and data releases must be independently versioned, discoverable, and reproducible.

## Evidence And Decision

The current upstream HPO release is `v2026-06-23`, published 2026-06-23. Its `hp.json` SHA-256 is `3b646565695329aa399e937883c68d5d424d0df5eaab2f22baa0e08d44fdbe87`. The current Phentrieve data release, `data-v2026-02-16`, indexes 19,389 active terms. A clean preparation of `v2026-06-23` produced 19,836 active terms (+447 net), 2,020 additional synonyms, 266 label changes, 309 definition changes, and 857 changed ancestor sets among retained terms. A new index release is therefore required.

The current bundle indexer records `hpo_version: latest` even when input data are pinned, and it can report success after an embedding batch fails. Both are release blockers because they make provenance false or can publish an incomplete index. They must be fixed before any bundle is created.

## Repository Boundary

`berntpopp/phentrieve` remains the software repository:

- Python CLI/library, API, frontend, docs, and software tests.
- Semantic software releases (`vX.Y.Z`) and GHCR API/frontend images.
- The reusable, versioned data-build workflow and release-verification code.

`berntpopp/phentrieve-data` becomes the data-distribution repository:

- No generated HPO database, vector index, or model cache is committed to Git.
- Each data release has a committed release specification and a matching GitHub draft release with assets.
- The release tag format is `hpo-vYYYY-MM-DD-rN`; `rN` permits a corrected or newly reproducible build of the same upstream HPO release without replacing historical bytes.
- GitHub immutable releases are enabled. Assets are uploaded to a draft, fully validated, and then published. The release contains `SHA256SUMS`, the data release manifest, and all bundle archives.

Existing assets under `berntpopp/phentrieve` remain available at their current URLs. They are documented as legacy rather than moved or deleted, preserving current Docker and CLI installations during migration.

## Release Contract

Every data release has a machine-readable specification containing:

- Data release tag and creation time.
- Upstream HPO version, immutable download URL, release date, and `hp.json` SHA-256.
- Phentrieve software version, full source commit SHA, and `uv.lock` SHA-256.
- ChromaDB/index format version and the expected document counts.
- The exact model matrix and the immutable Hugging Face revision for each model.
- Build environment information needed to interpret the result (Python, torch, CUDA, GPU), plus per-asset SHA-256.

Each individual bundle manifest retains its existing compatibility fields and is extended with the same pinned provenance. It must never contain the ambiguous value `latest` for HPO version, source revision, or model revision.

## Model Matrix And Expected Output

The data-release matrix is intentionally independent of `BENCHMARK_MODELS`, which currently resolves to BioLORD alone and is not the release matrix. The single source of truth defines these nine models:

1. `FremyCompany/BioLORD-2023-M`
2. `BAAI/bge-m3`
3. `sentence-transformers/LaBSE`
4. `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
5. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
6. `Alibaba-NLP/gte-multilingual-base`
7. `jinaai/jina-embeddings-v2-base-de`
8. `T-Systems-onsite/cross-en-de-roberta-sentence-transformer`
9. `sentence-transformers/distiluse-base-multilingual-cased-v2`

The initial release creates 19 artifacts:

- `phentrieve-data-v2026-06-23-minimal.tar.gz`.
- Nine `phentrieve-data-v2026-06-23-<model>.tar.gz` single-vector bundles.
- Nine `phentrieve-data-v2026-06-23-<model>-multivec.tar.gz` multivector bundles.

The prepared ontology must contain 19,836 active terms. Every single-vector collection must contain 19,836 documents. Every multivector collection must contain 63,428 documents (19,836 labels, 26,229 synonyms, and 17,363 definitions). The verifier rejects any other count.

## Build And Validation Flow

1. Create a clean, isolated release directory and fetch the pinned Phentrieve source commit.
2. Download `hp.json` explicitly, verify the upstream SHA-256, then prepare the SQLite database without re-downloading it.
3. Create and validate the minimal bundle.
4. Process one model and one index type at a time on the RTX 5090. Clear the index directory between bundles so an archive contains precisely one Chroma collection.
5. The indexer aborts at the first failed batch and verifies persisted collection count, HPO version, model, index type, and dimension before returning success.
6. Extract every bundle into an empty directory, verify checksums, assert the release contract, and run a retrieval smoke query for each model/index type.
7. Run the full Phentrieve quality gates plus focused data-release tests and a retrieval benchmark comparison against the previous BioLORD release.
8. Create a draft release in `berntpopp/phentrieve-data`, upload all 19 archives, checksums, and the release manifest, verify uploaded assets, then publish.
9. Switch Phentrieve downloader and Docker defaults to the data repository, release the software patch, and rebuild the container images using the verified multivector BioLORD bundle.

## GitHub Automation

The reusable workflow lives in the software repository and is called by a small workflow in `phentrieve-data`, pinned to a Phentrieve commit SHA. The caller owns the `github` context and `GITHUB_TOKEN`, so it can create the data repository's release without cross-repository write tokens. Local RTX builds use the same release specification and verification command, then upload with `gh release` to the dedicated repository.

The software-release workflow no longer listens for `release: created` to build HPO data. It only builds software artifacts/images, using a pinned data-release URL when an image needs a bundled default.

## Compatibility And Migration

- Add a configurable data-release repository setting to the downloader; default to `berntpopp/phentrieve-data` after its first release exists.
- Keep the old repository value available as an explicit compatibility override.
- Update Docker workflow/build arguments, Compose defaults, environment template, documentation, and release notes to use the new URL.
- The CLI detects release bundles through the configured repository API, not by hard-coded software repository URLs.
- Retain `data-v2026-02-16` unchanged in the software repository and explain its legacy status in the migration documentation.

## Security And Integrity

GitHub release assets are appropriate for the existing archive sizes, all of which are below GitHub's 2 GiB per-asset limit. A draft-first immutable release prevents later replacement of the release tag and bytes. The release verification record uses both `SHA256SUMS` and GitHub's asset digest/attestation verification.

Primary references:

- https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases
- https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity
- https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows
