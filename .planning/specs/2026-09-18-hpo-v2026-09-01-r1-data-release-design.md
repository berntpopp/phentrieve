# HPO v2026-09-01-r1 Data Release Specification & Design

- Date: 2026-09-18
- Author: Antigravity (Bioinformatics & ML Engineering)
- Status: Proposed / Ready for execution
- Target Release: `hpo-v2026-09-01-r1` in `berntpopp/phentrieve-data`
- Compute Target: NVIDIA GeForce RTX 5090 (32GB VRAM), PyTorch CUDA 13, cuDNN 9

---

## 1. Executive Summary & Bioinformatics Rationale

On September 3, 2026, the Human Phenotype Ontology consortium released **`v2026-09-01`**. The current production Phentrieve vector database and Docker distributions are pinned to **`hpo-v2026-06-23-r2`**.

### Ontological Deltas

| Dimension | `v2026-06-23-r2` | `v2026-09-01-r1` | Delta | Clinical Impact |
|---|---|---|---|---|
| **Active HPO Terms** | 19,836 | **19,894** | **+58 net** | Ingests 69 newly minted terms (e.g., `HP:0020357` *Head and neck squamous cell carcinoma*, `HP:6001534` *Limited thumb abduction*, `HP:6001524` *Poor oral mucosal wound healing*) |
| **Multi-Vector Chunks** | 63,428 | **63,586** | **+158 documents** | Expanded lexical coverage across synonyms (26,260) and clinical definitions (17,432) |
| **Obsoleted/Filtered** | Retained | Filtered | Synchronized | Eliminates false-positive mappings to newly obsoleted term IDs |
| **Upstream Digest** | `3b6465...` | `a7b3a012e7b4007a35a7cf8da35f2373b54cc16907f80d13d62470d31f833501` | Bit-verified | Cryptographic traceability to upstream release |

Recomputing and publishing `hpo-v2026-09-01-r1` resolves the semantic gap for downstream clinical phenotyping, aligns pre-built bundles with fresh CLI preparations (`phentrieve data prepare`), and maintains the decoupled release contract introduced in PR #295.

---

## 2. Release Contract Specification

The release contract strictly adheres to `DataReleaseSpec` in `phentrieve.data_processing.release_contract`:

```json
{
  "active_terms": 19894,
  "hpo_release_date": "2026-09-01",
  "hpo_sha256": "a7b3a012e7b4007a35a7cf8da35f2373b54cc16907f80d13d62470d31f833501",
  "hpo_source_url": "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-09-01/hp.json",
  "hpo_version": "v2026-09-01",
  "lockfile_sha256": "8f4fed04319294a90bac14febbeda2039d36a41afa52a8fadfac679f83ee8149",
  "models": [
    {
      "name": "FremyCompany/BioLORD-2023-M",
      "revision": "4ea2ea2c89ef63365f7fcd91406a0cd8ac36d2e2",
      "slug": "biolord",
      "trust_remote_code": false
    },
    {
      "name": "BAAI/bge-m3",
      "revision": "5617a9f61b028005a4858fdac845db406aefb181",
      "slug": "bge-m3",
      "trust_remote_code": false
    },
    {
      "name": "sentence-transformers/LaBSE",
      "revision": "836121a0533e5664b21c7aacc5d22951f2b8b25b",
      "slug": "labse",
      "trust_remote_code": false
    },
    {
      "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
      "revision": "4328cf26390c98c5e3c738b4460a05b95f4911f5",
      "slug": "mpnet-multi",
      "trust_remote_code": false
    },
    {
      "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
      "slug": "minilm-multi",
      "trust_remote_code": false
    },
    {
      "code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4",
      "name": "Alibaba-NLP/gte-multilingual-base",
      "revision": "9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
      "slug": "gte-multi",
      "trust_remote_code": true
    },
    {
      "name": "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
      "revision": "73fdad86ea7ac68989712ce2007ab43ae89a2ad7",
      "slug": "tsystems-ende",
      "trust_remote_code": false
    },
    {
      "name": "sentence-transformers/distiluse-base-multilingual-cased-v2",
      "revision": "bfe45d0732ca50787611c0fe107ba278c7f3f889",
      "slug": "distiluse-multi",
      "trust_remote_code": false
    }
  ],
  "multivector_documents": 63586,
  "phentrieve_version": "0.28.2",
  "release_tag": "hpo-v2026-09-01-r1",
  "source_commit": "278a6665bd99fcc631d0da62d16ca1a23fbee836"
}
```

---

## 3. RTX 5090 Hardware Acceleration & Compute Architecture

The build host features an **NVIDIA GeForce RTX 5090 (Blackwell, 32GB GDDR7 VRAM, Compute Capability 12.0)**. We exploit this configuration for fast, bit-accurate computation:

1. **Batch Size Tuning (`--batch-size 256`):**
   - Single-vector (19,894 documents): Processed in ~78 batches per model (~10–15s).
   - Multi-vector (63,586 documents): Processed in ~248 batches per model (~30–45s).
   - Maximum VRAM allocated under batch size 256 is ~6.2 GB (BAAI/bge-m3), leaving >25 GB headroom and eliminating any risk of CUDA OOM.
2. **Deterministic Float32 Vector Representation:**
   - Embeddings are computed and stored in full IEEE 754 float32 precision.
   - For `gte-multi`, the fixed rotary buffers (repaired in PR #346) maintain numeric reproducibility across Transformers 5.16.1.
3. **Serial Model Hygiene:**
   - Between each model build, `clear_model_registry()` and `shutil.rmtree(indexes_dir)` clean the ChromaDB workspace and unload GPU tensors, preventing cross-model memory fragmentation.
4. **Estimated Matrix Runtime:**
   - Database preparation: ~15s
   - 8 models x (Single + Multi): ~12–15 minutes total GPU compute
   - Tarball compression (`tar -czf`) & SHA256 checksumming: ~2–3 minutes
   - Full matrix generation completes in under 20 minutes locally.

---

## 4. Artifact Deliverables (17 Tarballs + Metadata)

| # | Artifact Filename | Vector Mode | Dimensions | Estimated Size |
|---|---|---|---|---|
| 1 | `phentrieve-data-v2026-09-01-minimal.tar.gz` | None (SQLite only) | N/A | ~2.8 MB |
| 2 | `phentrieve-data-v2026-09-01-biolord.tar.gz` | Single-vector | 768 | ~79 MB |
| 3 | `phentrieve-data-v2026-09-01-biolord-multivec.tar.gz` | Multi-vector | 768 | ~203 MB |
| 4 | `phentrieve-data-v2026-09-01-bge-m3.tar.gz` | Single-vector | 1024 | ~98 MB |
| 5 | `phentrieve-data-v2026-09-01-bge-m3-multivec.tar.gz` | Multi-vector | 1024 | ~261 MB |
| 6 | `phentrieve-data-v2026-09-01-labse.tar.gz` | Single-vector | 768 | ~79 MB |
| 7 | `phentrieve-data-v2026-09-01-labse-multivec.tar.gz` | Multi-vector | 768 | ~203 MB |
| 8 | `phentrieve-data-v2026-09-01-mpnet-multi.tar.gz` | Single-vector | 768 | ~80 MB |
| 9 | `phentrieve-data-v2026-09-01-mpnet-multi-multivec.tar.gz` | Multi-vector | 768 | ~203 MB |
| 10 | `phentrieve-data-v2026-09-01-minilm-multi.tar.gz` | Single-vector | 384 | ~52 MB |
| 11 | `phentrieve-data-v2026-09-01-minilm-multi-multivec.tar.gz` | Multi-vector | 384 | ~116 MB |
| 12 | `phentrieve-data-v2026-09-01-gte-multi.tar.gz` | Single-vector | 768 | ~80 MB |
| 13 | `phentrieve-data-v2026-09-01-gte-multi-multivec.tar.gz` | Multi-vector | 768 | ~203 MB |
| 14 | `phentrieve-data-v2026-09-01-tsystems-ende.tar.gz` | Single-vector | 768 | ~80 MB |
| 15 | `phentrieve-data-v2026-09-01-tsystems-ende-multivec.tar.gz` | Multi-vector | 768 | ~204 MB |
| 16 | `phentrieve-data-v2026-09-01-distiluse-multi.tar.gz` | Single-vector | 512 | ~61 MB |
| 17 | `phentrieve-data-v2026-09-01-distiluse-multi-multivec.tar.gz` | Multi-vector | 512 | ~145 MB |
| 18 | `SHA256SUMS` | Verification digests for all 17 archives |
| 19 | `release-manifest.json` | Full contract, document counts, and archive metadata |
| 20 | `verification-report.json` | Automated smoke-test and collection integrity verification |

---

## 5. Downstream Integration & Policy Alignment

Upon publication of `hpo-v2026-09-01-r1`:
1. In `berntpopp/phentrieve`:
   - Update default bundle URL in `api/Dockerfile`:
     `ARG BUNDLE_URL="https://github.com/berntpopp/phentrieve-data/releases/download/hpo-v2026-09-01-r1/phentrieve-data-v2026-09-01-biolord-multivec.tar.gz"`
   - Update `docker-compose.yml` and `.env.docker.template`.
   - Update `tests/unit/test_docker_data_release_policy.py` expectation to `"hpo-v2026-09-01-r1/"`.
   - Run benchmark comparison on 570 German terms and GeneReviews evaluation sets to quantify retrieval precision changes between `v2026-06-23` and `v2026-09-01`.
