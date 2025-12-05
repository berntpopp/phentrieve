# Pre-built Data Distribution System

**Status**: Proposed
**Created**: 2025-12-05
**Issue**: TBD (to be created after review)

---

## Executive Summary

Replace the current "build from scratch" approach for HPO databases and vector indexes with a **pre-built data distribution system** using GitHub Releases. This will:

- Reduce first-run setup time from **5-15 minutes** to **30 seconds**
- Eliminate redundant index building across installations
- Enable reproducible benchmarks with identical data
- Simplify Docker image builds
- Support multiple HPO versions and embedding models

---

## 1. Current State Analysis

### Data Artifacts Currently Built

| Artifact | Size | Build Time | Purpose |
|----------|------|------------|---------|
| `hpo_data.db` | 12 MB | ~30s | SQLite: HPO terms, graph metadata |
| `hp.json` | 21 MB | Download | Source HPO ontology |
| `indexes/chroma.sqlite3` | 45 MB | ~2-5 min | ChromaDB vector index |
| `indexes/{uuid}/` | 60 MB | - | ChromaDB data segments |
| **Total per model** | **~140 MB** | **5-15 min** | - |

### Current Build Flow

```
User runs `phentrieve data prepare`
    ↓
Download hp.json from GitHub (HPO releases)
    ↓
Parse JSON, build SQLite database
    ↓
User runs `phentrieve index build`
    ↓
Load embedding model (~500MB download first time)
    ↓
Embed 19,534 HPO terms
    ↓
Store in ChromaDB
```

**Problems:**
1. Every installation repeats the same work
2. Index build requires GPU/CPU compute
3. Docker builds are slow (rebuild on every deploy)
4. No data versioning or reproducibility
5. Different machines may produce slightly different indexes

---

## 2. Proposed Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  phentrieve-data Repository                  │
│                  (github.com/berntpopp/phentrieve-data)      │
├─────────────────────────────────────────────────────────────┤
│  Releases:                                                   │
│  ├── v2025.03.03-biolord-2023-m (HPO + BioLORD index)       │
│  ├── v2025.03.03-jina-de-v2 (HPO + Jina German index)       │
│  ├── v2025.03.03-bge-m3 (HPO + BGE-M3 index)                │
│  ├── v2025.03.03-hpo-only (HPO database only, no index)     │
│  ├── latest → v2025.03.03-biolord-2023-m                    │
│  ├── recommended → v2025.03.03-biolord-2023-m               │
│  └── benchmark → v2025.03.03-biolord-2023-m                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    phentrieve CLI                            │
│  `phentrieve data download --preset recommended`             │
│  `phentrieve data download --hpo-version v2025.03.03        │
│                            --model biolord-2023-m`           │
└─────────────────────────────────────────────────────────────┘
```

### Release Bundle Structure

Each release contains a `.tar.gz` archive:

```
phentrieve-data-v2025.03.03-biolord-2023-m.tar.gz
├── manifest.json           # Metadata about the bundle
├── hpo_data.db             # SQLite database
├── indexes/
│   ├── chroma.sqlite3      # ChromaDB metadata
│   └── {uuid}/             # ChromaDB segments
│       ├── data_level0.bin
│       ├── header.bin
│       ├── index_metadata.pickle
│       └── length.bin
└── checksums.sha256        # Integrity verification
```

### manifest.json Schema

```json
{
  "schema_version": "1.0.0",
  "bundle_id": "v2025.03.03-biolord-2023-m",
  "created_at": "2025-12-05T10:30:00Z",
  "created_by": "phentrieve-ci",

  "hpo": {
    "version": "v2025-03-03",
    "release_date": "2025-03-03",
    "term_count": 19534,
    "source_url": "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2025-03-03/hp.json"
  },

  "index": {
    "model_name": "FremyCompany/BioLORD-2023-M",
    "model_slug": "biolord-2023-m",
    "embedding_dimension": 768,
    "distance_metric": "cosine",
    "collection_name": "hpo_FremyCompany_BioLORD-2023-M",
    "document_count": 19534
  },

  "compatibility": {
    "phentrieve_min_version": "0.5.0",
    "chromadb_version": "0.5.23",
    "python_min_version": "3.9"
  },

  "checksums": {
    "hpo_data.db": "sha256:abc123...",
    "indexes/chroma.sqlite3": "sha256:def456..."
  }
}
```

---

## 3. Tag Strategy

### Semantic Tags

| Tag Pattern | Example | Purpose |
|-------------|---------|---------|
| `v{HPO_VERSION}-{MODEL_SLUG}` | `v2025.03.03-biolord-2023-m` | Specific version |
| `hpo-{VERSION}` | `hpo-v2025.03.03` | HPO-only (no index) |
| `latest` | → current default | Most recent stable |
| `recommended` | → tested for production | Quality-assured |
| `benchmark` | → used for benchmarks | Reproducibility |

### Model Slug Convention

```python
MODEL_SLUGS = {
    "FremyCompany/BioLORD-2023-M": "biolord-2023-m",
    "jinaai/jina-embeddings-v2-base-de": "jina-de-v2",
    "BAAI/bge-m3": "bge-m3",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet-multilingual",
}
```

---

## 4. CLI Commands

### New Commands

```bash
# List available pre-built bundles
phentrieve data list-bundles
# Output:
# Available bundles:
#   v2025.03.03-biolord-2023-m (140 MB) [recommended] [latest]
#   v2025.03.03-jina-de-v2 (142 MB)
#   v2025.03.03-bge-m3 (180 MB)
#   hpo-v2025.03.03 (12 MB) [hpo-only]

# Download recommended bundle
phentrieve data download --preset recommended

# Download specific version
phentrieve data download --bundle v2025.03.03-biolord-2023-m

# Download HPO-only (build index locally later)
phentrieve data download --preset hpo-only

# Download with custom data directory
phentrieve data download --preset recommended --data-dir /custom/path

# Verify bundle integrity
phentrieve data verify

# Show current data status
phentrieve data status
# Output:
# Data Directory: /home/user/.phentrieve/data
# HPO Database: v2025-03-03 (19,534 terms)
# Index: biolord-2023-m (19,534 embeddings)
# Bundle: v2025.03.03-biolord-2023-m
# Verified: ✓ checksums match
```

### Backward Compatibility

Existing commands continue to work:

```bash
# Still works - builds locally
phentrieve data prepare
phentrieve index build

# New shortcut - downloads pre-built
phentrieve data download --preset recommended
```

---

## 5. Implementation Plan

### Phase 1: Data Repository Setup (Week 1)

**Tasks:**
1. Create `berntpopp/phentrieve-data` repository
2. Set up GitHub Actions workflow for building bundles
3. Create bundle build script (`scripts/build_data_bundle.py`)
4. Generate first bundle with BioLORD model
5. Document bundle format in repository README

**Files to Create:**

```
phentrieve-data/
├── .github/
│   └── workflows/
│       ├── build-bundle.yml      # Manual trigger for new bundles
│       └── scheduled-build.yml   # Monthly HPO update check
├── scripts/
│   ├── build_bundle.py           # Bundle creation script
│   ├── verify_bundle.py          # Integrity verification
│   └── upload_release.py         # Upload to GitHub Releases
├── schemas/
│   └── manifest.schema.json      # JSON schema for manifest
├── README.md
└── BUNDLE_FORMAT.md
```

### Phase 2: CLI Integration (Week 2)

**Tasks:**
1. Add `phentrieve/data_distribution/` module
2. Implement download command with progress bar
3. Add bundle verification
4. Update `phentrieve data status` command
5. Add configuration for custom repository URL

**New Files in phentrieve:**

```
phentrieve/data_distribution/
├── __init__.py
├── bundle_downloader.py    # Download and extract bundles
├── bundle_verifier.py      # SHA256 verification
├── release_client.py       # GitHub Releases API client
└── manifest.py             # Manifest parsing/validation
```

**CLI Changes:**

```python
# phentrieve/cli/data_commands.py

@app.command("download")
def download_bundle(
    preset: Annotated[
        Optional[str],
        typer.Option("--preset", help="Preset bundle (latest, recommended, benchmark, hpo-only)")
    ] = None,
    bundle: Annotated[
        Optional[str],
        typer.Option("--bundle", help="Specific bundle version (e.g., v2025.03.03-biolord-2023-m)")
    ] = None,
    data_dir: Annotated[
        Optional[str],
        typer.Option("--data-dir", help="Custom data directory")
    ] = None,
    verify: Annotated[
        bool,
        typer.Option("--verify/--no-verify", help="Verify checksums after download")
    ] = True,
):
    """Download pre-built HPO data and indexes."""
    ...
```

### Phase 3: Docker Integration (Week 3)

**Tasks:**
1. Update Dockerfile to use pre-built bundles
2. Create multi-stage build with data layer
3. Add bundle selection via build args
4. Document Docker data management

**Dockerfile Changes:**

```dockerfile
# Build stage for downloading data
FROM python:3.11-slim AS data-downloader
ARG DATA_BUNDLE=recommended
RUN pip install phentrieve
RUN phentrieve data download --preset ${DATA_BUNDLE}

# Final stage
FROM python:3.11-slim AS runtime
COPY --from=data-downloader /root/.phentrieve/data /app/data
...
```

### Phase 4: Automation & CI (Week 4)

**Tasks:**
1. Monthly HPO version check workflow
2. Automated bundle building on HPO release
3. Bundle compatibility matrix
4. Integration tests with pre-built data

**GitHub Actions:**

```yaml
# .github/workflows/check-hpo-update.yml
name: Check HPO Update

on:
  schedule:
    - cron: '0 0 1 * *'  # First day of each month

jobs:
  check-update:
    runs-on: ubuntu-latest
    steps:
      - name: Check for new HPO release
        run: |
          LATEST=$(curl -s https://api.github.com/repos/obophenotype/human-phenotype-ontology/releases/latest | jq -r '.tag_name')
          echo "Latest HPO: $LATEST"
          # Compare with current and trigger build if different
```

---

## 6. Bundle Build Process

### Automated Build Script

```python
#!/usr/bin/env python3
"""Build a phentrieve data bundle for distribution."""

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

def build_bundle(
    hpo_version: str,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Build a complete data bundle."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Download HPO and build database
        from phentrieve.data_processing.hpo_parser import orchestrate_hpo_preparation
        orchestrate_hpo_preparation(data_dir_override=str(tmpdir))

        # 2. Build vector index
        from phentrieve.indexing.chromadb_orchestrator import build_index_for_model
        build_index_for_model(
            model_name=model_name,
            data_dir=tmpdir,
        )

        # 3. Create manifest
        manifest = create_manifest(
            hpo_version=hpo_version,
            model_name=model_name,
            data_dir=tmpdir,
        )

        # 4. Calculate checksums
        checksums = calculate_checksums(tmpdir)
        manifest["checksums"] = checksums

        # 5. Write manifest
        manifest_path = tmpdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # 6. Create tarball
        model_slug = get_model_slug(model_name)
        bundle_name = f"phentrieve-data-{hpo_version}-{model_slug}.tar.gz"
        bundle_path = output_dir / bundle_name

        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(tmpdir / "hpo_data.db", arcname="hpo_data.db")
            tar.add(tmpdir / "indexes", arcname="indexes")
            tar.add(manifest_path, arcname="manifest.json")
            tar.add(tmpdir / "checksums.sha256", arcname="checksums.sha256")

        return bundle_path
```

---

## 7. Download Implementation

### Release Client

```python
"""GitHub Releases API client for phentrieve-data repository."""

import httpx
from dataclasses import dataclass
from typing import Optional

REPO_URL = "https://api.github.com/repos/berntpopp/phentrieve-data"

@dataclass
class ReleaseAsset:
    name: str
    download_url: str
    size: int
    content_type: str

@dataclass
class Release:
    tag_name: str
    name: str
    published_at: str
    assets: list[ReleaseAsset]
    prerelease: bool

class ReleaseClient:
    def __init__(self, repo_url: str = REPO_URL):
        self.repo_url = repo_url
        self.client = httpx.Client(timeout=30.0)

    def list_releases(self) -> list[Release]:
        """List all available releases."""
        response = self.client.get(f"{self.repo_url}/releases")
        response.raise_for_status()
        return [self._parse_release(r) for r in response.json()]

    def get_release_by_tag(self, tag: str) -> Optional[Release]:
        """Get a specific release by tag name."""
        try:
            response = self.client.get(f"{self.repo_url}/releases/tags/{tag}")
            response.raise_for_status()
            return self._parse_release(response.json())
        except httpx.HTTPStatusError:
            return None

    def get_latest_release(self) -> Optional[Release]:
        """Get the latest release."""
        response = self.client.get(f"{self.repo_url}/releases/latest")
        response.raise_for_status()
        return self._parse_release(response.json())

    def download_asset(
        self,
        asset: ReleaseAsset,
        output_path: Path,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Download a release asset with progress reporting."""
        with self.client.stream("GET", asset.download_url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)

        return output_path
```

---

## 8. Configuration

### Environment Variables

```bash
# Custom data repository (for enterprise/private deployments)
PHENTRIEVE_DATA_REPO_URL=https://api.github.com/repos/myorg/phentrieve-data

# Skip verification (not recommended)
PHENTRIEVE_SKIP_VERIFY=false

# Offline mode (use local data only)
PHENTRIEVE_OFFLINE=false
```

### Config File (phentrieve.yaml)

```yaml
data_distribution:
  repository_url: "https://api.github.com/repos/berntpopp/phentrieve-data"
  default_preset: "recommended"
  auto_download: false  # Prompt user or auto-download on first run
  verify_checksums: true
  cache_releases: true  # Cache release metadata locally
```

---

## 9. Testing Strategy

### Unit Tests

```python
# tests/unit/data_distribution/test_bundle_downloader.py

def test_download_bundle_with_verification():
    """Test downloading and verifying a bundle."""
    ...

def test_manifest_validation():
    """Test manifest schema validation."""
    ...

def test_checksum_verification():
    """Test SHA256 checksum verification."""
    ...
```

### Integration Tests

```python
# tests/integration/test_prebuilt_data.py

def test_download_and_query_with_prebuilt_data():
    """Test full workflow with pre-built data."""
    # Download bundle
    download_bundle(preset="recommended", data_dir=tmpdir)

    # Query with pre-built index
    results = query_hpo("seizures", data_dir=tmpdir)

    assert len(results) > 0
    assert results[0]["hpo_id"].startswith("HP:")
```

### E2E Tests

```python
# tests/e2e/test_docker_with_prebuilt_data.py

def test_docker_container_with_prebuilt_data():
    """Test Docker container uses pre-built data correctly."""
    ...
```

---

## 10. Migration Path

### For Existing Users

1. **No breaking changes** - existing `phentrieve data prepare` still works
2. **Optional adoption** - users can choose to use pre-built data
3. **Gradual migration** - recommend pre-built data in documentation

### Documentation Updates

- Update README with pre-built data instructions
- Add "Quick Start" section using pre-built data
- Keep "Build from Source" section for advanced users
- Document bundle versioning and compatibility

---

## 11. Benefits Summary

| Benefit | Before | After |
|---------|--------|-------|
| **Setup Time** | 5-15 minutes | 30 seconds |
| **Reproducibility** | Varies by machine | Identical data |
| **Docker Build** | Slow (rebuild index) | Fast (copy pre-built) |
| **CI/CD** | Flaky (network/compute) | Reliable (cached) |
| **Disk Usage** | Same | Same |
| **Flexibility** | Build custom | Download OR build custom |

---

## 12. Security Considerations

1. **Checksum Verification**: SHA256 for all files
2. **Signed Releases**: Consider GPG signing releases
3. **HTTPS Only**: All downloads over HTTPS
4. **No Secrets in Bundles**: Bundles contain only public HPO data
5. **Audit Trail**: GitHub Actions logs for all builds

---

## 13. Future Enhancements

1. **Delta Updates**: Download only changed files between versions
2. **P2P Distribution**: IPFS for distributed downloads
3. **Hugging Face Hub**: Mirror bundles on HF Hub
4. **Model-Specific Indexes**: Pre-built indexes for all benchmark models
5. **Regional Mirrors**: CDN or regional mirrors for faster downloads
6. **Incremental Builds**: Only rebuild changed HPO terms

---

## 14. References

- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [GitHub REST API for Releases](https://docs.github.com/en/rest/releases)
- [HPO Downloads](http://human-phenotype-ontology.github.io/downloads.html)
- [Git LFS vs DVC Comparison](https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Python Semantic Release](https://github.com/python-semantic-release/python-semantic-release)

---

## 15. Decision Log

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| Storage Backend | GitHub Releases, Git LFS, DVC, HF Hub | GitHub Releases | Free, no dependencies, 2GB limit sufficient |
| Data Repository | Same repo, Separate repo | Separate repo | Clean separation, independent versioning |
| Bundle Format | ZIP, TAR.GZ, RAW | TAR.GZ | Standard, good compression, preserves permissions |
| Versioning | SemVer, HPO-based | HPO + Model combo | Clear provenance |

---

**Next Steps:**
1. Review and approve this plan
2. Create GitHub Issue for tracking
3. Create `phentrieve-data` repository
4. Implement Phase 1

