# Pre-built Data Distribution System

> **Issues**: [#133](https://github.com/berntpopp/phentrieve/issues/133) (Remove obsolete HPO terms), [#117](https://github.com/berntpopp/phentrieve/issues/117) (Pre-built data distribution)
> **Branch**: `feat/prebuilt-data-distribution-133-117`
> **Status**: Active Planning
> **Created**: 2025-12-09

---

## Executive Summary

This plan addresses two interconnected issues:

1. **Issue #133**: ~13.5% of HPO terms fail self-matching due to obsolete terms in the index
2. **Issue #117**: First-run setup takes 5-15 minutes, impacting UX and Docker builds

**Solution**: Create a robust pre-built data distribution system that:
- Filters obsolete HPO terms during data preparation
- Pre-computes embeddings for multiple models (9 benchmark models)
- Distributes data bundles via GitHub Releases
- Integrates with Docker for reproducible, fast builds
- Supports both "latest" HPO and pinned versions

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Architecture Design](#2-architecture-design)
3. [Implementation Phases](#3-implementation-phases)
4. [Technical Specifications](#4-technical-specifications)
5. [Bundle Format & Versioning](#5-bundle-format--versioning)
6. [CLI Commands](#6-cli-commands)
7. [Docker Integration](#7-docker-integration)
8. [GitHub Actions Workflows](#8-github-actions-workflows)
9. [Success Criteria](#9-success-criteria)
10. [Risk Mitigation](#10-risk-mitigation)

---

## 1. Problem Analysis

### 1.1 Obsolete Terms Issue (#133)

**Root Cause**: HPO JSON includes obsolete terms that should not be indexed.

**OBO Graphs JSON Format** ([obographs spec](https://github.com/geneontology/obographs)):
```json
{
  "id": "http://purl.obolibrary.org/obo/HP_0008036",
  "lbl": "obsolete Rod-cone dystrophy",
  "meta": {
    "deprecated": true,
    "basicPropertyValues": [
      {
        "pred": "http://purl.obolibrary.org/obo/IAO_0100001",
        "val": "http://purl.obolibrary.org/obo/HP_0000510"
      }
    ]
  }
}
```

**Detection Strategies** (per [HPO Obsoletion Wiki](https://github.com/obophenotype/human-phenotype-ontology/wiki/Obsoletion)):
1. `meta.deprecated == true` (OWL:deprecated)
2. Label prefix: `"obsolete "` (case-insensitive)
3. `term_replaced_by` annotation for migration tracking

**Impact**:
- ~2,600 obsolete terms (~13.5% of 19,534 total)
- Self-matching failures during embedding validation
- Retrieval accuracy degradation

### 1.2 Slow First-Run Issue (#117)

**Current Flow**:
```
User Install → phentrieve data prepare → Download HPO JSON (1-2 min)
            → Parse & Build SQLite (1-2 min)
            → phentrieve index build → Load model (2-5 min)
            → Generate embeddings (3-8 min per model)
```

**Total**: 5-15 minutes for single model, 30-60 minutes for all 9 models

**Proposed Flow**:
```
User Install → phentrieve data download --preset recommended
            → Download pre-built bundle (140MB, 1-2 min)
            → Verify checksums
            → Ready!
```

**Total**: < 2 minutes

---

## 2. Architecture Design

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION PIPELINE                         │
│                                                                       │
│  HPO Release ──► Filter Obsolete ──► SQLite DB ──► Embeddings        │
│  (hp.json)       (deprecated)        (hpo_data.db) (ChromaDB)        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BUNDLE PACKAGING                                  │
│                                                                       │
│  manifest.json ◄── SQLite + ChromaDB indexes + checksums            │
│                    tar.gz (~140MB per model)                         │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GITHUB RELEASES                                   │
│                                                                       │
│  phentrieve-data-{hpo_version}-{model_slug}.tar.gz                  │
│  Tags: v{hpo_version}-data, latest, recommended                     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTION CHANNELS                             │
│                                                                       │
│  CLI: phentrieve data download --model biolord                       │
│  Docker: Multi-stage build with pre-built data layer                │
│  Manual: GitHub Releases page direct download                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Principles

| Principle | Application |
|-----------|-------------|
| **DRY** | Single source of truth for obsolete detection logic |
| **KISS** | Simple tar.gz bundles, no custom formats |
| **SOLID** | Interface segregation for data sources (local vs remote) |
| **Modularity** | Separate concerns: parsing, filtering, packaging, distribution |
| **No Anti-patterns** | Avoid God classes, tight coupling, premature optimization |

---

## 3. Implementation Phases

### Phase 1: Obsolete Term Filtering (Issue #133)
**Priority**: HIGH | **Effort**: 1 day

1. Add `is_obsolete_term()` function to `hpo_parser.py`
2. Extend `_parse_hpo_json_to_graphs()` to filter obsolete terms
3. Store obsolete term metadata for reference (optional migration tracking)
4. Update tests to verify filtering

### Phase 2: Bundle Manifest & Packaging
**Priority**: HIGH | **Effort**: 1 day

1. Define `BundleManifest` dataclass
2. Create `phentrieve/data_processing/bundle_packager.py`
3. Implement checksum generation (SHA-256)
4. Add CLI command: `phentrieve data package`

### Phase 3: GitHub Releases Integration
**Priority**: HIGH | **Effort**: 1 day

1. Create `.github/workflows/build-data-bundles.yml`
2. Implement automated bundle building on HPO release
3. Upload bundles to GitHub Releases with proper tagging
4. Create release notes template

### Phase 4: CLI Download & Verification
**Priority**: HIGH | **Effort**: 1 day

1. Add `phentrieve data download` command
2. Add `phentrieve data verify` command
3. Add `phentrieve data list-bundles` command
4. Add `phentrieve data status` command
5. Implement progress bars and retry logic

### Phase 5: Docker Integration
**Priority**: MEDIUM | **Effort**: 0.5 days

1. Update API Dockerfile with optional pre-built data layer
2. Add build-time download from GitHub Releases
3. Configure cache layers for minimal rebuilds
4. Update docker-compose for volume mounting

### Phase 6: Documentation & Testing
**Priority**: MEDIUM | **Effort**: 0.5 days

1. Update CLAUDE.md with new commands
2. Add E2E tests for bundle download/verify
3. Add integration tests for obsolete filtering
4. Update README with distribution options

---

## 4. Technical Specifications

### 4.1 Obsolete Term Detection

**File**: `phentrieve/data_processing/hpo_parser.py`

```python
def is_obsolete_term(node_data: dict) -> bool:
    """
    Check if an HPO term is obsolete using OBO Graphs conventions.

    Detection criteria (per HPO Obsoletion Wiki):
    1. meta.deprecated == true (OWL:deprecated annotation)
    2. Label starts with "obsolete " (case-insensitive)

    Args:
        node_data: Raw node dictionary from HPO JSON

    Returns:
        True if term is obsolete, False otherwise
    """
    # Check deprecated flag in meta
    deprecated = safe_get_nested(node_data, "meta", "deprecated", default=False)
    if deprecated is True:
        return True

    # Check label prefix (case-insensitive)
    label = node_data.get("lbl", "")
    if label.lower().startswith("obsolete "):
        return True

    return False


def get_replacement_term(node_data: dict) -> Optional[str]:
    """
    Get the replacement term ID for an obsolete term.

    Uses IAO_0100001 (term replaced by) annotation.

    Args:
        node_data: Raw node dictionary from HPO JSON

    Returns:
        Replacement term ID (e.g., "HP:0000510") or None
    """
    basic_props = safe_get_list(node_data, "meta", "basicPropertyValues", default=[])
    for prop in basic_props:
        pred = prop.get("pred", "")
        if "IAO_0100001" in pred:  # term_replaced_by
            val = prop.get("val", "")
            return normalize_id(val) if val else None
    return None
```

### 4.2 Bundle Manifest Schema

**File**: `phentrieve/data_processing/bundle_manifest.py`

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json

@dataclass(frozen=True)
class EmbeddingModelInfo:
    """Embedding model metadata for bundle."""
    name: str                    # e.g., "FremyCompany/BioLORD-2023-M"
    dimension: int               # e.g., 768
    distance_metric: str         # e.g., "cosine"
    collection_name: str         # e.g., "hpo_FremyCompany_BioLORD-2023-M"

@dataclass
class BundleManifest:
    """
    Manifest for pre-built data bundles.

    Stored as manifest.json in bundle root.
    """
    # Versioning
    manifest_version: str = "1.0.0"
    bundle_format: str = "tar.gz"

    # HPO Data
    hpo_version: str = ""           # e.g., "v2025-03-03"
    hpo_release_date: str = ""      # e.g., "2025-03-03"
    hpo_source_url: str = ""        # GitHub release URL

    # Term Statistics
    total_terms: int = 0            # Before filtering
    active_terms: int = 0           # After filtering obsolete
    obsolete_terms: int = 0         # Filtered count

    # Embedding Model
    model: Optional[EmbeddingModelInfo] = None

    # Bundle Metadata
    created_at: str = ""            # ISO 8601 timestamp
    created_by: str = "phentrieve"  # Build system identifier
    phentrieve_version: str = ""    # Package version used

    # Checksums (SHA-256)
    checksums: dict = field(default_factory=dict)
    # {
    #   "hpo_data.db": "abc123...",
    #   "indexes/chromadb/": "def456...",
    # }

    def to_json(self) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "BundleManifest":
        """Deserialize manifest from JSON string."""
        data = json.loads(json_str)
        if data.get("model"):
            data["model"] = EmbeddingModelInfo(**data["model"])
        return cls(**data)
```

### 4.3 Bundle File Structure

```
phentrieve-data-v2025-03-03-biolord.tar.gz
├── manifest.json              # Bundle metadata
├── hpo_data.db               # SQLite database (~12 MB)
├── hp.json                   # Original HPO JSON (optional, for reference)
├── indexes/
│   └── FremyCompany_BioLORD-2023-M/
│       ├── chroma.sqlite3    # ChromaDB data
│       └── ...               # ChromaDB files
└── checksums.sha256          # SHA-256 checksums for all files
```

### 4.4 Version Naming Convention

**Bundle Filename Pattern**:
```
phentrieve-data-{hpo_version}-{model_slug}.tar.gz
```

**Examples**:
- `phentrieve-data-v2025-03-03-biolord.tar.gz` (BioLORD)
- `phentrieve-data-v2025-03-03-bge-m3.tar.gz` (BGE-M3)
- `phentrieve-data-v2025-03-03-labse.tar.gz` (LaBSE)

**Model Slug Mapping**:
```python
MODEL_SLUGS = {
    "FremyCompany/BioLORD-2023-M": "biolord",
    "jinaai/jina-embeddings-v2-base-de": "jina-de",
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer": "tsystems-ende",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet-multi",
    "sentence-transformers/distiluse-base-multilingual-cased-v2": "distiluse-multi",
    "BAAI/bge-m3": "bge-m3",
    "Alibaba-NLP/gte-multilingual-base": "gte-multi",
    "sentence-transformers/LaBSE": "labse",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "minilm-multi",
}
```

**GitHub Release Tags**:
- `data-v2025-03-03` - Specific HPO version
- `data-latest` - Latest available data (updated on new HPO releases)
- `data-recommended` - Recommended stable bundle (BioLORD model)

---

## 5. Bundle Format & Versioning

### 5.1 Bundle Types

| Type | Description | Size (est.) | Use Case |
|------|-------------|-------------|----------|
| **Minimal** | SQLite only (no embeddings) | ~15 MB | Custom model training |
| **Single Model** | SQLite + 1 ChromaDB index | ~140 MB | Production deployment |
| **Full** | SQLite + all 9 model indexes | ~1.2 GB | Benchmarking/research |

### 5.2 HPO Version Strategy

**Supported Modes**:

1. **Latest** (default): Automatically use newest HPO release
   ```yaml
   hpo_data:
     version: "latest"
   ```

2. **Pinned**: Use specific HPO version for reproducibility
   ```yaml
   hpo_data:
     version: "v2025-03-03"
   ```

**Version Resolution**:
```python
def resolve_hpo_version(version_spec: str) -> str:
    """
    Resolve HPO version specification to concrete version.

    Args:
        version_spec: "latest" or specific version like "v2025-03-03"

    Returns:
        Concrete version string (e.g., "v2025-03-03")
    """
    if version_spec == "latest":
        # Query GitHub API for latest HPO release
        return fetch_latest_hpo_version()
    return version_spec
```

### 5.3 Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Old bundle + new Phentrieve | Validate manifest version, warn if schema changes |
| New bundle + old Phentrieve | Graceful fallback if unknown fields |
| Missing manifest | Treat as legacy local build |

---

## 6. CLI Commands

### 6.1 New Commands

```bash
# Download pre-built data bundle
phentrieve data download [OPTIONS]
  --model TEXT        Model slug (biolord, bge-m3, etc.) or "all"
  --hpo-version TEXT  HPO version (default: latest)
  --preset TEXT       Preset: "recommended", "minimal", "full"
  --force            Overwrite existing data
  --verify           Verify checksums after download

# List available bundles
phentrieve data list-bundles [OPTIONS]
  --hpo-version TEXT  Filter by HPO version
  --model TEXT        Filter by model

# Verify local data integrity
phentrieve data verify [OPTIONS]
  --fix              Attempt to fix corrupted files by re-downloading

# Show data status
phentrieve data status
  # Output:
  # HPO Version: v2025-03-03 (active terms: 16,934, obsolete filtered: 2,600)
  # Database: /path/to/hpo_data.db (12.4 MB, verified)
  # Indexes:
  #   - BioLORD-2023-M: 16,934 vectors, 105 MB
  #   - BGE-M3: 16,934 vectors, 98 MB
  # Source: pre-built bundle (downloaded 2025-12-09)

# Package local data as bundle (for maintainers)
phentrieve data package [OPTIONS]
  --model TEXT        Model to include
  --output PATH       Output directory
  --include-hpo-json  Include original hp.json
```

### 6.2 Enhanced Existing Commands

```bash
# Enhanced data prepare with obsolete filtering
phentrieve data prepare [OPTIONS]
  --hpo-version TEXT   HPO version to download (default: from config)
  --include-obsolete   Include obsolete terms (for analysis)
  --force              Force re-download

# Enhanced index build with bundle awareness
phentrieve index build [OPTIONS]
  --use-bundle         Use pre-built bundle if available
  --model-name TEXT    Specific model or "all"
```

---

## 7. Docker Integration

### 7.1 Multi-Stage Build Strategy

```dockerfile
# =============================================================================
# Stage 1: Data Download (Optional - for pre-built bundles)
# =============================================================================
FROM alpine:3.20 AS data-downloader

ARG HPO_VERSION=latest
ARG MODEL_SLUG=biolord
ARG BUNDLE_URL=""

# Install tools
RUN apk add --no-cache curl tar

# Download pre-built bundle OR skip for local build
RUN if [ -n "$BUNDLE_URL" ]; then \
        echo "Downloading pre-built bundle..." && \
        curl -fsSL "$BUNDLE_URL" -o /bundle.tar.gz && \
        mkdir -p /data && \
        tar -xzf /bundle.tar.gz -C /data && \
        rm /bundle.tar.gz; \
    else \
        echo "No bundle URL provided, skipping pre-built data"; \
        mkdir -p /data; \
    fi

# =============================================================================
# Stage 2: Python Dependencies (existing)
# =============================================================================
FROM python:3.11-slim-bookworm AS python-deps
# ... existing dependency installation ...

# =============================================================================
# Stage 3: Runtime (enhanced)
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Copy pre-built data if available
COPY --from=data-downloader /data /phentrieve_data_mount

# Copy Python environment
COPY --from=python-deps /opt/venv /opt/venv

# ... rest of existing runtime stage ...
```

### 7.2 Docker Compose Configuration

```yaml
services:
  phentrieve_api:
    image: ghcr.io/berntpopp/phentrieve/api:latest
    build:
      context: .
      dockerfile: api/Dockerfile
      args:
        # Use pre-built bundle for faster builds
        BUNDLE_URL: "https://github.com/berntpopp/phentrieve/releases/download/data-v2025-03-03/phentrieve-data-v2025-03-03-biolord.tar.gz"
        HPO_VERSION: "v2025-03-03"
        MODEL_SLUG: "biolord"
    volumes:
      # Mount for persistent storage (overrides built-in data)
      - phentrieve_data:/phentrieve_data_mount
    environment:
      PHENTRIEVE_DATA_ROOT_DIR: /phentrieve_data_mount
```

### 7.3 Build Time Optimization

| Scenario | Build Time | Image Size |
|----------|------------|------------|
| **Current** (build from scratch) | 15-20 min | ~3.5 GB |
| **With pre-built bundle** | 3-5 min | ~3.5 GB |
| **Cached rebuild** | 30-60 sec | ~3.5 GB |

---

## 8. GitHub Actions Workflows

### 8.1 Bundle Build Workflow

**File**: `.github/workflows/build-data-bundles.yml`

```yaml
name: Build Data Bundles

on:
  # Manual trigger with version selection
  workflow_dispatch:
    inputs:
      hpo_version:
        description: 'HPO version (e.g., v2025-03-03) or "latest"'
        required: true
        default: 'latest'
      models:
        description: 'Models to build (comma-separated slugs or "all")'
        required: true
        default: 'biolord'
      create_release:
        description: 'Create GitHub Release'
        type: boolean
        default: true

  # Scheduled check for new HPO releases (monthly)
  schedule:
    - cron: '0 0 1 * *'  # First day of each month

permissions:
  contents: write  # For creating releases

jobs:
  check-hpo-version:
    runs-on: ubuntu-latest
    outputs:
      hpo_version: ${{ steps.resolve.outputs.version }}
      needs_build: ${{ steps.check.outputs.needs_build }}
    steps:
      - name: Resolve HPO version
        id: resolve
        run: |
          if [ "${{ inputs.hpo_version }}" = "latest" ] || [ -z "${{ inputs.hpo_version }}" ]; then
            VERSION=$(curl -sL https://api.github.com/repos/obophenotype/human-phenotype-ontology/releases/latest | jq -r '.tag_name')
          else
            VERSION="${{ inputs.hpo_version }}"
          fi
          echo "version=$VERSION" >> $GITHUB_OUTPUT

      - name: Check if bundle exists
        id: check
        run: |
          # Check if release already exists
          RELEASE_EXISTS=$(curl -sL "https://api.github.com/repos/${{ github.repository }}/releases/tags/data-${{ steps.resolve.outputs.version }}" | jq -r '.id // empty')
          if [ -z "$RELEASE_EXISTS" ]; then
            echo "needs_build=true" >> $GITHUB_OUTPUT
          else
            echo "needs_build=false" >> $GITHUB_OUTPUT
          fi

  build-bundles:
    needs: check-hpo-version
    if: needs.check-hpo-version.outputs.needs_build == 'true' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        model: ${{ fromJson(needs.parse-models.outputs.models) }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Prepare HPO data
        run: |
          uv run phentrieve data prepare \
            --hpo-version ${{ needs.check-hpo-version.outputs.hpo_version }}

      - name: Build index for model
        run: |
          uv run phentrieve index build \
            --model-name "${{ matrix.model }}"

      - name: Package bundle
        run: |
          uv run phentrieve data package \
            --model "${{ matrix.model }}" \
            --output ./bundles

      - name: Upload bundle artifact
        uses: actions/upload-artifact@v4
        with:
          name: bundle-${{ matrix.model }}
          path: ./bundles/*.tar.gz

  create-release:
    needs: [check-hpo-version, build-bundles]
    if: inputs.create_release
    runs-on: ubuntu-latest
    steps:
      - name: Download all bundle artifacts
        uses: actions/download-artifact@v4
        with:
          path: bundles
          pattern: bundle-*

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          tag_name: data-${{ needs.check-hpo-version.outputs.hpo_version }}
          name: "Data Bundle - HPO ${{ needs.check-hpo-version.outputs.hpo_version }}"
          body: |
            ## Pre-built Data Bundles

            **HPO Version**: ${{ needs.check-hpo-version.outputs.hpo_version }}
            **Build Date**: ${{ github.run_id }}

            ### Available Bundles

            | Model | Size | Download |
            |-------|------|----------|
            | BioLORD-2023-M | ~140 MB | [Download](./phentrieve-data-${{ needs.check-hpo-version.outputs.hpo_version }}-biolord.tar.gz) |

            ### Quick Install

            ```bash
            phentrieve data download --preset recommended
            ```

            ### Checksums

            See `checksums.sha256` in each bundle for verification.
          files: bundles/**/*.tar.gz
          fail_on_unmatched_files: false
```

### 8.2 Automated Monthly Check

The workflow includes a scheduled trigger (`cron: '0 0 1 * *'`) that:
1. Checks for new HPO releases
2. Compares with existing data bundles
3. Triggers build if new version available
4. Creates release with updated bundles

---

## 9. Success Criteria

### 9.1 Functional Requirements

| Requirement | Metric | Target |
|-------------|--------|--------|
| Obsolete term filtering | Terms filtered | ~2,600 (~13.5%) |
| Self-match accuracy | After filtering | >99% |
| Download time | Single bundle | < 2 minutes |
| Verification | Checksum validation | 100% coverage |
| Docker build | With pre-built data | < 5 minutes |

### 9.2 Quality Requirements

| Requirement | Metric | Target |
|-------------|--------|--------|
| Test coverage | New code | > 80% |
| Documentation | Commands documented | 100% |
| Backward compatibility | Existing workflows | No breaking changes |
| Error handling | Network failures | Graceful retry + fallback |

### 9.3 Supported Models (9 Benchmark Models)

| Model | Slug | Priority | Notes |
|-------|------|----------|-------|
| FremyCompany/BioLORD-2023-M | `biolord` | **Recommended** | Domain-specific biomedical |
| BAAI/bge-m3 | `bge-m3` | High | Strong multilingual |
| sentence-transformers/LaBSE | `labse` | High | Cross-lingual |
| jinaai/jina-embeddings-v2-base-de | `jina-de` | Medium | German-specific |
| T-Systems-onsite/cross-en-de-roberta-sentence-transformer | `tsystems-ende` | Medium | EN-DE cross-lingual |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | `mpnet-multi` | Medium | General multilingual |
| sentence-transformers/distiluse-base-multilingual-cased-v2 | `distiluse-multi` | Medium | Lightweight |
| Alibaba-NLP/gte-multilingual-base | `gte-multi` | Medium | GTE multilingual |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | `minilm-multi` | Low | Mini version |

---

## 10. Risk Mitigation

### 10.1 Identified Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| GitHub Release 2GB limit | High | Bundle size ~140MB, well under limit |
| HPO schema changes | Medium | Defensive parsing already implemented (Issue #23) |
| Network failures | Medium | Retry logic + local fallback |
| ChromaDB version mismatch | Medium | Pin ChromaDB version in manifest |
| Model version drift | Low | Store model name in manifest |

### 10.2 Fallback Strategy

```python
def get_data(prefer_prebuilt: bool = True) -> DataSource:
    """
    Data acquisition with graceful fallback.

    Priority:
    1. Pre-built bundle (if prefer_prebuilt and available)
    2. Local cached data
    3. Build from scratch (download HPO + generate embeddings)
    """
    if prefer_prebuilt:
        try:
            return download_prebuilt_bundle()
        except DownloadError:
            logger.warning("Pre-built bundle unavailable, falling back to local build")

    if local_data_exists():
        return load_local_data()

    return build_from_scratch()
```

---

## 11. File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `phentrieve/data_processing/bundle_manifest.py` | Bundle manifest dataclass |
| `phentrieve/data_processing/bundle_packager.py` | Bundle creation logic |
| `phentrieve/data_processing/bundle_downloader.py` | Bundle download + verify |
| `phentrieve/cli/data_download_commands.py` | New CLI commands |
| `.github/workflows/build-data-bundles.yml` | GitHub Actions workflow |
| `tests/unit/test_obsolete_filtering.py` | Unit tests for filtering |
| `tests/integration/test_bundle_download.py` | Integration tests for bundles |

### Modified Files

| File | Changes |
|------|---------|
| `phentrieve/data_processing/hpo_parser.py` | Add `is_obsolete_term()`, filter in parsing |
| `phentrieve/config.py` | Add `MODEL_SLUGS` mapping |
| `phentrieve/cli/data_commands.py` | Integrate new subcommands |
| `api/Dockerfile` | Add optional pre-built data stage |
| `docker-compose.yml` | Add bundle URL build arg |
| `CLAUDE.md` | Document new commands |

---

## 12. Implementation Checklist

### Phase 1: Obsolete Term Filtering ☐
- [ ] Add `is_obsolete_term()` function
- [ ] Add `get_replacement_term()` function
- [ ] Modify `_parse_hpo_json_to_graphs()` to filter
- [ ] Add statistics logging for filtered terms
- [ ] Write unit tests

### Phase 2: Bundle Packaging ☐
- [ ] Create `bundle_manifest.py`
- [ ] Create `bundle_packager.py`
- [ ] Implement checksum generation
- [ ] Add `phentrieve data package` command
- [ ] Write unit tests

### Phase 3: GitHub Releases ☐
- [ ] Create workflow file
- [ ] Test manual trigger
- [ ] Test scheduled trigger
- [ ] Verify release creation

### Phase 4: CLI Download ☐
- [ ] Create `bundle_downloader.py`
- [ ] Add `phentrieve data download` command
- [ ] Add `phentrieve data verify` command
- [ ] Add `phentrieve data list-bundles` command
- [ ] Add `phentrieve data status` command
- [ ] Write integration tests

### Phase 5: Docker Integration ☐
- [ ] Update Dockerfile with data stage
- [ ] Update docker-compose.yml
- [ ] Test build with pre-built data
- [ ] Document in CLAUDE.md

### Phase 6: Documentation ☐
- [ ] Update CLAUDE.md
- [ ] Update README.md
- [ ] Add troubleshooting guide
- [ ] Write E2E tests

---

## References

- [GitHub Releases Best Practices](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [HPO Obsoletion Wiki](https://github.com/obophenotype/human-phenotype-ontology/wiki/Obsoletion)
- [OBO Graphs Specification](https://github.com/geneontology/obographs)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [softprops/action-gh-release](https://github.com/softprops/action-gh-release)
- [svenstaro/upload-release-action](https://github.com/svenstaro/upload-release-action)
