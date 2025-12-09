# syntax=docker/dockerfile:1.12
# Pin to specific Dockerfile syntax version for reproducibility
# Updated to 1.12 for latest BuildKit features (2025 best practice)

# ============================================================================
# ARGUMENTS - Define all build arguments upfront
# ============================================================================
# Python 3.12 offers ~5% performance improvement and is the latest stable LTS
# Keep at 3.11 for now due to dependency compatibility (spaCy, torch, etc.)
ARG PYTHON_VERSION=3.11.11
ARG DEBIAN_VERSION=bookworm
# SQLite 3.51.1 fixes CVE-2025-7458 and CVE-2025-6965 (critical integer overflow vulnerabilities)
ARG SQLITE_VERSION=3510100
ARG SQLITE_YEAR=2025
# Note: SQLite 3.51+ uses SHA3-256 checksums (not SHA256)
ARG SQLITE_SHA3_256=9b2b1e73f577def1d5b75c5541555a7f42e6e073ad19f7a9118478389c9bbd9b

# =============================================================================
# DATA PREPARATION MODE CONFIGURATION (Issue #117)
# =============================================================================
# Three modes for HPO data preparation:
#
# MODE 1: Download pre-built bundle (DEFAULT - fastest, ~1-2 min)
#   - Set BUNDLE_URL to the bundle download URL
#   - Default: BioLORD bundle (recommended model with MRR@10: 0.823)
#   docker build --build-arg BUNDLE_URL="https://..." .
#
# MODE 2: Build indexes from scratch (~10-20 min)
#   - Set BUNDLE_URL="" and BUILD_INDEXES="true"
#   - Optionally set BUILD_MODEL for specific model
#   docker build --build-arg BUNDLE_URL="" --build-arg BUILD_INDEXES="true" .
#
# MODE 3: External volume mount (no embedded data)
#   - Set BUNDLE_URL="" and BUILD_INDEXES="false"
#   - Mount data volume at runtime: -v /path/to/data:/phentrieve_data_mount
#   docker build --build-arg BUNDLE_URL="" .
# =============================================================================

# Pre-built data bundle URL (Mode 1)
# Default: BioLORD bundle - recommended model with best MRR@10 (0.823)
ARG BUNDLE_URL="https://github.com/berntpopp/phentrieve/releases/download/data-v2025-11-24/phentrieve-data-v2025-11-24-biolord.tar.gz"

# Build indexes from scratch (Mode 2) - requires BUNDLE_URL=""
ARG BUILD_INDEXES="false"

# Model to use when building indexes (only used if BUILD_INDEXES="true")
ARG BUILD_MODEL="FremyCompany/BioLORD-2023-M"

# ============================================================================
# STAGE 1: SQLite Builder - Build custom SQLite with required extensions
# ============================================================================
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS sqlite-builder

ARG SQLITE_VERSION
ARG SQLITE_YEAR
ARG SQLITE_SHA3_256

# Install only necessary build tools
# Note: openssl is required for SHA3-256 checksum verification (SQLite 3.51+ uses SHA3)
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
        openssl \
    && rm -rf /var/lib/apt/lists/*

# Download, verify, and build SQLite with checksum validation
WORKDIR /build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Note: cd is used within RUN because SQLITE_VERSION is a build arg that can't be
# interpolated in WORKDIR, and we need atomic download-verify-build for security
# hadolint ignore=DL3003
RUN set -euxo pipefail && \
    # Download SQLite source (HTTPS for security) \
    wget -q https://www.sqlite.org/${SQLITE_YEAR}/sqlite-autoconf-${SQLITE_VERSION}.tar.gz && \
    # Verify SHA3-256 checksum (CRITICAL SECURITY - prevents MITM attacks) \
    # SQLite 3.51+ uses SHA3-256 instead of SHA256 \
    COMPUTED_HASH=$(openssl dgst -sha3-256 sqlite-autoconf-${SQLITE_VERSION}.tar.gz | awk '{print $2}') && \
    [ "$COMPUTED_HASH" = "${SQLITE_SHA3_256}" ] || { echo "SHA3-256 checksum mismatch!"; exit 1; } && \
    # Extract and build with optimizations \
    tar -xzf sqlite-autoconf-${SQLITE_VERSION}.tar.gz && \
    cd sqlite-autoconf-${SQLITE_VERSION} && \
    # Note: SQLite 3.51+ has JSON1 built-in (--enable-json1 removed)
    ./configure --prefix=/usr/local \
        --enable-fts5 \
        --enable-rtree && \
    make -j"$(nproc)" && \
    make install && \
    # Cleanup \
    cd /build && \
    rm -rf sqlite-autoconf-${SQLITE_VERSION}*
SHELL ["/bin/sh", "-c"]

# ============================================================================
# STAGE 2: Python Dependencies Builder - Install all Python packages
# ============================================================================
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS python-builder

# Copy SQLite from builder using --link for independent layer rebasing (2025 best practice)
# --link creates layers that can be rebased onto new base images without rebuilding
COPY --link --from=sqlite-builder /usr/local/lib/libsqlite3.* /usr/local/lib/
COPY --link --from=sqlite-builder /usr/local/bin/sqlite3 /usr/local/bin/
COPY --link --from=sqlite-builder /usr/local/include/sqlite3*.h /usr/local/include/

# Set SQLite library path (base image has no custom LD_LIBRARY_PATH to preserve)
ENV LD_LIBRARY_PATH=/usr/local/lib

# Install build dependencies for Python packages
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        libssl-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolation
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip, setuptools, wheel to patched versions
# - setuptools >= 78.1.1 fixes CVE-2025-47273 (path traversal/arbitrary file write)
# - setuptools >= 70.0.0 fixes CVE-2024-6345 (path traversal) and CVE-2022-40897 (RCE)
# - pip >= 24.1 fixes symbolic link extraction vulnerability
# hadolint ignore=DL3042,DL3013
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade "pip>=24.1" "setuptools>=78.1.1" wheel

WORKDIR /build

# Copy only dependency files first for better caching
COPY pyproject.toml ./

# Install dependencies in specific order to handle binary compatibility
# CRITICAL: Remove --no-cache-dir to leverage pip cache mount for 60-90% faster builds
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# hadolint ignore=DL3042,DL3013
RUN --mount=type=cache,target=/root/.cache/pip \
    set -euxo pipefail && \
    # Install NumPy 2.x first (modern ABI) \
    pip install "numpy>=2.0.0,<3.0.0" && \
    # Install spaCy 3.8+ (compatible with NumPy 2.x and Pydantic v2) \
    pip install "spacy>=3.8.0,<4.0.0" && \
    pip install \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl && \
    pip install \
        https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl && \
    # Install langdetect for language auto-detection (fixes #VULN-007) \
    pip install langdetect && \
    # Install API dependencies (FastAPI, uvicorn) \
    pip install \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.35.0" \
        "python-dotenv>=1.0.0" \
        "pydantic>=2.0.0" \
        "pydantic-settings>=2.0.0" && \
    # Install remaining dependencies from pyproject.toml \
    pip install .
SHELL ["/bin/sh", "-c"]

# ============================================================================
# STAGE 3: Data Bundle - Download pre-built data OR build indexes from scratch
# ============================================================================
# This stage handles data preparation in three modes:
#   Mode 1: BUNDLE_URL set (default) - Download pre-built bundle (~1-2 min)
#   Mode 2: BUILD_INDEXES=true - Build indexes from scratch (~10-20 min)
#   Mode 3: Neither - Placeholder for external volume mount
#
# Uses python-builder as base to support all modes (index building needs deps)
# Only /data directory is copied to runtime stage - doesn't affect final image size
FROM python-builder AS data-bundle

ARG BUNDLE_URL
ARG BUILD_INDEXES
ARG BUILD_MODEL

# Install wget for downloading bundle or HPO source data
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code (needed for BUILD_INDEXES mode)
# This is a no-op for bundle download mode but required for building
COPY pyproject.toml ./
COPY phentrieve/ ./phentrieve/

# Conditionally install phentrieve CLI for index building mode
# hadolint ignore=DL3013
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUILD_INDEXES" = "true" ]; then \
        echo "Installing phentrieve CLI for index building..." && \
        pip install --no-deps . && \
        echo "Phentrieve installed successfully"; \
    fi

# Set environment for phentrieve commands
ENV PHENTRIEVE_DATA_ROOT_DIR=/data

# Create data directory
WORKDIR /data

# Main data preparation logic - three modes
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN set -euxo pipefail && \
    if [ -n "$BUNDLE_URL" ]; then \
        echo "========================================" && \
        echo "MODE 1: Download pre-built bundle" && \
        echo "========================================" && \
        echo "Downloading from: $BUNDLE_URL" && \
        wget --progress=dot:giga -O bundle.tar.gz "$BUNDLE_URL" && \
        echo "Extracting bundle..." && \
        tar -xzf bundle.tar.gz && \
        rm bundle.tar.gz && \
        echo "" && \
        echo "Bundle contents:" && \
        ls -la /data/ && \
        echo "" && \
        echo "SUCCESS: Pre-built data bundle ready!"; \
    elif [ "$BUILD_INDEXES" = "true" ]; then \
        echo "========================================" && \
        echo "MODE 2: Build indexes from scratch" && \
        echo "Model: $BUILD_MODEL" && \
        echo "========================================" && \
        echo "" && \
        echo "Step 1/2: Downloading and preparing HPO data..." && \
        phentrieve data prepare && \
        echo "" && \
        echo "Step 2/2: Building vector indexes (this may take 10-20 minutes)..." && \
        phentrieve index build --model "$BUILD_MODEL" && \
        echo "" && \
        echo "Build complete! Contents:" && \
        ls -la /data/ && \
        echo "" && \
        echo "SUCCESS: Indexes built from scratch!"; \
    else \
        echo "========================================" && \
        echo "MODE 3: External volume mount" && \
        echo "========================================" && \
        echo "No embedded data - mount volume at runtime:" && \
        echo "  -v /path/to/data:/phentrieve_data_mount" && \
        echo "" && \
        mkdir -p /data/placeholder && \
        touch /data/placeholder/.external_mount && \
        echo "Placeholder created for external mount."; \
    fi
SHELL ["/bin/sh", "-c"]

# ============================================================================
# STAGE 4: Runtime - Minimal production image
# ============================================================================
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS runtime

ARG BUNDLE_URL

# Metadata labels (OCI standard)
LABEL org.opencontainers.image.title="Phentrieve API" \
      org.opencontainers.image.description="AI-powered HPO term mapping API" \
      org.opencontainers.image.vendor="Phentrieve" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/berntpopp/phentrieve" \
      org.opencontainers.image.documentation="https://github.com/berntpopp/phentrieve/blob/main/README.md" \
      org.opencontainers.image.authors="Bernt Popp <bernt.popp@gmail.com>"

# Install runtime dependencies only
# NOTE: Security updates should be handled by updating the base image version,
# not via apt-get upgrade (which is slow, non-cacheable, and non-reproducible).
# Pin base image versions in CI/CD and update them regularly instead.
# gosu is required for the entrypoint script to drop privileges after fixing permissions
# See: https://www.docker.com/blog/understanding-the-docker-user-instruction/
# hadolint ignore=DL3008,DL3009
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        curl \
        gosu \
    && rm -rf /var/lib/apt/lists/* \
    # Security hardening: Remove unnecessary setuid/setgid bits (CIS-DI-0008)
    # These are not needed for the API container runtime
    && find /usr/bin /usr/sbin /bin /sbin -perm /6000 -type f -exec chmod a-s {} \; || true \
    # Remove vulnerable setuptools from base image system packages (CVE-2025-47273, CVE-2024-6345)
    # The base python:3.11-slim-bookworm ships with setuptools 65.5.1 which has security vulnerabilities
    # We use a clean virtual environment with upgraded setuptools, so system setuptools is not needed
    && pip uninstall -y setuptools 2>/dev/null || true \
    && rm -rf /usr/local/lib/python*/site-packages/setuptools* \
              /usr/local/lib/python*/site-packages/_distutils_hack* \
              /usr/local/lib/python*/site-packages/pkg_resources* 2>/dev/null || true

# Copy SQLite runtime libraries using --link for independent layer rebasing
COPY --link --from=sqlite-builder /usr/local/lib/libsqlite3.* /usr/local/lib/
COPY --link --from=sqlite-builder /usr/local/bin/sqlite3 /usr/local/bin/
# Set SQLite library path (base image has no custom LD_LIBRARY_PATH to preserve)
ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy virtual environment from builder using --link for independent layer rebasing
COPY --link --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

# Create non-root user (SECURITY: CRITICAL)
# Using UID 10001 to avoid conflicts with host users
RUN groupadd -r -g 10001 phentrieve && \
    useradd -r -u 10001 -g phentrieve -m -d /app -s /sbin/nologin phentrieve

# Set working directory
WORKDIR /app

# Copy application code with correct ownership
COPY --chown=phentrieve:phentrieve phentrieve/ ./phentrieve/
COPY --chown=phentrieve:phentrieve api/ ./api/
# Copy root pyproject.toml for CLI version detection at runtime
# Note: api/pyproject.toml is already copied with api/ directory above
COPY --chown=phentrieve:phentrieve pyproject.toml ./

# Copy entrypoint script (must be executable)
# The entrypoint handles runtime permission fixing for bind mounts
# See: https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
COPY --chmod=755 api/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Create data mount point with correct permissions
RUN mkdir -p /phentrieve_data_mount && \
    chown -R phentrieve:phentrieve /phentrieve_data_mount

# Copy pre-built data bundle if BUNDLE_URL was provided (Issue #117)
# This enables building images with embedded HPO data for faster cold starts
# The bundle includes: hpo_data.db, indexes/, and manifest.json
# Using --link for independent layer rebasing (2025 best practice)
COPY --link --from=data-bundle --chown=phentrieve:phentrieve /data/ /phentrieve_data_mount/

# Create cache directory for HuggingFace models
RUN mkdir -p /app/.cache && \
    chown -R phentrieve:phentrieve /app/.cache

# Environment variables
# Note: PHENTRIEVE_DATA_DIR defaults to PHENTRIEVE_DATA_ROOT_DIR when not set
# The pre-built bundle puts hpo_data.db directly in the data root (not in a subdirectory)
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PHENTRIEVE_DATA_ROOT_DIR=/phentrieve_data_mount \
    PHENTRIEVE_INDEX_DIR=/phentrieve_data_mount/indexes \
    PHENTRIEVE_RESULTS_DIR=/phentrieve_data_mount/results \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

# NOTE: No USER directive here - entrypoint starts as root to fix permissions,
# then drops to non-root user (phentrieve:10001) before running the application.
# This follows Docker best practices for handling volume permissions:
# https://www.docker.com/blog/understanding-the-docker-user-instruction/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Use entrypoint script for runtime permission handling
# The script fixes volume permissions, then drops to non-root user
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
