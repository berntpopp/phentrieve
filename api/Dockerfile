# syntax=docker/dockerfile:1.11
# Pin to specific Dockerfile syntax version for reproducibility

# ============================================================================
# ARGUMENTS - Define all build arguments upfront
# ============================================================================
ARG PYTHON_VERSION=3.11.11
ARG DEBIAN_VERSION=bookworm
# SQLite 3.51.1 fixes CVE-2025-7458 and CVE-2025-6965 (critical integer overflow vulnerabilities)
ARG SQLITE_VERSION=3510100
ARG SQLITE_YEAR=2025
# Note: SQLite 3.51+ uses SHA3-256 checksums (not SHA256)
ARG SQLITE_SHA3_256=9b2b1e73f577def1d5b75c5541555a7f42e6e073ad19f7a9118478389c9bbd9b

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

# Copy SQLite from builder
COPY --from=sqlite-builder /usr/local/lib/libsqlite3.* /usr/local/lib/
COPY --from=sqlite-builder /usr/local/bin/sqlite3 /usr/local/bin/
COPY --from=sqlite-builder /usr/local/include/sqlite3*.h /usr/local/include/

# Set SQLite library path with conditional expansion to avoid undefined variable warning
ENV LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

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
# STAGE 3: Runtime - Minimal production image
# ============================================================================
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS runtime

# Metadata labels (OCI standard)
LABEL org.opencontainers.image.title="Phentrieve API" \
      org.opencontainers.image.description="AI-powered HPO term mapping API" \
      org.opencontainers.image.vendor="Phentrieve" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/berntpopp/phentrieve" \
      org.opencontainers.image.documentation="https://github.com/berntpopp/phentrieve/blob/main/README.md" \
      org.opencontainers.image.authors="Bernt Popp <bernt.popp@gmail.com>"

# Install runtime dependencies and apply security updates
# The apt-get upgrade fetches latest security patches for base image packages
# (addresses CVEs in glibc, openssl, perl, linux-pam, openldap, gnutls, ncurses, etc.)
# hadolint ignore=DL3008,DL3009
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    # Apply security updates to base image packages
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        curl \
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

# Copy SQLite runtime libraries
COPY --from=sqlite-builder /usr/local/lib/libsqlite3.* /usr/local/lib/
COPY --from=sqlite-builder /usr/local/bin/sqlite3 /usr/local/bin/
# Set SQLite library path with conditional expansion to avoid undefined variable warning
ENV LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Copy virtual environment from builder
COPY --from=python-builder /opt/venv /opt/venv
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

# Create data mount point with correct permissions
RUN mkdir -p /phentrieve_data_mount && \
    chown -R phentrieve:phentrieve /phentrieve_data_mount

# Create cache directory for HuggingFace models
RUN mkdir -p /app/.cache && \
    chown -R phentrieve:phentrieve /app/.cache

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PHENTRIEVE_DATA_ROOT_DIR=/phentrieve_data_mount \
    PHENTRIEVE_DATA_DIR=/phentrieve_data_mount/hpo_core_data \
    PHENTRIEVE_INDEX_DIR=/phentrieve_data_mount/indexes \
    PHENTRIEVE_RESULTS_DIR=/phentrieve_data_mount/results \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

# Switch to non-root user (SECURITY: CRITICAL)
USER phentrieve

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Use exec form for proper signal handling
ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
