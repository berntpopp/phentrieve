# syntax=docker/dockerfile:1.11
# Pin to specific Dockerfile syntax version for reproducibility

# ============================================================================
# ARGUMENTS - Define all build arguments upfront
# ============================================================================
ARG PYTHON_VERSION=3.11.11
ARG DEBIAN_VERSION=bookworm
ARG SQLITE_VERSION=3420000
ARG SQLITE_YEAR=2023
ARG SQLITE_SHA256=7abcfd161c6e2742ca5c6c0895d1f853c940f203304a0b49da4e1eca5d088ca6

# ============================================================================
# STAGE 1: SQLite Builder - Build custom SQLite with required extensions
# ============================================================================
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS sqlite-builder

ARG SQLITE_VERSION
ARG SQLITE_YEAR
ARG SQLITE_SHA256

# Install only necessary build tools
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download, verify, and build SQLite with checksum validation
WORKDIR /build
# hadolint ignore=DL3003
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN set -euxo pipefail && \
    # Download SQLite source (HTTPS for security) \
    wget -q https://www.sqlite.org/${SQLITE_YEAR}/sqlite-autoconf-${SQLITE_VERSION}.tar.gz && \
    # Verify checksum (CRITICAL SECURITY - prevents MITM attacks) \
    echo "${SQLITE_SHA256}  sqlite-autoconf-${SQLITE_VERSION}.tar.gz" | sha256sum -c - && \
    # Extract and build with optimizations \
    tar -xzf sqlite-autoconf-${SQLITE_VERSION}.tar.gz && \
    cd sqlite-autoconf-${SQLITE_VERSION} && \
    ./configure --prefix=/usr/local \
        --enable-fts5 \
        --enable-json1 \
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

# Set SQLite library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

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

# Upgrade pip, setuptools, wheel
# hadolint ignore=DL3042,DL3013
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

WORKDIR /build

# Copy only dependency files first for better caching
COPY pyproject.toml ./

# Install dependencies in specific order to handle binary compatibility
# CRITICAL: Remove --no-cache-dir to leverage pip cache mount for 60-90% faster builds
# hadolint ignore=DL3042,DL3013
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
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

# Pre-compile Python bytecode for faster startup
RUN python -m compileall -b $VIRTUAL_ENV

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

# Install only runtime dependencies
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy SQLite runtime libraries
COPY --from=sqlite-builder /usr/local/lib/libsqlite3.* /usr/local/lib/
COPY --from=sqlite-builder /usr/local/bin/sqlite3 /usr/local/bin/
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

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
