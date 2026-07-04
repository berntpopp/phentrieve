# Docker Build Timeout Fix Plan

## Status: Active
**Created**: 2025-11-24
**Issue**: API Docker image build exceeding 30-minute timeout in GitHub Actions

---

## Problem Analysis

### Symptoms
- `Build and Push API Image` job exceeds 30-minute timeout
- Error: `The job has exceeded the maximum execution time of 30m0s`
- Frontend build completes in ~2 minutes; API build times out
- Multiple consecutive builds are being cancelled due to concurrency

### Root Causes

1. **Heavy ML Dependencies**: The API Dockerfile installs:
   - PyTorch (~2GB+)
   - sentence-transformers
   - NumPy 2.x
   - spaCy with language models
   - ChromaDB with all dependencies

2. **Cache Inefficiency**: GitHub Actions cache (`type=gha`) has known issues:
   - Cache export can timeout after 600 seconds
   - [Issue #545](https://github.com/docker/build-push-action/issues/545): Cache export takes ~300s+
   - [Issue #1253](https://github.com/docker/build-push-action/issues/1253): Docker build fails with gha cache
   - API v2 migration deadline: April 15, 2025

3. **Build from Scratch**: Without effective caching, every build:
   - Compiles SQLite from source (~2-3 min)
   - Downloads and installs PyTorch (~5-10 min on slow connections)
   - Installs all Python dependencies (~10-15 min)
   - Total: 20-30+ minutes

4. **Hadolint Warnings** (non-blocking but indicate issues):
   - `DL3013`: Pin versions in pip (line 98)
   - `DL3003`: Use WORKDIR to switch directories (line 36)

---

## Research: 2025 Best Practices

### Sources Consulted
- [Docker Docs: GitHub Actions Cache](https://docs.docker.com/build/cache/backends/gha/)
- [Docker Docs: Cache Management](https://docs.docker.com/build/ci/github-actions/cache/)
- [HyperDX: Docker Buildx Caching](https://www.hyperdx.io/blog/docker-buildx-cache-with-github-actions)
- [Blacksmith: Cache is King](https://www.blacksmith.sh/blog/cache-is-king-a-guide-for-docker-layer-caching-in-github-actions)
- [DEPT: Speed up Docker Builds](https://engineering.deptagency.com/how-to-speed-up-docker-builds-in-github-actions)

### Key Recommendations

#### 1. Switch to Registry Cache (Recommended)
Registry cache is more reliable than `type=gha` for large images:

```yaml
cache-from: type=registry,ref=ghcr.io/${{ env.IMAGE_NAME_API }}:buildcache
cache-to: type=registry,ref=ghcr.io/${{ env.IMAGE_NAME_API }}:buildcache,mode=max
```

**Benefits**:
- No 10GB cache limit (uses container registry)
- `mode=max` caches all intermediate layers
- More reliable for large ML dependencies
- No export timeout issues

#### 2. Use `ghtoken` Parameter for GHA Cache
If keeping `type=gha`, add GitHub token for API v2 compatibility:

```yaml
- name: Build and push API image
  uses: docker/build-push-action@v6
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
  env:
    BUILDX_GHA_CACHE_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

#### 3. Increase Timeout
30 minutes is insufficient for initial builds. Recommended: 60 minutes.

```yaml
timeout-minutes: 60  # Increase from 30
```

#### 4. Optimize Dockerfile Layer Ordering
Current order causes cache invalidation too often. Reorder for better caching:

```dockerfile
# GOOD: Copy only pyproject.toml first, install deps, then copy code
COPY pyproject.toml ./
RUN pip install .
COPY . .

# BAD: Copying all source invalidates dependency cache
COPY . .
RUN pip install .
```

#### 5. Use Pre-built Base Image with Dependencies
Create a base image with heavy dependencies that rarely change:

```dockerfile
# base.Dockerfile - Build monthly or on dependency changes
FROM python:3.11-slim
RUN pip install torch sentence-transformers chromadb

# api.Dockerfile - Build on every push
FROM ghcr.io/berntpopp/phentrieve/api-base:latest
COPY . .
```

#### 6. Pin pip Versions (Fix Hadolint DL3013)
```dockerfile
RUN pip install \
    "numpy==2.1.3" \
    "spacy==3.8.0" \
    # etc.
```

---

## Recommended Fix Strategy

### Phase 1: Immediate Fixes (Quick Wins)

1. **Increase timeout to 60 minutes**
2. **Switch to registry cache** for reliability
3. **Add concurrency control** to prevent multiple builds

### Phase 2: Optimization

1. **Create base image** with ML dependencies
2. **Pin all pip versions** for reproducibility
3. **Optimize Dockerfile** layer ordering

### Phase 3: Long-term

1. **Consider self-hosted runners** for faster builds
2. **Implement build matrix** (build deps separately)
3. **Add build skip logic** when only docs/frontend change

---

## Implementation: Recommended Changes

### docker-publish.yml Changes

```yaml
# Add concurrency to prevent multiple long builds
concurrency:
  group: docker-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build-and-push-api:
    timeout-minutes: 60  # Increased from 30
    steps:
      - name: Build and push API image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./api/Dockerfile
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          # Switch to registry cache for reliability with large images
          cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME_API }}:buildcache
          cache-to: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME_API }}:buildcache,mode=max
          platforms: linux/amd64
```

### api/Dockerfile Changes

```dockerfile
# Line 36: Fix DL3003 - Already using WORKDIR, but RUN cd is flagged
# Change from:
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN set -euxo pipefail && \
    cd sqlite-autoconf-${SQLITE_VERSION} && \
    ...

# To:
WORKDIR /build/sqlite-autoconf-${SQLITE_VERSION}
RUN ./configure ... && make ...
WORKDIR /build

# Line 98: Fix DL3013 - Pin versions (already partially done)
# Consider pinning all versions explicitly
```

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Registry cache | Low | Falls back to full build if cache miss |
| Timeout increase | None | Just allows more time |
| Concurrency control | Low | May delay some builds |
| Dockerfile changes | Medium | Test locally first |

---

## Testing Plan

1. Create feature branch with changes
2. Test locally with `docker build --progress=plain`
3. Push and monitor GitHub Actions
4. Verify cache hits on subsequent builds
5. Confirm build time < 20 minutes with warm cache

---

## Success Metrics

- [ ] API Docker build completes within timeout
- [ ] Build time < 10 minutes with warm cache
- [ ] No Hadolint errors (warnings acceptable)
- [ ] Registry cache shows cache hits in logs

---

## References

- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Docker Buildx Cache Documentation](https://docs.docker.com/build/cache/)
- [GitHub Actions Cache Limits](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [Build Cache Issue #545](https://github.com/docker/build-push-action/issues/545)
- [Cache Export Timeout #1253](https://github.com/docker/build-push-action/issues/1253)
