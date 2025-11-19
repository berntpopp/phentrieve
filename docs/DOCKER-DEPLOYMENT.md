# Docker Deployment Guide

This guide covers deploying Phentrieve using Docker and Docker Compose for production environments.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Volume Permissions (Linux Only)](#volume-permissions-linux-only)
- [Configuration](#configuration)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

---

## Overview

Phentrieve provides production-ready Docker images with comprehensive security hardening:

- ✅ **Non-root execution** - API runs as UID 10001, frontend as UID 101
- ✅ **Resource limits** - CPU and memory constraints prevent resource exhaustion
- ✅ **Read-only filesystems** - Containers have immutable root filesystems
- ✅ **Dropped capabilities** - Minimal Linux capabilities (CAP_DROP: ALL)
- ✅ **Security options** - no-new-privileges, seccomp profiles
- ✅ **Health checks** - Built-in liveness and readiness probes

**Docker Images**:
- `ghcr.io/berntpopp/phentrieve/api:latest` - FastAPI backend
- `ghcr.io/berntpopp/phentrieve/frontend:latest` - Vue.js frontend (nginx)

---

## Prerequisites

**Required**:
- Docker 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose 2.0+ ([Install Compose](https://docs.docker.com/compose/install/))

**Recommended**:
- 8GB+ RAM (for embedding models)
- 10GB+ disk space (for HPO data and indexes)
- Linux, macOS, or Windows with WSL2

**Network Ports**:
- `8000` - API server (internal, proxied)
- `8080` - Frontend (nginx, proxied)

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/berntpopp/phentrieve.git
cd phentrieve
```

### 2. Setup Volume Permissions (Linux Only)

**Linux users must run this** to configure permissions for non-root containers:

```bash
sudo ./scripts/setup-docker-volumes.sh
```

**macOS/Windows users**: Skip this step (Docker Desktop handles permissions automatically).

### 3. Configure Environment

Create `.env` file with required variables:

```bash
# Copy example config
cp .env.example .env

# Edit with your values
nano .env
```

**Minimum required variables**:
```bash
# Data directories
PHENTRIEVE_HOST_DATA_DIR=/path/to/data
PHENTRIEVE_HOST_HF_CACHE_DIR=/path/to/data/hf_cache

# API configuration
VITE_API_URL_PUBLIC=http://your-domain.com/api

# Network (for Nginx Proxy Manager integration)
NPM_SHARED_NETWORK_NAME=npm_proxy_network
```

### 4. Prepare HPO Data

Download and prepare HPO ontology data:

```bash
# Option 1: Using local Python installation
phentrieve data prepare
phentrieve index build

# Option 2: Using Docker
docker-compose run --rm phentrieve_api phentrieve data prepare
docker-compose run --rm phentrieve_api phentrieve index build
```

### 5. Start Services

```bash
# Pull pre-built images (recommended)
docker-compose pull

# Start in background
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 6. Verify Deployment

```bash
# Check container health
docker-compose ps

# Test API
curl http://localhost:8000/api/v1/health

# Test frontend
curl http://localhost:8080/health
```

---

## Volume Permissions (Linux Only)

### Why Permission Setup is Required

Phentrieve containers run as **non-root users** for security:
- **API container**: `phentrieve` user (UID 10001, GID 10001)
- **Frontend container**: `nginx` user (UID 101, GID 101)

On **Linux hosts**, volume mounts require explicit permission setup because Docker strictly enforces UID/GID matching. On **macOS and Windows**, Docker Desktop handles this automatically.

### Platform-Specific Behavior

| Platform | Permission Setup Required? | Notes |
|----------|----------------------------|-------|
| **Linux** | ✅ **Yes** | Run `sudo ./scripts/setup-docker-volumes.sh` |
| **macOS** | ❌ No | Docker Desktop handles it automatically |
| **Windows (WSL2)** | ❌ No | Docker Desktop handles it automatically |

### Automated Setup (Recommended)

```bash
# Linux only - run with sudo
sudo ./scripts/setup-docker-volumes.sh
```

**What it does**:
1. Creates required directories (`indexes/`, `hf_cache/`)
2. Sets ownership to UID 10001:10001
3. Validates permissions
4. Provides next steps

### Manual Setup

If you prefer manual setup or the script doesn't work:

```bash
# Create directories
mkdir -p /path/to/data/indexes
mkdir -p /path/to/data/hf_cache

# Set ownership (Linux only)
sudo chown -R 10001:10001 /path/to/data/indexes
sudo chown -R 10001:10001 /path/to/data/hf_cache

# Verify permissions
ls -la /path/to/data/
# Should show: drwxr-xr-x 10001 10001 indexes/
#              drwxr-xr-x 10001 10001 hf_cache/
```

### Required Writable Volumes

The following volumes **must be writable** by UID 10001:

```yaml
# docker-compose.yml
volumes:
  # ChromaDB vector indexes (read-write)
  - ${PHENTRIEVE_HOST_DATA_DIR}/indexes:/phentrieve_data_mount/indexes:rw

  # Hugging Face model cache (read-write)
  - ${PHENTRIEVE_HOST_HF_CACHE_DIR}:/app/.cache/huggingface:rw
```

---

## Configuration

### Environment Variables

**Data Directories**:
```bash
PHENTRIEVE_HOST_DATA_DIR=/path/to/data           # Base data directory
PHENTRIEVE_HOST_HF_CACHE_DIR=/path/to/hf_cache   # Hugging Face cache (optional, defaults to $DATA_DIR/hf_cache)
```

**API Configuration**:
```bash
LOG_LEVEL_API=INFO                                # Logging level (DEBUG, INFO, WARN, ERROR)
VITE_API_URL_PUBLIC=http://your-domain.com/api    # Public API URL for frontend
```

**Network Configuration**:
```bash
NPM_SHARED_NETWORK_NAME=npm_proxy_network         # External network for reverse proxy
```

### Resource Limits

Default resource limits (defined in `docker-compose.yml`):

**API Container**:
```yaml
resources:
  limits:
    cpus: '4.0'       # Maximum 4 CPU cores
    memory: 8G        # Maximum 8GB RAM
  reservations:
    cpus: '1.0'       # Minimum 1 CPU core
    memory: 4G        # Minimum 4GB RAM
```

**Frontend Container**:
```yaml
resources:
  limits:
    cpus: '0.5'       # Maximum 0.5 CPU cores
    memory: 256M      # Maximum 256MB RAM
  reservations:
    cpus: '0.1'       # Minimum 0.1 CPU cores
    memory: 64M       # Minimum 64MB RAM
```

**Customizing Limits**:

Create `docker-compose.override.yml`:

```yaml
services:
  phentrieve_api:
    deploy:
      resources:
        limits:
          cpus: '8.0'    # Increase for better performance
          memory: 16G
```

### Health Checks

**API Health Check**:
```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"]
  interval: 30s       # Check every 30 seconds
  timeout: 10s        # 10 second timeout
  retries: 5          # 5 retries before marking unhealthy
  start_period: 180s  # 3 minute grace period for model loading
```

**Frontend Health Check**:
```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 10s
```

---

## Production Deployment

### Reverse Proxy Integration (Nginx Proxy Manager)

Phentrieve is designed to work with **Nginx Proxy Manager** (NPM) for SSL termination and routing.

**1. Create external network (if not exists)**:
```bash
docker network create npm_proxy_network
```

**2. Configure NPM proxy hosts**:

**API Proxy Host**:
- Domain: `api.your-domain.com`
- Forward Hostname: `phentrieve_api`
- Forward Port: `8000`
- Websockets: ✅ Enabled
- SSL: ✅ Request SSL certificate

**Frontend Proxy Host**:
- Domain: `your-domain.com`
- Forward Hostname: `phentrieve_frontend`
- Forward Port: `8080`
- SSL: ✅ Request SSL certificate

**3. Update environment**:
```bash
# .env
VITE_API_URL_PUBLIC=https://api.your-domain.com/api
NPM_SHARED_NETWORK_NAME=npm_proxy_network
```

### Building Custom Images

**Development build** (local changes):
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
```

**Production build** (from source):
```bash
docker-compose build --no-cache
```

**Multi-platform build** (for deployment to different architectures):
```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/berntpopp/phentrieve/api:custom \
  -f api/Dockerfile \
  --push .
```

### Data Backup

**Recommended backup strategy**:

```bash
#!/bin/bash
# backup-phentrieve.sh

BACKUP_DIR="/backups/phentrieve/$(date +%Y%m%d)"
DATA_DIR="${PHENTRIEVE_HOST_DATA_DIR}"

mkdir -p "$BACKUP_DIR"

# Backup HPO database (critical)
cp "${DATA_DIR}/hpo_data.db" "${BACKUP_DIR}/"

# Backup vector indexes (large, but reproducible)
tar -czf "${BACKUP_DIR}/indexes.tar.gz" "${DATA_DIR}/indexes/"

# Backup configuration
cp .env "${BACKUP_DIR}/"

echo "Backup complete: ${BACKUP_DIR}"
```

**Restoration**:
```bash
# Stop containers
docker-compose down

# Restore data
cp /backups/phentrieve/20250119/hpo_data.db /path/to/data/
tar -xzf /backups/phentrieve/20250119/indexes.tar.gz -C /path/to/data/

# Fix permissions (Linux)
sudo chown -R 10001:10001 /path/to/data/indexes

# Restart
docker-compose up -d
```

### Monitoring

**Container health**:
```bash
# Check status
docker-compose ps

# Health check logs
docker inspect phentrieve_api | jq '.[0].State.Health'
```

**Resource usage**:
```bash
# Live stats
docker stats phentrieve_api phentrieve_frontend

# Memory usage
docker exec phentrieve_api ps aux --sort=-%mem | head
```

**Logs**:
```bash
# Follow all logs
docker-compose logs -f

# API only (last 100 lines)
docker-compose logs --tail=100 -f phentrieve_api

# Export logs
docker-compose logs --no-color > phentrieve-logs-$(date +%Y%m%d).txt
```

---

## Troubleshooting

### Permission Denied Errors

**Error**:
```
Error: [Errno 13] Permission denied: '/phentrieve_data_mount/indexes/...'
```

**Solution (Linux)**:
```bash
# Run the setup script
sudo ./scripts/setup-docker-volumes.sh

# Or manually fix permissions
sudo chown -R 10001:10001 /path/to/data/indexes
sudo chown -R 10001:10001 /path/to/data/hf_cache
```

**Solution (macOS/Windows)**:
This error should not occur on macOS or Windows. If it does:
1. Verify Docker Desktop is running
2. Check volume mounts in `docker-compose.yml`
3. Restart Docker Desktop

### Container Fails to Start

**Check logs**:
```bash
docker-compose logs phentrieve_api
```

**Common issues**:

1. **Out of memory** during model loading:
   ```bash
   # Increase memory limit in docker-compose.override.yml
   resources:
     limits:
       memory: 16G
   ```

2. **Missing HPO data**:
   ```bash
   # Prepare data
   docker-compose run --rm phentrieve_api phentrieve data prepare
   ```

3. **Port conflict**:
   ```bash
   # Check if ports in use
   sudo lsof -i :8000
   sudo lsof -i :8080

   # Change ports in docker-compose.yml
   ports:
     - "8001:8000"  # API
     - "8081:8080"  # Frontend
   ```

### Slow Model Loading

**Symptoms**: Container takes >5 minutes to become healthy

**Solutions**:

1. **Increase health check start period**:
   ```yaml
   healthcheck:
     start_period: 300s  # 5 minutes
   ```

2. **Pre-download models**:
   ```bash
   # Download models before starting
   docker-compose run --rm phentrieve_api python -c "
   from sentence_transformers import SentenceTransformer
   SentenceTransformer('FremyCompany/BioLORD-2023-M')
   "
   ```

3. **Use persistent cache**:
   Ensure `PHENTRIEVE_HOST_HF_CACHE_DIR` is set and persistent.

### Network Issues

**Error**: Frontend can't reach API

**Check**:
```bash
# Verify containers are on same network
docker network inspect phentrieve_internal_net

# Test API from frontend container
docker exec phentrieve_frontend wget -O- http://phentrieve_api:8000/api/v1/health
```

**Solution**:
Ensure both containers are on `phentrieve_internal_net` network.

---

## Security Considerations

### Non-Root Execution

**Why it matters**:
- Prevents container escape vulnerabilities
- Limits damage if container is compromised
- Follows principle of least privilege
- Required for Kubernetes security policies

**Implementation**:
```yaml
services:
  phentrieve_api:
    user: "10001:10001"  # phentrieve:phentrieve
```

### Read-Only Root Filesystem

**Why it matters**:
- Prevents malicious code from modifying container
- Makes container immutable
- Detects unauthorized file changes

**Writable directories** (via tmpfs):
```yaml
tmpfs:
  - /tmp:uid=10001,gid=10001,mode=1777,size=1G
  - /app/.cache:uid=10001,gid=10001,mode=0755,size=2G
```

### Resource Limits

**Why it matters**:
- Prevents resource exhaustion attacks
- Ensures fair resource allocation
- Protects host system

**Monitoring**:
```bash
# Check if limits are being hit
docker stats phentrieve_api

# Look for container restarts (OOM)
docker inspect phentrieve_api | jq '.[0].State.Restarted'
```

### Secrets Management

**Never commit secrets to git**:

```bash
# Use .env for local development
echo ".env" >> .gitignore

# Use Docker secrets for production
docker secret create phentrieve_db_password ./db_password.txt
```

**In docker-compose.yml**:
```yaml
services:
  phentrieve_api:
    secrets:
      - db_password
secrets:
  db_password:
    external: true
```

### SSL/TLS

**Always use HTTPS in production**:
- Terminate SSL at reverse proxy (Nginx Proxy Manager)
- Use Let's Encrypt for free certificates
- Enable HTTP to HTTPS redirect
- Set HSTS headers

---

## Additional Resources

- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Nginx Proxy Manager Documentation](https://nginxproxymanager.com/)
- [Phentrieve API Documentation](http://localhost:8000/docs)
- [Phentrieve GitHub Repository](https://github.com/berntpopp/phentrieve)

---

## Getting Help

**Issues**:
- [GitHub Issues](https://github.com/berntpopp/phentrieve/issues)

**Logs**:
When reporting issues, include:
```bash
# System info
docker version
docker-compose version
uname -a

# Container logs
docker-compose logs --tail=100

# Container health
docker-compose ps
```

**Community**:
- Check existing GitHub issues for solutions
- Include detailed error messages and logs in new issues
