#!/usr/bin/env bash
# Setup Docker volume permissions for Phentrieve
#
# This script ensures that Docker volume directories have correct permissions
# for both local development and Docker container usage.
#
# Features:
# - Creates necessary directories for data, indexes, and HuggingFace cache
# - Sets permissions for dual-access (local user + Docker container UID 10001)
# - Supports multiple permission methods: sudo, Docker-based, or world-writable
#
# Platform behavior:
# - Linux: Sets world-writable (777) permissions for dual local/Docker access
# - macOS: Docker Desktop handles permissions automatically (no setup needed)
# - Windows: Docker Desktop handles permissions automatically (no setup needed)
#
# Usage:
#   ./scripts/setup-docker-volumes.sh          # Tries Docker-based or world-writable
#   sudo ./scripts/setup-docker-volumes.sh     # Uses sudo for stricter permissions
#
# Environment variables:
#   PHENTRIEVE_HOST_DATA_DIR    - Base data directory (default: ./data)
#   PHENTRIEVE_HOST_HF_CACHE_DIR - Hugging Face cache (default: $DATA_DIR/hf_cache)

set -euo pipefail

# Configuration
PHENTRIEVE_UID=10001
PHENTRIEVE_GID=10001
DATA_DIR="${PHENTRIEVE_HOST_DATA_DIR:-./data}"
HF_CACHE_DIR="${PHENTRIEVE_HOST_HF_CACHE_DIR:-${DATA_DIR}/hf_cache}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================"
echo "Phentrieve Docker Volume Setup"
echo "========================================"
echo ""

# Create directories if they don't exist
echo "Creating directories..."
mkdir -p "${DATA_DIR}/indexes"
mkdir -p "${HF_CACHE_DIR}"
echo "✓ Directories created"
echo ""

# Function to fix permissions using Docker container
fix_permissions_with_docker() {
    echo "Using Docker container to fix permissions..."
    if docker run --rm -v "$(pwd)/${DATA_DIR}:/data" alpine:3.20 sh -c 'chmod -R 777 /data/indexes /data/hf_cache 2>/dev/null || true' 2>/dev/null; then
        echo -e "${GREEN}✓ Permissions set via Docker${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Docker-based permission fix failed${NC}"
        return 1
    fi
}

# Function to fix permissions with sudo
fix_permissions_with_sudo() {
    echo "Setting ownership to UID ${PHENTRIEVE_UID}:${PHENTRIEVE_GID}..."
    chown -R "${PHENTRIEVE_UID}:${PHENTRIEVE_GID}" "${DATA_DIR}/indexes"
    chown -R "${PHENTRIEVE_UID}:${PHENTRIEVE_GID}" "${HF_CACHE_DIR}"
    # Also set world-writable for dual-access in development
    chmod -R 777 "${DATA_DIR}/indexes" "${HF_CACHE_DIR}"
    echo -e "${GREEN}✓ Permissions set successfully${NC}"
}

# Function to fix permissions with chmod only (no root needed)
fix_permissions_with_chmod() {
    echo "Setting world-writable permissions for dual-access..."
    if chmod -R 777 "${DATA_DIR}/indexes" "${HF_CACHE_DIR}" 2>/dev/null; then
        echo -e "${GREEN}✓ Permissions set with chmod${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  chmod failed (may need different ownership)${NC}"
        return 1
    fi
}

# Check platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Platform: Linux"
    echo "→ Permission setup for dual-access (local + Docker)"
    echo ""

    # Check if running as root or with sudo
    if [ "$EUID" -eq 0 ]; then
        echo -e "${CYAN}Running as root - using sudo method${NC}"
        fix_permissions_with_sudo
    else
        echo "Not running as root - trying alternative methods..."
        echo ""

        # Try chmod first (works if user owns the directories)
        if fix_permissions_with_chmod; then
            echo ""
        # Try Docker-based fix if chmod fails
        elif command -v docker &> /dev/null && fix_permissions_with_docker; then
            echo ""
        else
            echo ""
            echo -e "${RED}⚠️  Permission fix failed${NC}"
            echo ""
            echo "Please run with sudo:"
            echo "  ${YELLOW}sudo $0${NC}"
            echo ""
            echo "Or manually fix permissions:"
            echo "  sudo chmod -R 777 ${DATA_DIR}/indexes ${HF_CACHE_DIR}"
            echo ""
            exit 1
        fi
    fi

    echo ""
    echo "Directories configured:"
    echo "  - ${DATA_DIR}/indexes"
    echo "  - ${HF_CACHE_DIR}"

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Platform: macOS"
    echo "→ No permission setup needed"
    echo ""
    echo -e "${GREEN}✓ Docker Desktop handles permissions automatically${NC}"

elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "Platform: Windows"
    echo "→ No permission setup needed"
    echo ""
    echo -e "${GREEN}✓ Docker Desktop handles permissions automatically${NC}"

else
    echo -e "${YELLOW}Platform: Unknown ($OSTYPE)${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Warning: Unknown platform${NC}"
    echo ""
    echo "You may need to manually set permissions on:"
    echo "  - ${DATA_DIR}/indexes"
    echo "  - ${HF_CACHE_DIR}"
    echo ""
    echo "Try: chmod -R 777 ${DATA_DIR}/indexes ${HF_CACHE_DIR}"
fi

echo ""
echo "========================================"
echo "Next Steps"
echo "========================================"
echo ""
echo "1. Configure environment variables in .env file:"
echo "   export PHENTRIEVE_HOST_DATA_DIR=${DATA_DIR}"
echo "   export PHENTRIEVE_HOST_HF_CACHE_DIR=${HF_CACHE_DIR}"
echo ""
echo "2. Start Phentrieve with Docker Compose:"
echo "   docker-compose up"
echo ""
echo "3. Access the services:"
echo "   - API:      http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Frontend: http://localhost:8080"
echo ""
