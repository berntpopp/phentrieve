#!/usr/bin/env bash
# Setup Docker volume permissions for Phentrieve
#
# This script ensures that Docker volume directories have correct permissions
# for the non-root phentrieve user (UID 10001).
#
# Platform behavior:
# - Linux: Requires manual permission setup (this script handles it)
# - macOS: Docker Desktop handles permissions automatically (no setup needed)
# - Windows: Docker Desktop handles permissions automatically (no setup needed)
#
# Usage:
#   sudo ./scripts/setup-docker-volumes.sh
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

# Check platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Platform: Linux"
    echo "→ Permission setup required"
    echo ""

    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}⚠️  Error: Not running as root${NC}"
        echo ""
        echo "This script must be run with sudo on Linux to change ownership."
        echo ""
        echo "Run: ${YELLOW}sudo $0${NC}"
        echo ""
        exit 1
    fi

    # Set ownership
    echo "Setting ownership to UID ${PHENTRIEVE_UID}:${PHENTRIEVE_GID}..."
    chown -R "${PHENTRIEVE_UID}:${PHENTRIEVE_GID}" "${DATA_DIR}/indexes"
    chown -R "${PHENTRIEVE_UID}:${PHENTRIEVE_GID}" "${HF_CACHE_DIR}"

    echo -e "${GREEN}✓ Permissions set successfully${NC}"
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
    echo "  - ${DATA_DIR}/indexes (UID:GID = ${PHENTRIEVE_UID}:${PHENTRIEVE_GID})"
    echo "  - ${HF_CACHE_DIR} (UID:GID = ${PHENTRIEVE_UID}:${PHENTRIEVE_GID})"
    echo ""
    echo "On Linux, run:"
    echo "  sudo chown -R ${PHENTRIEVE_UID}:${PHENTRIEVE_GID} ${DATA_DIR}/indexes"
    echo "  sudo chown -R ${PHENTRIEVE_UID}:${PHENTRIEVE_GID} ${HF_CACHE_DIR}"
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
