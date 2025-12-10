#!/bin/bash
# =============================================================================
# Docker Entrypoint Script for Phentrieve API
# =============================================================================
# This script handles runtime permission management and data synchronization
# for non-root containers following Docker best practices:
# - https://www.docker.com/blog/understanding-the-docker-user-instruction/
# - https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
#
# The script starts as root, syncs embedded data if needed, fixes volume
# permissions, then drops to non-root user (phentrieve:10001) using gosu.
#
# Data Sync Feature (Issue #142):
# When deployed with volume mounts, the embedded data bundle may be newer than
# mounted data. This script detects version mismatches via manifest.json and
# automatically syncs embedded data to the mounted volume.
# =============================================================================

set -e

# Configuration
PHENTRIEVE_UID="${PHENTRIEVE_UID:-10001}"
PHENTRIEVE_GID="${PHENTRIEVE_GID:-10001}"
DATA_DIR="${PHENTRIEVE_DATA_ROOT_DIR:-/phentrieve_data_mount}"
CACHE_DIR="${HF_HOME:-/app/.cache/huggingface}"
EMBEDDED_DATA_DIR="/app/embedded_data"

# Data sync behavior configuration
# Set PHENTRIEVE_DATA_SYNC=false to disable automatic data sync
DATA_SYNC_ENABLED="${PHENTRIEVE_DATA_SYNC:-true}"
# Set PHENTRIEVE_DATA_FORCE_SYNC=true to always sync (useful for debugging)
DATA_FORCE_SYNC="${PHENTRIEVE_DATA_FORCE_SYNC:-false}"

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[entrypoint]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[entrypoint]${NC} $1"
}

log_error() {
    echo -e "${RED}[entrypoint]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[entrypoint]${NC} $1"
}

# Function to extract version from manifest.json
# Uses Python's json module for reliable parsing (no jq dependency)
get_manifest_version() {
    local manifest_path="$1"
    if [ -f "$manifest_path" ]; then
        python3 -c "
import json
import sys
try:
    with open('$manifest_path', 'r') as f:
        data = json.load(f)
    # Prefer hpo_version, fallback to phentrieve_version
    version = data.get('hpo_version', data.get('phentrieve_version', 'unknown'))
    print(version)
except Exception as e:
    print('unknown', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || echo "unknown"
    else
        echo "missing"
    fi
}

# Function to get full manifest info for logging
get_manifest_info() {
    local manifest_path="$1"
    if [ -f "$manifest_path" ]; then
        python3 -c "
import json
try:
    with open('$manifest_path', 'r') as f:
        data = json.load(f)
    hpo_ver = data.get('hpo_version', 'unknown')
    phentrieve_ver = data.get('phentrieve_version', 'unknown')
    created = data.get('created_at', 'unknown')
    print(f'HPO: {hpo_ver}, Phentrieve: {phentrieve_ver}, Created: {created}')
except:
    print('Unable to parse manifest')
" 2>/dev/null || echo "Unable to read manifest"
    else
        echo "Manifest not found"
    fi
}

# Function to sync embedded data to mounted volume
sync_embedded_data() {
    log_info "Syncing embedded data to mounted volume..."

    # Ensure embedded data exists
    if [ ! -d "$EMBEDDED_DATA_DIR" ] || [ ! -f "$EMBEDDED_DATA_DIR/manifest.json" ]; then
        log_warn "No embedded data found at $EMBEDDED_DATA_DIR - skipping sync"
        return 1
    fi

    # Create target directory if needed
    mkdir -p "$DATA_DIR"

    # Copy all embedded data to mounted volume
    # Using cp -a to preserve permissions and timestamps
    # Using rsync-like approach with cp for atomic updates
    log_info "Copying data files..."

    # Copy manifest first
    cp -a "$EMBEDDED_DATA_DIR/manifest.json" "$DATA_DIR/manifest.json"

    # Copy HPO database if exists
    if [ -f "$EMBEDDED_DATA_DIR/hpo_data.db" ]; then
        log_info "  - Copying hpo_data.db..."
        cp -a "$EMBEDDED_DATA_DIR/hpo_data.db" "$DATA_DIR/hpo_data.db"
    fi

    # Copy hp.json if exists
    if [ -f "$EMBEDDED_DATA_DIR/hp.json" ]; then
        log_info "  - Copying hp.json..."
        cp -a "$EMBEDDED_DATA_DIR/hp.json" "$DATA_DIR/hp.json"
    fi

    # Copy indexes directory if exists
    if [ -d "$EMBEDDED_DATA_DIR/indexes" ]; then
        log_info "  - Copying indexes/..."
        rm -rf "$DATA_DIR/indexes" 2>/dev/null || true
        cp -a "$EMBEDDED_DATA_DIR/indexes" "$DATA_DIR/indexes"
    fi

    log_info "Data sync completed successfully!"
    return 0
}

# Function to check if data sync is needed
check_data_sync() {
    # Skip if sync is disabled
    if [ "$DATA_SYNC_ENABLED" != "true" ]; then
        log_info "Data sync disabled (PHENTRIEVE_DATA_SYNC=false)"
        return 1
    fi

    # Force sync if requested
    if [ "$DATA_FORCE_SYNC" = "true" ]; then
        log_info "Force sync enabled (PHENTRIEVE_DATA_FORCE_SYNC=true)"
        return 0
    fi

    # Check for embedded data
    local embedded_manifest="$EMBEDDED_DATA_DIR/manifest.json"
    local mounted_manifest="$DATA_DIR/manifest.json"

    if [ ! -f "$embedded_manifest" ]; then
        log_debug "No embedded data manifest found - sync not needed"
        return 1
    fi

    # Get versions
    local embedded_version
    local mounted_version
    embedded_version=$(get_manifest_version "$embedded_manifest")
    mounted_version=$(get_manifest_version "$mounted_manifest")

    log_info "Checking data versions:"
    log_info "  Embedded: $(get_manifest_info "$embedded_manifest")"
    log_info "  Mounted:  $(get_manifest_info "$mounted_manifest")"

    # Sync needed if:
    # 1. Mounted data is missing (no manifest)
    # 2. Mounted data version is different from embedded
    # 3. Mounted data has no HPO database

    if [ "$mounted_version" = "missing" ]; then
        log_info "Mounted data not found - sync needed"
        return 0
    fi

    if [ ! -f "$DATA_DIR/hpo_data.db" ]; then
        log_info "HPO database missing in mounted volume - sync needed"
        return 0
    fi

    if [ "$embedded_version" != "$mounted_version" ]; then
        log_info "Version mismatch detected (embedded: $embedded_version, mounted: $mounted_version) - sync needed"
        return 0
    fi

    log_info "Data versions match - no sync needed"
    return 1
}

# Function to fix permissions on a directory
fix_permissions() {
    local dir="$1"
    local desc="$2"

    if [ -d "$dir" ]; then
        # Check if directory is writable by the phentrieve user
        if ! gosu phentrieve test -w "$dir" 2>/dev/null; then
            log_info "Fixing permissions on $desc ($dir)..."
            chown -R "${PHENTRIEVE_UID}:${PHENTRIEVE_GID}" "$dir" 2>/dev/null || \
                log_warn "Could not chown $dir - may need host-side permission fix"
            chmod -R u+rwX "$dir" 2>/dev/null || true
        else
            log_info "$desc permissions OK"
        fi
    else
        log_info "Creating $desc directory..."
        mkdir -p "$dir"
        chown "${PHENTRIEVE_UID}:${PHENTRIEVE_GID}" "$dir"
        chmod 755 "$dir"
    fi
}

# Function to verify required data files exist
verify_data() {
    local hpo_db="$DATA_DIR/hpo_data.db"
    local indexes="$DATA_DIR/indexes"

    if [ ! -f "$hpo_db" ]; then
        log_error "HPO database not found: $hpo_db"
        log_error "Please ensure HPO data is available via:"
        log_error "  - Pre-built bundle (embedded in image)"
        log_error "  - Volume mount with prepared data"
        log_error "  - Running 'phentrieve data prepare' first"
        return 1
    fi

    if [ ! -d "$indexes" ]; then
        log_warn "Index directory not found: $indexes"
        log_warn "Vector search may not work without indexes"
    fi

    log_info "HPO database found: $hpo_db"
    return 0
}

# Main entrypoint logic
main() {
    log_info "Starting Phentrieve API entrypoint..."
    log_info "Running as: $(id)"

    # If running as root, handle data sync, fix permissions, and switch to phentrieve user
    if [ "$(id -u)" = "0" ]; then
        log_info "Running as root - handling data sync and permissions..."

        # Check if data sync is needed and perform sync
        if check_data_sync; then
            if ! sync_embedded_data; then
                log_warn "Data sync failed - continuing with existing data"
            fi
        fi

        # Fix permissions on data directory
        fix_permissions "$DATA_DIR" "data directory"

        # Fix permissions on index directory if it exists
        if [ -d "$DATA_DIR/indexes" ]; then
            fix_permissions "$DATA_DIR/indexes" "indexes directory"
        fi

        # Fix permissions on cache directory
        fix_permissions "$CACHE_DIR" "HuggingFace cache"

        # Verify required data exists
        if ! verify_data; then
            log_error "Data verification failed - exiting"
            exit 1
        fi

        log_info "Dropping to phentrieve user (UID ${PHENTRIEVE_UID})..."

        # Execute the command as the phentrieve user using gosu
        # Use exec to replace this process
        exec gosu phentrieve "$@"
    else
        log_info "Already running as non-root user"

        # Verify data exists
        if ! verify_data; then
            log_error "Data verification failed - exiting"
            exit 1
        fi

        # Execute the command directly
        exec "$@"
    fi
}

# Run main with all arguments
main "$@"
