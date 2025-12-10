#!/bin/bash
# =============================================================================
# Docker Entrypoint Script for Phentrieve API
# =============================================================================
# This script handles runtime permission management for non-root containers
# following Docker best practices:
# - https://www.docker.com/blog/understanding-the-docker-user-instruction/
# - https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
#
# The script starts as root, fixes volume permissions, then drops to non-root
# user (phentrieve:10001) before running the application using gosu.
# =============================================================================

set -e

# Configuration
PHENTRIEVE_UID="${PHENTRIEVE_UID:-10001}"
PHENTRIEVE_GID="${PHENTRIEVE_GID:-10001}"
DATA_DIR="${PHENTRIEVE_DATA_ROOT_DIR:-/phentrieve_data_mount}"
CACHE_DIR="${HF_HOME:-/app/.cache/huggingface}"

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

    # If running as root, fix permissions and switch to phentrieve user
    if [ "$(id -u)" = "0" ]; then
        log_info "Running as root - fixing permissions before dropping privileges..."

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
