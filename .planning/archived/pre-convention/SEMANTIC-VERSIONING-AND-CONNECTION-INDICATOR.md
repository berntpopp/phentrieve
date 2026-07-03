# Semantic Versioning & Connection Indicator Implementation Plan

**Status**: 🟡 Active
**Priority**: High
**Created**: 2025-11-21
**Estimated Effort**: Medium (2-3 days)

## Executive Summary

Implement proper semantic versioning (SemVer) for Phentrieve's three independent components (CLI/Library, API, Frontend) starting with alpha versions (0.x.x), and add a connection status indicator in the frontend to show API health and connection time.

**Goals**:
- ✅ Independent version management for CLI, API, and Frontend
- ✅ Start with alpha versions (0.x.x) for initial development
- ✅ Build-time version injection for frontend (zero runtime cost)
- ✅ Runtime version reading for backend (always current)
- ✅ Public `/version` API endpoint for version aggregation
- ✅ Connection status indicator in frontend (similar to kidney-genetics-db)
- ✅ Follow DRY, KISS, SOLID principles - no antipatterns

---

## Table of Contents

1. [Semantic Versioning Strategy](#1-semantic-versioning-strategy)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Backend Version Management](#phase-1-backend-version-management)
4. [Phase 2: Frontend Version Management](#phase-2-frontend-version-management)
5. [Phase 3: CLI Version Management](#phase-3-cli-version-management)
6. [Phase 4: Connection Status Indicator](#phase-4-connection-status-indicator)
7. [Phase 5: UI Components](#phase-5-ui-components)
8. [Testing Strategy](#testing-strategy)
9. [Best Practices & Principles](#best-practices--principles)
10. [References](#references)

---

## 1. Semantic Versioning Strategy

### 1.1 Version Format

Following [Semantic Versioning 2.0.0](https://semver.org/):

```
0.MINOR.PATCH[-PRERELEASE][+BUILD]
```

**Components**:
- **MAJOR** (0): Stays at 0 during alpha development (unstable API)
- **MINOR**: Incremented for new features (backwards-compatible)
- **PATCH**: Incremented for bug fixes (backwards-compatible)
- **PRERELEASE** (optional): `-alpha.N`, `-beta.N`, `-rc.N`
- **BUILD** (optional): `+20251121.abc123` (date + git hash)

**Examples**:
```
0.1.0        - First minor release (alpha stage)
0.1.1        - Bug fix
0.2.0        - New feature
0.2.0-alpha.1 - Pre-release for feature branch
1.0.0        - First stable release (exit alpha)
```

### 1.2 Alpha Development Philosophy

**Per SemVer 2.0.0 Spec**:
> "Major version zero (0.y.z) is for initial development. Anything MAY change at any time. The public API SHOULD NOT be considered stable."

**Our Strategy**:
- Start all components at `0.1.0` (initial release)
- Stay in `0.x.x` range until API is stable
- Breaking changes increment MINOR version (0.1.0 → 0.2.0)
- Features/fixes increment PATCH version (0.1.0 → 0.1.1)
- Move to `1.0.0` when ready for production (stable API contract)

**Configuration** (Python Semantic Release):
```toml
[tool.semantic_release]
allow_zero_version = true      # Enable 0.x.x versions
major_on_zero = false          # Breaking changes bump MINOR during 0.x.x
```

### 1.3 Independent Component Versioning

Each component evolves independently:

| Component | Purpose | Current Version | Next Release |
|-----------|---------|----------------|--------------|
| **CLI/Library** | Python package (`phentrieve`) | 0.0.0 → **0.1.0** | Feature: HPO extraction |
| **API** | FastAPI backend service | 0.0.0 → **0.1.0** | Initial API endpoints |
| **Frontend** | Vue.js web interface | 0.0.0 → **0.1.0** | Basic UI features |

**Rationale**:
- CLI changes don't require API version bump (and vice versa)
- Frontend can iterate UI independently of backend
- Clearer communication of what changed where
- Follows microservices/modular architecture principles

### 1.4 Version Synchronization Points

While components version independently, **coordination is needed for**:

1. **Breaking API Changes** → Both API and Frontend must coordinate
2. **New CLI Features** → May require API support → Coordinate versions
3. **Major Releases** → All components move to same MAJOR version (0.x.x → 1.0.0)

**Communication via `/version` endpoint**:
```json
{
  "cli": { "version": "0.2.1", "name": "phentrieve", "type": "Python CLI" },
  "api": { "version": "0.3.0", "name": "phentrieve-api", "type": "FastAPI" },
  "frontend": { "version": "0.1.5", "name": "phentrieve-frontend", "type": "Vue.js" },
  "environment": "development",
  "timestamp": "2025-11-21T10:30:00Z"
}
```

---

## 2. Architecture Overview

### 2.1 Version Flow Diagram

```mermaid
graph TB
    subgraph "Build Time"
        PY[pyproject.toml<br/>version: 0.1.0]
        PKG[package.json<br/>version: 0.1.0]

        PY -->|Read at runtime| CLI[CLI __init__.py<br/>__version__]
        PY -->|Read at runtime| API[API version.py<br/>get_api_version()]
        PKG -->|Vite inject| FE[Frontend<br/>__APP_VERSION__]
    end

    subgraph "Runtime"
        CLI --> CLIVER[CLI --version]
        API --> APIVER[GET /version]
        FE --> FEVER[version.js utils]

        APIVER --> AGG[Aggregated Versions]
        FEVER --> AGG

        AGG --> UI[AppFooter.vue<br/>Version Display]
    end

    subgraph "Connection Monitoring"
        HEALTH[GET /health]
        POLL[Periodic Health Check<br/>Every 30s]

        HEALTH --> POLL
        POLL --> STATUS[Connection Status]
        STATUS --> INDIC[Status Indicator<br/>in AppFooter]
    end
```

### 2.2 Component Responsibilities

| Component | Responsibility | Source of Truth |
|-----------|---------------|-----------------|
| **pyproject.toml** | CLI & API version source | Single source for Python |
| **package.json** | Frontend version source | Single source for Node.js |
| **CLI `__init__.py`** | Expose CLI version | Reads from pyproject.toml |
| **API `version.py`** | Version utilities | Reads from pyproject.toml |
| **API `/version` endpoint** | Aggregate all versions | Combines CLI, API, Frontend |
| **Frontend `version.js`** | Version utilities | Build-time injection + API fetch |
| **`AppFooter.vue`** | UI display | Consumes version.js |
| **GET `/health`** | API health check | Always returns 200 OK when alive |
| **`api-health.js`** | Connection monitoring | Polls /health every 30s |

### 2.3 Design Principles Applied

**DRY (Don't Repeat Yourself)**:
- ✅ Single source of truth: `pyproject.toml` for Python, `package.json` for Node
- ✅ Reuse version utilities across CLI/API (shared `version.py`)
- ✅ Shared components (AppFooter) for version display

**KISS (Keep It Simple, Stupid)**:
- ✅ No complex version calculation - read from files
- ✅ Simple polling for health checks (not WebSocket for REST API)
- ✅ Minimal dependencies (use built-in `tomllib` for Python 3.11+)

**SOLID Principles**:
- ✅ **S**ingle Responsibility: Each module handles one concern
- ✅ **O**pen/Closed: Version utilities extensible without modification
- ✅ **L**iskov Substitution: Version readers return consistent format
- ✅ **I**nterface Segregation: Small, focused interfaces
- ✅ **D**ependency Inversion: Depend on abstractions (version interface)

**No Antipatterns**:
- ❌ No hardcoded versions in multiple places
- ❌ No polling when not needed (use reactive refs, not setInterval in components)
- ❌ No tight coupling between components
- ❌ No premature optimization (profile first)
- ❌ No magic numbers/strings (use constants)

---

## Phase 1: Backend Version Management

### 1.1 Update pyproject.toml

**File**: `pyproject.toml`

**Changes**:
```toml
[project]
name = "phentrieve"
version = "0.1.0"  # Update from 0.0.0
# ... rest of project config

[tool.semantic_release]
# Enable alpha versioning (0.x.x)
allow_zero_version = true
major_on_zero = false  # Breaking changes bump MINOR during 0.x

# Version location (for semantic-release to update)
version_toml = ["pyproject.toml:project.version"]

# Version variables to update across codebase
version_variables = [
    "phentrieve/__init__.py:__version__",
]

# Commit message parsing (Conventional Commits)
commit_parser = "angular"
```

**Rationale**:
- Single source of truth for Python packages
- `semantic_release` can automatically bump versions
- Clear alpha development configuration

---

### 1.2 Create API Version Utility Module

**File**: `api/version.py` (new file)

```python
"""
Version management utilities for Phentrieve API.

Provides functions to read version information from pyproject.toml
and aggregate versions from all components.

Following DRY, KISS, SOLID principles:
- Single source of truth (pyproject.toml)
- Simple implementation (tomllib built-in)
- Cached for performance (lru_cache)
"""

import os
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_api_version() -> str:
    """
    Read API version from pyproject.toml.

    Uses Python 3.11+ built-in tomllib (no external dependencies).
    Cached for performance - call .cache_clear() in tests.

    Returns:
        Version string (e.g., "0.1.0") or "unknown" on error

    Example:
        >>> version = get_api_version()
        >>> print(version)
        '0.1.0'
    """
    try:
        import tomllib  # Python 3.11+ built-in

        # Navigate to pyproject.toml (one level up from api/)
        api_dir = Path(__file__).parent
        pyproject_path = api_dir.parent / "pyproject.toml"

        if not pyproject_path.exists():
            logger.warning(f"pyproject.toml not found: {pyproject_path}")
            return "unknown"

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            version = data.get("project", {}).get("version", "unknown")

        logger.debug(f"API version loaded: {version}")
        return version

    except ImportError:
        logger.error("tomllib not available (requires Python 3.11+)")
        return "unknown"
    except Exception as e:
        logger.error(f"Failed to read API version: {e}")
        return "unknown"


def get_all_versions() -> dict[str, Any]:
    """
    Aggregate versions from all Phentrieve components.

    Returns:
        Dictionary with version info for API, CLI, environment, timestamp

    Example:
        >>> versions = get_all_versions()
        >>> versions["api"]["version"]
        '0.1.0'
        >>> versions["cli"]["version"]
        '0.1.0'
    """
    api_version = get_api_version()

    # CLI version is same as API version (both from pyproject.toml)
    # Frontend version is fetched separately by frontend code

    return {
        "cli": {
            "version": api_version,
            "name": "phentrieve",
            "type": "Python CLI/Library"
        },
        "api": {
            "version": api_version,
            "name": "phentrieve-api",
            "type": "FastAPI"
        },
        "environment": os.getenv("ENV", os.getenv("ENVIRONMENT", "development")),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
```

**Key Design Decisions**:
- ✅ **`@lru_cache`**: Reads file once, caches result (performance)
- ✅ **`tomllib`**: Python 3.11+ built-in (no dependencies)
- ✅ **Error handling**: Returns "unknown" on failure (graceful degradation)
- ✅ **Logging**: Debug logs for visibility, error logs for failures
- ✅ **Type hints**: Clear function signatures for maintainability
- ✅ **Docstrings**: Comprehensive documentation with examples

---

### 1.3 Create Version API Endpoint

**File**: `api/routers/system.py` (new file or extend existing)

```python
"""
System endpoints for health checks, version info, and API metadata.

Public endpoints (no authentication required) for monitoring and debugging.
"""

import logging
from fastapi import APIRouter

from api.version import get_all_versions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/system", tags=["System"])


@router.get("/version")
async def get_version() -> dict:
    """
    Get version information for all Phentrieve components.

    **Public endpoint** - no authentication required.
    Useful for monitoring, debugging, and UI version display.

    Returns:
        {
            "cli": {
                "version": "0.1.0",
                "name": "phentrieve",
                "type": "Python CLI/Library"
            },
            "api": {
                "version": "0.1.0",
                "name": "phentrieve-api",
                "type": "FastAPI"
            },
            "environment": "development",
            "timestamp": "2025-11-21T10:30:00+00:00"
        }
    """
    logger.info("Version information requested")

    versions = get_all_versions()

    logger.debug(
        f"Version info returned: CLI={versions['cli']['version']}, "
        f"API={versions['api']['version']}, "
        f"Environment={versions['environment']}"
    )

    return versions


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint for monitoring and connection status.

    **Public endpoint** - no authentication required.
    Always returns 200 OK when API is alive.

    Returns:
        {
            "status": "healthy",
            "service": "phentrieve-api",
            "version": "0.1.0",
            "timestamp": "2025-11-21T10:30:00+00:00"
        }
    """
    from api.version import get_api_version
    from datetime import datetime, timezone

    return {
        "status": "healthy",
        "service": "phentrieve-api",
        "version": get_api_version(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
```

**Endpoint Design**:
- ✅ **Public endpoints**: No auth required (useful for monitoring)
- ✅ **Structured logging**: Info + debug logs with context
- ✅ **Async/await**: Follows FastAPI best practices
- ✅ **Clear responses**: JSON with version metadata
- ✅ **Health check**: Simple 200 OK (connection verification)

---

### 1.4 Register System Router

**File**: `api/main.py`

```python
# Add to existing imports
from api.routers import system  # New import

# Add to router registration
app.include_router(system.router)
```

---

### 1.5 Update CLI Version Exposure

**File**: `phentrieve/__init__.py`

```python
"""
Phentrieve - HPO term extraction and mapping toolkit.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("phentrieve")
except PackageNotFoundError:
    # Package not installed, read from pyproject.toml
    import tomllib
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
        __version__ = data["project"]["version"]

__all__ = ["__version__"]
```

**Key Features**:
- ✅ **`importlib.metadata`**: Read version from installed package (standard)
- ✅ **Fallback**: Read from pyproject.toml if not installed (dev mode)
- ✅ **Zero dependencies**: Uses built-in Python modules only

---

## Phase 2: Frontend Version Management

### 2.1 Update package.json

**File**: `frontend/package.json`

```json
{
  "name": "phentrieve-frontend",
  "version": "0.1.0",
  "description": "Phentrieve web interface for HPO term mapping",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "vue": "^3.5.13",
    "vuetify": "^3.7.5",
    "pinia": "^2.2.8"
  }
}
```

---

### 2.2 Configure Vite for Version Injection

**File**: `frontend/vite.config.js`

```javascript
import { defineConfig } from 'vite'
import { readFileSync } from 'fs'
import vue from '@vitejs/plugin-vue'
import vuetify from 'vite-plugin-vuetify'

// Read version from package.json at build time
const packageJson = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf-8')
)

export default defineConfig({
  plugins: [
    vue(),
    vuetify({ autoImport: true })
  ],

  // Inject version as global constant (build-time, zero runtime cost)
  define: {
    __APP_VERSION__: JSON.stringify(packageJson.version)
  },

  // ... rest of existing config
})
```

**Key Features**:
- ✅ **Build-time injection**: Version baked into bundle (zero runtime overhead)
- ✅ **Global constant**: `__APP_VERSION__` accessible everywhere
- ✅ **Type-safe**: Can add TypeScript declaration if needed

---

### 2.3 Add TypeScript Declaration (Optional but Recommended)

**File**: `frontend/src/types/globals.d.ts` (new file)

```typescript
/**
 * Global constants injected by Vite at build time.
 */

/**
 * Application version from package.json.
 * Injected at build time by Vite's define config.
 *
 * @example
 * console.log(__APP_VERSION__) // "0.1.0"
 */
declare const __APP_VERSION__: string
```

---

### 2.4 Create Frontend Version Utility

**File**: `frontend/src/utils/version.js` (new file)

```javascript
/**
 * Version management utilities for Phentrieve frontend.
 *
 * Provides functions to:
 * - Get frontend version (build-time injected)
 * - Fetch API version from backend
 * - Aggregate all component versions
 *
 * Following DRY, KISS, SOLID principles:
 * - Single source of truth (package.json via Vite)
 * - Simple implementation (no complex logic)
 * - Graceful degradation (works even if API is down)
 */

import api from '@/api/axios'

/**
 * Get frontend version (build-time injection from package.json).
 *
 * Version is baked into bundle at build time - zero runtime cost.
 *
 * @returns {string} Frontend version (e.g., "0.1.0")
 *
 * @example
 * const version = getFrontendVersion()
 * console.log(version) // "0.1.0"
 */
export function getFrontendVersion() {
  return __APP_VERSION__  // Injected by Vite at build time
}

/**
 * Fetch all component versions from API and combine with frontend version.
 *
 * Makes HTTP GET request to /api/v1/system/version endpoint.
 * Gracefully handles API failures - returns frontend version even if API is down.
 *
 * @returns {Promise<Object>} All component versions
 *
 * @example
 * const versions = await getAllVersions()
 * console.log(versions.frontend.version) // "0.1.0"
 * console.log(versions.api.version)      // "0.1.0" or "unknown"
 * console.log(versions.cli.version)      // "0.1.0" or "unknown"
 */
export async function getAllVersions() {
  try {
    const response = await api.get('/system/version')

    // Combine API response with frontend version
    return {
      ...response.data,  // CLI + API versions from backend
      frontend: {
        version: getFrontendVersion(),
        name: 'phentrieve-frontend',
        type: 'Vue.js'
      }
    }
  } catch (error) {
    console.error('Failed to fetch versions from API:', error)

    // Graceful degradation - return frontend version even if API fails
    return {
      frontend: {
        version: getFrontendVersion(),
        name: 'phentrieve-frontend',
        type: 'Vue.js'
      },
      api: { version: 'unknown', name: 'phentrieve-api', type: 'FastAPI' },
      cli: { version: 'unknown', name: 'phentrieve', type: 'Python CLI' },
      environment: 'unknown',
      timestamp: new Date().toISOString()
    }
  }
}
```

**Key Design Decisions**:
- ✅ **Graceful degradation**: Returns frontend version even if API is down
- ✅ **Centralized logic**: Single source for version operations
- ✅ **Error handling**: Catches API failures, provides fallback
- ✅ **JSDoc comments**: Clear documentation for IDE autocomplete

---

## Phase 3: CLI Version Management

### 3.1 Add --version Flag to CLI

**File**: `phentrieve/cli/__init__.py`

```python
"""
Phentrieve CLI entrypoint.
"""

import typer
from phentrieve import __version__

app = typer.Typer(
    name="phentrieve",
    help="Human Phenotype Ontology (HPO) term extraction and mapping toolkit",
    add_completion=False
)

def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(f"phentrieve version {__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True
    )
):
    """Phentrieve CLI - HPO term extraction toolkit."""
    pass
```

**Usage**:
```bash
$ phentrieve --version
phentrieve version 0.1.0

$ phentrieve -v
phentrieve version 0.1.0
```

---

## Phase 4: Connection Status Indicator

### 4.1 Create API Health Service

**File**: `frontend/src/services/api-health.js` (new file)

```javascript
/**
 * API health monitoring service for connection status tracking.
 *
 * Polls /api/v1/system/health endpoint every 30 seconds to verify API connectivity.
 * Provides reactive connection status for UI components.
 *
 * Design patterns:
 * - Singleton: Single health monitor across entire app
 * - Observer: Components subscribe to connection status changes
 * - Retry: Auto-retry on failures with configurable interval
 *
 * Following KISS principle: Simple polling, no WebSocket complexity for REST API.
 */

import { ref } from 'vue'
import api from '@/api/axios'

class ApiHealthService {
  constructor() {
    /**
     * Reactive connection status (true = connected, false = disconnected).
     * Bind directly to Vue components for automatic UI updates.
     * @type {import('vue').Ref<boolean>}
     */
    this.connected = ref(true)

    /**
     * Timestamp of last successful health check.
     * @type {import('vue').Ref<Date|null>}
     */
    this.lastCheck = ref(null)

    /**
     * Response time of last health check in milliseconds.
     * @type {import('vue').Ref<number|null>}
     */
    this.responseTime = ref(null)

    /**
     * Interval between health checks (milliseconds).
     * @type {number}
     */
    this.checkInterval = 30000  // 30 seconds

    /**
     * Timeout for health check requests (milliseconds).
     * @type {number}
     */
    this.requestTimeout = 5000  // 5 seconds

    /**
     * Interval timer ID for cleanup.
     * @type {number|null}
     * @private
     */
    this.intervalId = null
  }

  /**
   * Perform a single health check.
   * Measures response time and updates connection status.
   *
   * @returns {Promise<boolean>} True if healthy, false otherwise
   */
  async checkHealth() {
    const startTime = performance.now()

    try {
      const response = await api.get('/system/health', {
        timeout: this.requestTimeout
      })

      const endTime = performance.now()
      const responseTimeMs = Math.round(endTime - startTime)

      // Update reactive state
      this.connected.value = response.status === 200
      this.lastCheck.value = new Date()
      this.responseTime.value = responseTimeMs

      console.log(`[API Health] Connected (${responseTimeMs}ms)`)

      return true
    } catch (error) {
      console.error('[API Health] Check failed:', error.message)

      // Update reactive state
      this.connected.value = false
      this.lastCheck.value = new Date()
      this.responseTime.value = null

      return false
    }
  }

  /**
   * Start periodic health monitoring.
   * Performs initial check immediately, then repeats every checkInterval.
   */
  startMonitoring() {
    if (this.intervalId) {
      console.warn('[API Health] Monitoring already started')
      return
    }

    console.log(`[API Health] Starting monitoring (every ${this.checkInterval / 1000}s)`)

    // Initial check
    this.checkHealth()

    // Periodic checks
    this.intervalId = setInterval(() => {
      this.checkHealth()
    }, this.checkInterval)
  }

  /**
   * Stop periodic health monitoring.
   * Call this in component's onUnmounted to prevent memory leaks.
   */
  stopMonitoring() {
    if (this.intervalId) {
      console.log('[API Health] Stopping monitoring')
      clearInterval(this.intervalId)
      this.intervalId = null
    }
  }
}

// Singleton instance - shared across entire app
export const apiHealthService = new ApiHealthService()

/**
 * Vue composable for using API health service in components.
 *
 * Provides reactive refs that automatically update UI on connection changes.
 *
 * @returns {Object} API health service interface
 *
 * @example
 * // In Vue component
 * import { onMounted, onUnmounted } from 'vue'
 * import { useApiHealth } from '@/services/api-health'
 *
 * const { connected, responseTime, startMonitoring, stopMonitoring } = useApiHealth()
 *
 * onMounted(() => startMonitoring())
 * onUnmounted(() => stopMonitoring())
 *
 * // In template
 * <div>API Status: {{ connected ? 'Online' : 'Offline' }}</div>
 * <div v-if="responseTime">Response Time: {{ responseTime }}ms</div>
 */
export function useApiHealth() {
  return {
    connected: apiHealthService.connected,
    lastCheck: apiHealthService.lastCheck,
    responseTime: apiHealthService.responseTime,
    checkHealth: () => apiHealthService.checkHealth(),
    startMonitoring: () => apiHealthService.startMonitoring(),
    stopMonitoring: () => apiHealthService.stopMonitoring()
  }
}
```

**Key Design Decisions**:
- ✅ **Singleton pattern**: One instance for entire app (efficient)
- ✅ **Reactive refs**: Automatic UI updates via Vue reactivity
- ✅ **Composable API**: Vue 3 Composition API style
- ✅ **Performance tracking**: Measures response time for diagnostics
- ✅ **Lifecycle management**: Start/stop methods for cleanup
- ✅ **Simple polling**: No WebSocket complexity (KISS principle)

---

## Phase 5: UI Components

### 5.1 Create AppFooter Component

**File**: `frontend/src/components/AppFooter.vue` (new file)

```vue
<template>
  <v-footer app class="bg-surface-light py-1" style="min-height: 48px">
    <v-container class="py-0">
      <v-row align="center" justify="space-between" no-gutters>

        <!-- Left: Version Information Button -->
        <v-col cols="auto" class="py-0">
          <v-menu location="top" :close-on-content-click="false">
            <template #activator="{ props: menuProps }">
              <v-btn variant="text" size="small" v-bind="menuProps">
                <v-icon icon="mdi-information-outline" size="small" start />
                v{{ frontendVersion }}
              </v-btn>
            </template>

            <!-- Version Details Popup -->
            <v-card min-width="340">
              <v-card-title class="d-flex align-center">
                <v-icon icon="mdi-package-variant" class="mr-2" />
                Version Information

                <v-spacer />

                <v-btn
                  icon="mdi-refresh"
                  variant="text"
                  size="small"
                  :loading="loading"
                  @click="refreshVersions"
                />
              </v-card-title>

              <v-divider />

              <v-card-text>
                <!-- Frontend Version -->
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-vuejs" color="success" />
                  </template>
                  <v-list-item-title>Frontend</v-list-item-title>
                  <v-list-item-subtitle>
                    {{ frontendVersion }} (Vue.js)
                  </v-list-item-subtitle>
                </v-list-item>

                <!-- API Version -->
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-api" color="primary" />
                  </template>
                  <v-list-item-title>API</v-list-item-title>
                  <v-list-item-subtitle>
                    {{ apiVersion }} (FastAPI)
                  </v-list-item-subtitle>
                </v-list-item>

                <!-- CLI Version -->
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-console" color="info" />
                  </template>
                  <v-list-item-title>CLI</v-list-item-title>
                  <v-list-item-subtitle>
                    {{ cliVersion }} (Python)
                  </v-list-item-subtitle>
                </v-list-item>

                <v-divider class="my-2" />

                <!-- Environment Badge -->
                <div class="d-flex align-center justify-space-between">
                  <span class="text-caption text-medium-emphasis">Environment:</span>
                  <v-chip
                    :color="getEnvironmentColor(environment)"
                    size="small"
                    label
                  >
                    {{ environment }}
                  </v-chip>
                </div>

                <!-- Last Updated -->
                <div v-if="timestamp" class="text-caption text-medium-emphasis mt-2">
                  Updated: {{ formatTimestamp(timestamp) }}
                </div>
              </v-card-text>
            </v-card>
          </v-menu>
        </v-col>

        <!-- Center: Connection Status Indicator -->
        <v-col cols="auto" class="py-0">
          <v-chip
            :color="apiConnected ? 'success' : 'error'"
            size="x-small"
            label
            class="mx-2"
          >
            <v-icon size="x-small" start>
              {{ apiConnected ? 'mdi-lan-connect' : 'mdi-lan-disconnect' }}
            </v-icon>
            API {{ apiConnected ? 'Online' : 'Offline' }}
            <span v-if="apiConnected && responseTime" class="ml-1">
              ({{ responseTime }}ms)
            </span>
          </v-chip>
        </v-col>

        <!-- Right: Project Links -->
        <v-col cols="auto" class="py-0">
          <v-btn
            href="https://github.com/berntpopp/phentrieve"
            target="_blank"
            variant="text"
            size="small"
          >
            <v-icon icon="mdi-github" size="small" start />
            GitHub
          </v-btn>
        </v-col>

      </v-row>
    </v-container>
  </v-footer>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { getAllVersions } from '@/utils/version'
import { useApiHealth } from '@/services/api-health'

// Version state
const frontendVersion = ref(__APP_VERSION__)  // Build-time injected
const apiVersion = ref('...')
const cliVersion = ref('...')
const environment = ref('...')
const timestamp = ref(null)
const loading = ref(false)

// API health monitoring
const {
  connected: apiConnected,
  responseTime,
  startMonitoring,
  stopMonitoring
} = useApiHealth()

/**
 * Fetch versions from API and update state.
 */
async function refreshVersions() {
  loading.value = true

  try {
    const versions = await getAllVersions()

    // Update state
    frontendVersion.value = versions.frontend.version
    apiVersion.value = versions.api?.version || 'unknown'
    cliVersion.value = versions.cli?.version || 'unknown'
    environment.value = versions.environment || 'unknown'
    timestamp.value = versions.timestamp

    console.log('[AppFooter] Versions refreshed:', versions)
  } catch (error) {
    console.error('[AppFooter] Failed to refresh versions:', error)
    apiVersion.value = 'error'
    cliVersion.value = 'error'
  } finally {
    loading.value = false
  }
}

/**
 * Get color for environment badge.
 */
function getEnvironmentColor(env) {
  const colors = {
    production: 'success',
    staging: 'warning',
    development: 'info'
  }
  return colors[env] || 'default'
}

/**
 * Format timestamp for display.
 */
function formatTimestamp(ts) {
  if (!ts) return 'Unknown'

  try {
    const date = new Date(ts)
    return date.toLocaleString()
  } catch (error) {
    return 'Invalid'
  }
}

// Lifecycle hooks
onMounted(() => {
  refreshVersions()      // Fetch versions on mount
  startMonitoring()      // Start API health checks
})

onUnmounted(() => {
  stopMonitoring()       // Stop health checks on unmount (prevent memory leaks)
})
</script>

<style scoped>
/* Compact footer styling */
.v-footer {
  border-top: thin solid rgba(var(--v-border-color), var(--v-border-opacity));
}
</style>
```

**Key Features**:
- ✅ **Compact design**: Minimal footer height (48px)
- ✅ **Version popup**: Detailed info on click (no clutter)
- ✅ **Connection indicator**: Real-time API status + response time
- ✅ **Environment badge**: Color-coded (prod=green, staging=yellow, dev=blue)
- ✅ **Auto-refresh**: Fetches versions on mount
- ✅ **Lifecycle management**: Starts/stops monitoring properly
- ✅ **Vuetify components**: Consistent Material Design

---

### 5.2 Register AppFooter in App.vue

**File**: `frontend/src/App.vue`

```vue
<template>
  <v-app>
    <!-- Existing navigation/toolbar -->
    <v-app-bar app>
      <!-- ... existing app bar content ... -->
    </v-app-bar>

    <!-- Main content -->
    <v-main>
      <router-view />
    </v-main>

    <!-- Add footer -->
    <AppFooter />
  </v-app>
</template>

<script setup>
import AppFooter from '@/components/AppFooter.vue'
</script>
```

---

## Testing Strategy

### Backend Tests

**File**: `tests/unit/api/test_version.py` (new file)

```python
"""
Tests for API version management.
"""

import pytest
from api.version import get_api_version, get_all_versions


def test_get_api_version():
    """Test that API version can be read from pyproject.toml."""
    version = get_api_version()

    assert version is not None
    assert version != "unknown"
    assert "." in version  # Should be semantic version (x.y.z)


def test_get_api_version_caching():
    """Test that get_api_version uses LRU cache correctly."""
    version1 = get_api_version()
    version2 = get_api_version()

    assert version1 == version2
    assert version1 is version2  # Same object (cached)


def test_get_all_versions():
    """Test aggregation of all component versions."""
    versions = get_all_versions()

    assert "cli" in versions
    assert "api" in versions
    assert "environment" in versions
    assert "timestamp" in versions

    assert versions["cli"]["version"] is not None
    assert versions["api"]["version"] is not None


def test_version_endpoint(client):
    """Test /api/v1/system/version endpoint."""
    response = client.get("/api/v1/system/version")

    assert response.status_code == 200

    data = response.json()
    assert "cli" in data
    assert "api" in data
    assert "environment" in data


def test_health_endpoint(client):
    """Test /api/v1/system/health endpoint."""
    response = client.get("/api/v1/system/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
```

---

### Frontend Tests

**File**: `frontend/src/utils/version.test.js` (new file)

```javascript
import { describe, it, expect, vi } from 'vitest'
import { getFrontendVersion, getAllVersions } from './version'

describe('Version Utils', () => {
  it('getFrontendVersion returns build-time injected version', () => {
    const version = getFrontendVersion()

    expect(version).toBeDefined()
    expect(typeof version).toBe('string')
    expect(version).toMatch(/^\d+\.\d+\.\d+/)  // Matches x.y.z pattern
  })

  it('getAllVersions fetches from API and combines with frontend', async () => {
    // Mock API response
    vi.mock('@/api/axios', () => ({
      default: {
        get: vi.fn(() => Promise.resolve({
          data: {
            api: { version: '0.1.0' },
            cli: { version: '0.1.0' }
          }
        }))
      }
    }))

    const versions = await getAllVersions()

    expect(versions).toHaveProperty('frontend')
    expect(versions).toHaveProperty('api')
    expect(versions).toHaveProperty('cli')
    expect(versions.frontend.version).toBeDefined()
  })

  it('getAllVersions handles API failure gracefully', async () => {
    // Mock API error
    vi.mock('@/api/axios', () => ({
      default: {
        get: vi.fn(() => Promise.reject(new Error('Network error')))
      }
    }))

    const versions = await getAllVersions()

    // Should still return frontend version
    expect(versions.frontend.version).toBeDefined()
    expect(versions.api.version).toBe('unknown')
    expect(versions.cli.version).toBe('unknown')
  })
})
```

---

## Best Practices & Principles

### DRY (Don't Repeat Yourself)

✅ **Single source of truth**:
- Python versions: `pyproject.toml`
- Frontend versions: `package.json`
- No duplicate version strings

✅ **Reuse utilities**:
- `api/version.py` shared by CLI and API
- `frontend/src/utils/version.js` shared across components

✅ **Avoid duplication**:
- Version reading logic in one place only
- Connection status managed by single service

---

### KISS (Keep It Simple, Stupid)

✅ **Simple solutions**:
- Read versions from files (don't calculate)
- Polling for health checks (no WebSocket overkill for REST API)
- Built-in modules (no external dependencies where possible)

✅ **Avoid complexity**:
- No fancy version calculation algorithms
- No complex state machines for connection status
- No premature optimization

---

### SOLID Principles

✅ **Single Responsibility Principle (SRP)**:
- `version.py` → Version reading only
- `api-health.js` → Health monitoring only
- `AppFooter.vue` → UI display only

✅ **Open/Closed Principle (OCP)**:
- Version utilities extensible without modification
- New version sources can be added without changing existing code

✅ **Liskov Substitution Principle (LSP)**:
- All version readers return consistent format (version string)
- All health checkers return boolean (connected/disconnected)

✅ **Interface Segregation Principle (ISP)**:
- Small, focused interfaces (useApiHealth(), getAllVersions())
- Components depend only on what they need

✅ **Dependency Inversion Principle (DIP)**:
- Depend on abstractions (version interface, health interface)
- Not dependent on concrete implementations

---

### No Antipatterns

❌ **Avoided antipatterns**:
- **God Object**: No single object doing everything
- **Hardcoded Values**: No version strings in multiple places
- **Tight Coupling**: Components loosely coupled via interfaces
- **Magic Numbers**: Constants defined clearly (checkInterval = 30000)
- **Premature Optimization**: Profile first, optimize later
- **Reinventing the Wheel**: Use built-in modules (tomllib, importlib.metadata)

---

## References

### Documentation

- [Semantic Versioning 2.0.0](https://semver.org/)
- [Python Semantic Release](https://python-semantic-release.readthedocs.io/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Vite Build-time Constants](https://vitejs.dev/config/shared-options.html#define)
- [Vue 3 Composition API](https://vuejs.org/guide/extras/composition-api-faq.html)

### Inspiration

- **kidney-genetics-db** repository (version management and WebSocket patterns)
- [Microsoft Engineering Playbook - Component Versioning](https://microsoft.github.io/code-with-engineering-playbook/source-control/component-versioning/)
- [Mastering Monorepo Versioning](https://amarchenko.dev/blog/2023-09-26-versioning/)

---

## Implementation Checklist

### Phase 1: Backend ✅
- [ ] Update `pyproject.toml` with `0.1.0` and semantic-release config
- [ ] Create `api/version.py` utility module
- [ ] Create `api/routers/system.py` with `/version` and `/health` endpoints
- [ ] Register system router in `api/main.py`
- [ ] Update `phentrieve/__init__.py` with `__version__`
- [ ] Add `--version` flag to CLI
- [ ] Write backend tests (`tests/unit/api/test_version.py`)

### Phase 2: Frontend ✅
- [ ] Update `frontend/package.json` to `0.1.0`
- [ ] Configure Vite for version injection (`vite.config.js`)
- [ ] Add TypeScript declarations (`types/globals.d.ts`)
- [ ] Create `frontend/src/utils/version.js`
- [ ] Write frontend tests (`version.test.js`)

### Phase 3: Connection Status ✅
- [ ] Create `frontend/src/services/api-health.js`
- [ ] Test health monitoring service

### Phase 4: UI Components ✅
- [ ] Create `frontend/src/components/AppFooter.vue`
- [ ] Register AppFooter in `App.vue`
- [ ] Test UI in browser (manual)

### Phase 5: Documentation ✅
- [ ] Update `CLAUDE.md` with version management info
- [ ] Add version management to README
- [ ] Document API endpoints in OpenAPI schema

---

## Success Criteria

✅ **Version Management**:
- CLI shows version with `phentrieve --version`
- API `/version` endpoint returns all component versions
- Frontend displays version in footer
- All versions start at `0.1.0` (alpha)

✅ **Connection Indicator**:
- Footer shows green "API Online" when connected
- Footer shows red "API Offline" when disconnected
- Response time displayed when available
- Auto-refreshes every 30 seconds

✅ **Code Quality**:
- 0 linting errors
- 0 type errors (mypy)
- All tests passing
- Follows DRY, KISS, SOLID principles

✅ **User Experience**:
- Version info always accessible (footer)
- Clear visual feedback for connection status
- No performance impact (build-time injection + caching)
- Graceful degradation (works even if API is down)

---

**END OF PLAN**
