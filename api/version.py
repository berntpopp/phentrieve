"""
Version management utilities for Phentrieve API.

Provides functions to read version information from pyproject.toml files
and aggregate versions from all components.

Version sources:
- CLI: /pyproject.toml (main package version)
- API: /api/pyproject.toml (API-specific version)
- Frontend: /frontend/package.json (managed separately)

Following DRY, KISS, SOLID principles:
- Single source of truth per component (separate pyproject.toml files)
- Simple implementation (tomllib built-in)
- Cached for performance (lru_cache)
"""

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_toml() -> Any:
    """Load tomllib module (Python 3.11+) or tomli fallback."""
    try:
        import tomllib

        return tomllib
    except ImportError:
        import tomli as tomllib

        return tomllib


def _read_version_from_toml(pyproject_path: Path, component: str) -> str:
    """
    Read version from a pyproject.toml file.

    Args:
        pyproject_path: Path to pyproject.toml file
        component: Component name for logging (e.g., "API", "CLI")

    Returns:
        Version string or "unknown" on error
    """
    try:
        tomllib = _load_toml()

        if not pyproject_path.exists():
            logger.warning(f"{component} pyproject.toml not found: {pyproject_path}")
            return "unknown"

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            version: str = data.get("project", {}).get("version", "unknown")

        logger.debug(f"{component} version loaded: {version}")
        return version

    except ImportError as e:
        logger.error(f"tomllib/tomli not available: {e}")
        return "unknown"
    except Exception as e:
        logger.error(f"Failed to read {component} version: {e}")
        return "unknown"


@lru_cache(maxsize=1)
def get_api_version() -> str:
    """
    Read API version from api/pyproject.toml.

    The API has its own pyproject.toml for independent versioning.
    Cached for performance - call .cache_clear() in tests.

    Returns:
        Version string (e.g., "0.4.0") or "unknown" on error

    Example:
        >>> version = get_api_version()
        >>> print(version)
        '0.4.0'
    """
    api_dir = Path(__file__).parent
    pyproject_path = api_dir / "pyproject.toml"
    return _read_version_from_toml(pyproject_path, "API")


@lru_cache(maxsize=1)
def get_cli_version() -> str:
    """
    Read CLI/library version from root pyproject.toml.

    The CLI uses the main package pyproject.toml at the project root.
    Cached for performance - call .cache_clear() in tests.

    Returns:
        Version string (e.g., "0.8.0") or "unknown" on error

    Example:
        >>> version = get_cli_version()
        >>> print(version)
        '0.8.0'
    """
    api_dir = Path(__file__).parent
    pyproject_path = api_dir.parent / "pyproject.toml"
    return _read_version_from_toml(pyproject_path, "CLI")


def get_all_versions() -> dict[str, Any]:
    """
    Aggregate versions from all Phentrieve components.

    Each component has its own version source:
    - CLI: /pyproject.toml (main package)
    - API: /api/pyproject.toml (API-specific)
    - Frontend: Fetched separately by frontend code from package.json

    Returns:
        Dictionary with version info for API, CLI, environment, timestamp

    Example:
        >>> versions = get_all_versions()
        >>> versions["api"]["version"]
        '0.4.0'
        >>> versions["cli"]["version"]
        '0.8.0'
    """
    return {
        "cli": {
            "version": get_cli_version(),
            "name": "phentrieve",
            "type": "Python CLI/Library",
        },
        "api": {
            "version": get_api_version(),
            "name": "phentrieve-api",
            "type": "FastAPI",
        },
        "environment": os.getenv("ENV", os.getenv("ENVIRONMENT", "development")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
