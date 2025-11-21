"""
Version management utilities for Phentrieve API.

Provides functions to read version information from pyproject.toml
and aggregate versions from all components.

Following DRY, KISS, SOLID principles:
- Single source of truth (pyproject.toml)
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


@lru_cache(maxsize=1)
def get_api_version() -> str:
    """
    Read API version from pyproject.toml.

    Uses Python 3.11+ built-in tomllib or falls back to tomli for Python 3.10.
    Cached for performance - call .cache_clear() in tests.

    Returns:
        Version string (e.g., "0.2.0") or "unknown" on error

    Example:
        >>> version = get_api_version()
        >>> print(version)
        '0.2.0'
    """
    try:
        # Try Python 3.11+ built-in tomllib first
        try:
            import tomllib
        except ImportError:
            # Fall back to tomli for Python 3.10
            import tomli as tomllib  # type: ignore

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

    except ImportError as e:
        logger.error(f"tomllib/tomli not available: {e}")
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
        '0.2.0'
        >>> versions["cli"]["version"]
        '0.2.0'
    """
    api_version = get_api_version()

    # CLI version is same as API version (both from pyproject.toml)
    # Frontend version is fetched separately by frontend code

    return {
        "cli": {
            "version": api_version,
            "name": "phentrieve",
            "type": "Python CLI/Library",
        },
        "api": {
            "version": api_version,
            "name": "phentrieve-api",
            "type": "FastAPI",
        },
        "environment": os.getenv("ENV", os.getenv("ENVIRONMENT", "development")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
