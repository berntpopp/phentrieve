"""
System endpoints for version info and API metadata.

Public endpoints (no authentication required) for monitoring and debugging.
"""

import logging

from fastapi import APIRouter

from api.routers.health import health_payload
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
                "version": "0.18.0",
                "name": "phentrieve",
                "type": "Python CLI/Library"
            },
            "api": {
                "version": "0.9.1",
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


@router.get("/health", include_in_schema=False)
async def health_check() -> dict:
    """Deprecated alias of ``GET /api/v1/health``; returns the same payload."""
    return health_payload()
