"""
System endpoints for health checks, version info, and API metadata.

Public endpoints (no authentication required) for monitoring and debugging.
"""

import logging

from fastapi import APIRouter

from api.version import get_all_versions, get_api_version

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
                "version": "0.2.0",
                "name": "phentrieve",
                "type": "Python CLI/Library"
            },
            "api": {
                "version": "0.2.0",
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
            "version": "0.2.0",
            "timestamp": "2025-11-21T10:30:00+00:00"
        }
    """
    from datetime import datetime, timezone

    return {
        "status": "healthy",
        "service": "phentrieve-api",
        "version": get_api_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
