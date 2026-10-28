from datetime import UTC, datetime

from fastapi import APIRouter

from api.version import get_api_version

# Create router without any tags or prefixes - these will be set in main.py
router = APIRouter()


def health_payload() -> dict:
    """Build the health response shared by every health route."""
    return {
        "status": "healthy",
        "service": "phentrieve-api",
        "version": get_api_version(),
        "timestamp": datetime.now(UTC).isoformat(),
    }


# Canonical route is /api/v1/health (no trailing slash, so Docker probes get a
# 200 instead of a 307 redirect). The trailing-slash form is kept as an alias.
@router.get("")
@router.get("/", include_in_schema=False)
async def health_check() -> dict:
    """
    Health check endpoint for container orchestration and connection status.

    **Public endpoint** - no authentication required.
    Always returns 200 OK when the API is alive.
    """
    return health_payload()
