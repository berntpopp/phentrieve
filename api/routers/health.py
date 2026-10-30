from datetime import UTC, datetime

from fastapi import APIRouter, Response, status

from api.dependencies import LOADED_SBERT_MODELS
from api.version import get_api_version
from phentrieve.config import DEFAULT_MODEL

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


@router.get("/ready", include_in_schema=False)
async def readiness_check(response: Response) -> dict:
    """
    Report whether the default embedding model is loaded.

    The API accepts requests while the model still loads in the background, and
    queries answer 503 until then. Liveness stays on the health route, so a slow
    model load does not mark the container unhealthy.
    """
    ready = DEFAULT_MODEL in LOADED_SBERT_MODELS
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        response.headers["Retry-After"] = "30"
    return {**health_payload(), "status": "ready" if ready else "loading"}
