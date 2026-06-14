import logging
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import api.config as api_config
from api.config import (
    ALLOWED_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    LOG_LEVEL,
)
from api.dependencies import (
    cleanup_model_caches,
    get_dense_retriever_dependency,
    get_sbert_model_dependency,
)
from api.routers import (
    config_info_router,
    health,
    phenopacket_router,
    query_router,
    similarity_router,
    system,
    text_processing_router,
)
from api.schemas.errors import ErrorResponse
from api.version import get_api_version
from phentrieve.config import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_MULTI_VECTOR,
)

logger = logging.getLogger(__name__)
# Configure logging for the API using config value
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))

# Single source of truth for the versioned API path prefix (used by the routers
# and to place the interactive docs/OpenAPI behind the reverse proxy).
API_V1_PREFIX = "/api/v1"


# =============================================================================
# MCP (Model Context Protocol) HTTP Mounting
# =============================================================================
# When ENABLE_MCP_HTTP=true, mount MCP at /mcp on the same domain as the API.
# This enables LLM clients to access HPO term extraction tools via HTTP.
#
# Production URL: https://phentrieve.example.com/mcp
# Local dev URL:  http://localhost:8734/mcp
#
# Environment variables:
#   ENABLE_MCP_HTTP=true         - Enable MCP HTTP endpoint
#   PHENTRIEVE_MCP_ENABLE_HTTP=true  - Alternative env var
# =============================================================================
def _try_mount_mcp(target_app: FastAPI) -> None:
    """Attempt to mount MCP server at /mcp if enabled and dependencies available."""
    try:
        from api.mcp.config import is_mcp_http_enabled

        if not is_mcp_http_enabled():
            return

        # Import MCP facade factory (requires mcp optional dependency)
        from api.mcp.server import mount_phentrieve_mcp_facade

        mount_phentrieve_mcp_facade(target_app)
        logger.info("Phentrieve MCP facade mounted at /mcp using Streamable HTTP")
    except ImportError:
        # fastmcp / mcp optional dependency not installed - silently skip
        logger.debug("MCP dependencies not available - skipping /mcp mount")
    except Exception as e:
        # Log unexpected errors but don't crash the API
        logger.warning(f"Failed to mount MCP server: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Phentrieve API starting up. Attempting to pre-load default models...")
    default_trust_remote_code = True  # For default models known to need it
    try:
        # Pre-load default SBERT model for retrieval
        await get_sbert_model_dependency(
            model_name_requested=DEFAULT_MODEL,
            device_override=DEFAULT_DEVICE,
            trust_remote_code=default_trust_remote_code,
        )
        # Pre-load the retriever associated with the default SBERT model
        # Uses DEFAULT_MULTI_VECTOR to pre-load the correct index type
        await get_dense_retriever_dependency(
            sbert_model_name_for_retriever=DEFAULT_MODEL,
            multi_vector=DEFAULT_MULTI_VECTOR,
        )
        logger.info(
            f"Default SBERT model '{DEFAULT_MODEL}' and its retriever "
            f"(multi_vector={DEFAULT_MULTI_VECTOR}) pre-loading tasks initiated."
        )

    except HTTPException as http_exc:
        if http_exc.status_code == 503 and "is being prepared" in str(http_exc.detail):
            logger.warning(
                f"API Startup: A default model is still loading in the background: {http_exc.detail}"
            )
        elif http_exc.status_code == 503 and "failed to load" in str(http_exc.detail):
            logger.error(
                f"API Startup CRITICAL FAILURE: A default model ('{DEFAULT_MODEL}') failed to load: {http_exc.detail}. The API will start, but this model will be unavailable."
            )
        else:
            logger.error(
                f"API Startup: HTTP error during initial model dependency call: {http_exc.status_code} - {http_exc.detail}",
                exc_info=True,
            )
    except Exception as e:
        logger.error(
            f"API Startup: Unexpected general error during model pre-loading attempt: {e}",
            exc_info=True,
        )

    logger.info(
        "API startup: Default model pre-loading initiation complete (actual loading may be in background)."
    )

    # Optionally seed a pre-verified dev account (no-op unless configured).
    if api_config.PHENTRIEVE_AUTH_ENABLED:
        from api.auth.seed import seed_user_from_config

        seed_user_from_config()

    # Mount MCP if enabled (moved from module-level to lifespan)
    _try_mount_mcp(app)
    mcp_http_app = getattr(app.state, "phentrieve_mcp_http_app", None)

    try:
        if mcp_http_app is None:
            yield  # This is where the application runs
        else:
            # Enter the FastMCP ASGI app's lifespan to start the
            # StreamableHTTP session manager for the mounted /mcp sub-app.
            async with mcp_http_app.router.lifespan_context(app):
                yield
    finally:
        # Shutdown — cancel outstanding loads, then clear caches
        await cleanup_model_caches()
        logger.info("Shutting down Phentrieve API...")


def create_app() -> FastAPI:
    """Application factory for the Phentrieve API."""
    application = FastAPI(
        title="Phentrieve API",
        version=get_api_version(),
        lifespan=lifespan,
        # Serve the interactive docs/OpenAPI under the API prefix so they are
        # reachable through the frontend reverse proxy (which only forwards /api
        # and /mcp). At the app root they would be shadowed by the SPA.
        docs_url=f"{API_V1_PREFIX}/docs",
        redoc_url=f"{API_V1_PREFIX}/redoc",
        openapi_url=f"{API_V1_PREFIX}/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=CORS_ALLOW_CREDENTIALS,
        allow_methods=CORS_ALLOW_METHODS,
        allow_headers=CORS_ALLOW_HEADERS,
    )

    def _status_slug(status_code: int, fallback: str = "http_error") -> str:
        """Derive a machine-readable slug from an HTTP status code."""
        try:
            return HTTPStatus(status_code).phrase.lower().replace(" ", "_")
        except ValueError:
            return fallback

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Render HTTPException via ErrorResponse.

        Registers on StarletteHTTPException (the base class) so that both
        FastAPI routing 404s and explicit HTTPException raises in routers
        are intercepted.
        """
        slug = _status_slug(exc.status_code)
        body = ErrorResponse(
            status_code=exc.status_code,
            error=slug,
            # Pass exc.detail through unchanged — preserves str details AND
            # preserves dict/list details (e.g. similarity_router returns a dict).
            detail=exc.detail if exc.detail is not None else slug,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(exclude_none=True),
            headers=getattr(exc, "headers", None) or None,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Render Pydantic/FastAPI validation errors (422) via ErrorResponse.

        Without this, FastAPI returns its default {"detail": [...]} shape
        which clients would have to parse separately from our ErrorResponse
        contract.
        """
        body = ErrorResponse(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            error="unprocessable_entity",
            detail=jsonable_encoder(exc.errors()),
        )
        return JSONResponse(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            content=body.model_dump(exclude_none=True),
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all 500 handler so unhandled exceptions still conform to
        the ErrorResponse contract.

        Logs the full traceback server-side and returns a generic
        ``internal_server_error`` payload without leaking exception
        internals to clients.
        """
        logger.exception("Unhandled exception in request path: %s", type(exc).__name__)
        body = ErrorResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            error="internal_server_error",
            detail="An unexpected error occurred.",
        )
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content=body.model_dump(exclude_none=True),
        )

    application.include_router(
        query_router.router, prefix=f"{API_V1_PREFIX}/query", tags=["HPO Term Query"]
    )
    application.include_router(
        health.router, prefix=f"{API_V1_PREFIX}/health", tags=["Health Check"]
    )
    application.include_router(system.router)
    application.include_router(
        similarity_router.router,
        prefix=f"{API_V1_PREFIX}/similarity",
        tags=["HPO Term Similarity"],
    )
    application.include_router(phenopacket_router.router)
    application.include_router(config_info_router.router, prefix=API_V1_PREFIX)
    application.include_router(
        text_processing_router.router, tags=["Text Processing and HPO Extraction"]
    )

    @application.get(API_V1_PREFIX, include_in_schema=False)
    @application.get(f"{API_V1_PREFIX}/", include_in_schema=False)
    async def api_v1_root() -> RedirectResponse:
        """Redirect the API base path to the interactive Swagger docs."""
        return RedirectResponse(url=application.docs_url or f"{API_V1_PREFIX}/docs")

    if api_config.PHENTRIEVE_AUTH_ENABLED:
        if not api_config.PHENTRIEVE_AUTH_JWT_SECRET:
            message = (
                "PHENTRIEVE_AUTH_ENABLED is set but PHENTRIEVE_AUTH_JWT_SECRET is "
                "empty. Set a strong secret to enable authentication."
            )
            if api_config.PHENTRIEVE_ENV.strip().lower() == "production":
                raise RuntimeError(message)
            logger.error(
                "%s Auth routes will be mounted but tokens will fail.", message
            )
        from api.auth import router as auth_router

        application.include_router(auth_router.router)
        logger.info("Authentication enabled: mounted /api/v1/auth routes.")

    @application.get(
        "/",
        tags=["API Information"],
        summary="API Information",
        description="Returns information about the Phentrieve API and available endpoints.",
    )
    async def root():
        """API Information endpoint.

        Returns information about the Phentrieve API, its purpose, and available endpoints.
        For full API documentation, visit the /docs endpoint.
        """
        return {
            "api": "Phentrieve API",
            "version": get_api_version(),
            "description": "API for Human Phenotype Ontology (HPO) term retrieval and semantic similarity calculation",
            "endpoints": {
                "HPO Term Query": {
                    "description": "Query HPO terms based on research phenotype text",
                    "endpoints": [
                        {
                            "path": "/api/v1/query/",
                            "methods": ["GET", "POST"],
                            "description": "Retrieve relevant HPO terms for research phenotype text",
                        }
                    ],
                },
                "Text Processing": {
                    "description": "Process research phenotype text to extract HPO terms with advanced configuration",
                    "endpoints": [
                        {
                            "path": "/api/v1/text/process",
                            "methods": ["POST"],
                            "description": "Process raw research phenotype text with customizable chunking and assertion detection settings",
                        }
                    ],
                },
                "HPO Term Similarity": {
                    "description": "Calculate semantic similarity between HPO terms",
                    "endpoints": [
                        {
                            "path": "/api/v1/similarity/{term1_id}/{term2_id}",
                            "methods": ["GET"],
                            "description": "Calculate semantic similarity between two HPO terms",
                        }
                    ],
                },
                "Health Check": {
                    "description": "API health monitoring",
                    "endpoints": [
                        {
                            "path": "/api/v1/health/",
                            "methods": ["GET"],
                            "description": "Check API health status",
                        }
                    ],
                },
                "Documentation": {
                    "description": "API documentation",
                    "endpoints": [
                        {
                            "path": "/docs",
                            "methods": ["GET"],
                            "description": "Swagger UI documentation",
                        },
                        {
                            "path": "/redoc",
                            "methods": ["GET"],
                            "description": "ReDoc documentation",
                        },
                    ],
                },
            },
        }

    return application


# Module-level app for uvicorn and existing imports
app = create_app()
