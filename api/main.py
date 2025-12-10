import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to Python path
# This needs to be before other project-specific imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import (  # noqa: E402
    ALLOWED_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    LOG_LEVEL,
)
from api.dependencies import (  # noqa: E402
    get_cross_encoder_dependency,
    get_dense_retriever_dependency,
    get_sbert_model_dependency,
)
from api.routers import (  # noqa: E402
    config_info_router,
    health,
    query_router,
    similarity_router,
    system,
    text_processing_router,
)
from phentrieve.config import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_MULTI_VECTOR,
    DEFAULT_RERANKER_MODEL,
)

logger = logging.getLogger(__name__)
# Configure logging for the API using config value
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))


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

        # Attempt to pre-load default reranker (optional, good for responsiveness)
        if DEFAULT_RERANKER_MODEL:
            await get_cross_encoder_dependency(
                reranker_model_name=DEFAULT_RERANKER_MODEL,
                device_override=DEFAULT_DEVICE,
            )
            logger.info(
                f"Default reranker '{DEFAULT_RERANKER_MODEL}' pre-loading task initiated."
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
    yield  # This is where the application runs

    # Shutdown logic (if needed)
    logger.info("Shutting down Phentrieve API...")


app = FastAPI(title="Phentrieve API", version="0.1.0", lifespan=lifespan)

# Configure CORS using config values
# Origins can be customized via ALLOWED_ORIGINS env var or api/local_api_config.env
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)
# This content was moved to the lifespan context manager


# Include routers
app.include_router(query_router.router, prefix="/api/v1/query", tags=["HPO Term Query"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health Check"])
app.include_router(
    system.router
)  # System router (version, health) - prefix set in router
app.include_router(
    similarity_router.router, prefix="/api/v1/similarity", tags=["HPO Term Similarity"]
)
app.include_router(config_info_router.router, prefix="/api/v1")
app.include_router(
    text_processing_router.router, tags=["Text Processing and HPO Extraction"]
)


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
def _try_mount_mcp() -> None:
    """Attempt to mount MCP server at /mcp if enabled and dependencies available."""
    try:
        from api.mcp.config import is_mcp_http_enabled

        if not is_mcp_http_enabled():
            return

        # Import MCP server factory (requires fastapi-mcp optional dependency)
        from api.mcp.server import create_mcp_server

        mcp = create_mcp_server(app)
        mcp.mount()  # Mounts at /mcp by default
        logger.info("MCP server mounted at /mcp (ENABLE_MCP_HTTP=true)")
    except ImportError:
        # fastapi-mcp not installed - silently skip
        logger.debug("MCP dependencies not available - skipping /mcp mount")
    except Exception as e:
        # Log unexpected errors but don't crash the API
        logger.warning(f"Failed to mount MCP server: {e}")


# Mount MCP if enabled (runs at module load time)
_try_mount_mcp()


@app.get(
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
        "version": "0.1.0",
        "description": "API for Human Phenotype Ontology (HPO) term retrieval and semantic similarity calculation",
        "endpoints": {
            "HPO Term Query": {
                "description": "Query HPO terms based on clinical text",
                "endpoints": [
                    {
                        "path": "/api/v1/query/",
                        "methods": ["GET", "POST"],
                        "description": "Retrieve relevant HPO terms for clinical text",
                    }
                ],
            },
            "Text Processing": {
                "description": "Process clinical text to extract HPO terms with advanced configuration",
                "endpoints": [
                    {
                        "path": "/api/v1/text/process",
                        "methods": ["POST"],
                        "description": "Process raw clinical text with customizable chunking, reranking, and assertion detection settings",
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
