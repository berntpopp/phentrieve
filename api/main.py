import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to Python path
# This needs to be before other project-specific imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routers import (  # noqa: E402
    query_router,
    health,
    similarity_router,
    config_info_router,
    text_processing_router,
)
from api.dependencies import (  # noqa: E402
    get_sbert_model_dependency,
    get_dense_retriever_dependency,
    get_cross_encoder_dependency,
)
from phentrieve.config import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_DEVICE,
)

logger = logging.getLogger(__name__)
# Configure logging for the API
logging.basicConfig(level=logging.INFO)


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
        await get_dense_retriever_dependency(
            sbert_model_name_for_retriever=DEFAULT_MODEL
        )
        logger.info(
            f"Default SBERT model '{DEFAULT_MODEL}' and its retriever pre-loading tasks initiated."
        )

        # Attempt to pre-load default rerankers (optional, good for responsiveness)
        if DEFAULT_RERANKER_MODEL:
            await get_cross_encoder_dependency(
                reranker_model_name=DEFAULT_RERANKER_MODEL,
                device_override=DEFAULT_DEVICE,
            )
            logger.info(
                f"Default cross-lingual reranker '{DEFAULT_RERANKER_MODEL}' pre-loading task initiated."
            )
        if DEFAULT_MONOLINGUAL_RERANKER_MODEL:
            await get_cross_encoder_dependency(
                reranker_model_name=DEFAULT_MONOLINGUAL_RERANKER_MODEL,
                device_override=DEFAULT_DEVICE,
            )
            logger.info(
                f"Default monolingual reranker '{DEFAULT_MONOLINGUAL_RERANKER_MODEL}' pre-loading task initiated."
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

# Configure CORS
# Adjust origins as needed for your frontend development and production
origins = [
    "http://localhost:8080",  # Default Vue CLI dev server
    "http://localhost:3000",  # Common React/Next.js dev server
    "http://localhost:5173",  # Vite default port
    "https://phentrieve.kidney-genetics.org",  # Production frontend URL
    # Add your production frontend URL when deployed (placeholder if others needed)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# This content was moved to the lifespan context manager


# Include routers
app.include_router(query_router.router, prefix="/api/v1/query", tags=["HPO Term Query"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health Check"])
app.include_router(
    similarity_router.router, prefix="/api/v1/similarity", tags=["HPO Term Similarity"]
)
app.include_router(config_info_router.router, prefix="/api/v1")
app.include_router(
    text_processing_router.router, tags=["Text Processing and HPO Extraction"]
)


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
