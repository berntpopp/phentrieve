import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routers import query_router, health
from api.dependencies import (
    get_sbert_model_dependency,
    get_dense_retriever_dependency,
    get_cross_encoder_dependency,
)
from phentrieve.config import (
    DEFAULT_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_DEVICE,
)

logger = logging.getLogger(__name__)
# Configure logging for the API
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Phentrieve API", version="0.1.0")

# Configure CORS
# Adjust origins as needed for your frontend development and production
origins = [
    "http://localhost:8080",  # Default Vue CLI dev server
    "http://localhost:3000",  # Common React/Next.js dev server
    "http://localhost:5173",  # Vite default port
    # Add your production frontend URL when deployed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Phentrieve API starting up. Pre-loading default models...")
    try:
        # Pre-load default SBERT model for retrieval
        sbert_retrieval_model = await get_sbert_model_dependency(
            model_name_requested=DEFAULT_MODEL, device_override=DEFAULT_DEVICE
        )
        # Pre-load the retriever associated with the default SBERT model
        await get_dense_retriever_dependency(
            sbert_model_name_for_retriever=DEFAULT_MODEL
        )
        logger.info(
            f"Default SBERT model '{DEFAULT_MODEL}' and its retriever pre-loaded."
        )

        # Attempt to pre-load default rerankers (optional, good for responsiveness)
        if DEFAULT_RERANKER_MODEL:
            await get_cross_encoder_dependency(
                reranker_model_name=DEFAULT_RERANKER_MODEL,
                device_override=DEFAULT_DEVICE,
            )
            logger.info(
                f"Default cross-lingual reranker '{DEFAULT_RERANKER_MODEL}' pre-loading attempted."
            )
        if DEFAULT_MONOLINGUAL_RERANKER_MODEL:
            await get_cross_encoder_dependency(
                reranker_model_name=DEFAULT_MONOLINGUAL_RERANKER_MODEL,
                device_override=DEFAULT_DEVICE,
            )
            logger.info(
                f"Default monolingual reranker '{DEFAULT_MONOLINGUAL_RERANKER_MODEL}' pre-loading attempted."
            )

    except Exception as e:
        logger.error(f"Error during API startup model pre-loading: {e}", exc_info=True)
    logger.info("API startup model pre-loading complete (or attempted).")


# Include routers
app.include_router(query_router.router, prefix="/api/v1/query", tags=["HPO Term Query"])
app.include_router(health.router, prefix="/api/v1", tags=["Health Check"])
# Add other routers here for text_processing if created later


@app.get("/")
async def root():
    return {"message": "Welcome to Phentrieve API. See /docs for details."}
