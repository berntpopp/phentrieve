"""
Tests for the configuration information API endpoint.

This file contains tests to ensure the Phentrieve API's configuration
and information endpoint is working correctly.
"""

import pytest
from fastapi.testclient import TestClient
from phentrieve import config as phentrieve_config

# Import the main FastAPI app
from api.main import app

client = TestClient(app)


def test_get_config_info():
    """Test the /api/v1/info endpoint."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()

    # Test response structure
    assert "available_embedding_models" in data
    assert "default_embedding_model" in data
    assert "available_reranker_models" in data
    assert "default_reranker_model" in data
    assert "available_monolingual_reranker_models" in data
    assert "default_monolingual_reranker_model" in data
    assert "default_parameters" in data
    assert "chunking_config" in data
    assert "hpo_data_status" in data

    # Test embedding models
    assert len(data["available_embedding_models"]) == len(
        phentrieve_config.BENCHMARK_MODELS
    )
    assert data["default_embedding_model"] == phentrieve_config.DEFAULT_MODEL

    # Test reranker models
    assert data["default_reranker_model"] == phentrieve_config.DEFAULT_RERANKER_MODEL
    assert (
        data["default_monolingual_reranker_model"]
        == phentrieve_config.DEFAULT_MONOLINGUAL_RERANKER_MODEL
    )

    # Test default parameters
    assert (
        data["default_parameters"]["similarity_threshold"]
        == phentrieve_config.DEFAULT_SIMILARITY_THRESHOLD
    )
    assert (
        data["default_parameters"]["reranker_mode"]
        == phentrieve_config.DEFAULT_RERANKER_MODE
    )
    assert data["default_parameters"]["top_k"] == phentrieve_config.DEFAULT_TOP_K
    assert (
        data["default_parameters"]["enable_reranker"]
        == phentrieve_config.DEFAULT_ENABLE_RERANKER
    )
    assert (
        data["default_parameters"]["rerank_candidate_count"]
        == phentrieve_config.DEFAULT_RERANK_CANDIDATE_COUNT
    )
    assert (
        data["default_parameters"]["similarity_formula"]
        == phentrieve_config.DEFAULT_SIMILARITY_FORMULA
    )
    assert data["default_parameters"]["language"] == phentrieve_config.DEFAULT_LANGUAGE

    # Test chunking config
    assert "available_strategies" in data["chunking_config"]
    assert "default_strategy" in data["chunking_config"]
    assert "sliding_window" in data["chunking_config"]["available_strategies"]

    # Test HPO data status
    assert "ancestors_loaded" in data["hpo_data_status"]
    assert "depths_loaded" in data["hpo_data_status"]
    assert isinstance(data["hpo_data_status"]["ancestors_loaded"], bool)
    assert isinstance(data["hpo_data_status"]["depths_loaded"], bool)


def test_get_config_info_model_details():
    """Test that model details in the response match expected format."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()

    # Check structure of embedding models
    for model in data["available_embedding_models"]:
        assert "id" in model
        assert "description" in model
        assert "is_default" in model
        assert isinstance(model["is_default"], bool)

        # Check that default model is marked correctly
        if model["id"] == phentrieve_config.DEFAULT_MODEL:
            assert model["is_default"] is True

    # Check structure of reranker models
    for model in data["available_reranker_models"]:
        assert "id" in model
        assert "description" in model
        assert "is_default" in model

        # Default reranker should be marked as default
        if model["id"] == phentrieve_config.DEFAULT_RERANKER_MODEL:
            assert model["is_default"] is True
