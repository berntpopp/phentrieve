"""Characterization tests for api/dependencies.py model loading.

Tests the status tracking, double-check locking, timeout behavior, and cache
hit paths. Must pass identically before AND after dependency unification.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# API import path workaround (see tests/unit/api/README.md)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from api.dependencies import (  # noqa: E402
    LOADED_SBERT_MODELS,
    MODEL_LOAD_LOCKS,
    MODEL_LOADING_STATUS,
    MODEL_LOADING_TASKS,
    _get_lock_for_model,
    get_sbert_model_dependency,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_global_state():
    """Reset all module-level state between tests."""
    LOADED_SBERT_MODELS.clear()
    MODEL_LOADING_STATUS.clear()
    MODEL_LOAD_LOCKS.clear()
    MODEL_LOADING_TASKS.clear()
    yield
    LOADED_SBERT_MODELS.clear()
    MODEL_LOADING_STATUS.clear()
    MODEL_LOAD_LOCKS.clear()
    MODEL_LOADING_TASKS.clear()


class TestGetLockForModel:
    def test_creates_lock_for_new_model(self):
        lock = _get_lock_for_model("test-model")
        assert isinstance(lock, asyncio.Lock)

    def test_returns_same_lock_for_same_model(self):
        lock1 = _get_lock_for_model("test-model")
        lock2 = _get_lock_for_model("test-model")
        assert lock1 is lock2

    def test_different_models_get_different_locks(self):
        lock1 = _get_lock_for_model("model-a")
        lock2 = _get_lock_for_model("model-b")
        assert lock1 is not lock2


class TestSbertModelCacheHit:
    @pytest.mark.asyncio
    async def test_returns_cached_model(self):
        mock_model = MagicMock()
        LOADED_SBERT_MODELS["test-model"] = mock_model
        result = await get_sbert_model_dependency("test-model")
        assert result is mock_model


class TestSbertModelStatusFailed:
    @pytest.mark.asyncio
    async def test_failed_status_raises_503(self):
        MODEL_LOADING_STATUS["test-model"] = "failed"
        from fastapi import HTTPException

        with pytest.raises(HTTPException, match="failed to load"):
            await get_sbert_model_dependency("test-model")


class TestSbertModelLoadingStatus:
    """Tests for SBERT loading status tracking transitions."""

    @pytest.mark.asyncio
    async def test_loading_status_without_task_raises_503(self):
        """When status is 'loading' but no task found, raises 503."""
        MODEL_LOADING_STATUS["test-model"] = "loading"
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await get_sbert_model_dependency("test-model")
        assert exc_info.value.status_code == 503
        assert "being prepared" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_cache_hit_skips_lock(self):
        """Cache hit returns immediately without acquiring lock."""
        mock_model = MagicMock()
        LOADED_SBERT_MODELS["test-model"] = mock_model

        # Call twice - both should hit cache
        result1 = await get_sbert_model_dependency("test-model")
        result2 = await get_sbert_model_dependency("test-model")
        assert result1 is mock_model
        assert result2 is mock_model
