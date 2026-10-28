"""Characterization tests for api/dependencies.py model loading.

Tests the status tracking, double-check locking, timeout behavior, and cache
hit paths. Must pass identically before AND after dependency unification.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

import api.dependencies as dependencies
from api.dependencies import (
    LOADED_SBERT_MODELS,
    MODEL_LOAD_FAILED_AT,
    MODEL_LOAD_LOCKS,
    MODEL_LOADING_STATUS,
    MODEL_LOADING_TASKS,
    _get_lock_for_model,
    cleanup_model_caches,
    get_sbert_model_dependency,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_global_state():
    """Reset all module-level state between tests."""
    LOADED_SBERT_MODELS.clear()
    MODEL_LOADING_STATUS.clear()
    MODEL_LOAD_FAILED_AT.clear()
    MODEL_LOAD_LOCKS.clear()
    MODEL_LOADING_TASKS.clear()
    yield
    LOADED_SBERT_MODELS.clear()
    MODEL_LOADING_STATUS.clear()
    MODEL_LOAD_FAILED_AT.clear()
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
    async def test_recent_failure_raises_503_with_retry_after(self):
        MODEL_LOADING_STATUS["test-model"] = "failed"
        MODEL_LOAD_FAILED_AT["test-model"] = time.monotonic()
        from fastapi import HTTPException

        with pytest.raises(HTTPException, match="failed to load") as exc_info:
            await get_sbert_model_dependency("test-model")
        assert exc_info.value.status_code == 503
        assert exc_info.value.headers is not None
        assert int(exc_info.value.headers["Retry-After"]) >= 1

    @pytest.mark.asyncio
    async def test_failure_is_retried_after_cooldown(self, monkeypatch):
        """A failed load must not disable the model until restart (#416)."""
        MODEL_LOADING_STATUS["test-model"] = "failed"
        MODEL_LOAD_FAILED_AT["test-model"] = (
            time.monotonic() - dependencies.MODEL_LOAD_RETRY_COOLDOWN_SECONDS - 1
        )
        mock_model = MagicMock()
        load_calls: list[str] = []

        def fake_load_embedding_model(*, model_name, **_kwargs):
            load_calls.append(model_name)
            return mock_model

        monkeypatch.setattr(
            dependencies, "load_embedding_model", fake_load_embedding_model
        )

        result = await get_sbert_model_dependency("test-model")

        assert result is mock_model
        assert load_calls == ["test-model"]
        assert MODEL_LOADING_STATUS["test-model"] == "loaded"
        assert "test-model" not in MODEL_LOAD_FAILED_AT

    @pytest.mark.asyncio
    async def test_failed_load_records_failure_time(self, monkeypatch):
        def failing_load(**_kwargs):
            raise OSError("download failed")

        monkeypatch.setattr(dependencies, "load_embedding_model", failing_load)
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            await get_sbert_model_dependency("test-model")
        assert MODEL_LOADING_STATUS["test-model"] == "failed"
        assert "test-model" in MODEL_LOAD_FAILED_AT


class TestCleanupModelCaches:
    @pytest.mark.asyncio
    async def test_cleanup_clears_embedding_registry(self, monkeypatch):
        """Models must actually be released, not only dropped from API caches."""
        calls: list[bool] = []
        monkeypatch.setattr(
            dependencies, "clear_model_registry", lambda: calls.append(True)
        )
        LOADED_SBERT_MODELS["test-model"] = MagicMock()
        MODEL_LOAD_FAILED_AT["other-model"] = 0.0

        await cleanup_model_caches()

        assert calls == [True]
        assert not LOADED_SBERT_MODELS
        assert not MODEL_LOAD_FAILED_AT


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
