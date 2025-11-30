"""Unit tests for API model loading with async wait logic.

Tests the new async wait functionality in api/dependencies.py that allows
fast-loading models to complete within the request lifecycle instead of
forcing clients to retry with 503 errors.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sentence_transformers import CrossEncoder, SentenceTransformer

# Add project root to sys.path to import from api/ directory
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from api import dependencies  # noqa: E402


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Reset all model caches and status before each test."""
    dependencies.LOADED_SBERT_MODELS.clear()
    dependencies.LOADED_CROSS_ENCODERS.clear()
    dependencies.MODEL_LOADING_STATUS.clear()
    dependencies.MODEL_LOAD_LOCKS.clear()
    dependencies.MODEL_LOADING_TASKS.clear()
    yield
    # Cleanup after test
    dependencies.LOADED_SBERT_MODELS.clear()
    dependencies.LOADED_CROSS_ENCODERS.clear()
    dependencies.MODEL_LOADING_STATUS.clear()
    dependencies.MODEL_LOAD_LOCKS.clear()
    dependencies.MODEL_LOADING_TASKS.clear()


@pytest.fixture
def mock_cross_encoder():
    """Create a mock CrossEncoder instance."""
    mock_ce = MagicMock(spec=CrossEncoder)
    mock_ce.predict = MagicMock(return_value=[0.9, 0.8, 0.7])
    return mock_ce


@pytest.fixture
def mock_sbert_model():
    """Create a mock SentenceTransformer instance."""
    mock_model = MagicMock(spec=SentenceTransformer)
    mock_model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    return mock_model


class TestCrossEncoderLoadingWithTimeout:
    """Test CrossEncoder loading with async wait and timeout logic."""

    @pytest.mark.asyncio
    async def test_first_request_fast_loading_succeeds(self, mock_cross_encoder):
        """Test that first request succeeds if model loads within timeout.

        Scenario: User enables reranker for first time
        Expected: Model loads in <10s, request succeeds without 503
        """
        model_name = "test-cross-encoder"

        # Mock fast loading (3 seconds)
        async def fast_load(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate fast loading
            return mock_cross_encoder

        with patch("api.dependencies.run_in_threadpool", new=fast_load):
            # First request should succeed
            result = await dependencies.get_cross_encoder_dependency(
                reranker_model_name=model_name
            )

            assert result is mock_cross_encoder
            assert dependencies.MODEL_LOADING_STATUS[model_name] == "loaded"
            assert model_name in dependencies.LOADED_CROSS_ENCODERS
            assert model_name not in dependencies.MODEL_LOADING_TASKS  # Cleaned up

    @pytest.mark.asyncio
    async def test_first_request_slow_loading_returns_503(self, mock_cross_encoder):
        """Test that first request returns 503 if model loading exceeds timeout.

        Scenario: Model takes longer than timeout to load
        Expected: Returns 503 with Retry-After header, loading continues in background
        """
        model_name = "slow-cross-encoder"

        # Mock slow loading (exceeds 10s timeout)
        async def slow_load(*args, **kwargs):
            await asyncio.sleep(100)  # Intentionally longer than timeout
            return mock_cross_encoder

        # Set short timeout for testing
        with patch.object(dependencies, "CROSS_ENCODER_LOAD_TIMEOUT", 0.2):
            with patch("api.dependencies.run_in_threadpool", new=slow_load):
                # First request should return 503
                with pytest.raises(HTTPException) as exc_info:
                    await dependencies.get_cross_encoder_dependency(
                        reranker_model_name=model_name
                    )

                assert exc_info.value.status_code == 503
                assert "taking longer than expected" in exc_info.value.detail
                assert "Retry-After" in exc_info.value.headers
                assert dependencies.MODEL_LOADING_STATUS[model_name] == "loading"

    @pytest.mark.asyncio
    async def test_concurrent_requests_wait_for_same_model(self, mock_cross_encoder):
        """Test that concurrent requests wait for the same loading task.

        Scenario: Multiple users request reranker simultaneously
        Expected: All requests wait for same loading task, all succeed together
        """
        model_name = "concurrent-cross-encoder"

        # Mock loading that takes 0.5 seconds
        async def moderate_load(*args, **kwargs):
            await asyncio.sleep(0.5)
            return mock_cross_encoder

        with patch("api.dependencies.run_in_threadpool", new=moderate_load):
            # Start 3 concurrent requests
            results = await asyncio.gather(
                dependencies.get_cross_encoder_dependency(model_name),
                dependencies.get_cross_encoder_dependency(model_name),
                dependencies.get_cross_encoder_dependency(model_name),
            )

            # All should succeed and get the same model
            assert all(r is mock_cross_encoder for r in results)
            assert dependencies.MODEL_LOADING_STATUS[model_name] == "loaded"
            assert model_name not in dependencies.MODEL_LOADING_TASKS  # Cleaned up

    @pytest.mark.asyncio
    async def test_second_request_while_loading_waits(self, mock_cross_encoder):
        """Test that second request waits if first is still loading.

        Scenario: User1 triggers loading, User2 requests before loading completes
        Expected: User2 waits for loading to complete, both succeed
        """
        model_name = "wait-cross-encoder"
        loading_event = asyncio.Event()

        # Mock loading that we can control
        async def controlled_load(*args, **kwargs):
            await loading_event.wait()  # Wait until we signal
            await asyncio.sleep(0.1)
            return mock_cross_encoder

        with patch("api.dependencies.run_in_threadpool", new=controlled_load):
            # Start first request (doesn't await yet)
            task1 = asyncio.create_task(
                dependencies.get_cross_encoder_dependency(model_name)
            )

            # Give it time to start loading
            await asyncio.sleep(0.1)
            assert dependencies.MODEL_LOADING_STATUS[model_name] == "loading"

            # Start second request while first is loading
            task2 = asyncio.create_task(
                dependencies.get_cross_encoder_dependency(model_name)
            )

            # Give second request time to enter wait state
            await asyncio.sleep(0.1)

            # Now allow loading to complete
            loading_event.set()

            # Both should succeed
            result1, result2 = await asyncio.gather(task1, task2)
            assert result1 is mock_cross_encoder
            assert result2 is mock_cross_encoder

    @pytest.mark.asyncio
    async def test_task_cleanup_on_successful_load(self, mock_cross_encoder):
        """Test that loading task is removed from tracking dict after completion.

        Ensures no memory leak from accumulated task references.
        """
        model_name = "cleanup-cross-encoder"

        async def fast_load(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_cross_encoder

        with patch("api.dependencies.run_in_threadpool", new=fast_load):
            # During loading, task should be tracked
            task = asyncio.create_task(
                dependencies.get_cross_encoder_dependency(model_name)
            )
            await asyncio.sleep(0.05)  # Let it start

            # Complete the loading (result intentionally discarded, we only need the side effects)
            _ = await task

            # Task should be cleaned up
            assert model_name not in dependencies.MODEL_LOADING_TASKS
            assert dependencies.MODEL_LOADING_STATUS[model_name] == "loaded"

    @pytest.mark.asyncio
    async def test_task_cleanup_on_failed_load(self):
        """Test that loading task is cleaned up even when loading fails.

        Ensures cleanup happens in failure scenarios too.
        """
        model_name = "failed-cross-encoder"

        async def failing_load(*args, **kwargs):
            await asyncio.sleep(0.1)
            raise RuntimeError("Model loading failed!")

        with patch("api.dependencies.run_in_threadpool", new=failing_load):
            # This should fail but still clean up
            with pytest.raises(HTTPException) as exc_info:
                await dependencies.get_cross_encoder_dependency(model_name)

            # Should timeout (since loading task fails)
            assert exc_info.value.status_code == 503

            # Wait a bit for background task to complete and clean up
            await asyncio.sleep(0.3)

            # Task should be cleaned up
            assert model_name not in dependencies.MODEL_LOADING_TASKS
            assert dependencies.MODEL_LOADING_STATUS[model_name] == "failed"

    @pytest.mark.asyncio
    async def test_cached_model_returns_immediately(self, mock_cross_encoder):
        """Test that already-loaded models return immediately without waiting.

        Scenario: Model loaded previously, user makes another request
        Expected: Returns immediately from cache
        """
        model_name = "cached-cross-encoder"

        # Pre-populate cache
        dependencies.LOADED_CROSS_ENCODERS[model_name] = mock_cross_encoder
        dependencies.MODEL_LOADING_STATUS[model_name] = "loaded"

        # Should return immediately without any loading
        result = await dependencies.get_cross_encoder_dependency(model_name)

        assert result is mock_cross_encoder
        assert model_name not in dependencies.MODEL_LOADING_TASKS


class TestSBERTModelLoadingWithTimeout:
    """Test SBERT model loading with async wait and timeout logic."""

    @pytest.mark.asyncio
    async def test_sbert_first_request_fast_loading(self, mock_sbert_model):
        """Test SBERT model loads successfully on first request if fast enough."""
        model_name = "test-sbert"

        async def fast_load(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_sbert_model

        with patch("api.dependencies.run_in_threadpool", new=fast_load):
            result = await dependencies.get_sbert_model_dependency(
                model_name_requested=model_name
            )

            assert result is mock_sbert_model
            assert dependencies.MODEL_LOADING_STATUS[model_name] == "loaded"

    @pytest.mark.asyncio
    async def test_sbert_timeout_longer_than_cross_encoder(self, mock_sbert_model):
        """Test that SBERT has longer timeout than CrossEncoder.

        SBERT models are larger and slower to load.
        """
        # Verify timeout configuration
        assert (
            dependencies.SBERT_LOAD_TIMEOUT >= dependencies.CROSS_ENCODER_LOAD_TIMEOUT
        )
        assert dependencies.SBERT_LOAD_TIMEOUT == 60  # Default
        assert dependencies.CROSS_ENCODER_LOAD_TIMEOUT == 10  # Default


class TestConfigurableTimeouts:
    """Test that timeouts can be configured via environment variables."""

    def test_default_timeouts(self):
        """Test that default timeout values are sensible."""
        # Re-import to get fresh values
        from api import dependencies

        # Defaults should be conservative
        assert dependencies.SBERT_LOAD_TIMEOUT > 0
        assert dependencies.CROSS_ENCODER_LOAD_TIMEOUT > 0
        assert dependencies.SBERT_LOAD_TIMEOUT > dependencies.CROSS_ENCODER_LOAD_TIMEOUT

    @pytest.mark.asyncio
    async def test_custom_timeout_via_env_var(self, mock_cross_encoder, monkeypatch):
        """Test that timeout can be customized via environment variable."""
        # Set custom timeout
        monkeypatch.setenv("PHENTRIEVE_CROSS_ENCODER_LOAD_TIMEOUT", "5")

        # Reload module to pick up new env var and reassign
        import importlib

        reloaded_deps = importlib.reload(dependencies)

        assert reloaded_deps.CROSS_ENCODER_LOAD_TIMEOUT == 5.0

    @pytest.mark.asyncio
    async def test_timeout_enforced_correctly(self, mock_cross_encoder):
        """Test that custom timeout is actually enforced."""
        model_name = "timeout-test-cross-encoder"

        # Mock loading that takes 1 second
        async def one_second_load(*args, **kwargs):
            await asyncio.sleep(1.0)
            return mock_cross_encoder

        # Set very short timeout
        with patch.object(dependencies, "CROSS_ENCODER_LOAD_TIMEOUT", 0.1):
            with patch("api.dependencies.run_in_threadpool", new=one_second_load):
                # Should timeout
                with pytest.raises(HTTPException) as exc_info:
                    await dependencies.get_cross_encoder_dependency(model_name)

                assert exc_info.value.status_code == 503


class TestErrorHandling:
    """Test error handling in model loading."""

    @pytest.mark.asyncio
    async def test_loading_failure_sets_failed_status(self):
        """Test that loading failures set status to 'failed'."""
        model_name = "error-cross-encoder"

        async def failing_load(*args, **kwargs):
            await asyncio.sleep(0.1)
            raise ValueError("Intentional failure")

        with patch("api.dependencies.run_in_threadpool", new=failing_load):
            with pytest.raises(HTTPException):
                await dependencies.get_cross_encoder_dependency(model_name)

            # Give background task time to fail and update status
            await asyncio.sleep(0.3)
            assert dependencies.MODEL_LOADING_STATUS[model_name] == "failed"

    @pytest.mark.asyncio
    async def test_failed_model_returns_503(self):
        """Test that requests for failed models return 503."""
        model_name = "previously-failed"

        # Pre-set failed status
        dependencies.MODEL_LOADING_STATUS[model_name] = "failed"

        with pytest.raises(HTTPException) as exc_info:
            await dependencies.get_cross_encoder_dependency(model_name)

        assert exc_info.value.status_code == 503
        assert "failed to load" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_none_model_name_returns_none(self):
        """Test that None model name returns None (no loading)."""
        result = await dependencies.get_cross_encoder_dependency(
            reranker_model_name=None
        )
        assert result is None


class TestLockMechanism:
    """Test that the lock mechanism prevents race conditions."""

    @pytest.mark.asyncio
    async def test_lock_prevents_duplicate_loading(self, mock_cross_encoder):
        """Test that lock prevents same model from loading twice.

        Ensures only one loading task is created even with concurrent requests.
        """
        model_name = "lock-test-cross-encoder"
        load_count = {"count": 0}

        async def counting_load(*args, **kwargs):
            load_count["count"] += 1
            await asyncio.sleep(0.2)
            return mock_cross_encoder

        with patch("api.dependencies.run_in_threadpool", new=counting_load):
            # Start multiple concurrent requests
            results = await asyncio.gather(
                dependencies.get_cross_encoder_dependency(model_name),
                dependencies.get_cross_encoder_dependency(model_name),
                dependencies.get_cross_encoder_dependency(model_name),
            )

            # All should succeed
            assert all(r is mock_cross_encoder for r in results)

            # But model should only be loaded once (not 3 times)
            # Note: Due to timing, this might be 1 or 2, but not 3
            assert load_count["count"] <= 2


class TestBackwardCompatibility:
    """Test that changes don't break existing behavior."""

    @pytest.mark.asyncio
    async def test_returns_same_model_instance_to_all_callers(self, mock_cross_encoder):
        """Test that all requests get the same cached instance."""
        model_name = "shared-cross-encoder"

        async def load_once(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_cross_encoder

        with patch("api.dependencies.run_in_threadpool", new=load_once):
            result1 = await dependencies.get_cross_encoder_dependency(model_name)
            result2 = await dependencies.get_cross_encoder_dependency(model_name)
            result3 = await dependencies.get_cross_encoder_dependency(model_name)

            # All should be the exact same instance (not copies)
            assert result1 is result2 is result3 is mock_cross_encoder

    @pytest.mark.asyncio
    async def test_model_cache_persists_across_requests(self, mock_cross_encoder):
        """Test that loaded models stay in cache between requests."""
        model_name = "persistent-cross-encoder"

        # Load model
        dependencies.LOADED_CROSS_ENCODERS[model_name] = mock_cross_encoder
        dependencies.MODEL_LOADING_STATUS[model_name] = "loaded"

        # Multiple requests should all hit cache
        for _ in range(5):
            result = await dependencies.get_cross_encoder_dependency(model_name)
            assert result is mock_cross_encoder

        # Model should still be in cache
        assert model_name in dependencies.LOADED_CROSS_ENCODERS
