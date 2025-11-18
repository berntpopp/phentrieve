"""Unit tests for embedding model caching functionality.

Tests the thread-safe model registry to ensure:
1. Models are cached and reused correctly
2. Thread safety works properly
3. force_reload bypasses cache
4. clear_model_registry works
5. Device switching works correctly
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from phentrieve.embeddings import (
    clear_model_registry,
    get_cached_models,
    load_embedding_model,
)


class TestModelCaching:
    """Tests for model caching behavior."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_model_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_model_registry()

    def test_model_loaded_once_per_name(self):
        """Verify same model name returns cached instance."""
        # Mock SentenceTransformer to avoid actual model loading
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.to.return_value = mock_model
            mock_st.return_value = mock_model

            # Load model twice
            model1 = load_embedding_model("test-model")
            model2 = load_embedding_model("test-model")

            # Should be the exact same object
            assert model1 is model2

            # SentenceTransformer should only be instantiated once
            assert mock_st.call_count == 1

    def test_force_reload_bypasses_cache(self):
        """Verify force_reload creates new instance."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.to.return_value = mock_model
            mock_st.return_value = mock_model

            # Load model normally
            model1 = load_embedding_model("test-model")

            # Load with force_reload
            mock_st.return_value = MagicMock()  # Return different mock
            mock_st.return_value.device = "cpu"
            mock_st.return_value.to.return_value = mock_st.return_value
            model2 = load_embedding_model("test-model", force_reload=True)

            # Should be different objects
            assert model1 is not model2

            # SentenceTransformer should be instantiated twice
            assert mock_st.call_count == 2

    def test_different_models_not_cached_together(self):
        """Verify different model names get separate instances."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model_a = MagicMock()
            mock_model_a.device = "cpu"
            mock_model_a.to.return_value = mock_model_a

            mock_model_b = MagicMock()
            mock_model_b.device = "cpu"
            mock_model_b.to.return_value = mock_model_b

            mock_st.side_effect = [mock_model_a, mock_model_b]

            # Load two different models
            model_a = load_embedding_model("model-a")
            model_b = load_embedding_model("model-b")

            # Should be different objects
            assert model_a is not model_b

            # Both should be instantiated
            assert mock_st.call_count == 2

    def test_clear_registry_removes_models(self):
        """Verify clear_model_registry removes cached models."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.to.return_value = mock_model
            mock_st.return_value = mock_model

            # Load model
            load_embedding_model("test-model")
            assert len(get_cached_models()) == 1

            # Clear registry
            clear_model_registry()
            assert len(get_cached_models()) == 0

            # Next load should create new instance
            mock_st.return_value = MagicMock()
            mock_st.return_value.device = "cpu"
            mock_st.return_value.to.return_value = mock_st.return_value
            load_embedding_model("test-model")

            # Should have been instantiated twice total
            assert mock_st.call_count == 2

    def test_get_cached_models_returns_correct_list(self):
        """Verify get_cached_models returns list of cached model names."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.to.return_value = mock_model
            mock_st.return_value = mock_model

            # Initially empty
            assert get_cached_models() == []

            # Load models
            load_embedding_model("model-a")
            assert "model-a" in get_cached_models()

            load_embedding_model("model-b")
            cached = get_cached_models()
            assert len(cached) == 2
            assert "model-a" in cached
            assert "model-b" in cached

    def test_device_parameter_used(self):
        """Verify device parameter is passed correctly."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.to.return_value = mock_model
            mock_st.return_value = mock_model

            # Load with specific device
            load_embedding_model("test-model", device="cpu")

            # Model should be moved to CPU
            mock_model.to.assert_called_with("cpu")

    def test_mps_device_detection(self):
        """Verify MPS device is detected when available."""
        with (
            patch("phentrieve.embeddings.SentenceTransformer") as mock_st,
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_built", return_value=True),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            mock_model = MagicMock()
            mock_model.device = "mps"
            mock_model.to.return_value = mock_model
            mock_st.return_value = mock_model

            # Load without specifying device (should auto-detect MPS)
            load_embedding_model("test-model")

            # Model should be moved to MPS
            mock_model.to.assert_called_with("mps")

    def test_default_model_used_when_none(self):
        """Verify DEFAULT_BIOLORD_MODEL is used when model_name is None."""
        from phentrieve.config import DEFAULT_BIOLORD_MODEL

        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.to.return_value = mock_model
            mock_st.return_value = mock_model

            # Load with None model_name
            load_embedding_model(model_name=None)

            # Should have loaded default model
            assert DEFAULT_BIOLORD_MODEL in get_cached_models()


class TestThreadSafety:
    """Tests for thread safety of model caching."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_model_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_model_registry()

    def test_concurrent_loads_return_same_instance(self):
        """Verify concurrent loads don't create duplicates."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            # Simulate slow model loading
            def slow_load(*args, **kwargs):
                time.sleep(0.1)
                model = MagicMock()
                model.device = "cpu"
                model.to.return_value = model
                return model

            mock_st.side_effect = slow_load

            models = []
            errors = []

            def load_model():
                try:
                    m = load_embedding_model("test-model")
                    models.append(m)
                except Exception as e:
                    errors.append(e)

            # Launch 10 concurrent threads
            threads = [threading.Thread(target=load_model) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # No errors should have occurred
            assert len(errors) == 0

            # All threads should get a model
            assert len(models) == 10

            # All threads should get the same instance
            assert all(m is models[0] for m in models)

            # SentenceTransformer should only be instantiated once
            assert mock_st.call_count == 1

    def test_concurrent_different_models(self):
        """Verify concurrent loads of different models work correctly."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:

            def create_mock_model(*args, **kwargs):
                model = MagicMock()
                model.device = "cpu"
                model.to.return_value = model
                return model

            mock_st.side_effect = create_mock_model

            models_a = []
            models_b = []

            def load_model_a():
                m = load_embedding_model("model-a")
                models_a.append(m)

            def load_model_b():
                m = load_embedding_model("model-b")
                models_b.append(m)

            # Launch threads for both models
            threads = [threading.Thread(target=load_model_a) for _ in range(5)]
            threads.extend([threading.Thread(target=load_model_b) for _ in range(5)])

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should succeed
            assert len(models_a) == 5
            assert len(models_b) == 5

            # All instances of model_a should be the same
            assert all(m is models_a[0] for m in models_a)

            # All instances of model_b should be the same
            assert all(m is models_b[0] for m in models_b)

            # model_a and model_b should be different
            assert models_a[0] is not models_b[0]

    def test_clear_registry_thread_safe(self):
        """Verify clear_model_registry is thread-safe."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.to.return_value = mock_model
            mock_st.return_value = mock_model

            # Load a model
            load_embedding_model("test-model")

            results = []

            def clear_and_check():
                clear_model_registry()
                cached = get_cached_models()
                results.append(len(cached))

            # Clear registry from multiple threads
            threads = [threading.Thread(target=clear_and_check) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All clears should succeed (registry should be empty)
            assert all(r == 0 for r in results)


class TestDeviceSwitching:
    """Tests for device switching behavior."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_model_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_model_registry()

    def test_cached_model_moved_to_different_device(self):
        """Verify cached model can be moved to different device."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.device = "cpu"

            def mock_to(device):
                mock_model.device = device
                return mock_model

            mock_model.to.side_effect = mock_to
            mock_st.return_value = mock_model

            # Load on CPU
            model1 = load_embedding_model("test-model", device="cpu")
            assert str(model1.device) == "cpu"

            # Load on "cuda" (same model, different device)
            model2 = load_embedding_model("test-model", device="cuda")

            # Should be same cached instance
            assert model1 is model2

            # Device should have been updated
            assert str(model2.device) == "cuda"

            # to() should have been called for device switch
            assert mock_model.to.call_count >= 2  # Initial + device switch

    def test_device_switch_not_called_if_same_device(self):
        """Verify .to() not called unnecessarily when device matches."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.to.return_value = mock_model
            mock_st.return_value = mock_model

            # Load on CPU
            model1 = load_embedding_model("test-model", device="cpu")
            to_call_count_after_first_load = mock_model.to.call_count

            # Load again on CPU (no device switch needed)
            model2 = load_embedding_model("test-model", device="cpu")

            # Should be same instance
            assert model1 is model2

            # to() should not have been called again (device already matches)
            assert mock_model.to.call_count == to_call_count_after_first_load


class TestErrorHandling:
    """Tests for error handling in model loading."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_model_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_model_registry()

    def test_loading_error_raises_value_error(self):
        """Verify loading errors are properly raised."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            mock_st.side_effect = Exception("Model not found")

            # Should raise ValueError with descriptive message
            with pytest.raises(
                ValueError, match="Error loading SentenceTransformer model"
            ):
                load_embedding_model("nonexistent-model")

    def test_failed_load_not_cached(self):
        """Verify failed loads don't cache broken models."""
        with patch("phentrieve.embeddings.SentenceTransformer") as mock_st:
            # First call fails
            mock_st.side_effect = Exception("Model not found")

            with pytest.raises(ValueError):
                load_embedding_model("test-model")

            # Model should not be in cache
            assert "test-model" not in get_cached_models()

            # Second call should attempt to load again
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.to.return_value = mock_model
            mock_st.side_effect = None
            mock_st.return_value = mock_model

            # This should succeed
            model = load_embedding_model("test-model")
            assert model is not None
            assert "test-model" in get_cached_models()
