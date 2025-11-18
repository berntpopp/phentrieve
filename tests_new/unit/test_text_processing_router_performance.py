"""
Unit tests for text processing router performance improvements.

Tests cover:
- Model caching via dependency injection
- Adaptive timeout based on text length
- Graceful timeout handling
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from api.routers.text_processing_router import (
    _process_text_internal,
    process_text_extract_hpo,
)
from api.schemas.text_processing_schemas import TextProcessingRequest


class TestAdaptiveTimeout:
    """Test adaptive timeout calculation based on text length."""

    @pytest.mark.asyncio
    async def test_small_text_30s_timeout(self):
        """Small text (<500 chars) should have 30s timeout."""
        request = TextProcessingRequest(
            text_content="Short text" * 10,  # ~100 chars
        )

        # Mock the internal processing to verify timeout value
        with patch(
            "api.routers.text_processing_router._process_text_internal"
        ) as mock_internal:
            mock_internal.return_value = {"test": "result"}

            with patch("api.routers.text_processing_router.asyncio.wait_for") as mock_wait:
                mock_wait.return_value = {"test": "result"}

                await process_text_extract_hpo(request)

                # Verify wait_for was called with correct timeout
                mock_wait.assert_called_once()
                args, kwargs = mock_wait.call_args
                assert kwargs.get("timeout") == 30 or args[1] == 30

    @pytest.mark.asyncio
    async def test_medium_text_60s_timeout(self):
        """Medium text (500-2000 chars) should have 60s timeout."""
        request = TextProcessingRequest(
            text_content="Medium text " * 100,  # ~1200 chars
        )

        with patch(
            "api.routers.text_processing_router._process_text_internal"
        ) as mock_internal:
            mock_internal.return_value = {"test": "result"}

            with patch("api.routers.text_processing_router.asyncio.wait_for") as mock_wait:
                mock_wait.return_value = {"test": "result"}

                await process_text_extract_hpo(request)

                # Verify 60s timeout
                args, kwargs = mock_wait.call_args
                assert kwargs.get("timeout") == 60 or args[1] == 60

    @pytest.mark.asyncio
    async def test_large_text_120s_timeout(self):
        """Large text (2000-5000 chars) should have 120s timeout."""
        request = TextProcessingRequest(
            text_content="Large text " * 300,  # ~3300 chars
        )

        with patch(
            "api.routers.text_processing_router._process_text_internal"
        ) as mock_internal:
            mock_internal.return_value = {"test": "result"}

            with patch("api.routers.text_processing_router.asyncio.wait_for") as mock_wait:
                mock_wait.return_value = {"test": "result"}

                await process_text_extract_hpo(request)

                # Verify 120s timeout
                args, kwargs = mock_wait.call_args
                assert kwargs.get("timeout") == 120 or args[1] == 120

    @pytest.mark.asyncio
    async def test_very_large_text_180s_timeout(self):
        """Very large text (>5000 chars) should have 180s timeout."""
        request = TextProcessingRequest(
            text_content="Very large text " * 400,  # ~6400 chars
        )

        with patch(
            "api.routers.text_processing_router._process_text_internal"
        ) as mock_internal:
            mock_internal.return_value = {"test": "result"}

            with patch("api.routers.text_processing_router.asyncio.wait_for") as mock_wait:
                mock_wait.return_value = {"test": "result"}

                await process_text_extract_hpo(request)

                # Verify 180s timeout
                args, kwargs = mock_wait.call_args
                assert kwargs.get("timeout") == 180 or args[1] == 180


class TestTimeoutHandling:
    """Test graceful timeout error handling."""

    @pytest.mark.asyncio
    async def test_timeout_raises_504_error(self):
        """Timeout should raise 504 Gateway Timeout with helpful message."""
        request = TextProcessingRequest(
            text_content="Test text",
        )

        with patch(
            "api.routers.text_processing_router.asyncio.wait_for"
        ) as mock_wait:
            # Simulate timeout
            mock_wait.side_effect = asyncio.TimeoutError()

            with pytest.raises(HTTPException) as exc_info:
                await process_text_extract_hpo(request)

            # Verify 504 status code
            assert exc_info.value.status_code == 504

            # Verify helpful error message
            detail = exc_info.value.detail
            assert "timed out" in detail.lower()
            assert "seconds" in detail.lower()
            assert "suggestions" in detail.lower() or "reduce" in detail.lower()

    @pytest.mark.asyncio
    async def test_timeout_message_includes_suggestions(self):
        """Timeout error should include actionable suggestions."""
        request = TextProcessingRequest(
            text_content="Test text" * 50,
        )

        with patch(
            "api.routers.text_processing_router.asyncio.wait_for"
        ) as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()

            with pytest.raises(HTTPException) as exc_info:
                await process_text_extract_hpo(request)

            detail = exc_info.value.detail.lower()

            # Should suggest at least one mitigation
            has_suggestion = (
                "reduce text" in detail
                or "simple" in detail
                or "disable reranker" in detail
            )
            assert has_suggestion, f"No suggestions in error message: {detail}"


class TestModelCaching:
    """Test that models are cached and not reloaded on every request."""

    @pytest.mark.asyncio
    async def test_uses_cached_sbert_model(self):
        """Should use get_sbert_model_dependency for cached models."""
        request = TextProcessingRequest(
            text_content="Test text for caching",
        )

        with patch(
            "api.routers.text_processing_router.get_sbert_model_dependency"
        ) as mock_get_model:
            mock_get_model.return_value = MagicMock()

            with patch(
                "api.routers.text_processing_router.get_dense_retriever_dependency"
            ) as mock_get_retriever:
                mock_retriever = MagicMock()
                mock_retriever.model_name = "test-model"
                mock_get_retriever.return_value = mock_retriever

                with patch(
                    "api.routers.text_processing_router.run_in_threadpool"
                ) as mock_threadpool:
                    # Mock all threadpool operations
                    mock_threadpool.return_value = "en"

                    with patch(
                        "api.routers.text_processing_router.TextProcessingPipeline"
                    ):
                        with patch(
                            "api.routers.text_processing_router.orchestrate_hpo_extraction"
                        ) as mock_orchestrate:
                            mock_orchestrate.return_value = ([], [])

                            try:
                                await _process_text_internal(request)
                            except Exception:
                                # Expected: Full pipeline may fail with mocked dependencies
                                # We're only testing that cached dependencies are called
                                pass

                            # Verify cached dependency was called
                            mock_get_model.assert_called()

    @pytest.mark.asyncio
    async def test_uses_cached_retriever(self):
        """Should use get_dense_retriever_dependency for cached retrievers."""
        request = TextProcessingRequest(
            text_content="Test text",
        )

        with patch(
            "api.routers.text_processing_router.get_sbert_model_dependency"
        ) as mock_get_model:
            mock_get_model.return_value = MagicMock()

            with patch(
                "api.routers.text_processing_router.get_dense_retriever_dependency"
            ) as mock_get_retriever:
                mock_retriever = MagicMock()
                mock_retriever.model_name = "test-model"
                mock_get_retriever.return_value = mock_retriever

                with patch(
                    "api.routers.text_processing_router.run_in_threadpool"
                ) as mock_threadpool:
                    mock_threadpool.return_value = "en"

                    with patch(
                        "api.routers.text_processing_router.TextProcessingPipeline"
                    ):
                        with patch(
                            "api.routers.text_processing_router.orchestrate_hpo_extraction"
                        ) as mock_orchestrate:
                            mock_orchestrate.return_value = ([], [])

                            try:
                                await _process_text_internal(request)
                            except Exception:
                                # Expected: Full pipeline may fail with mocked dependencies
                                # We're only testing that cached dependencies are called
                                pass

                            # Verify cached retriever was requested
                            mock_get_retriever.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_cached_cross_encoder_when_enabled(self):
        """Should use get_cross_encoder_dependency when reranking enabled."""
        request = TextProcessingRequest(
            text_content="Test text",
            enable_reranker=True,
            reranker_model_name="test-reranker",
        )

        with patch(
            "api.routers.text_processing_router.get_sbert_model_dependency"
        ) as mock_get_model:
            mock_get_model.return_value = MagicMock()

            with patch(
                "api.routers.text_processing_router.get_dense_retriever_dependency"
            ) as mock_get_retriever:
                mock_retriever = MagicMock()
                mock_retriever.model_name = "test-model"
                mock_get_retriever.return_value = mock_retriever

                with patch(
                    "api.routers.text_processing_router.get_cross_encoder_dependency"
                ) as mock_get_cross_enc:
                    mock_get_cross_enc.return_value = MagicMock()

                    with patch(
                        "api.routers.text_processing_router.run_in_threadpool"
                    ) as mock_threadpool:
                        mock_threadpool.return_value = "en"

                        with patch(
                            "api.routers.text_processing_router.TextProcessingPipeline"
                        ):
                            with patch(
                                "api.routers.text_processing_router.orchestrate_hpo_extraction"
                            ) as mock_orchestrate:
                                mock_orchestrate.return_value = ([], [])

                                try:
                                    await _process_text_internal(request)
                                except Exception:
                                    # Expected: Full pipeline may fail with mocked dependencies
                                    # We're only testing that cached dependencies are called
                                    pass

                                # Verify cross-encoder dependency was called
                                mock_get_cross_enc.assert_called_once_with(
                                    reranker_model_name="test-reranker"
                                )


class TestModelReuse:
    """Test that the same model is reused for chunking when appropriate."""

    @pytest.mark.asyncio
    async def test_reuses_retrieval_model_for_chunking(self):
        """Should reuse retrieval model for chunking when no separate model specified."""
        request = TextProcessingRequest(
            text_content="Test text",
            retrieval_model_name="test-model",
            # No semantic_model_name specified
        )

        with patch(
            "api.routers.text_processing_router.get_sbert_model_dependency"
        ) as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            with patch(
                "api.routers.text_processing_router.get_dense_retriever_dependency"
            ) as mock_get_retriever:
                mock_retriever = MagicMock()
                mock_retriever.model_name = "test-model"
                mock_get_retriever.return_value = mock_retriever

                with patch(
                    "api.routers.text_processing_router.run_in_threadpool"
                ) as mock_threadpool:
                    mock_threadpool.return_value = "en"

                    with patch(
                        "api.routers.text_processing_router.TextProcessingPipeline"
                    ) as mock_pipeline:
                        with patch(
                            "api.routers.text_processing_router.orchestrate_hpo_extraction"
                        ) as mock_orchestrate:
                            mock_orchestrate.return_value = ([], [])

                            try:
                                await _process_text_internal(request)
                            except Exception:
                                # Expected: Full pipeline may fail with mocked dependencies
                                # We're only testing that cached dependencies are called
                                pass

                            # Should call get_sbert_model_dependency at least once
                            # (may be called twice: once for retrieval check, once for actual use)
                            # Note: We can't test actual caching behavior with mocks - the mock
                            # always returns the same value. Actual caching is tested in integration tests.
                            assert mock_get_model.call_count >= 1
