"""Unit tests for text processing router helper functions."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.llm_quota import DailyQuotaStore, hash_subject_key
from api.main import app
from api.routers.text_processing_router import (
    QuotaExceededError,
    _get_chunking_config_for_api,
    _get_trust_remote_code_for_model,
    _prepare_standard_request_context,
    _process_text_via_shared_service,
    _validate_response_chunk_references,
)
from api.schemas.text_processing_schemas import (
    AggregatedHPOTermAPI,
    ProcessedChunkAPI,
    TextAttributionSpanAPI,
    TextProcessingRequest,
)
from phentrieve.config import BENCHMARK_MODELS, DEFAULT_MODEL

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_text_processing_router_returns_llm_meta(client, monkeypatch):
    monkeypatch.setattr(
        "api.routers.text_processing_router.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": "gpt-5.4-mini",
                "llm_mode": "two_phase",
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Patient had recurrent seizures.",
            "extraction_backend": "llm",
            "llm_model": "gpt-5.4-mini",
            "llm_mode": "two_phase",
        },
    )

    assert response.status_code == 200
    assert response.json()["meta"]["extraction_backend"] == "llm"


def test_text_processing_router_rejects_llm_without_model(client):
    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Patient had recurrent seizures.",
            "extraction_backend": "llm",
        },
    )

    assert response.status_code == 422
    assert "llm_model" in response.text


def test_text_processing_router_returns_429_when_quota_exhausted(client, monkeypatch):
    monkeypatch.setattr(
        "api.config.PHENTRIEVE_ENV",
        "production",
        raising=False,
    )

    def raise_quota(*args, **kwargs):
        raise QuotaExceededError(
            quota_used=3,
            quota_limit=3,
            quota_remaining=0,
            usage_date_utc="2026-04-15",
        )

    monkeypatch.setattr(
        "api.routers.text_processing_router.check_llm_quota_or_raise",
        raise_quota,
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "note",
            "extraction_backend": "llm",
            "llm_model": "gpt-5.4-mini",
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"]["quota_remaining"] == 0


def test_text_processing_router_returns_503_when_subject_resolution_is_untrusted(
    client, monkeypatch
):
    monkeypatch.setattr(
        "api.config.PHENTRIEVE_ENV",
        "production",
        raising=False,
    )
    monkeypatch.setattr(
        "api.routers.text_processing_router.resolve_subject_ip",
        lambda **kwargs: None,
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "note",
            "extraction_backend": "llm",
            "llm_model": "gpt-5.4-mini",
        },
    )

    assert response.status_code == 503
    assert "trusted anonymous subject" in response.json()["detail"]


def test_text_processing_router_counts_successes_and_skips_failed_requests(
    tmp_path, monkeypatch
):
    quota_db = tmp_path / "llm_quota.db"
    monkeypatch.setattr(
        "api.config.PHENTRIEVE_ENV",
        "production",
        raising=False,
    )
    monkeypatch.setattr(
        "api.config.PHENTRIEVE_TRUSTED_PROXY_CIDRS",
        "172.16.0.0/12",
        raising=False,
    )
    monkeypatch.setattr(
        "api.config.PHENTRIEVE_LLM_DAILY_LIMIT",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        "api.config.PHENTRIEVE_LLM_QUOTA_DB_PATH",
        str(quota_db),
        raising=False,
    )

    service_state = {"mode": "success"}

    def run_service(**kwargs):
        if service_state["mode"] == "fail":
            raise RuntimeError("llm backend failed")
        return {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": kwargs.get("llm_model", "gpt-5.4-mini"),
                "llm_mode": kwargs.get("llm_mode", "two_phase"),
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }

    monkeypatch.setattr(
        "api.routers.text_processing_router.run_full_text_service",
        run_service,
    )

    store = DailyQuotaStore(quota_db, daily_limit=2)
    headers = {"X-Forwarded-For": "203.0.113.5"}
    payload = {
        "text": "note",
        "extraction_backend": "llm",
        "llm_model": "gpt-5.4-mini",
    }
    subject_key = hash_subject_key("203.0.113.5")
    usage_date_utc = datetime.now(UTC).date().isoformat()

    with TestClient(
        app,
        raise_server_exceptions=False,
        client=("172.18.0.10", 50000),
    ) as local_client:
        first = local_client.post("/api/v1/text/process", json=payload, headers=headers)
        assert first.status_code == 200
        assert first.json()["meta"]["quota_limit"] == 2
        assert first.json()["meta"]["quota_remaining"] == 1
        assert (
            store.get_status(
                subject_key=subject_key,
                usage_date_utc=usage_date_utc,
            ).quota_used
            == 1
        )

        service_state["mode"] = "fail"
        failed = local_client.post(
            "/api/v1/text/process",
            json=payload,
            headers=headers,
        )
        assert failed.status_code == 500
        assert (
            store.get_status(
                subject_key=subject_key,
                usage_date_utc=usage_date_utc,
            ).quota_used
            == 1
        )

        service_state["mode"] = "success"
        second = local_client.post(
            "/api/v1/text/process",
            json=payload,
            headers=headers,
        )
        assert second.status_code == 200
        assert second.json()["meta"]["quota_limit"] == 2
        assert second.json()["meta"]["quota_remaining"] == 0
        assert (
            store.get_status(
                subject_key=subject_key,
                usage_date_utc=usage_date_utc,
            ).quota_used
            == 2
        )

        exhausted = local_client.post(
            "/api/v1/text/process",
            json=payload,
            headers=headers,
        )
        assert exhausted.status_code == 429
        assert exhausted.json()["detail"]["quota_remaining"] == 0


def test_text_processing_router_returns_standard_extraction_backend_contract(
    client, monkeypatch
):
    monkeypatch.setattr(
        "api.routers.text_processing_router.get_sbert_model_dependency",
        AsyncMock(return_value=MagicMock(model_name="FremyCompany/BioLORD-2023-M")),
    )
    monkeypatch.setattr(
        "api.routers.text_processing_router.get_dense_retriever_dependency",
        AsyncMock(return_value=MagicMock(model_name="FremyCompany/BioLORD-2023-M")),
    )
    monkeypatch.setattr(
        "api.routers.text_processing_router.run_full_text_service",
        lambda **kwargs: {
            "meta": {"extraction_backend": "standard"},
            "processed_chunks": [
                {
                    "chunk_id": 1,
                    "text": "Patient had recurrent seizures.",
                    "status": "affirmed",
                    "hpo_matches": [
                        {"id": "HP:0001250", "name": "Seizure", "score": 0.91}
                    ],
                    "start_char": 0,
                    "end_char": 31,
                }
            ],
            "aggregated_hpo_terms": [
                {
                    "id": "HP:0001250",
                    "name": "Seizure",
                    "confidence": 0.91,
                    "status": "affirmed",
                    "evidence_count": 1,
                    "chunks": [0],
                    "text_attributions": [
                        {
                            "chunk_idx": 0,
                            "start_char": 15,
                            "end_char": 24,
                            "matched_text_in_chunk": "seizures",
                        }
                    ],
                    "score": 0.91,
                }
            ],
        },
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text_content": "Patient had recurrent seizures.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["extraction_backend"] == "standard"
    assert body["meta"]["effective_language"] == "en"
    assert body["meta"]["effective_retrieval_model"] == "FremyCompany/BioLORD-2023-M"
    assert (
        body["meta"]["request_parameters"]["text"] == "Patient had recurrent seizures."
    )
    assert body["meta"]["num_processed_chunks"] == 1
    assert body["processed_chunks"][0]["hpo_matches"][0]["hpo_id"] == "HP:0001250"
    assert body["aggregated_hpo_terms"][0]["hpo_id"] == "HP:0001250"


class TestGetChunkingConfigForApi:
    """Test _get_chunking_config_for_api function."""

    @pytest.mark.parametrize(
        "strategy_name",
        [
            "simple",
            "semantic",
            "detailed",
            "sliding_window",
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_all_strategies_return_valid_config(self, strategy_name: str):
        """Test all strategies return valid configuration."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy=strategy_name,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - config should be a non-empty list of dicts
        assert isinstance(config, list)
        assert len(config) > 0
        assert all(isinstance(c, dict) for c in config)

    def test_unknown_strategy_uses_default(self):
        """Test unknown strategy falls back to default."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="unknown_nonexistent_strategy",
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should return valid config (default)
        assert isinstance(config, list)
        assert len(config) > 0

    def test_custom_sliding_window_parameters_applied(self):
        """Test custom sliding window parameters are applied correctly."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="semantic",
            window_size=15,
            step_size=3,
            split_threshold=0.8,
            min_segment_length=10,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - find sliding_window component
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        assert sw_component is not None

        sw_config = sw_component["config"]
        assert sw_config["window_size_tokens"] == 15
        assert sw_config["step_size_tokens"] == 3
        assert sw_config["splitting_threshold"] == 0.8
        assert sw_config["min_split_segment_length_words"] == 10

    def test_default_parameters_when_none_provided(self):
        """Test default parameters used when not specified in request."""
        # Arrange - Note: TextProcessingRequest has its own defaults
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="semantic",
            # Request defaults: ws=2, ss=1, th=0.3, msl=1
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should have sliding_window component with applied params
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        if sw_component:  # May not exist for all strategies
            assert "window_size_tokens" in sw_component["config"]
            assert "step_size_tokens" in sw_component["config"]

    def test_simple_strategy_returns_without_modification(self):
        """Test simple strategy returns config without sliding window params."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="simple",
            window_size=100,  # Should be ignored for simple strategy
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should return valid config
        assert isinstance(config, list)
        assert len(config) > 0

    def test_sliding_window_strategy_uses_params_directly(self):
        """Test sliding_window strategy passes params to config function."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="sliding_window",
            window_size=20,
            step_size=5,
            split_threshold=0.6,
            min_segment_length=8,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - params should be applied via config function
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        assert sw_component is not None

        sw_config = sw_component["config"]
        assert sw_config["window_size_tokens"] == 20
        assert sw_config["step_size_tokens"] == 5
        assert sw_config["splitting_threshold"] == 0.6
        assert sw_config["min_split_segment_length_words"] == 8

    def test_case_insensitive_strategy_name(self):
        """Test strategy name is case-insensitive."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="SEMANTIC",  # Uppercase
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should work same as lowercase
        assert isinstance(config, list)
        assert len(config) > 0

    @pytest.mark.parametrize(
        "strategy",
        [
            "semantic",
            "detailed",
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_all_applicable_strategies_receive_params(self, strategy: str):
        """Test all strategies that need params receive them."""
        # Arrange
        custom_ws = 25
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy=strategy,
            window_size=custom_ws,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should have sliding_window with custom param
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        if sw_component:  # Some strategies may not have sliding_window
            assert sw_component["config"]["window_size_tokens"] == custom_ws


class TestTextProcessingModelValidation:
    """Test model allowlist and request-path value preservation."""

    SUPPORTED_NON_DEFAULT_MODEL = next(
        (model for model in BENCHMARK_MODELS if model != DEFAULT_MODEL),
        None,
    )

    @pytest.mark.asyncio
    async def test_rejects_non_allowlisted_retrieval_model_name(self):
        """Unsupported retrieval models should return a 400 before loading."""
        request = TextProcessingRequest(
            text_content="test text",
            language="en",
            retrieval_model_name="not-allowlisted-model",
        )

        with patch(
            "api.routers.text_processing_router.get_sbert_model_dependency"
        ) as mock_get_model:
            mock_get_model.return_value = MagicMock()

            with patch(
                "api.routers.text_processing_router.get_dense_retriever_dependency"
            ) as mock_get_retriever:
                mock_get_retriever.return_value = MagicMock()

                with patch(
                    "api.routers.text_processing_router.TextProcessingPipeline"
                ) as mock_pipeline_cls:
                    mock_pipeline = MagicMock()
                    mock_pipeline.process = MagicMock()
                    mock_pipeline_cls.return_value = mock_pipeline

                    async def fake_run_in_threadpool(func, *args, **kwargs):
                        raise AssertionError(f"Unexpected callable: {func!r}")

                    with patch(
                        "api.routers.text_processing_router.run_in_threadpool",
                        side_effect=fake_run_in_threadpool,
                    ):
                        with pytest.raises(HTTPException) as exc_info:
                            await _prepare_standard_request_context(request)

        assert exc_info.value.status_code == 400
        assert "retrieval_model_name" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_accepts_explicit_default_retrieval_model_and_uses_safe_loading(
        self,
    ):
        """Allowlisted default model should pass through to loading with trust_remote_code=True."""
        request = TextProcessingRequest(
            text_content="test text",
            language="en",
            retrieval_model_name=DEFAULT_MODEL,
        )

        with patch(
            "api.routers.text_processing_router.get_sbert_model_dependency"
        ) as mock_get_model:
            mock_get_model.return_value = MagicMock()

            with patch(
                "api.routers.text_processing_router.get_dense_retriever_dependency"
            ) as mock_get_retriever:
                mock_get_retriever.return_value = MagicMock()

                with patch(
                    "api.routers.text_processing_router.TextProcessingPipeline"
                ) as mock_pipeline_cls:
                    mock_pipeline = MagicMock()
                    mock_pipeline.process = MagicMock()
                    mock_pipeline_cls.return_value = mock_pipeline

                    async def fake_run_in_threadpool(func, *args, **kwargs):
                        raise AssertionError(f"Unexpected callable: {func!r}")

                    with patch(
                        "api.routers.text_processing_router.run_in_threadpool",
                        side_effect=fake_run_in_threadpool,
                    ):
                        await _prepare_standard_request_context(request)

        mock_get_model.assert_called_once_with(
            model_name_requested=DEFAULT_MODEL,
            trust_remote_code=True,
        )

    @pytest.mark.asyncio
    async def test_omitted_semantic_model_name_falls_back_to_retrieval_model(self):
        """When semantic_model_name is omitted, chunking should reuse the retrieval model."""
        if self.SUPPORTED_NON_DEFAULT_MODEL is None:
            pytest.skip("No configured non-default benchmark models available")

        request = TextProcessingRequest(
            text_content="test text",
            language="en",
            retrieval_model_name=self.SUPPORTED_NON_DEFAULT_MODEL,
        )

        with patch(
            "api.routers.text_processing_router.get_sbert_model_dependency"
        ) as mock_get_model:
            mock_get_model.return_value = MagicMock()

            with patch(
                "api.routers.text_processing_router.get_dense_retriever_dependency"
            ) as mock_get_retriever:
                mock_get_retriever.return_value = MagicMock()

                with patch(
                    "api.routers.text_processing_router.TextProcessingPipeline"
                ) as mock_pipeline_cls:
                    mock_pipeline = MagicMock()
                    mock_pipeline.process = MagicMock()
                    mock_pipeline_cls.return_value = mock_pipeline

                    async def fake_run_in_threadpool(func, *args, **kwargs):
                        raise AssertionError(f"Unexpected callable: {func!r}")

                    with patch(
                        "api.routers.text_processing_router.run_in_threadpool",
                        side_effect=fake_run_in_threadpool,
                    ):
                        await _prepare_standard_request_context(request)

        mock_get_model.assert_called_once_with(
            model_name_requested=self.SUPPORTED_NON_DEFAULT_MODEL,
            trust_remote_code=_get_trust_remote_code_for_model(
                self.SUPPORTED_NON_DEFAULT_MODEL
            ),
        )

    @pytest.mark.asyncio
    async def test_accepts_supported_non_default_retrieval_model(self):
        """Configured non-default models should remain valid API choices."""
        if self.SUPPORTED_NON_DEFAULT_MODEL is None:
            pytest.skip("No configured non-default benchmark models available")

        request = TextProcessingRequest(
            text_content="test text",
            language="en",
            retrieval_model_name=self.SUPPORTED_NON_DEFAULT_MODEL,
            semantic_model_name=self.SUPPORTED_NON_DEFAULT_MODEL,
        )

        with patch(
            "api.routers.text_processing_router.get_sbert_model_dependency"
        ) as mock_get_model:
            mock_get_model.return_value = MagicMock()

            with patch(
                "api.routers.text_processing_router.get_dense_retriever_dependency"
            ) as mock_get_retriever:
                mock_get_retriever.return_value = MagicMock()

                with patch(
                    "api.routers.text_processing_router.TextProcessingPipeline"
                ) as mock_pipeline_cls:
                    mock_pipeline = MagicMock()
                    mock_pipeline.process = MagicMock()
                    mock_pipeline_cls.return_value = mock_pipeline

                    async def fake_run_in_threadpool(func, *args, **kwargs):
                        raise AssertionError(f"Unexpected callable: {func!r}")

                    with patch(
                        "api.routers.text_processing_router.run_in_threadpool",
                        side_effect=fake_run_in_threadpool,
                    ):
                        await _prepare_standard_request_context(request)

        mock_get_model.assert_called_once()
        _, kwargs = mock_get_model.call_args
        assert kwargs == {
            "model_name_requested": self.SUPPORTED_NON_DEFAULT_MODEL,
            "trust_remote_code": _get_trust_remote_code_for_model(
                self.SUPPORTED_NON_DEFAULT_MODEL
            ),
        }

    @pytest.mark.asyncio
    async def test_preserves_explicit_zero_values_for_extraction_backend(self):
        """Explicit 0.0 values should reach the shared-service kwargs unchanged."""
        request = TextProcessingRequest(
            text_content="test text",
            language="en",
            chunk_retrieval_threshold=0.0,
            aggregated_term_confidence=0.0,
        )

        standard_context = {
            "actual_language": "en",
            "retrieval_model_name": DEFAULT_MODEL,
            "chunking_pipeline_config": [{"type": "simple"}],
            "retriever": MagicMock(),
            "text_pipeline": MagicMock(sbert_model=MagicMock()),
        }
        captured_kwargs: dict[str, object] = {}

        with patch(
            "api.routers.text_processing_router._prepare_standard_request_context",
            AsyncMock(return_value=standard_context),
        ):
            with patch(
                "api.routers.text_processing_router.run_full_text_service",
                side_effect=lambda **kwargs: (
                    captured_kwargs.update(kwargs)
                    or {
                        "meta": {"extraction_backend": "standard"},
                        "processed_chunks": [],
                        "aggregated_hpo_terms": [],
                    }
                ),
            ):
                await _process_text_via_shared_service(request)

        assert captured_kwargs["chunk_retrieval_threshold"] == 0.0
        assert captured_kwargs["min_confidence_for_aggregated"] == 0.0

    @pytest.mark.asyncio
    async def test_standard_extraction_backend_shared_service_path_validates_adapted_response(
        self,
    ):
        """Standard shared-service responses should still hit the invariant check."""
        request = TextProcessingRequest(
            text_content="test text",
            language="en",
        )

        standard_context = {
            "actual_language": "en",
            "retrieval_model_name": DEFAULT_MODEL,
            "chunking_pipeline_config": [{"type": "simple"}],
            "retriever": MagicMock(),
            "text_pipeline": MagicMock(sbert_model=MagicMock()),
        }

        with patch(
            "api.routers.text_processing_router._prepare_standard_request_context",
            AsyncMock(return_value=standard_context),
        ):
            with patch(
                "api.routers.text_processing_router.run_full_text_service",
                return_value={
                    "meta": {"extraction_backend": "standard"},
                    "processed_chunks": [
                        {
                            "chunk_id": 1,
                            "text": "chunk",
                            "status": "affirmed",
                            "hpo_matches": [{"id": "HP:0001250", "name": "Seizure"}],
                        }
                    ],
                    "aggregated_hpo_terms": [
                        {
                            "id": "HP:0001250",
                            "name": "Seizure",
                            "confidence": 0.9,
                            "status": "affirmed",
                            "evidence_count": 1,
                            "chunks": [0],
                            "text_attributions": [],
                            "score": 0.9,
                        }
                    ],
                },
            ):
                with patch(
                    "api.routers.text_processing_router._validate_response_chunk_references"
                ) as mock_validate:
                    await _process_text_via_shared_service(request)

        mock_validate.assert_called_once()


class TestValidateResponseChunkReferences:
    """Test _validate_response_chunk_references validation function."""

    def test_valid_references_pass_validation(self):
        """Test valid chunk references pass all checks."""
        # Arrange
        chunks = [
            ProcessedChunkAPI(
                chunk_id=1,
                text="First chunk with seizures",
                status="affirmed",
            ),
            ProcessedChunkAPI(
                chunk_id=2,
                text="Second chunk with autism",
                status="affirmed",
            ),
        ]
        terms = [
            AggregatedHPOTermAPI(
                hpo_id="HP:0001250",
                name="Seizure",
                confidence=0.9,
                status="affirmed",
                evidence_count=1,
                source_chunk_ids=[1],
                text_attributions=[
                    TextAttributionSpanAPI(
                        chunk_id=1,
                        matched_text_in_chunk="seizures",
                        start_char=17,
                        end_char=25,
                    )
                ],
                top_evidence_chunk_id=1,
            ),
            AggregatedHPOTermAPI(
                hpo_id="HP:0000729",
                name="Autism",
                confidence=0.85,
                status="affirmed",
                evidence_count=1,
                source_chunk_ids=[2],
                text_attributions=[
                    TextAttributionSpanAPI(
                        chunk_id=2,
                        matched_text_in_chunk="autism",
                        start_char=18,
                        end_char=24,
                    )
                ],
                top_evidence_chunk_id=2,
            ),
        ]

        # Act - should not raise
        _validate_response_chunk_references(chunks, terms)

        # Assert - verify the data structures are intact after validation
        assert len(chunks) == 2
        assert len(terms) == 2
        assert terms[0].id == "HP:0001250"
        assert terms[1].id == "HP:0000729"

    def test_non_sequential_chunk_ids_fail_validation(self):
        """Test non-sequential chunk IDs trigger assertion."""
        # Arrange - chunk IDs are 1 and 3 (missing 2)
        chunks = [
            ProcessedChunkAPI(
                chunk_id=1,
                text="First chunk",
                status="affirmative",
            ),
            ProcessedChunkAPI(
                chunk_id=3,  # ❌ Should be 2
                text="Third chunk",
                status="affirmative",
            ),
        ]
        terms: list[AggregatedHPOTermAPI] = []

        # Act & Assert
        with pytest.raises(AssertionError, match="Chunk IDs not sequential 1-based"):
            _validate_response_chunk_references(chunks, terms)

    def test_non_1_based_chunk_ids_fail_validation(self):
        """Test chunk IDs not starting at 1 trigger assertion."""
        # Arrange - chunk IDs start at 0
        chunks = [
            ProcessedChunkAPI(
                chunk_id=0,  # ❌ Should be 1
                text="First chunk",
                status="affirmative",
            ),
            ProcessedChunkAPI(
                chunk_id=1,
                text="Second chunk",
                status="affirmative",
            ),
        ]
        terms: list[AggregatedHPOTermAPI] = []

        # Act & Assert
        with pytest.raises(AssertionError, match="Chunk IDs not sequential 1-based"):
            _validate_response_chunk_references(chunks, terms)

    def test_invalid_source_chunk_id_fails_validation(self):
        """Test invalid source_chunk_id triggers assertion."""
        # Arrange
        chunks = [
            ProcessedChunkAPI(
                chunk_id=1,
                text="Only chunk",
                status="affirmative",
            ),
        ]
        terms = [
            AggregatedHPOTermAPI(
                hpo_id="HP:0001250",
                name="Seizure",
                confidence=0.9,
                status="affirmed",
                evidence_count=1,
                source_chunk_ids=[1, 2, 3],  # ❌ 2 and 3 don't exist
                text_attributions=[],
            ),
        ]

        # Act & Assert
        with pytest.raises(
            AssertionError, match="has invalid source_chunk_ids.*{2, 3}"
        ):
            _validate_response_chunk_references(chunks, terms)

    def test_invalid_text_attribution_chunk_id_fails_validation(self):
        """Test invalid text_attribution chunk_id triggers assertion."""
        # Arrange
        chunks = [
            ProcessedChunkAPI(
                chunk_id=1,
                text="Only chunk",
                status="affirmative",
            ),
        ]
        terms = [
            AggregatedHPOTermAPI(
                hpo_id="HP:0001250",
                name="Seizure",
                confidence=0.9,
                status="affirmed",
                evidence_count=1,
                source_chunk_ids=[1],
                text_attributions=[
                    TextAttributionSpanAPI(
                        chunk_id=99,  # ❌ Doesn't exist
                        matched_text_in_chunk="seizure",
                        start_char=0,
                        end_char=7,
                    )
                ],
            ),
        ]

        # Act & Assert
        with pytest.raises(
            AssertionError,
            match="has text_attribution with invalid chunk_id 99",
        ):
            _validate_response_chunk_references(chunks, terms)

    def test_invalid_top_evidence_chunk_id_fails_validation(self):
        """Test invalid top_evidence_chunk_id triggers assertion."""
        # Arrange
        chunks = [
            ProcessedChunkAPI(
                chunk_id=1,
                text="Only chunk",
                status="affirmative",
            ),
        ]
        terms = [
            AggregatedHPOTermAPI(
                hpo_id="HP:0001250",
                name="Seizure",
                confidence=0.9,
                status="affirmed",
                evidence_count=1,
                source_chunk_ids=[1],
                text_attributions=[],
                top_evidence_chunk_id=42,  # ❌ Doesn't exist
            ),
        ]

        # Act & Assert
        with pytest.raises(
            AssertionError,
            match="has invalid top_evidence_chunk_id 42",
        ):
            _validate_response_chunk_references(chunks, terms)

    def test_none_top_evidence_chunk_id_passes_validation(self):
        """Test None top_evidence_chunk_id is valid (optional field)."""
        # Arrange
        chunks = [
            ProcessedChunkAPI(
                chunk_id=1,
                text="Only chunk",
                status="affirmative",
            ),
        ]
        terms = [
            AggregatedHPOTermAPI(
                hpo_id="HP:0001250",
                name="Seizure",
                confidence=0.9,
                status="affirmed",
                evidence_count=1,
                source_chunk_ids=[1],
                text_attributions=[],
                top_evidence_chunk_id=None,  # ✅ Optional
            ),
        ]

        # Act - should not raise
        _validate_response_chunk_references(chunks, terms)

        # Assert - None top_evidence_chunk_id is accepted
        assert terms[0].top_evidence_chunk_id is None

    def test_empty_chunks_and_terms_pass_validation(self):
        """Test empty chunks and terms pass validation."""
        # Arrange
        chunks: list[ProcessedChunkAPI] = []
        terms: list[AggregatedHPOTermAPI] = []

        # Act - should not raise
        _validate_response_chunk_references(chunks, terms)

        # Assert - empty lists remain empty
        assert len(chunks) == 0
        assert len(terms) == 0

    def test_multiple_text_attributions_all_valid(self):
        """Test multiple text attributions with valid chunk IDs."""
        # Arrange
        chunks = [
            ProcessedChunkAPI(
                chunk_id=1,
                text="First chunk",
                status="affirmative",
            ),
            ProcessedChunkAPI(
                chunk_id=2,
                text="Second chunk",
                status="affirmative",
            ),
        ]
        terms = [
            AggregatedHPOTermAPI(
                hpo_id="HP:0001250",
                name="Seizure",
                confidence=0.9,
                status="affirmed",
                evidence_count=1,
                source_chunk_ids=[1, 2],
                text_attributions=[
                    TextAttributionSpanAPI(
                        chunk_id=1,
                        matched_text_in_chunk="seizure",
                        start_char=0,
                        end_char=7,
                    ),
                    TextAttributionSpanAPI(
                        chunk_id=2,
                        matched_text_in_chunk="epilepsy",
                        start_char=0,
                        end_char=8,
                    ),
                ],
            ),
        ]

        # Act - should not raise
        _validate_response_chunk_references(chunks, terms)

        # Assert - verify attributions span both chunks
        assert len(terms) == 1
        assert len(terms[0].text_attributions) == 2
        assert terms[0].text_attributions[0].chunk_id == 1
        assert terms[0].text_attributions[1].chunk_id == 2
