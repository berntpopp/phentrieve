"""Unit tests for text processing service boundaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from api.schemas.text_processing_schemas import TextProcessingRequest
from phentrieve.config import DEFAULT_MODEL

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_prepare_standard_context_rejects_unsupported_retrieval_model() -> None:
    from api.services.text_processing_context import (
        prepare_standard_text_processing_context,
    )

    request = TextProcessingRequest(
        text="Patient has seizures.",
        language="en",
        retrieval_model_name="not-allowlisted-model",
    )

    with pytest.raises(HTTPException) as exc_info:
        await prepare_standard_text_processing_context(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Unsupported retrieval_model_name: not-allowlisted-model. "
        "Unsupported retrieval model: not-allowlisted-model. "
        "Allowed values: FremyCompany/BioLORD-2023-M, "
        "jinaai/jina-embeddings-v2-base-de, "
        "T-Systems-onsite/cross-en-de-roberta-sentence-transformer, "
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2, "
        "sentence-transformers/distiluse-base-multilingual-cased-v2, "
        "BAAI/bge-m3, Alibaba-NLP/gte-multilingual-base, "
        "sentence-transformers/LaBSE, "
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2."
    )


@pytest.mark.asyncio
async def test_prepare_standard_context_loads_policy_resolved_dependencies(
    monkeypatch,
) -> None:
    from api.services import text_processing_context

    retrieval_model = MagicMock(name="retrieval_model")
    retriever = MagicMock(name="retriever")
    pipeline = MagicMock(name="pipeline")
    pipeline.sbert_model = retrieval_model

    get_sbert = AsyncMock(return_value=retrieval_model)
    get_retriever = AsyncMock(return_value=retriever)
    pipeline_cls = MagicMock(return_value=pipeline)

    monkeypatch.setattr(
        text_processing_context,
        "get_sbert_model_dependency",
        get_sbert,
    )
    monkeypatch.setattr(
        text_processing_context,
        "get_dense_retriever_dependency",
        get_retriever,
    )
    monkeypatch.setattr(
        text_processing_context,
        "TextProcessingPipeline",
        pipeline_cls,
    )

    request = TextProcessingRequest(
        text="Patient has seizures.",
        language="en",
        retrieval_model_name=DEFAULT_MODEL,
        semantic_model_name=None,
    )

    context = await text_processing_context.prepare_standard_text_processing_context(
        request
    )

    get_sbert.assert_awaited_once_with(
        model_name_requested=DEFAULT_MODEL,
        trust_remote_code=True,
    )
    get_retriever.assert_awaited_once_with(sbert_model_name_for_retriever=DEFAULT_MODEL)
    pipeline_cls.assert_called_once()
    assert context["actual_language"] == "en"
    assert context["retrieval_model_name"] == DEFAULT_MODEL
    assert context["retriever"] is retriever
    assert context["text_pipeline"] is pipeline


@pytest.mark.asyncio
async def test_execute_llm_ignores_client_llm_target_overrides(monkeypatch) -> None:
    from api.services import text_processing_execution

    captured_kwargs: dict[str, object] = {}

    request = TextProcessingRequest(
        text="Patient has seizures.",
        extraction_backend="llm",
    )
    object.__setattr__(request, "llm_provider", "openai")
    object.__setattr__(request, "llm_model", "gpt-5.4")
    object.__setattr__(request, "llm_base_url", "https://evil.example/v1")

    def fake_resolve_public_llm_target(**kwargs):
        assert kwargs == {}
        return SimpleNamespace(
            provider="server-provider",
            model="server-model",
            base_url=None,
        )

    def fake_run_full_text_service(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "meta": {"extraction_backend": "llm"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }

    async def fake_run_in_threadpool(func, **kwargs):
        return func(**kwargs)

    monkeypatch.setattr(
        text_processing_execution,
        "resolve_public_llm_target",
        fake_resolve_public_llm_target,
    )
    monkeypatch.setattr(
        text_processing_execution,
        "run_full_text_service",
        fake_run_full_text_service,
    )
    monkeypatch.setattr(
        text_processing_execution,
        "run_in_threadpool",
        fake_run_in_threadpool,
    )

    await text_processing_execution.execute_llm_text_processing(request)

    assert captured_kwargs["llm_provider"] == "server-provider"
    assert captured_kwargs["llm_model"] == "server-model"
    assert captured_kwargs["llm_base_url"] is None
