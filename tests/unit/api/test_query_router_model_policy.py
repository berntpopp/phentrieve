"""Tests for query router retrieval model policy enforcement."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from phentrieve.config import DEFAULT_MODEL

pytestmark = pytest.mark.unit


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "api.main.get_sbert_model_dependency",
        AsyncMock(return_value=MagicMock(model_name=DEFAULT_MODEL)),
    )
    monkeypatch.setattr(
        "api.main.get_dense_retriever_dependency",
        AsyncMock(return_value=MagicMock(model_name=DEFAULT_MODEL)),
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_query_rejects_unsupported_model_before_retriever_load(
    client,
    monkeypatch,
) -> None:
    async def fail_if_called(**_kwargs):
        raise AssertionError("unsupported model should not reach retriever loading")

    monkeypatch.setattr(
        "api.routers.query_router.get_dense_retriever_dependency",
        fail_if_called,
    )

    response = client.post(
        "/api/v1/query/",
        json={
            "text": "Patient has seizures.",
            "model_name": "attacker/BioLORD-remote-code",
        },
    )

    assert response.status_code == 400
    assert "Unsupported retrieval model" in response.text


def test_query_omitted_model_name_uses_default_policy_model(
    client,
    monkeypatch,
) -> None:
    captured_retriever_kwargs: dict[str, object] = {}
    captured_query_kwargs: dict[str, object] = {}

    async def fake_get_dense_retriever_dependency(**kwargs):
        captured_retriever_kwargs.update(kwargs)
        return MagicMock(model_name=DEFAULT_MODEL)

    async def fake_execute_hpo_retrieval_for_api(**kwargs):
        captured_query_kwargs.update(kwargs)
        return {"results": [], "original_query_assertion_status": "affirmed"}

    monkeypatch.setattr(
        "api.routers.query_router.get_dense_retriever_dependency",
        fake_get_dense_retriever_dependency,
    )
    monkeypatch.setattr(
        "api.routers.query_router.execute_hpo_retrieval_for_api",
        fake_execute_hpo_retrieval_for_api,
    )
    monkeypatch.setattr(
        "api.routers.query_router._resolve_query_language",
        lambda text, language=None, default_language="en": language or default_language,
    )

    response = client.post(
        "/api/v1/query/",
        json={"text": "Patient has seizures."},
    )

    assert response.status_code == 200
    assert response.json()["model_used_for_retrieval"] == DEFAULT_MODEL
    assert captured_retriever_kwargs["sbert_model_name_for_retriever"] == DEFAULT_MODEL
    assert captured_query_kwargs["retriever"].model_name == DEFAULT_MODEL
