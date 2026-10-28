"""Tests for query router retrieval model policy enforcement."""

import asyncio
import warnings
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

    def fake_execute_hpo_retrieval_for_api(**kwargs):
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


@pytest.fixture
def stub_query_path(monkeypatch):
    """Stub retriever loading and retrieval; record what the router passes."""
    captured: dict[str, dict[str, object]] = {"retriever": {}, "query": {}}

    async def fake_get_dense_retriever_dependency(**kwargs):
        captured["retriever"].update(kwargs)
        return MagicMock(model_name=kwargs["sbert_model_name_for_retriever"])

    def fake_execute_hpo_retrieval_for_api(**kwargs):
        # #405: blocking retrieval must run in a worker thread, not on the loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            kwargs["_ran_off_event_loop"] = True
        else:
            kwargs["_ran_off_event_loop"] = False
        captured["query"].update(kwargs)
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
    return captured


def test_post_query_runs_retrieval_off_event_loop(client, stub_query_path) -> None:
    response = client.post("/api/v1/query/", json={"text": "Patient has seizures."})

    assert response.status_code == 200
    assert stub_query_path["query"]["_ran_off_event_loop"] is True


def test_get_query_injects_awaited_retriever_for_requested_model(
    client, stub_query_path
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        response = client.get(
            "/api/v1/query/",
            params={"text": "seizures", "model_name": DEFAULT_MODEL},
        )

    assert response.status_code == 200
    assert response.json()["model_used_for_retrieval"] == DEFAULT_MODEL
    assert stub_query_path["retriever"]["sbert_model_name_for_retriever"] == (
        DEFAULT_MODEL
    )
    # The router must receive a real retriever, not an un-awaited coroutine.
    assert stub_query_path["query"]["retriever"].model_name == DEFAULT_MODEL
    assert stub_query_path["query"]["_ran_off_event_loop"] is True


def test_get_query_rejects_unsupported_model(client, stub_query_path) -> None:
    response = client.get(
        "/api/v1/query/",
        params={"text": "seizures", "model_name": "attacker/remote-code"},
    )

    assert response.status_code == 400
    assert stub_query_path["retriever"] == {}


@pytest.mark.parametrize(
    "params",
    [
        {"text": "seizures", "num_results": 100},
        {"text": "seizures", "num_results": 30, "include_details": True},
        {"text": "seizures", "query_assertion_preference": "bogus"},
    ],
)
def test_get_query_invalid_params_return_422(client, stub_query_path, params) -> None:
    response = client.get("/api/v1/query/", params=params)

    assert response.status_code == 422
    assert stub_query_path["query"] == {}


def test_get_query_openapi_has_no_stray_model_parameter(client) -> None:
    params = client.get("/api/v1/openapi.json").json()["paths"]["/api/v1/query/"][
        "get"
    ]["parameters"]
    names = [p["name"] for p in params]

    assert "model" not in names
    assert names.count("model_name") == 1
