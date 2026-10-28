"""Startup warm-up of the default Chroma index (#419)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.main import create_app

pytestmark = pytest.mark.unit


@pytest.fixture
def stub_sbert(monkeypatch):
    monkeypatch.setattr(
        "api.main.get_sbert_model_dependency", AsyncMock(return_value=MagicMock())
    )


def _start_app() -> None:
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        assert client.get("/").status_code == 200


def test_lifespan_warms_default_index(monkeypatch, stub_sbert) -> None:
    retriever = MagicMock()
    monkeypatch.setattr(
        "api.main.get_dense_retriever_dependency", AsyncMock(return_value=retriever)
    )

    _start_app()

    retriever.query.assert_called_once_with("warmup", n_results=1)


def test_lifespan_survives_failed_warmup_query(monkeypatch, stub_sbert) -> None:
    retriever = MagicMock()
    retriever.query.side_effect = RuntimeError("collection missing")
    monkeypatch.setattr(
        "api.main.get_dense_retriever_dependency", AsyncMock(return_value=retriever)
    )

    _start_app()

    retriever.query.assert_called_once()


def test_lifespan_survives_missing_index(monkeypatch, stub_sbert) -> None:
    monkeypatch.setattr(
        "api.main.get_dense_retriever_dependency",
        AsyncMock(side_effect=HTTPException(status_code=503, detail="no index")),
    )

    _start_app()
