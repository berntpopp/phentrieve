"""Tests for research-use guardrails on text-bearing API endpoints."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas.text_processing_schemas import TextProcessingResponseAPI

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _empty_text_response() -> TextProcessingResponseAPI:
    return TextProcessingResponseAPI.model_validate(
        {
            "meta": {"extraction_backend": "standard"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }
    )


def test_text_processing_requires_research_ack_in_public_hosted_mode(
    client, monkeypatch
):
    monkeypatch.setattr("api.config.PHENTRIEVE_PUBLIC_HOSTED_MODE", True, raising=False)
    monkeypatch.setattr(
        "api.config.PHENTRIEVE_REQUIRE_RESEARCH_ACK", False, raising=False
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Research note mentions recurrent seizures.",
            "extraction_backend": "standard",
        },
    )

    assert response.status_code == 428
    assert response.json()["detail"]["error_code"] == "research_use_ack_required"


def test_text_processing_accepts_research_ack_in_public_hosted_mode(
    client, monkeypatch
):
    monkeypatch.setattr("api.config.PHENTRIEVE_PUBLIC_HOSTED_MODE", True, raising=False)
    monkeypatch.setattr(
        "api.config.PHENTRIEVE_REQUIRE_RESEARCH_ACK", False, raising=False
    )
    monkeypatch.setattr(
        "api.routers.text_processing_router._process_text_via_shared_service",
        AsyncMock(return_value=_empty_text_response()),
    )

    response = client.post(
        "/api/v1/text/process",
        headers={"X-Phentrieve-Research-Use-Acknowledged": "true"},
        json={
            "text": "Research note mentions recurrent seizures.",
            "extraction_backend": "standard",
        },
    )

    assert response.status_code == 200
    assert response.json()["meta"]["extraction_backend"] == "standard"


def test_public_hosted_mode_allows_llm_after_research_ack(client, monkeypatch):
    monkeypatch.setattr("api.config.PHENTRIEVE_PUBLIC_HOSTED_MODE", True, raising=False)
    monkeypatch.setattr(
        "api.config.PHENTRIEVE_REQUIRE_RESEARCH_ACK", False, raising=False
    )
    monkeypatch.setattr(
        "api.routers.text_processing_router._process_text_via_shared_service",
        AsyncMock(
            return_value=TextProcessingResponseAPI.model_validate(
                {
                    "meta": {
                        "extraction_backend": "llm",
                        "llm_model": "gpt-5.4-mini",
                    },
                    "processed_chunks": [],
                    "aggregated_hpo_terms": [],
                }
            )
        ),
    )

    response = client.post(
        "/api/v1/text/process",
        headers={"X-Phentrieve-Research-Use-Acknowledged": "true"},
        json={
            "text": "Research note mentions recurrent seizures.",
            "extraction_backend": "llm",
            "llm_model": "gpt-5.4-mini",
        },
    )

    assert response.status_code == 200
    assert response.json()["meta"]["extraction_backend"] == "llm"
