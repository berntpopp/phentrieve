"""REST LLM-error fallback and LLM-path retriever reuse (#418, #420)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas.text_processing_schemas import TextProcessingRequest
from api.services import text_processing_execution

pytestmark = pytest.mark.unit

_EMPTY_RESULT = {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def stub_retrievers(monkeypatch):
    """Avoid real model/index loads on both the standard and the LLM path."""
    retriever = MagicMock(model=MagicMock())
    for target in (
        "api.services.text_processing_execution.get_dense_retriever_dependency",
        "api.services.text_processing_context.get_dense_retriever_dependency",
        "api.services.text_processing_context.get_sbert_model_dependency",
    ):
        monkeypatch.setattr(target, AsyncMock(return_value=retriever))
    monkeypatch.setattr(
        "api.services.text_processing_context.TextProcessingPipeline", MagicMock()
    )
    return retriever


# --------------------------------------------------------------------------- #
# #418: REST allow_standard_fallback covers LLM backend errors
# --------------------------------------------------------------------------- #
def _flaky_llm_service(**kwargs):
    if kwargs["extraction_backend"] == "llm":
        raise RuntimeError("provider exploded")
    return {
        "meta": {"extraction_backend": "standard"},
        "processed_chunks": [],
        "aggregated_hpo_terms": [],
    }


def test_llm_backend_error_falls_back_to_standard_when_allowed(
    client, monkeypatch, stub_retrievers
):
    monkeypatch.setattr("api.config.PHENTRIEVE_LLM_QUOTA_ENFORCE", "false")
    monkeypatch.setattr(
        "api.services.text_processing_execution.run_full_text_service",
        _flaky_llm_service,
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Patient had recurrent seizures.",
            "language": "en",
            "extraction_backend": "llm",
        },
        headers={"X-Phentrieve-Allow-Standard-Fallback": "true"},
    )

    assert response.status_code == 200
    meta = response.json()["meta"]
    assert meta["fallback_reason"] == "llm_backend_error"
    assert meta["extraction_backend"] == "standard"
    assert "provider exploded" not in response.text


def test_llm_backend_error_body_flag_also_enables_fallback(
    client, monkeypatch, stub_retrievers
):
    monkeypatch.setattr("api.config.PHENTRIEVE_LLM_QUOTA_ENFORCE", "false")
    monkeypatch.setattr(
        "api.services.text_processing_execution.run_full_text_service",
        _flaky_llm_service,
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Patient had recurrent seizures.",
            "language": "en",
            "extraction_backend": "llm",
            "allow_standard_fallback": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["meta"]["fallback_reason"] == "llm_backend_error"


def test_llm_backend_error_without_fallback_is_500(
    client, monkeypatch, stub_retrievers
):
    monkeypatch.setattr("api.config.PHENTRIEVE_LLM_QUOTA_ENFORCE", "false")
    monkeypatch.setattr(
        "api.services.text_processing_execution.run_full_text_service",
        _flaky_llm_service,
    )

    response = client.post(
        "/api/v1/text/process",
        json={"text": "Patient had recurrent seizures.", "extraction_backend": "llm"},
    )

    assert response.status_code == 500


def test_llm_backend_fallback_does_not_consume_quota(
    tmp_path, monkeypatch, stub_retrievers
):
    from api.llm_quota import hash_subject_key, read_llm_quota

    quota_db = tmp_path / "llm_quota.db"
    monkeypatch.setattr("api.config.PHENTRIEVE_LLM_QUOTA_ENFORCE", "true")
    monkeypatch.setattr("api.config.PHENTRIEVE_LLM_DAILY_LIMIT", 2)
    monkeypatch.setattr("api.config.PHENTRIEVE_LLM_QUOTA_DB_PATH", str(quota_db))
    monkeypatch.setattr("api.config.PHENTRIEVE_TRUSTED_PROXY_CIDRS", "172.16.0.0/12")
    monkeypatch.setattr(
        "api.services.text_processing_execution.run_full_text_service",
        _flaky_llm_service,
    )

    with TestClient(
        app, raise_server_exceptions=False, client=("172.18.0.10", 50000)
    ) as local_client:
        response = local_client.post(
            "/api/v1/text/process",
            json={
                "text": "Patient had recurrent seizures.",
                "language": "en",
                "extraction_backend": "llm",
                "allow_standard_fallback": True,
            },
            headers={"X-Forwarded-For": "203.0.113.7"},
        )

    assert response.status_code == 200
    meta = response.json()["meta"]
    assert meta["fallback_reason"] == "llm_backend_error"
    assert "quota_remaining" not in meta
    # The failed LLM attempt's reservation was released.
    status = read_llm_quota(subject_key=hash_subject_key("203.0.113.7"), daily_limit=2)
    assert status.quota_used == 0


# --------------------------------------------------------------------------- #
# #420: the LLM path reuses the cached retriever
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_llm_execution_passes_cached_retriever(monkeypatch, stub_retrievers):
    captured: dict[str, object] = {}

    async def fake_run_in_threadpool(func, **kwargs):
        captured.update(kwargs)
        return _EMPTY_RESULT

    monkeypatch.setattr(
        text_processing_execution, "run_in_threadpool", fake_run_in_threadpool
    )

    await text_processing_execution.execute_llm_text_processing(
        TextProcessingRequest(text="Patient has seizures.", extraction_backend="llm")
    )

    dependency = text_processing_execution.get_dense_retriever_dependency
    dependency.assert_awaited_once()  # type: ignore[attr-defined]
    assert captured["retriever"] is stub_retrievers
    assert captured["sbert_model_for_semantic_chunking"] is stub_retrievers.model
