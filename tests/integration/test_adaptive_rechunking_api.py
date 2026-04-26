"""API parity for adaptive re-chunking.

Plan B Phase 9 / Task 12: verify that ``TextProcessingRequest`` accepts an
optional ``adaptive_rechunking`` block, that the router forwards it to the
shared full-text service as an ``AdaptiveRechunkingConfig`` instance, and
that the response surfaces ``meta.adaptive_rechunking`` when populated by
the backend.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


def _standard_context() -> dict[str, object]:
    from phentrieve.config import DEFAULT_MODEL

    return {
        "actual_language": "en",
        "retrieval_model_name": DEFAULT_MODEL,
        "chunking_pipeline_config": [{"type": "simple"}],
        "retriever": MagicMock(),
        "text_pipeline": MagicMock(sbert_model=MagicMock()),
    }


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


class TestAdaptiveRechunkingAPI:
    """Adaptive re-chunking parity between the CLI and the HTTP API."""

    def test_request_with_adaptive_rechunking_forwards_config(self, client):
        """Request with adaptive_rechunking block forwards a resolved config."""
        from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig

        captured_kwargs: dict[str, object] = {}

        def fake_run(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {
                "meta": {
                    "extraction_backend": "standard",
                    "adaptive_rechunking": {
                        "enabled": True,
                        "trigger_count": 1,
                        "subdivisions": 2,
                        "applied": True,
                    },
                },
                "processed_chunks": [],
                "aggregated_hpo_terms": [],
            }

        with patch(
            "api.routers.text_processing_router._prepare_standard_request_context",
            AsyncMock(return_value=_standard_context()),
        ):
            with patch(
                "api.routers.text_processing_router.run_full_text_service",
                side_effect=fake_run,
            ):
                response = client.post(
                    "/api/v1/text/process",
                    json={
                        "text": "Patient with seizures.",
                        "extraction_backend": "standard",
                        "adaptive_rechunking": {
                            "enabled": True,
                            "quality_threshold": 0.5,
                            "margin_threshold": 0.02,
                        },
                    },
                )

        assert response.status_code == 200, response.text

        # The router must hand the shared service a frozen config dataclass
        # rather than the raw Pydantic block.
        forwarded = captured_kwargs.get("adaptive_rechunking")
        assert isinstance(forwarded, AdaptiveRechunkingConfig)
        assert forwarded.enabled is True
        assert forwarded.quality_threshold == 0.5
        assert forwarded.margin_threshold == 0.02

        body = response.json()
        assert body["meta"]["adaptive_rechunking"]["enabled"] is True
        assert body["meta"]["adaptive_rechunking"]["trigger_count"] == 1

    def test_request_without_adaptive_rechunking_omits_field(self, client):
        """Default request must not forward an adaptive config nor surface meta."""
        captured_kwargs: dict[str, object] = {}

        def fake_run(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {
                "meta": {"extraction_backend": "standard"},
                "processed_chunks": [],
                "aggregated_hpo_terms": [],
            }

        with patch(
            "api.routers.text_processing_router._prepare_standard_request_context",
            AsyncMock(return_value=_standard_context()),
        ):
            with patch(
                "api.routers.text_processing_router.run_full_text_service",
                side_effect=fake_run,
            ):
                response = client.post(
                    "/api/v1/text/process",
                    json={
                        "text": "Patient with seizures.",
                        "extraction_backend": "standard",
                    },
                )

        assert response.status_code == 200, response.text
        assert "adaptive_rechunking" not in captured_kwargs
        body = response.json()
        assert "adaptive_rechunking" not in body.get("meta", {})

    def test_adaptive_rechunking_request_field_rejects_unknown_keys(self, client):
        """The request schema forbids unknown adaptive_rechunking keys (extra=forbid)."""
        response = client.post(
            "/api/v1/text/process",
            json={
                "text": "Patient with seizures.",
                "adaptive_rechunking": {
                    "enabled": True,
                    "this_key_does_not_exist": 1.0,
                },
            },
        )
        assert response.status_code == 422
