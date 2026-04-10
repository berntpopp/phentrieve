"""Tests for the ErrorResponse Pydantic schema and global handler."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.schemas.errors import ErrorResponse

pytestmark = pytest.mark.unit


class TestErrorResponseSchema:
    def test_schema_fields(self):
        """ErrorResponse must have status_code, error, detail, and optional request_id."""
        resp = ErrorResponse(
            status_code=422,
            error="unprocessable_entity",
            detail="Field 'num_results' out of range",
        )
        assert resp.status_code == 422
        assert resp.error == "unprocessable_entity"
        assert resp.detail == "Field 'num_results' out of range"
        assert resp.request_id is None

    def test_schema_accepts_request_id(self):
        resp = ErrorResponse(
            status_code=500,
            error="internal_server_error",
            detail="Boom",
            request_id="abc-123",
        )
        assert resp.request_id == "abc-123"

    def test_schema_rejects_missing_fields(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ErrorResponse(status_code=500)  # type: ignore[call-arg]


class TestGlobalExceptionHandler:
    def test_http_exception_returns_error_response_shape(self):
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        # Hit a nonexistent endpoint that causes FastAPI to raise a 404 HTTPException.
        response = client.get("/api/v1/nonexistent-endpoint")
        assert response.status_code == 404
        body = response.json()
        # Must conform to ErrorResponse, not the raw FastAPI default shape.
        assert "status_code" in body
        assert "error" in body
        assert "detail" in body
        assert body["status_code"] == response.status_code
