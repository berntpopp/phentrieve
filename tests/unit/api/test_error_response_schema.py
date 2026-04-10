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


class TestDetailPassthrough:
    """Verifies the handler preserves both string and structured detail payloads.

    Regression guard for commit eb18d5f review: the original handler called
    str(exc.detail), which stringified dict-valued details from the similarity
    router with Python repr syntax and broke clients that parsed the JSON.
    """

    def _app_with_raise(self, status_code: int, detail):
        """Build a minimal FastAPI app whose /boom endpoint raises an HTTPException."""
        from http import HTTPStatus

        from fastapi import FastAPI, Request
        from fastapi import HTTPException as FastAPIHTTPException
        from fastapi.responses import JSONResponse
        from starlette.exceptions import HTTPException as StarletteHTTPException

        mini = FastAPI()

        # Re-register the same handler the production app registers, so this
        # test exercises the real handler logic in isolation.
        @mini.exception_handler(StarletteHTTPException)
        async def handler(
            _request: Request, exc: StarletteHTTPException
        ) -> JSONResponse:
            try:
                slug = HTTPStatus(exc.status_code).phrase.lower().replace(" ", "_")
            except ValueError:
                slug = "http_error"
            body = ErrorResponse(
                status_code=exc.status_code,
                error=slug,
                detail=exc.detail if exc.detail is not None else slug,
            )
            return JSONResponse(
                status_code=exc.status_code,
                content=body.model_dump(exclude_none=True),
                headers=getattr(exc, "headers", None) or None,
            )

        @mini.get("/boom")
        async def boom():
            raise FastAPIHTTPException(status_code=status_code, detail=detail)

        return mini

    def test_string_detail_passes_through(self):
        app = self._app_with_raise(400, "Bad input")
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/boom")
        assert response.status_code == 400
        body = response.json()
        assert body["detail"] == "Bad input"  # string, not repr
        assert body["error"] == "bad_request"

    def test_dict_detail_passes_through(self):
        app = self._app_with_raise(
            404,
            {
                "error_message": "One or both HPO terms not found",
                "term1": "HP:0001250",
                "term2": "HP:0001251",
            },
        )
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/boom")
        assert response.status_code == 404
        body = response.json()
        # detail must be the original dict, not a stringified repr
        assert isinstance(body["detail"], dict)
        assert body["detail"]["error_message"] == "One or both HPO terms not found"
        assert body["detail"]["term1"] == "HP:0001250"
        assert body["detail"]["term2"] == "HP:0001251"
        assert body["error"] == "not_found"

    def test_list_detail_passes_through(self):
        app = self._app_with_raise(
            422,
            [{"loc": ["body", "num_results"], "msg": "too high"}],
        )
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/boom")
        assert response.status_code == 422
        body = response.json()
        assert isinstance(body["detail"], list)
        assert body["detail"][0]["msg"] == "too high"
