"""Standardized error response schema for the Phentrieve API.

Every 4xx/5xx response from the Phentrieve API is rendered through this
schema via the exception handlers registered in api.main.create_app():

- ``StarletteHTTPException`` → covers explicit ``HTTPException`` raises
  in routers AND FastAPI's routing 404s (``HTTPException`` is a subclass).
- ``RequestValidationError`` → covers Pydantic validation failures (422).
- ``Exception`` → catch-all 500 handler for anything else that escapes
  a router. Logs the full traceback server-side and returns a generic
  ``internal_server_error`` payload without leaking exception details.

This gives API consumers a single stable error shape regardless of the
failure mode.
"""

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response envelope.

    Every 4xx and 5xx response from the Phentrieve API conforms to this
    shape. The ``error`` field is a machine-readable slug derived from
    the HTTP status (e.g. "bad_request", "not_found"). The ``detail``
    field is the original HTTPException detail — a string for simple
    errors, or a dict/list for structured error payloads. Structured
    details are passed through without stringification so clients can
    parse their fields directly.
    """

    status_code: int = Field(
        ...,
        description="HTTP status code",
        examples=[422, 503],
    )
    error: str = Field(
        ...,
        description="Machine-readable error slug (snake_case)",
        examples=["unprocessable_entity", "service_unavailable"],
    )
    detail: str | dict[str, Any] | list[Any] = Field(
        ...,
        description=(
            "Human-readable error description, or a structured payload for errors "
            "that carry additional context (e.g. the similarity router returns a dict "
            "with term1/term2/error_message fields on 404)."
        ),
        examples=[
            "Field 'num_results' must be between 1 and 50.",
            {"error_message": "Term not found", "term1": "HP:0001250"},
        ],
    )
    request_id: str | None = Field(
        default=None,
        description="Optional request correlation ID, if the server attaches one.",
    )
