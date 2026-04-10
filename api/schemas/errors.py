"""Standardized error response schema for the Phentrieve API.

All HTTPException responses are rendered through this schema via a
global exception handler registered in api.main.create_app(), giving
API consumers a single stable error shape.
"""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response envelope.

    Every 4xx and 5xx response from the Phentrieve API conforms to this
    shape. The ``error`` field is a machine-readable slug derived from
    the HTTP status (e.g. "bad_request", "not_found"). The ``detail``
    field is the human-readable explanation from the HTTPException.
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
    detail: str = Field(
        ...,
        description="Human-readable error description",
        examples=["Field 'num_results' must be between 1 and 50."],
    )
    request_id: str | None = Field(
        default=None,
        description="Optional request correlation ID, if the server attaches one.",
    )
