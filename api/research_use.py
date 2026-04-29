"""Research-use guardrails shared by API routers."""

from fastapi import HTTPException, Request, status

import api.config as api_config

RESEARCH_ACK_HEADER = "x-phentrieve-research-use-acknowledged"
RESEARCH_ACK_HEADER_DISPLAY = "X-Phentrieve-Research-Use-Acknowledged"
RESEARCH_USE_LIMITATION = (
    "Research use only. Phentrieve maps text to HPO terms for research, "
    "education, and knowledge discovery. It must not be used for diagnosis, "
    "treatment selection, patient triage, or other clinical decision-making."
)


def research_ack_openapi_parameter() -> dict[str, object]:
    """Return the OpenAPI parameter for research-use acknowledgement."""
    return {
        "name": RESEARCH_ACK_HEADER_DISPLAY,
        "in": "header",
        "required": False,
        "schema": {
            "type": "string",
            "enum": ["true"],
        },
        "description": (
            "Required when public-hosted or research-ack mode is enabled. "
            "Set to `true` after presenting the research-use limitation."
        ),
    }


def is_research_ack_required() -> bool:
    """Return whether text-bearing endpoints require acknowledgement."""
    return bool(
        api_config.PHENTRIEVE_PUBLIC_HOSTED_MODE
        or api_config.PHENTRIEVE_REQUIRE_RESEARCH_ACK
    )


def require_research_use_acknowledgement(request: Request) -> None:
    """Require a research-use acknowledgement header when configured."""
    if not is_research_ack_required():
        return

    if request.headers.get(RESEARCH_ACK_HEADER, "").lower() == "true":
        return

    raise HTTPException(
        status_code=status.HTTP_428_PRECONDITION_REQUIRED,
        detail={
            "error_code": "research_use_ack_required",
            "message": (
                "This endpoint is available for research use only. Send "
                f"{RESEARCH_ACK_HEADER_DISPLAY}: true after presenting the "
                "research-use limitation to the user."
            ),
            "required_header": RESEARCH_ACK_HEADER_DISPLAY,
            "intended_use": RESEARCH_USE_LIMITATION,
        },
    )
