"""L1: empty / whitespace-only text is rejected at validation time instead of
being silently accepted and returning success with empty results.

B3: a blank-text rejection must surface as a value-level validation envelope
(the actual reason), not the unknown-argument-NAME template that lists the
parameter names in allowed_values.
"""

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from api.mcp.envelope import build_arg_error_envelope
from api.mcp.middleware import ArgValidationMiddleware
from api.mcp.tools._common import LanguageArg, TextArg

pytestmark = pytest.mark.unit

_TA = TypeAdapter(TextArg)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", "  \n  "])
def test_blank_text_rejected(blank):
    with pytest.raises(ValidationError):
        _TA.validate_python(blank)


def test_real_text_accepted():
    assert _TA.validate_python("progressive ataxia") == "progressive ataxia"


def test_value_error_envelope_uses_message_not_param_names():
    """B3: build_arg_error_envelope must emit the value-level message for a
    value error on a *valid* argument, without allowed_values=param-names."""
    env = build_arg_error_envelope(
        tool_name="phentrieve_search_hpo_terms",
        loc="text",
        error_type="value_error",
        valid_params=["text", "language", "num_results"],
        signature="phentrieve_search_hpo_terms(text, language=, num_results=)",
        suggestion=None,
        constraints=None,
        value_message="text must not be empty or whitespace-only.",
    )
    assert env["error_code"] == "validation_failed"
    assert env["field"] == "text"
    assert "empty" in env["message"].lower()
    # The signature hint is preserved...
    assert env["hint"] == "phentrieve_search_hpo_terms(text, language=, num_results=)"
    # ...but allowed_values must NOT enumerate the parameter names.
    allowed = env.get("allowed_values", [])
    assert "language" not in allowed
    assert "num_results" not in allowed


class _ToolArgs(BaseModel):
    text: TextArg
    language: LanguageArg = None


@pytest.mark.parametrize("blank", ["   ", ""])
def test_blank_text_error_result_is_value_level(blank):
    """B3 end-to-end: the middleware's error envelope for a blank-text
    ValidationError is value-level, not the unknown-arg-name template."""
    with pytest.raises(ValidationError) as ei:
        _ToolArgs(text=blank)
    schema = _ToolArgs.model_json_schema()
    result = ArgValidationMiddleware()._error_result(
        "phentrieve_search_hpo_terms", ["text", "language"], schema, ei.value
    )
    env = result.structured_content
    assert env["error_code"] == "validation_failed"
    assert env["field"] == "text"
    # Not the name-error template.
    assert "argument names are listed" not in env["message"].lower()
    allowed = env.get("allowed_values", [])
    assert "text" not in allowed
    assert "language" not in allowed
