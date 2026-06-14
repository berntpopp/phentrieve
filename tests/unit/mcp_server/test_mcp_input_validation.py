"""L1: empty / whitespace-only text is rejected at validation time instead of
being silently accepted and returning success with empty results."""

import pytest
from pydantic import TypeAdapter, ValidationError

from api.mcp.tools._common import TextArg

pytestmark = pytest.mark.unit

_TA = TypeAdapter(TextArg)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", "  \n  "])
def test_blank_text_rejected(blank):
    with pytest.raises(ValidationError):
        _TA.validate_python(blank)


def test_real_text_accepted():
    assert _TA.validate_python("progressive ataxia") == "progressive ataxia"
