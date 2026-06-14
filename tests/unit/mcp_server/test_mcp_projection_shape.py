"""D7/D13 -- projected aggregated terms carry text_attributions uniformly and
drop null padding so the MCP payload is not bloated with default-valued keys.
"""

import pytest

from api.mcp.projection import project_aggregated_terms_for_mcp

pytestmark = pytest.mark.unit


def test_text_attributions_key_always_present():
    out = project_aggregated_terms_for_mcp(
        [
            {
                "id": "HP:1",
                "name": "A",
                "score": 0.9,
                "assertion_status": "affirmed",
                "text_attributions": [{"start_char": 0, "end_char": 1}],
            },
            {"id": "HP:2", "name": "B", "score": 0.8, "assertion_status": "affirmed"},
        ]
    )
    assert all("text_attributions" in t for t in out)
    assert out[1]["text_attributions"] == []


def test_null_padding_dropped():
    out = project_aggregated_terms_for_mcp(
        [
            {
                "id": "HP:1",
                "name": "A",
                "score": 0.9,
                "assertion_status": "affirmed",
                "start_char": None,
                "end_char": None,
            }
        ]
    )[0]
    assert "start_char" not in out
    assert "end_char" not in out
    assert out["hpo_id"] == "HP:1"
