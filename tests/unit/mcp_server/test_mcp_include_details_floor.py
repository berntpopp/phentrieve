"""M5: include_details=True is honored at compact verbosity.

Default response_mode is compact, which normally drops detail fields. When the
caller explicitly requests include_details=True, definition/synonyms must survive
compact (otherwise the advertised flag is a silent no-op at the default mode)."""

import pytest

from api.mcp.shaping import apply_response_mode

pytestmark = pytest.mark.unit

_PAYLOAD = {
    "results": [
        {
            "hpo_id": "HP:0001250",
            "label": "Seizure",
            "similarity": 0.9,
            "definition": "An epileptic seizure.",
            "synonyms": ["fit", "convulsion"],
        }
    ]
}


def test_compact_drops_details_by_default():
    out = apply_response_mode(_PAYLOAD, "compact")
    item = out["results"][0]
    assert "definition" not in item
    assert "synonyms" not in item


def test_compact_keeps_details_when_requested():
    out = apply_response_mode(
        _PAYLOAD, "compact", keep_detail_fields=("definition", "synonyms")
    )
    item = out["results"][0]
    assert item["definition"] == "An epileptic seizure."
    assert item["synonyms"] == ["fit", "convulsion"]


def test_minimal_keeps_requested_details_too():
    out = apply_response_mode(_PAYLOAD, "minimal", keep_detail_fields=("definition",))
    item = out["results"][0]
    assert item["definition"] == "An epileptic seizure."
