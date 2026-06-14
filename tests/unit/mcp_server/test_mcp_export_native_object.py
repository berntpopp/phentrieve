"""D4 -- export returns a native phenopacket object (string kept for compat).
D11 -- client-supplied phenotypes get honest provenance, not ``legacy_dict``.
"""

import json

import pytest

from api.mcp.service_adapters import export_phenopacket_service

pytestmark = pytest.mark.unit


def _export(*, sidecar=False):
    return export_phenopacket_service(
        case_id="CASE-1",
        case_label=None,
        input_text=None,
        subject=None,
        phenotypes=[
            {"hpo_id": "HP:0001250", "label": "Seizure", "assertion": "affirmed"}
        ],
        include_annotation_sidecar=sidecar,
    )


def test_export_returns_native_phenopacket_object():
    out = _export()
    assert isinstance(out.get("phenopacket"), dict), "native phenopacket object missing"
    assert out["phenopacket"].get("id")
    # string form kept for back-compat and must be equivalent
    assert json.loads(out["phenopacket_json"]) == out["phenopacket"]


def test_client_supplied_provenance_is_honest():
    out = _export(sidecar=True)
    blob = json.dumps(out)
    assert "legacy_dict" not in blob
    assert "client_supplied" in blob
