"""Unit tests for the phenopacket export API router."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_phenopacket_router_exports_bundle_with_optional_sidecar(client):
    response = client.post(
        "/api/v1/phenopackets/export",
        json={
            "case_id": "case-1",
            "case_label": "Case 1",
            "input_text": "Patient had recurrent seizures.",
            "include_annotation_sidecar": True,
            "phenotypes": [
                {
                    "hpo_id": "HP:0001250",
                    "label": "Seizure",
                    "assertion_status": "affirmed",
                    "source_chunk_ids": [1],
                    "text_attributions": [
                        {
                            "chunk_id": 1,
                            "start_char": 8,
                            "end_char": 26,
                            "matched_text_in_chunk": "recurrent seizures",
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "phenopacket_json" in payload
    assert payload["annotation_sidecar"]["phenopacket_id"]
