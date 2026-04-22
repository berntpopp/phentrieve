"""Unit tests for the phenopacket export API router."""

import json

import pytest
from fastapi.testclient import TestClient
from google.protobuf.json_format import Parse
from phenopackets import Phenopacket

from api.main import app

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_phenopacket_router_exports_bundle_with_optional_sidecar_and_negation(client):
    response = client.post(
        "/api/v1/phenopackets/export",
        json={
            "case_id": "case-1",
            "case_label": "Case 1",
            "input_text": "Patient had recurrent seizures.",
            "include_annotation_sidecar": True,
            "subject": {
                "id": "patient-1",
                "sex": "FEMALE",
                "dateOfBirth": "2010-05-15T00:00:00.000Z",
            },
            "phenotypes": [
                {
                    "hpo_id": "HP:0001250",
                    "label": "Seizure",
                    "assertion_status": "negated",
                    "certainty": "high",
                    "confidence": 0.91,
                    "evidence_text": "Patient had recurrent seizures.",
                    "source_mode": "manual_review",
                    "match_method": "workspace_collection",
                    "source_chunk_ids": [1],
                    "text_attributions": [
                        {
                            "chunk_id": 1,
                            "start_char": 8,
                            "end_char": 26,
                            "matched_text_in_chunk": "recurrent seizures",
                        }
                    ],
                },
                {
                    "hpo_id": "HP:0001290",
                    "label": "Generalized tonic-clonic seizure",
                    "assertion_status": "affirmed",
                    "source_chunk_ids": [2],
                    "text_attributions": [
                        {
                            "chunk_id": 2,
                            "start_char": 31,
                            "end_char": 41,
                            "matched_text_in_chunk": "convulsion",
                        }
                    ],
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "phenopacket_json" in payload
    assert payload["annotation_sidecar"]["phenopacket_id"]

    phenopacket = json.loads(payload["phenopacket_json"])
    assert phenopacket["id"] == "case-1"
    assert phenopacket["phenotypicFeatures"][0]["type"]["id"] == "HP:0001250"
    assert phenopacket["phenotypicFeatures"][0]["excluded"] is True
    assert phenopacket["subject"] == {
        "id": "patient-1",
        "sex": "FEMALE",
        "dateOfBirth": "2010-05-15T00:00:00.000Z",
    }
    assert any(
        ref["id"] == "phentrieve:case_label" and ref["description"] == "Case 1"
        for ref in phenopacket["metaData"]["externalReferences"]
    )
    Parse(payload["phenopacket_json"], Phenopacket(), ignore_unknown_fields=False)

    annotations = payload["annotation_sidecar"]["annotations"]
    assert annotations[0]["assertion"] == "negated"
    assert annotations[0]["certainty"] == "high"
    assert annotations[0]["confidence"] == 0.91
    assert annotations[0]["evidence_text"] == "Patient had recurrent seizures."
    assert annotations[0]["chunk_refs"] == [1]
    assert annotations[0]["spans"] == [
        {
            "start_char": 8,
            "end_char": 26,
            "text": "recurrent seizures",
        }
    ]
    assert annotations[0]["provenance"] == {
        "source_mode": "manual_review",
        "match_method": "workspace_collection",
    }
    assert annotations[1]["chunk_refs"] == [2]


def test_phenopacket_router_maps_request_payload_to_exporter_shape(client, monkeypatch):
    captured_call = {}

    def fake_export_phenopacket_bundle(**kwargs):
        captured_call.update(kwargs)
        return {
            "phenopacket_json": '{"id":"packet-1"}',
            "annotation_sidecar": {
                "schema_version": "1.0.0",
                "artifact_type": "phenotype_annotation_bundle",
                "generated_by": {
                    "tool": "phentrieve",
                    "version": "0.0.0-test",
                },
                "phenopacket_id": "packet-1",
                "annotations": [],
            },
        }

    monkeypatch.setattr(
        "api.routers.phenopacket_router.export_phenopacket_bundle",
        fake_export_phenopacket_bundle,
    )

    response = client.post(
        "/api/v1/phenopackets/export",
        json={
            "case_id": "case-2",
            "case_label": "Case 2",
            "input_text": "No seizures were reported.",
            "include_annotation_sidecar": True,
            "subject": {
                "id": "subject-2",
                "sex": "MALE",
                "dateOfBirth": "2000-01-01T00:00:00.000Z",
            },
            "phenotypes": [
                {
                    "hpo_id": "HP:0001250",
                    "label": "Seizure",
                    "assertion_status": "negated",
                    "certainty": "high",
                    "confidence": 0.91,
                    "evidence_text": "No seizures were reported.",
                    "source_mode": "manual_review",
                    "match_method": "workspace_collection",
                    "source_chunk_ids": [4],
                    "text_attributions": [
                        {
                            "chunk_id": 4,
                            "start_char": 3,
                            "end_char": 11,
                            "matched_text_in_chunk": "seizures",
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert captured_call["input_text"] == "No seizures were reported."
    assert captured_call["include_annotation_sidecar"] is True
    assert captured_call["aggregated_results"] == [
        {
            "hpo_id": "HP:0001250",
            "label": "Seizure",
            "assertion": "negated",
            "assertion_status": "negated",
            "certainty": "high",
            "confidence": 0.91,
            "evidence_text": "No seizures were reported.",
            "source_mode": "manual_review",
            "match_method": "workspace_collection",
            "chunk_refs": [4],
            "spans": [
                {
                    "start_char": 3,
                    "end_char": 11,
                    "evidence_text": "seizures",
                    "chunk_refs": [4],
                }
            ],
        }
    ]


def test_phenopacket_router_response_matches_declared_response_model(client):
    response = client.post(
        "/api/v1/phenopackets/export",
        json={
            "case_id": "case-3",
            "case_label": "Case 3",
            "phenotypes": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"phenopacket_json", "annotation_sidecar"}
    assert isinstance(payload["phenopacket_json"], str)
    assert payload["annotation_sidecar"] is None


@pytest.mark.parametrize(
    ("subject_payload", "expected_error_fragment"),
    [
        (
            {
                "id": "subject-invalid-sex",
                "sex": "ROBOT",
                "dateOfBirth": "2010-05-15T00:00:00.000Z",
            },
            "subject.sex",
        ),
        (
            {
                "id": "subject-invalid-date",
                "sex": "FEMALE",
                "dateOfBirth": "not-a-date",
            },
            "subject.dateOfBirth",
        ),
    ],
)
def test_phenopacket_router_rejects_invalid_subject_metadata(
    client, subject_payload, expected_error_fragment
):
    response = client.post(
        "/api/v1/phenopackets/export",
        json={
            "case_id": "case-invalid-subject",
            "subject": subject_payload,
            "phenotypes": [],
        },
    )

    assert response.status_code == 422
    assert expected_error_fragment in response.text
