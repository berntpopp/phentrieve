"""Tests for reproducible HPO preparation provenance."""

import hashlib
import json

import pytest

from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.data_processing.hpo_parser import (
    compute_file_sha256,
    prepare_hpo_data,
    verify_hpo_json_sha256,
)

pytestmark = pytest.mark.unit


def _minimal_hpo_json() -> dict:
    return {
        "graphs": [
            {
                "nodes": [
                    {"id": "HP:0000001", "lbl": "All", "meta": {}},
                    {
                        "id": "HP:0000118",
                        "lbl": "Phenotypic abnormality",
                        "meta": {},
                    },
                ],
                "edges": [
                    {
                        "sub": "HP:0000118",
                        "pred": "is_a",
                        "obj": "HP:0000001",
                    }
                ],
            }
        ]
    }


def test_verify_hpo_json_sha256_rejects_an_unexpected_file(tmp_path):
    hpo_path = tmp_path / "hp.json"
    hpo_path.write_text("known ontology input", encoding="utf-8")
    expected_sha256 = hashlib.sha256(hpo_path.read_bytes()).hexdigest()

    assert compute_file_sha256(hpo_path) == expected_sha256
    assert verify_hpo_json_sha256(hpo_path, expected_sha256) == expected_sha256

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_hpo_json_sha256(hpo_path, "0" * 64)


def test_prepare_records_pinned_hpo_provenance(tmp_path):
    hpo_path = tmp_path / "hp.json"
    db_path = tmp_path / "hpo.db"
    hpo_path.write_text(json.dumps(_minimal_hpo_json()), encoding="utf-8")
    expected_sha256 = hashlib.sha256(hpo_path.read_bytes()).hexdigest()

    success, error, resolved_version = prepare_hpo_data(
        hpo_file_path=hpo_path,
        db_path=db_path,
        hpo_version="v2026-06-23",
        expected_sha256=expected_sha256,
    )

    assert success, error
    assert resolved_version == "v2026-06-23"

    db = HPODatabase(db_path)
    assert db.get_metadata("hpo_version") == "v2026-06-23"
    assert db.get_metadata("hpo_release_date") == "2026-06-23"
    assert db.get_metadata("hpo_source_sha256") == expected_sha256
    assert db.get_metadata("hpo_download_date") is not None
    db.close()
