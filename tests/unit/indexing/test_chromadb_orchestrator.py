"""Tests for provenance propagation during index orchestration."""

import pytest

from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.indexing.chromadb_orchestrator import orchestrate_index_building

pytestmark = pytest.mark.unit


def test_orchestrator_passes_database_provenance_to_index_build(mocker, tmp_path):
    db_path = tmp_path / "hpo_data.db"
    db = HPODatabase(db_path)
    db.initialize_schema()
    db.set_metadata("hpo_version", "v2026-06-23")
    db.set_metadata("hpo_source_sha256", "a" * 64)
    db.close()

    mocker.patch(
        "phentrieve.indexing.chromadb_orchestrator.load_hpo_terms",
        return_value=[{"id": "HP:0000001"}],
    )
    mocker.patch(
        "phentrieve.indexing.chromadb_orchestrator.create_hpo_documents",
        return_value=(["All"], [{"hpo_id": "HP:0000001"}], ["HP:0000001"]),
    )
    mocker.patch(
        "phentrieve.indexing.chromadb_orchestrator.load_embedding_model",
        return_value=object(),
    )
    build_index = mocker.patch(
        "phentrieve.indexing.chromadb_orchestrator.build_chromadb_index",
        return_value=True,
    )

    success = orchestrate_index_building(
        model_name_arg="unit/model",
        data_dir_override=str(tmp_path),
        index_dir_override=str(tmp_path / "indexes"),
        model_revision="b" * 40,
    )

    assert success
    assert build_index.call_args.kwargs["hpo_version"] == "v2026-06-23"
    assert build_index.call_args.kwargs["hpo_source_sha256"] == "a" * 64
    assert build_index.call_args.kwargs["model_revision"] == "b" * 40
