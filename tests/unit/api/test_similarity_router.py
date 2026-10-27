"""Unit tests for the HPO term similarity router.

The router delegates to the shared ``compute_term_similarity`` (#415) and only
translates its errors into HTTP responses. A tiny prepared HPO database stands
in for the real ontology data.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.similarity_router import router
from phentrieve.evaluation.metrics import load_hpo_graph_data
from tests.unit.tiny_hpo_db import use_hpo_data_dir, write_tiny_hpo_db

pytestmark = pytest.mark.unit

URL = "/api/v1/similarity/HP:0001250/HP:0002133"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    use_hpo_data_dir(monkeypatch, directory)
    yield directory
    load_hpo_graph_data.cache_clear()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/similarity")
    return TestClient(app)


def test_returns_score_labels_and_lca(data_dir, client):
    write_tiny_hpo_db(data_dir)

    response = client.get(URL, params={"formula": "simple_resnik_like"})

    assert response.status_code == 200
    body = response.json()
    assert body["term1_id"] == "HP:0001250"
    assert body["term1_label"] == "Seizure"
    assert body["term2_id"] == "HP:0002133"
    assert body["term2_label"] == "Status epilepticus"
    assert body["formula_used"] == "simple_resnik_like"
    assert 0.0 < body["similarity_score"] < 1.0
    assert body["lca_details"] == {
        "id": "HP:0000707",
        "label": "Abnormality of the nervous system",
        "depth": 2,
    }
    assert body["error_message"] is None


def test_invalid_formula_falls_back_to_default(data_dir, client):
    write_tiny_hpo_db(data_dir)

    response = client.get(URL, params={"formula": "bogus"})

    assert response.status_code == 200
    assert response.json()["formula_used"] == "hybrid"


def test_unknown_term_returns_404_with_detail(data_dir, client):
    write_tiny_hpo_db(data_dir)

    response = client.get("/api/v1/similarity/HP:0001250/HP:9999999")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["term1_id"] == "HP:0001250"
    assert detail["term1_label"] == "Seizure"
    assert detail["term2_id"] == "HP:9999999"
    assert detail["term2_label"] is None
    assert detail["formula_used"] == "hybrid"
    assert detail["similarity_score"] == 0.0
    assert "HP:9999999" in detail["error_message"]


def test_missing_data_returns_503_then_recovers_once_prepared(data_dir, client):
    """#417: "data missing" must not be cached for the process lifetime."""
    response = client.get(URL)
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]

    write_tiny_hpo_db(data_dir)

    response = client.get(URL)
    assert response.status_code == 200
    assert response.json()["term1_label"] == "Seizure"


def test_unexpected_error_returns_500(data_dir, client, mocker):
    write_tiny_hpo_db(data_dir)
    mocker.patch(
        "api.routers.similarity_router.compute_term_similarity",
        side_effect=RuntimeError("boom"),
    )

    response = client.get(URL)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["error_message"] == "An internal server error occurred."
    assert "boom" not in response.text
