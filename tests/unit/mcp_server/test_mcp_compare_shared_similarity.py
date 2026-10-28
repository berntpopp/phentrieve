"""compare_hpo_terms_service uses the shared similarity computation (#415).

Runs against a tiny prepared HPO database, so it does not depend on
`phentrieve data prepare` having been run.
"""

from __future__ import annotations

import pytest

from api.mcp.envelope import McpToolError
from api.mcp.service_adapters import compare_hpo_terms_service
from phentrieve.evaluation.metrics import SimilarityFormula, load_hpo_graph_data
from phentrieve.evaluation.term_similarity import compute_term_similarity
from tests.unit.tiny_hpo_db import use_hpo_data_dir, write_tiny_hpo_db

pytestmark = pytest.mark.unit

T1, T2 = "HP:0001250", "HP:0002133"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    use_hpo_data_dir(monkeypatch, directory)
    yield directory
    load_hpo_graph_data.cache_clear()


def test_compact_payload_matches_shared_computation(data_dir):
    write_tiny_hpo_db(data_dir)

    res = compare_hpo_terms_service(term1_id=T1, term2_id=T2, formula="hybrid")

    shared = compute_term_similarity(T1, T2, SimilarityFormula.HYBRID)
    assert res == {
        "term1_id": T1,
        "term2_id": T2,
        "formula_used": "hybrid",
        "similarity_score": shared.similarity_score,
    }


def test_lca_details_shape_and_values(data_dir):
    write_tiny_hpo_db(data_dir)

    res = compare_hpo_terms_service(
        term1_id=T1, term2_id=T2, formula="hybrid", include_lca_details=True
    )

    assert res["lca_details"] == {
        "mica": {
            "hpo_id": "HP:0000707",
            "label": "Abnormality of the nervous system",
            "depth": 2,
        },
        "lca_depth": 2,
        "term1": {
            "hpo_id": T1,
            "label": "Seizure",
            "depth": 3,
            "normalized_depth": 1.0,
        },
        "term2": {
            "hpo_id": T2,
            "label": "Status epilepticus",
            "depth": 3,
            "normalized_depth": 1.0,
        },
        "path_length": 2,
    }


def test_unknown_term_is_not_found(data_dir):
    write_tiny_hpo_db(data_dir)

    with pytest.raises(McpToolError) as exc_info:
        compare_hpo_terms_service(term1_id=T1, term2_id="HP:9999999", formula="hybrid")
    assert exc_info.value.error_code == "not_found"
    assert "HP:9999999" in exc_info.value.message


def test_missing_ontology_data_is_temporarily_unavailable(data_dir):
    with pytest.raises(McpToolError) as exc_info:
        compare_hpo_terms_service(term1_id=T1, term2_id=T2, formula="hybrid")
    assert exc_info.value.error_code == "temporarily_unavailable"
