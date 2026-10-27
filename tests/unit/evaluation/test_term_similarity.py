"""Tests for the shared HPO term-similarity computation (#415, #417)."""

from __future__ import annotations

import pytest

from phentrieve.evaluation.metrics import (
    SimilarityFormula,
    calculate_semantic_similarity,
    load_hpo_graph_data,
)
from phentrieve.evaluation.term_similarity import (
    HPODataUnavailableError,
    HPOTermNotFoundError,
    compute_term_similarity,
    load_hpo_labels,
)
from tests.unit.tiny_hpo_db import use_hpo_data_dir, write_tiny_hpo_db

pytestmark = pytest.mark.unit


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    use_hpo_data_dir(monkeypatch, directory)
    yield directory
    load_hpo_graph_data.cache_clear()


def test_computes_score_labels_and_lca(data_dir):
    write_tiny_hpo_db(data_dir)

    result = compute_term_similarity(
        "HP:0001250", "HP:0002133", SimilarityFormula.HYBRID, with_lca=True
    )

    assert result.term1_label == "Seizure"
    assert result.term2_label == "Status epilepticus"
    assert result.similarity_score == calculate_semantic_similarity(
        "HP:0001250", "HP:0002133", formula=SimilarityFormula.HYBRID
    )
    assert 0.0 < result.similarity_score < 1.0
    assert (result.lca_id, result.lca_label, result.lca_depth) == (
        "HP:0000707",
        "Abnormality of the nervous system",
        2,
    )
    assert (result.term1_depth, result.term2_depth, result.max_depth) == (3, 3, 3)


def test_without_lca_leaves_lca_fields_empty(data_dir):
    write_tiny_hpo_db(data_dir)

    result = compute_term_similarity("HP:0001250", "HP:0002133")

    assert result.lca_id is None
    assert result.lca_label is None
    assert result.lca_depth is None
    assert result.formula is SimilarityFormula.HYBRID


def test_missing_terms_raise_not_found(data_dir):
    write_tiny_hpo_db(data_dir)

    with pytest.raises(HPOTermNotFoundError) as exc_info:
        compute_term_similarity("HP:0001250", "HP:9999999")
    assert exc_info.value.missing == ["HP:9999999"]


def test_missing_data_raises_unavailable(data_dir):
    with pytest.raises(HPODataUnavailableError):
        compute_term_similarity("HP:0001250", "HP:0002133")
    assert load_hpo_labels(["HP:0001250"]) == {}


def test_data_prepared_after_a_failed_load_is_picked_up(data_dir):
    """#417: the "data missing" result must not be cached."""
    assert load_hpo_graph_data() == ({}, {})
    with pytest.raises(HPODataUnavailableError):
        compute_term_similarity("HP:0001250", "HP:0002133")

    write_tiny_hpo_db(data_dir)

    ancestors, depths = load_hpo_graph_data()
    assert depths["HP:0001250"] == 3
    result = compute_term_similarity("HP:0001250", "HP:0002133")
    assert result.term1_label == "Seizure"
