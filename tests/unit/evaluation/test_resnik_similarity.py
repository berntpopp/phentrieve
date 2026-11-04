"""Hybrid similarity uses graph distances to the LCA, not depth differences (#475)."""

import pytest

from phentrieve.evaluation import metrics
from phentrieve.evaluation.metrics import calculate_resnik_similarity

# R -> A -> B -> E, R -> L, X is a child of E and of L.
# X has shortest-path depth 2 (via L), its ancestor E has depth 3.
_ANCESTORS = {
    "R": {"R"},
    "A": {"A", "R"},
    "B": {"B", "A", "R"},
    "E": {"E", "B", "A", "R"},
    "L": {"L", "R"},
    "X": {"X", "E", "B", "A", "R", "L"},
    "Y": {"Y", "E", "B", "A", "R"},
}
_DEPTHS = {"R": 0, "A": 1, "B": 2, "E": 3, "L": 1, "X": 2, "Y": 4}


@pytest.fixture
def tiny_dag(monkeypatch):
    monkeypatch.setattr(metrics, "load_hpo_graph_data", lambda: (_ANCESTORS, _DEPTHS))


def test_descendant_with_shorter_depth_is_not_identical(tiny_dag):
    # Old formula: depth(X) - depth(E) = -1, clamped to 0, similarity 1.0.
    assert calculate_resnik_similarity("X", "E") < 1.0


def test_cancelling_depth_differences_are_not_identical(tiny_dag):
    # Old formula: (4 - 3) + (2 - 3) = 0, similarity 1.0 without a warning.
    assert calculate_resnik_similarity("Y", "X") < 1.0


def test_identical_terms_are_one(tiny_dag):
    assert calculate_resnik_similarity("X", "X") == 1.0


def test_real_hpo_pairs_from_issue_are_below_one():
    ancestors, _ = metrics.load_hpo_graph_data()
    pairs = [
        ("HP:0002202", "HP:0010742"),
        ("HP:0010742", "HP:0000969"),
        ("HP:0001744", "HP:0025408"),
    ]
    if not ancestors or any(t not in ancestors for pair in pairs for t in pair):
        pytest.skip("Real HPO graph data is not available")

    for term1, term2 in pairs:
        assert calculate_resnik_similarity(term1, term2) < 1.0
