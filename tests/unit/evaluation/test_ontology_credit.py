import pytest

from phentrieve.evaluation.metrics import SimilarityFormula
from phentrieve.evaluation.ontology_credit import (
    MatchKind,
    OntologyCreditConfig,
    build_ontology_credit_config,
    calculate_pair_credit,
    load_hpo_graph_data,
)


def test_exact_pair_receives_full_credit(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: ({"HP:1": {"HP:1"}}, {"HP:1": 1}),
    )

    credit = calculate_pair_credit("HP:1", "HP:1")

    assert credit.credit == 1.0
    assert credit.match_kind == MatchKind.EXACT


def test_direct_descendant_receives_high_credit(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:parent": {"HP:root", "HP:parent"},
                "HP:child": {"HP:root", "HP:parent", "HP:child"},
            },
            {"HP:root": 0, "HP:parent": 1, "HP:child": 2},
        ),
    )

    credit = calculate_pair_credit("HP:child", "HP:parent")

    assert credit.credit == 0.95
    assert credit.match_kind == MatchKind.DESCENDANT
    assert credit.distance == 1


def test_descendant_credit_records_semantic_similarity(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:parent": {"HP:root", "HP:parent"},
                "HP:child": {"HP:root", "HP:parent", "HP:child"},
            },
            {"HP:root": 0, "HP:parent": 1, "HP:child": 2},
        ),
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.calculate_semantic_similarity",
        lambda *_args, **_kwargs: 0.42,
    )

    credit = calculate_pair_credit("HP:child", "HP:parent")

    assert credit.match_kind == MatchKind.DESCENDANT
    assert credit.semantic_similarity == pytest.approx(0.42)


def test_direct_ancestor_receives_lower_credit(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:parent": {"HP:root", "HP:parent"},
                "HP:child": {"HP:root", "HP:parent", "HP:child"},
            },
            {"HP:root": 0, "HP:parent": 1, "HP:child": 2},
        ),
    )

    credit = calculate_pair_credit("HP:parent", "HP:child")

    assert credit.credit == 0.85
    assert credit.match_kind == MatchKind.ANCESTOR
    assert credit.distance == 1


def test_dag_descendant_distance_uses_parent_edges_not_global_depth(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:parent": {"HP:root", "HP:parent"},
                "HP:mid": {"HP:root", "HP:parent", "HP:mid"},
                "HP:child": {"HP:root", "HP:parent", "HP:mid", "HP:child"},
            },
            {"HP:root": 0, "HP:parent": 1, "HP:mid": 2, "HP:child": 2},
        ),
    )

    credit = calculate_pair_credit("HP:child", "HP:parent")

    assert credit.credit == pytest.approx(0.90)
    assert credit.match_kind == MatchKind.DESCENDANT
    assert credit.distance == 2


def test_pair_credit_does_not_reuse_stale_parent_cache(monkeypatch):
    ancestors = {
        "HP:parent": {"HP:root", "HP:parent"},
        "HP:mid": {"HP:root", "HP:parent", "HP:mid"},
        "HP:child": {"HP:root", "HP:parent", "HP:child"},
    }
    depths = {"HP:root": 0, "HP:parent": 1, "HP:mid": 2, "HP:child": 2}
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (ancestors, depths),
    )

    direct_credit = calculate_pair_credit("HP:child", "HP:parent")
    ancestors["HP:child"] = {"HP:root", "HP:parent", "HP:mid", "HP:child"}

    updated_credit = calculate_pair_credit("HP:child", "HP:parent")

    assert direct_credit.distance == 1
    assert updated_credit.distance == 2


def test_assertion_independent_sibling_credit_uses_minimum(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:a": {"HP:root", "HP:parent", "HP:a"},
                "HP:b": {"HP:root", "HP:parent", "HP:b"},
                "HP:parent": {"HP:root", "HP:parent"},
            },
            {"HP:root": 0, "HP:parent": 1, "HP:a": 2, "HP:b": 2},
        ),
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.calculate_semantic_similarity",
        lambda *_args, **_kwargs: 0.2,
    )

    credit = calculate_pair_credit("HP:a", "HP:b")

    assert credit.credit == 0.65
    assert credit.match_kind == MatchKind.SIBLING


def test_dag_terms_sharing_non_parent_ancestor_are_not_siblings(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:p": {"HP:root", "HP:p"},
                "HP:q": {"HP:root", "HP:p", "HP:q"},
                "HP:r": {"HP:root", "HP:p", "HP:r"},
                "HP:a": {"HP:root", "HP:p", "HP:q", "HP:a"},
                "HP:b": {"HP:root", "HP:p", "HP:r", "HP:b"},
            },
            {
                "HP:root": 0,
                "HP:p": 1,
                "HP:q": 2,
                "HP:r": 2,
                "HP:a": 2,
                "HP:b": 2,
            },
        ),
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.calculate_semantic_similarity",
        lambda *_args, **_kwargs: 0.2,
    )

    credit = calculate_pair_credit("HP:a", "HP:b")

    assert credit.credit == 0.45
    assert credit.match_kind == MatchKind.COUSIN


def test_unrelated_below_floor_receives_zero(monkeypatch):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:a": {"HP:root", "HP:a"},
                "HP:b": {"HP:other", "HP:b"},
            },
            {"HP:root": 0, "HP:other": 0, "HP:a": 1, "HP:b": 1},
        ),
    )
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.calculate_semantic_similarity",
        lambda *_args, **_kwargs: 0.29,
    )

    credit = calculate_pair_credit("HP:a", "HP:b")

    assert credit.credit == 0.0
    assert credit.match_kind == MatchKind.UNRELATED


def test_semantic_similarity_failures_are_logged(monkeypatch, caplog):
    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.load_hpo_graph_data",
        lambda: (
            {
                "HP:a": {"HP:root", "HP:a"},
                "HP:b": {"HP:other", "HP:b"},
            },
            {"HP:root": 0, "HP:other": 0, "HP:a": 1, "HP:b": 1},
        ),
    )

    def fail_similarity(*_args, **_kwargs):
        raise RuntimeError("similarity backend failed")

    monkeypatch.setattr(
        "phentrieve.evaluation.ontology_credit.calculate_semantic_similarity",
        fail_similarity,
    )

    with caplog.at_level("WARNING"):
        credit = calculate_pair_credit("HP:a", "HP:b")

    assert credit.match_kind == MatchKind.UNRELATED
    assert "Failed to calculate semantic similarity" in caplog.text


def test_build_ontology_credit_config_validates_options():
    config = build_ontology_credit_config(
        semantic_floor=0.25,
        similarity_formula="simple_resnik_like",
    )

    assert config.semantic_floor == 0.25
    assert config.similarity_formula == SimilarityFormula.SIMPLE_RESNIK_LIKE


@pytest.mark.parametrize("semantic_floor", [-0.1, 1.1])
def test_build_ontology_credit_config_rejects_invalid_floor(semantic_floor):
    with pytest.raises(ValueError, match="ontology_semantic_floor"):
        build_ontology_credit_config(
            semantic_floor=semantic_floor,
            similarity_formula="hybrid",
        )


def test_build_ontology_credit_config_rejects_invalid_formula():
    with pytest.raises(ValueError, match="Invalid ontology similarity formula"):
        build_ontology_credit_config(
            semantic_floor=0.25,
            similarity_formula="simple",
        )


def test_real_hpo_intellectual_disability_child_credit():
    ancestors, _ = load_hpo_graph_data()
    if not ancestors:
        pytest.skip("HPO graph data is not available")
    if "HP:0001256" not in ancestors or "HP:0001249" not in ancestors:
        pytest.skip("Real HPO graph data lacks required test terms")

    credit = calculate_pair_credit(
        "HP:0001256",
        "HP:0001249",
        OntologyCreditConfig(semantic_floor=0.30),
    )

    assert credit.match_kind == MatchKind.DESCENDANT
    assert credit.credit >= 0.90
