"""Unit tests for phentrieve.analysis.ontology_fidelity."""

from __future__ import annotations

import math

import pytest

# ------------------------------------------------------------
# Shared tiny DAG used across tests.
#
#            HP:0000001  (root, depth 0)
#              /      \
#        HP:0001      HP:0002         (depth 1; top-level branches)
#          /  \          |
#     HP:0010 HP:0011  HP:0020        (depth 2)
#                        |
#                     HP:0030         (depth 3)
#
# HP:0011 has two parents: HP:0001 AND HP:0002  (multi-parent case).
# ------------------------------------------------------------


@pytest.fixture
def tiny_dag() -> tuple[dict[str, set[str]], dict[str, int]]:
    """Return (ancestors, depths) for the fixture DAG above.

    Convention matches HPODatabase.load_graph_data: ancestors[t] does NOT
    include t itself.
    """
    ancestors = {
        "HP:0000001": set(),
        "HP:0001": {"HP:0000001"},
        "HP:0002": {"HP:0000001"},
        "HP:0010": {"HP:0000001", "HP:0001"},
        "HP:0011": {"HP:0000001", "HP:0001", "HP:0002"},
        "HP:0020": {"HP:0000001", "HP:0002"},
        "HP:0030": {"HP:0000001", "HP:0002", "HP:0020"},
    }
    depths = {
        "HP:0000001": 0,
        "HP:0001": 1,
        "HP:0002": 1,
        "HP:0010": 2,
        "HP:0011": 2,
        "HP:0020": 2,
        "HP:0030": 3,
    }
    return ancestors, depths


def test_build_descendants_index_matches_manual_inverse(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import build_descendants_index

    ancestors, _ = tiny_dag
    descendants = build_descendants_index(ancestors)

    # Each term is its own descendant by our convention.
    assert descendants["HP:0030"] == {"HP:0030"}
    assert descendants["HP:0020"] == {"HP:0020", "HP:0030"}
    assert descendants["HP:0002"] == {"HP:0002", "HP:0011", "HP:0020", "HP:0030"}
    assert descendants["HP:0001"] == {"HP:0001", "HP:0010", "HP:0011"}
    assert descendants["HP:0000001"] == set(ancestors.keys())


def test_information_content_root_is_zero(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        information_content,
    )

    ancestors, _ = tiny_dag
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants, root="HP:0000001")

    assert ic["HP:0000001"] == pytest.approx(0.0)


def test_information_content_leaf_is_log_n(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        information_content,
    )

    ancestors, _ = tiny_dag
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants, root="HP:0000001")

    n_total = len(ancestors)
    # HP:0030 is a leaf: |descendants|=1, IC = -log(1/N) = log(N)
    assert ic["HP:0030"] == pytest.approx(math.log(n_total))


def test_information_content_monotone_with_depth(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        information_content,
    )

    ancestors, _ = tiny_dag
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants, root="HP:0000001")

    # A descendant must have IC >= its ancestor.
    assert ic["HP:0030"] >= ic["HP:0020"] >= ic["HP:0002"] >= ic["HP:0000001"]


def test_graph_shortest_path_identical_terms_is_zero(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import graph_shortest_path

    ancestors, depths = tiny_dag
    assert graph_shortest_path("HP:0030", "HP:0030", ancestors, depths) == 0


def test_graph_shortest_path_known_pairs(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import graph_shortest_path

    ancestors, depths = tiny_dag

    # HP:0010 -> HP:0001 -> HP:0011 : distance 2
    assert graph_shortest_path("HP:0010", "HP:0011", ancestors, depths) == 2
    # HP:0030 -> HP:0020 -> HP:0002 : distance 2
    assert graph_shortest_path("HP:0030", "HP:0002", ancestors, depths) == 2
    # HP:0010 (branch 1) and HP:0020 (branch 2): LCA is root (depth 0).
    # distance = 2 + 2 - 0 = 4
    assert graph_shortest_path("HP:0010", "HP:0020", ancestors, depths) == 4
    # Multi-parent: HP:0011 has two parents; LCA with HP:0020 is HP:0002 (depth 1).
    # depths: HP:0011=2, HP:0020=2; distance = 2 + 2 - 2*1 = 2
    assert graph_shortest_path("HP:0011", "HP:0020", ancestors, depths) == 2


def test_resnik_similarity_identical_terms_equals_own_ic(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        information_content,
        resnik_similarity,
    )

    ancestors, _ = tiny_dag
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)

    assert resnik_similarity("HP:0030", "HP:0030", ancestors, ic) == pytest.approx(
        ic["HP:0030"]
    )


def test_resnik_similarity_siblings_is_ic_of_parent(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        information_content,
        resnik_similarity,
    )

    ancestors, _ = tiny_dag
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)

    # HP:0010 and HP:0011 share HP:0001 as a parent; HP:0001 is also their
    # most-informative common ancestor in the tiny DAG (higher IC than root,
    # and HP:0011 shares HP:0002 with HP:0010 only via root).
    got = resnik_similarity("HP:0010", "HP:0011", ancestors, ic)
    assert got == pytest.approx(ic["HP:0001"])


def test_resnik_similarity_cross_branch_is_root_ic(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        information_content,
        resnik_similarity,
    )

    ancestors, _ = tiny_dag
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)

    # HP:0010 is only under HP:0001. HP:0020 is only under HP:0002. The only
    # common ancestor is the root, so Resnik = IC(root) = 0.
    assert resnik_similarity("HP:0010", "HP:0020", ancestors, ic) == pytest.approx(0.0)


def test_top_level_branch_for_depth_0_term_is_none(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import top_level_branch

    ancestors, depths = tiny_dag
    assert top_level_branch("HP:0000001", ancestors, depths) == (None, frozenset())


def test_top_level_branch_for_depth_1_term_is_self(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import top_level_branch

    ancestors, depths = tiny_dag
    branch, all_branches = top_level_branch("HP:0001", ancestors, depths)
    assert branch == "HP:0001"
    assert all_branches == frozenset({"HP:0001"})


def test_top_level_branch_single_parent_chain(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import top_level_branch

    ancestors, depths = tiny_dag
    branch, all_branches = top_level_branch("HP:0030", ancestors, depths)
    assert branch == "HP:0002"
    assert all_branches == frozenset({"HP:0002"})


def test_top_level_branch_multi_parent_picks_lex_smallest(tiny_dag):
    from phentrieve.analysis.ontology_fidelity import top_level_branch

    ancestors, depths = tiny_dag
    # HP:0011 has depth-1 ancestors HP:0001 and HP:0002; "HP:0001" < "HP:0002".
    branch, all_branches = top_level_branch("HP:0011", ancestors, depths)
    assert branch == "HP:0001"
    assert all_branches == frozenset({"HP:0001", "HP:0002"})


def test_sample_pairs_shape_and_bounds():
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import sample_pairs

    rng = np.random.default_rng(42)
    pairs = sample_pairs(n_terms=100, n_pairs=500, rng=rng)
    assert pairs.shape == (500, 2)
    assert pairs.min() >= 0
    assert pairs.max() < 100
    # No self-pairs.
    assert (pairs[:, 0] != pairs[:, 1]).all()


def test_sample_pairs_deterministic_given_seed():
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import sample_pairs

    a = sample_pairs(1000, 200, np.random.default_rng(7))
    b = sample_pairs(1000, 200, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_sample_pairs_clamps_when_requested_exceeds_available():
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import sample_pairs

    # For 5 terms there are only 5*4/2 = 10 unordered distinct pairs.
    pairs = sample_pairs(5, 100, np.random.default_rng(0))
    assert pairs.shape[0] <= 10
    assert (pairs[:, 0] != pairs[:, 1]).all()


def test_global_distance_correlation_returns_expected_keys(tiny_dag):
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        global_distance_correlation,
        information_content,
    )

    ancestors, depths = tiny_dag
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)

    term_ids = list(depths.keys())
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((len(term_ids), 16)).astype(np.float32)

    result = global_distance_correlation(
        term_ids, embeddings, ancestors, depths, ic, n_pairs=20, seed=0
    )

    assert set(result) >= {
        "spearman_shortest_path",
        "spearman_resnik",
        "n_pairs",
    }
    assert isinstance(result["n_pairs"], int)
    assert result["n_pairs"] <= 20


def test_global_distance_correlation_seeded_is_deterministic(tiny_dag):
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        global_distance_correlation,
        information_content,
    )

    ancestors, depths = tiny_dag
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)

    term_ids = list(depths.keys())
    rng = np.random.default_rng(7)
    embeddings = rng.standard_normal((len(term_ids), 16)).astype(np.float32)

    a = global_distance_correlation(
        term_ids, embeddings, ancestors, depths, ic, n_pairs=15, seed=11
    )
    b = global_distance_correlation(
        term_ids, embeddings, ancestors, depths, ic, n_pairs=15, seed=11
    )
    assert a == b


def test_per_term_fidelity_bounds_0_to_1(tiny_dag):
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        information_content,
        per_term_fidelity,
    )

    ancestors, depths = tiny_dag
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)

    term_ids = list(depths.keys())
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((len(term_ids), 8)).astype(np.float32)

    rows = per_term_fidelity(term_ids, embeddings, ancestors, descendants, ic, k=3)
    assert len(rows) == len(term_ids)
    for row in rows:
        assert 0.0 <= row["fidelity"] <= 1.0
        assert len(row["nn_embedding"]) == 3
        assert len(row["nn_dag"]) == 3
        # Self must be excluded from both neighborhoods.
        assert row["id"] not in row["nn_embedding"]
        assert row["id"] not in row["nn_dag"]


def test_per_term_fidelity_tiebreak_is_lexicographic():
    """Equal Resnik scores across candidates break by ascending HPO ID."""
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        information_content,
        per_term_fidelity,
    )

    # Flat DAG: root + three depth-1 children. All sibling pairs have
    # Resnik = IC(root) = 0, so ties. HP:0010 < HP:0020 < HP:0030.
    ancestors = {
        "HP:0000001": set(),
        "HP:0010": {"HP:0000001"},
        "HP:0020": {"HP:0000001"},
        "HP:0030": {"HP:0000001"},
    }
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)
    term_ids = ["HP:0000001", "HP:0010", "HP:0020", "HP:0030"]
    embeddings = np.eye(4, dtype=np.float32)

    rows = per_term_fidelity(term_ids, embeddings, ancestors, descendants, ic, k=2)
    by_id = {r["id"]: r for r in rows}
    # For HP:0030 the two lex-smallest non-self terms are HP:0000001 and HP:0010.
    assert by_id["HP:0030"]["nn_dag"] == ["HP:0000001", "HP:0010"]


def test_per_term_fidelity_identical_embeddings_yields_uniform_baseline():
    """If all embeddings are identical, embedding k-NN is just the lex-smallest
    non-self terms, which matches the Resnik-tiebreak fallback for a flat DAG --
    so fidelity should be 1.0 on every term."""
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import (
        build_descendants_index,
        information_content,
        per_term_fidelity,
    )

    ancestors = {
        "HP:0000001": set(),
        "HP:0010": {"HP:0000001"},
        "HP:0020": {"HP:0000001"},
        "HP:0030": {"HP:0000001"},
        "HP:0040": {"HP:0000001"},
    }
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)
    term_ids = sorted(ancestors.keys())
    embeddings = np.ones((len(term_ids), 4), dtype=np.float32)

    rows = per_term_fidelity(term_ids, embeddings, ancestors, descendants, ic, k=2)
    for row in rows:
        assert row["fidelity"] == pytest.approx(1.0)


def test_branch_knn_purity_monocluster_equals_one():
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import branch_knn_purity

    term_ids = ["HP:0010", "HP:0011", "HP:0012"]
    branch_map = {"HP:0010": "HP:0001", "HP:0011": "HP:0001", "HP:0012": "HP:0001"}
    # All identical embeddings -> every neighbor in the same branch.
    embeddings = np.ones((3, 4), dtype=np.float32)
    result = branch_knn_purity(term_ids, embeddings, branch_map, k=2)
    assert result["overall"] == pytest.approx(1.0)
    assert result["per_branch"]["HP:0001"] == pytest.approx(1.0)


def test_branch_knn_purity_excludes_root_from_denominator():
    """Terms whose branch is None (the HPO root) must not appear in per_branch
    totals and must not contribute to the overall mean."""
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import branch_knn_purity

    term_ids = ["HP:0000001", "HP:0010", "HP:0011"]
    branch_map = {"HP:0000001": None, "HP:0010": "HP:0001", "HP:0011": "HP:0001"}
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    result = branch_knn_purity(term_ids, embeddings, branch_map, k=1)
    assert "HP:0001" in result["per_branch"]
    assert None not in result["per_branch"]
    # HP:0010's 1-NN is HP:0011 (same branch) -> purity 1.0.
    # HP:0011's 1-NN is HP:0010 (same branch) -> purity 1.0.
    # HP:0000001 excluded from denominator.
    assert result["overall"] == pytest.approx(1.0)
    assert result["n_evaluated"] == 2
