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
