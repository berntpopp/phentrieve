"""Pure functions for ontology/embedding fidelity analysis.

No I/O, no matplotlib, no ChromaDB. Inputs are plain dicts and numpy arrays.

Conventions
-----------
- ancestors[t] does NOT include t itself (matches HPODatabase.load_graph_data).
- descendants[t] DOES include t itself (so |descendants| >= 1 always, IC
  is always well-defined).
"""

from __future__ import annotations

import math


def build_descendants_index(
    ancestors: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Invert an ancestors map into a descendants map.

    Each term is defined as its own descendant, so |descendants(t)| >= 1
    for every term.
    """
    descendants: dict[str, set[str]] = {t: {t} for t in ancestors}
    for term, anc_set in ancestors.items():
        for anc in anc_set:
            # anc may be outside the ancestors map only if the DAG is malformed;
            # ignore silently rather than crash — correlates with DB integrity.
            if anc in descendants:
                descendants[anc].add(term)
    return descendants


def information_content(
    descendants: dict[str, set[str]],
    root: str = "HP:0000001",
) -> dict[str, float]:
    """Resnik information content: IC(t) = -log(|descendants(t)| / |descendants(root)|).

    IC(root) == 0 exactly. Leaves have max IC.
    """
    if root not in descendants:
        raise KeyError(f"Root {root!r} not in descendants map")
    root_count = len(descendants[root])
    if root_count <= 0:
        raise ValueError("Root has zero descendants; descendants map is empty.")
    return {
        term: -math.log(len(desc) / root_count) for term, desc in descendants.items()
    }


def _include_self(ancestors_of_t: set[str], t: str) -> set[str]:
    """Return ancestors(t) ∪ {t}. Useful for LCA computation."""
    return ancestors_of_t | {t}


def graph_shortest_path(
    u: str,
    v: str,
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
) -> int:
    """Shortest-path distance on the HPO is_a DAG.

    dist(u, v) = depth(u) + depth(v) - 2 * depth(LCA(u, v))

    LCA is chosen as the common (ancestor-or-self) term with maximum depth.
    Ties between equal-depth common ancestors are irrelevant for the distance
    formula, so we do not break them deterministically.
    """
    if u == v:
        return 0
    common = _include_self(ancestors[u], u) & _include_self(ancestors[v], v)
    if not common:
        raise ValueError(
            f"No common ancestor between {u!r} and {v!r}; "
            "graph is disconnected or root not reachable."
        )
    lca_depth = max(depths[c] for c in common)
    return depths[u] + depths[v] - 2 * lca_depth


def resnik_similarity(
    u: str,
    v: str,
    ancestors: dict[str, set[str]],
    ic: dict[str, float],
) -> float:
    """Resnik similarity: IC of the most-informative common ancestor.

    Returns 0.0 when the only common ancestor is the root (IC(root) == 0).
    """
    if u == v:
        return ic[u]
    common = _include_self(ancestors[u], u) & _include_self(ancestors[v], v)
    if not common:
        raise ValueError(
            f"No common ancestor between {u!r} and {v!r}; "
            "graph is disconnected or root not reachable."
        )
    return max(ic[c] for c in common)
