# Ontology–Embedding Fidelity Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`.planning/specs/2026-04-17-ontology-embedding-fidelity-design.md`](../specs/2026-04-17-ontology-embedding-fidelity-design.md) (commit `1bb3326`)
**Issue:** [#34](https://github.com/berntpopp/phentrieve/issues/34)
**Branch:** `feat/ontology-embedding-fidelity` (in a git worktree at `~/worktrees/phentrieve-ontology-fidelity/`)

**Goal:** Ship a standalone Python script (`scripts/analyze_embedding_ontology.py`) that loads BioLORD-2023-M embeddings for indexed HPO terms, computes four ontology-fidelity metrics, and writes a timestamped directory of correlation results + five plots, answering whether the language model recapitulates the curated HPO DAG.

**Architecture:** A new pure-function library module `phentrieve/analysis/ontology_fidelity.py` holds the testable math (graph distances, Resnik similarity, per-term fidelity, distance correlation). A thin cache module `phentrieve/analysis/embedding_cache.py` reads embeddings from the existing ChromaDB collection on first use and writes `.npy`/`.json` to a per-model cache directory. The orchestrating `scripts/analyze_embedding_ontology.py` wires them together and handles UMAP + plotting.

**Tech Stack:** Python 3.10+, numpy, scikit-learn, scipy (transitive), matplotlib, umap-learn (new, optional extra), plotly (new, optional extra), pytest, uv.

---

## File Structure

**Create:**
- `phentrieve/analysis/__init__.py` — empty package marker
- `phentrieve/analysis/ontology_fidelity.py` — pure math (descendants, IC, shortest-path, Resnik, top-level branch, sampling, correlations, fidelity, purity, depth corr)
- `phentrieve/analysis/embedding_cache.py` — ChromaDB → `.npy` + `hpo_ids.json` cache
- `scripts/analyze_embedding_ontology.py` — orchestrator (arg parsing, HPO DB access, cache call, metrics assembly, UMAP, plotting, writers)
- `tests/unit/analysis/__init__.py` — empty package marker
- `tests/unit/analysis/test_ontology_fidelity.py` — unit tests for math module
- `tests/unit/analysis/test_embedding_cache.py` — unit tests for cache module
- `tests/integration/analysis/__init__.py` — empty package marker
- `tests/integration/analysis/test_analyze_script_smoke.py` — end-to-end smoke test on 200-term subset

**Modify:**
- `pyproject.toml` — add `[project.optional-dependencies]` group `analysis` with `umap-learn>=0.5.6` and `plotly>=5.22.0`
- `scripts/README.md` — document the script (invocation + outputs)

**Not modified:** `phentrieve/config.py`, `phentrieve/utils.py`, `phentrieve/data_processing/*`, any CLI command module. This feature intentionally sits outside the CLI surface.

---

## Task 0: Worktree setup

**Files:**
- Operates on the parent repo, not the worktree yet.

- [ ] **Step 1: Create the worktree and branch**

```bash
cd /home/bernt-popp/development/phentrieve
git fetch origin
git worktree add -b feat/ontology-embedding-fidelity ~/worktrees/phentrieve-ontology-fidelity main
```

Expected: `Preparing worktree (new branch 'feat/ontology-embedding-fidelity')` and `HEAD is now at <sha> ...`.

- [ ] **Step 2: Switch into the worktree and verify branch**

```bash
cd ~/worktrees/phentrieve-ontology-fidelity
git status
git branch --show-current
```

Expected: clean working tree on `feat/ontology-embedding-fidelity`.

- [ ] **Step 3: Install dev dependencies in the worktree**

```bash
make install-dev
```

Expected: `uv sync` completes without errors. **All subsequent commands run from `~/worktrees/phentrieve-ontology-fidelity`** unless stated otherwise.

- [ ] **Step 4: Confirm existing test suite is green on the base branch**

```bash
make check && make typecheck-fast && make test
```

Expected: all three pass. If any fail, stop — do not begin implementation on an already-broken base.

---

## Task 1: Scaffold `phentrieve/analysis` package

**Files:**
- Create: `phentrieve/analysis/__init__.py`
- Create: `tests/unit/analysis/__init__.py`
- Create: `tests/integration/analysis/__init__.py`

- [ ] **Step 1: Create the package markers**

```bash
mkdir -p phentrieve/analysis tests/unit/analysis tests/integration/analysis
```

Write `phentrieve/analysis/__init__.py`:

```python
"""Analysis helpers for ontology/embedding correlation."""
```

Write `tests/unit/analysis/__init__.py` and `tests/integration/analysis/__init__.py`: each is a single empty line (or just `""`). Keep them minimal.

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "import phentrieve.analysis; print(phentrieve.analysis.__doc__)"
```

Expected: prints `Analysis helpers for ontology/embedding correlation.`

- [ ] **Step 3: Commit**

```bash
git add phentrieve/analysis/__init__.py tests/unit/analysis/__init__.py tests/integration/analysis/__init__.py
git commit -m "feat(analysis): scaffold analysis package"
```

---

## Task 2: `information_content` + `build_descendants_index` (TDD)

**Files:**
- Create: `phentrieve/analysis/ontology_fidelity.py`
- Create: `tests/unit/analysis/test_ontology_fidelity.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/analysis/test_ontology_fidelity.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

Expected: 4 failures with `ModuleNotFoundError: No module named 'phentrieve.analysis.ontology_fidelity'` (or `ImportError`).

- [ ] **Step 3: Write the minimal implementation**

Write `phentrieve/analysis/ontology_fidelity.py`:

```python
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
from collections.abc import Iterable


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
        term: -math.log(len(desc) / root_count)
        for term, desc in descendants.items()
    }
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run lint and format**

```bash
make lint-fix && make format
```

Expected: no errors (formatters may touch files; review with `git diff`).

- [ ] **Step 6: Commit**

```bash
git add phentrieve/analysis/ontology_fidelity.py tests/unit/analysis/test_ontology_fidelity.py
git commit -m "feat(analysis): add descendants index and information content"
```

---

## Task 3: `graph_shortest_path` (TDD)

**Files:**
- Modify: `phentrieve/analysis/ontology_fidelity.py`
- Modify: `tests/unit/analysis/test_ontology_fidelity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/analysis/test_ontology_fidelity.py`:

```python
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
```

- [ ] **Step 2: Run and confirm failure**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py::test_graph_shortest_path_known_pairs -v
```

Expected: `ImportError: cannot import name 'graph_shortest_path' ...`.

- [ ] **Step 3: Implement**

Append to `phentrieve/analysis/ontology_fidelity.py`:

```python
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
```

- [ ] **Step 4: Run tests and confirm pass**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

Expected: all existing tests still pass, plus 2 new.

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add phentrieve/analysis/ontology_fidelity.py tests/unit/analysis/test_ontology_fidelity.py
git commit -m "feat(analysis): add graph_shortest_path via LCA"
```

---

## Task 4: `resnik_similarity` (TDD)

**Files:**
- Modify: `phentrieve/analysis/ontology_fidelity.py`
- Modify: `tests/unit/analysis/test_ontology_fidelity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/analysis/test_ontology_fidelity.py`:

```python
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
```

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v -k resnik
```

Expected: 3 failures on `ImportError`.

- [ ] **Step 3: Implement**

Append to `phentrieve/analysis/ontology_fidelity.py`:

```python
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
```

- [ ] **Step 4: Run and confirm pass**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add phentrieve/analysis/ontology_fidelity.py tests/unit/analysis/test_ontology_fidelity.py
git commit -m "feat(analysis): add resnik_similarity"
```

---

## Task 5: `top_level_branch` with multi-parent determinism (TDD)

**Files:**
- Modify: `phentrieve/analysis/ontology_fidelity.py`
- Modify: `tests/unit/analysis/test_ontology_fidelity.py`

- [ ] **Step 1: Add failing tests**

Append:

```python
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
```

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v -k top_level_branch
```

Expected: 4 failures.

- [ ] **Step 3: Implement**

Append:

```python
def top_level_branch(
    term_id: str,
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
) -> tuple[str | None, frozenset[str]]:
    """Return (deterministic_branch, all_depth_1_ancestors_including_self_if_depth1).

    Rules (from spec's determinism section):
    - depth 0 (the root): returns (None, empty frozenset).
    - depth 1: returns (term_id, {term_id}).
    - depth > 1: collect all depth-1 ancestors of term_id; the deterministic
      branch is the lexicographically smallest HPO ID in that set.
    """
    d = depths.get(term_id)
    if d is None:
        raise KeyError(f"Unknown term {term_id!r}")
    if d == 0:
        return None, frozenset()
    if d == 1:
        return term_id, frozenset({term_id})

    depth_one_ancs = {a for a in ancestors[term_id] if depths.get(a) == 1}
    if not depth_one_ancs:
        # Should not happen in a well-formed HPO DAG, but guard anyway.
        return None, frozenset()
    return min(depth_one_ancs), frozenset(depth_one_ancs)
```

- [ ] **Step 4: Run and confirm pass**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add phentrieve/analysis/ontology_fidelity.py tests/unit/analysis/test_ontology_fidelity.py
git commit -m "feat(analysis): add top_level_branch with multi-parent tiebreak"
```

---

## Task 6: `sample_pairs` (TDD)

**Files:**
- Modify: `phentrieve/analysis/ontology_fidelity.py`
- Modify: `tests/unit/analysis/test_ontology_fidelity.py`

- [ ] **Step 1: Add failing tests**

Append:

```python
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
```

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v -k sample_pairs
```

Expected: 3 failures on `ImportError`.

- [ ] **Step 3: Implement**

Append to `phentrieve/analysis/ontology_fidelity.py`:

```python
import numpy as np  # noqa: E402 — kept near use-sites for readers


def sample_pairs(
    n_terms: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return `n_pairs` distinct unordered index pairs in [0, n_terms), as (n_pairs, 2) int64.

    - No self-pairs ((i, i) excluded).
    - If n_pairs exceeds the total number of distinct unordered pairs, the
      result is clamped to the maximum available (len <= n_terms*(n_terms-1)/2).
    - Deterministic given the same rng state.
    """
    if n_terms < 2:
        return np.empty((0, 2), dtype=np.int64)

    max_pairs = n_terms * (n_terms - 1) // 2
    target = min(n_pairs, max_pairs)

    # Sample with rejection — simpler and faster than enumerating pairs for
    # realistic n_terms (~18k) and target (~50k). Fall back to exhaustive
    # enumeration when target is close to max_pairs.
    if target >= max_pairs * 0.5:
        rows, cols = np.triu_indices(n_terms, k=1)
        all_pairs = np.stack([rows, cols], axis=1)
        idx = rng.choice(len(all_pairs), size=target, replace=False)
        return all_pairs[idx].astype(np.int64)

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(target * 10, 1000)
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        a, b = rng.integers(0, n_terms, size=2)
        if a == b:
            continue
        u, v = (int(a), int(b)) if a < b else (int(b), int(a))
        if (u, v) in seen:
            continue
        seen.add((u, v))
        out.append((u, v))

    return np.array(out, dtype=np.int64)
```

- [ ] **Step 4: Confirm pass**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add phentrieve/analysis/ontology_fidelity.py tests/unit/analysis/test_ontology_fidelity.py
git commit -m "feat(analysis): add deterministic pair sampling"
```

---

## Task 7: `global_distance_correlation` (TDD)

**Files:**
- Modify: `phentrieve/analysis/ontology_fidelity.py`
- Modify: `tests/unit/analysis/test_ontology_fidelity.py`

- [ ] **Step 1: Add failing tests**

Append:

```python
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

    a = global_distance_correlation(term_ids, embeddings, ancestors, depths, ic, n_pairs=15, seed=11)
    b = global_distance_correlation(term_ids, embeddings, ancestors, depths, ic, n_pairs=15, seed=11)
    assert a == b
```

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v -k global_distance
```

Expected: 2 failures on `ImportError`.

- [ ] **Step 3: Implement**

Append:

```python
def _cosine_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise cosine distance between paired rows of a and b. Shape (n,)."""
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    # Avoid division-by-zero; zero-norm rows get 1.0 (max distance).
    sim = np.where(denom > 0, num / np.maximum(denom, 1e-12), 0.0)
    return 1.0 - sim


def global_distance_correlation(
    term_ids: list[str],
    embeddings: np.ndarray,
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
    ic: dict[str, float],
    n_pairs: int = 50_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Spearman ρ between embedding cosine distance and (shortest-path, Resnik).

    Returns a dict with keys 'spearman_shortest_path', 'spearman_resnik',
    'n_pairs'. Uses sampled pairs; seed controls pair selection.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    pairs = sample_pairs(len(term_ids), n_pairs, rng)
    if pairs.size == 0:
        return {
            "spearman_shortest_path": float("nan"),
            "spearman_resnik": float("nan"),
            "n_pairs": 0,
        }

    u_ids = [term_ids[i] for i in pairs[:, 0]]
    v_ids = [term_ids[i] for i in pairs[:, 1]]

    cosine = _cosine_distance_rows(embeddings[pairs[:, 0]], embeddings[pairs[:, 1]])
    shortest = np.array(
        [graph_shortest_path(u, v, ancestors, depths) for u, v in zip(u_ids, v_ids)],
        dtype=np.float64,
    )
    resnik = np.array(
        [resnik_similarity(u, v, ancestors, ic) for u, v in zip(u_ids, v_ids)],
        dtype=np.float64,
    )

    # Spearman expects distance-like arrays to correlate in the same direction.
    # Cosine distance is larger for dissimilar pairs; shortest-path also larger;
    # so a positive ρ means agreement. Resnik is a similarity — negate for agreement.
    rho_sp, _ = spearmanr(cosine, shortest)
    rho_rk, _ = spearmanr(cosine, -resnik)

    return {
        "spearman_shortest_path": float(rho_sp) if rho_sp == rho_sp else float("nan"),
        "spearman_resnik": float(rho_rk) if rho_rk == rho_rk else float("nan"),
        "n_pairs": int(len(pairs)),
    }
```

- [ ] **Step 4: Confirm pass**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

Expected: all pass. (scipy is transitive via scikit-learn, already installed.)

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add phentrieve/analysis/ontology_fidelity.py tests/unit/analysis/test_ontology_fidelity.py
git commit -m "feat(analysis): add global distance correlation (Spearman ρ)"
```

---

## Task 8: `per_term_fidelity` with self-exclusion + tie-break (TDD)

**Files:**
- Modify: `phentrieve/analysis/ontology_fidelity.py`
- Modify: `tests/unit/analysis/test_ontology_fidelity.py`

- [ ] **Step 1: Add failing tests**

Append:

```python
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

    rows = per_term_fidelity(
        term_ids, embeddings, ancestors, descendants, ic, k=3
    )
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
    non-self terms, which matches the Resnik-tiebreak fallback for a flat DAG —
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
```

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v -k per_term_fidelity
```

Expected: 3 failures on `ImportError`.

- [ ] **Step 3: Implement**

Append:

```python
def _embedding_knn(
    embeddings: np.ndarray, k: int
) -> np.ndarray:
    """Return indices of the k nearest non-self neighbors per row, cosine metric.

    Shape: (n, k) of int64. Self is always excluded.
    """
    from sklearn.neighbors import NearestNeighbors

    # k+1 because the query point itself is always its own nearest neighbor.
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    _, idx = nn.kneighbors(embeddings, return_distance=True)
    # idx[:, 0] is the self-hit in almost all cases. Drop it robustly:
    n = idx.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        row = [j for j in idx[i] if j != i][:k]
        # Pad in case of degenerate metric behavior (duplicate self rows, etc.)
        while len(row) < k:
            row.append(row[-1] if row else 0)
        out[i] = row
    return out


def _resnik_top_k(
    query: str,
    candidates: list[str],
    ancestors: dict[str, set[str]],
    ic: dict[str, float],
    k: int,
) -> list[str]:
    """Return the k candidates with highest Resnik similarity to `query`,
    ties broken by ascending HPO ID. Query itself is excluded.
    """
    scored = []
    for c in candidates:
        if c == query:
            continue
        scored.append((-resnik_similarity(query, c, ancestors, ic), c))
    # (-resnik, id): primary ascending on -resnik (i.e. descending resnik);
    # secondary ascending on id — this is exactly the tie-break rule.
    scored.sort()
    return [c for _, c in scored[:k]]


def per_term_fidelity(
    term_ids: list[str],
    embeddings: np.ndarray,
    ancestors: dict[str, set[str]],
    descendants: dict[str, set[str]],
    ic: dict[str, float],
    k: int = 10,
) -> list[dict]:
    """Per-term fidelity: |embedding-kNN ∩ Resnik-top-k| / k, in [0, 1].

    Returns a list of {id, fidelity, nn_embedding, nn_dag} dicts, one per term,
    in the same order as `term_ids`.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    nn_idx = _embedding_knn(embeddings, k)
    rows: list[dict] = []
    for i, query in enumerate(term_ids):
        nn_embedding = [term_ids[j] for j in nn_idx[i]]
        nn_dag = _resnik_top_k(query, term_ids, ancestors, ic, k)
        overlap = len(set(nn_embedding) & set(nn_dag))
        rows.append(
            {
                "id": query,
                "fidelity": overlap / k,
                "nn_embedding": nn_embedding,
                "nn_dag": nn_dag,
            }
        )
    return rows
```

- [ ] **Step 4: Confirm pass**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add phentrieve/analysis/ontology_fidelity.py tests/unit/analysis/test_ontology_fidelity.py
git commit -m "feat(analysis): add per-term fidelity with self-exclusion and lex tiebreak"
```

---

## Task 9: `branch_knn_purity` (TDD)

**Files:**
- Modify: `phentrieve/analysis/ontology_fidelity.py`
- Modify: `tests/unit/analysis/test_ontology_fidelity.py`

- [ ] **Step 1: Add failing tests**

Append:

```python
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
```

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v -k branch_knn_purity
```

Expected: 2 failures on `ImportError`.

- [ ] **Step 3: Implement**

Append:

```python
def branch_knn_purity(
    term_ids: list[str],
    embeddings: np.ndarray,
    branch_map: dict[str, str | None],
    k: int = 10,
) -> dict:
    """Mean per-term branch purity over the embedding k-NN.

    Terms whose branch is None (i.e. the HPO root) are excluded from both the
    per-branch breakdown and the overall mean.

    Returns {'overall': float, 'per_branch': {branch_id: float}, 'n_evaluated': int}.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    nn_idx = _embedding_knn(embeddings, k)

    per_branch_scores: dict[str, list[float]] = {}
    all_scores: list[float] = []

    for i, query in enumerate(term_ids):
        query_branch = branch_map.get(query)
        if query_branch is None:
            continue
        neighbor_branches = [branch_map.get(term_ids[j]) for j in nn_idx[i]]
        matches = sum(1 for nb in neighbor_branches if nb == query_branch)
        score = matches / k
        all_scores.append(score)
        per_branch_scores.setdefault(query_branch, []).append(score)

    overall = float(np.mean(all_scores)) if all_scores else float("nan")
    per_branch = {b: float(np.mean(s)) for b, s in per_branch_scores.items()}
    return {
        "overall": overall,
        "per_branch": per_branch,
        "n_evaluated": len(all_scores),
    }
```

- [ ] **Step 4: Confirm pass**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add phentrieve/analysis/ontology_fidelity.py tests/unit/analysis/test_ontology_fidelity.py
git commit -m "feat(analysis): add branch k-NN purity"
```

---

## Task 10: `depth_correlation` (TDD)

**Files:**
- Modify: `phentrieve/analysis/ontology_fidelity.py`
- Modify: `tests/unit/analysis/test_ontology_fidelity.py`

- [ ] **Step 1: Add failing test**

Append:

```python
def test_depth_correlation_returns_scalar_in_expected_range():
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import depth_correlation

    term_ids = [f"HP:{i:07d}" for i in range(20)]
    depths = {tid: i % 5 for i, tid in enumerate(term_ids)}
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((20, 8)).astype(np.float32)

    rho = depth_correlation(term_ids, embeddings, depths)
    assert isinstance(rho, float)
    assert -1.0 <= rho <= 1.0
```

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v -k depth_correlation
```

- [ ] **Step 3: Implement**

Append:

```python
def depth_correlation(
    term_ids: list[str],
    embeddings: np.ndarray,
    depths: dict[str, int],
) -> float:
    """Spearman ρ between term depth and Euclidean distance from the aligned centroid.

    The centroid is computed on the provided embeddings (the aligned set),
    not on any larger matrix.
    """
    from scipy.stats import spearmanr

    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    depth_arr = np.array([depths[tid] for tid in term_ids], dtype=np.float64)
    rho, _ = spearmanr(depth_arr, dists)
    return float(rho) if rho == rho else float("nan")
```

- [ ] **Step 4: Confirm pass**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py -v
```

Expected: all tests pass, including the earlier ones.

- [ ] **Step 5: Run coverage check and commit**

```bash
uv run pytest tests/unit/analysis/test_ontology_fidelity.py --cov=phentrieve.analysis.ontology_fidelity --cov-report=term-missing
```

Expected: coverage on `ontology_fidelity.py` is ≥ 85% (spec target).

```bash
make lint-fix && make format
git add phentrieve/analysis/ontology_fidelity.py tests/unit/analysis/test_ontology_fidelity.py
git commit -m "feat(analysis): add depth correlation metric"
```

---

## Task 11: Embedding cache module (TDD)

**Files:**
- Create: `phentrieve/analysis/embedding_cache.py`
- Create: `tests/unit/analysis/test_embedding_cache.py`

- [ ] **Step 1: Write failing tests**

Write `tests/unit/analysis/test_embedding_cache.py`:

```python
"""Unit tests for phentrieve.analysis.embedding_cache."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_client_with_collection(ids, embeddings) -> MagicMock:
    """Return a mock chromadb.PersistentClient that exposes a collection
    whose .get(include=['embeddings']) returns the given ids/embeddings."""
    client = MagicMock()
    collection = MagicMock()
    collection.get.return_value = {
        "ids": list(ids),
        "embeddings": [list(row) for row in embeddings],
    }
    client.get_collection.return_value = collection
    return client


def test_embedding_cache_first_call_reads_chroma_and_writes_files(tmp_path):
    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    ids = ["HP:0000001", "HP:0000002", "HP:0000003"]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    client = _make_mock_client_with_collection(ids, vecs)

    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=client,
    ):
        got_ids, got_vecs = load_cached_embeddings(
            model_name="FremyCompany/BioLORD-2023-M",
            index_dir_override=str(tmp_path),
        )

    assert got_ids == ids
    np.testing.assert_allclose(got_vecs, vecs)

    cache_root = tmp_path / "ontology_fidelity_cache"
    subdirs = list(cache_root.iterdir())
    assert len(subdirs) == 1
    cache_dir = subdirs[0]
    assert (cache_dir / "embeddings.npy").exists()
    assert (cache_dir / "hpo_ids.json").exists()
    assert (cache_dir / "meta.json").exists()
    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["model_name"] == "FremyCompany/BioLORD-2023-M"
    assert meta["n_terms"] == 3
    assert meta["dim"] == 2


def test_embedding_cache_second_call_skips_chroma(tmp_path):
    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    ids = ["HP:0000001", "HP:0000002"]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    client = _make_mock_client_with_collection(ids, vecs)

    # Warm the cache.
    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=client,
    ):
        load_cached_embeddings("FremyCompany/BioLORD-2023-M", index_dir_override=str(tmp_path))

    # Second call must not construct PersistentClient at all.
    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        side_effect=AssertionError("should not be called"),
    ):
        got_ids, got_vecs = load_cached_embeddings(
            "FremyCompany/BioLORD-2023-M", index_dir_override=str(tmp_path)
        )

    assert got_ids == ids
    np.testing.assert_allclose(got_vecs, vecs)


def test_embedding_cache_refresh_overwrites(tmp_path):
    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    ids_v1 = ["HP:0000001", "HP:0000002"]
    vecs_v1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids_v2 = ["HP:0000001", "HP:0000002", "HP:0000003"]
    vecs_v2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=_make_mock_client_with_collection(ids_v1, vecs_v1),
    ):
        load_cached_embeddings("FremyCompany/BioLORD-2023-M", index_dir_override=str(tmp_path))

    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=_make_mock_client_with_collection(ids_v2, vecs_v2),
    ):
        got_ids, got_vecs = load_cached_embeddings(
            "FremyCompany/BioLORD-2023-M",
            refresh=True,
            index_dir_override=str(tmp_path),
        )

    assert got_ids == ids_v2
    assert got_vecs.shape == (3, 2)


def test_embedding_cache_missing_collection_raises(tmp_path):
    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    client = MagicMock()
    client.get_collection.side_effect = ValueError("not found")
    with patch(
        "phentrieve.analysis.embedding_cache.chromadb.PersistentClient",
        return_value=client,
    ):
        with pytest.raises(FileNotFoundError):
            load_cached_embeddings("FremyCompany/BioLORD-2023-M", index_dir_override=str(tmp_path))
```

- [ ] **Step 2: Run and confirm failure**

```bash
uv run pytest tests/unit/analysis/test_embedding_cache.py -v
```

Expected: 4 failures on `ImportError`.

- [ ] **Step 3: Implement**

Write `phentrieve/analysis/embedding_cache.py`:

```python
"""Read HPO embeddings once from ChromaDB, then serve from a local `.npy` cache.

Cache layout:
    <index_dir>/ontology_fidelity_cache/<collection_name>/
        embeddings.npy    # (N, D) float32
        hpo_ids.json      # ["HP:0000001", ...] aligned with embeddings rows
        meta.json         # {"model_name", "collection_name", "written_at",
                          #  "n_terms", "dim"}

index_dir defaults to phentrieve.utils.get_default_index_dir(), override via
`index_dir_override`. collection_name comes from
phentrieve.utils.generate_collection_name(model_name).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np

from phentrieve.utils import generate_collection_name, get_default_index_dir

logger = logging.getLogger(__name__)


def _cache_dir_for(index_dir: Path, collection_name: str) -> Path:
    return index_dir / "ontology_fidelity_cache" / collection_name


def _read_cache(cache_dir: Path) -> tuple[list[str], np.ndarray] | None:
    """Return (ids, embeddings) if the cache is valid, else None.

    Any missing/malformed file yields None (caller will refresh from Chroma).
    """
    emb_path = cache_dir / "embeddings.npy"
    ids_path = cache_dir / "hpo_ids.json"
    meta_path = cache_dir / "meta.json"
    if not (emb_path.exists() and ids_path.exists() and meta_path.exists()):
        return None
    try:
        embeddings = np.load(emb_path)
        ids = json.loads(ids_path.read_text())
        meta = json.loads(meta_path.read_text())
    except (ValueError, OSError, json.JSONDecodeError) as e:
        logger.error("Ontology-fidelity cache unreadable at %s: %s", cache_dir, e)
        return None
    if not isinstance(ids, list) or len(ids) != embeddings.shape[0]:
        logger.error(
            "Ontology-fidelity cache mismatch: %d ids vs %d embedding rows",
            len(ids) if isinstance(ids, list) else -1,
            embeddings.shape[0],
        )
        return None
    if meta.get("n_terms") != embeddings.shape[0]:
        logger.error("Ontology-fidelity cache meta.n_terms mismatch")
        return None
    return list(ids), embeddings


def _write_cache(
    cache_dir: Path,
    ids: list[str],
    embeddings: np.ndarray,
    model_name: str,
    collection_name: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings.astype(np.float32))
    (cache_dir / "hpo_ids.json").write_text(json.dumps(list(ids)))
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "collection_name": collection_name,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "n_terms": int(embeddings.shape[0]),
                "dim": int(embeddings.shape[1]),
            }
        )
    )


def _clear_cache(cache_dir: Path) -> None:
    for fname in ("embeddings.npy", "hpo_ids.json", "meta.json"):
        p = cache_dir / fname
        if p.exists():
            p.unlink()


def _read_from_chroma(
    index_dir: Path, collection_name: str
) -> tuple[list[str], np.ndarray]:
    try:
        client = chromadb.PersistentClient(path=str(index_dir))
        collection = client.get_collection(collection_name)
    except Exception as e:  # chromadb raises a variety of types
        raise FileNotFoundError(
            f"ChromaDB collection {collection_name!r} not found in {index_dir}. "
            "Run 'phentrieve index build --model-name ...' first."
        ) from e

    got = collection.get(include=["embeddings"])
    ids = list(got["ids"])
    embeddings = np.asarray(got["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError(
            f"Malformed Chroma payload: {len(ids)} ids vs embeddings shape "
            f"{embeddings.shape}"
        )
    return ids, embeddings


def load_cached_embeddings(
    model_name: str,
    refresh: bool = False,
    index_dir_override: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return (hpo_ids, embeddings) for the given model.

    First call: queries ChromaDB, writes cache, returns arrays.
    Later calls: reads cache only. `refresh=True` forces a re-read.

    Raises FileNotFoundError if the ChromaDB collection for `model_name`
    does not exist.
    """
    index_dir = Path(index_dir_override) if index_dir_override else get_default_index_dir()
    collection_name = generate_collection_name(model_name)
    cache_dir = _cache_dir_for(index_dir, collection_name)

    if not refresh:
        cached = _read_cache(cache_dir)
        if cached is not None:
            logger.info("Loaded ontology-fidelity cache from %s", cache_dir)
            return cached
        if any(
            (cache_dir / f).exists() for f in ("embeddings.npy", "hpo_ids.json", "meta.json")
        ):
            logger.warning(
                "Ontology-fidelity cache at %s is partial/malformed; refreshing from Chroma",
                cache_dir,
            )

    _clear_cache(cache_dir)
    ids, embeddings = _read_from_chroma(index_dir, collection_name)
    _write_cache(cache_dir, ids, embeddings, model_name, collection_name)
    logger.info(
        "Wrote ontology-fidelity cache: %d terms × %d dims → %s",
        len(ids),
        embeddings.shape[1],
        cache_dir,
    )
    return ids, embeddings
```

- [ ] **Step 4: Confirm pass**

```bash
uv run pytest tests/unit/analysis/test_embedding_cache.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add phentrieve/analysis/embedding_cache.py tests/unit/analysis/test_embedding_cache.py
git commit -m "feat(analysis): add ChromaDB->npy embedding cache with refresh semantics"
```

---

## Task 12: Add `analysis` optional dependency group

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Confirm there is no existing `[project.optional-dependencies]` section**

```bash
grep -n "optional-dependencies" pyproject.toml || echo "NONE FOUND"
```

If a section already exists: add the new `analysis` group inside it. If not: add the section (use the snippet below as-is).

- [ ] **Step 2: Add the new extra**

Append to `pyproject.toml` (after the `dependencies = [ ... ]` list, before the `[project.scripts]` section):

```toml
[project.optional-dependencies]
analysis = [
    "umap-learn>=0.5.6",
    "plotly>=5.22.0",
]
```

If the section exists, add the `analysis = [...]` entry inside it instead.

- [ ] **Step 3: Install the extra in the dev environment**

```bash
uv sync --extra analysis
```

Expected: `umap-learn` and `plotly` (and `numba`/`llvmlite` pulled by umap-learn) are installed. No errors.

- [ ] **Step 4: Verify imports**

```bash
uv run python -c "import umap, plotly.express; print(umap.__version__, plotly.__version__)"
```

Expected: version strings, no error.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(deps): add optional analysis extra (umap-learn, plotly)"
```

---

## Task 13: Orchestrator — argument parsing + logging + HPO DB loading

**Files:**
- Create: `scripts/analyze_embedding_ontology.py`

- [ ] **Step 1: Write the script skeleton**

Write `scripts/analyze_embedding_ontology.py`:

```python
#!/usr/bin/env python3
"""Ontology–embedding fidelity analysis.

Loads HPO graph data + cached BioLORD embeddings, computes four correlation
metrics and produces five plots in a timestamped output directory.

Run `python scripts/analyze_embedding_ontology.py --help` for usage.

This is a standalone research script — intentionally not exposed via the
`phentrieve` CLI. See .planning/specs/2026-04-17-ontology-embedding-fidelity-design.md
for the full contract.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("analyze_embedding_ontology")


@dataclass
class Args:
    model_name: str
    output_dir: Path
    k: int
    n_pairs: int
    umap_neighbors: int
    umap_min_dist: float
    metric: str
    sample: int | None
    seed: int
    skip_interactive: bool
    refresh_cache: bool
    log_level: str


def parse_args(argv: list[str] | None = None) -> Args:
    p = argparse.ArgumentParser(
        description="Ontology–embedding fidelity analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", default="FremyCompany/BioLORD-2023-M")
    p.add_argument(
        "--output-dir",
        default="data/results/ontology_fidelity",
        type=Path,
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-pairs", type=int, default=50_000)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-interactive", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args(argv)
    return Args(
        model_name=ns.model_name,
        output_dir=ns.output_dir,
        k=ns.k,
        n_pairs=ns.n_pairs,
        umap_neighbors=ns.umap_neighbors,
        umap_min_dist=ns.umap_min_dist,
        metric=ns.metric,
        sample=ns.sample,
        seed=ns.seed,
        skip_interactive=ns.skip_interactive,
        refresh_cache=ns.refresh_cache,
        log_level=ns.log_level,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_hpo_bundle(
    data_dir_override: str | None = None,
) -> tuple[list[dict], dict[str, set[str]], dict[str, int], str | None, Path]:
    """Open HPODatabase directly, load terms + graph + version, close. Hard-fail
    on missing DB (unlike document_creator.load_hpo_terms which soft-fails)."""
    from phentrieve.config import DEFAULT_HPO_DB_FILENAME
    from phentrieve.data_processing.hpo_database import HPODatabase
    from phentrieve.utils import get_default_data_dir, resolve_data_path

    data_dir = resolve_data_path(data_dir_override, "data_dir", get_default_data_dir)
    db_path = data_dir / DEFAULT_HPO_DB_FILENAME
    if not db_path.exists():
        raise FileNotFoundError(
            f"HPO database not found: {db_path}. "
            "Run 'phentrieve data prepare' first."
        )
    db = HPODatabase(db_path)
    try:
        terms = db.load_all_terms()
        ancestors, depths = db.load_graph_data()
        hpo_version = db.get_metadata("hpo_version")
    finally:
        db.close()
    logger.info(
        "HPO DB loaded: %d terms, version=%s, path=%s",
        len(terms),
        hpo_version,
        db_path,
    )
    return terms, ancestors, depths, hpo_version, db_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        terms, ancestors, depths, hpo_version, db_path = load_hpo_bundle()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    logger.info("Loaded %d HPO terms (DB: %s)", len(terms), db_path)
    # Further stages added in subsequent tasks.
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Dry-run with `--help`**

```bash
uv run python scripts/analyze_embedding_ontology.py --help
```

Expected: argparse help text listing all flags with defaults.

- [ ] **Step 3: Dry-run against the real HPO DB**

```bash
uv run python scripts/analyze_embedding_ontology.py --log-level DEBUG
```

Expected: logs show the DB path and term count, exit code 0. If the HPO DB is missing on this machine: `phentrieve data prepare` first.

- [ ] **Step 4: Commit**

```bash
make lint-fix && make format
git add scripts/analyze_embedding_ontology.py
git commit -m "feat(scripts): scaffold ontology fidelity orchestrator (HPO DB + argparse)"
```

---

## Task 14: Orchestrator — embedding cache load + alignment

**Files:**
- Modify: `scripts/analyze_embedding_ontology.py`

- [ ] **Step 1: Extend the script with alignment logic**

Replace the `main` function in `scripts/analyze_embedding_ontology.py` with:

```python
def align_terms_and_embeddings(
    terms: list[dict],
    hpo_ids: list[str],
    embeddings: "np.ndarray",
    symmetric_diff_tolerance: float = 0.05,
) -> tuple[list[dict], "np.ndarray"]:
    """Filter both sides to the intersection of (HPO DB IDs) and (cache IDs).

    - If |symmetric difference| / min(|A|, |B|) > tolerance, raise ValueError —
      the cache is likely stale.
    - Otherwise, log a warning with counts and proceed on the intersection.
    - Preserves cache row order for returned embeddings.
    """
    import numpy as np

    db_ids = {t["id"] for t in terms}
    cache_ids = set(hpo_ids)
    inter = db_ids & cache_ids
    only_db = db_ids - cache_ids
    only_cache = cache_ids - db_ids

    sym_diff = len(only_db) + len(only_cache)
    denom = max(min(len(db_ids), len(cache_ids)), 1)
    if sym_diff / denom > symmetric_diff_tolerance:
        raise ValueError(
            f"HPO DB / embedding cache divergence too large: "
            f"{sym_diff} mismatched IDs out of min({len(db_ids)}, {len(cache_ids)}) "
            f"({sym_diff / denom:.1%} > {symmetric_diff_tolerance:.0%}). "
            "Rebuild the index or refresh the cache (--refresh-cache)."
        )
    if sym_diff:
        logger.warning(
            "HPO DB / embedding cache mismatch: %d only in DB, %d only in cache",
            len(only_db),
            len(only_cache),
        )
    if not inter:
        raise ValueError("Zero overlap between HPO DB and embedding cache.")

    keep_cache_rows = [i for i, hid in enumerate(hpo_ids) if hid in inter]
    aligned_embeddings = embeddings[keep_cache_rows]
    aligned_ids_set = {hpo_ids[i] for i in keep_cache_rows}
    aligned_terms = [t for t in terms if t["id"] in aligned_ids_set]

    # Sort aligned_terms into the same order as aligned_embeddings rows.
    ordered_ids = [hpo_ids[i] for i in keep_cache_rows]
    by_id = {t["id"]: t for t in aligned_terms}
    aligned_terms = [by_id[i] for i in ordered_ids]

    return aligned_terms, aligned_embeddings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        terms, ancestors, depths, hpo_version, db_path = load_hpo_bundle()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    from phentrieve.analysis.embedding_cache import load_cached_embeddings

    try:
        hpo_ids, embeddings = load_cached_embeddings(
            model_name=args.model_name,
            refresh=args.refresh_cache,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    try:
        aligned_terms, aligned_embeddings = align_terms_and_embeddings(
            terms, hpo_ids, embeddings
        )
    except ValueError as e:
        logger.error("%s", e)
        return 1

    if args.sample is not None and args.sample < len(aligned_terms):
        import numpy as np

        rng = np.random.default_rng(args.seed)
        sample_idx = rng.choice(len(aligned_terms), size=args.sample, replace=False)
        sample_idx.sort()
        aligned_terms = [aligned_terms[i] for i in sample_idx]
        aligned_embeddings = aligned_embeddings[sample_idx]
        logger.info("Sampled %d terms (seed=%d)", len(aligned_terms), args.seed)
    elif args.sample is not None:
        logger.info(
            "Requested --sample %d >= %d aligned terms; using all.",
            args.sample,
            len(aligned_terms),
        )

    logger.info(
        "Aligned: %d terms × %d dims", len(aligned_terms), aligned_embeddings.shape[1]
    )

    # Metrics, UMAP, and writers are added in subsequent tasks.
    return 0
```

- [ ] **Step 2: Smoke-run**

```bash
uv run python scripts/analyze_embedding_ontology.py --log-level DEBUG
```

Expected: the script loads HPO, loads/caches embeddings (first run will read Chroma and write to `data/indexes/ontology_fidelity_cache/phentrieve_FremyCompany_BioLORD-2023-M/`), prints aligned count, exits 0.

If ChromaDB does not contain the BioLORD collection on this machine: `phentrieve index build --model-name FremyCompany/BioLORD-2023-M` first.

- [ ] **Step 3: Commit**

```bash
make lint-fix && make format
git add scripts/analyze_embedding_ontology.py
git commit -m "feat(scripts): add embedding cache load and alignment to orchestrator"
```

---

## Task 15: Orchestrator — metrics assembly + summary.json + per-term CSV

**Files:**
- Modify: `scripts/analyze_embedding_ontology.py`

- [ ] **Step 1: Add metrics computation and CSV/JSON writers**

Add the following functions to `scripts/analyze_embedding_ontology.py` (above `main`):

```python
def compute_metrics(
    aligned_terms: list[dict],
    aligned_embeddings: "np.ndarray",
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
    args: Args,
) -> tuple[dict, list[dict], dict[str, tuple[str | None, frozenset[str]]]]:
    """Run all four metrics. Return (summary_dict, per_term_rows, branch_info)."""
    from phentrieve.analysis.ontology_fidelity import (
        branch_knn_purity,
        build_descendants_index,
        depth_correlation,
        global_distance_correlation,
        information_content,
        per_term_fidelity,
        top_level_branch,
    )

    term_ids = [t["id"] for t in aligned_terms]
    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)

    branch_info = {
        tid: top_level_branch(tid, ancestors, depths) for tid in term_ids
    }
    branch_map: dict[str, str | None] = {tid: branch_info[tid][0] for tid in term_ids}
    multi_parent_count = sum(1 for tid in term_ids if len(branch_info[tid][1]) > 1)

    corr = global_distance_correlation(
        term_ids, aligned_embeddings, ancestors, depths, ic,
        n_pairs=args.n_pairs, seed=args.seed,
    )
    fidelity_rows = per_term_fidelity(
        term_ids, aligned_embeddings, ancestors, descendants, ic, k=args.k
    )
    purity = branch_knn_purity(term_ids, aligned_embeddings, branch_map, k=args.k)
    depth_rho = depth_correlation(term_ids, aligned_embeddings, depths)

    summary: dict = {
        "metrics": {
            "global_distance_correlation": corr,
            "per_term_fidelity_mean": float(
                sum(r["fidelity"] for r in fidelity_rows) / max(len(fidelity_rows), 1)
            ),
            "branch_knn_purity": purity,
            "depth_correlation_spearman": depth_rho,
        },
        "meta": {
            "multi_parent_term_count": multi_parent_count,
            "n_terms_analyzed": len(term_ids),
            "embedding_dim": int(aligned_embeddings.shape[1]),
        },
    }
    return summary, fidelity_rows, branch_info


def write_per_term_csv(
    out_path: Path,
    fidelity_rows: list[dict],
    aligned_terms: list[dict],
    branch_info: dict[str, tuple[str | None, frozenset[str]]],
    depths: dict[str, int],
) -> None:
    import csv

    labels = {t["id"]: t.get("label", "") for t in aligned_terms}
    rows_sorted = sorted(fidelity_rows, key=lambda r: r["fidelity"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hpo_id", "label", "branch", "all_branches", "depth", "fidelity", "rank"])
        for rank, row in enumerate(rows_sorted, start=1):
            tid = row["id"]
            branch, all_branches = branch_info[tid]
            w.writerow(
                [
                    tid,
                    labels.get(tid, ""),
                    branch if branch is not None else "",
                    ";".join(sorted(all_branches)),
                    depths.get(tid, -1),
                    f"{row['fidelity']:.6f}",
                    rank,
                ]
            )


def write_summary_json(
    out_path: Path,
    summary: dict,
    args: Args,
    hpo_version: str | None,
    db_path: Path,
    aligned_n: int,
) -> None:
    import json

    payload = {
        **summary,
        "config": {
            "model_name": args.model_name,
            "hpo_db_path": str(db_path),
            "hpo_version": hpo_version,
            "k": args.k,
            "n_pairs": args.n_pairs,
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "metric": args.metric,
            "sample": args.sample,
            "seed": args.seed,
        },
        "aligned_n_terms": aligned_n,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
```

- [ ] **Step 2: Wire them into `main`**

Replace the `return 0` tail of `main` with:

```python
    from datetime import datetime

    from phentrieve.utils import get_model_slug

    slug = get_model_slug(args.model_name)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = args.output_dir / f"{slug}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_root)

    summary, fidelity_rows, branch_info = compute_metrics(
        aligned_terms, aligned_embeddings, ancestors, depths, args
    )
    write_per_term_csv(
        out_root / "per_term_fidelity.csv",
        fidelity_rows,
        aligned_terms,
        branch_info,
        depths,
    )
    write_summary_json(
        out_root / "summary.json",
        summary,
        args,
        hpo_version,
        db_path,
        len(aligned_terms),
    )
    logger.info("Metrics written. Next: UMAP and plots.")
    return 0
```

- [ ] **Step 3: Verify `get_model_slug` is importable**

```bash
uv run python -c "from phentrieve.utils import get_model_slug; print(get_model_slug('FremyCompany/BioLORD-2023-M'))"
```

Expected: a slug string like `FremyCompany_BioLORD-2023-M`. If the import fails, adjust the import above to use `generate_collection_name` instead (which strips the `phentrieve_` prefix) and update accordingly.

- [ ] **Step 4: Smoke-run with `--sample` to keep it fast**

```bash
uv run python scripts/analyze_embedding_ontology.py --sample 300 --n-pairs 1000 --log-level DEBUG
```

Expected: `per_term_fidelity.csv` and `summary.json` appear under `data/results/ontology_fidelity/<slug>_<ts>/`; exit code 0. Verify files:

```bash
ls data/results/ontology_fidelity/
head data/results/ontology_fidelity/*/per_term_fidelity.csv
cat data/results/ontology_fidelity/*/summary.json | head -40
```

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add scripts/analyze_embedding_ontology.py
git commit -m "feat(scripts): compute metrics and write summary.json + per-term CSV"
```

---

## Task 16: Orchestrator — UMAP + static PNG plots + coords CSV

**Files:**
- Modify: `scripts/analyze_embedding_ontology.py`

- [ ] **Step 1: Add UMAP + plotting functions**

Add above `main`:

```python
def run_umap(
    embeddings: "np.ndarray",
    args: Args,
) -> "np.ndarray":
    """Return 2-D UMAP coordinates with shape (n, 2)."""
    import umap

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.metric,
        n_components=2,
        random_state=args.seed,
    )
    return reducer.fit_transform(embeddings)


def write_umap_coords_csv(
    out_path: Path,
    aligned_terms: list[dict],
    coords: "np.ndarray",
    branch_info: dict[str, tuple[str | None, frozenset[str]]],
    fidelity_by_id: dict[str, float],
    depths: dict[str, int],
) -> None:
    import csv

    labels = {t["id"]: t.get("label", "") for t in aligned_terms}
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hpo_id", "label", "umap_x", "umap_y", "branch", "fidelity", "depth"])
        for (term, (x, y)) in zip(aligned_terms, coords):
            tid = term["id"]
            branch, _ = branch_info[tid]
            w.writerow(
                [
                    tid,
                    labels.get(tid, ""),
                    f"{x:.6f}",
                    f"{y:.6f}",
                    branch if branch is not None else "",
                    f"{fidelity_by_id.get(tid, float('nan')):.6f}",
                    depths.get(tid, -1),
                ]
            )


def plot_umap_by_branch(
    out_path: Path,
    coords: "np.ndarray",
    aligned_terms: list[dict],
    branch_info: dict[str, tuple[str | None, frozenset[str]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = [branch_info[t["id"]][0] or "ROOT" for t in aligned_terms]
    unique = sorted(set(branches))
    cmap = plt.get_cmap("tab20", len(unique))
    color_of = {b: cmap(i) for i, b in enumerate(unique)}
    colors = [color_of[b] for b in branches]

    fig, ax = plt.subplots(figsize=(12, 9), dpi=120)
    ax.scatter(coords[:, 0], coords[:, 1], s=3, c=colors, alpha=0.7, linewidths=0)
    ax.set_title("HPO embedding UMAP — colored by top-level branch")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_of[b], label=b, markersize=6)
        for b in unique
    ]
    ax.legend(handles=handles, loc="best", fontsize=7, ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_umap_by_fidelity(
    out_path: Path,
    coords: "np.ndarray",
    aligned_terms: list[dict],
    fidelity_by_id: dict[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fid = [fidelity_by_id.get(t["id"], 0.0) for t in aligned_terms]
    fig, ax = plt.subplots(figsize=(12, 9), dpi=120)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], s=4, c=fid, cmap="RdBu", vmin=0.0, vmax=1.0,
        alpha=0.8, linewidths=0,
    )
    ax.set_title("HPO embedding UMAP — colored by per-term fidelity")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("fidelity (0=disagree, 1=agree)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_distance_correlation(
    out_path: Path,
    aligned_terms: list[dict],
    aligned_embeddings: "np.ndarray",
    ancestors: dict[str, set[str]],
    depths: dict[str, int],
    args: Args,
) -> dict[str, float]:
    """Hexbin: cosine distance vs (shortest-path, Resnik), two subplots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from phentrieve.analysis.ontology_fidelity import (
        _cosine_distance_rows,  # type: ignore[attr-defined]
        build_descendants_index,
        graph_shortest_path,
        information_content,
        resnik_similarity,
        sample_pairs,
    )
    from scipy.stats import spearmanr

    descendants = build_descendants_index(ancestors)
    ic = information_content(descendants)
    term_ids = [t["id"] for t in aligned_terms]
    rng = np.random.default_rng(args.seed)
    pairs = sample_pairs(len(term_ids), args.n_pairs, rng)
    if pairs.size == 0:
        return {"spearman_shortest_path": float("nan"), "spearman_resnik": float("nan")}

    cos = _cosine_distance_rows(
        aligned_embeddings[pairs[:, 0]], aligned_embeddings[pairs[:, 1]]
    )
    u_ids = [term_ids[i] for i in pairs[:, 0]]
    v_ids = [term_ids[i] for i in pairs[:, 1]]
    shortest = np.array(
        [graph_shortest_path(u, v, ancestors, depths) for u, v in zip(u_ids, v_ids)],
        dtype=np.float64,
    )
    resnik = np.array(
        [resnik_similarity(u, v, ancestors, ic) for u, v in zip(u_ids, v_ids)],
        dtype=np.float64,
    )
    rho_sp, _ = spearmanr(cos, shortest)
    rho_rk, _ = spearmanr(cos, -resnik)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    ax1.hexbin(shortest, cos, gridsize=40, cmap="viridis", mincnt=1)
    ax1.set_xlabel("HPO shortest-path distance")
    ax1.set_ylabel("embedding cosine distance")
    ax1.set_title(f"shortest-path  ρ={rho_sp:.3f}  (n={len(pairs)})")
    ax2.hexbin(-resnik, cos, gridsize=40, cmap="viridis", mincnt=1)
    ax2.set_xlabel("−Resnik similarity  (larger = more distant)")
    ax2.set_ylabel("embedding cosine distance")
    ax2.set_title(f"Resnik  ρ={rho_rk:.3f}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return {"spearman_shortest_path": float(rho_sp), "spearman_resnik": float(rho_rk)}


def plot_branch_fidelity(
    out_path: Path,
    purity: dict,
    fidelity_rows: list[dict],
    branch_info: dict[str, tuple[str | None, frozenset[str]]],
) -> None:
    """Horizontal bar chart of per-branch MEAN FIDELITY (not purity).
    Purity is a separate metric surfaced in summary.json.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_branch: dict[str, list[float]] = {}
    for row in fidelity_rows:
        branch, _ = branch_info[row["id"]]
        if branch is None:
            continue
        by_branch.setdefault(branch, []).append(row["fidelity"])
    ordered = sorted(by_branch.items(), key=lambda kv: np.mean(kv[1]))
    branches = [b for b, _ in ordered]
    means = [float(np.mean(v)) for _, v in ordered]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(branches))), dpi=120)
    ax.barh(branches, means, color="#3b82f6")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("mean per-term fidelity")
    ax.set_title("Fidelity by top-level HPO branch")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
```

- [ ] **Step 2: Extend `main`**

Before the `return 0` at the end of `main`, insert:

```python
    logger.info("Running UMAP (n_neighbors=%d, min_dist=%s) ...", args.umap_neighbors, args.umap_min_dist)
    coords = run_umap(aligned_embeddings, args)
    fidelity_by_id = {r["id"]: r["fidelity"] for r in fidelity_rows}

    write_umap_coords_csv(
        out_root / "umap_coords.csv",
        aligned_terms,
        coords,
        branch_info,
        fidelity_by_id,
        depths,
    )
    plot_umap_by_branch(
        out_root / "umap_by_branch.png", coords, aligned_terms, branch_info
    )
    plot_umap_by_fidelity(
        out_root / "umap_by_fidelity.png", coords, aligned_terms, fidelity_by_id
    )
    plot_distance_correlation(
        out_root / "distance_correlation.png",
        aligned_terms,
        aligned_embeddings,
        ancestors,
        depths,
        args,
    )
    plot_branch_fidelity(
        out_root / "branch_fidelity.png",
        summary["metrics"]["branch_knn_purity"],
        fidelity_rows,
        branch_info,
    )
    logger.info("Static plots written.")
```

- [ ] **Step 3: Smoke-run**

```bash
uv run python scripts/analyze_embedding_ontology.py --sample 500 --n-pairs 2000 --log-level INFO
ls data/results/ontology_fidelity/*/
```

Expected: all of `summary.json`, `per_term_fidelity.csv`, `umap_coords.csv`, `umap_by_branch.png`, `umap_by_fidelity.png`, `distance_correlation.png`, `branch_fidelity.png` are written. Open one PNG to verify it renders.

- [ ] **Step 4: Commit**

```bash
make lint-fix && make format
git add scripts/analyze_embedding_ontology.py
git commit -m "feat(scripts): add UMAP and four static plots + coords CSV"
```

---

## Task 17: Orchestrator — interactive Plotly HTML

**Files:**
- Modify: `scripts/analyze_embedding_ontology.py`

- [ ] **Step 1: Add the Plotly plot function**

Add above `main`:

```python
def plot_umap_interactive(
    out_path: Path,
    coords: "np.ndarray",
    aligned_terms: list[dict],
    branch_info: dict[str, tuple[str | None, frozenset[str]]],
    fidelity_by_id: dict[str, float],
) -> None:
    """Single HTML with two traces (branch + fidelity) toggled by legend groups."""
    import plotly.graph_objects as go

    ids = [t["id"] for t in aligned_terms]
    labels = [t.get("label", "") for t in aligned_terms]
    defs = [t.get("definition", "") or "" for t in aligned_terms]
    branches = [branch_info[t["id"]][0] or "ROOT" for t in aligned_terms]
    fid = [fidelity_by_id.get(t["id"], float("nan")) for t in aligned_terms]

    hover = [
        f"<b>{hid}</b><br>{lab}<br>branch: {br}<br>fidelity: {f:.3f}<br>{d[:160]}"
        for hid, lab, br, f, d in zip(ids, labels, branches, fid, defs)
    ]

    unique_branches = sorted(set(branches))
    branch_to_color = {b: i for i, b in enumerate(unique_branches)}
    branch_colors = [branch_to_color[b] for b in branches]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=branch_colors,
                colorscale="Viridis",
                showscale=False,
                opacity=0.8,
            ),
            text=hover,
            hoverinfo="text",
            name="by branch",
            visible=True,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=fid,
                colorscale="RdBu",
                cmin=0.0,
                cmax=1.0,
                showscale=True,
                colorbar=dict(title="fidelity"),
                opacity=0.85,
            ),
            text=hover,
            hoverinfo="text",
            name="by fidelity",
            visible=False,
        )
    )
    fig.update_layout(
        title="HPO embedding UMAP — interactive",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.08,
                buttons=[
                    dict(
                        label="by branch",
                        method="update",
                        args=[{"visible": [True, False]}],
                    ),
                    dict(
                        label="by fidelity",
                        method="update",
                        args=[{"visible": [False, True]}],
                    ),
                ],
            )
        ],
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
```

- [ ] **Step 2: Wire it into `main`**

After the static plots block in `main`, add:

```python
    if not args.skip_interactive:
        try:
            plot_umap_interactive(
                out_root / "umap_interactive.html",
                coords,
                aligned_terms,
                branch_info,
                fidelity_by_id,
            )
            logger.info("Interactive HTML written.")
        except ImportError:
            logger.error(
                "plotly not installed. Install with: uv sync --extra analysis  "
                "(or pass --skip-interactive to skip this output)."
            )
            return 1
    else:
        logger.info("Skipping interactive HTML (--skip-interactive).")
```

- [ ] **Step 3: Smoke-run**

```bash
uv run python scripts/analyze_embedding_ontology.py --sample 500 --n-pairs 2000
ls data/results/ontology_fidelity/*/umap_interactive.html
```

Expected: an HTML file is present. Open it in a browser to verify both traces toggle.

- [ ] **Step 4: Test the --skip-interactive path**

```bash
uv run python scripts/analyze_embedding_ontology.py --sample 300 --skip-interactive
```

Expected: no `umap_interactive.html` in the new output dir.

- [ ] **Step 5: Commit**

```bash
make lint-fix && make format
git add scripts/analyze_embedding_ontology.py
git commit -m "feat(scripts): add interactive Plotly HTML with branch/fidelity toggle"
```

---

## Task 18: Integration smoke test

**Files:**
- Create: `tests/integration/analysis/test_analyze_script_smoke.py`

- [ ] **Step 1: Write the integration test**

Write `tests/integration/analysis/test_analyze_script_smoke.py`:

```python
"""End-to-end smoke test for scripts/analyze_embedding_ontology.py.

Runs the orchestrator against the real HPO SQLite DB but mocks the embedding
cache (so ChromaDB is never touched and UMAP operates on tiny synthetic vectors).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "analyze_embedding_ontology.py"


pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _write_stub_cache(tmp_path: Path, term_ids: list[str]) -> Path:
    """Write a fake ontology-fidelity cache with deterministic synthetic vectors."""
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((len(term_ids), 16)).astype(np.float32)

    # Mirrors the real cache layout.
    from phentrieve.utils import generate_collection_name

    collection = generate_collection_name("FremyCompany/BioLORD-2023-M")
    cache_dir = tmp_path / "indexes" / "ontology_fidelity_cache" / collection
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", vecs)
    (cache_dir / "hpo_ids.json").write_text(json.dumps(term_ids))
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_name": "FremyCompany/BioLORD-2023-M",
                "collection_name": collection,
                "written_at": "2026-04-17T00:00:00+00:00",
                "n_terms": len(term_ids),
                "dim": 16,
            }
        )
    )
    return cache_dir


def test_analyze_script_smoke(tmp_path, monkeypatch):
    """Run the script against a real HPO DB with a stubbed embedding cache."""
    from phentrieve.data_processing.hpo_database import HPODatabase
    from phentrieve.utils import get_default_data_dir

    db_path = get_default_data_dir() / "hpo_data.db"
    if not db_path.exists():
        pytest.skip("HPO SQLite DB not present on this machine")

    db = HPODatabase(db_path)
    try:
        all_terms = db.load_all_terms()
    finally:
        db.close()
    # Take the first 200 for speed.
    term_ids = [t["id"] for t in all_terms[:200]]

    # Point PHENTRIEVE_INDEX_DIR at a temp tree and write a stubbed cache there.
    tmp_index_root = tmp_path
    _write_stub_cache(tmp_index_root, term_ids)
    monkeypatch.setenv("PHENTRIEVE_INDEX_DIR", str(tmp_index_root / "indexes"))

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--sample",
        "150",
        "--n-pairs",
        "300",
        "--k",
        "5",
        "--umap-neighbors",
        "10",
        "--output-dir",
        str(out_dir),
        "--log-level",
        "INFO",
        "--skip-interactive",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    # Exactly one run subdir.
    run_dirs = list(out_dir.iterdir())
    assert len(run_dirs) == 1
    run = run_dirs[0]

    expected = [
        "summary.json",
        "per_term_fidelity.csv",
        "umap_coords.csv",
        "umap_by_branch.png",
        "umap_by_fidelity.png",
        "distance_correlation.png",
        "branch_fidelity.png",
    ]
    for name in expected:
        assert (run / name).exists(), f"missing {name}; dir has {list(run.iterdir())}"

    summary = json.loads((run / "summary.json").read_text())
    assert "metrics" in summary
    assert "config" in summary
    assert summary["config"]["model_name"] == "FremyCompany/BioLORD-2023-M"
```

- [ ] **Step 2: Confirm `PHENTRIEVE_INDEX_DIR` overrides `get_default_index_dir`**

```bash
grep -n "PHENTRIEVE_INDEX_DIR\|index_dir" phentrieve/utils.py | head -20
```

If `PHENTRIEVE_INDEX_DIR` is **not** the correct env var name for overriding the index dir in this repo, replace the `monkeypatch.setenv(...)` line in the test with the correct mechanism (for example, construct the script command with a `--index-dir` flag if one exists, or use `phentrieve/phentrieve.yaml`-based override). If no env var/flag exists, add a `--index-dir` flag to the script by plumbing `index_dir_override` through `load_cached_embeddings` and re-run. Leave a note in the commit message.

- [ ] **Step 3: Run the smoke test**

```bash
uv run pytest tests/integration/analysis/test_analyze_script_smoke.py -v -m slow
```

Expected: `1 passed`. Skipped if the HPO DB isn't present locally.

- [ ] **Step 4: Commit**

```bash
make lint-fix && make format
git add tests/integration/analysis/test_analyze_script_smoke.py
git commit -m "test(analysis): add integration smoke test for analyze script"
```

---

## Task 19: Documentation — scripts/README.md

**Files:**
- Modify: `scripts/README.md`

- [ ] **Step 1: Read the current README**

```bash
uv run cat scripts/README.md | head -60
```

Use `Read` to inspect the file and pick an insertion point (typically a new section after the existing scripts).

- [ ] **Step 2: Append a new section**

Append to `scripts/README.md`:

```markdown
### `analyze_embedding_ontology.py`

Ontology–embedding fidelity analysis. Loads BioLORD HPO embeddings from the
existing ChromaDB collection (cached to `.npy` on first run), computes four
correlation metrics between the embedding space and the curated HPO DAG, and
produces a timestamped results directory with a summary JSON, a per-term
fidelity CSV, and five plots.

**Prerequisites:**

1. Phentrieve installed with the `analysis` extra:
   ```bash
   uv sync --extra analysis
   ```
2. HPO SQLite DB prepared:
   ```bash
   phentrieve data prepare
   ```
3. BioLORD index built:
   ```bash
   phentrieve index build --model-name FremyCompany/BioLORD-2023-M
   ```

**Usage:**

```bash
python scripts/analyze_embedding_ontology.py \
    --model-name FremyCompany/BioLORD-2023-M \
    --k 10 \
    --n-pairs 50000
```

Full flag set: see `--help`.

**Outputs** — under `data/results/ontology_fidelity/<model-slug>_<YYYYMMDD-HHMMSS>/`:

| File | Description |
|---|---|
| `summary.json` | Spearman ρ (shortest-path & Resnik), mean per-term fidelity, branch k-NN purity, depth correlation, config echo. |
| `per_term_fidelity.csv` | Per-term fidelity, sorted ascending (worst first). |
| `umap_coords.csv` | UMAP coordinates for each HPO term. |
| `umap_by_branch.png` | UMAP colored by top-level HPO branch. |
| `umap_by_fidelity.png` | UMAP colored by per-term fidelity (RdBu). |
| `umap_interactive.html` | Interactive Plotly; toggle between branch and fidelity colorings. |
| `distance_correlation.png` | Hexbin of embedding cosine vs graph distance (two panels). |
| `branch_fidelity.png` | Per-branch mean fidelity bar chart. |

See [`.planning/specs/2026-04-17-ontology-embedding-fidelity-design.md`](../.planning/specs/2026-04-17-ontology-embedding-fidelity-design.md)
for the full contract and metric definitions.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/README.md
git commit -m "docs(scripts): document analyze_embedding_ontology.py"
```

---

## Task 20: Final quality gate

**Files:**
- None (verification only).

- [ ] **Step 1: Run the full required check trio**

```bash
make check && make typecheck-fast && make test
```

Expected: all pass. If any fail — stop, investigate, fix. Do not claim completion until green.

- [ ] **Step 2: Run the analysis unit tests with coverage**

```bash
uv run pytest tests/unit/analysis/ --cov=phentrieve.analysis --cov-report=term-missing
```

Expected: ≥ 85% line coverage on `phentrieve/analysis/*`. If below, add tests for uncovered branches (most likely: malformed cache paths, `refresh=True` overwrite path).

- [ ] **Step 3: Run the integration smoke test**

```bash
uv run pytest tests/integration/analysis/ -v -m slow
```

Expected: smoke test passes (or skips with a clear "HPO DB not present" reason on a fresh checkout).

- [ ] **Step 4: Tag the branch ready for PR**

```bash
git log --oneline main..HEAD
```

Expected: ~14 atomic commits, one per task (Tasks 1–3 combined into the scaffold commit, others one-per-task). Confirm the list is clean.

- [ ] **Step 5: Push and open a PR**

```bash
git push -u origin feat/ontology-embedding-fidelity
gh pr create --title "feat: ontology–embedding fidelity analysis script" --body "$(cat <<'EOF'
## Summary
Implements the spec at `.planning/specs/2026-04-17-ontology-embedding-fidelity-design.md` (commit 1bb3326).

Ships a standalone research script `scripts/analyze_embedding_ontology.py` that:
- Loads BioLORD-2023-M embeddings once from ChromaDB (cached to `.npy`).
- Computes four ontology-fidelity metrics against the curated HPO DAG: global distance correlation (Spearman ρ for shortest-path and Resnik), per-term fidelity, branch k-NN purity, depth correlation.
- Produces a timestamped output directory with summary JSON, per-term fidelity CSV, UMAP coords CSV, four static PNG plots, and an interactive Plotly HTML.

No CLI change — the script is invoked directly per the user's request.

Addresses #34 (proposes closing with a pointer to the script).

## Test plan
- [ ] `make check && make typecheck-fast && make test` green
- [ ] `uv run pytest tests/unit/analysis/` green, ≥ 85% coverage
- [ ] `uv run pytest tests/integration/analysis/ -m slow` green on a machine with the HPO DB
- [ ] Manual run on full dataset produces all 8 expected artifacts

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review against the spec

Spec sections → task that implements each:

- **Inputs & outputs table (spec §"User-facing artifact")** → Tasks 13–17 (orchestrator + writers); integration asserts all 8 files (Task 18).
- **Metric 1: global distance correlation** → Task 7 (library) + Task 15 (orchestrator call).
- **Metric 2: per-term fidelity** → Task 8 (library) + Task 15 (orchestrator call + CSV writer).
- **Metric 3: branch k-NN purity** → Task 9 (library) + Task 15 (orchestrator call).
- **Metric 4: depth correlation** → Task 10 (library) + Task 15 (orchestrator call).
- **Determinism — k-NN self-exclusion** → Task 8, `_embedding_knn` + test `test_per_term_fidelity_bounds_0_to_1`.
- **Determinism — Resnik tie-break** → Task 8, `_resnik_top_k` + test `test_per_term_fidelity_tiebreak_is_lexicographic`.
- **Determinism — multi-parent top-level branch** → Task 5, `top_level_branch` + test `test_top_level_branch_multi_parent_picks_lex_smallest`.
- **Determinism — zero-descendant leaves / IC(root)=0** → Task 2 tests.
- **Determinism — depth-0 handling** → Task 5, `test_top_level_branch_for_depth_0_term_is_none`; Task 9, `test_branch_knn_purity_excludes_root_from_denominator`.
- **Determinism — seed sole entropy** → Task 6 (`test_sample_pairs_deterministic_given_seed`), Task 7 (`test_global_distance_correlation_seeded_is_deterministic`), Task 16 (`umap.UMAP(random_state=args.seed)`).
- **Determinism — centroid on aligned set** → Task 10 (`depth_correlation` works on whatever embedding matrix is passed; orchestrator passes aligned only).
- **Determinism — 5% alignment threshold** → Task 14, `align_terms_and_embeddings`.
- **Cache path layout** → Task 11, `_cache_dir_for` + writers; verified by `test_embedding_cache_first_call_reads_chroma_and_writes_files`.
- **`load_hpo_bundle` uses `HPODatabase` directly** → Task 13.
- **Dependencies: `analysis` optional extra** → Task 12.
- **Error handling — missing HPO DB hard-fails** → Task 13.
- **Error handling — missing Chroma collection hard-fails** → Task 11, `_read_from_chroma`.
- **Error handling — malformed cache → delete + retry** → Task 11, `_read_cache` returns None + logger.warning in `load_cached_embeddings`.
- **Error handling — zero intersection aborts** → Task 14.
- **Plots 1–4 PNG + 5 Plotly HTML** → Task 16 + Task 17.
- **UMAP coords CSV** → Task 16.
- **summary.json config echo** → Task 15 (`write_summary_json`).
- **Testing — unit names listed in spec** → Tasks 2–10 (by name; re-check in Task 20 coverage output).
- **Testing — integration smoke** → Task 18.
- **Testing — coverage ≥ 85%** → Task 20 gate.
- **Docs — scripts/README.md** → Task 19.
- **Worktree-based feature work** → Task 0.

No gaps identified. Placeholders scan: each step contains concrete code or a concrete command. Type consistency: `per_term_fidelity` returns `list[dict]` with keys `{id, fidelity, nn_embedding, nn_dag}` in Task 8, and Tasks 15–16 access those keys by the same names. `top_level_branch` returns `tuple[str | None, frozenset[str]]` in Task 5, and Tasks 15–16 unpack it as `(branch, all_branches)`. `branch_knn_purity` returns `{'overall', 'per_branch', 'n_evaluated'}` in Task 9, consumed only by `summary.json` in Task 15.

Two open questions from the spec were resolved during the plan and noted here so readers don't need to cross-reference:
- **Distance-correlation hexbin: two subplots in one PNG.** Implemented in Task 16, `plot_distance_correlation`.
- **Plotly HTML: single file with toggle.** Implemented in Task 17, `plot_umap_interactive`.
