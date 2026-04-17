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

import numpy as np


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
    """Spearman rho between embedding cosine distance and (shortest-path, Resnik).

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
        [
            graph_shortest_path(u, v, ancestors, depths)
            for u, v in zip(u_ids, v_ids, strict=True)
        ],
        dtype=np.float64,
    )
    resnik = np.array(
        [
            resnik_similarity(u, v, ancestors, ic)
            for u, v in zip(u_ids, v_ids, strict=True)
        ],
        dtype=np.float64,
    )

    # Spearman expects distance-like arrays to correlate in the same direction.
    # Cosine distance is larger for dissimilar pairs; shortest-path also larger;
    # so a positive rho means agreement. Resnik is a similarity -- negate for agreement.
    rho_sp, _ = spearmanr(cosine, shortest)
    rho_rk, _ = spearmanr(cosine, -resnik)

    return {
        "spearman_shortest_path": float(rho_sp) if rho_sp == rho_sp else float("nan"),
        "spearman_resnik": float(rho_rk) if rho_rk == rho_rk else float("nan"),
        "n_pairs": int(len(pairs)),
    }


def _embedding_knn(embeddings: np.ndarray, k: int) -> np.ndarray:
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
    # secondary ascending on id -- this is exactly the tie-break rule.
    scored.sort()
    return [c for _, c in scored[:k]]


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


def per_term_fidelity(
    term_ids: list[str],
    embeddings: np.ndarray,
    ancestors: dict[str, set[str]],
    descendants: dict[str, set[str]],
    ic: dict[str, float],
    k: int = 10,
) -> list[dict]:
    """Per-term fidelity: |embedding-kNN intersection Resnik-top-k| / k, in [0, 1].

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
