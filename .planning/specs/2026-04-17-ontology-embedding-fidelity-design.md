# Ontology–Embedding Fidelity Analysis Script

**Date:** 2026-04-17
**Issue:** [#34 — HPO term embedding visualization with UMAP](https://github.com/berntpopp/phentrieve/issues/34)
**Status:** Design, awaiting implementation plan
**Branch:** `feat/ontology-embedding-fidelity`

## Problem

Phentrieve relies on BioLORD-2023-M embeddings to retrieve HPO terms from clinical text. We have no quantitative evidence that the embedding geometry reflects the curated HPO DAG, and no way to see *where* language agrees or disagrees with the ontology. Issue #34 proposes a visualization CLI; this spec extends the intent into a correlation analysis that produces both numbers and plots, so we can answer "does the language model recapitulate the curated ontology, and if not, where does it fail?"

## Goal

Ship a one-shot research script that, for a given indexed model (default BioLORD-2023-M), emits a timestamped output directory containing four correlation metrics, a per-term fidelity table, and five plots. The script is standalone (not a CLI subcommand) and reuses the existing HPO database and ChromaDB collections without forcing a re-index.

## Non-goals

- Not exposed via `phentrieve` CLI — script only (`scripts/analyze_embedding_ontology.py`).
- No model-vs-model comparison.
- No Vue frontend integration.
- No Nomic Atlas upload.
- No multi-vector index support (single-vector only for this pass).
- No re-indexing. Embeddings are read from the existing ChromaDB collection once and cached in a dedicated per-model cache directory (see "Embedding cache" for the exact path).
- No hosting or deployment of generated HTML plots — they are static artifacts.

## Out-of-spec (future work)

- Exposing this as a `phentrieve visualize` CLI command family.
- Multi-model comparison dashboards.
- Per-term fidelity surfaced in the Vue frontend (e.g., retrieval-panel side indicator).

## User-facing artifact

A single Python script invoked from the repo root:

```bash
python scripts/analyze_embedding_ontology.py \
    --model-name FremyCompany/BioLORD-2023-M \
    --k 10 \
    --n-pairs 50000
```

Produces `data/results/ontology_fidelity/<model-slug>_<YYYYMMDD-HHMMSS>/` containing:

| File | Description |
|---|---|
| `summary.json` | Headline numbers, per-branch breakdown, config echo (model, HPO DB path, HPO version if available, k, n-pairs, seed, UMAP params, timestamp). |
| `per_term_fidelity.csv` | Columns: `hpo_id`, `label`, `branch`, `depth`, `fidelity`, `rank`. Sorted ascending by fidelity (worst first). |
| `umap_coords.csv` | Columns: `hpo_id`, `label`, `umap_x`, `umap_y`, `branch`, `fidelity`, `depth`. |
| `umap_by_branch.png` | Plot 1. Static scatter colored by top-level HPO branch. |
| `umap_by_fidelity.png` | Plot 2. Same UMAP layout, colored by per-term fidelity (diverging colormap). |
| `umap_interactive.html` | Plots 1+2 as Plotly (hover shows id/label/definition/branch/fidelity). |
| `distance_correlation.png` | Plot 3. Hexbin of graph distance (shortest-path & Resnik panels) vs embedding cosine distance. Spearman ρ annotated. |
| `branch_fidelity.png` | Plot 4. Horizontal bar chart of mean fidelity per top-level HPO branch. |

## Metrics

Four metrics, all computed once per run.

1. **Global distance correlation.** Spearman ρ between pairwise cosine distance (embedding) and (a) shortest-path distance on the HPO is_a DAG, (b) Resnik similarity. Computed on `--n-pairs` sampled term pairs (default 50 000). Reports both correlations plus sample size.
2. **Per-term fidelity.** For each term *t*, compute the *k* nearest terms by embedding cosine and the *k* most Resnik-similar terms from the DAG. Fidelity(*t*) = |embedding-kNN ∩ DAG-kNN| / k, in [0, 1]. Exported as `per_term_fidelity.csv`; mean reported in summary.
3. **k-NN branch purity.** For each term, what fraction of its *k* embedding nearest neighbors share its top-level HPO branch. Mean overall + per-branch breakdown.
4. **Depth correlation.** Spearman ρ between term depth in the DAG and Euclidean distance from the centroid of the embedding cloud. Single scalar; surfaces whether deeper (more specific) terms tend toward the periphery of the embedding space.

All four live in `summary.json` under a `metrics` key with clearly-named subfields.

### Determinism rules

The above definitions are ambiguous for several edge cases in the HPO DAG. The rules below are binding — they define the computation and the tests lock them in.

- **k-NN self-exclusion.** Both the embedding k-NN and the DAG k-NN exclude the query term itself. `k` refers to the count *after* exclusion. Implemented by fitting a `NearestNeighbors` with `n_neighbors=k+1` and dropping the self-hit.
- **Resnik top-k tie-breaking.** Candidates are sorted by Resnik similarity descending; ties are broken by ascending HPO ID (lexicographic). If fewer than *k* non-self candidates have Resnik > 0, the remaining slots are filled with the lexicographically smallest non-self HPO IDs that have not been chosen yet. (A term deep in a sparse branch may otherwise have < *k* meaningful DAG neighbors.) This fallback is rare — we log a count of terms that hit it.
- **Multi-parent top-level branch.** HPO is a DAG; a term can have multiple depth-1 ancestors. For deterministic branch assignment we take the set of depth-1 ancestors reachable from *t* (plus *t* itself if `depth(t) == 1`), sort ascending by HPO ID, and pick the first. The full set is recorded in the per-term CSV as `all_branches` (semicolon-separated) for audit. `summary.json` notes the count of multi-parent terms that hit this tiebreaker.
- **Branch purity** uses the single deterministic assignment from the previous rule. A neighbor "shares" the branch iff its own deterministic branch equals the query term's. Purity = share-count / *k*.
- **Zero-descendant terms (leaves).** `descendants(t)` is defined to always include *t* itself, so `|descendants| ≥ 1` for every term and IC is always well-defined. IC(root) = 0 by construction. IC of a leaf = `log(N)` where N = `|descendants(root)|`.
- **Depth-0 terms.** Only the HPO root (`HP:0000001`) has depth 0. It has no top-level branch (`branch = None`), is excluded from branch purity, and its per-term fidelity and depth correlation contributions are still computed normally (it is still a valid k-NN query).
- **Randomness.** `--seed` is the sole entropy source. It seeds (a) the pair sampler for metric 1, (b) `umap.UMAP(random_state=seed)`, (c) any other stochastic step. Reruns with the same seed, same embeddings, and same HPO DB must produce byte-identical `summary.json` metrics and `per_term_fidelity.csv`.
- **Centroid for depth correlation.** Computed on the aligned term set (post HPO↔embedding intersection), not on the full embedding matrix.
- **Alignment warnings are non-fatal below threshold.** If the symmetric difference between HPO DB term IDs and cached embedding IDs is > 5% of either side, abort with an error — the index is likely stale. Otherwise log a warning with counts and proceed on the intersection.

## Architecture

Two new Python modules, following the established `scripts/` + `phentrieve/` split.

### `phentrieve/analysis/ontology_fidelity.py` (new module)

Pure functions, no I/O, no matplotlib. Public surface:

```python
def build_descendants_index(ancestors: dict[str, set[str]]) -> dict[str, set[str]]: ...
def graph_shortest_path(u: str, v: str, ancestors, depths) -> int: ...
def information_content(descendants, root: str = "HP:0000001") -> dict[str, float]: ...
def resnik_similarity(u: str, v: str, ancestors, ic) -> float: ...
def top_level_branch(term_id: str, ancestors, depths) -> str | None: ...
def sample_pairs(n_terms: int, n_pairs: int, rng) -> np.ndarray: ...
def global_distance_correlation(
    term_ids, embeddings, ancestors, depths, ic, n_pairs=50_000, seed=42
) -> dict[str, float]: ...
def per_term_fidelity(
    term_ids, embeddings, ancestors, descendants, ic, k=10
) -> list[dict]: ...
def branch_knn_purity(
    term_ids, embeddings, branch_map, k=10
) -> dict[str, float]: ...
def depth_correlation(term_ids, embeddings, depths) -> float: ...
```

Inputs are plain dicts and numpy arrays; no database or ChromaDB dependencies. This is the unit-testable core.

### `phentrieve/analysis/embedding_cache.py` (new module)

```python
def load_cached_embeddings(
    model_name: str,
    refresh: bool = False,
    index_dir_override: str | None = None,
) -> tuple[list[str], np.ndarray]: ...
```

**Cache path.** ChromaDB stores all single-vector collections in one shared directory (`data/indexes/chroma.sqlite3` plus Chroma-managed subdirs) — it is *not* a per-collection layout. To avoid collisions across models, the fidelity cache lives in a dedicated sibling directory:

```
<index_dir>/ontology_fidelity_cache/<collection_name>/
    embeddings.npy      # shape (N, D), float32
    hpo_ids.json        # ["HP:0000001", ...] — same order as rows in embeddings.npy
    meta.json           # {"model_name": ..., "collection_name": ..., "written_at": ISO8601, "n_terms": N, "dim": D}
```

- `index_dir` comes from `phentrieve.utils.get_default_index_dir()` (overridable via `index_dir_override`, matching the orchestrator's interface).
- `collection_name` is the single-vector name returned by the existing `generate_collection_name(model_name)` helper (no `_multi` suffix — this pass explicitly excludes multi-vector).
- The cache directory is created on first write; `.gitignore` already covers `data/indexes/`.

**Behavior.**
- First call (or `refresh=True`): resolve the ChromaDB collection via existing single-vector helpers, call `collection.get(include=["embeddings"])` in one shot (≈18k × 768 floats → ~55 MB, acceptable), write `embeddings.npy`, `hpo_ids.json`, and `meta.json`, return the arrays.
- Subsequent calls: read `hpo_ids.json` and `embeddings.npy` directly — no Chroma dependency loaded at runtime.
- `refresh=True` deletes the three files before rewriting.
- If the cache exists but is malformed (missing file, mismatched row count between ids and array, unparsable meta), log an error and fall back to a full refresh rather than crashing.
- If the source ChromaDB collection does not exist, raise `FileNotFoundError` with a message pointing to `phentrieve index build --model-name ...`.

### `scripts/analyze_embedding_ontology.py` (new script)

Orchestrator. Responsibilities in order:

1. Parse CLI flags (see "Flags" below).
2. `setup_logging` at the requested level.
3. Open the HPO database directly: resolve its path via `phentrieve.utils` (`resolve_data_path` + `DEFAULT_HPO_DB_FILENAME`), and if the file does not exist, raise `FileNotFoundError` pointing to `phentrieve data prepare` and exit 1. *Do not* use `phentrieve.data_processing.document_creator.load_hpo_terms()` — its soft-fail (log + return `[]`) swallows this error and violates the hard-fail contract in "Error handling".
4. Construct `HPODatabase(db_path)` and, in a single open/close cycle, call:
   - `load_all_terms()` → list of `{id, label, definition, synonyms, comments}` records.
   - `load_graph_data()` → `(ancestors, depths)`.
   - `get_metadata("hpo_version")` → `hpo_version` string (may be `None`; record `null` in summary.json).
   - Close the connection immediately after.
5. Call `load_cached_embeddings(model_name, refresh=...)`.
6. Align `hpo_ids` ↔ loaded terms (build the intersection; apply the alignment rule from "Determinism rules" — abort if the symmetric difference is > 5%, otherwise warn and proceed).
7. Build `descendants`, `ic`, `branch_map` via analysis module.
8. Compute metrics 1–4.
9. Fit UMAP (`umap.UMAP(n_neighbors=..., min_dist=..., metric=..., random_state=seed)`).
10. Write CSVs, PNGs, optional HTML, and `summary.json` (which echoes `hpo_db_path`, `hpo_version`, `collection_name`, `embedding_dim`, `n_terms`, and the flag values).
11. Return exit code.

Script is ~200–300 lines, mostly glue; plotting functions live in-file since they are output-shaped and not reused.

### Flags

```
--model-name STR        default: FremyCompany/BioLORD-2023-M
--output-dir PATH       default: data/results/ontology_fidelity/
--k INT                 default: 10
--n-pairs INT           default: 50000
--umap-neighbors INT    default: 15
--umap-min-dist FLOAT   default: 0.1
--metric STR            default: cosine
--sample INT            default: None  (sample N HPO terms; None = all)
--seed INT              default: 42
--skip-interactive      flag — skip Plotly HTML
--refresh-cache         flag — re-read embeddings from ChromaDB
--log-level STR         default: INFO
```

## Data flow

```
HPO DB (sqlite)          ChromaDB collection
    |                            |
    v                            v
ancestors, depths       embeddings.npy cache (first run)
    |                            |
    +------- align ids ----------+
                |
                v
        analysis module (pure)
          |        |        |
          v        v        v
       metrics  fidelity  UMAP coords
          |        |        |
          +--------+--------+
                   |
                   v
          plotting + writers
                   |
                   v
    data/results/ontology_fidelity/<slug>_<ts>/
```

## Performance

- Cold runtime target: ≤ 5 min (includes ChromaDB read).
- Warm runtime target: ≤ 3 min.
- Peak memory target: ≤ 4 GB (during UMAP).
- `--n-pairs 50000` keeps graph-distance computation in-memory and yields a stable Spearman ρ; tunable.
- k-NN uses `sklearn.neighbors.NearestNeighbors(metric='cosine')` — avoids materializing the full 18k × 18k cosine matrix.
- `--skip-interactive` skips the ~30s Plotly HTML serialization.
- `--sample` is for smoke testing only, not real runs.

## Error handling

- Missing HPO DB → raise `FileNotFoundError`, message points to `phentrieve data prepare`, exit code 1. Explicitly not the log-and-return-empty behavior of `load_hpo_terms()`.
- Missing ChromaDB collection for model → raise `FileNotFoundError` from `load_cached_embeddings`, message points to `phentrieve index build --model-name ...`, exit code 1.
- ID mismatch between HPO DB and cached embeddings → if the symmetric difference exceeds 5% of either set, abort with exit code 1 (likely stale index); otherwise log a warning with counts and proceed on the intersection (see "Determinism rules").
- Zero intersection → error, exit code 1.
- Malformed cache (missing file, row-count mismatch, unparsable `meta.json`) → log error, delete the cache, re-read from ChromaDB once, retry. If still malformed → exit code 1.
- UMAP failure → propagate exception with context; no silent swallowing.
- Any `--sample` value ≥ total terms is clamped to the total (with a log info line).

## Dependencies

Audit of the current `pyproject.toml` core deps:

- `matplotlib>=3.10.8` — **already core.** Used as-is.
- `scikit-learn>=1.4.0` — **already core.** Used as-is (pulls `scipy` transitively, so `scipy` is available without a direct pin).
- `pandas>=2.0.0`, `numpy>=2.0.0`, `networkx>=3.0` — **already core.** Used as-is.

Genuinely new dependencies — added under a new optional extra `analysis` in `pyproject.toml`:

```
[project.optional-dependencies]
analysis = [
    "umap-learn>=0.5.6",
    "plotly>=5.22.0",
]
```

Rationale for optional (not core):
- `umap-learn` pulls `numba` + `llvmlite`, a heavy install not needed by the CLI, API, MCP server, or test suite.
- `plotly` is only used by the interactive HTML output; static PNGs cover the common case.
- The script gracefully reports a friendly install hint (`pip install 'phentrieve[analysis]'` or `uv sync --extra analysis`) if the imports fail at top-level, rather than a bare `ModuleNotFoundError`.

Install: `uv sync --extra analysis`. `scipy` is *not* pinned directly; if CI surfaces a version issue we add an explicit pin in a follow-up commit. No changes to the core install set.

## Testing

All tests under `tests/unit/analysis/` and `tests/integration/analysis/`, mirroring the existing layout.

### Unit tests (`tests/unit/analysis/test_ontology_fidelity.py`)

- `test_graph_shortest_path_known_pairs` — hand-built 5-term DAG.
- `test_graph_shortest_path_identical_terms_is_zero`.
- `test_resnik_similarity_identical_terms_equals_own_ic`.
- `test_information_content_root_is_zero`.
- `test_build_descendants_index_matches_manual_inverse`.
- `test_top_level_branch_for_depth_1_term_is_self`.
- `test_top_level_branch_for_depth_0_term_is_none`.
- `test_top_level_branch_multi_parent_picks_lex_smallest` — locks in the multi-parent determinism rule.
- `test_per_term_fidelity_bounds_0_to_1`.
- `test_per_term_fidelity_excludes_self_from_knn` — both embedding-kNN and DAG-kNN.
- `test_per_term_fidelity_tiebreak_is_lexicographic` — construct a DAG with tied Resnik scores.
- `test_per_term_fidelity_identical_embeddings_yields_uniform_baseline`.
- `test_global_distance_correlation_returns_expected_keys`.
- `test_global_distance_correlation_seeded_is_deterministic` — two runs with the same seed produce identical ρ.
- `test_branch_knn_purity_monocluster_equals_one`.
- `test_branch_knn_purity_excludes_root_from_denominator`.
- `test_depth_correlation_returns_scalar_in_expected_range`.

### Unit test for cache

- `test_embedding_cache_writes_and_reloads` (uses temp dir, mocks ChromaDB collection).
- `test_embedding_cache_refresh_overwrites`.

### Integration smoke test (`tests/integration/analysis/test_analyze_script_smoke.py`, marked `slow`)

- Runs `scripts/analyze_embedding_ontology.py --sample 200` against the real HPO DB with patched `load_cached_embeddings` returning synthetic vectors.
- Asserts output directory contains all 8 expected files.
- Asserts `summary.json` parses and contains required keys.
- Asserts `per_term_fidelity.csv` has expected columns and row count.
- Does not hit ChromaDB; does not fit full UMAP on real BioLORD embeddings.

### Coverage expectation

All new code under `phentrieve/analysis/` must reach ≥ 85% line coverage. Script orchestration may be below that since some branches are exercised only in integration.

## Documentation

- `scripts/README.md` — add section documenting `analyze_embedding_ontology.py` with example invocation and description of outputs.
- Project docs — no new mkdocs page; a short entry in the analysis/tools section if one exists, otherwise the script README is the canonical reference.

## Git / PR process

- Work happens on branch `feat/ontology-embedding-fidelity` in a worktree (`~/worktrees/phentrieve-ontology-fidelity/`).
- Atomic commits: one per module (`analysis/embedding_cache.py`, `analysis/ontology_fidelity.py`, `scripts/analyze_embedding_ontology.py`, tests, pyproject extra, docs).
- PR references #34 and proposes closing it with a pointer to the script.

## Open questions deferred to implementation plan

- Whether the distance-correlation hexbin renders both shortest-path and Resnik in a single figure (two subplots) or separate files; leaning toward two subplots.
- Plotly figure: single HTML with two tabs/buttons toggling color scheme, or two separate HTMLs. Leaning toward single HTML with toggle.

These are decided during the implementation plan (next step), not this spec.
