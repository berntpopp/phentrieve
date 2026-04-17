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
- No re-indexing. Embeddings are read from the existing ChromaDB collection once and cached next to it.
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

Four metrics, all computed once per run:

1. **Global distance correlation.** Spearman ρ between pairwise cosine distance (embedding) and (a) shortest-path distance on the HPO is_a DAG, (b) Resnik similarity. Computed on `--n-pairs` sampled term pairs (default 50 000). Reports both correlations plus sample size.
2. **Per-term fidelity.** For each term *t*, compute the *k* nearest terms by embedding cosine and the *k* most Resnik-similar terms from the DAG. Fidelity(*t*) = |embedding-kNN ∩ DAG-kNN| / k, in [0, 1]. Exported as `per_term_fidelity.csv`; mean reported in summary.
3. **k-NN branch purity.** For each term, what fraction of its *k* embedding nearest neighbors share its top-level HPO branch. Mean overall + per-branch breakdown.
4. **Depth correlation.** Spearman ρ between term depth in the DAG and Euclidean distance from the centroid of the embedding cloud. Single scalar; surfaces whether deeper (more specific) terms tend toward the periphery of the embedding space.

All four live in `summary.json` under a `metrics` key with clearly-named subfields.

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
    model_name: str, refresh: bool = False
) -> tuple[list[str], np.ndarray]: ...
```

On first call: queries the existing ChromaDB collection via the current Phentrieve helpers, writes `embeddings.npy` + `hpo_ids.json` next to the collection directory, returns the arrays. On subsequent calls: reads from disk. `refresh=True` forces a re-read.

The cache location is derived from the same path logic used by `phentrieve/indexing/chromadb_orchestrator.py`, so no new configuration is introduced. Collection resolution uses the existing single-vector helpers.

### `scripts/analyze_embedding_ontology.py` (new script)

Orchestrator. Responsibilities in order:

1. Parse CLI flags (see "Flags" below).
2. `setup_logging` at the requested level.
3. Load HPO graph via `HPODatabase.load_graph_data()` → `ancestors`, `depths`.
4. Load HPO term metadata (id, label, definition) via `load_hpo_terms`.
5. Call `load_cached_embeddings(model_name, refresh=...)`.
6. Align `hpo_ids` ↔ loaded terms (filter to intersection; warn on mismatches).
7. Build `descendants`, `ic`, `branch_map` via analysis module.
8. Compute metrics 1–4.
9. Fit UMAP (`umap.UMAP(n_neighbors=..., min_dist=..., metric=..., random_state=seed)`).
10. Write CSVs, PNGs, optional HTML, and `summary.json`.
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

- Missing HPO DB → error message pointing to `phentrieve data prepare`, exit code 1.
- Missing ChromaDB collection for model → error message pointing to `phentrieve index build --model-name ...`, exit code 1.
- ID mismatch between graph and embeddings → log warning with counts, proceed on intersection.
- Zero intersection → error, exit code 1.
- UMAP failure → propagate exception with context; no silent swallowing.
- Any `--sample` value ≥ total terms is clamped silently (with a log info line).

## Dependencies

Added to `pyproject.toml` under a new optional extra `analysis`:

- `umap-learn`
- `matplotlib`
- `plotly`
- `scipy` (verify if already transitive; pin if not)

Install: `uv sync --extra analysis`. Core runtime install stays lean. scikit-learn is already an indirect dependency via sentence-transformers; the spec explicitly depends on it and pins if necessary.

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
- `test_per_term_fidelity_bounds_0_to_1`.
- `test_per_term_fidelity_identical_embeddings_yields_uniform_baseline`.
- `test_global_distance_correlation_returns_expected_keys`.
- `test_branch_knn_purity_monocluster_equals_one`.
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

- Exact HPO version echoing: whether we embed the version string from the SQLite schema or compute a hash of the `hp.json` bundle.
- Whether the distance-correlation hexbin renders both shortest-path and Resnik in a single figure (two subplots) or separate files; leaning toward two subplots.
- Plotly figure: single HTML with two tabs/buttons toggling color scheme, or two separate HTMLs. Leaning toward single HTML with toggle.

These are decided during the implementation plan (next step), not this spec.
