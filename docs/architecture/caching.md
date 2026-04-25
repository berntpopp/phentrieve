# Caching Strategy

Phentrieve uses two distinct caching patterns, each for a different purpose. This document explains when to use which.

## Pattern 1 — `cachetools.TTLCache` with `_cache_lock` (API model cache)

**Where**: `api/dependencies.py`

**What it caches**:
- `LOADED_SBERT_MODELS` — loaded `SentenceTransformer` instances, keyed by model name
- `LOADED_RETRIEVERS` — loaded `DenseRetriever` instances, keyed by `f"retriever_for_{name}_multi={bool}"`
- `MODEL_LOADING_STATUS` — per-model load state machine (`loading` / `loaded` / `failed`)
- `MODEL_LOAD_LOCKS` — per-model asyncio locks for concurrent-request deduplication

**Why TTLCache**:
- Long-running API processes can see many distinct model names over their lifetime (e.g. a benchmark workload that rotates through models). Unbounded dicts would leak.
- TTL (3600s) ensures stale entries are reclaimed even under write pressure.
- `maxsize=10` for the model caches, `maxsize=50` for the tracking caches (bigger because the entries are tiny).
- `_cache_lock` (threading.Lock) serializes mutations — `TTLCache` itself is not thread-safe.
- All writes use `with _cache_lock:`; all reads use `.get(key)` which is safe even when an entry expires between check and read.

**Eviction**: Automatic on TTL expiry; LRU eviction when over `maxsize`.

**Lifecycle**: `cleanup_model_caches()` runs in the FastAPI lifespan shutdown. It first cancels any in-flight background loading tasks via `asyncio.shield` + `gather(return_exceptions=True)`, then clears all four caches under `_cache_lock`.

## Pattern 2 — `@functools.lru_cache` (process-wide singletons)

**Where**:
- `phentrieve/utils.py::load_user_config` (maxsize=1) — the YAML config dict, loaded once per process
- `phentrieve/utils.py::normalize_id` (maxsize=512) — HPO ID normalization results (e.g. `HP:0001250` → `HP:0001250`); bounded by the HPO term count
- `phentrieve/evaluation/metrics.py::_load_hpo_graph_data_impl` (maxsize=1) — the ~2M ontology ancestor graph, loaded once; `load_hpo_graph_data()` is a thin wrapper that normalises its `db_path` argument before delegating here
- `phentrieve/retrieval/details_enrichment.py::get_shared_database` (maxsize=1) — one SQLite `HPODatabase` connection per db path
- `phentrieve/config.py::_load_yaml_config` (maxsize=1) — wraps `load_user_config()` with a lazy import to avoid circular imports
- `phentrieve/retrieval/aggregation.py::_parse_formula` (maxsize=32) — parsed AST for custom scoring formulas; bounded by the number of distinct formula strings
- `api/routers/similarity_router.py::_get_hpo_label_map_api` (maxsize=1) — HPO label map for the similarity API router, loaded once on first request
- `api/version.py::get_api_version`, `get_cli_version` (maxsize=1 each) — package version strings read from `pyproject.toml` files

**Why `@lru_cache` and not TTL**:
- These are **true singletons**: the underlying data is immutable within a process run. Reloading would waste memory and time with no behavioral change.
- The key space is small and bounded: one config file, one DB path, one HPO JSON, one ontology graph. Unlike the API model cache, a CLI run or benchmark pass does not rotate through dozens of distinct keys.
- Python's `@lru_cache` is thread-safe for reads.

**Note on `phentrieve/embeddings.py`**: The embedding model cache does NOT use `@lru_cache`. Instead it uses a custom `_MODEL_REGISTRY` dict with a `_REGISTRY_LOCK` (`threading.Lock`). This gives finer control: models can be moved between devices and the registry updated atomically, which `@lru_cache` does not support. Use `clear_model_registry()` to free the cache.

**When to NOT use `@lru_cache`**:
- Anything where the key space is user-controlled and effectively unbounded (request parameters, query strings). Use `TTLCache` with a bounded `maxsize` instead.
- Anything with a lifecycle managed by an app factory (FastAPI startup/shutdown). Use module-level `TTLCache` + `cleanup_*` in lifespan hooks.
- Mutable state. `@lru_cache` caches **outputs**, not state — do not use it for anything that writes.
- Anything needing device-aware cache invalidation (see the embeddings registry pattern instead).

## Decision matrix

| Situation | Use |
|---|---|
| API request-scoped, key may grow unbounded | `TTLCache(maxsize=N, ttl=T)` + lock |
| Process-wide singleton, immutable data, bounded keys | `@lru_cache(maxsize=M)` |
| Per-request memoization | FastAPI dependency with `Depends()`, no caching |
| Coroutine/async safety across the event loop | `asyncio.Lock` per key (see `_get_lock_for_model`) |
| Mutable cache with device-aware invalidation | Custom dict + `threading.Lock` (see `_MODEL_REGISTRY`) |
| Tests need to reset caches | Both patterns expose `.cache_clear()` — call it in fixtures |

## Audit checklist

When adding a new cache:
- [ ] Is the key space bounded? (If no → TTLCache)
- [ ] Is the data mutable during the process run? (If yes → don't cache, or use custom registry)
- [ ] Is there a lifecycle hook to clear it? (If it's module-level, add it to the relevant `cleanup_*` function)
- [ ] Are tests cleaning up? (`cache.cache_clear()` in `setup_method` / fixture teardown)
- [ ] Is there a concurrency concern? (If multi-threaded writes possible → wrap in `_cache_lock`)

## History

- **Pre-PR #191**: `api/dependencies.py` used **unbounded** module-level dicts for model and retriever caches.
- **PR #191 (2026-04-10)**: Model caches converted to `TTLCache`, `_cache_lock` introduced, `cleanup_model_caches()` wired into lifespan shutdown.
- **PR #191 follow-up**: `MODEL_LOADING_STATUS` and `MODEL_LOAD_LOCKS` also converted to `TTLCache` to close the unbounded-tracking-dict gap.
