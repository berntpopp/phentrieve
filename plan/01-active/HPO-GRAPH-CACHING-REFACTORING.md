# HPO Graph Caching Refactoring

**Status**: Ready for Implementation
**Created**: 2025-11-19
**Priority**: Medium
**Estimated Effort**: 2-3 hours
**Risk Level**: Low
**Last Updated**: 2025-11-19

---

## Objective

Refactor global state-based caching in `phentrieve/evaluation/metrics.py` to use thread-safe, testable `@lru_cache` pattern, eliminating anti-pattern while following SOLID, KISS, and DRY principles.

---

## Problem Statement

### Current Anti-Pattern

**Location**: `phentrieve/evaluation/metrics.py:51-52`

```python
# Global caches for HPO graph data
_hpo_ancestors: dict[str, set[str]] | None = None
_hpo_term_depths: dict[str, int] | None = None
```

### Issues Identified

1. **❌ Mutable Global State**: Makes testing difficult (test pollution between tests)
2. **❌ Not Thread-Safe**: Race conditions in multi-threaded FastAPI environment
3. **❌ Hidden Dependencies**: Functions implicitly depend on global state
4. **❌ Hard to Test**: Requires patching module-level variables
5. **❌ Inconsistent Pattern**: Other parts of codebase use `@lru_cache` (`utils.py:217`, `api/routers/similarity_router.py:32`)

### Why NOT Singleton Pattern

The initially proposed Singleton pattern (`__new__`-based class) is **rejected** because:

- ❌ Over-engineered for a simple data loading function
- ❌ Still requires manual thread-safety implementation
- ❌ Non-Pythonic approach (violates PEP 20)
- ❌ Harder to test than functional approach
- ❌ Violates KISS principle
- ❌ Doesn't match existing codebase patterns

---

## Solution: `@lru_cache` Pattern

### Why This is Correct

| Criterion | Global Variables | Singleton Class | **`@lru_cache`** ✅ |
|-----------|------------------|-----------------|---------------------|
| Thread-safe | ❌ | ⚠️ Needs locks | ✅ Built-in |
| Testable | ❌ | ⚠️ Reset logic | ✅ `.cache_clear()` |
| Pythonic | ❌ | ❌ | ✅ Idiomatic |
| KISS | ⚠️ Simple but flawed | ❌ Complex | ✅ Minimal |
| DRY | ✅ | ❌ Reinvents wheel | ✅ Uses stdlib |
| Matches codebase | ❌ | ❌ | ✅ Yes |
| Concurrency | ❌ Unsafe | ⚠️ Manual | ✅ Lock-protected |

### Design Principles Applied

- **SOLID**: Single Responsibility - function just loads data
- **KISS**: Simplest solution that solves the problem
- **DRY**: Reuses `@lru_cache` pattern already in codebase
- **Pythonic**: Idiomatic use of decorators and pure functions

---

## Current State Analysis

### Files Affected

**Primary File:**
- `phentrieve/evaluation/metrics.py` - Contains global state and loading function

**Usage Locations:**
- `phentrieve/cli/similarity_commands.py` - CLI commands
- `api/routers/similarity_router.py` - API endpoint (line 107)
- `api/routers/config_info_router.py` - Config info endpoint (line 117)
- `phentrieve/evaluation/runner.py` - Benchmark evaluation (line 126)
- `phentrieve/evaluation/semantic_metrics.py` - Semantic similarity calculations

**Test Files:**
- `tests/unit/cli/test_similarity_commands.py` - Mocks `load_hpo_graph_data`
- `tests/unit/core/test_semantic_metrics.py` - Uses metrics functions

### Existing Caching Patterns in Codebase

Already using `@lru_cache` in:
1. `phentrieve/utils.py:217` - `load_user_config()`
2. `phentrieve/utils.py:328` - `normalize_id()`
3. `phentrieve/config.py:289` - Config loading
4. `api/config.py:35` - API config
5. `api/routers/similarity_router.py:32` - `_get_hpo_label_map_api()`

**This refactoring brings `load_hpo_graph_data()` in line with established patterns.**

---

## Implementation Plan

### Phase 1: Core Refactoring ⚙️

**File**: `phentrieve/evaluation/metrics.py`

**Tasks:**

1. **Add import** (top of file, line ~14):
   ```python
   from functools import lru_cache
   ```

2. **Remove global variables** (lines 51-52):
   ```python
   # DELETE THESE LINES:
   # _hpo_ancestors: dict[str, set[str]] | None = None
   # _hpo_term_depths: dict[str, int] | None = None
   ```

3. **Add decorator to function** (line 55):
   ```python
   @lru_cache(maxsize=1)  # Thread-safe cache for HPO graph data
   def load_hpo_graph_data(
       db_path: str | None = None,
       ancestors_path: str | None = None,  # Deprecated, kept for compatibility
       depths_path: str | None = None,  # Deprecated, kept for compatibility
   ) -> tuple[dict[str, set[str]], dict[str, int]]:
   ```

4. **Remove global statement** (line 76):
   ```python
   # DELETE THIS LINE:
   # global _hpo_ancestors, _hpo_term_depths
   ```

5. **Remove cache check** (lines 78-81):
   ```python
   # DELETE THESE LINES:
   # if _hpo_ancestors is not None and _hpo_term_depths is not None:
   #     logging.debug("Using cached HPO graph data")
   #     return _hpo_ancestors, _hpo_term_depths
   ```

6. **Remove global assignments** (line 104):
   ```python
   # CHANGE:
   # _hpo_ancestors, _hpo_term_depths = db.load_graph_data()

   # TO:
   ancestors, depths = db.load_graph_data()
   ```

7. **Update logging statements** (lines 107-133):
   - Change `_hpo_ancestors` → `ancestors`
   - Change `_hpo_term_depths` → `depths`

8. **Update return statement** (line 134):
   ```python
   # CHANGE:
   # return _hpo_ancestors, _hpo_term_depths

   # TO:
   return ancestors, depths
   ```

9. **Update docstring**:
   ```python
   """
   Load precomputed HPO graph data from SQLite database.

   Args:
       db_path: Path to the HPO SQLite database file (preferred)
       ancestors_path: (Deprecated) Path to ancestors pickle file - for backward compatibility
       depths_path: (Deprecated) Path to depths pickle file - for backward compatibility

   Returns:
       Tuple of (ancestors_dict, depths_dict)
       - ancestors_dict: {term_id: set of ancestor IDs}
       - depths_dict: {term_id: depth from root}

   Note:
       Results are cached using @lru_cache. Call `load_hpo_graph_data.cache_clear()`
       to reset the cache if needed (e.g., in tests or after data updates).
       Returns empty dictionaries if database not found or loading fails.
   """
   ```

**Estimated Time**: 30 minutes
**Risk**: Low - Function signature unchanged, all callers work as-is

---

### Phase 2: Test Updates 🧪

**Files**:
- `tests/conftest.py` (root fixtures)
- `tests/unit/conftest.py` (unit test fixtures)
- `tests/unit/cli/test_similarity_commands.py`

**Tasks:**

1. **Add opt-in cache-clearing fixtures** to `tests/conftest.py`:

   **Note**: Originally planned with `autouse=True`, but implemented as opt-in fixtures to avoid
   test suite slowdown. Most unit tests mock `load_hpo_graph_data` and don't need cache clearing.
   Only integration tests that load real HPO data should use these fixtures.

   ```python
   import pytest

   @pytest.fixture
   def fresh_hpo_graph_data():
       """
       Opt-in fixture for tests that need fresh HPO graph data.

       Clears the cache before and after the test to ensure isolation.
       Use this fixture when your test:
       - Needs to load real HPO data (not mocked)
       - Requires fresh data without cached state
       - Tests caching behavior itself
       """
       from phentrieve.evaluation.metrics import load_hpo_graph_data

       # Clear cache before test
       load_hpo_graph_data.cache_clear()

       # Load fresh data
       data = load_hpo_graph_data()

       yield data

       # Clear cache after test for isolation
       load_hpo_graph_data.cache_clear()

   @pytest.fixture
   def clear_hpo_cache():
       """
       Minimal fixture that just clears the HPO graph data cache.

       Use this when you need cache clearing but don't need the actual data.
       Useful for integration tests that call functions which internally
       load HPO data.
       """
       from phentrieve.evaluation.metrics import load_hpo_graph_data

       # Clear before test
       load_hpo_graph_data.cache_clear()

       yield

       # Clear after test
       load_hpo_graph_data.cache_clear()
   ```

2. **Remove global variable reset logic** from existing tests:
   - Search for `_hpo_ancestors` and `_hpo_term_depths` in test files
   - Remove any direct assignments like `sim_module._hpo_ancestors = None`
   - These are no longer needed with cache clearing fixture

3. **Verify existing mocks still work**:
   - Tests already mock `load_hpo_graph_data` at function level
   - No changes needed to mock decorators (they work the same)

4. **Run test suite**:
   ```bash
   # Fast verification
   pytest tests/unit/cli/test_similarity_commands.py -v

   # Full test suite
   make test
   ```

**Estimated Time**: 30 minutes
**Risk**: Low - Tests already mock at function level

---

### Phase 3: Verification & Testing ✅

**Tasks:**

1. **Run full test suite**:
   ```bash
   make test
   ```
   Expected: All tests pass (115+ unit/integration tests)

2. **Type checking**:
   ```bash
   make typecheck-fast
   ```
   Expected: 0 mypy errors (maintain current standard)

3. **Linting**:
   ```bash
   make check
   ```
   Expected: 0 Ruff errors

4. **Manual verification** - Test cache behavior:
   ```python
   # In Python REPL or test script
   from phentrieve.evaluation.metrics import load_hpo_graph_data

   # First call - loads from database
   ancestors1, depths1 = load_hpo_graph_data()
   print(f"Loaded: {len(ancestors1)} ancestors, {len(depths1)} depths")

   # Second call - returns cached result (fast)
   ancestors2, depths2 = load_hpo_graph_data()
   assert ancestors1 is ancestors2  # Same object (cached)
   assert depths1 is depths2

   # Clear cache
   load_hpo_graph_data.cache_clear()

   # Third call - reloads from database
   ancestors3, depths3 = load_hpo_graph_data()
   assert ancestors3 is not ancestors1  # New objects
   ```

5. **API integration test**:
   ```bash
   # Start API locally
   cd api && python run_api_local.py

   # Test similarity endpoint
   curl http://localhost:8734/api/similarity/HP:0001197/HP:0000750
   ```
   Expected: Returns similarity score successfully

6. **CLI integration test**:
   ```bash
   phentrieve similarity calculate HP:0001197 HP:0000750
   ```
   Expected: Returns similarity calculation

**Estimated Time**: 30 minutes
**Risk**: Low

---

### Phase 4: Documentation Updates 📚

**Files:**
- `CLAUDE.md` (project instructions)
- `plan/STATUS.md` (project status)

**Tasks:**

1. **Update `CLAUDE.md`** - Architecture section:
   ```markdown
   ### Key Architecture Patterns

   **Caching Pattern:**
   - Use `@lru_cache` for expensive one-time data loading
   - Examples:
     - `load_hpo_graph_data()` - HPO ontology graph structure
     - `load_user_config()` - Configuration loading
     - `_get_hpo_label_map_api()` - HPO label mapping
   - Thread-safe by default (Python 3.9+)
   - Clear cache in tests: `function_name.cache_clear()`
   ```

2. **Update this plan status**:
   - Move from `01-active/` to `02-completed/`
   - Add completion date

3. **Add comments in code** (`metrics.py`):
   ```python
   @lru_cache(maxsize=1)  # Thread-safe cache for HPO graph data
   def load_hpo_graph_data(...) -> tuple[dict[str, set[str]], dict[str, int]]:
       """
       ...

       Thread Safety:
           This function is thread-safe due to @lru_cache's built-in locking.
           Multiple concurrent calls will wait for the first load to complete.
       """
   ```

**Estimated Time**: 30 minutes
**Risk**: None

---

## Success Criteria

- [ ] Global variables `_hpo_ancestors` and `_hpo_term_depths` removed
- [ ] `@lru_cache(maxsize=1)` decorator added to `load_hpo_graph_data()`
- [ ] All references to global state removed from function body
- [ ] Test fixture added for cache clearing
- [ ] All 115+ tests pass (`make test`)
- [ ] 0 type errors (`make typecheck-fast`)
- [ ] 0 linting errors (`make check`)
- [ ] Manual cache behavior verified (load, cache hit, clear, reload)
- [ ] API endpoint works correctly
- [ ] CLI command works correctly
- [ ] Documentation updated

---

## Rollback Plan

If issues arise, revert with:

```bash
# 1. Git reset to previous commit
git log --oneline -5  # Find commit before changes
git reset --hard <commit-hash>

# 2. Or manual revert of metrics.py changes:
#    - Remove @lru_cache decorator
#    - Add back global variables
#    - Add back global statement
#    - Add back cache check logic
#    - Restore global assignments

# 3. Remove test fixture from conftest.py

# 4. Verify rollback
make test
make typecheck-fast
make check
```

**Recovery Time**: 10 minutes

---

## Benefits

### Immediate Benefits

1. **Thread Safety**: No more race conditions in FastAPI concurrent requests
2. **Testability**: Easy cache clearing with `.cache_clear()`
3. **Code Quality**: Eliminates global state anti-pattern
4. **Consistency**: Matches existing codebase patterns
5. **Maintainability**: Simpler, more Pythonic code

### Long-Term Benefits

1. **Scalability**: Safe for multi-threaded/async environments
2. **Debugging**: Clearer function dependencies
3. **Refactoring**: Easier to move/reorganize code
4. **Onboarding**: Familiar pattern for new developers
5. **Future-Proof**: Aligned with Python best practices

---

## Alternative Approaches Considered

### 1. Singleton Pattern with `__new__` (Rejected)

**Proposed Pattern:**
```python
class HPOGraphService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ancestors = None
            cls._instance._depths = None
        return cls._instance
```

**Rejected Because:**
- ❌ Over-engineered for simple data loading
- ❌ Violates KISS principle
- ❌ Requires manual thread-safety implementation
- ❌ Non-Pythonic (violates PEP 20: "Simple is better than complex")
- ❌ Harder to test than functional approach
- ❌ More boilerplate code to maintain

### 2. FastAPI Dependency Injection Only (Rejected)

**Proposed Pattern:**
```python
# In api/dependencies.py
async def get_hpo_graph_data_dependency():
    return load_hpo_graph_data()
```

**Rejected Because:**
- ⚠️ Only works for API layer
- ⚠️ CLI and evaluation code still need direct access
- ⚠️ HPO graph data is an "ambient dependency" (used everywhere)
- ℹ️ Can be added later as optional enhancement

### 3. Service Locator Pattern (Rejected)

**Rejected Because:**
- ❌ Over-engineering for this use case
- ❌ Introduces indirection without clear benefit
- ❌ Not needed for stateless data loading

---

## Related Documentation

- **Codebase Patterns**: See `phentrieve/utils.py:217` for similar `@lru_cache` usage
- **Testing Patterns**: See `tests/conftest.py` for fixture patterns
- **FastAPI Best Practices**: [Dependencies with yield](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/)
- **Python `lru_cache` Docs**: [functools.lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache)

---

## Implementation Checklist

### Pre-Implementation
- [ ] Read this plan thoroughly
- [ ] Create feature branch: `git checkout -b refactor/hpo-graph-caching`
- [ ] Ensure all tests pass on main: `make test`
- [ ] Backup current state

### Phase 1: Core Refactoring
- [ ] Add `from functools import lru_cache` import
- [ ] Remove global variable declarations (lines 51-52)
- [ ] Add `@lru_cache(maxsize=1)` decorator to function
- [ ] Remove `global` statement (line 76)
- [ ] Remove cache check logic (lines 78-81)
- [ ] Update variable assignments (line 104: `ancestors, depths = ...`)
- [ ] Update all logging statements (replace `_hpo_ancestors` → `ancestors`, `_hpo_term_depths` → `depths`)
- [ ] Update return statement (line 134: `return ancestors, depths`)
- [ ] Update function docstring
- [ ] Run `make check` (formatting/linting)

### Phase 2: Test Updates
- [ ] Add cache-clearing fixture to `tests/conftest.py`
- [ ] Remove any global variable reset logic from tests
- [ ] Run `pytest tests/unit/cli/test_similarity_commands.py -v`
- [ ] Run `make test` (full test suite)

### Phase 3: Verification
- [ ] Run `make typecheck-fast` (0 errors expected)
- [ ] Run `make check` (0 errors expected)
- [ ] Manual cache behavior test (REPL verification)
- [ ] API integration test (`curl` endpoint)
- [ ] CLI integration test (`phentrieve similarity`)

### Phase 4: Documentation
- [ ] Update `CLAUDE.md` architecture section
- [ ] Add inline code comments about thread-safety
- [ ] Update this plan status to completed

### Finalization
- [ ] Review all changes: `git diff`
- [ ] Commit with message: `refactor: Replace global HPO graph cache with @lru_cache for thread-safety`
- [ ] Push feature branch
- [ ] Create PR with this plan as description
- [ ] Wait for CI to pass
- [ ] Merge to main
- [ ] Move plan to `02-completed/`

---

## Timeline

**Total Estimated Time**: 2-3 hours

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Core Refactoring | 30 min | None |
| Phase 2: Test Updates | 30 min | Phase 1 |
| Phase 3: Verification | 30 min | Phase 1-2 |
| Phase 4: Documentation | 30 min | Phase 1-3 |
| Buffer/Review | 30-60 min | All phases |

**Recommended Approach**: Execute all phases in one session for consistency.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Tests fail after refactoring | Low | Medium | Comprehensive test suite, fixture for cache clearing |
| Breaking API compatibility | Very Low | High | Function signature unchanged |
| Performance regression | Very Low | Medium | Same caching behavior, verify with benchmarks |
| Thread-safety issues | Very Low | Medium | `lru_cache` is thread-safe by design |
| Merge conflicts | Low | Low | Small, focused change in one file |

**Overall Risk Level**: **LOW** ✅

---

## Post-Implementation Review

After completion, evaluate:

1. **Did we achieve thread-safety?** (Test with concurrent API requests)
2. **Are tests easier to write?** (Compare before/after test complexity)
3. **Is the code more maintainable?** (Developer feedback)
4. **Any unexpected issues?** (Monitor logs, error reports)
5. **Performance impact?** (Benchmark before/after if needed)

Document findings in completion notes when moving to `02-completed/`.

---

## Questions & Answers

**Q: Why not use a dependency injection framework like `dependency-injector`?**
A: Overkill for this use case. The codebase doesn't use DI frameworks, and `@lru_cache` is simpler and sufficient.

**Q: What if we need to reload data without restarting the server?**
A: Call `load_hpo_graph_data.cache_clear()` to force reload on next access. Can expose via admin API endpoint if needed.

**Q: Is `lru_cache` truly thread-safe?**
A: Yes, since Python 3.2. It uses an internal lock (`RLock`) to prevent race conditions.

**Q: What about cache invalidation when HPO data updates?**
A: Current behavior returns empty dicts on error. After data update, call `.cache_clear()` or restart the process.

**Q: Can we cache with different `db_path` parameters?**
A: Yes! `lru_cache` caches based on function arguments. Different `db_path` values will have separate cache entries.

**Q: Why `maxsize=1` instead of larger cache?**
A: Only one HPO database is loaded per application instance. `maxsize=1` is explicit about this behavior.

---

**End of Plan**
