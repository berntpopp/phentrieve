# Benchmark Data Reorganization - Simplified Plan

**Status:** 🔄 Active (Refactored)
**Created:** 2025-11-18 (Refactored after expert review)
**Priority:** High
**Effort:** 1 day (6-8 hours)
**Complexity:** Low (KISS compliant)

---

## Problem Statement

**Current State:**
- Benchmark test data lives in `data/test_cases/` (outside test structure)
- Recent refactoring of test data to `tests/data/` didn't include benchmarks
- No regression tests for benchmark/visualization commands after refactoring
- Inconsistent with new annotation data organization (`tests/data/en/`, `tests/data/de/`)

**Core Issue:** Benchmark data is isolated from test suite and lacks regression testing.

**Solution:** Move to `tests/data/benchmarks/` and add integration tests. Simple, atomic, done.

---

## Design Principles

This plan follows:
- ✅ **KISS** - Keep It Simple, Stupid
- ✅ **DRY** - Don't Repeat Yourself
- ✅ **YAGNI** - You Aren't Gonna Need It
- ✅ **SOLID** - Single Responsibility, focused modules
- ✅ **Atomic Changes** - One clean commit, no gradual migration
- ✅ **Zero Tech Debt** - No compatibility layers or deprecated code

---

## Target Structure

```
tests/data/
├── benchmarks/
│   ├── README.md                    # Dataset documentation
│   └── multilingual/
│       ├── tiny_v1.json             # 9 cases - quick testing
│       ├── small_v1.json            # 9 cases - small eval
│       ├── 70cases_gemini_v1.json   # 70 cases - Gemini translated
│       ├── 70cases_o3_v1.json       # 70 cases - O3 translated
│       ├── 200cases_gemini_v1.json  # 200 cases - large Gemini
│       └── 200cases_o3_v1.json      # 200 cases - large O3
├── de/phentrieve/                   # Existing: German annotations
└── en/phenobert/                    # Existing: English annotations

data/
├── results/                         # Benchmark outputs (unchanged)
└── test_cases/                      # DELETED after migration
```

**Naming Convention:** `{size}_{variant}_v{version}.json`
- Simpler than original (directory provides language context)
- Example: `70cases_gemini_v1.json` not `benchmark_ml_70cases_gemini_v1.json`

---

## Implementation

### Phase 1: Reorganize Data (2-3 hours)

#### 1.1 Create Directory Structure
```bash
mkdir -p tests/data/benchmarks/multilingual
```

#### 1.2 Move Files with Simplified Names
```bash
cd data/test_cases

# Copy with new names (cp not mv - safety first)
cp sample_test_cases.json ../../tests/data/benchmarks/multilingual/tiny_v1.json
cp test_cases_small.json ../../tests/data/benchmarks/multilingual/small_v1.json
cp expanded_test_70cases_gemini25translated.json ../../tests/data/benchmarks/multilingual/70cases_gemini_v1.json
cp expanded_test_70cases_o3translated.json ../../tests/data/benchmarks/multilingual/70cases_o3_v1.json
cp expanded_test_200cases_gemini25translated.json ../../tests/data/benchmarks/multilingual/200cases_gemini_v1.json
cp expanded_test_200cases_o3translated.json ../../tests/data/benchmarks/multilingual/200cases_o3_v1.json
```

**Verify integrity:**
```bash
# Compare file sizes
ls -lh data/test_cases/*.json
ls -lh tests/data/benchmarks/multilingual/*.json
```

#### 1.3 Create Documentation
Create `tests/data/benchmarks/README.md`:

```markdown
# Benchmark Evaluation Datasets

Curated datasets for evaluating HPO retrieval performance.

## Structure

- `multilingual/` - Mixed-language datasets (German, English, etc.)
- `de/` - German-only datasets (future)
- `en/` - English-only datasets (future)

## Datasets

| File | Cases | Variant | Description |
|------|-------|---------|-------------|
| tiny_v1.json | 9 | Manual | Quick testing, default dataset |
| small_v1.json | 9 | Manual | Small evaluation set |
| 70cases_gemini_v1.json | 70 | AI-translated | Gemini 2.5 translated |
| 70cases_o3_v1.json | 70 | AI-translated | O3 translated |
| 200cases_gemini_v1.json | 200 | AI-translated | Large Gemini set |
| 200cases_o3_v1.json | 200 | AI-translated | Large O3 set |

## Format

```json
[
  {
    "description": "Optional human-readable description",
    "text": "Clinical text in target language",
    "expected_hpo_ids": ["HP:0001234", "HP:0005678"]
  }
]
```

## Usage

### CLI
```bash
# Default (uses tiny_v1.json)
phentrieve benchmark run

# Specify dataset
phentrieve benchmark run --test-file multilingual/70cases_gemini_v1.json
```

### Python
```python
from phentrieve.data_processing.test_data_loader import load_test_data

# Relative path from project root
dataset = load_test_data("tests/data/benchmarks/multilingual/tiny_v1.json")
```

### Tests
```python
def test_something(benchmark_data_dir):
    dataset = load_test_data(str(benchmark_data_dir / "multilingual/tiny_v1.json"))
    assert len(dataset) == 9
```

## Adding Datasets

1. Add JSON file to appropriate directory
2. Update table above
3. Verify with: `pytest tests/unit/cli/test_benchmark_commands.py::TestBenchmarkDataLoading`

## Version History

- **v1** (2025-11-18): Initial migration from `data/test_cases/`
```

#### 1.4 Update Configuration
Edit `phentrieve/config.py`:

```python
# Line 18: Update or add these constants
BENCHMARK_DATA_DIR = Path("tests/data/benchmarks")
DEFAULT_BENCHMARK_FILE = "multilingual/tiny_v1.json"

# DELETE these if they exist:
# DEFAULT_TEST_CASES_SUBDIR = "test_cases"  # REMOVE
```

#### 1.5 Update Benchmark Orchestrator
Edit `phentrieve/evaluation/benchmark_orchestrator.py`:

Find line ~118-122 (the default test file logic):
```python
# OLD CODE:
if not test_file:
    data_dir = get_default_data_dir()
    test_cases_dir = data_dir / DEFAULT_TEST_CASES_SUBDIR
    test_file = str(test_cases_dir / "sample_test_cases.json")

# NEW CODE:
if not test_file:
    from phentrieve.config import BENCHMARK_DATA_DIR, DEFAULT_BENCHMARK_FILE
    project_root = Path(__file__).parent.parent.parent
    test_file = str(project_root / BENCHMARK_DATA_DIR / DEFAULT_BENCHMARK_FILE)
```

#### 1.6 Update Sample Data Creator
Edit `phentrieve/data_processing/test_data_loader.py`:

Update `create_sample_test_data()` function (around line 60-68):
```python
def create_sample_test_data(output_file: Optional[str] = None) -> list[dict[str, Any]]:
    """Create a sample test dataset if none exists."""
    if output_file is None:
        from phentrieve.config import BENCHMARK_DATA_DIR, DEFAULT_BENCHMARK_FILE
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / BENCHMARK_DATA_DIR / DEFAULT_BENCHMARK_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = str(output_path)

    # ... rest remains the same
```

#### 1.7 Search and Replace
```bash
# Search for any hardcoded references
grep -r "data/test_cases" phentrieve/ tests/ --exclude-dir=__pycache__

# Update any found references manually
```

#### 1.8 Test Changes
```bash
# Run existing tests
make test

# Test CLI command
phentrieve benchmark run --cpu --debug

# Should use new default path
```

#### 1.9 Clean Up
```bash
# Delete old directory
rm -rf data/test_cases

# Update .gitignore if needed
```

**Acceptance Criteria:**
- ✅ All 6 files in new location with simpler names
- ✅ Config updated (2 constants only)
- ✅ Orchestrator uses new default path
- ✅ CLI works without arguments
- ✅ All existing tests pass
- ✅ Old directory deleted

---

### Phase 2: Add Integration Tests (3-4 hours)

#### 2.1 Add Conftest Helper
Edit `tests/conftest.py`, add:

```python
from pathlib import Path

@pytest.fixture
def benchmark_data_dir():
    """Return path to benchmark data directory."""
    return Path(__file__).parent / "data" / "benchmarks"
```

That's it. No fixture proliferation. Tests can use this + `load_test_data()` directly.

#### 2.2 Add Integration Tests
Edit `tests/unit/cli/test_benchmark_commands.py`, add new test class at end:

```python
# Add at end of file, after existing tests

# =============================================================================
# Integration Tests (Using Real Benchmark Data)
# =============================================================================


class TestBenchmarkDataLoading:
    """Integration tests for benchmark data loading.

    These tests verify benchmark datasets can be loaded and have valid structure.
    Uses actual files from tests/data/benchmarks/ (not mocked).
    """

    @pytest.mark.integration
    def test_all_datasets_loadable(self, benchmark_data_dir):
        """Verify all benchmark datasets can be loaded."""
        from phentrieve.data_processing.test_data_loader import load_test_data

        benchmark_files = list(benchmark_data_dir.rglob("*.json"))
        benchmark_files = [f for f in benchmark_files if f.name != "datasets.json"]

        assert len(benchmark_files) == 6, f"Expected 6 datasets, found {len(benchmark_files)}"

        for dataset_file in benchmark_files:
            dataset = load_test_data(str(dataset_file))

            # Basic validation
            assert dataset is not None, f"Failed to load {dataset_file.name}"
            assert len(dataset) > 0, f"Empty dataset: {dataset_file.name}"
            assert isinstance(dataset, list), f"Dataset not a list: {dataset_file.name}"

    @pytest.mark.integration
    def test_dataset_structure_valid(self, benchmark_data_dir):
        """Verify all datasets have required fields and correct structure."""
        from phentrieve.data_processing.test_data_loader import load_test_data
        import re

        hpo_pattern = re.compile(r"^HP:\d{7}$")

        for dataset_file in benchmark_data_dir.rglob("*.json"):
            if dataset_file.name == "datasets.json":
                continue

            dataset = load_test_data(str(dataset_file))

            for i, case in enumerate(dataset):
                # Required fields
                assert "text" in case, f"{dataset_file.name} case {i}: missing 'text'"
                assert "expected_hpo_ids" in case, f"{dataset_file.name} case {i}: missing 'expected_hpo_ids'"

                # Field types
                assert isinstance(case["text"], str), f"{dataset_file.name} case {i}: text not string"
                assert isinstance(case["expected_hpo_ids"], list), f"{dataset_file.name} case {i}: expected_hpo_ids not list"

                # Field content
                assert len(case["text"]) > 0, f"{dataset_file.name} case {i}: empty text"
                assert len(case["expected_hpo_ids"]) > 0, f"{dataset_file.name} case {i}: no expected HPO IDs"

                # HPO ID format
                for hpo_id in case["expected_hpo_ids"]:
                    assert hpo_pattern.match(hpo_id), \
                        f"{dataset_file.name} case {i}: invalid HPO ID format: {hpo_id}"

    @pytest.mark.integration
    def test_expected_case_counts(self, benchmark_data_dir):
        """Verify datasets have expected number of cases."""
        from phentrieve.data_processing.test_data_loader import load_test_data

        expected_counts = {
            "tiny_v1.json": 9,
            "small_v1.json": 9,
            "70cases_gemini_v1.json": 70,
            "70cases_o3_v1.json": 70,
            "200cases_gemini_v1.json": 200,
            "200cases_o3_v1.json": 200,
        }

        for filename, expected_count in expected_counts.items():
            dataset_path = benchmark_data_dir / "multilingual" / filename
            dataset = load_test_data(str(dataset_path))

            actual_count = len(dataset) if dataset else 0
            assert actual_count == expected_count, \
                f"{filename}: expected {expected_count} cases, got {actual_count}"

    @pytest.mark.integration
    def test_default_dataset_loads(self):
        """Verify default dataset path works."""
        from phentrieve.config import BENCHMARK_DATA_DIR, DEFAULT_BENCHMARK_FILE
        from phentrieve.data_processing.test_data_loader import load_test_data
        from pathlib import Path

        # Construct path like the orchestrator does
        project_root = Path(__file__).parent.parent.parent
        test_file = str(project_root / BENCHMARK_DATA_DIR / DEFAULT_BENCHMARK_FILE)

        dataset = load_test_data(test_file)
        assert dataset is not None
        assert len(dataset) == 9  # tiny_v1.json has 9 cases


class TestBenchmarkCommandsIntegration:
    """Integration tests for benchmark CLI commands with real data.

    These tests verify the benchmark pipeline works end-to-end.
    Marked as slow - skip in fast test runs with: pytest -m "not slow"
    """

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires HPO data prepared locally")
    def test_benchmark_run_tiny_dataset(self, benchmark_data_dir, tmp_path):
        """Test running benchmark with tiny dataset (end-to-end).

        NOTE: This test requires:
        - HPO data prepared: phentrieve data prepare
        - Vector index built: phentrieve index build

        Unskip and run manually when HPO data available.
        """
        from phentrieve.evaluation.benchmark_orchestrator import orchestrate_benchmark

        test_file = str(benchmark_data_dir / "multilingual" / "tiny_v1.json")

        results = orchestrate_benchmark(
            test_file=test_file,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold=0.1,
            cpu=True,
            debug=True,
            results_dir_override=str(tmp_path),
        )

        # Verify results structure
        assert results is not None
        assert isinstance(results, dict)

        # Check expected metrics exist
        expected_metrics = ["mrr", "hit_rate_at_1", "hit_rate_at_5", "avg_semantic_similarity"]
        for metric in expected_metrics:
            assert metric in results, f"Missing metric: {metric}"

        # Sanity check metric ranges
        assert 0 <= results["mrr"] <= 1, "MRR out of range"
        assert 0 <= results["hit_rate_at_1"] <= 1, "Hit rate out of range"

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires HPO data - run manually")
    def test_benchmark_comparison_pipeline(self, benchmark_data_dir, tmp_path):
        """Test benchmark comparison with multiple results (end-to-end).

        NOTE: Requires HPO data. Unskip for manual testing.
        """
        # This would test the full comparison pipeline
        # Implement when needed for regression testing
        pass

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires HPO data - run manually")
    def test_benchmark_visualization_pipeline(self, benchmark_data_dir, tmp_path):
        """Test benchmark visualization generation (end-to-end).

        NOTE: Requires HPO data. Unskip for manual testing.
        """
        # This would test visualization generation
        # Implement when needed for regression testing
        pass
```

#### 2.3 Run Tests
```bash
# Run new integration tests
pytest tests/unit/cli/test_benchmark_commands.py::TestBenchmarkDataLoading -v

# Run all tests
make test

# Check for any failures
```

**Acceptance Criteria:**
- ✅ Simple fixture in conftest.py (just path, no logic)
- ✅ All 6 datasets load successfully
- ✅ Datasets have valid structure
- ✅ Case counts match expected values
- ✅ Default dataset works
- ✅ Framework for end-to-end tests (skipped until HPO data ready)
- ✅ All tests pass

---

### Phase 3: Documentation (1 hour)

#### 3.1 Update CLAUDE.md
Edit `CLAUDE.md`, find "Data Directory Structure" section and add:

```markdown
### Benchmark Test Data

**Location**: `tests/data/benchmarks/`

Benchmark datasets for evaluating HPO retrieval performance:
- `multilingual/` - Mixed-language datasets (default location)
- `de/` - German-only datasets (future)
- `en/` - English-only datasets (future)

**Available Datasets**:
- `tiny_v1.json` (9 cases) - Quick testing, used by default
- `70cases_gemini_v1.json` (70 cases) - Medium evaluation
- `200cases_gemini_v1.json` (200 cases) - Comprehensive evaluation

**Usage**:
```bash
# Use default tiny dataset
phentrieve benchmark run

# Specify dataset by relative path
phentrieve benchmark run --test-file multilingual/70cases_gemini_v1.json

# Or absolute path
phentrieve benchmark run --test-file tests/data/benchmarks/multilingual/tiny_v1.json
```

**In Tests**:
```python
def test_something(benchmark_data_dir):
    from phentrieve.data_processing.test_data_loader import load_test_data
    dataset = load_test_data(str(benchmark_data_dir / "multilingual/tiny_v1.json"))
    assert len(dataset) == 9
```

See `tests/data/benchmarks/README.md` for complete dataset documentation.
```

#### 3.2 Update plan/STATUS.md
Edit `plan/STATUS.md`, add to recent achievements:

```markdown
### Benchmark Data Reorganization (2025-11-18)
- ✅ Migrated benchmark data from `data/test_cases/` to `tests/data/benchmarks/`
- ✅ Simplified naming convention (directory provides context)
- ✅ Added integration tests verifying dataset loading (6 new tests)
- ✅ Updated configuration and documentation
- ✅ Zero tech debt - clean atomic refactor
```

#### 3.3 Verify Documentation
```bash
# Check all docs are consistent
grep -r "data/test_cases" . --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=node_modules

# Should find nothing (or only in this plan document)
```

**Acceptance Criteria:**
- ✅ CLAUDE.md has benchmark data section
- ✅ STATUS.md updated with achievement
- ✅ No references to old path remain
- ✅ benchmarks/README.md comprehensive

---

## Validation Checklist

Before committing, verify:

### Functionality
- [ ] `phentrieve benchmark run` works (uses default tiny_v1.json)
- [ ] `phentrieve benchmark run --test-file multilingual/70cases_gemini_v1.json` works
- [ ] All existing tests pass: `make test`
- [ ] New integration tests pass: `pytest tests/unit/cli/test_benchmark_commands.py::TestBenchmarkDataLoading -v`

### Code Quality
- [ ] No linting errors: `make lint`
- [ ] No formatting issues: `make check`
- [ ] No type errors: `make typecheck-fast`

### Data Integrity
- [ ] All 6 files in new location
- [ ] File sizes match originals
- [ ] JSON files are valid (can be loaded)
- [ ] Old directory deleted

### Configuration
- [ ] Only 2 config constants (BENCHMARK_DATA_DIR, DEFAULT_BENCHMARK_FILE)
- [ ] No deprecated constants remain
- [ ] No compatibility flags

### Documentation
- [ ] CLAUDE.md updated
- [ ] STATUS.md updated
- [ ] benchmarks/README.md exists and complete
- [ ] No references to `data/test_cases` in docs

---

## What This Plan Does NOT Include

Following YAGNI principle, we explicitly exclude:

### ❌ NOT Included (Unnecessary Complexity)
1. **Backward compatibility layer** - Atomic change, no gradual migration
2. **Legacy path support** - Old directory deleted immediately
3. **datasets.json metadata** - README.md sufficient
4. **6+ specialized fixtures** - One simple path fixture is enough
5. **Migration guide** - Internal change, CLAUDE.md update sufficient
6. **Deprecation warnings** - Not a public API
7. **Complex validation** - Basic checks in tests sufficient
8. **Multiple test files** - Add to existing file

### ✅ Can Add Later If Needed
- Pydantic models for validation (when actually needed)
- More fixtures (when pattern emerges)
- Language-specific directories (de/, en/) (when we have datasets)
- Generated metadata (when automation required)
- Additional integration tests (when HPO data ready)

---

## Timeline

**Total: 1 Day (6-8 hours)**

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Reorganize Data | 2-3 hours | None |
| 2. Integration Tests | 3-4 hours | Phase 1 complete |
| 3. Documentation | 1 hour | Phase 1-2 complete |

**Slack**: ~1 hour for unexpected issues

---

## Risk Assessment

### Low Risk ✅
- **Small change scope**: Moving files and updating paths
- **Atomic commit**: All changes tested together
- **No external impact**: Internal file organization only
- **Reversible**: Git revert if needed

### Mitigations
- Copy files first (cp not mv) - verify before deleting originals
- Run tests after each phase
- Single PR with all changes - reviewed as unit
- Clear rollback: `git revert <commit-hash>`

---

## Success Criteria

### Must Have ✅
1. All benchmark data in `tests/data/benchmarks/`
2. CLI works with default and specified datasets
3. All existing tests pass (no regressions)
4. New integration tests pass (6 tests minimum)
5. Old `data/test_cases/` deleted
6. Documentation updated (CLAUDE.md, STATUS.md, README.md)
7. Zero linting/type errors
8. One clean atomic commit

### Quality Metrics
- **Complexity**: Low (no compatibility layers)
- **Tech Debt**: Zero (no deprecated code)
- **Test Coverage**: +6 integration tests
- **Timeline**: ≤ 1 day actual effort

---

## Comparison: Old Plan vs. Refactored Plan

| Aspect | Original Plan | Refactored Plan |
|--------|---------------|-----------------|
| **Timeline** | 3-4 days | 1 day |
| **Phases** | 6 phases | 3 phases |
| **Compatibility Layer** | Complex fallback logic | None - atomic change |
| **Legacy Support** | Yes with warnings | No - clean break |
| **Config Constants** | 4+ with flags | 2 simple constants |
| **Fixtures** | 6+ dataset-specific | 1 path helper |
| **Test Files** | 3 new files | Add to existing |
| **Migration Guide** | Full document | Paragraph in CLAUDE.md |
| **Metadata Files** | datasets.json | README.md only |
| **Naming** | 5 parts | 2-3 parts |
| **Tech Debt** | High | Zero |
| **Complexity** | High | Low |
| **Lines Changed** | ~500+ | ~150 |

---

## Implementation Order

### Commit 1: Reorganize Data & Update Code
- Phase 1 complete (all 1.1-1.9 steps)
- Ensure all tests still pass
- Single commit message:
  ```
  refactor: Migrate benchmark data to tests/data/benchmarks/

  - Move 6 benchmark datasets from data/test_cases/ to tests/data/benchmarks/multilingual/
  - Simplify naming convention (directory provides language context)
  - Update config.py to use new paths
  - Update benchmark_orchestrator.py default path
  - Update test_data_loader.py sample creation
  - Delete old data/test_cases/ directory
  - Add tests/data/benchmarks/README.md

  Breaking changes: None (CLI behavior unchanged)
  ```

### Commit 2: Add Integration Tests
- Phase 2 complete
- Commit message:
  ```
  test: Add integration tests for benchmark data loading

  - Add TestBenchmarkDataLoading class (6 new tests)
  - Add TestBenchmarkCommandsIntegration class (framework for E2E tests)
  - Add benchmark_data_dir fixture to conftest.py
  - Verify all datasets loadable and valid structure

  Tests: +6 integration tests
  ```

### Commit 3: Documentation
- Phase 3 complete
- Commit message:
  ```
  docs: Update documentation for benchmark data reorganization

  - Update CLAUDE.md with new benchmark data location
  - Update plan/STATUS.md with achievement
  - Verify no stale references to old path
  ```

**Or**: Squash all 3 into single atomic commit (preferred for clean history).

---

## Rollback Plan

If issues discovered after merge:

### Immediate Rollback (< 5 minutes)
```bash
# Revert the commit(s)
git revert <commit-hash>
git push
```

### Partial Rollback (Keep tests, revert data move)
```bash
# Manual fix:
# 1. git revert <commit-hash>
# 2. Restore data/test_cases/ from git history
# 3. Keep new integration tests but mark as skip
# 4. Update config to use old path temporarily
```

### Why Rollback Is Easy
- Single commit or small commit series
- No gradual migration complexity
- No compatibility layers to untangle
- Simple git revert restores everything

---

## Principles Applied

### KISS (Keep It Simple, Stupid) ✅
- No compatibility layers
- No complex path resolution
- Minimal configuration
- Simple naming convention
- Atomic refactor

### DRY (Don't Repeat Yourself) ✅
- One fixture (path only), not 6+
- Reuse existing `load_test_data()` function
- README.md is single source of truth

### YAGNI (You Aren't Gonna Need It) ✅
- No datasets.json
- No migration guide
- No deprecation period
- No premature abstractions

### SOLID ✅
- **Single Responsibility**: Each module has one job
- **Open/Closed**: Can add datasets without changing code
- **Dependency Inversion**: Depend on paths (config) not hardcoded values

---

## FAQ

**Q: Why no backward compatibility for old path?**
A: This is internal file organization, not a public API. No external users depend on these paths. Atomic change is simpler and cleaner.

**Q: What if we have scripts using old paths?**
A: Update them in the same commit. Should be <5 references total. Atomic change prevents partial migration issues.

**Q: Why delete old directory immediately?**
A: Two sources of truth create confusion. Clean break prevents tech debt accumulation.

**Q: Why not use Pydantic for validation?**
A: YAGNI - basic checks in tests are sufficient. Add Pydantic later if complex validation needed.

**Q: Why one fixture instead of six?**
A: DRY - all fixtures would do the same thing. Factory pattern or direct function usage is simpler.

**Q: Why 1 day instead of 3-4 days?**
A: This is fundamentally: move 6 files + update 3-4 code locations + add tests. Simple refactor, not complex migration.

**Q: Can we add more datasets later?**
A: Yes! Just add JSON to appropriate directory and update README. Structure supports growth without code changes.

---

## Appendix: File Mapping

| Old Path | New Path | Size | Cases |
|----------|----------|------|-------|
| data/test_cases/sample_test_cases.json | tests/data/benchmarks/multilingual/tiny_v1.json | 1.8K | 9 |
| data/test_cases/test_cases_small.json | tests/data/benchmarks/multilingual/small_v1.json | 1.8K | 9 |
| data/test_cases/expanded_test_70cases_gemini25translated.json | tests/data/benchmarks/multilingual/70cases_gemini_v1.json | 12K | 70 |
| data/test_cases/expanded_test_70cases_o3translated.json | tests/data/benchmarks/multilingual/70cases_o3_v1.json | 12K | 70 |
| data/test_cases/expanded_test_200cases_gemini25translated.json | tests/data/benchmarks/multilingual/200cases_gemini_v1.json | 32K | 200 |
| data/test_cases/expanded_test_200cases_o3translated.json | tests/data/benchmarks/multilingual/200cases_o3_v1.json | 33K | 200 |

---

## Ready to Implement?

This plan is:
- ✅ Simple and focused
- ✅ KISS/DRY/YAGNI compliant
- ✅ Realistic timeline (1 day)
- ✅ Zero tech debt
- ✅ Low risk
- ✅ Clear acceptance criteria

**Next step**: Review and approve, then implement Phase 1 → 2 → 3.

Or: Ask clarifying questions if anything unclear.

---

**End of Simplified Plan**
