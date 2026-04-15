# Stream C: Infrastructure & Test Quality — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden test infrastructure, add pre-commit hooks, and remove dead code/config issues — establishing the foundation for Streams A and B.

**Architecture:** Independent quick-win commits followed by test infrastructure improvements. Each commit is atomic and passes `make all`. No behavioral changes to application code — only tooling, config, and test quality.

**Tech Stack:** pytest, ruff, mypy, pre-commit, pyproject.toml

**Branch:** `improve/infrastructure`

**Spec:** `docs/superpowers/specs/2026-04-09-code-quality-improvements-design.md` (Stream C)

---

### Task 1: Delete Dead Code

**Files:**
- Delete: `frontend/src/composables/useDisclaimer.js`
- Delete: `tests/unit/cli/test_query_commands.py.disabled`
- Delete: `tests/unit/cli/conftest.py.disabled`

- [ ] **Step 1: Verify useDisclaimer.js is unused**

Run:
```bash
grep -r "useDisclaimer" frontend/src/ --include="*.vue" --include="*.js" --include="*.ts"
```
Expected: Only hits in `useDisclaimer.js` itself and possibly `stores/disclaimer.js` (which is the replacement). No component imports it.

- [ ] **Step 2: Delete the three dead files**

```bash
rm frontend/src/composables/useDisclaimer.js
rm tests/unit/cli/test_query_commands.py.disabled
rm tests/unit/cli/conftest.py.disabled
```

- [ ] **Step 3: Verify build still works**

Run:
```bash
make frontend-build && make test
```
Expected: Both pass with zero errors.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: delete dead code (useDisclaimer.js, disabled test files)

Remove 3 files identified as dead code in code quality review:
- frontend/src/composables/useDisclaimer.js — superseded by Pinia store
- tests/unit/cli/test_query_commands.py.disabled — stale disabled tests
- tests/unit/cli/conftest.py.disabled — unused conftest with mocking strategy"
```

---

### Task 2: Fix Ruff Target Version

**Files:**
- Modify: `pyproject.toml:92`

- [ ] **Step 1: Fix the target version**

In `pyproject.toml`, change line 92:

```toml
# Before:
target-version = "py39"

# After:
target-version = "py310"
```

This aligns Ruff's target with the project's actual minimum Python version (`>=3.10` in `pyproject.toml:6`).

- [ ] **Step 2: Verify no new lint errors**

Run:
```bash
make check
```
Expected: Zero errors. Some UP (pyupgrade) rules may now fire for Python 3.10+ syntax — if so, `make check` auto-fixes them since `fix = true` is set.

- [ ] **Step 3: Run type checker to confirm no regressions**

Run:
```bash
make typecheck-fast
```
Expected: Zero errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml && git commit -m "fix: align ruff target-version with project minimum (py39 -> py310)

Project requires >=3.10 but ruff was targeting py39, potentially missing
Python 3.10+ syntax improvements."
```

---

### Task 3: Add Pre-commit Hooks

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Install pre-commit**

Run:
```bash
uv pip install pre-commit
```

- [ ] **Step 2: Create .pre-commit-config.yaml**

Create `.pre-commit-config.yaml` at the project root:

```yaml
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.4
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
```

**Spec deviation: mypy hook omitted.** The design spec at `docs/superpowers/specs/2026-04-09-code-quality-improvements-design.md:287` included a `mirrors-mypy` hook. After research, this is omitted because:
1. mypy in pre-commit runs from scratch on every commit (no daemon), taking 30-60s on this codebase
2. The project already uses `make typecheck-fast` (mypy daemon, ~3s) which developers run via `make all`
3. The pre-commit docs recommend against slow hooks that discourage frequent commits
4. CI enforces mypy as a required check, so untyped code cannot merge

Ruff in pre-commit provides the fast feedback loop (<1s). mypy stays in `make all` and CI.

- [ ] **Step 3: Install the hooks**

Run:
```bash
pre-commit install
```

- [ ] **Step 4: Run on all files to verify**

Run:
```bash
pre-commit run --all-files
```
Expected: All hooks pass. If trailing-whitespace or end-of-file-fixer makes changes, stage and re-run.

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml && git commit -m "chore: add pre-commit hooks (ruff-check, ruff-format, standard hooks)

Closes the feedback loop on code quality — developers can no longer
commit unformatted or unlinted code. Hooks: trailing-whitespace,
end-of-file-fixer, check-yaml, check-added-large-files,
check-merge-conflict, ruff-check --fix, ruff-format."
```

---

### Import Path Decision (Non-task — explicit constraint for this stream)

**Decision: Keep current PYTHONPATH/Makefile approach.** The `sys.path` hack in `tests/unit/api/test_dependencies_model_loading.py:17-19` and the `PYTHONPATH=$PWD` in `make test-api` stay as-is for this cycle.

**Rationale:** Six approaches to fixing pytest import paths for the `api` module were attempted and all failed (documented in `tests/unit/api/README.md:67-86`). Root cause: pytest assertion rewriting runs before any path configuration is processed. The `src` layout refactoring is the proper fix but is out of scope (see spec Out of Scope section).

**Implication:** All API test run commands in this plan use `PYTHONPATH=$PWD` or `make test` (which sets it via Makefile). New API test files in Stream A must include the same `sys.path` workaround at the top.

---

### Task 4: Centralize Shared Test Fixtures

**Files:**
- Modify: `tests/unit/conftest.py`
- Modify: `tests/unit/api/test_dependencies_model_loading.py`

- [ ] **Step 1: Read current fixtures in both files**

Read `tests/unit/conftest.py` and `tests/unit/api/test_dependencies_model_loading.py` to identify duplicated mock fixtures (e.g., `mock_sbert_model`, `mock_cross_encoder`).

- [ ] **Step 2: Add shared mock fixtures to unit conftest**

Add to `tests/unit/conftest.py`:

```python
from unittest.mock import MagicMock


@pytest.fixture
def mock_sbert_model():
    """Shared mock SentenceTransformer model for unit tests."""
    model = MagicMock()
    model.encode.return_value = [[0.1] * 384]
    model.get_sentence_embedding_dimension.return_value = 384
    return model


@pytest.fixture
def mock_cross_encoder():
    """Shared mock CrossEncoder for unit tests."""
    encoder = MagicMock()
    encoder.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
    return encoder
```

- [ ] **Step 3: Update test_dependencies_model_loading.py to use shared fixtures**

Remove the locally-defined `mock_sbert_model` and `mock_cross_encoder` fixtures from the file. The shared fixtures from `conftest.py` will be picked up automatically by pytest.

- [ ] **Step 4: Verify tests pass**

Run:
```bash
PYTHONPATH=$PWD make test
```
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/conftest.py tests/unit/api/test_dependencies_model_loading.py && git commit -m "refactor: centralize mock_sbert_model and mock_cross_encoder fixtures

Move duplicated mock fixtures to tests/unit/conftest.py so all unit
tests share the same definitions. Reduces maintenance burden."
```

---

### Task 5: Fix Zero-Assertion Tests (Priority Batch)

**Files:**
- Modify: Multiple test files (audit first, then fix)

- [ ] **Step 1: Find all zero-assertion test functions**

Run:
```bash
python3 -c "
import ast, sys, pathlib

test_dir = pathlib.Path('tests')
no_assert = []
for f in sorted(test_dir.rglob('test_*.py')):
    try:
        tree = ast.parse(f.read_text())
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
            has_assert = any(
                isinstance(n, ast.Assert) or
                (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
                 isinstance(n.value.func, ast.Attribute) and
                 n.value.func.attr in ('assert_called', 'assert_called_once', 'assert_called_with',
                                        'assert_called_once_with', 'assert_not_called'))
                for n in ast.walk(node)
            )
            # Check for pytest.raises usage (counts as an assertion)
            has_raises = any(
                isinstance(n, ast.Call) and
                isinstance(getattr(n, 'func', None), ast.Attribute) and
                getattr(n.func, 'attr', '') == 'raises'
                for n in ast.walk(node)
            )
            if not has_assert and not has_raises:
                no_assert.append(f'{f}:{node.lineno}:{node.name}')

for item in no_assert:
    print(item)
print(f'\nTotal: {len(no_assert)} tests with no assertions')
"
```

- [ ] **Step 2: Fix each zero-assertion test**

There are exactly **18 zero-assertion tests** across 6 files. Fix each one:

**File 1: `tests/unit/api/test_text_processing_router.py` (4 tests)**
- `test_valid_references_pass_validation:204` — call the validator, assert it returns without error AND assert on the returned structure
- `test_none_top_evidence_chunk_id_passes_validation:395` — assert the response object has expected fields
- `test_empty_chunks_and_terms_pass_validation:421` — assert the validated response has empty lists
- `test_multiple_text_attributions_all_valid:430` — assert the number of attributions matches input

**File 2: `tests/unit/cli/test_benchmark_integration.py` (2 tests)**
- `test_benchmark_comparison_pipeline:176` — `@pytest.mark.skip` (requires HPO data). Leave as-is — already skipped. Add a comment documenting why.
- `test_benchmark_visualization_pipeline:187` — same, already skipped.

**File 3: `tests/unit/core/test_assertion_detection.py` (1 test)**
- `test_bidirectional_direction:413` — assert on the returned direction enum value

**File 4: `tests/unit/data_processing/test_bundle_packager.py` (1 test)**
- `test_passes_when_checksums_match:156` — assert the function returns True or completes without exception AND assert no side effects

**File 5: `tests/unit/data_processing/test_hpo_database.py` (1 test)**
- `test_close_idempotent:404` — assert the database connection is closed (check attribute or call)

**File 6: `tests/unit/mcp/test_mcp_server.py` (1 test)**
- `test_mcp_check_installed_returns_false_without_package:53` — assert the return value is False

**File 7: `tests/unit/phenopacket_utils/test_phenopacket_utils.py` (8 tests)**
- `test_format_as_phenopacket_v2_empty:8` — assert result is a valid dict with expected keys (id, subject, phenotypicFeatures)
- `test_format_as_phenopacket_v2_empty_both:17` — assert same structure with empty phenotypicFeatures list
- `test_format_as_phenopacket_v2_basic_aggregated:26` — assert phenotypicFeatures length matches input terms
- `test_format_as_phenopacket_v2_sorting:48` — assert phenotypicFeatures are sorted by confidence descending
- `test_format_as_phenopacket_v2_evidence:70` — assert evidence objects have expected structure
- `test_format_as_phenopacket_v2_chunk_results:95` — assert chunk-based results map correctly
- `test_format_as_phenopacket_v2_metadata:160` — assert metadata fields (created, createdBy, resources) are present
- `test_format_as_phenopacket_v2_with_metadata_parameters:182` — assert subject ID and sex values are set from parameters

- [ ] **Step 3: Verify all tests pass**

Run:
```bash
make test
```
Expected: All tests pass.

- [ ] **Step 4: Verify no zero-assertion tests remain**

Re-run the script from Step 1. Expected: `Total: 0 tests with no assertions`.

- [ ] **Step 5: Commit**

```bash
git add tests/ && git commit -m "test: add assertions to all zero-assertion test functions

Audit found 18 tests with no assertions, giving false confidence.
Added meaningful assertions (value checks, match= on pytest.raises)
to every test function."
```

---

### Task 6: Add Parametrize to Repetitive Tests

**Files:**
- Modify: `tests/unit/core/test_assertion_detection.py`
- Modify: `tests/unit/data_processing/test_hpo_parser_edge_cases.py`
- Modify: `tests/unit/api/test_text_processing_router.py`

- [ ] **Step 1: Read all three target files and identify parametrize candidates**

Read each file, find groups of 3+ test functions that share the same body structure but differ only in input values.

Run this to find candidate groups:
```bash
for f in tests/unit/core/test_assertion_detection.py tests/unit/data_processing/test_hpo_parser_edge_cases.py tests/unit/api/test_text_processing_router.py; do
  echo "=== $f ==="
  grep -n "def test_" "$f" | head -30
done
```

- [ ] **Step 2: Refactor test_assertion_detection.py**

Read the file. Look for test functions in the assertion detection test that test the same detection function with different inputs (different languages, different rule types, different assertion categories). Group these into `@pytest.mark.parametrize` calls.

For each group:
1. Identify the common function body
2. Extract the varying parts as parameters
3. Replace N functions with 1 parametrized function
4. Ensure each parametrize case has a descriptive ID using `pytest.param(..., id="...")`

```python
# Example of the parametrize ID pattern to use:
@pytest.mark.parametrize("direction,expected", [
    pytest.param("forward", Direction.FORWARD, id="forward-direction"),
    pytest.param("backward", Direction.BACKWARD, id="backward-direction"),
    pytest.param("bidirectional", Direction.BIDIRECTIONAL, id="bidirectional-direction"),
])
def test_direction_parsing(direction, expected):
    result = parse_direction(direction)
    assert result == expected
```

- [ ] **Step 3: Refactor test_hpo_parser_edge_cases.py and test_text_processing_router.py**

Apply the same parametrize extraction to both files. Focus on groups of validation tests that test the same validator with different inputs (valid vs invalid, different field combinations).

- [ ] **Step 4: Run tests and verify count changed**

```bash
make test 2>&1 | tail -3
```
Expected: All pass. Total test count may increase (parametrized tests expand) but number of test *functions* decreases.

- [ ] **Step 3: Verify all tests pass**

Run:
```bash
make test
```
Expected: All pass. Test count may change (parametrized tests expand to N sub-tests).

- [ ] **Step 4: Commit**

```bash
git add tests/ && git commit -m "refactor: convert repetitive tests to @pytest.mark.parametrize

Converted repetitive test patterns in assertion_detection,
hpo_parser_edge_cases, and text_processing_router to parametrized
tests. Reduces duplication and makes it easy to add new test cases."
```

---

### Task 7: Fix Marker Consistency

**Files:**
- Modify: `tests/unit/data_processing/test_hpo_parser_helpers.py`
- Modify: `tests/unit/data_processing/test_hpo_parser_edge_cases.py`
- Modify: `tests/unit/cli/test_minimal.py`
- Modify: `tests/unit/phenopacket_utils/test_phenopacket_utils.py`

- [ ] **Step 1: Add missing markers**

Add `@pytest.mark.unit` to each file's module level or to each test class/function. Use the `pytestmark` module variable for simplicity:

```python
# Add at top of each file, after imports:
import pytest

pytestmark = pytest.mark.unit
```

- [ ] **Step 2: Verify markers are applied**

Run:
```bash
pytest tests/unit/ -m unit --collect-only 2>&1 | tail -5
```
Expected: All unit tests are collected.

- [ ] **Step 3: Commit**

```bash
git add tests/ && git commit -m "test: add missing @pytest.mark.unit markers to 4 test files

Ensures consistent marker usage across all unit test files for
proper test selection with -m unit."
```

---

### Task 8: Re-enable Coverage Threshold

**Files:**
- Modify: `pyproject.toml:188-194`

- [ ] **Step 1: Set coverage threshold**

Current coverage baseline is **45%** (8507 statements, 4653 missed, measured 2026-04-09). Set the threshold to 40% (5% below actual) to provide buffer for test variance while preventing regression:

In `pyproject.toml`, change line 194:

```toml
# Before:
    # "--cov-fail-under=80",  # TODO: Enable after migration complete

# After:
    "--cov-fail-under=40",  # Baseline: 45% as of 2026-04-09. Ratchet up as coverage improves.
```

- [ ] **Step 3: Verify make test passes with the threshold**

Run:
```bash
make test
```
Expected: Passes. Coverage is above the threshold.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml && git commit -m "test: re-enable coverage threshold at current baseline

Set --cov-fail-under to current baseline. This prevents coverage
regression. Threshold should be ratcheted up as Streams A and B
add characterization tests."
```

---

### Task 9: Run Verification Gates

- [ ] **Step 1: Gate 1 — Lint and type check**

Run:
```bash
make check && make typecheck-fast
```
Expected: Zero errors in both.

- [ ] **Step 2: Gate 2 — All tests pass**

Run:
```bash
make test
```
Expected: All tests pass.

- [ ] **Step 3: Gate 3 — Pre-commit runs clean**

Run:
```bash
pre-commit run --all-files
```
Expected: All hooks pass.

- [ ] **Step 4: Gate 4 — Zero-assertion audit**

Run the zero-assertion detection script from Task 5 Step 1.
Expected: `Total: 0 tests with no assertions`.

- [ ] **Step 5: Gate 5 — Marker audit**

Run:
```bash
# Check that all unit test files have the unit marker
for f in $(find tests/unit -name "test_*.py" -not -name "__*"); do
  if ! grep -q "pytest.mark.unit\|pytestmark.*unit" "$f"; then
    echo "MISSING MARKER: $f"
  fi
done
```
Expected: No output (all files have markers).

- [ ] **Step 6: Gate 6 — Coverage gate enabled**

Run:
```bash
grep "cov-fail-under" pyproject.toml
```
Expected: Line is un-commented with a numeric threshold.
