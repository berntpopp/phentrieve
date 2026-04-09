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

Note: mypy hook is omitted intentionally — it's too slow for pre-commit and the project uses `make typecheck-fast` (mypy daemon) instead. Ruff handles the fast feedback loop; mypy runs via `make all` before commits.

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

For each test found, add a meaningful assertion. Common patterns:

**Pattern A — test that calls a function but doesn't assert:**
```python
# Before:
def test_something():
    result = my_function(input)

# After:
def test_something():
    result = my_function(input)
    assert result is not None
    assert isinstance(result, ExpectedType)
    assert result.key == expected_value
```

**Pattern B — test using pytest.raises without match:**
```python
# Before:
def test_invalid_input():
    with pytest.raises(ValueError):
        my_function(bad_input)

# After:
def test_invalid_input():
    with pytest.raises(ValueError, match="expected error substring"):
        my_function(bad_input)
```

**Pattern C — test that's truly empty/placeholder:**
```python
# Delete the test function entirely if it provides no value
```

Work through the priority files first:
1. `tests/unit/api/test_dependencies_model_loading.py`
2. `tests/unit/api/test_text_processing_router_performance.py`
3. `tests/unit/cli/test_benchmark_integration.py`

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

Audit found N tests with no assertions, giving false confidence.
Added meaningful assertions (value checks, match= on pytest.raises)
to every test function."
```

---

### Task 6: Add Parametrize to Repetitive Tests

**Files:**
- Modify: `tests/unit/core/test_assertion_detection.py`
- Modify: `tests/unit/data_processing/test_hpo_parser_edge_cases.py`
- Modify: `tests/unit/api/test_text_processing_router.py`

- [ ] **Step 1: Read test_assertion_detection.py and identify repetitive patterns**

Look for test functions that follow the same structure but differ only in input values (rule type, language, expected output). These are parametrize candidates.

- [ ] **Step 2: Refactor repetitive assertion detection tests**

Example transformation:

```python
# Before: N separate test functions with same structure
def test_negation_rule_en():
    result = detect_assertion("no fever", "en")
    assert result.status == "negated"

def test_negation_rule_de():
    result = detect_assertion("kein Fieber", "de")
    assert result.status == "negated"

# After: Single parametrized test
@pytest.mark.parametrize("text,lang,expected_status", [
    ("no fever", "en", "negated"),
    ("kein Fieber", "de", "negated"),
    # ... more cases
])
def test_negation_detection(text, lang, expected_status):
    result = detect_assertion(text, lang)
    assert result.status == expected_status
```

Apply the same pattern to `test_hpo_parser_edge_cases.py` and `test_text_processing_router.py`.

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

- [ ] **Step 1: Determine current coverage baseline**

Run:
```bash
make test-cov 2>&1 | grep "TOTAL"
```
Note the current total coverage percentage.

- [ ] **Step 2: Set coverage threshold at baseline**

In `pyproject.toml`, un-comment and set the threshold to 5% below current actual (giving buffer for test variance):

```toml
# Before:
    # "--cov-fail-under=80",  # TODO: Enable after migration complete

# After (example, adjust number to actual baseline):
    "--cov-fail-under=40",  # Ratchet up as coverage improves
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
