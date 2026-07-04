# Testing Coverage Expansion Plan (REVISED)

**Status:** Active - Ready for Implementation
**Created:** 2025-11-17
**Revised:** 2025-11-17 (Senior Review - Critical Flaws Fixed)
**Priority:** High
**Related Issue:** [#37: Implement comprehensive testing suite](https://github.com/berntpopp/phentrieve/issues/37)

---

## Executive Summary

Issue #37 requested comprehensive testing. **Infrastructure is complete** (pytest, CI/CD, 263 tests), but **coverage is only 8%**. This revised plan focuses on **quality over quantity**, testing **critical paths to 70-80%** rather than all code to 30%.

### Current State ✅
- ✅ 263 tests (206 unit/integration + 57 E2E)
- ✅ pytest + pytest-cov configured
- ✅ Test structure (unit/, integration/, e2e/)
- ✅ CI/CD integration (GitHub Actions)
- ✅ Make commands (`make test`, `make test-cov`)

### Critical Findings ⚠️
- ❌ **8% coverage** (industry standard: 70-80%)
- ❌ **Zero coverage on critical modules** (API routes, retrieval)
- ❌ **Missing anti-pattern prevention** (risk of low-quality tests)
- ⚠️ **Risk:** Testing easy code, missing critical paths

---

## Objectives (REVISED)

### Primary Goal: Quality Over Quantity

**NOT THIS:**
> ❌ "Increase coverage from 8% to 30%"
> Problem: Could hit 30% by testing trivial getters/setters

**THIS:**
> ✅ **Test all critical paths to 70-80% coverage**
> ✅ **Test high-risk modules to 90%+ coverage**
> ✅ **Let overall coverage be a natural byproduct**

### Coverage Targets by Module Type

| Module Type | Target | Rationale |
|-------------|--------|-----------|
| **Critical** (auth, query, retrieval) | 90%+ | High risk, high use |
| **Important** (utils, processing, CLI) | 70%+ | Widely used, medium risk |
| **Supporting** (config, formatters) | 50%+ | Low complexity, low risk |
| **Low Priority** (visualization, plotting) | 30%+ | Not core functionality |

**Overall target: 60-70% coverage** as natural result of testing critical code.

### Guiding Principles

**Based on 2024 Research:**
> "Projects with over 80% test coverage have 30% lower bug density than those with less than 50%"
>
> "Aspire to 100% and you'll hit 80%; aspire to 80% and you'll hit 40%"
>
> "The 20% left uncovered is probably the 20% that needs testing most"

**Focus:**
1. ✅ Test behavior, not implementation
2. ✅ Critical paths first, percentage second
3. ✅ Edge cases and error handling
4. ✅ Fast, independent, maintainable tests
5. ❌ Don't chase coverage percentage
6. ❌ Don't test trivial code for metrics

---

## Pytest Anti-Patterns (MUST AVOID)

### ❌ Anti-Pattern #1: Testing Private Methods

```python
# DON'T DO THIS
def test_internal_helper():
    obj = MyClass()
    result = obj._private_method()  # WRONG!
    assert result == expected
```

**Why it's wrong:** Makes refactoring impossible. Private methods are tested indirectly through public API.

**Do this instead:**
```python
# CORRECT - test public API
def test_public_method_behavior():
    obj = MyClass()
    result = obj.public_method()  # Tests private methods indirectly
    assert result == expected
```

### ❌ Anti-Pattern #2: Test Interdependencies

```python
# DON'T DO THIS
class TestUserFlow:
    user_id = None  # Shared state!

    def test_create_user(self):
        self.user_id = create_user("John")  # Sets global state

    def test_get_user(self):
        user = get_user(self.user_id)  # Depends on previous test!
```

**Why it's wrong:** Tests break when run in isolation or different order.

**Do this instead:**
```python
# CORRECT - independent tests
def test_create_user():
    user_id = create_user("John")
    assert user_id is not None

def test_get_user():
    user_id = create_user("Test")  # Create own data
    user = get_user(user_id)
    assert user.name == "Test"
```

### ❌ Anti-Pattern #3: File-Based Fixtures

```python
# DON'T DO THIS
def test_api():
    # Dozens of JSON files in tests/fixtures/
    data = json.load(open("tests/fixtures/sample1.json"))
    data2 = json.load(open("tests/fixtures/sample2.json"))
    # Hard to maintain, unclear what data is
```

**Why it's wrong (from research):**
> "Having dozens of JSONs or files stored as data fixtures is wrong - create functions that generate those on demand"

**Do this instead:**
```python
# CORRECT - generate data in code
@pytest.fixture
def sample_hpo_term():
    """Generate test HPO term data."""
    return {
        "id": "HP:0000001",
        "name": "Seizure",
        "definition": "Test definition"
    }

@pytest.fixture
def hpo_term_factory():
    """Factory for custom HPO terms."""
    def _create(**overrides):
        defaults = {"id": "HP:0000001", "name": "Seizure"}
        return {**defaults, **overrides}
    return _create
```

### ❌ Anti-Pattern #4: Assertion Roulette

```python
# DON'T DO THIS
def test_user_creation():
    user = create_user("John", 30, "john@example.com")
    assert user.name == "John"
    assert user.age == 30
    assert user.email == "john@example.com"
    # Which one failed? Message doesn't tell you!
```

**Why it's wrong:** First failure hides subsequent issues. Unclear what's being tested.

**Do this instead:**
```python
# CORRECT - focused tests
def test_user_has_correct_name():
    user = create_user("John")
    assert user.name == "John", "User name should match input"

def test_user_has_correct_age():
    user = create_user("John", age=30)
    assert user.age == 30, "User age should match input"

# OR use parametrize for similar assertions
@pytest.mark.parametrize("field,value", [
    ("name", "John"),
    ("age", 30),
    ("email", "john@example.com"),
])
def test_user_fields(field, value):
    user = create_user("John", 30, "john@example.com")
    assert getattr(user, field) == value
```

### ❌ Anti-Pattern #5: Over-Mocking

```python
# DON'T DO THIS - mocking everything
def test_api_query():
    with patch("chromadb.Client"):
        with patch("sentence_transformers.SentenceTransformer"):
            with patch("phentrieve.retrieval.dense_retriever"):
                # What are we even testing? Our mocks?
                response = api_query("test")
```

**Why it's wrong:** Mocking too much makes test meaningless. Won't catch integration bugs.

**Do this instead:**
```python
# CORRECT - strategic mocking

# UNIT TEST: Mock external dependencies only
def test_retriever_filters_results():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["HP:001", "HP:002"]],
        "distances": [[0.9, 0.1]]
    }

    retriever = DenseRetriever(collection=mock_collection)
    results = retriever.filter_by_threshold(threshold=0.5)
    # Test filtering logic, not ChromaDB
    assert len(results) == 1

# INTEGRATION TEST: Use real lightweight instance
def test_retriever_with_real_db(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("test")
    collection.add(documents=["seizure"], ids=["HP:001"])

    retriever = DenseRetriever(collection=collection)
    results = retriever.query("seizure", top_k=1)
    # Test real integration
    assert results[0]["id"] == "HP:001"
```

---

## Mocking Strategy (REVISED)

### Three Layers, Three Strategies

```
┌─────────────────────────────────────────┐
│ E2E Tests (Docker)                      │ NO MOCKING
│ - Full system with real containers     │ Test: Complete workflows
│ - Real ChromaDB, real models, real API │ Slow: 5-10 min total
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Integration Tests                       │ MINIMAL MOCKING
│ - Real lightweight instances            │ Mock: External APIs only
│ - In-memory ChromaDB, temp files        │ Real: Component interactions
│ - Test component interactions           │ Medium: 30-60 sec total
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Unit Tests                              │ STRATEGIC MOCKING
│ - Isolated business logic               │ Mock: Slow/external deps
│ - Mock ChromaDB, ML models, network     │ Real: Business logic
│ - Test single function/class            │ Fast: <1 sec each
└─────────────────────────────────────────┘
```

### When to Mock (and When NOT To)

**✅ DO Mock:**
- External APIs (network calls)
- ML model inference (slow, non-deterministic)
- Database connections (in unit tests)
- File I/O (in unit tests)
- Time/random (non-deterministic)

**❌ DON'T Mock:**
- Business logic (defeats the purpose!)
- Simple utilities (test real implementation)
- Integration tests (use real instances)
- E2E tests (defeats the purpose!)

### Example: Three-Layer Testing

```python
# UNIT TEST: Mock ChromaDB client
def test_dense_retriever_filters_low_scores():
    """Test filtering logic in isolation."""
    mock_results = [
        {"id": "HP:001", "score": 0.9},
        {"id": "HP:002", "score": 0.3},  # Below threshold
    ]

    retriever = DenseRetriever()
    filtered = retriever.filter_results(mock_results, threshold=0.5)

    assert len(filtered) == 1
    assert filtered[0]["id"] == "HP:001"

# INTEGRATION TEST: Real in-memory ChromaDB
def test_retriever_with_lightweight_db(tmp_path):
    """Test retriever with real ChromaDB instance."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("test")
    collection.add(
        documents=["seizure", "tremor"],
        ids=["HP:001", "HP:002"]
    )

    retriever = DenseRetriever(collection=collection)
    results = retriever.query("epilepsy", top_k=1)

    assert len(results) == 1
    assert results[0]["id"] in ["HP:001", "HP:002"]

# E2E TEST: Full Docker stack (in tests/e2e/)
def test_api_query_end_to_end(api_service):
    """Test complete query workflow via HTTP."""
    response = requests.post(
        f"{api_service}/api/v1/query",
        json={"query_text": "seizures", "top_k": 5}
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
    # Real API, real DB, real models - full integration
```

---

## Strategic Coverage Plan (REVISED)

### Phase 1: Critical Modules (Target: 90%+ coverage)

**Duration:** 3 weeks
**Why First:** High risk, high use, high impact

#### 1.1 API Authentication & Authorization (Priority: CRITICAL)
**Current:** 0%
**Target:** 95%
**Modules:**
- `api/dependencies.py` - Auth dependency injection
- `api/middleware/` - Auth middleware (if exists)

**Why Critical:** Security vulnerability if auth bypassed.

**Test Coverage:**
```python
def test_query_requires_authentication():
    # Test: Unauthenticated request fails

def test_invalid_token_rejected():
    # Test: Malformed/expired tokens rejected

def test_rate_limiting_enforced():
    # Test: Too many requests blocked
```

#### 1.2 Query & Retrieval System (Priority: CRITICAL)
**Current:** dense_retriever.py at 13%, reranker.py at 22%
**Target:** 90%
**Effort:** 1 week

**Modules:**
- `api/routers/query_router.py` (0% → 90%)
- `phentrieve/retrieval/dense_retriever.py` (13% → 90%)
- `phentrieve/retrieval/reranker.py` (22% → 90%)
- `phentrieve/retrieval/query_orchestrator.py` (0% → 70%)

**Why Critical:** Core functionality - if this breaks, product doesn't work.

**Test Strategy:**
```python
# Critical paths to test:
- Valid query returns results
- Invalid query_text raises error
- Empty query returns empty results
- top_k parameter works correctly
- Threshold filtering works
- Re-ranking improves results
- Error handling (timeout, DB unavailable)
- Edge cases (special characters, long text)
```

#### 1.3 Text Processing Pipeline (Priority: HIGH)
**Current:** 10%
**Target:** 80%
**Effort:** 1 week

**Modules:**
- `api/routers/text_processing_router.py` (0% → 80%)
- `phentrieve/text_processing/pipeline.py` (10% → 80%)
- `phentrieve/text_processing/chunkers.py` (11% → 60%)

**Why Critical:** Core text processing - errors affect all results.

**Critical Test Cases:**
```python
- Pipeline processes clinical text correctly
- All 7 chunking strategies work
- Assertion detection works (negation, uncertainty)
- HPO extraction works
- Error handling (malformed input)
- Edge cases (empty text, very long text, special chars)
```

---

### Phase 2: Important Utilities (Target: 70%+ coverage)

**Duration:** 2 weeks
**Why Second:** Widely used across codebase

#### 2.1 Core Utilities (Priority: HIGH)
**Current:** 30%
**Target:** 80%
**Effort:** 1 week

**Modules:**
- `phentrieve/utils.py` (30% → 80%)
- `phentrieve/embeddings.py` (19% → 70%)
- `phentrieve/config.py` (87% → 95%)

**Why Important:** Used by all other modules - bugs cascade.

**Test Strategy:**
```python
@pytest.mark.parametrize("input_id,expected", [
    ("http://purl.obolibrary.org/obo/HP_0000001", "HP:0000001"),
    ("HP:0000001", "HP:0000001"),
    ("HP_0000001", "HP:0000001"),
    ("INVALID", ValueError),  # Test error handling
])
def test_normalize_hpo_id(input_id, expected):
    if expected == ValueError:
        with pytest.raises(ValueError):
            normalize_id(input_id)
    else:
        assert normalize_id(input_id) == expected
```

#### 2.2 Output Formatters (Priority: MEDIUM)
**Current:** 8%
**Target:** 70%
**Effort:** 3 days

**Modules:**
- `phentrieve/retrieval/output_formatters.py` (8% → 70%)

**Test All Formats:**
```python
def test_format_as_text_valid_structure()
def test_format_as_json_valid_structure()
def test_format_as_jsonl_valid_structure()
def test_empty_results_handled_gracefully()
```

---

### Phase 3: CLI & Supporting Code (Target: 50%+ coverage)

**Duration:** 1 week
**Why Third:** Less critical than API, but important for usability

#### 3.1 CLI Commands (Priority: MEDIUM)
**Current:** Most at 0%
**Target:** 60%
**Effort:** 1 week

**Modules:**
- `phentrieve/cli/query_commands.py` (0% → 60%)
- `phentrieve/cli/data_commands.py` (0% → 50%)
- `phentrieve/cli/index_commands.py` (0% → 50%)

**Test Strategy:**
```python
from typer.testing import CliRunner

def test_query_command_basic():
    runner = CliRunner()
    result = runner.invoke(app, ["query", "seizures"])
    assert result.exit_code == 0
    assert "HP:" in result.stdout

def test_query_command_invalid_input():
    runner = CliRunner()
    result = runner.invoke(app, ["query", ""])
    assert result.exit_code != 0
    assert "Error" in result.stdout
```

**Note:** We already have CLI tests (expand them, don't rewrite).

---

### Phase 4: Low Priority Modules (Target: 30%+)

**Defer or minimal coverage:**
- Visualization (`phentrieve/visualization/`) - Not core functionality
- Evaluation (`phentrieve/evaluation/`) - Benchmarking only
- Data processing (`phentrieve/data_processing/`) - Run rarely, complex

**Rationale:** Limited ROI, focus on core product functionality.

---

## Implementation Strategy

### Data-Driven Prioritization

**BEFORE writing ANY tests, analyze:**

```bash
# 1. Find most-changed files (likely most buggy)
git log --since="6 months ago" --format=format: --name-only \
  | grep "\.py$" \
  | sort | uniq -c | sort -rn | head -20

# 2. Find files with most lines (complex = risky)
find phentrieve/ api/ -name "*.py" -exec wc -l {} + \
  | sort -rn | head -20

# 3. Check production errors (if available)
# - Which modules cause 500 errors?
# - Which APIs have highest failure rate?
```

**Use this data to prioritize, not assumptions!**

### Test Development Workflow

```bash
# 1. Identify critical module
git log --oneline phentrieve/retrieval/dense_retriever.py | head

# 2. Check current coverage
pytest tests/unit/retrieval/test_dense_retriever.py \
  --cov=phentrieve/retrieval/dense_retriever.py \
  --cov-report=term-missing

# 3. Write tests for uncovered critical paths
# Focus on: error handling, edge cases, business logic

# 4. Verify coverage increase
pytest tests/unit/retrieval/test_dense_retriever.py \
  --cov=phentrieve/retrieval/dense_retriever.py \
  --cov-report=term-missing

# 5. Run full suite (catch regressions)
make test

# 6. Run linters (catch anti-patterns)
make lint-tests  # Uses flake8-pytest-style

# 7. Commit with meaningful message
git commit -m "test(retrieval): Add edge case tests for dense retriever

- Test empty query handling
- Test invalid threshold values
- Test ChromaDB connection errors
- Coverage: 13% → 45% (critical paths covered)"
```

### Fixture Organization (DRY Principle)

```
tests/
├── conftest.py                    # Shared fixtures (all tests)
│   ├── @pytest.fixture sample_hpo_data
│   ├── @pytest.fixture temp_config_file
│   └── @pytest.fixture hpo_term_factory
│
├── unit/
│   ├── conftest.py                # Unit test fixtures
│   │   ├── @pytest.fixture mock_chroma_client
│   │   └── @pytest.fixture mock_embedding_model
│   │
│   ├── api/
│   │   ├── conftest.py            # API-specific fixtures
│   │   │   ├── @pytest.fixture api_client
│   │   │   └── @pytest.fixture mock_dependencies
│   │   └── test_*.py
│   │
│   └── core/
│       ├── conftest.py            # Core lib fixtures
│       └── test_*.py
│
└── integration/
    ├── conftest.py                # Integration fixtures
    │   ├── @pytest.fixture test_db (real)
    │   └── @pytest.fixture temp_index_dir
    └── test_*.py
```

**Fixture Example (Factory Pattern - DRY):**
```python
# tests/conftest.py
@pytest.fixture
def hpo_term_factory():
    """Factory for creating HPO terms with custom attributes."""
    def _create(**overrides):
        defaults = {
            "id": "HP:0000001",
            "name": "Seizure",
            "definition": "Abnormal electrical discharge",
            "synonyms": []
        }
        return {**defaults, **overrides}
    return _create

# Usage in tests
def test_with_custom_term(hpo_term_factory):
    term = hpo_term_factory(name="Epileptic seizure")
    assert term["name"] == "Epileptic seizure"
    assert term["id"] == "HP:0000001"  # Defaults preserved
```

---

## Quality Gates (MANDATORY)

### Pre-Merge Checklist

**Before merging any test PR, verify:**

#### 1. Test Quality
- [ ] No anti-patterns (private methods, dependencies, file fixtures)
- [ ] Clear, descriptive test names
- [ ] Each test has single purpose (KISS)
- [ ] No magic numbers or strings
- [ ] Proper use of fixtures (DRY)

#### 2. Test Coverage
- [ ] Critical paths covered (>90%)
- [ ] Edge cases tested
- [ ] Error handling tested
- [ ] NOT just hitting coverage percentage

#### 3. Performance
- [ ] Unit tests: <1s each
- [ ] Integration tests: <5s each
- [ ] All tests pass consistently (no flaky tests)

#### 4. Code Quality
- [ ] `make lint-tests` passes (flake8-pytest-style)
- [ ] `make test` passes (all tests)
- [ ] `make typecheck` passes (mypy)
- [ ] No test warnings

#### 5. Documentation
- [ ] Tests have docstrings explaining purpose
- [ ] Complex logic has comments
- [ ] Fixtures are documented

### flake8-pytest-style Integration

**Add to project:**
```bash
# Add linter
uv add --dev flake8-pytest-style

# Add to Makefile
lint-tests:
    @echo "Linting tests for pytest anti-patterns..."
    flake8 tests/ --select=PT --show-source
```

**What it catches:**
- ✅ Incorrect fixture usage
- ✅ Test function naming violations
- ✅ Marker misuse
- ✅ Assert statement issues
- ✅ Parametrize problems

---

## Test Structure & Naming

### Directory Structure (KISS)

```
tests/
├── conftest.py                         # Shared fixtures
├── unit/                               # Fast, isolated (<1s each)
│   ├── api/
│   │   ├── test_query_router.py        # NEW - Critical!
│   │   ├── test_text_processing_router.py  # ✅ Exists (expand)
│   │   └── test_similarity_router.py   # NEW
│   ├── cli/
│   │   ├── test_query_commands.py      # ✅ Exists (expand)
│   │   ├── test_data_commands.py       # NEW
│   │   └── test_index_commands.py      # NEW
│   ├── core/
│   │   ├── test_utils.py               # ✅ Exists (expand)
│   │   ├── test_embeddings.py          # ✅ Exists (expand)
│   │   └── test_config.py              # NEW
│   └── retrieval/
│       ├── test_dense_retriever.py     # ✅ Exists (expand)
│       ├── test_reranker.py            # ✅ Exists (expand)
│       └── test_output_formatters.py   # ✅ Exists
├── integration/                        # Component interaction (<30s total)
│   ├── test_chunking_pipeline_integration.py  # ✅ Exists
│   ├── test_query_workflow.py          # NEW - End-to-end query
│   └── test_index_building.py          # NEW - Index creation
└── e2e/                                # Full Docker stack
    ├── test_api_e2e.py                 # ✅ Exists (expand)
    ├── test_docker_security.py         # ✅ Exists
    └── test_docker_health.py           # ✅ Exists
```

### Naming Conventions (Self-Documenting)

```python
# Pattern: test_<function>_<condition>_<expected_behavior>

# Good examples:
def test_normalize_id_with_uri_returns_hpo_format()
def test_normalize_id_with_invalid_format_raises_error()
def test_query_endpoint_with_empty_text_returns_400()
def test_dense_retriever_with_low_threshold_returns_more_results()
def test_reranker_with_nli_mode_improves_scores()

# Class naming:
class TestDenseRetriever:
    """Tests for dense retriever functionality."""

    def test_query_returns_results(self):
        pass

    def test_filter_by_threshold_removes_low_scores(self):
        pass
```

---

## Timeline & Effort (REALISTIC)

### Phase 1: Critical Modules (90%+ coverage)
**Duration:** 3 weeks
**Effort:** ~120 hours

- **Week 1:** API routes & auth (query, text processing, auth)
- **Week 2:** Core retrieval (dense retriever, reranker, orchestrator)
- **Week 3:** Text processing pipeline (chunkers, assertion detection)

### Phase 2: Important Utilities (70%+ coverage)
**Duration:** 2 weeks
**Effort:** ~80 hours

- **Week 1:** Core utils (normalization, similarity, embeddings)
- **Week 2:** Formatters, config, helpers

### Phase 3: CLI & Supporting (50%+ coverage)
**Duration:** 1 week
**Effort:** ~40 hours

- CLI commands (query, data, index)
- Integration tests
- Documentation

### Phase 4: Review & Cleanup
**Duration:** 1 week
**Effort:** ~40 hours

- Code review feedback
- Test refactoring (DRY improvements)
- Performance optimization
- Documentation updates

### Total Timeline
**7 weeks** for 60-70% quality coverage

**Note:** Can be parallelized with multiple developers.

---

## Success Metrics (REVISED)

### Quantitative (Secondary)
- ✅ Overall coverage: 8% → **60-70%**
- ✅ Critical modules: **90%+**
- ✅ Important modules: **70%+**
- ✅ Test count: 263 → **500-600 tests**
- ✅ Test speed: Unit tests <1 min, integration <2 min

### Qualitative (PRIMARY)
- ✅ **All critical paths tested** (query, retrieval, auth, processing)
- ✅ **All edge cases covered** (empty input, invalid data, errors)
- ✅ **Tests are maintainable** (DRY, clear naming, good fixtures)
- ✅ **Tests are meaningful** (catch real bugs, not implementation details)
- ✅ **No anti-patterns** (passes flake8-pytest-style)
- ✅ **Fast execution** (enable TDD workflow)

### The Real Test
> "Can we confidently refactor code without breaking tests?"
> "Do tests catch real bugs in code review?"
> "Can new developers understand tests easily?"

If YES to all three → Success! ✅

---

## Documentation Updates

### 1. README.md - Add Testing Section

```markdown
## Testing

Phentrieve maintains high test coverage (60-70%) with focus on critical paths.

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/unit/api/test_query_router.py -v

# Run E2E tests (requires Docker)
make test-e2e
```

### Test Structure

- `tests/unit/` - Fast, isolated tests (<1s each, 90%+ critical path coverage)
- `tests/integration/` - Component interaction tests (<30s total)
- `tests/e2e/` - Docker-based end-to-end tests (full system validation)

### Writing Tests

See [CONTRIBUTING.md](CONTRIBUTING.md#testing) for guidelines.

**Key principles:**
- Test behavior, not implementation
- Focus on critical paths and edge cases
- Keep tests fast and independent
- Avoid anti-patterns (see CONTRIBUTING.md)
```

### 2. CONTRIBUTING.md - Testing Guidelines

```markdown
## Testing Guidelines

### Test Philosophy

**Quality > Quantity**
- 90% coverage of critical paths > 100% coverage of trivial code
- Focus on edge cases and error handling
- Tests should catch real bugs, not just increase percentages

### Writing Good Tests

**DO:**
- ✅ Test behavior, not implementation
- ✅ Use descriptive names: `test_function_when_condition_then_expected`
- ✅ Keep tests simple (KISS) - if setup is >10 lines, refactor
- ✅ Make tests independent (no shared state)
- ✅ Use fixtures for setup (DRY)
- ✅ Parametrize for multiple inputs
- ✅ Test edge cases and errors

**DON'T:**
- ❌ Test private methods (test through public API)
- ❌ Create test dependencies (each test must run independently)
- ❌ Use file-based fixtures (generate data in code)
- ❌ Over-mock (mock only slow/external dependencies)
- ❌ Test implementation details (tests should survive refactoring)

### Anti-Patterns to Avoid

See examples in our testing plan: `plan/01-active/TESTING-COVERAGE-EXPANSION-PLAN.md`

### Test Review Checklist

Before submitting test PR:
- [ ] No anti-patterns (pytest-style linter passes)
- [ ] Tests are fast (<1s for unit tests)
- [ ] Tests are independent (can run in any order)
- [ ] Clear, descriptive names
- [ ] Edge cases and errors tested
- [ ] Critical paths covered
```

---

## Acceptance Criteria (Issue #37) - REVISED

| Criterion | Status | Notes |
|-----------|--------|-------|
| ✅ pytest framework | DONE | Configured in pyproject.toml |
| ✅ pytest-cov plugin | DONE | Coverage reporting works |
| ✅ tests/ directory structure | DONE | unit/, integration/, e2e/ |
| ⚠️ Unit tests for utilities | PARTIAL | Need 70%+ coverage (currently 30%) |
| ✅ Integration tests | DONE | Good coverage |
| ⚠️ CLI tests with CliRunner | PARTIAL | Expand from 0% to 60% |
| ❌ Coverage >30% baseline | **8%** | **Expand to 60-70% (revised target)** |
| ✅ Simple `pytest` command | DONE | Works via Make commands |
| ✅ CI/CD integration | DONE | GitHub Actions configured |
| ⚠️ Documentation | PARTIAL | Add to README/CONTRIBUTING |

**REVISED Target:** 60-70% overall, 90%+ critical paths

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Writing low-value tests for coverage** | High | High | Focus on critical paths first, use quality gates |
| **Over-mocking creates false confidence** | Medium | High | Clarified mocking strategy, integration tests with real instances |
| **Tests slow down development** | Medium | High | Performance requirements (<1s unit, <30s integration) |
| **Brittle tests break on refactoring** | Medium | Medium | Test behavior not implementation, code reviews |
| **Test interdependencies** | Low | Medium | Enforce independence in quality gates |
| **Missing anti-patterns** | Medium | Medium | Add flake8-pytest-style linter |
| **Timeline pressure → cutting corners** | Medium | High | Realistic 7-week timeline, no shortcuts on quality |

---

## Next Steps

1. **Review & Approve** - Stakeholder sign-off on revised plan
2. **Add Tooling** - Install flake8-pytest-style linter
3. **Data Analysis** - Run git churn analysis to validate priorities
4. **Create Branch** - `feature/testing-coverage-expansion`
5. **Phase 1 Week 1** - Start with API query router (critical path)
6. **Incremental PRs** - Small, focused PRs per module
7. **Track Progress** - Update plan with completion status
8. **Close Issue #37** - When 60%+ coverage achieved with quality! 🎉

---

## References

### Research & Best Practices
- [pytest Best Practices (2024)](https://pytest-with-eric.com/pytest-best-practices/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Test Coverage Best Practices](https://medium.com/@keployio/mastering-python-test-coverage-tools-tips-and-best-practices-11daf699d79b)
- [Why 100% Coverage Isn't the Goal](https://blog.ndepend.com/aim-100-percent-test-coverage/)
- [80% Coverage Standard](https://stackoverflow.com/questions/90002/what-is-a-reasonable-code-coverage-for-unit-tests-and-why)

### Internal Docs
- `CLAUDE.md` - Pre-commit checklist (make check, make typecheck-fast, make test)
- `plan/02-completed/TESTING-MODERNIZATION-PLAN.md` - Previous testing work
- Issue #37 - Original testing requirements

### Key Quotes
> "Projects with over 80% test coverage have 30% lower bug density"
>
> "Aspire to 100% and you'll hit 80%; aspire to 80% and you'll hit 40%"
>
> "The 20% left uncovered is probably the 20% that needs testing most"

---

**Status:** Ready for Implementation (REVISED)
**Next Action:** Review revised plan, then start Phase 1 Week 1 (API query router)
**Target:** Close issue #37 with 60-70% quality coverage, 90%+ critical paths
**Approach:** Quality over quantity, critical paths first, no anti-patterns
