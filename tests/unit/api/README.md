# API Unit Tests

## Overview

Comprehensive unit tests for FastAPI API layer (schemas, routers, dependencies).

**Status**: 30 tests written, pytest import configuration in progress

## Running API Tests

### Quick Start (Recommended)

Use the Makefile target which handles PYTHONPATH automatically:

```bash
make test-api         # Run API tests
make test-api-cov     # Run with coverage report
```

### Manual Execution

If running tests manually, you may need to set PYTHONPATH:

```bash
# From project root
PYTHONPATH=$PWD python3 -m pytest tests/unit/api/ -v

# With coverage
PYTHONPATH=$PWD python3 -m pytest tests/unit/api/ --cov=api --cov-report=term-missing
```

## Test Coverage

### Schemas (30 tests, 100% coverage target)

- ✅ `QueryRequest`, `QueryResponse`, `HPOResultItem` (14 tests)
- ✅ `SimilarityRequest`, `SimilarityResponse`, `SimilarityPairResult` (6 tests)
- ✅ `TextProcessingRequest`, `TextProcessingResponse`, `ExtractedHPOTerm` (6 tests)
- ✅ `ConfigInfoResponse` (4 tests)

**Coverage**: All Pydantic validation (field types, defaults, min/max, edge cases)

### Routers (Planned)

- ⏳ `health.py` - Basic health check endpoint
- ⏳ `query_router.py` - Main query endpoint with mocked dependencies
- ⏳ `similarity_router.py` - Similarity search endpoint
- ⏳ `text_processing_router.py` - Text processing endpoint

### Dependencies (Planned)

- ⏳ Model caching and dependency injection tests

## Test Structure

```
tests/unit/api/
├── README.md (this file)
├── __init__.py
└── test_schemas.py (30 tests)
```

## Known Issues

### Pytest Import Path Configuration

**Issue**: pytest's assertion rewriting phase occurs before PYTHONPATH/pythonpath configuration is processed, causing "ModuleNotFoundError: No module named 'api.schemas'" even though the module is installed.

**Root Cause**: pytest uses `/site-packages/_pytest/assertion/rewrite.py` which runs before:
- `conftest.py` hooks (pytest_configure, module-level sys.path)
- `pyproject.toml` pythonpath setting
- PYTHONPATH environment variable processing

**Workarounds Attempted**:
1. ✗ `pyproject.toml` `pythonpath = "."`
2. ✗ `pytest.ini` `pythonpath = .`
3. ✗ `conftest.py` pytest_configure hook
4. ✗ Module-level `sys.path.insert(0, ".")` in conftest.py
5. ✗ `PYTHONPATH` environment variable
6. ✗ `--import-mode=importlib`

**Current Solution**: Using Makefile targets that set PYTHONPATH

**Status**: **ACCEPTED** - This workaround is the official solution for API tests. The `make test-api` and `make test-api-cov` targets handle PYTHONPATH configuration automatically.

**Future Consideration**: The `src` layout refactoring was attempted but deferred to avoid regressions and allow more thorough research into pytest 9.0 configuration. This may be revisited as a separate, dedicated refactoring task.

## Future Improvements

1. **Consider `src` Layout**: Restructure project to use `src/phentrieve/` and `src/api/`
   - Benefit: Eliminates import path issues
   - Cost: Requires refactoring imports throughout project

2. **Add Router Tests**: Test FastAPI endpoints with `TestClient`

3. **Add Dependency Tests**: Test model loading, caching, and dependency injection

## Contributing

When adding new API tests:

1. Follow AAA pattern (Arrange, Act, Assert)
2. Use descriptive test names: `test_<function>_<scenario>_<expected>`
3. Mock external dependencies (models, databases, file I/O)
4. Test edge cases (empty inputs, validation errors, boundary conditions)
5. Use the Makefile targets for consistent test execution

## References

- [Pytest Import Mechanisms](https://docs.pytest.org/en/stable/explanation/pythonpath.html)
- [Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
