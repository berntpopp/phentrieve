# Testing Baseline (Phase 0)

**Date**: 2025-11-15
**Status**: Baseline Measured

## Current State

### Test Statistics
- **Total tests**: 87
- **Passing**: 84
- **Failing**: 3
- **Test duration**: 91.27 seconds

### Failing Tests
1. `tests/cli/test_query_commands.py::test_query_output_file_write_error` - Assertion error (error message format mismatch)
2. `tests/test_assertion_detection.py::TestDependencyAssertionDetector::test_negation_detection` - Negation detection not working
3. `tests/test_assertion_detection.py::TestDependencyAssertionDetector::test_normality_detection_after_refactoring` - Normality detection not working

### Coverage
- **Statement coverage**: 0% (4915 statements, 0 covered)
- **Reason**: Tests are heavily mocked, actual code paths not executed
- **Files measured**: `phentrieve/` and `api/` packages

### Test Structure
```
tests/
├── api/
│   ├── test_config_info_router.py
│   └── test_text_processing_router.py
├── cli/
│   ├── test_query_commands.py
│   └── test_similarity_commands.py
├── test_assertion_detection.py
├── test_basic_chunkers.py
├── test_chunking_pipeline_integration.py
├── test_resource_loader.py
├── test_semantic_metrics.py
└── test_sliding_window_chunker.py
```

### Test Distribution
- **API tests**: 2 files
- **CLI tests**: 2 files
- **Core/Library tests**: 6 files
- **Test style**: Mixed (unittest.TestCase + pytest)

## Baseline Files
- `coverage-baseline.json`: Full coverage report
- `baseline-coverage.log`: Test execution log

## Next Steps
Phase 1: Foundation - Create new test structure and migrate tests
