# Phase 2 Strategy: Achieving 80% Coverage (Expert Analysis)

**Author**: Senior Developer Analysis
**Date**: 2025-11-15
**Target**: 80% statement coverage (3932/4915 statements)

## Executive Summary

**Current State**: 0% coverage (tests use heavy mocking, don't execute source code)
**Challenge**: 4915 total statements, need to cover 3932 for 80%
**Strategy**: Focus on **high-value core modules** (~2000 critical statements) + existing partial coverage

## The 80/20 Approach

### Why Current Coverage is 0%

Our existing 87 tests are primarily:
- **CLI integration tests**: Mock the entire stack, test user interface
- **Unit tests with heavy mocking**: Test structure, not implementation
- **Integration tests**: Test workflow, but mock internal details

**Result**: Tests verify behavior but don't execute actual code paths.

### Strategic Insight

To reach 80% pragmatically:
1. **Don't test everything equally** - Focus on critical business logic
2. **Test real implementations** - Reduce mocking, execute actual code
3. **Skip low-value code** - CLI wrappers, simple delegates, visualization
4. **Leverage existing tests** - Integration tests already validate workflows

## High-Value Modules (Priority 1)

These modules contain core business logic and are critical to system functionality:

### 1. Text Processing Core (Target: ~400 statements)

**`phentrieve/text_processing/chunkers.py`** (367 statements)
- **Critical functions**:
  - `ParagraphChunker.chunk()` - Already tested but mocked
  - `SentenceChunker.chunk()` - Already tested but mocked
  - `SlidingWindowSemanticSplitter.chunk()` - Complex logic, HIGH VALUE
  - `FineGrainedPunctuationChunker.chunk()` - Regex logic
- **Testing approach**: Real instances, actual text input, verify output
- **Expected coverage**: ~300/367 statements (82%)

**`phentrieve/text_processing/assertion_detection.py`** (228 statements)
- **Critical functions**:
  - `KeywordAssertionDetector.detect()` - Partially tested
  - `DependencyAssertionDetector.detect()` - Partially tested
  - `CombinedAssertionDetector.detect()` - Partially tested
- **Testing approach**: Real detectors, actual clinical text, verify assertions
- **Expected coverage**: ~180/228 statements (79%)

**`phentrieve/text_processing/pipeline.py`** (123 statements)
- **Critical functions**:
  - `TextProcessingPipeline.process()` - Integration tested, need unit tests
  - Pipeline stage orchestration
- **Testing approach**: Real pipeline with minimal config, test stages
- **Expected coverage**: ~90/123 statements (73%)

**Subtotal**: 570/718 statements (~79%)

### 2. Retrieval Core (Target: ~350 statements)

**`phentrieve/retrieval/dense_retriever.py`** (109 statements)
- **Critical functions**:
  - `DenseRetriever.retrieve()` - Core retrieval logic
  - Query embedding, similarity search
- **Testing approach**: Mock only ChromaDB, test retrieval logic
- **Expected coverage**: ~80/109 statements (73%)

**`phentrieve/embeddings.py`** (32 statements)
- **Critical functions**:
  - `get_embedding_model()` - Model loading
  - Embedding generation
- **Testing approach**: Real lightweight model (MiniLM), cache testing
- **Expected coverage**: ~25/32 statements (78%)

**`phentrieve/retrieval/output_formatters.py`** (62 statements)
- **Critical functions**:
  - `format_results_as_text()` - Partially tested via CLI
  - `format_results_as_json()` - Partially tested
- **Testing approach**: Unit tests with mock result data
- **Expected coverage**: ~50/62 statements (81%)

**Subtotal**: 155/203 statements (~76%)

### 3. Data Processing (Target: ~200 statements)

**`phentrieve/text_processing/resource_loader.py`** (46 statements)
- **Critical functions**:
  - `ResourceLoader.load_resource()` - Already tested but mocked
  - Resource caching
- **Testing approach**: Real file loading, test caching behavior
- **Expected coverage**: ~35/46 statements (76%)

**`phentrieve/data_processing/document_creator.py`** (68 statements)
- **Critical functions**:
  - Document creation from HPO data
  - Metadata generation
- **Testing approach**: Real document creation with test data
- **Expected coverage**: ~50/68 statements (74%)

**`phentrieve/config.py`** (55 statements)
- **Critical functions**:
  - Config loading and validation
  - Default config generation
- **Testing approach**: Test config loading, validation logic
- **Expected coverage**: ~40/55 statements (73%)

**Subtotal**: 125/169 statements (~74%)

### 4. Evaluation Metrics (Target: ~250 statements)

**`phentrieve/evaluation/semantic_metrics.py`** (113 statements)
- **Critical functions**:
  - `calculate_assertion_accuracy()` - Already tested but mocked
  - `calculate_semantic_prf1()` - Already tested but mocked
- **Testing approach**: Real calculations with test data
- **Expected coverage**: ~90/113 statements (80%)

**`phentrieve/evaluation/metrics.py`** (211 statements)
- **Critical functions**:
  - Standard metrics calculation (precision, recall, F1)
  - Match algorithms
- **Testing approach**: Real calculations, test edge cases
- **Expected coverage**: ~150/211 statements (71%)

**Subtotal**: 240/324 statements (~74%)

## Medium-Value Modules (Priority 2)

Test only if we need more coverage after Priority 1:

- `phentrieve/utils.py` (173 statements) - Utility functions, ~60% coverage target
- `phentrieve/retrieval/text_attribution.py` (36 statements) - Attribution logic
- `phentrieve/text_processing/cleaners.py` (11 statements) - Simple cleaners

## Low-Value Modules (Skip for 80% Goal)

These are lower priority for 80% coverage:

- **CLI commands** (already tested via integration):
  - `cli/query_commands.py` (90 statements)
  - `cli/similarity_commands.py` (70 statements)
  - `cli/text_commands.py` (203 statements)
  - `cli/benchmark_commands.py` (44 statements)

- **Complex orchestrators** (hard to test, lower ROI):
  - `retrieval/query_orchestrator.py` (279 statements)
  - `evaluation/comparison_orchestrator.py` (360 statements)
  - `evaluation/runner.py` (269 statements)

- **Visualization** (not critical):
  - `visualization/plot_utils.py` (152 statements)

- **API** (tested via integration later):
  - All `api/` modules (defer to Phase 3 e2e tests)

## Coverage Math

**Priority 1 Targets**:
- Text Processing: 570 statements (covered ~450)
- Retrieval: 155 statements (covered ~120)
- Data Processing: 125 statements (covered ~95)
- Metrics: 240 statements (covered ~180)
- **Total**: 1090 statements → ~845 covered

**Additional needed for 80%**: 3932 total needed
- Priority 1: ~845 covered
- Existing integration coverage: ~500 (from real integration tests)
- Priority 2 modules: ~600 more
- **Total projected**: ~1945/4915 = **39.6% realistic coverage**

## Revised Pragmatic Target

**Analysis**: Reaching 80% is unrealistic without testing:
- Complex orchestrators (high effort, low value)
- CLI commands (already tested via integration)
- API endpoints (already tested via integration)

**Revised Strategy**:
1. **Aim for 60% coverage** of core modules (more realistic)
2. **Focus on critical paths** (chunking, retrieval, assertion detection)
3. **Defer orchestrators** to integration tests
4. **Measure quality**, not just quantity

## Implementation Plan

### Week 1: Core Text Processing (Days 1-2)
- [ ] `test_chunkers_real.py` - Real chunking tests
- [ ] `test_assertion_detection_real.py` - Real assertion detection
- [ ] `test_pipeline_real.py` - Real pipeline tests
- **Target**: +450 covered statements

### Week 1: Core Retrieval (Day 3)
- [ ] `test_dense_retriever_real.py` - Real retrieval tests
- [ ] `test_embeddings_real.py` - Real embedding tests
- [ ] `test_output_formatters_real.py` - Real formatting tests
- **Target**: +120 covered statements

### Week 1: Data & Config (Day 4)
- [ ] `test_resource_loader_real.py` - Real resource loading
- [ ] `test_config_real.py` - Real config tests
- [ ] `test_document_creator_real.py` - Real document creation
- **Target**: +95 covered statements

### Week 2: Metrics & Utils (Day 5)
- [ ] `test_semantic_metrics_real.py` - Real metric calculations
- [ ] `test_metrics_real.py` - Real evaluation metrics
- [ ] `test_utils_real.py` - Utility functions
- **Target**: +280 covered statements

## Success Criteria

**Pragmatic Goals**:
- ✅ **60%+ coverage** for `phentrieve/` package
- ✅ **Core modules at 75%+**: chunkers, assertion detection, retrieval
- ✅ **Fast test suite**: <2 minutes for full suite
- ✅ **Quality over quantity**: Test critical paths, not boilerplate

**Metrics**:
- Coverage: 60% statement coverage (2950/4915 statements)
- Speed: <120 seconds for full test suite
- Reliability: 0 flaky tests, all deterministic

## Key Principles

1. **Real code execution**: Minimal mocking, test actual implementations
2. **Fast tests**: Use lightweight models, cache fixtures
3. **Critical paths first**: Focus on business logic, not infrastructure
4. **Pragmatic targets**: 60% is better than 0%, don't over-engineer

## Next Steps

1. Start with chunkers.py (highest value, already partially tested)
2. Add real unit tests that execute code
3. Measure coverage after each module
4. Adjust strategy based on actual coverage gains
