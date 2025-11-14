# Type Annotation Fixes - Comprehensive Plan

## Overview
- **Total Errors**: 158 across 29 files
- **Strategy**: Fix systematically by category, test incrementally
- **Goal**: 100% mypy compliance without regressions

---

## Error Categories & Fixes

### 1. Implicit Optional Errors (PEP 484) - ~40 errors ⚠️ HIGH PRIORITY

**Issue**: Functions with `default=None` but type hint doesn't include `| None`

**Files Affected**:
- `phentrieve/evaluation/metrics.py` (lines 56, 56)
- `phentrieve/data_processing/hpo_parser.py` (lines 410-413)
- `phentrieve/evaluation/result_analyzer.py` (line 143)
- `phentrieve/evaluation/comparison_orchestrator.py` (line 240)
- `phentrieve/text_processing/chunkers.py` (line 799)
- `phentrieve/indexing/chromadb_indexer.py` (line 33)
- `phentrieve/retrieval/query_orchestrator.py` (lines 43, 70, 297)
- `phentrieve/evaluation/runner.py` (lines 54, 55, 60)
- `phentrieve/evaluation/benchmark_orchestrator.py` (lines 50, 52, 57, 65)

**Fix Pattern**:
```python
# BEFORE
def func(arg: str = None):
    ...

# AFTER
def func(arg: str | None = None):
    ...
```

**Automated Fix**: Use `typing.Optional` or Python 3.10+ union syntax `T | None`

---

### 2. no-any-return Errors - ~15 errors ⚠️ HIGH PRIORITY

**Issue**: Functions return `Any` but declare specific return types

**Files Affected**:
- `phentrieve/utils.py` (lines 284, 399, 448)
- `phentrieve/data_processing/test_data_loader.py` (line 42)
- `phentrieve/data_processing/hpo_parser.py` (line 97)
- `phentrieve/cli/utils.py` (line 226)
- `phentrieve/retrieval/reranker.py` (line 47)
- `phentrieve/retrieval/query_orchestrator.py` (line 63)
- `api/routers/text_processing_router.py` (lines 66, 79, 92, 94, 112, 125, 138, 155)

**Fix Pattern**:
```python
# BEFORE
def get_path() -> Path:
    return config.get("path")  # Returns Any

# AFTER
def get_path() -> Path:
    result = config.get("path")
    return Path(result) if result else Path(".")
```

**Strategy**: Add type casts or assertions to convert `Any` to specific types

---

### 3. var-annotated Errors - ~10 errors 🔧 MEDIUM PRIORITY

**Issue**: Variables need explicit type annotations

**Files Affected**:
- `phentrieve/retrieval/text_attribution.py` (line 55) - `matched_spans`
- `phentrieve/data_processing/hpo_parser.py` (line 272) - `queue`
- `phentrieve/evaluation/comparison_orchestrator.py` (lines 302, 311) - `hr_by_k`, `ont_by_k`
- `phentrieve/text_processing/assertion_detection.py` (lines 545, 547) - `keyword_details`, `dependency_details`
- `phentrieve/evaluation/runner.py` (lines 181, 182, 186, 187)
- `phentrieve/evaluation/full_text_runner.py` (line 102)

**Fix Pattern**:
```python
# BEFORE
matched_spans = set()

# AFTER
matched_spans: set[tuple[int, int]] = set()
```

---

### 4. Undefined Global Variables - ~15 errors 🔴 CRITICAL

**Issue**: Module-level globals not defined before use

**File**: `phentrieve/retrieval/query_orchestrator.py`

**Variables**:
- `_global_model`
- `_global_retriever`
- `_global_cross_encoder`
- `_global_query_assertion_detector`

**Fix**:
```python
# Add at module level (top of file)
from typing import Optional
from sentence_transformers import SentenceTransformer

_global_model: Optional[SentenceTransformer] = None
_global_retriever: Optional[DenseRetriever] = None
_global_cross_encoder: Optional[CrossEncoder] = None
_global_query_assertion_detector: Optional[CombinedAssertionDetector] = None
```

---

### 5. Function Call Signature Mismatches - ~40 errors 🔴 CRITICAL

**Issue**: Function calls don't match current signatures

**Major Issues**:

**A. `rerank_with_cross_encoder` calls** (10+ errors)
- Wrong parameter names: `results` → `query_result`, `cross_encoder` → `cross_encoder_model`, `top_k` → missing
- File: `query_orchestrator.py`

**B. Missing required arguments** (5 errors)
- `api/routers/query_router.py` line 75: Missing `reranker_model`, `monolingual_reranker_model`, etc.

**C. Wrong parameter names** (10 errors)
- `orchestrate_hpo_extraction`: `similarity_threshold_per_chunk` → correct param name
- File: `full_text_runner.py`

**D. Type mismatches in parameters** (15 errors)
- `str | None` passed where `str` expected
- `int | None` passed where `int` expected

**Fix Strategy**:
1. Read function signature from definition
2. Update all call sites to match
3. Add None guards where needed

---

### 6. union-attr Errors (None checks) - ~5 errors 🔧 MEDIUM PRIORITY

**Issue**: Accessing attributes on potentially None values

**Files**:
- `comparison_orchestrator.py` (lines 872, 889) - `Match[str] | None`
- `text_processing_router.py` (line 46) - `str | None`

**Fix Pattern**:
```python
# BEFORE
result = pattern.match(text)
value = result.group(1)

# AFTER
result = pattern.match(text)
if result:
    value = result.group(1)
else:
    value = ""
```

---

### 7. Assignment Type Mismatches - ~15 errors 🔧 MEDIUM PRIORITY

**Issues**:
- `phentrieve/evaluation/metrics.py:180` - `int | float` assigned to `int`
- `phentrieve/text_processing/assertion_detection.py:76,82` - `None` assigned to `Language`
- `phentrieve/text_processing/chunkers.py:618` - `None` assigned to `Pattern[str]`
- `phentrieve/evaluation/comparison_orchestrator.py:862` - `list[Any]` to `ndarray`

**Fix Pattern**:
```python
# BEFORE
count: int = metric_value  # metric_value is int | float

# AFTER
count: int | float = metric_value
# OR
count: int = int(metric_value)
```

---

### 8. API Schema Issues - ~13 errors 🔴 CRITICAL

**Files**:
- `api/schemas/text_processing_schemas.py` (line 95)
- `api/routers/similarity_router.py` (lines 159, 163)
- `api/routers/query_router.py` (lines 75, 82, 85)
- `api/dependencies.py` (lines 60, 62)

**Issues**:
- Pydantic schema mismatches
- Missing required fields
- Wrong Literal types

**Fix Strategy**:
1. Review Pydantic v2 best practices
2. Ensure all required fields provided
3. Use correct Literal types

---

### 9. TypedDict/Indexing Errors - ~5 errors 🔧 MEDIUM PRIORITY

**Files**:
- `phentrieve/retrieval/dense_retriever.py` (lines 224, 227, 231, 233)
- `phentrieve/evaluation/metrics.py` (line 364)

**Issues**:
- Unknown TypedDict keys
- Invalid string indexing
- None indexing

**Fix Pattern**:
```python
# BEFORE
similarities = embeddings[0]  # embeddings is list[...] | None

# AFTER
if embeddings:
    similarities = embeddings[0]
```

---

### 10. Pipeline Type Variance Issues - ~10 errors 🔧 MEDIUM PRIORITY

**File**: `phentrieve/text_processing/pipeline.py`

**Issue**: List declared as `list[ParagraphChunker]` but contains subtypes

**Lines**: 112, 117, 120, 123, 149, 201, 230, 242, 244

**Fix**:
```python
# BEFORE
chunkers: list[ParagraphChunker] = []
chunkers.append(SentenceChunker(...))  # Wrong!

# AFTER
chunkers: list[TextChunker] = []
chunkers.append(SentenceChunker(...))  # Correct
```

---

## Implementation Plan

### Phase 1: Critical Fixes (2-3 hours)
1. ✅ Define global variables in `query_orchestrator.py`
2. ✅ Fix function signature mismatches in `query_orchestrator.py`
3. ✅ Fix API schema issues in `api/` modules
4. ✅ Fix pipeline variance issues

### Phase 2: High Priority (2-3 hours)
5. ✅ Fix all Implicit Optional errors (40 errors)
6. ✅ Fix no-any-return errors (15 errors)

### Phase 3: Medium Priority (1-2 hours)
7. ✅ Fix var-annotated errors (10 errors)
8. ✅ Fix union-attr errors (5 errors)
9. ✅ Fix assignment type mismatches (15 errors)
10. ✅ Fix TypedDict/indexing errors (5 errors)

### Phase 4: Testing & Validation (1 hour)
11. ✅ Run `make typecheck` - should pass with 0 errors
12. ✅ Run `make test` - ensure no regressions
13. ✅ Run `make lint` - ensure code quality
14. ✅ Commit with descriptive message

---

## Testing Strategy

After each phase:
```bash
# Type check
make typecheck

# Run tests
make test

# Ensure no new issues
git diff
```

**Success Criteria**:
- ✅ 0 mypy errors
- ✅ All 84/87 tests still passing (3 pre-existing failures)
- ✅ No Ruff violations
- ✅ Code remains readable and maintainable

---

## Best Practices References

1. **PEP 484** - Type Hints: https://peps.python.org/pep-0484/
2. **PEP 604** - Union Types: https://peps.python.org/pep-0604/
3. **mypy docs** - Common Issues: https://mypy.readthedocs.io/en/stable/common_issues.html
4. **Pydantic v2** - Type Annotations: https://docs.pydantic.dev/latest/

---

## Risk Mitigation

1. **Create branch**: `feature/type-annotations-fix`
2. **Incremental commits**: After each phase
3. **Test coverage**: Run full test suite after each commit
4. **Rollback plan**: Git reset if tests fail

---

## Estimated Time: 6-9 hours total

**Breakdown**:
- Phase 1 (Critical): 2-3 hours
- Phase 2 (High Priority): 2-3 hours
- Phase 3 (Medium Priority): 1-2 hours
- Phase 4 (Testing): 1 hour
