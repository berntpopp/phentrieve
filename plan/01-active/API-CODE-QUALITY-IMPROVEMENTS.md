# API Code Quality Improvements Plan (REVISED)

**Status:** In Progress
**Created:** 2025-01-17
**Revised:** 2025-01-17 (Senior Review: Anti-patterns removed, focus sharpened)
**Target Completion:** 2025-01-19
**Priority:** High
**Related Issues:**
- [#42: Add comprehensive tests for chunk index mapping](https://github.com/berntpopp/phentrieve/issues/42)
- [#41: Refactor chunking config generation to eliminate code duplication](https://github.com/berntpopp/phentrieve/issues/41)

---

## Objective

**Focus on actual problems, avoid over-engineering.**

Improve code quality and reliability through:
1. **Eliminate real duplication** in chunking config (extract duplicated loop only)
2. **Add comprehensive tests** for chunk index alignment (the real value)
3. **Add assertion-based validation** to catch bugs in development

**What we're NOT doing:**
- ❌ Creating utility modules for trivial arithmetic (`+ 1`)
- ❌ Complex dictionary mappings with lambdas
- ❌ Optional/conditional validation (inconsistent behavior)
- ❌ Over-abstraction without clear benefit

**Principles:**
- **KISS**: Simple extraction, explicit code
- **DRY**: Extract actual duplication (the loop), not trivial operations
- **YAGNI**: Don't create abstractions until proven necessary
- **Explicit > Implicit**: `idx + 1` is clearer than `convert_idx_to_id(idx)`

---

## Success Criteria

### Issue #41: Chunking Config Refactoring
- [ ] Extract `_apply_sliding_window_params()` helper function (single responsibility)
- [ ] Refactor `_get_chunking_config_for_api()` to eliminate 70 lines of duplication
- [ ] Reduce function length from ~100 to ~50 lines
- [ ] Maintain identical API behavior for all 7 chunking strategies
- [ ] Add unit tests for helper function (>95% coverage)
- [ ] Keep if/elif structure (explicit is better than clever)
- [ ] Pass mypy type checking (0 errors)
- [ ] Pass Ruff linting (0 errors)

### Issue #42: Chunk Index Alignment Tests
- [ ] Add `_validate_response_chunk_references()` assertion helper
- [ ] Integrate validation using `__debug__` (zero production cost)
- [ ] Create comprehensive integration test suite (all strategies, edge cases)
- [ ] Verify chunk ID alignment across all response components
- [ ] Test edge cases: empty text, single chunk, multiple chunks
- [ ] **Keep explicit `idx + 1` conversions** (no utility module)
- [ ] Pass mypy type checking (0 errors)
- [ ] Pass Ruff linting (0 errors)
- [ ] Achieve >90% coverage for integration test scenarios

### General Quality Gates
- [ ] All 157 existing tests pass (no regressions)
- [ ] All 42 E2E Docker tests pass
- [ ] CI/CD pipeline passes
- [ ] No performance degradation
- [ ] Code is simpler, not more complex

---

## Critical Review: Anti-Patterns Identified & Avoided

### ❌ Anti-Pattern #1: Complex Dictionary Mapping (Original Plan)
```python
# DON'T DO THIS - Complex, hard to type, hard to maintain
strategy_mapping: dict[str, tuple[Callable[[], Generator[...]], bool]] = {
    "sliding_window": (lambda: get_sliding_window_config_with_params(...), False),
}
```

**Problems:**
- Lambda with closure (captures local variables)
- Inconsistent callable signatures
- Unreadable type hints
- Magic boolean parameter
- Mixing data and behavior

**Solution:** Keep simple if/elif with extracted helper (see implementation).

### ❌ Anti-Pattern #2: Utility Module for Trivial Operations (Original Plan)
```python
# DON'T DO THIS - Over-abstraction
def convert_chunk_idx_to_api_id(idx: int) -> int:
    """50 lines of docstring for: return idx + 1"""
    if idx < 0:
        raise ValueError(...)
    return idx + 1
```

**Problems:**
- Indirection without value
- False sense of safety (validation at wrong layer)
- Obscures simple arithmetic
- YAGNI violation

**Solution:** Keep explicit `idx + 1` conversions, validate at response level.

### ❌ Anti-Pattern #3: Optional Validation (Original Plan)
```python
# DON'T DO THIS - Inconsistent behavior
if logger.isEnabledFor(logging.DEBUG):
    validate_chunk_ids_exist(...)  # Only validates in debug mode
```

**Problems:**
- Different behavior in debug vs production
- Heisenbugs (bugs that disappear when debugging)
- Testing doesn't catch what production doesn't validate

**Solution:** Use `__debug__` assertions (compile-time, consistent).

---

## Context & Background

### Issue #41: Code Duplication Analysis

**Location:** `api/routers/text_processing_router.py:71-141`

**Current state:** 70 lines of duplicated code across 6 strategies:
```python
# Repeated 6 times with identical logic
for component in config:
    if component.get("type") == "sliding_window":
        component["config"].update({
            "window_size_tokens": cfg_window_size,
            "step_size_tokens": cfg_step_size,
            "splitting_threshold": cfg_split_threshold,
            "min_split_segment_length_words": cfg_min_segment_length,
        })
```

**Root cause:** Loop logic extracted to functions, but parameter application not extracted.

**Impact:** Maintenance burden, inconsistency risk when adding parameters.

### Issue #42: Index Alignment Analysis

**Location:** `api/routers/text_processing_router.py:286, 323, 345-346, 356-357, 362`

**Current state:** Index conversions at 5 locations:
- Line 286: `chunk_id=idx + 1` ← processed chunks
- Line 323: `chunk_id = chunk_idx + 1` ← detailed results
- Line 345-346: `chunk_id=attribution.get("chunk_idx", 0) + 1` ← attributions
- Line 356-357: `source_chunk_ids = [chunk_idx + 1 for ...]` ← aggregated terms
- Line 362: `top_evidence_chunk_id = top_evidence_chunk_idx + 1` ← top evidence

**Real problem:** Not the conversions themselves (explicit is good), but **lack of validation** that references are consistent.

**Impact:** Potential misalignment between:
- `aggregated_hpo_terms.source_chunk_ids` and `processed_chunks`
- `text_attributions.chunk_id` and `processed_chunks`
- `top_evidence_chunk_id` and `processed_chunks`

**Solution:** Comprehensive tests + assertion-based validation, not abstraction.

---

## Implementation Steps

### Phase 1: Issue #41 - Extract Duplicated Loop

#### Step 1.1: Create Helper Function (Single Responsibility)

**File:** `api/routers/text_processing_router.py`

Add before `_get_chunking_config_for_api`:

```python
def _apply_sliding_window_params(
    config: list[dict[str, Any]],
    window_size: int,
    step_size: int,
    threshold: float,
    min_segment_length: int,
) -> None:
    """
    Apply sliding window parameters to chunking configuration components.

    Modifies config in-place by updating parameters for any components
    with type='sliding_window'.

    Args:
        config: Chunking pipeline configuration (modified in-place)
        window_size: Window size in tokens
        step_size: Step size in tokens
        threshold: Similarity threshold for splitting (0.0-1.0)
        min_segment_length: Minimum segment length in words
    """
    for component in config:
        if component.get("type") == "sliding_window":
            component["config"].update({
                "window_size_tokens": window_size,
                "step_size_tokens": step_size,
                "splitting_threshold": threshold,
                "min_split_segment_length_words": min_segment_length,
            })
```

**Why this design:**
- ✅ Single responsibility: only updates sliding window params
- ✅ Simple signature: easy to type hint, test, understand
- ✅ In-place modification: clear from signature (`-> None`)
- ✅ Focused: does one thing well

#### Step 1.2: Refactor Main Function (Keep Explicit)

**File:** `api/routers/text_processing_router.py`

Replace `_get_chunking_config_for_api` with:

```python
def _get_chunking_config_for_api(
    request: TextProcessingRequest,
) -> list[dict[str, Any]]:
    """
    Get chunking configuration based on request strategy and parameters.

    Args:
        request: Text processing request with strategy and parameters

    Returns:
        Chunking pipeline configuration list
    """
    strategy_name = (
        request.chunking_strategy.lower()
        if request.chunking_strategy
        else "sliding_window_punct_conj_cleaned"
    )

    # Extract parameters with defaults
    ws = request.window_size if request.window_size is not None else 7
    ss = request.step_size if request.step_size is not None else 1
    th = request.split_threshold if request.split_threshold is not None else 0.5
    msl = request.min_segment_length if request.min_segment_length is not None else 3

    logger.debug(
        f"API: Building config for '{strategy_name}': "
        f"ws={ws}, ss={ss}, th={th}, msl={msl}"
    )

    # Strategy selection - explicit if/elif is GOOD for small sets
    if strategy_name == "simple":
        return list(get_simple_chunking_config())

    elif strategy_name == "sliding_window":
        # Special case: takes params directly, no post-processing needed
        return list(get_sliding_window_config_with_params(ws, ss, th, msl))

    # All other strategies: get base config, then apply params
    elif strategy_name == "semantic":
        config = list(get_semantic_chunking_config())
    elif strategy_name == "detailed":
        config = list(get_detailed_chunking_config())
    elif strategy_name == "sliding_window_cleaned":
        config = list(get_sliding_window_cleaned_config())
    elif strategy_name == "sliding_window_punct_cleaned":
        config = list(get_sliding_window_punct_cleaned_config())
    elif strategy_name == "sliding_window_punct_conj_cleaned":
        config = list(get_sliding_window_punct_conj_cleaned_config())
    else:
        # Unknown strategy - use default with warning
        logger.warning(
            f"API: Unknown strategy '{strategy_name}', "
            f"using sliding_window_punct_conj_cleaned"
        )
        config = list(get_sliding_window_punct_conj_cleaned_config())

    # Apply sliding window parameters to config
    _apply_sliding_window_params(config, ws, ss, th, msl)
    return config
```

**Why this design:**
- ✅ Explicit if/elif: clear, readable, maintainable
- ✅ Single helper call: eliminates all duplication
- ✅ Type-safe: simple types, easy for mypy
- ✅ Self-documenting: logic flow is obvious
- ✅ Easy to extend: adding new strategy is straightforward

**Line count:** ~100 lines → ~50 lines (mission accomplished)

#### Step 1.3: Add Unit Tests

**File:** `tests/unit/api/test_text_processing_router.py` (new)

```python
"""Unit tests for text processing router helper functions."""

import pytest
from typing import Any

from api.routers.text_processing_router import (
    _apply_sliding_window_params,
    _get_chunking_config_for_api,
)
from api.schemas.text_processing_schemas import TextProcessingRequest


pytestmark = pytest.mark.unit


class TestApplySlidingWindowParams:
    """Test _apply_sliding_window_params helper function."""

    def test_updates_sliding_window_component(self):
        """Test sliding window component parameters are updated."""
        # Arrange
        config = [
            {
                "type": "sliding_window",
                "config": {
                    "window_size_tokens": 5,
                    "step_size_tokens": 1,
                },
            },
            {
                "type": "other_component",
                "config": {"param": "value"},
            },
        ]

        # Act
        _apply_sliding_window_params(
            config=config,
            window_size=10,
            step_size=2,
            threshold=0.7,
            min_segment_length=5,
        )

        # Assert
        sw_config = config[0]["config"]
        assert sw_config["window_size_tokens"] == 10
        assert sw_config["step_size_tokens"] == 2
        assert sw_config["splitting_threshold"] == 0.7
        assert sw_config["min_split_segment_length_words"] == 5

        # Other component unchanged
        assert config[1]["config"] == {"param": "value"}

    def test_handles_config_without_sliding_window(self):
        """Test gracefully handles config without sliding_window component."""
        # Arrange
        config = [{"type": "other", "config": {}}]

        # Act - should not raise
        _apply_sliding_window_params(config, 10, 2, 0.7, 5)

        # Assert - no changes
        assert config == [{"type": "other", "config": {}}]

    def test_updates_multiple_sliding_window_components(self):
        """Test updates all sliding_window components if multiple exist."""
        # Arrange
        config = [
            {"type": "sliding_window", "config": {}},
            {"type": "other", "config": {}},
            {"type": "sliding_window", "config": {}},
        ]

        # Act
        _apply_sliding_window_params(config, 15, 3, 0.8, 10)

        # Assert - both sliding_window components updated
        assert config[0]["config"]["window_size_tokens"] == 15
        assert config[2]["config"]["window_size_tokens"] == 15


class TestGetChunkingConfigForApi:
    """Test _get_chunking_config_for_api function."""

    @pytest.mark.parametrize(
        "strategy_name",
        [
            "simple",
            "semantic",
            "detailed",
            "sliding_window",
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_all_strategies_return_valid_config(self, strategy_name: str):
        """Test all strategies return valid configuration."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy=strategy_name,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert
        assert isinstance(config, list)
        assert len(config) > 0
        assert all(isinstance(c, dict) for c in config)
        assert all("type" in c and "config" in c for c in config)

    def test_unknown_strategy_uses_default(self):
        """Test unknown strategy falls back to default."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="unknown_nonexistent_strategy",
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should return valid config (default)
        assert isinstance(config, list)
        assert len(config) > 0

    def test_custom_sliding_window_parameters_applied(self):
        """Test custom sliding window parameters are applied correctly."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="semantic",
            window_size=15,
            step_size=3,
            split_threshold=0.8,
            min_segment_length=10,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - find sliding_window component
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        assert sw_component is not None

        sw_config = sw_component["config"]
        assert sw_config["window_size_tokens"] == 15
        assert sw_config["step_size_tokens"] == 3
        assert sw_config["splitting_threshold"] == 0.8
        assert sw_config["min_split_segment_length_words"] == 10

    def test_default_parameters_when_none_provided(self):
        """Test default parameters used when not specified in request."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy="semantic",
            # No custom params - should use defaults
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should have sliding_window with some params
        sw_component = next(
            (c for c in config if c.get("type") == "sliding_window"), None
        )
        assert sw_component is not None
        assert "window_size_tokens" in sw_component["config"]
        assert "step_size_tokens" in sw_component["config"]

    def test_none_chunking_strategy_uses_default(self):
        """Test None/missing chunking_strategy uses default."""
        # Arrange
        request = TextProcessingRequest(
            text_content="test text",
            chunking_strategy=None,
        )

        # Act
        config = _get_chunking_config_for_api(request)

        # Assert - should return valid config
        assert isinstance(config, list)
        assert len(config) > 0
```

**Coverage target:** >95% for helper function

---

### Phase 2: Issue #42 - Comprehensive Testing + Validation

#### Step 2.1: Add Assertion-Based Validation Helper

**File:** `api/routers/text_processing_router.py`

Add before the `@router.post("/process")` endpoint:

```python
def _validate_response_chunk_references(
    processed_chunks: list[ProcessedChunkAPI],
    aggregated_terms: list[AggregatedHPOTermAPI],
) -> None:
    """
    Validate chunk ID references in API response for internal consistency.

    This function checks invariants that must hold for a valid response:
    1. Chunk IDs are sequential and 1-based
    2. All source_chunk_ids reference existing chunks
    3. All text_attribution chunk_ids reference existing chunks
    4. top_evidence_chunk_id references an existing chunk (if present)

    This is called under __debug__ (Python assertions enabled) to catch
    bugs during development/testing without production overhead.

    Args:
        processed_chunks: List of processed chunks with chunk_id
        aggregated_terms: List of aggregated HPO terms with chunk references

    Raises:
        AssertionError: If any invariant is violated
    """
    total_chunks = len(processed_chunks)
    chunk_ids = {chunk.chunk_id for chunk in processed_chunks}

    # Invariant 1: Chunk IDs are sequential 1-based
    expected_ids = set(range(1, total_chunks + 1))
    assert chunk_ids == expected_ids, (
        f"Chunk IDs not sequential 1-based. "
        f"Expected {expected_ids}, got {chunk_ids}"
    )

    # Invariant 2: All source_chunk_ids reference existing chunks
    for term in aggregated_terms:
        invalid_source_ids = set(term.source_chunk_ids) - chunk_ids
        assert not invalid_source_ids, (
            f"HPO term {term.hpo_id} has invalid source_chunk_ids: "
            f"{invalid_source_ids} (valid range: 1-{total_chunks})"
        )

    # Invariant 3: All text_attribution chunk_ids reference existing chunks
    for term in aggregated_terms:
        for attr in term.text_attributions:
            assert attr.chunk_id in chunk_ids, (
                f"HPO term {term.hpo_id} has text_attribution with invalid "
                f"chunk_id {attr.chunk_id} (valid range: 1-{total_chunks})"
            )

    # Invariant 4: top_evidence_chunk_id references existing chunk (if present)
    for term in aggregated_terms:
        if term.top_evidence_chunk_id is not None:
            assert term.top_evidence_chunk_id in chunk_ids, (
                f"HPO term {term.hpo_id} has invalid top_evidence_chunk_id "
                f"{term.top_evidence_chunk_id} (valid range: 1-{total_chunks})"
            )
```

**Why this design:**
- ✅ Assertion-based: only runs when Python assertions enabled (`python`, not `python -O`)
- ✅ Clear invariants: documents what must be true
- ✅ Informative errors: tells you exactly what's wrong
- ✅ Zero production cost: `__debug__` is compile-time constant
- ✅ Consistent: always validates or never validates (no conditional logic)

#### Step 2.2: Integrate Validation into Endpoint

**File:** `api/routers/text_processing_router.py`

In `process_text_extract_hpo` function, add **before the final return**:

```python
        # ... existing code to build response ...

        # Validate response invariants (only when assertions enabled)
        if __debug__:
            _validate_response_chunk_references(
                api_processed_chunks, api_aggregated_hpo_terms
            )

        return TextProcessingResponseAPI(
            meta=response_meta,
            processed_chunks=api_processed_chunks,
            aggregated_hpo_terms=api_aggregated_hpo_terms,
        )
```

**Why `__debug__`:**
- Compile-time constant (not runtime check like `if DEBUG:`)
- Completely removed when Python run with `-O` flag
- Production deployments use `-O` → zero overhead
- Development/testing always has assertions → catches bugs early

#### Step 2.3: Keep Explicit Index Conversions (No Utility Module)

**File:** `api/routers/text_processing_router.py`

**DO NOT** change existing conversions. Keep them explicit:

```python
# Line 286: KEEP AS-IS
chunk_id=idx + 1,  # 0-based to 1-based for API

# Line 323: KEEP AS-IS
chunk_id = chunk_idx + 1  # 0-based to 1-based

# Line 345-346: KEEP AS-IS
chunk_id=attribution.get("chunk_idx", 0) + 1,  # 0-based to 1-based

# Line 356-357: KEEP AS-IS
source_chunk_ids = [
    chunk_idx + 1  # 0-based to 1-based
    for chunk_idx in term_data.get("chunks", [])
]

# Line 362: KEEP AS-IS
top_evidence_chunk_id = top_evidence_chunk_idx + 1  # 0-based to 1-based
```

**Why keep explicit:**
- ✅ Clear intent: `idx + 1` immediately shows conversion
- ✅ No indirection: no need to jump to utility function
- ✅ YAGNI: we don't need abstraction for simple arithmetic
- ✅ Comments clarify: explain the conversion inline
- ✅ Validation catches errors: tests validate correctness, not implementation

#### Step 2.4: Add Comprehensive Integration Tests

**File:** `tests/integration/test_text_processing_chunk_alignment.py` (new)

```python
"""Integration tests for chunk index alignment in text processing API.

These tests validate that chunk IDs remain consistent across all components
of the API response, preventing off-by-one errors and invalid references.
"""

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from api.main import app

    return TestClient(app)


class TestChunkIdConsistency:
    """Test chunk ID consistency across API response components."""

    def test_chunk_ids_are_sequential_one_based(self, api_client):
        """Chunk IDs must be sequential starting from 1."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia. No heart disease.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        chunks = data["processed_chunks"]
        chunk_ids = [c["chunk_id"] for c in chunks]

        # Invariant: sequential 1-based IDs
        assert chunk_ids == list(range(1, len(chunks) + 1)), (
            f"Chunk IDs should be sequential 1-based, got {chunk_ids}"
        )

    def test_source_chunk_ids_reference_existing_chunks(self, api_client):
        """All source_chunk_ids must reference existing processed chunks."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia. No heart disease.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        valid_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}

        for term in data["aggregated_hpo_terms"]:
            for chunk_id in term["source_chunk_ids"]:
                assert chunk_id in valid_chunk_ids, (
                    f"Term {term['hpo_id']} references non-existent chunk {chunk_id}. "
                    f"Valid IDs: {valid_chunk_ids}"
                )

    def test_text_attribution_chunk_ids_valid(self, api_client):
        """All text_attribution chunk_ids must reference existing chunks."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        valid_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}

        for term in data["aggregated_hpo_terms"]:
            for attribution in term["text_attributions"]:
                assert attribution["chunk_id"] in valid_chunk_ids, (
                    f"Term {term['hpo_id']} attribution references "
                    f"non-existent chunk {attribution['chunk_id']}"
                )

    def test_top_evidence_chunk_id_valid_when_present(self, api_client):
        """top_evidence_chunk_id must reference existing chunk if not None."""
        # Arrange
        request = {
            "text_content": "Patient has seizures.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        valid_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}

        for term in data["aggregated_hpo_terms"]:
            top_chunk_id = term.get("top_evidence_chunk_id")
            if top_chunk_id is not None:
                assert top_chunk_id in valid_chunk_ids, (
                    f"Term {term['hpo_id']} top_evidence_chunk_id {top_chunk_id} "
                    f"does not reference an existing chunk"
                )


class TestChunkAlignmentAcrossStrategies:
    """Test chunk alignment for all chunking strategies."""

    @pytest.mark.parametrize(
        "strategy",
        [
            "simple",
            "semantic",
            "detailed",
            "sliding_window",
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_strategy_maintains_chunk_alignment(self, api_client, strategy):
        """All chunking strategies must maintain valid chunk references."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia. Tremor noted. No heart disease.",
            "language": "en",
            "chunking_strategy": strategy,
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate chunk ID consistency
        valid_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}

        for term in data["aggregated_hpo_terms"]:
            # Check source_chunk_ids
            for chunk_id in term["source_chunk_ids"]:
                assert chunk_id in valid_chunk_ids, (
                    f"Strategy '{strategy}': Term {term['hpo_id']} "
                    f"references invalid chunk {chunk_id}"
                )

            # Check text_attributions
            for attribution in term["text_attributions"]:
                assert attribution["chunk_id"] in valid_chunk_ids, (
                    f"Strategy '{strategy}': Term {term['hpo_id']} "
                    f"attribution has invalid chunk_id {attribution['chunk_id']}"
                )

            # Check top_evidence_chunk_id
            if term.get("top_evidence_chunk_id") is not None:
                assert term["top_evidence_chunk_id"] in valid_chunk_ids, (
                    f"Strategy '{strategy}': Term {term['hpo_id']} "
                    f"has invalid top_evidence_chunk_id"
                )


class TestEdgeCases:
    """Test edge cases for chunk index alignment."""

    def test_single_chunk_all_references_to_one(self, api_client):
        """Single chunk case - all references should be to chunk_id 1."""
        # Arrange
        request = {
            "text_content": "Seizures",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should have exactly 1 chunk
        assert len(data["processed_chunks"]) == 1
        assert data["processed_chunks"][0]["chunk_id"] == 1

        # All references should be to chunk_id 1
        for term in data["aggregated_hpo_terms"]:
            assert all(cid == 1 for cid in term["source_chunk_ids"]), (
                f"Single chunk scenario: all source_chunk_ids should be 1, "
                f"got {term['source_chunk_ids']}"
            )

    def test_empty_text_graceful_handling(self, api_client):
        """Empty text should be handled gracefully."""
        # Arrange
        request = {
            "text_content": "",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert - should handle gracefully (200 or 400 acceptable)
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            # Should have valid structure
            assert "processed_chunks" in data
            assert "aggregated_hpo_terms" in data
            assert isinstance(data["processed_chunks"], list)
            assert isinstance(data["aggregated_hpo_terms"], list)

    def test_whitespace_only_text_handling(self, api_client):
        """Whitespace-only text should be handled gracefully."""
        # Arrange
        request = {
            "text_content": "   \n\t   ",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            assert "processed_chunks" in data
            assert "aggregated_hpo_terms" in data

    def test_very_long_text_maintains_alignment(self, api_client):
        """Long text with many chunks maintains alignment."""
        # Arrange - create long text
        sentences = [
            "Patient has seizures.",
            "Ataxia noted.",
            "No heart disease.",
            "Tremor observed.",
            "Muscle weakness present.",
        ]
        long_text = " ".join(sentences * 5)  # 25 sentences

        request = {
            "text_content": long_text,
            "language": "en",
            "chunking_strategy": "simple",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should have multiple chunks
        num_chunks = len(data["processed_chunks"])
        assert num_chunks > 1

        # All chunk IDs should be valid
        valid_chunk_ids = set(range(1, num_chunks + 1))
        actual_chunk_ids = {c["chunk_id"] for c in data["processed_chunks"]}
        assert actual_chunk_ids == valid_chunk_ids

        # All references should be valid
        for term in data["aggregated_hpo_terms"]:
            assert all(
                1 <= cid <= num_chunks for cid in term["source_chunk_ids"]
            ), f"Invalid source_chunk_ids in long text: {term['source_chunk_ids']}"


class TestHPOMatchesInChunks:
    """Test HPO matches within processed chunks structure."""

    def test_hpo_matches_structure_valid(self, api_client):
        """HPO matches in chunks should have valid structure."""
        # Arrange
        request = {
            "text_content": "Patient has seizures and ataxia.",
            "language": "en",
        }

        # Act
        response = api_client.post("/api/v1/text/process", json=request)

        # Assert
        assert response.status_code == 200
        data = response.json()

        for chunk in data["processed_chunks"]:
            # Each chunk should have valid structure
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "status" in chunk
            assert "hpo_matches" in chunk

            # hpo_matches should be a list
            assert isinstance(chunk["hpo_matches"], list)

            # Each match should have required fields
            for match in chunk["hpo_matches"]:
                assert "hpo_id" in match
                assert "name" in match
                assert "score" in match
                assert isinstance(match["score"], (int, float))
                assert match["score"] >= 0.0
```

**Coverage target:** >90% for integration test scenarios

**Why this test suite adds value:**
- ✅ Tests actual behavior: validates invariants that must hold
- ✅ Catches real bugs: invalid references, off-by-one errors
- ✅ Comprehensive: all strategies, all edge cases
- ✅ Self-documenting: test names explain requirements
- ✅ Regression prevention: any future changes must pass these

---

### Phase 3: Quality Assurance

#### Step 3.1: Type Checking

```bash
# Fast incremental check (recommended)
make typecheck-fast

# Full check
make typecheck
```

**Acceptance:** 0 mypy errors

**Common issues to watch:**
- Ensure `_apply_sliding_window_params` has proper type hints
- Ensure `_validate_response_chunk_references` has proper type hints
- Check `list[dict[str, Any]]` return types are correct

#### Step 3.2: Linting & Formatting

```bash
# Format and lint
make check

# Or separately
make format  # Fix formatting
make lint    # Check linting
```

**Acceptance:** 0 Ruff errors

**Expected changes:**
- May need to adjust line lengths
- May need to add/remove trailing commas
- Ensure docstrings follow project style

#### Step 3.3: Unit Tests

```bash
# Run new unit tests
pytest tests/unit/api/test_text_processing_router.py -v

# With coverage
pytest tests/unit/api/test_text_processing_router.py --cov=api.routers.text_processing_router --cov-report=term-missing
```

**Acceptance:**
- All tests pass
- >95% coverage for `_apply_sliding_window_params`
- >90% coverage for `_get_chunking_config_for_api`

#### Step 3.4: Integration Tests

```bash
# Run new integration tests
pytest tests/integration/test_text_processing_chunk_alignment.py -v

# With coverage
pytest tests/integration/test_text_processing_chunk_alignment.py --cov=api.routers.text_processing_router -v
```

**Acceptance:**
- All tests pass (all strategies, all edge cases)
- No failures in chunk alignment validation

#### Step 3.5: Full Test Suite (Regression Check)

```bash
# Run all tests
make test

# Or with coverage
make test-cov
```

**Acceptance:**
- All 157 existing tests pass
- New tests added to count
- No regressions

#### Step 3.6: E2E Docker Tests

```bash
# Full E2E test suite
make test-e2e
```

**Acceptance:**
- All 42 E2E tests pass
- No Docker-related issues

---

## Dependencies

### Prerequisites
- ✅ Python 3.9+ environment
- ✅ pytest, mypy, Ruff installed (via `make install-dev`)
- ✅ Docker for E2E tests
- ✅ Existing test infrastructure

### Blocked By
- None - can start immediately

### Blocks
- None - independent improvement

---

## Rollback Plan

### Granular Rollback (Preferred)

Each phase committed separately for easy rollback:

```bash
# View recent commits
git log --oneline -10

# Revert specific commit (keeps history)
git revert <commit-hash>

# Or reset to before changes (rewrites history - use carefully)
git reset --hard <commit-before-changes>
```

### Full Rollback

If major issues discovered:

```bash
# Create rollback branch
git checkout -b rollback-api-quality-improvements

# Reset to before all changes
git reset --hard <commit-before-phase-1>

# Force push (coordinate with team)
git push origin rollback-api-quality-improvements --force

# Create rollback PR
gh pr create --title "Rollback: API Quality Improvements" --body "Reason: ..."
```

### Verification After Rollback

```bash
make check      # Lint/format
make typecheck  # Type check
make test       # All tests
make test-e2e   # E2E tests
```

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Breaking API response format** | High | Very Low | Integration tests validate exact response structure |
| **Type checking regressions** | Medium | Low | Incremental mypy checks after each step |
| **Performance degradation** | Medium | Very Low | `__debug__` validation has zero production cost |
| **Test flakiness** | Low | Low | Use deterministic test data, avoid time dependencies |
| **Merge conflicts** | Low | Medium | Work in feature branch, coordinate with team |

---

## Performance Considerations

### Expected Impact

**Phase 1 (Refactoring):**
- Impact: None (same logic, just reorganized)
- Overhead: Zero

**Phase 2 (Validation):**
- Development: Assertions run, negligible overhead
- Production: `__debug__` is `False`, validation completely removed by Python
- Overhead: Zero in production

### Benchmark (if needed)

```bash
# Before changes
time pytest tests/integration/ -k "text_processing"

# After changes
time pytest tests/integration/ -k "text_processing"
```

**Expected:** No measurable difference

---

## Documentation Updates

### Code Documentation
- ✅ Docstrings for all new functions (Google style)
- ✅ Type hints for all parameters and returns
- ✅ Inline comments explaining `__debug__` usage

### Project Documentation
- Update `CLAUDE.md` if new patterns introduced
- Document test file locations

### No API Documentation Changes
- API contract unchanged
- Response format identical

---

## Timeline

| Phase | Task | Time Estimate |
|-------|------|---------------|
| **Phase 1** | Extract helper + refactor | 1-2 hours |
| **Phase 1** | Unit tests | 1 hour |
| **Phase 2** | Validation helper | 0.5 hours |
| **Phase 2** | Integration tests | 2-3 hours |
| **Phase 3** | QA (type/lint/test) | 1 hour |
| **Total** | | **5.5-7.5 hours** |

**Target Completion:** 2025-01-19 (2 days, single developer)

---

## Success Metrics

### Code Quality
- ✅ 0 mypy errors (maintain standard)
- ✅ 0 Ruff errors (maintain standard)
- ✅ >95% coverage for new utility functions
- ✅ >90% coverage for integration tests

### Maintainability
- ✅ 70 lines of duplication eliminated
- ✅ Clear, explicit code (if/elif over complex mapping)
- ✅ Single responsibility functions
- ✅ Self-documenting test suite

### Reliability
- ✅ Assertion-based validation catches bugs in development
- ✅ Comprehensive tests prevent regressions
- ✅ All edge cases covered

### Simplicity
- ✅ No over-engineering (no utility module for `+ 1`)
- ✅ Explicit conversions (readable, maintainable)
- ✅ Simple extraction (no clever tricks)

---

## Anti-Patterns Avoided

This plan explicitly avoids:

❌ **Complex dictionary mappings** with lambdas and closures
❌ **Utility modules** for trivial operations (`+ 1`)
❌ **Optional/conditional validation** (inconsistent behavior)
❌ **Over-abstraction** without clear benefit
❌ **Magic booleans** in configuration tuples
❌ **Mixing data and behavior** in dictionaries

Instead, we follow:

✅ **KISS**: Simple extraction of duplicated loop
✅ **DRY**: One function for one repeated block
✅ **YAGNI**: Don't create abstractions until proven necessary
✅ **Explicit > Implicit**: `idx + 1` clearer than function call
✅ **Test behavior**: Validate invariants, not implementation

---

## Resources

### Best Practices
- [Real Python: Refactoring](https://realpython.com/python-refactoring/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Python Assertions](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)

### Internal Docs
- `plan/02-completed/TESTING-MODERNIZATION-PLAN.md`
- `CLAUDE.md`

### Related Issues
- [#41: Refactor chunking config](https://github.com/berntpopp/phentrieve/issues/41)
- [#42: Chunk index mapping tests](https://github.com/berntpopp/phentrieve/issues/42)

---

## Post-Implementation Checklist

- [ ] Phase 1: Helper function implemented and tested
- [ ] Phase 1: Refactoring complete (100→50 lines)
- [ ] Phase 1: Unit tests pass (>95% coverage)
- [ ] Phase 2: Validation helper implemented
- [ ] Phase 2: Integration tests complete (all strategies, edge cases)
- [ ] Phase 3: Type checking passes (0 errors)
- [ ] Phase 3: Linting passes (0 errors)
- [ ] Phase 3: All 157+ tests pass
- [ ] Phase 3: All 42 E2E tests pass
- [ ] Code review completed
- [ ] PR created and merged
- [ ] Issues #41 and #42 closed
- [ ] Plan moved to `plan/02-completed/`

---

**Status:** Ready to implement
**Next Action:** Begin Phase 1 - Extract `_apply_sliding_window_params()` helper
**Approach:** Focused, pragmatic, no over-engineering
