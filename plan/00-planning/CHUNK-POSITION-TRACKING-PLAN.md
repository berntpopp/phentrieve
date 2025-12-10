# Chunk Position Tracking - Implementation Plan

**Status:** Ready for Implementation
**Created:** 2025-12-10
**Issues:** [#132](https://github.com/berntpopp/phentrieve/issues/132), [#36](https://github.com/berntpopp/phentrieve/issues/36)
**Estimated Time:** 6-8 hours

---

## Summary

Add character position tracking (`start_char`, `end_char`) to chunks, enabling text highlighting in frontend and accurate benchmark comparisons with PhenoBERT format.

**Approach:** Calculate positions at pipeline output using **pre-cleaned** text (before whitespace normalization). No changes to chunker interfaces.

---

## File Changes Overview

| File | Action | Lines |
|------|--------|-------|
| `phentrieve/text_processing/spans.py` | **CREATE** | ~50 |
| `phentrieve/text_processing/__init__.py` | MODIFY | +2 |
| `phentrieve/text_processing/pipeline.py` | MODIFY | +25 |
| `api/schemas/text_processing_schemas.py` | MODIFY | +8 |
| `api/routers/text_processing_router.py` | MODIFY | +6 |
| `phentrieve/phenopackets/utils.py` | MODIFY | +6 |
| `phentrieve/benchmark/extraction_benchmark.py` | MODIFY | +8, -4 |
| `tests/unit/test_spans.py` | **CREATE** | ~60 |

---

## 1. Create `spans.py`

**File:** `phentrieve/text_processing/spans.py` (NEW)

```python
"""Text span with position tracking."""

from dataclasses import dataclass
from typing import Any, Optional
import re


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Immutable text span with document position.

    Attributes:
        text: The text content
        start_char: Start position in document (0-indexed)
        end_char: End position (exclusive, like Python slicing)
    """
    text: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if self.start_char < 0:
            raise ValueError(f"start_char must be >= 0, got {self.start_char}")
        if self.end_char < self.start_char:
            raise ValueError(f"end_char must be >= start_char")

    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start_char": self.start_char, "end_char": self.end_char}


def find_span_in_text(
    needle: str,
    haystack: str,
    search_start: int = 0,
) -> Optional[TextSpan]:
    """Find text in document, returning position.

    Uses exact match first, then whitespace-normalized fallback.
    """
    if not needle or not haystack:
        return None

    # Exact match
    pos = haystack.find(needle, search_start)
    if pos != -1:
        return TextSpan(text=haystack[pos:pos + len(needle)], start_char=pos, end_char=pos + len(needle))

    # Whitespace-normalized fallback
    norm_needle = re.sub(r"\s+", " ", needle.strip())
    norm_haystack = re.sub(r"\s+", " ", haystack)
    pos = norm_haystack.find(norm_needle, search_start)
    if pos != -1:
        return TextSpan(text=needle, start_char=pos, end_char=pos + len(norm_needle))

    return None
```

---

## 2. Update `__init__.py`

**File:** `phentrieve/text_processing/__init__.py`

**After line 19** (after cleaners import), add:
```python
from phentrieve.text_processing.spans import TextSpan, find_span_in_text
```

**In `__all__` list (line 22-34)**, add before closing bracket:
```python
    "TextSpan",
    "find_span_in_text",
```

---

## 3. Update `pipeline.py`

**File:** `phentrieve/text_processing/pipeline.py`

### 3.1 Add import (after line 31)

```python
from phentrieve.text_processing.spans import find_span_in_text
```

### 3.2 Modify `process()` method signature (line 279)

**Replace:**
```python
    def process(self, raw_text: str) -> list[dict[str, Any]]:
```

**With:**
```python
    def process(self, raw_text: str, include_positions: bool = False) -> list[dict[str, Any]]:
```

### 3.3 Update docstring (lines 280-296)

Add to Args section:
```python
            include_positions: If True, include start_char/end_char in output
```

Add to Returns section:
```python
                    'start_char': int,          # Document position (if include_positions=True)
                    'end_char': int,            # Document position (if include_positions=True)
```

### 3.4 Store original text (after line 302, before normalization)

**Add after `return []`:**
```python
        # Store original for position tracking (BEFORE normalization)
        original_text = raw_text
```

### 3.5 Replace the final chunk processing loop (lines 364-391)

**Replace the entire loop:**
```python
        processed_chunks_with_assertion: list[dict[str, Any]] = []
        for idx, final_text_chunk in enumerate(final_raw_chunks_text_only):
            cleaned_final_chunk = clean_internal_newlines_and_extra_spaces(
                final_text_chunk
            )
            if not cleaned_final_chunk:
                continue

            assertion_status, assertion_details = self.assertion_detector.detect(
                cleaned_final_chunk
            )

            source_info_for_chunk = (
                current_source_info_list[idx]
                if idx < len(current_source_info_list)
                else ["unknown_source"]
            )

            processed_chunks_with_assertion.append(
                {
                    "text": cleaned_final_chunk,
                    "status": assertion_status,  # This is the AssertionStatus Enum object
                    "assertion_details": assertion_details,
                    "source_indices": {
                        "processing_stages": source_info_for_chunk
                    },  # Simplified source info
                }
            )
```

**With:**
```python
        processed_chunks_with_assertion: list[dict[str, Any]] = []
        search_start = 0  # Track position for sequential search

        for idx, final_text_chunk in enumerate(final_raw_chunks_text_only):
            # Calculate position BEFORE cleaning (using pre-cleaned text)
            start_char, end_char = -1, -1
            if include_positions:
                span = find_span_in_text(final_text_chunk, original_text, search_start)
                if span:
                    start_char, end_char = span.start_char, span.end_char
                    search_start = span.end_char

            cleaned_final_chunk = clean_internal_newlines_and_extra_spaces(
                final_text_chunk
            )
            if not cleaned_final_chunk:
                continue

            assertion_status, assertion_details = self.assertion_detector.detect(
                cleaned_final_chunk
            )

            source_info_for_chunk = (
                current_source_info_list[idx]
                if idx < len(current_source_info_list)
                else ["unknown_source"]
            )

            chunk_data: dict[str, Any] = {
                "text": cleaned_final_chunk,
                "status": assertion_status,
                "assertion_details": assertion_details,
                "source_indices": {"processing_stages": source_info_for_chunk},
            }

            if include_positions:
                chunk_data["start_char"] = start_char
                chunk_data["end_char"] = end_char

            processed_chunks_with_assertion.append(chunk_data)
```

---

## 4. Update API Schema

**File:** `api/schemas/text_processing_schemas.py`

### 4.1 Add to `ProcessedChunkAPI` (after line 131)

**After `hpo_matches` field, add:**
```python
    start_char: Optional[int] = Field(
        default=None,
        description="Start position in original document (0-indexed). None if not tracked.",
    )
    end_char: Optional[int] = Field(
        default=None,
        description="End position in original document (exclusive). None if not tracked.",
    )
```

### 4.2 Add to `TextProcessingRequest` (after line 107, after `include_details`)

```python
    include_chunk_positions: bool = Field(
        default=False,
        description="Include character positions (start_char, end_char) for each chunk.",
    )
```

---

## 5. Update API Router

**File:** `api/routers/text_processing_router.py`

### 5.1 Update pipeline call (lines 301-303)

**Replace:**
```python
        processed_chunks_list = await run_in_threadpool(
            text_pipeline.process, request.text_content
        )
```

**With:**
```python
        processed_chunks_list = await run_in_threadpool(
            text_pipeline.process,
            request.text_content,
            include_positions=request.include_chunk_positions,
        )
```

### 5.2 Update ProcessedChunkAPI creation (lines 310-316)

**Replace:**
```python
            api_processed_chunks.append(
                ProcessedChunkAPI(
                    chunk_id=idx + 1,  # 1-based for display/API
                    text=p_chunk["text"],
                    status=p_chunk["status"].value,  # Convert Enum to string
                    assertion_details=p_chunk.get("assertion_details"),
                )
            )
```

**With:**
```python
            api_processed_chunks.append(
                ProcessedChunkAPI(
                    chunk_id=idx + 1,
                    text=p_chunk["text"],
                    status=p_chunk["status"].value,
                    assertion_details=p_chunk.get("assertion_details"),
                    start_char=p_chunk.get("start_char"),
                    end_char=p_chunk.get("end_char"),
                )
            )
```

---

## 6. Update Phenopacket Output

**File:** `phentrieve/phenopackets/utils.py`

### 6.1 Update `_format_from_chunk_results()` (lines 193-215)

**Replace lines 193-215:**
```python
    for chunk_result in chunk_results:
        chunk_idx = chunk_result.get("chunk_idx", 0)
        chunk_text = chunk_result.get("chunk_text", "")
        matches = chunk_result.get("matches", [])

        for match in matches:
            hpo_id = match.get("id", "")
            hpo_name = match.get("name", "")
            score = match.get("score", 0.0)
            assertion_status = match.get("assertion_status")

            # Create OntologyClass for the feature type
            feature_type = OntologyClass(id=hpo_id, label=hpo_name)

            # Build description with confidence, chunk info, and source text
            # Note: No rank is provided since rankings are not comparable across chunks
            description_parts = [
                f"Phentrieve retrieval confidence: {score:.4f}",
                f"Chunk: {chunk_idx + 1}",
                f"Source text: {chunk_text}",
            ]
            if assertion_status:
                description_parts.insert(1, f"Assertion: {assertion_status}")
```

**With:**
```python
    for chunk_result in chunk_results:
        chunk_idx = chunk_result.get("chunk_idx", 0)
        chunk_text = chunk_result.get("chunk_text", "")
        start_char = chunk_result.get("start_char", -1)
        end_char = chunk_result.get("end_char", -1)
        matches = chunk_result.get("matches", [])

        for match in matches:
            hpo_id = match.get("id", "")
            hpo_name = match.get("name", "")
            score = match.get("score", 0.0)
            assertion_status = match.get("assertion_status")

            feature_type = OntologyClass(id=hpo_id, label=hpo_name)

            description_parts = [
                f"Phentrieve retrieval confidence: {score:.4f}",
                f"Chunk: {chunk_idx + 1}",
            ]
            if start_char >= 0 and end_char >= 0:
                description_parts.append(f"Start: {start_char}")
                description_parts.append(f"End: {end_char}")
            if assertion_status:
                description_parts.append(f"Assertion: {assertion_status}")
            description_parts.append(f"Source text: {chunk_text}")
```

---

## 7. Update Benchmark

**File:** `phentrieve/benchmark/extraction_benchmark.py`

### 7.1 Replace chunk position computation (lines 474-485)

**Replace:**
```python
        # Pre-compute chunk positions in full text
        chunk_positions: dict[int, tuple[int, int]] = {}
        last_end = 0
        for chunk_info in extraction_details.get("processed_chunks", []):
            chunk_idx = chunk_info["chunk_idx"]
            chunk_text = chunk_info["text"]
            start, end = self._find_chunk_position_in_text(
                chunk_text, full_text, last_end
            )
            chunk_positions[chunk_idx] = (start, end)
            if end > 0:
                last_end = end  # Continue searching from where we left off
```

**With:**
```python
        # Use pipeline-provided positions (fall back to search if unavailable)
        chunk_positions: dict[int, tuple[int, int]] = {}
        last_end = 0
        for chunk_info in extraction_details.get("processed_chunks", []):
            chunk_idx = chunk_info["chunk_idx"]
            start = chunk_info.get("start_char", -1)
            end = chunk_info.get("end_char", -1)

            # Fallback to string search if positions not provided
            if start < 0 or end < 0:
                chunk_text = chunk_info["text"]
                start, end = self._find_chunk_position_in_text(chunk_text, full_text, last_end)

            chunk_positions[chunk_idx] = (start, end)
            if end > 0:
                last_end = end
```

---

## 8. Create Unit Tests

**File:** `tests/unit/test_spans.py` (NEW)

```python
"""Unit tests for TextSpan and find_span_in_text."""

import pytest
from phentrieve.text_processing.spans import TextSpan, find_span_in_text


class TestTextSpan:
    """Tests for TextSpan dataclass."""

    def test_creation(self):
        span = TextSpan("hello", 0, 5)
        assert span.text == "hello"
        assert span.start_char == 0
        assert span.end_char == 5

    def test_str_returns_text(self):
        span = TextSpan("hello", 10, 15)
        assert str(span) == "hello"

    def test_len_returns_text_length(self):
        span = TextSpan("hello", 10, 15)
        assert len(span) == 5

    def test_immutable(self):
        span = TextSpan("hello", 0, 5)
        with pytest.raises(AttributeError):
            span.text = "world"

    def test_validation_negative_start(self):
        with pytest.raises(ValueError, match="start_char must be >= 0"):
            TextSpan("hello", -1, 5)

    def test_validation_end_before_start(self):
        with pytest.raises(ValueError, match="end_char must be >= start_char"):
            TextSpan("hello", 10, 5)

    def test_to_dict(self):
        span = TextSpan("hello", 10, 15)
        assert span.to_dict() == {"text": "hello", "start_char": 10, "end_char": 15}


class TestFindSpanInText:
    """Tests for find_span_in_text function."""

    def test_exact_match(self):
        span = find_span_in_text("world", "hello world")
        assert span is not None
        assert span.start_char == 6
        assert span.end_char == 11
        assert span.text == "world"

    def test_not_found(self):
        assert find_span_in_text("xyz", "hello world") is None

    def test_empty_inputs(self):
        assert find_span_in_text("", "hello") is None
        assert find_span_in_text("hello", "") is None

    def test_search_start(self):
        text = "hello hello"
        span1 = find_span_in_text("hello", text, search_start=0)
        span2 = find_span_in_text("hello", text, search_start=1)
        assert span1 is not None and span1.start_char == 0
        assert span2 is not None and span2.start_char == 6

    def test_whitespace_fallback(self):
        # Multiple spaces in haystack, single in needle
        span = find_span_in_text("hello world", "hello  world")
        assert span is not None

    def test_unicode_german(self):
        span = find_span_in_text("Trinkschwäche", "Kind mit Trinkschwäche")
        assert span is not None
        assert span.start_char == 9

    def test_unicode_french(self):
        span = find_span_in_text("été", "L'été est chaud")
        assert span is not None
        assert span.start_char == 2
```

---

## 9. Integration Test

**File:** `tests/integration/test_pipeline_positions.py` (NEW)

```python
"""Integration tests for pipeline position tracking."""

import pytest


class TestPipelinePositions:
    """Tests for position tracking in TextProcessingPipeline."""

    def test_default_no_positions(self, text_processing_pipeline):
        """Default: positions not included."""
        chunks = text_processing_pipeline.process("Patient has seizures.")
        for chunk in chunks:
            assert "start_char" not in chunk
            assert "end_char" not in chunk

    def test_with_positions(self, text_processing_pipeline):
        """Positions included when requested."""
        text = "Patient has seizures. No fever."
        chunks = text_processing_pipeline.process(text, include_positions=True)

        for chunk in chunks:
            assert "start_char" in chunk
            assert "end_char" in chunk
            if chunk["start_char"] >= 0:
                assert chunk["end_char"] <= len(text)

    def test_sequential_positions(self, text_processing_pipeline):
        """Positions are sequential (no overlaps)."""
        text = "Pain noted. Pain increased. Pain resolved."
        chunks = text_processing_pipeline.process(text, include_positions=True)

        positions = [(c["start_char"], c["end_char"]) for c in chunks if c["start_char"] >= 0]
        for i in range(1, len(positions)):
            assert positions[i][0] >= positions[i-1][1], "Positions should not overlap"
```

---

## Testing Commands

```bash
# Run new unit tests
pytest tests/unit/test_spans.py -v

# Run integration tests
pytest tests/integration/test_pipeline_positions.py -v

# Run all tests
make test

# Type check
make typecheck-fast

# Lint
make check
```

---

## Verification Checklist

- [ ] `make check` passes (0 lint errors)
- [ ] `make typecheck-fast` passes (0 type errors)
- [ ] `make test` passes (all tests green)
- [ ] API endpoint returns positions when `include_chunk_positions=true`
- [ ] Phenopacket output includes `Start: X | End: Y` in description
- [ ] Benchmark uses pipeline positions instead of string search

---

## Rollback

All changes are backward compatible:
- `include_positions=False` default preserves existing behavior
- API fields are `Optional` with `None` default
- Benchmark falls back to string search if positions unavailable

```bash
git revert <commit-range>
```
