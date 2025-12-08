# Phenopacket Input Metadata Enhancement

**Issue**: Query/input text is not included in phenopacket metadata output
**Date**: 2025-12-08
**Status**: ✅ Implemented (2025-12-08)

## Problem Statement

When using `phentrieve query --interactive` or `phentrieve text` with phenopacket output format, the original input query/text is not preserved in the output metadata. This makes it difficult to:

1. Trace which input produced specific HPO term matches
2. Reproduce results or audit the retrieval process
3. Build downstream pipelines that need input-output correlation

### Current Output (Missing Input)

```json
{
  "metaData": {
    "created": "2025-12-08T10:47:29.119027Z",
    "createdBy": "phentrieve 0.4.0",
    "resources": [...],
    "externalReferences": [
      {"id": "phentrieve:embedding_model", "description": "FremyCompany/BioLORD-2023-M"}
    ]
  }
}
```

### Desired Output (With Input)

```json
{
  "metaData": {
    "externalReferences": [
      {"id": "phentrieve:embedding_model", "description": "FremyCompany/BioLORD-2023-M"},
      {"id": "phentrieve:input_text", "description": "kleinwuchs"}
    ]
  }
}
```

---

## Current Architecture Analysis

### Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT SOURCES                                  │
├─────────────────┬─────────────────┬─────────────────────────────────────┤
│ query command   │ text command    │ text --interactive                  │
│ (single query)  │ (full text)     │ (session queries)                   │
└────────┬────────┴────────┬────────┴──────────────┬──────────────────────┘
         │                 │                        │
         ▼                 ▼                        ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────────────┐
│ orchestrate_    │ │ orchestrate_    │ │ Interactive session with        │
│ query()         │ │ hpo_extraction()│ │ stored last_results             │
└────────┬────────┘ └────────┬────────┘ └──────────────┬──────────────────┘
         │                   │                          │
         ▼                   ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    format_as_phenopacket_v2()                           │
│  Parameters:                                                             │
│  - aggregated_results (query style)                                      │
│  - chunk_results (text style with chunk_text)                           │
│  - embedding_model, reranker_model                                       │
│  - ❌ NO input_text parameter currently                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Current Function Signature

**File**: `phentrieve/phenopackets/utils.py:84-93`

```python
def format_as_phenopacket_v2(
    aggregated_results: Optional[list[dict[str, Any]]] = None,
    chunk_results: Optional[list[dict[str, Any]]] = None,
    phentrieve_version: Optional[str] = None,
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
    hpo_version: Optional[str] = None,
) -> str:
```

### Call Sites Analysis

| Location | Command | Input Available | Currently Passed |
|----------|---------|-----------------|------------------|
| `query_commands.py:101-105` | `query` | `text` variable | ❌ Not passed |
| `text_commands.py:1023-1027` | `text process` | `raw_text` variable | ❌ Not passed |
| `text_interactive.py:494-498` | `text --interactive` | `user_input` in session | ❌ Not passed |
| `text_interactive.py:475-492` | `!p` export | Chunk texts embedded | ✅ In chunk_results |

### Key Finding: Partial Implementation Exists

The **interactive text mode** already embeds `chunk_text` in each chunk result:

```python
chunk_results_with_text = [
    {
        "chunk_idx": chunk_result.get("chunk_idx", i),
        "chunk_text": chunk_text,  # ✅ Source text preserved
        "matches": chunk_result.get("matches", []),
    }
]
```

This is used in evidence descriptions but **not** in top-level metadata.

---

## Proposed Solution

### Design Principles Applied

| Principle | Application |
|-----------|-------------|
| **DRY** | Single parameter `input_text` threads through all formatters |
| **KISS** | Use existing `externalReferences` pattern - no new structures |
| **SOLID** | |
| - Single Responsibility | Each function does one thing: format, create, serialize |
| - Open/Closed | Add parameter without changing existing behavior |
| - Interface Segregation | Optional parameter maintains backward compatibility |
| **Modular** | Change contained in `phenopackets/utils.py` + 3 call sites |

### Implementation Plan

#### Phase 1: Core Formatter Enhancement

**File**: `phentrieve/phenopackets/utils.py`

1. **Update function signature** (line 84):
```python
def format_as_phenopacket_v2(
    aggregated_results: Optional[list[dict[str, Any]]] = None,
    chunk_results: Optional[list[dict[str, Any]]] = None,
    phentrieve_version: Optional[str] = None,
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
    hpo_version: Optional[str] = None,
    input_text: Optional[str] = None,  # NEW: Original query/input text
) -> str:
```

2. **Thread parameter to internal functions**:
   - `_format_from_chunk_results()` → pass `input_text`
   - `_format_from_aggregated_results()` → pass `input_text`
   - Both call `_create_phenopacket_json()` with `input_text`

3. **Update `_create_phenopacket_json()`** (line 312):
```python
def _create_phenopacket_json(
    phenotypic_features: list,
    phentrieve_version: str,
    embedding_model: Optional[str],
    reranker_model: Optional[str],
    hpo_version: str,
    input_text: Optional[str] = None,  # NEW
) -> str:
```

4. **Add to external references** (after line 352):
```python
# Add input text reference (truncate if too long)
if input_text:
    # Truncate long texts for metadata (full text in evidence descriptions)
    truncated = input_text[:500] + "..." if len(input_text) > 500 else input_text
    external_references.append(
        ExternalReference(
            id="phentrieve:input_text",
            description=truncated,
        )
    )
```

#### Phase 2: Update Call Sites

**1. Query Command** (`query_commands.py:101-105`):
```python
phenopacket_json = format_as_phenopacket_v2(
    aggregated_results=aggregated_results,
    embedding_model=model_name,
    reranker_model=reranker_model if enable_reranker else None,
    input_text=text,  # NEW: Pass original query
)
```

**2. Text Command** (`text_commands.py:1023-1027`):
```python
phenopacket_json = format_as_phenopacket_v2(
    chunk_results=aggregated_results,
    embedding_model=embedding_model,
    reranker_model=reranker_model,
    input_text=raw_text,  # NEW: Pass original full text
)
```

**3. Interactive Mode** (`text_interactive.py:494-498`):
```python
phenopacket_json = format_as_phenopacket_v2(
    chunk_results=chunk_results_with_text,
    embedding_model=retrieval_model,
    reranker_model=reranker_model if enable_reranker else None,
    input_text=user_input,  # NEW: Pass session input
)
```

#### Phase 3: For Text Command - Add Chunk Summary

For the `text` command, also include chunk count in metadata:

```python
if input_text and chunk_results:
    external_references.append(
        ExternalReference(
            id="phentrieve:chunk_count",
            description=str(len(chunk_results)),
        )
    )
```

---

## Expected Output After Implementation

### Query Command Output

```json
{
  "id": "phentrieve-phenopacket-...",
  "phenotypicFeatures": [...],
  "metaData": {
    "created": "2025-12-08T10:47:29.119027Z",
    "createdBy": "phentrieve 0.4.0",
    "resources": [
      {
        "id": "hp",
        "name": "human phenotype ontology",
        "version": "v2025-03-03"
      }
    ],
    "phenopacketSchemaVersion": "2.0.2",
    "externalReferences": [
      {"id": "phentrieve:embedding_model", "description": "FremyCompany/BioLORD-2023-M"},
      {"id": "phentrieve:input_text", "description": "kleinwuchs"}
    ]
  }
}
```

### Text Command Output (with chunks)

```json
{
  "metaData": {
    "externalReferences": [
      {"id": "phentrieve:embedding_model", "description": "FremyCompany/BioLORD-2023-M"},
      {"id": "phentrieve:input_text", "description": "Patient presents with short stature and developmental delay..."},
      {"id": "phentrieve:chunk_count", "description": "5"}
    ]
  }
}
```

---

## File Change Summary

| File | Changes |
|------|---------|
| `phentrieve/phenopackets/utils.py` | Add `input_text` parameter, thread through functions, add to external references |
| `phentrieve/cli/query_commands.py` | Pass `text` to formatter (1 line) |
| `phentrieve/cli/text_commands.py` | Pass `raw_text` to formatter (1 line) |
| `phentrieve/cli/text_interactive.py` | Pass `user_input` to formatter (1 line) |
| `tests/` | Add tests for new parameter |

**Total LOC changed**: ~30-40 lines

---

## Testing Plan

1. **Unit Tests** (`tests/unit/test_phenopacket_utils.py`):
   - Test `input_text` appears in external references
   - Test truncation for texts > 500 chars
   - Test `None` input_text produces no reference (backward compat)
   - Test `chunk_count` appears for chunk-based results

2. **Integration Tests**:
   - `phentrieve query "seizures" --output-format phenopacket` → verify input_text
   - `phentrieve text process "..." --output-format phenopacket` → verify input_text + chunk_count

3. **Manual Verification**:
   - Interactive mode with `!t` toggle and query → verify metadata

---

## Backward Compatibility

✅ **Fully backward compatible**:
- `input_text` parameter is optional with default `None`
- Existing code continues to work unchanged
- Output only enhanced when parameter is provided

---

## Recommendation

**Proceed with implementation** - This is a low-risk, high-value enhancement that:
1. Follows existing patterns (externalReferences)
2. Requires minimal code changes (~40 lines)
3. Maintains full backward compatibility
4. Improves traceability and auditability

**Estimated effort**: 1-2 hours including tests
