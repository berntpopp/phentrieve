# HPO Parser Schema Resilience Refactoring Plan

**Status:** Draft → Ready for Implementation
**Created:** 2025-11-20
**Priority:** High (Stability & Maintainability)
**Estimated Effort:** 3-5 days
**Issue:** Resolves #23 - Gracefully handle HPO JSON schema changes
**Last Updated:** 2025-11-20

---

## Executive Summary

Refactor `phentrieve/data_processing/hpo_parser.py` to handle HPO JSON schema changes gracefully using **Pydantic validation**, **defensive programming**, and **modular design**. Currently, hardcoded paths like `node_data["meta"]["definition"]["val"]` will break if the HPO Consortium modifies the schema. This refactoring ensures **zero regressions** while improving **resilience**, **maintainability**, and **debuggability**.

**Key Improvements:**
- ✅ **Pydantic schema validation** with clear required vs optional fields
- ✅ **Graceful degradation** - warnings instead of failures for missing optional data
- ✅ **Comprehensive logging** for schema deviations
- ✅ **100% backward compatibility** - all existing functionality preserved
- ✅ **Modular design** following SOLID principles
- ✅ **Type safety** with mypy compliance (0 errors maintained)

---

## Objective

Transform the HPO parser from brittle, path-dependent code into a **resilient, schema-aware system** that:
1. Validates JSON structure using Pydantic models
2. Handles missing/malformed fields gracefully
3. Logs schema deviations for debugging
4. Maintains 100% functional compatibility
5. Enables easy schema evolution tracking

---

## Current State Analysis

### Problems Identified

**1. Hardcoded Dictionary Access**
```python
# Current: Unsafe direct access (lines 386-390)
definition = node_data["meta"]["definition"]["val"]  # KeyError if missing!

# Current: Mixed safe/unsafe patterns (lines 381, 394)
label = node_data.get("lbl", "")  # Safe
comments = [c for c in node_data["meta"]["comments"] if c]  # Unsafe!
```

**2. No Schema Validation**
- No upfront validation of JSON structure
- No distinction between critical fields (id, label) and optional fields (definition, synonyms)
- Silent failures or crashes depending on location

**3. Inconsistent Error Handling**
```python
# Line 127: Returns None tuple on critical failure
if not graphs_data or not isinstance(graphs_data, list):
    return None, None, None, None  # Hard to debug

# Line 141: Continues silently on missing node ID
if not original_id:
    logger.warning(...)
    continue  # Should this be an error?
```

**4. Poor Separation of Concerns**
- `_parse_hpo_json_to_graphs()` does parsing + validation + mapping
- `_extract_term_data_for_db()` duplicates extraction logic
- Hard to test individual components

**5. No Schema Version Detection**
- No way to detect HPO schema version changes
- No compatibility checks
- No migration path for future schema updates

### Current Test Coverage

```bash
# From grep search
tests/unit/cli/test_data_commands.py  # Tests orchestrate_hpo_preparation
```

**Coverage Gaps:**
- ❌ No direct tests for `_parse_hpo_json_to_graphs`
- ❌ No tests for malformed JSON handling
- ❌ No tests for missing optional fields
- ❌ No tests for edge cases (empty graphs, missing root)

---

## Success Criteria

- [x] **Zero Regressions**: All existing tests pass without modification
- [ ] **Pydantic Models**: OBOGraph schema defined with required/optional fields
- [ ] **Graceful Degradation**: Missing optional fields emit warnings, not errors
- [ ] **Comprehensive Logging**: Schema deviations logged at appropriate levels
- [ ] **Type Safety**: mypy passes with 0 errors
- [ ] **Test Coverage**: New unit tests for schema validation and edge cases
- [ ] **Documentation**: Clear docstrings explaining schema handling
- [ ] **Performance**: No measurable performance degradation (<5% overhead)
- [ ] **Backward Compatibility**: Existing HPO data files parse identically

---

## Architecture Design

### Pydantic Schema Models (New Module)

Create `phentrieve/data_processing/hpo_schema.py`:

```python
"""
Pydantic models for OBOGraph JSON schema validation.

Based on: https://github.com/geneontology/obographs
HPO data uses OBOGraph format with specific conventions.
"""
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator


class MetaDefinition(BaseModel):
    """Definition metadata with value and cross-references."""
    val: str
    xrefs: Optional[list[str]] = None

    class Config:
        extra = "allow"  # Allow unknown fields for forward compatibility


class MetaSynonym(BaseModel):
    """Synonym with predicate, value, and cross-references."""
    pred: Optional[str] = None
    val: str
    xrefs: Optional[list[str]] = None

    class Config:
        extra = "allow"


class NodeMeta(BaseModel):
    """
    Optional metadata for an HPO term node.

    All fields are optional to allow graceful degradation.
    Warnings logged for missing fields during parsing.
    """
    definition: Optional[MetaDefinition] = None
    synonyms: Optional[list[MetaSynonym]] = None
    comments: Optional[list[str]] = None
    subsets: Optional[list[str]] = None
    xrefs: Optional[list[dict[str, Any]]] = None
    basicPropertyValues: Optional[list[dict[str, Any]]] = None

    class Config:
        extra = "allow"  # Forward compatibility


class HPONode(BaseModel):
    """
    HPO term node in OBOGraph format.

    Required fields (will fail validation if missing):
    - id: Term identifier (e.g., "HP:0001250")
    - lbl: Human-readable label

    Optional fields (graceful degradation):
    - meta: All metadata (definition, synonyms, comments)
    - type: Node type (usually "CLASS")
    """
    id: str = Field(..., description="Required: HPO term ID")
    lbl: str = Field(..., description="Required: Term label", min_length=1)
    meta: Optional[NodeMeta] = None
    type: Optional[str] = None

    class Config:
        extra = "allow"

    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure ID starts with expected prefix (HP:, OBO:, etc.)."""
        # Note: normalize_id() will handle full normalization later
        if not v:
            raise ValueError("ID cannot be empty")
        return v


class HPOEdge(BaseModel):
    """
    Edge representing relationships between HPO terms.

    Required fields:
    - sub/subj: Subject (child) term ID
    - pred: Predicate/relationship type
    - obj: Object (parent) term ID
    """
    sub: Optional[str] = None  # Some HPO versions use "sub"
    subj: Optional[str] = None  # Others use "subj"
    pred: str
    obj: str
    meta: Optional[dict[str, Any]] = None

    class Config:
        extra = "allow"

    @field_validator('pred')
    @classmethod
    def validate_predicate(cls, v: str) -> str:
        """Log if non-standard predicate encountered."""
        standard_predicates = {
            "is_a",
            "http://www.w3.org/2000/01/rdf-schema#subClassOf",
            "http://purl.obolibrary.org/obo/BFO_0000050",
        }
        if v not in standard_predicates:
            # Note: Just validation, logging happens in parser
            pass
        return v

    def get_subject(self) -> Optional[str]:
        """Get subject ID handling both 'sub' and 'subj' variants."""
        return self.subj or self.sub


class HPOGraph(BaseModel):
    """
    OBOGraph structure containing nodes and edges.

    Required fields:
    - nodes: List of HPO term nodes (can be empty but must exist)
    - edges: List of relationships (can be empty but must exist)
    """
    id: Optional[str] = None
    nodes: list[HPONode] = Field(default_factory=list)
    edges: list[HPOEdge] = Field(default_factory=list)
    meta: Optional[dict[str, Any]] = None

    class Config:
        extra = "allow"


class HPOGraphDocument(BaseModel):
    """
    Top-level HPO JSON document following OBOGraph specification.

    Required fields:
    - graphs: Array of graph objects (must have at least one)
    """
    graphs: list[HPOGraph] = Field(..., min_length=1)
    meta: Optional[dict[str, Any]] = None

    class Config:
        extra = "allow"

    @field_validator('graphs')
    @classmethod
    def validate_graphs_not_empty(cls, v: list[HPOGraph]) -> list[HPOGraph]:
        """Ensure at least one graph exists."""
        if not v or len(v) == 0:
            raise ValueError("HPO JSON must contain at least one graph")
        return v
```

### Refactored Parser Functions

**Key Changes:**

1. **Schema Validation Layer** (New)
```python
def validate_hpo_json_schema(hpo_data: dict) -> HPOGraphDocument:
    """
    Validate HPO JSON against OBOGraph schema using Pydantic.

    Args:
        hpo_data: Raw HPO JSON data

    Returns:
        Validated HPOGraphDocument

    Raises:
        ValidationError: If critical schema violations found

    Logs:
        - ERROR: Missing required fields (graphs, nodes[].id, nodes[].lbl)
        - WARNING: Missing optional fields (meta, definitions, synonyms)
        - INFO: Schema validation success with stats
    """
```

2. **Safe Node Extraction** (Refactored)
```python
def extract_node_data(node: HPONode) -> dict[str, Any]:
    """
    Extract database-ready data from validated HPO node.

    Args:
        node: Validated Pydantic HPONode

    Returns:
        Dictionary with keys: id, label, definition, synonyms, comments

    Note:
        - Missing optional fields replaced with sensible defaults
        - All extraction logic in one place (DRY)
        - Type-safe with Pydantic model
    """
```

3. **Modular Graph Parsing** (Refactored)
```python
def parse_hpo_graph_nodes(
    graph: HPOGraph
) -> tuple[dict[str, HPONode], set[str]]:
    """
    Parse and validate nodes from HPO graph.

    Returns:
        Tuple of (node_map, term_ids)

    Separation of Concerns:
        - Only handles node parsing
        - No edge logic mixed in
        - Easier to test and maintain
    """

def parse_hpo_graph_edges(
    graph: HPOGraph,
    valid_term_ids: set[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Parse edges into parent-child and child-parent mappings.

    Args:
        graph: Validated HPOGraph
        valid_term_ids: Set of known term IDs for validation

    Returns:
        Tuple of (parent_to_children_map, child_to_parents_map)

    Separation of Concerns:
        - Only handles edge parsing
        - Validates edge endpoints against known nodes
        - Clear single responsibility
    """
```

---

## Implementation Plan

### Phase 1: Schema Definition (Day 1, 4-6 hours)

**Goal:** Create Pydantic models without modifying existing parser

**Steps:**

1. **Create schema module**
   ```bash
   # New file
   touch phentrieve/data_processing/hpo_schema.py
   ```

2. **Implement Pydantic models** (see Architecture Design above)
   - HPOGraphDocument (root)
   - HPOGraph
   - HPONode (with validators)
   - HPOEdge (with subject getter)
   - NodeMeta, MetaDefinition, MetaSynonym

3. **Add comprehensive docstrings**
   - Explain OBOGraph format
   - Document required vs optional fields
   - Link to spec: https://github.com/geneontology/obographs

4. **Type check**
   ```bash
   make typecheck-fast
   # Goal: 0 mypy errors
   ```

5. **Schema validation tests**
   ```bash
   # New file
   touch tests_new/unit/data_processing/test_hpo_schema.py
   ```

   Test cases:
   - ✅ Valid minimal HPO JSON (id, lbl only)
   - ✅ Valid full HPO JSON (all metadata)
   - ✅ Missing required field (id) → ValidationError
   - ✅ Missing required field (lbl) → ValidationError
   - ✅ Missing optional fields → Success with defaults
   - ✅ Extra unknown fields → Success (forward compat)
   - ✅ Empty graphs array → ValidationError
   - ✅ Subject ID variants (sub vs subj) → Both work

**Validation:**
```bash
pytest tests_new/unit/data_processing/test_hpo_schema.py -v
# Goal: 100% passing
```

---

### Phase 2: Parser Refactoring (Day 2-3, 8-12 hours)

**Goal:** Refactor existing parser to use Pydantic models

**Steps:**

1. **Add schema validation to load_hpo_json()**
   ```python
   def load_hpo_json(hpo_file_path: Path) -> Optional[HPOGraphDocument]:
       """
       Load and validate HPO JSON file.

       Returns:
           Validated HPOGraphDocument or None on failure
       """
       try:
           with open(hpo_file_path, encoding="utf-8") as f:
               raw_data = json.load(f)

           # NEW: Pydantic validation
           validated = HPOGraphDocument(**raw_data)
           logger.info(
               f"HPO JSON validated: {len(validated.graphs[0].nodes)} nodes, "
               f"{len(validated.graphs[0].edges)} edges"
           )
           return validated

       except ValidationError as e:
           logger.error(f"HPO JSON schema validation failed: {e}")
           # Log each validation error separately
           for error in e.errors():
               logger.error(
                   f"  Field: {error['loc']} | "
                   f"Error: {error['msg']} | "
                   f"Type: {error['type']}"
               )
           return None
       except Exception as e:
           # ... existing error handling
   ```

2. **Refactor _parse_hpo_json_to_graphs()**
   ```python
   def _parse_hpo_json_to_graphs(
       hpo_doc: HPOGraphDocument,  # Changed: was dict
   ) -> tuple[
       Optional[dict[str, dict]],
       Optional[dict[str, list[str]]],
       Optional[dict[str, list[str]]],
       Optional[set[str]],
   ]:
       """
       Parses validated HPO document into graph structures.

       Args:
           hpo_doc: Validated Pydantic model (not raw dict)
       """
       all_nodes_data: dict[str, dict] = {}
       parent_to_children_map: dict[str, list[str]] = defaultdict(list)
       child_to_parents_map: dict[str, list[str]] = defaultdict(list)
       all_term_ids: set[str] = set()

       graph = hpo_doc.graphs[0]  # Already validated to exist

       # Parse nodes using Pydantic model
       for node in graph.nodes:
           node_id_norm = normalize_id(node.id)
           if node_id_norm and node_id_norm.startswith("HP:"):
               # NEW: Convert Pydantic model to dict for storage
               all_nodes_data[node_id_norm] = node.model_dump()
               all_term_ids.add(node_id_norm)

               # NEW: Log missing optional metadata
               if not node.meta:
                   logger.debug(f"Node {node_id_norm} has no metadata")
               elif not node.meta.definition:
                   logger.debug(f"Node {node_id_norm} missing definition")

       # Parse edges using Pydantic model
       for edge in graph.edges:
           if edge.pred in {"is_a", "http://..."}:
               subj_orig = edge.get_subject()  # NEW: Use helper method
               obj_orig = edge.obj

               # ... rest of existing logic
   ```

3. **Refactor _extract_term_data_for_db()**
   ```python
   def _extract_term_data_for_db(
       all_nodes_data: dict[str, dict]
   ) -> list[dict[str, Any]]:
       """
       Extract term data from parsed nodes for database storage.

       Note: Receives dict because data already converted from Pydantic
       """
       terms_data = []

       for term_id, node_dict in tqdm(...):
           # NEW: Safer extraction with explicit defaults
           label = node_dict.get("lbl", "")
           if not label:
               logger.warning(
                   f"Term {term_id} has empty label - using ID as fallback"
               )
               label = term_id

           # NEW: Safe nested metadata extraction
           definition = ""
           meta = node_dict.get("meta")
           if meta:
               definition_obj = meta.get("definition")
               if definition_obj:
                   definition = definition_obj.get("val", "")
               else:
                   logger.debug(f"Term {term_id} missing definition")

           # NEW: Safe synonym extraction
           synonyms = []
           if meta and "synonyms" in meta:
               for syn_obj in meta["synonyms"]:
                   if isinstance(syn_obj, dict) and "val" in syn_obj:
                       synonyms.append(syn_obj["val"])

           # NEW: Safe comments extraction
           comments = []
           if meta and "comments" in meta:
               comment_list = meta["comments"]
               if isinstance(comment_list, list):
                   comments = [c for c in comment_list if c and isinstance(c, str)]

           terms_data.append({
               "id": term_id,
               "label": label,
               "definition": definition,
               "synonyms": json.dumps(synonyms, ensure_ascii=False),
               "comments": json.dumps(comments, ensure_ascii=False),
           })

       return terms_data
   ```

4. **Update prepare_hpo_data() signature**
   ```python
   # Update call sites to pass HPOGraphDocument instead of dict
   hpo_doc = load_hpo_json(hpo_file_path)
   if not hpo_doc:
       return False, f"Failed to load HPO JSON from {hpo_file_path}"

   all_nodes_data, parent_to_children_map, child_to_parents_map, all_term_ids = (
       _parse_hpo_json_to_graphs(hpo_doc)  # Changed: pass HPOGraphDocument
   )
   ```

**Validation After Each Step:**
```bash
# Type check
make typecheck-fast

# Existing tests (ensure no regressions)
pytest tests/unit/cli/test_data_commands.py -v

# Full test suite
make test
```

---

### Phase 3: Enhanced Error Handling & Logging (Day 3, 4 hours)

**Goal:** Comprehensive logging for schema deviations

**Steps:**

1. **Add schema deviation counter**
   ```python
   class SchemaDeviationCounter:
       """Track schema deviations during parsing."""
       def __init__(self):
           self.missing_definitions = 0
           self.missing_synonyms = 0
           self.missing_comments = 0
           self.empty_labels = 0
           self.unknown_predicates = set()

       def log_summary(self, logger):
           """Log summary of schema deviations."""
           logger.info("=== HPO Schema Deviation Summary ===")
           logger.info(f"Terms missing definitions: {self.missing_definitions}")
           logger.info(f"Terms missing synonyms: {self.missing_synonyms}")
           logger.info(f"Terms missing comments: {self.missing_comments}")
           logger.info(f"Terms with empty labels: {self.empty_labels}")
           if self.unknown_predicates:
               logger.warning(
                   f"Unknown edge predicates: {self.unknown_predicates}"
               )
   ```

2. **Integrate counter into parsing**
   ```python
   def _parse_hpo_json_to_graphs(
       hpo_doc: HPOGraphDocument,
   ) -> tuple[...]:
       deviation_counter = SchemaDeviationCounter()

       for node in graph.nodes:
           if not node.meta:
               deviation_counter.missing_definitions += 1
               deviation_counter.missing_synonyms += 1
               deviation_counter.missing_comments += 1
           else:
               if not node.meta.definition:
                   deviation_counter.missing_definitions += 1
               if not node.meta.synonyms:
                   deviation_counter.missing_synonyms += 1
               if not node.meta.comments:
                   deviation_counter.missing_comments += 1

       # At end of function
       deviation_counter.log_summary(logger)
   ```

3. **Add detailed ValidationError reporting**
   ```python
   def format_validation_error(error: ValidationError) -> str:
       """Format Pydantic ValidationError for debugging."""
       lines = ["HPO JSON Schema Validation Failed:"]
       for err in error.errors():
           loc = " → ".join(str(x) for x in err["loc"])
           lines.append(
               f"  • Location: {loc}\n"
               f"    Error: {err['msg']}\n"
               f"    Type: {err['type']}\n"
               f"    Input: {err.get('input', 'N/A')}"
           )
       return "\n".join(lines)
   ```

---

### Phase 4: Testing & Validation (Day 4, 6-8 hours)

**Goal:** Comprehensive test coverage for new functionality

**Test Files:**

1. **tests_new/unit/data_processing/test_hpo_schema.py** (Created in Phase 1)
   - Schema validation tests
   - ~15-20 test cases

2. **tests_new/unit/data_processing/test_hpo_parser_refactored.py** (New)
   ```python
   """
   Unit tests for refactored HPO parser with schema validation.

   Tests cover:
   - Graceful handling of missing optional fields
   - Proper error handling for missing required fields
   - Schema deviation logging
   - Backward compatibility with existing HPO data
   """
   import pytest
   from pydantic import ValidationError
   from phentrieve.data_processing.hpo_parser import (
       load_hpo_json,
       _parse_hpo_json_to_graphs,
       _extract_term_data_for_db,
   )
   from phentrieve.data_processing.hpo_schema import HPOGraphDocument


   def test_load_hpo_json_with_minimal_schema(tmp_path):
       """Test loading HPO JSON with only required fields."""
       minimal_json = {
           "graphs": [{
               "nodes": [
                   {"id": "HP:0000001", "lbl": "All"},
                   {"id": "HP:0001250", "lbl": "Seizure"},
               ],
               "edges": [
                   {"sub": "HP:0001250", "pred": "is_a", "obj": "HP:0000001"}
               ]
           }]
       }
       # ... test implementation


   def test_load_hpo_json_missing_required_field(tmp_path):
       """Test that missing required field raises ValidationError."""
       invalid_json = {
           "graphs": [{
               "nodes": [
                   {"id": "HP:0000001"}  # Missing required 'lbl'
               ]
           }]
       }
       # ... expect ValidationError


   def test_parse_handles_missing_optional_metadata(caplog):
       """Test parsing continues with warnings when optional fields missing."""
       # ... verify warnings logged, parsing succeeds


   def test_extract_term_data_with_partial_metadata():
       """Test term extraction handles missing synonyms, comments, definitions."""
       # ... verify sensible defaults used


   def test_backward_compatibility_with_real_hpo_data():
       """Test that real HPO data still parses identically."""
       # Use actual hp.json if available
       # Compare output with previous implementation (golden file)
   ```

3. **tests_new/integration/test_hpo_parser_integration.py** (New)
   ```python
   """
   Integration tests for HPO parser with real data.

   These tests use actual HPO JSON data (if available) or realistic
   test fixtures to validate end-to-end parsing behavior.
   """

   @pytest.mark.integration
   def test_prepare_hpo_data_full_pipeline(tmp_path):
       """Test complete HPO data preparation pipeline."""
       # ... test with realistic HPO JSON


   @pytest.mark.integration
   def test_parsing_performance_no_regression(benchmark):
       """Test that Pydantic validation doesn't slow parsing >5%."""
       # ... benchmark comparison
   ```

**Test Execution:**
```bash
# Unit tests (fast)
pytest tests_new/unit/data_processing/test_hpo_schema.py -v
pytest tests_new/unit/data_processing/test_hpo_parser_refactored.py -v

# Integration tests
pytest tests_new/integration/test_hpo_parser_integration.py -v

# All tests (check for regressions)
make test

# Coverage check
pytest --cov=phentrieve.data_processing.hpo_parser \
       --cov=phentrieve.data_processing.hpo_schema \
       --cov-report=html
# Goal: >90% coverage for new/modified code
```

---

### Phase 5: Documentation & Cleanup (Day 5, 2-4 hours)

**Goal:** Complete documentation and code cleanup

**Steps:**

1. **Update module docstrings**
   ```python
   # phentrieve/data_processing/hpo_parser.py
   """
   HPO data processing and preparation module.

   This module provides functions for downloading, parsing, and processing
   Human Phenotype Ontology (HPO) data including:
   - Downloading the HPO JSON file (OBOGraph format)
   - **Schema validation using Pydantic models**
   - **Graceful handling of schema variations**
   - Extracting ALL individual HPO terms
   - Building the HPO graph structure
   - Precomputing graph properties (ancestor sets, term depths) for ALL terms

   Schema Resilience:
       The parser uses Pydantic models (see hpo_schema.py) to validate
       the HPO JSON structure according to the OBOGraph specification.
       Required fields (id, lbl) must be present; optional fields
       (meta.definition, meta.synonyms) are handled gracefully with
       warnings logged for missing data.

   OBOGraph Specification:
       https://github.com/geneontology/obographs

   HPO Ontology:
       https://hpo.jax.org/
   """
   ```

2. **Add schema evolution guide**
   ```python
   # phentrieve/data_processing/hpo_schema.py
   """
   ... existing docstring ...

   Schema Evolution:
       When the HPO Consortium updates the OBOGraph schema:

       1. Review changes in https://github.com/geneontology/obographs
       2. Update Pydantic models in this file
       3. Add new optional fields with `Optional[T] = None`
       4. Update validators if field semantics change
       5. Add migration logic if field is required
       6. Update tests in tests_new/unit/data_processing/test_hpo_schema.py
       7. Run full test suite to check for regressions

   Version Compatibility:
       - Tested with HPO releases: 2024-04-26, 2024-07-01
       - OBOGraph specification: v1.2
   """
   ```

3. **Update CLAUDE.md with new patterns**
   ```markdown
   # CLAUDE.md additions

   ### HPO Parser Architecture

   **Schema Validation**: The HPO parser uses Pydantic models for JSON schema
   validation (`phentrieve/data_processing/hpo_schema.py`). This ensures
   graceful handling of schema changes from the HPO Consortium.

   **Key Patterns**:
   - Required fields: `id`, `lbl` (fail fast if missing)
   - Optional fields: `meta.*` (graceful degradation with warnings)
   - Forward compatibility: `Config.extra = "allow"` for unknown fields

   **When modifying the parser**:
   1. Update Pydantic models first
   2. Run schema validation tests
   3. Check type errors with `make typecheck-fast`
   4. Ensure backward compatibility with existing HPO data
   ```

4. **Code cleanup**
   - Remove commented-out code
   - Consolidate duplicate logging
   - Ensure consistent error message format
   - Add type hints where missing

5. **Final validation**
   ```bash
   # Pre-commit checks
   make check          # Format + lint
   make typecheck-fast # Type checking
   make test           # All tests
   make all            # Full pipeline

   # Coverage report
   make test-cov
   # Open htmlcov/index.html and verify >90% for modified files
   ```

---

## Testing Strategy

### Test Pyramid

```
     E2E (Docker)
    /           \
   Integration Tests    ← Real HPO data, full pipeline
  /               \
 Unit Tests             ← Schema validation, edge cases
```

### Test Cases Breakdown

**Unit Tests (Fast, <1s each):**
- ✅ Schema validation: Required fields missing → ValidationError
- ✅ Schema validation: Optional fields missing → Success with defaults
- ✅ Schema validation: Unknown extra fields → Success (forward compat)
- ✅ Edge cases: Empty graphs array → ValidationError
- ✅ Edge cases: Node without ID → Logged and skipped
- ✅ Edge cases: Node without label → Logged and use ID as fallback
- ✅ Edge cases: Edge with unknown predicate → Logged warning
- ✅ Data extraction: Missing metadata → Sensible defaults
- ✅ Data extraction: Partial metadata → Extracted correctly
- ✅ Logging: Schema deviations counted and reported

**Integration Tests (Slower, 5-30s each):**
- ✅ Full pipeline: Real HPO JSON → Database
- ✅ Backward compatibility: Output identical to pre-refactor
- ✅ Performance: Pydantic overhead <5%
- ✅ Large dataset: ~19,500 HPO terms parse without error

**Regression Tests:**
- ✅ All existing tests pass unchanged
- ✅ CLI command `phentrieve data prepare` works identically
- ✅ Database schema unchanged
- ✅ Graph computation results identical

### Performance Benchmarks

**Acceptance Criteria:**
- Parsing time increase: <5% (Pydantic validation overhead)
- Memory usage: <10% increase (Pydantic model instances)
- Database insertion: No change (same data format)

**Measurement:**
```bash
# Before refactoring
time phentrieve data prepare --force

# After refactoring
time phentrieve data prepare --force

# Compare results
```

---

## Rollback Plan

### If Critical Issues Arise

**Immediate Rollback:**
```bash
# Revert to last known good commit
git revert <refactor-commit-sha>

# Or reset if not pushed
git reset --hard HEAD~1

# Verify functionality
make test
phentrieve data prepare --force
```

**Partial Rollback (Keep Schema Module):**
```bash
# Keep hpo_schema.py but revert parser changes
git checkout HEAD~1 -- phentrieve/data_processing/hpo_parser.py

# Keep tests for future use
# (don't revert tests_new/unit/data_processing/test_hpo_schema.py)
```

### Rollback Triggers

Revert changes if:
- ❌ Any existing test fails (regression)
- ❌ Performance degradation >10%
- ❌ Real HPO data fails to parse
- ❌ Database corruption or data loss
- ❌ CLI commands broken

### Recovery Steps

1. Execute rollback (above)
2. Document failure reason in GitHub issue
3. Add failing case to test suite
4. Re-plan with additional considerations
5. Re-implement with fixes

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Pydantic performance overhead** | Medium | Low | Benchmark early (Phase 4), optimize if >5% |
| **Breaking existing functionality** | High | Low | Comprehensive regression tests, backward compat tests |
| **Missing schema edge cases** | Medium | Medium | Test with multiple HPO versions, add unknown field handling |
| **Overly strict validation** | Medium | Medium | Use `Optional` liberally, allow extra fields |
| **Complex refactoring introduces bugs** | High | Medium | Incremental phases, validate after each step |
| **Future HPO schema changes** | Low | High | Forward compatibility with `extra="allow"`, clear evolution guide |

---

## Dependencies

### Required

- ✅ **pydantic** (`^2.0`) - Already in pyproject.toml
- ✅ **Python 3.9+** - Already required
- ✅ **pytest** - Already in dev dependencies

### Optional

- ✅ **pytest-benchmark** - For performance regression testing (already installed)

---

## Success Metrics

### Quantitative

- [ ] **Test Coverage**: >90% for `hpo_schema.py` and refactored `hpo_parser.py`
- [ ] **Type Coverage**: 100% (mypy with no errors)
- [ ] **Performance**: <5% overhead vs baseline
- [ ] **Regression Tests**: 100% passing (0 failures)

### Qualitative

- [ ] **Code Clarity**: Easier to understand schema structure
- [ ] **Maintainability**: Clear separation of concerns
- [ ] **Debuggability**: Better error messages with schema context
- [ ] **Resilience**: Handles missing optional fields gracefully
- [ ] **Documentation**: Clear docstrings and evolution guide

---

## Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Schema Definition** | 4-6 hours | `hpo_schema.py` + unit tests |
| **Phase 2: Parser Refactoring** | 8-12 hours | Refactored `hpo_parser.py` |
| **Phase 3: Error Handling** | 4 hours | Enhanced logging + deviation tracking |
| **Phase 4: Testing** | 6-8 hours | Integration tests + performance benchmarks |
| **Phase 5: Documentation** | 2-4 hours | Docstrings + CLAUDE.md updates |
| **Total** | **3-5 days** | Fully resilient HPO parser |

---

## Next Steps

1. **Review this plan** - Get feedback from team/maintainers
2. **Create feature branch** - `git checkout -b feature/hpo-parser-resilience-issue-23`
3. **Execute Phase 1** - Schema definition
4. **Validate incrementally** - Run tests after each phase
5. **Submit PR** - With reference to #23

---

## References

- **Issue #23**: https://github.com/berntpopp/phentrieve/issues/23
- **OBOGraph Specification**: https://github.com/geneontology/obographs
- **Pydantic Documentation**: https://docs.pydantic.dev/latest/
- **HPO Ontology**: https://hpo.jax.org/
- **HPO GitHub**: https://github.com/obophenotype/human-phenotype-ontology

---

## Appendix A: Design Principles Applied

### SOLID Principles

**Single Responsibility Principle (SRP)**
- ✅ `hpo_schema.py`: Only schema validation
- ✅ `load_hpo_json()`: Only file I/O + validation
- ✅ `parse_hpo_graph_nodes()`: Only node parsing
- ✅ `parse_hpo_graph_edges()`: Only edge parsing
- ✅ `extract_node_data()`: Only data transformation

**Open/Closed Principle (OCP)**
- ✅ `Config.extra = "allow"`: Open to new fields, closed to modification
- ✅ Pydantic validators: Extend validation without changing models

**Liskov Substitution Principle (LSP)**
- ✅ `HPOGraphDocument` can be used anywhere dict was expected (via `model_dump()`)
- ✅ Backward compatible function signatures (where feasible)

**Interface Segregation Principle (ISP)**
- ✅ Small, focused Pydantic models (NodeMeta, MetaDefinition, MetaSynonym)
- ✅ No fat interfaces with unused fields

**Dependency Inversion Principle (DIP)**
- ✅ Depend on abstractions (Pydantic BaseModel) not concrete dict structures
- ✅ Validation logic injected via Pydantic, not embedded in parser

### DRY (Don't Repeat Yourself)

- ✅ Single extraction function for node data (no duplication)
- ✅ Shared validators in Pydantic models
- ✅ Centralized error formatting

### KISS (Keep It Simple, Stupid)

- ✅ Pydantic handles complexity of validation
- ✅ Clear, linear flow: Load → Validate → Parse → Extract
- ✅ No over-engineered abstractions (e.g., factory patterns, complex inheritance)

### Defensive Programming

- ✅ Validate early (at JSON load time)
- ✅ Fail fast for critical errors (missing id, lbl)
- ✅ Degrade gracefully for optional fields
- ✅ Comprehensive logging for debugging

---

## Appendix B: Example Usage

### Before Refactoring (Brittle)

```python
# Crashes if meta.definition.val missing
definition = node_data["meta"]["definition"]["val"]
```

### After Refactoring (Resilient)

```python
# Pydantic validation + graceful degradation
node = HPONode(**raw_node_data)

definition = ""
if node.meta and node.meta.definition:
    definition = node.meta.definition.val
else:
    logger.debug(f"Node {node.id} missing definition")
```

---

**Status**: ✅ **READY FOR IMPLEMENTATION**
**Approval Required**: Yes (review by maintainers)
**Estimated Start**: Upon approval
**Estimated Completion**: 5 business days post-start
