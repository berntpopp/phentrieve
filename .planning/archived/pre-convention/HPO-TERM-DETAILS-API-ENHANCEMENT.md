# HPO Term Details in API Responses - Refactoring Plan

**Status:** Active
**Created:** 2025-11-20
**Priority:** High
**Related Issue:** [#24](https://github.com/berntpopp/phentrieve/issues/24)
**Estimated Complexity:** Medium

---

## Executive Summary

Enhance API responses to include optional HPO term details (definitions and synonyms) without impacting performance. This feature enables richer client experiences while maintaining backward compatibility through an opt-in flag.

**Key Principle:** Build incrementally with modular, testable components following DRY, KISS, and SOLID principles.

---

## Table of Contents

1. [Objective](#objective)
2. [Success Criteria](#success-criteria)
3. [Current State Analysis](#current-state-analysis)
4. [Architecture & Design Principles](#architecture--design-principles)
5. [Implementation Plan](#implementation-plan)
6. [Testing Strategy](#testing-strategy)
7. [UI/UX Design](#uiux-design)
8. [Performance Considerations](#performance-considerations)
9. [Rollback Plan](#rollback-plan)
10. [References](#references)

---

## Objective

Enable API consumers to optionally retrieve detailed HPO term information (definitions and synonyms) through a single request, eliminating the need for separate lookups and improving client-side efficiency.

**Core Requirements:**
- ✅ Add `include_details` parameter to `QueryRequest` (default: `false`)
- ✅ Extend `HPOResultItem` with optional `definition` and `synonyms` fields
- ✅ Fetch details efficiently from SQLite database (`hpo_data.db`)
- ✅ Maintain backward compatibility (no breaking changes)
- ✅ Preserve acceptable response times (< 300ms for 10 results with details)
- ✅ Implement in Python CLI/library first, then API
- ✅ Add elegant UI controls in Vue frontend (expandable panels)

---

## Success Criteria

### Backend (CLI + API)
- [ ] Database layer has efficient batch lookup method (`get_terms_by_ids`)
- [ ] CLI supports `--include-details` flag with formatted output
- [ ] API request schema includes `include_details: Optional[bool] = False`
- [ ] API response schema includes optional `definition` and `synonyms` fields
- [ ] Details are `null`/omitted when `include_details=false` (default)
- [ ] All existing tests pass without modification
- [ ] New tests achieve ≥90% coverage for new code paths
- [ ] Performance benchmarks show < 100ms overhead for details lookup (10 terms)
- [ ] Type checking passes with 0 mypy errors

### Frontend
- [ ] Options panel includes "Show term details" toggle
- [ ] Results display uses Vuetify expansion panels for details
- [ ] Definition displayed in readable format
- [ ] Synonyms shown as comma-separated chips
- [ ] Toggle state persists in Pinia store
- [ ] Loading indicators during API calls
- [ ] Responsive design works on mobile
- [ ] i18n support for all new UI text (EN, DE, ES, FR, NL)

### Documentation & Quality
- [ ] CLAUDE.md updated with new API parameters
- [ ] API docs (OpenAPI/Swagger) reflect schema changes
- [ ] CLI `--help` includes details about new flag
- [ ] Code follows existing style conventions (Ruff, ESLint)
- [ ] No antipatterns introduced (checked in code review)

---

## Current State Analysis

### Database Schema (Already Optimal)

**Table:** `hpo_terms`
```sql
CREATE TABLE IF NOT EXISTS hpo_terms (
    id TEXT PRIMARY KEY,              -- HP:0000123
    label TEXT NOT NULL,
    definition TEXT,                  -- Already exists!
    synonyms TEXT,                    -- Already exists! (JSON array)
    comments TEXT,                    -- JSON array
    created_at TEXT DEFAULT (datetime('now'))
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_hpo_terms_label ON hpo_terms(label);
```

**Key Findings:**
- ✅ Definition and synonyms fields already exist in database
- ✅ Synonyms stored as JSON array: `["syn1", "syn2"]`
- ✅ Index on `label` exists, but no index on `id` needed (PRIMARY KEY is implicit)
- ✅ Database connection uses `Row` factory for dict-like access
- ✅ Connection configured for multi-threading (`check_same_thread=False`)

**Existing Database Methods:**
- `load_all_terms()` - Loads ALL terms (inefficient for API)
- `get_label_map()` - Optimized for labels only
- ⚠️ **MISSING**: `get_terms_by_ids(ids: list[str])` - Batch lookup needed

### API Architecture (Schema Updates Required)

**Current Schema:** `api/schemas/query_schemas.py`

```python
# QueryRequest - needs one field
class QueryRequest(BaseModel):
    text: str = Field(...)
    model_name: Optional[str] = None
    # ... other fields ...
    # MISSING: include_details field

# HPOResultItem - needs two fields
class HPOResultItem(BaseModel):
    hpo_id: str
    label: str
    similarity: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    original_rank: Optional[int] = None
    # MISSING: definition and synonyms fields
```

**API Endpoint:** `api/routers/query_router.py`
- `POST /` - Main query endpoint (needs details enrichment logic)
- Calls `execute_hpo_retrieval_for_api()` in `phentrieve/retrieval/api_helpers.py`
- Returns `QueryResponse` with list of `HPOResultItem`

**Data Flow:**
1. Request → `run_hpo_query()` endpoint
2. → `execute_hpo_retrieval_for_api()` helper
3. → `DenseRetriever.query()` returns IDs + labels + scores
4. → Optional reranking with cross-encoder
5. → Format results as `HPOResultItem` list
6. ⚠️ **NEW**: Enrich with details if `include_details=true`
7. → Return `QueryResponse`

### Frontend Architecture (New UI Controls Needed)

**Components:**
- `frontend/src/components/QueryInterface.vue` - Query form (add toggle)
- `frontend/src/components/ResultsDisplay.vue` - Results list (add expansion panels)
- `frontend/src/components/SimilarityScore.vue` - Score chips (reuse)

**State Management (Pinia):**
- Store location: `frontend/src/stores/` (likely new `settings` store)
- Persist toggle state across queries

**Vuetify Components:**
- `v-expansion-panels` - For expandable term details
- `v-chip` - For synonyms display
- `v-switch` or `v-checkbox` - For toggle in options

---

## Architecture & Design Principles

### 1. Single Responsibility Principle (SOLID)

**Database Layer:**
- `HPODatabase` handles ONLY database operations
- New method: `get_terms_by_ids()` - single responsibility: batch fetch

**API Layer:**
- Request schema validates input
- Response schema validates output
- Business logic in dedicated helper function

**Frontend Layer:**
- `QueryInterface` handles form inputs
- `ResultsDisplay` handles result rendering
- Separate settings store for UI preferences

### 2. Don't Repeat Yourself (DRY)

**Shared Logic:**
- Database lookup logic centralized in `HPODatabase`
- Details enrichment as reusable function: `enrich_with_details()`
- Frontend expansion panel as reusable sub-component

**Configuration:**
- Default `include_details=False` defined once in schema
- UI toggle default from Pinia store (single source of truth)

### 3. Keep It Simple, Stupid (KISS)

**Simple Solutions:**
- Use existing database schema (no migrations)
- Straightforward SQL query with `IN` clause
- Simple conditional logic: `if include_details: add_details()`
- Standard Vuetify components (no custom CSS needed)

**Avoid Over-Engineering:**
- ❌ NO caching layer (premature optimization)
- ❌ NO new microservice (monolith sufficient)
- ❌ NO GraphQL (REST sufficient)
- ❌ NO complex query builder (raw SQL clear)

### 4. Open/Closed Principle (SOLID)

**Extension Points:**
- New optional fields can be added without modifying core logic
- Details enrichment function accepts any ID list
- Frontend expansion panel accepts any detail structure

**Closed for Modification:**
- Core retrieval logic unchanged
- Existing API contracts unchanged (backward compatible)
- Database schema unchanged

### 5. Modularization

**Module Boundaries:**
```
phentrieve/
  data_processing/
    hpo_database.py              # NEW: get_terms_by_ids()
  retrieval/
    api_helpers.py               # NEW: enrich_with_details()

api/
  schemas/
    query_schemas.py             # MODIFY: Add fields
  routers/
    query_router.py              # MODIFY: Pass include_details flag

frontend/
  src/
    components/
      QueryInterface.vue         # MODIFY: Add toggle
      ResultsDisplay.vue         # MODIFY: Add expansion panels
      TermDetailsPanel.vue       # NEW: Reusable detail display
    stores/
      settings.ts                # NEW: UI preferences store
```

---

## Implementation Plan

### Phase 1: Database Layer (Foundation)

**Estimated Time:** 2 hours

#### Step 1.1: Add Batch Lookup Method

**File:** `phentrieve/data_processing/hpo_database.py`

**Implementation:**
```python
def get_terms_by_ids(self, term_ids: list[str]) -> dict[str, dict[str, Any]]:
    """
    Efficiently fetch multiple HPO terms by IDs in a single query.

    Args:
        term_ids: List of HPO term IDs (e.g., ["HP:0001250", "HP:0002119"])

    Returns:
        Dictionary mapping {term_id: term_data} where term_data includes:
        - id: str
        - label: str
        - definition: str (empty string if missing)
        - synonyms: list[str] (deserialized from JSON)
        - comments: list[str] (deserialized from JSON)

    Performance:
        - Single SQL query using IN clause (O(n) where n = len(term_ids))
        - Returns empty dict if term_ids is empty (short-circuit)
        - Handles missing terms gracefully (skips, logs warning)

    Example:
        >>> db = HPODatabase(Path("hpo_data.db"))
        >>> terms = db.get_terms_by_ids(["HP:0001250", "HP:0002119"])
        >>> terms["HP:0001250"]["definition"]
        "Recurrent seizures affecting..."
    """
    if not term_ids:
        logger.debug("Empty term_ids list provided to get_terms_by_ids")
        return {}

    conn = self.get_connection()

    # Use parameterized query for SQL injection safety
    placeholders = ",".join("?" * len(term_ids))
    query = f"""
        SELECT id, label, definition, synonyms, comments
        FROM hpo_terms
        WHERE id IN ({placeholders})
    """

    cursor = conn.execute(query, term_ids)

    terms_map = {}
    for row in cursor:
        term_data = dict(row)
        # Deserialize JSON fields (consistent with load_all_terms)
        term_data["synonyms"] = json.loads(term_data["synonyms"] or "[]")
        term_data["comments"] = json.loads(term_data["comments"] or "[]")
        terms_map[term_data["id"]] = term_data

    # Log warning for any missing terms
    found_ids = set(terms_map.keys())
    missing_ids = set(term_ids) - found_ids
    if missing_ids:
        logger.warning(
            f"Terms not found in database: {sorted(missing_ids)[:5]}"
            f"{'...' if len(missing_ids) > 5 else ''}"
        )

    logger.debug(
        f"Fetched {len(terms_map)}/{len(term_ids)} terms from database"
    )
    return terms_map
```

**Testing:**
```python
# tests/test_hpo_database.py (new test)
def test_get_terms_by_ids_batch_lookup(hpo_db_with_sample_data):
    """Test efficient batch lookup of multiple terms."""
    term_ids = ["HP:0000001", "HP:0000118", "HP:0000707"]

    result = hpo_db_with_sample_data.get_terms_by_ids(term_ids)

    assert len(result) == 3
    assert "HP:0000001" in result
    assert result["HP:0000001"]["label"] == "All"
    assert isinstance(result["HP:0000001"]["synonyms"], list)
    assert isinstance(result["HP:0000001"]["definition"], str)

def test_get_terms_by_ids_empty_list():
    """Test empty input returns empty dict."""
    db = HPODatabase(":memory:")
    db.initialize_schema()

    result = db.get_terms_by_ids([])

    assert result == {}

def test_get_terms_by_ids_missing_terms(hpo_db_with_sample_data):
    """Test handling of non-existent term IDs."""
    term_ids = ["HP:0000001", "HP:9999999"]  # One valid, one invalid

    result = hpo_db_with_sample_data.get_terms_by_ids(term_ids)

    assert len(result) == 1
    assert "HP:0000001" in result
    assert "HP:9999999" not in result
```

**Performance Benchmark:**
```python
# tests/benchmark_database.py (optional, for documentation)
import time

def benchmark_get_terms_by_ids():
    """Benchmark batch vs individual lookups."""
    db = HPODatabase(Path("data/hpo_data.db"))
    term_ids = ["HP:{:07d}".format(i) for i in range(1, 101)]  # 100 terms

    # Batch lookup
    start = time.perf_counter()
    batch_result = db.get_terms_by_ids(term_ids)
    batch_time = time.perf_counter() - start

    print(f"Batch lookup (100 terms): {batch_time*1000:.2f}ms")
    # Expected: < 50ms
```

---

### Phase 2: CLI Integration (Library Layer)

**Estimated Time:** 3 hours

#### Step 2.1: Add Details Enrichment Helper

**File:** `phentrieve/retrieval/details_enrichment.py` (NEW)

**Purpose:** Reusable function for enriching results with details (used by CLI and API)

```python
"""
HPO term details enrichment utilities.

Provides reusable functions for adding definitions and synonyms to query results.
"""

import logging
from pathlib import Path
from typing import Any

from phentrieve.config import DEFAULT_HPO_DB_FILENAME
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.utils import get_default_data_dir, resolve_data_path

logger = logging.getLogger(__name__)


def enrich_results_with_details(
    results: list[dict[str, Any]],
    data_dir_override: str | None = None,
) -> list[dict[str, Any]]:
    """
    Enrich HPO query results with definitions and synonyms from database.

    This function takes a list of query results (containing at minimum hpo_id
    and label fields) and enriches them with definition and synonyms fields
    by performing a single batch lookup against the HPO database.

    Args:
        results: List of result dicts, each containing at minimum:
                 - hpo_id: str (e.g., "HP:0001250")
                 - label: str
                 Additional fields (similarity, scores) are preserved.
        data_dir_override: Optional override for data directory path

    Returns:
        Enriched results list with added fields:
        - definition: str | None (None if missing in database)
        - synonyms: list[str] | None (None if missing in database)

        Original fields are preserved. Order is maintained.

    Performance:
        - Single database query for all terms (batch lookup)
        - O(n) where n = len(results)
        - Typical: < 50ms for 10-50 terms

    Error Handling:
        - If database unavailable, logs error and returns results unchanged
        - If term not found, adds None for definition/synonyms
        - Never raises exceptions (graceful degradation)

    Example:
        >>> results = [
        ...     {"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95},
        ...     {"hpo_id": "HP:0002119", "label": "Ventriculomegaly", "similarity": 0.89}
        ... ]
        >>> enriched = enrich_results_with_details(results)
        >>> enriched[0]["definition"]
        "A seizure is an intermittent abnormality..."
        >>> enriched[0]["synonyms"]
        ["Seizures", "Epileptic seizure"]
    """
    if not results:
        logger.debug("Empty results list, skipping details enrichment")
        return results

    # Extract HPO IDs from results
    hpo_ids = [result["hpo_id"] for result in results]

    # Resolve database path
    try:
        data_dir = resolve_data_path(
            data_dir_override, "data_dir", get_default_data_dir
        )
        db_path = data_dir / DEFAULT_HPO_DB_FILENAME

        if not db_path.exists():
            logger.error(
                f"HPO database not found at {db_path}. "
                "Details enrichment skipped. "
                "Run 'phentrieve data prepare' to generate database."
            )
            # Add None fields gracefully
            for result in results:
                result["definition"] = None
                result["synonyms"] = None
            return results

        # Batch lookup from database
        db = HPODatabase(db_path)
        terms_map = db.get_terms_by_ids(hpo_ids)
        db.close()

        # Enrich results with details
        for result in results:
            hpo_id = result["hpo_id"]
            term_data = terms_map.get(hpo_id)

            if term_data:
                # Add definition (empty string → None for API clarity)
                definition = term_data.get("definition", "")
                result["definition"] = definition if definition else None

                # Add synonyms (empty list → None for API clarity)
                synonyms = term_data.get("synonyms", [])
                result["synonyms"] = synonyms if synonyms else None
            else:
                # Term not found in database
                result["definition"] = None
                result["synonyms"] = None
                logger.warning(
                    f"Term {hpo_id} not found in database during enrichment"
                )

        logger.debug(
            f"Enriched {len(results)} results with details "
            f"({len(terms_map)} terms found in database)"
        )
        return results

    except Exception as e:
        logger.error(
            f"Error during details enrichment: {e}", exc_info=True
        )
        # Graceful degradation: add None fields
        for result in results:
            result.setdefault("definition", None)
            result.setdefault("synonyms", None)
        return results
```

**Testing:**
```python
# tests_new/unit/test_details_enrichment.py (NEW)
import pytest
from phentrieve.retrieval.details_enrichment import enrich_results_with_details


def test_enrich_empty_results():
    """Test enrichment with empty results list."""
    result = enrich_results_with_details([])
    assert result == []


def test_enrich_with_valid_terms(tmp_path, create_test_hpo_db):
    """Test successful enrichment with valid HPO terms."""
    # Setup test database
    db_path = tmp_path / "hpo_data.db"
    create_test_hpo_db(
        db_path,
        terms=[
            {
                "id": "HP:0001250",
                "label": "Seizure",
                "definition": "A seizure is an abnormal...",
                "synonyms": '["Seizures", "Epileptic seizure"]',
                "comments": "[]",
            }
        ],
    )

    # Test enrichment
    results = [
        {"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95}
    ]

    enriched = enrich_results_with_details(
        results, data_dir_override=str(tmp_path)
    )

    assert len(enriched) == 1
    assert enriched[0]["definition"] == "A seizure is an abnormal..."
    assert enriched[0]["synonyms"] == ["Seizures", "Epileptic seizure"]
    assert enriched[0]["similarity"] == 0.95  # Original fields preserved


def test_enrich_with_missing_term(tmp_path, create_test_hpo_db):
    """Test enrichment when term not in database."""
    db_path = tmp_path / "hpo_data.db"
    create_test_hpo_db(db_path, terms=[])  # Empty database

    results = [
        {"hpo_id": "HP:9999999", "label": "Unknown", "similarity": 0.5}
    ]

    enriched = enrich_results_with_details(
        results, data_dir_override=str(tmp_path)
    )

    assert enriched[0]["definition"] is None
    assert enriched[0]["synonyms"] is None


def test_enrich_without_database(tmp_path):
    """Test graceful degradation when database doesn't exist."""
    results = [
        {"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95}
    ]

    enriched = enrich_results_with_details(
        results, data_dir_override=str(tmp_path)  # No database here
    )

    # Should return results with None fields
    assert enriched[0]["definition"] is None
    assert enriched[0]["synonyms"] is None
```

#### Step 2.2: Add CLI Flag and Output Formatting

**File:** `phentrieve/cli/query_commands.py` (or main CLI file)

**Changes:**
```python
@app.command()
def query(
    text: str = typer.Argument(..., help="Clinical text to query"),
    # ... existing parameters ...
    include_details: bool = typer.Option(
        False,
        "--include-details",
        "-d",
        help="Include HPO term definitions and synonyms in output",
    ),
):
    """
    Query HPO terms from clinical text.

    Examples:
        # Basic query
        phentrieve query "patient has seizures"

        # Query with details
        phentrieve query "patient has seizures" --include-details
    """
    # ... existing query logic ...

    # After getting results from retrieval
    if include_details:
        from phentrieve.retrieval.details_enrichment import (
            enrich_results_with_details,
        )
        results = enrich_results_with_details(results, data_dir_override)

    # Format output (update formatter to handle details)
    format_results_for_cli(results, include_details=include_details)
```

**File:** `phentrieve/retrieval/output_formatters.py`

**Update existing formatter:**
```python
def format_results_for_cli(
    results: list[dict[str, Any]],
    include_details: bool = False,
) -> None:
    """
    Format query results for CLI display.

    Args:
        results: List of HPO result dictionaries
        include_details: Whether to display definitions and synonyms
    """
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, result in enumerate(results, 1):
        # Existing output
        console.print(f"\n[bold cyan]{i}. {result['hpo_id']}[/bold cyan]")
        console.print(f"   Label: {result['label']}")

        if result.get("similarity"):
            console.print(f"   Similarity: {result['similarity']:.3f}")

        # NEW: Details section
        if include_details:
            definition = result.get("definition")
            if definition:
                console.print(f"\n   [dim]Definition:[/dim]")
                console.print(f"   {definition}")

            synonyms = result.get("synonyms")
            if synonyms:
                console.print(f"\n   [dim]Synonyms:[/dim]")
                console.print(f"   {', '.join(synonyms)}")
```

**Testing:**
```bash
# Manual testing commands
phentrieve query "patient has seizures" --include-details
phentrieve query "fever and headache" -d  # Short flag

# Expected output format:
# 1. HP:0001250
#    Label: Seizure
#    Similarity: 0.954
#
#    Definition:
#    A seizure is an intermittent abnormality of nervous system physiology...
#
#    Synonyms:
#    Seizures, Epileptic seizure
```

---

### Phase 3: API Integration (Backend)

**Estimated Time:** 2 hours

#### Step 3.1: Update API Schemas

**File:** `api/schemas/query_schemas.py`

**Changes:**
```python
class QueryRequest(BaseModel):
    """Request schema for HPO term query."""

    text: str = Field(
        ..., min_length=1, description="Clinical text to query for HPO terms."
    )
    # ... existing fields ...

    # NEW FIELD
    include_details: bool = Field(
        False,
        description=(
            "Include HPO term definitions and synonyms in the response. "
            "Default is False for performance. "
            "When True, adds 'definition' and 'synonyms' fields to each result."
        ),
    )


class HPOResultItem(BaseModel):
    """Single HPO term result."""

    hpo_id: str
    label: str
    similarity: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    original_rank: Optional[int] = None

    # NEW FIELDS (Optional for backward compatibility)
    definition: Optional[str] = Field(
        None,
        description=(
            "HPO term definition. Only populated when "
            "include_details=true in request."
        ),
    )
    synonyms: Optional[list[str]] = Field(
        None,
        description=(
            "List of term synonyms. Only populated when "
            "include_details=true in request."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.954,
                "cross_encoder_score": 0.876,
                "original_rank": 1,
                "definition": "A seizure is an intermittent abnormality...",
                "synonyms": ["Seizures", "Epileptic seizure"],
            }
        }
```

**Backward Compatibility Check:**
- ✅ New fields are `Optional` with `None` default
- ✅ Pydantic will omit `None` fields from JSON (if `exclude_none=True`)
- ✅ Existing clients unaffected (new field defaults to `False`)
- ✅ OpenAPI docs will show new fields as optional

#### Step 3.2: Update API Endpoint Logic

**File:** `phentrieve/retrieval/api_helpers.py`

**Update `execute_hpo_retrieval_for_api()` function:**
```python
async def execute_hpo_retrieval_for_api(
    text: str,
    language: str,
    retriever: DenseRetriever,
    num_results: int,
    similarity_threshold: float,
    enable_reranker: bool,
    cross_encoder: Optional[CrossEncoder],
    rerank_count: int,
    reranker_mode: str,
    translation_dir_path: Optional[str],
    detect_query_assertion: bool = True,
    query_assertion_language: Optional[str] = None,
    query_assertion_preference: str = "dependency",
    include_details: bool = False,  # NEW PARAMETER
    debug: bool = False,
) -> dict[str, Any]:
    """
    Execute HPO term retrieval for API requests.

    Args:
        ... (existing args) ...
        include_details: Whether to enrich results with definitions/synonyms

    Returns:
        Dictionary with query results including optional details
    """
    # ... existing logic ...

    # Format results as HPOResultItem compatible structure
    formatted_results = []
    for item in hpo_embeddings_results:
        result_item = {
            "hpo_id": item["hpo_id"],
            "label": item["label"],
            "similarity": item["similarity"],
        }

        # Add reranking info if available
        if "cross_encoder_score" in item:
            result_item["cross_encoder_score"] = item["cross_encoder_score"]
        if "original_rank" in item:
            result_item["original_rank"] = item["original_rank"]

        formatted_results.append(result_item)

    # NEW: Enrich with details if requested
    if include_details:
        from phentrieve.retrieval.details_enrichment import (
            enrich_results_with_details,
        )
        formatted_results = enrich_results_with_details(formatted_results)

    # Create result dictionary
    result_dict = {
        "query_text_processed": segment_to_process,
        "results": formatted_results,
        "original_query_assertion_status": (
            original_query_assertion_status.value
            if original_query_assertion_status
            else None
        ),
    }

    return result_dict
```

**File:** `api/routers/query_router.py`

**Update endpoint to pass flag:**
```python
@router.post("/", response_model=QueryResponse)
async def run_hpo_query(
    request: QueryRequest,
    retriever: DenseRetriever = Depends(get_retriever_for_request),
):
    """Execute an HPO term query with full control over parameters."""
    # ... existing logic ...

    # Call the core HPO retrieval logic
    query_results_dict = await execute_hpo_retrieval_for_api(
        text=request.text,
        language=language_to_use,
        retriever=retriever,
        num_results=request.num_results,
        similarity_threshold=request.similarity_threshold,
        enable_reranker=request.enable_reranker
        and (cross_encoder_instance is not None),
        cross_encoder=cross_encoder_instance,
        rerank_count=request.rerank_count,
        reranker_mode=request.reranker_mode,
        translation_dir_path=resolved_translation_dir,
        detect_query_assertion=request.detect_query_assertion,
        query_assertion_language=request.query_assertion_language,
        query_assertion_preference=request.query_assertion_preference,
        include_details=request.include_details,  # NEW PARAMETER
        debug=False,
    )

    # ... rest of existing logic ...
```

**Testing:**
```python
# tests_new/unit/test_api_query_endpoint.py (update existing tests)

def test_query_without_details(test_client):
    """Test query with include_details=false (default)."""
    response = test_client.post(
        "/api/query/",
        json={
            "text": "patient has seizures",
            "num_results": 5,
            # include_details omitted (defaults to false)
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Results should NOT have definition/synonyms
    assert len(data["results"]) > 0
    assert "hpo_id" in data["results"][0]
    assert "label" in data["results"][0]
    assert "definition" not in data["results"][0] or data["results"][0]["definition"] is None
    assert "synonyms" not in data["results"][0] or data["results"][0]["synonyms"] is None


def test_query_with_details(test_client):
    """Test query with include_details=true."""
    response = test_client.post(
        "/api/query/",
        json={
            "text": "patient has seizures",
            "num_results": 5,
            "include_details": True,  # NEW PARAMETER
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Results SHOULD have definition/synonyms (or None if missing)
    assert len(data["results"]) > 0
    result = data["results"][0]

    assert "hpo_id" in result
    assert "label" in result
    assert "definition" in result  # Present (may be None)
    assert "synonyms" in result  # Present (may be None)

    # At least one result should have non-null details (assuming good test data)
    has_definition = any(r.get("definition") for r in data["results"])
    has_synonyms = any(r.get("synonyms") for r in data["results"])
    assert has_definition or has_synonyms  # At least one should have details


def test_query_details_performance(test_client, benchmark):
    """Benchmark performance overhead of details enrichment."""

    def query_with_details():
        response = test_client.post(
            "/api/query/",
            json={
                "text": "patient has seizures",
                "num_results": 10,
                "include_details": True,
            },
        )
        return response.json()

    result = benchmark(query_with_details)

    # Assert performance (adjust based on actual benchmarks)
    assert benchmark.stats["mean"] < 0.3  # < 300ms average
```

---

### Phase 4: Frontend Integration (UI/UX)

**Estimated Time:** 4 hours

#### Step 4.1: Add Settings Store for UI Preferences

**File:** `frontend/src/stores/settings.ts` (NEW)

```typescript
/**
 * Settings store for user preferences
 *
 * Manages UI preferences like term details visibility.
 * State is persisted to localStorage for user convenience.
 */

import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export const useSettingsStore = defineStore('settings', () => {
  // State
  const showTermDetails = ref<boolean>(false)

  // Load from localStorage on initialization
  const loadSettings = () => {
    try {
      const saved = localStorage.getItem('phentrieve_show_term_details')
      if (saved !== null) {
        showTermDetails.value = JSON.parse(saved)
      }
    } catch (e) {
      console.warn('Failed to load settings from localStorage:', e)
    }
  }

  // Watch for changes and persist to localStorage
  watch(showTermDetails, (newValue) => {
    try {
      localStorage.setItem('phentrieve_show_term_details', JSON.stringify(newValue))
    } catch (e) {
      console.warn('Failed to save settings to localStorage:', e)
    }
  })

  // Initialize on store creation
  loadSettings()

  return {
    showTermDetails,
  }
})
```

**Testing:**
```typescript
// frontend/src/stores/__tests__/settings.spec.ts (NEW)

import { setActivePinia, createPinia } from 'pinia'
import { useSettingsStore } from '../settings'

describe('Settings Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  it('initializes with default value', () => {
    const store = useSettingsStore()
    expect(store.showTermDetails).toBe(false)
  })

  it('persists changes to localStorage', () => {
    const store = useSettingsStore()

    store.showTermDetails = true

    const saved = localStorage.getItem('phentrieve_show_term_details')
    expect(JSON.parse(saved!)).toBe(true)
  })

  it('loads persisted value on initialization', () => {
    // Pre-populate localStorage
    localStorage.setItem('phentrieve_show_term_details', 'true')

    const store = useSettingsStore()

    expect(store.showTermDetails).toBe(true)
  })
})
```

#### Step 4.2: Add Toggle to Query Interface

**File:** `frontend/src/components/QueryInterface.vue`

**Changes:**
```vue
<template>
  <v-card>
    <!-- Existing query form -->

    <!-- NEW: Options Section (Collapsible) -->
    <v-expansion-panels v-model="optionsPanelOpen" class="mt-2">
      <v-expansion-panel>
        <v-expansion-panel-title>
          <v-icon start>mdi-cog</v-icon>
          {{ $t('queryInterface.optionsTitle', 'Query Options') }}
        </v-expansion-panel-title>

        <v-expansion-panel-text>
          <!-- Existing options (if any) -->

          <!-- NEW: Term Details Toggle -->
          <v-switch
            v-model="settingsStore.showTermDetails"
            color="primary"
            :label="$t('queryInterface.showTermDetails', 'Show term details (definitions & synonyms)')"
            :hint="$t('queryInterface.showTermDetailsHint', 'Includes detailed information in search results. May slightly increase response time.')"
            persistent-hint
            hide-details="auto"
          />
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <!-- Existing submit button -->
  </v-card>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useSettingsStore } from '@/stores/settings'

const settingsStore = useSettingsStore()
const optionsPanelOpen = ref<number | undefined>(undefined) // Collapsed by default

// Update API call to include details flag
const submitQuery = async () => {
  // ... existing logic ...

  const requestBody = {
    text: queryText.value,
    num_results: numResults.value,
    // ... other parameters ...
    include_details: settingsStore.showTermDetails, // NEW PARAMETER
  }

  const response = await fetch('/api/query/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  })

  // ... handle response ...
}
</script>
```

#### Step 4.3: Add Expandable Details in Results Display

**File:** `frontend/src/components/TermDetailsPanel.vue` (NEW)

```vue
<template>
  <div v-if="hasDetails" class="term-details pa-3">
    <!-- Definition Section -->
    <div v-if="definition" class="mb-3">
      <div class="text-subtitle-2 text-medium-emphasis mb-1">
        <v-icon size="small" class="mr-1">mdi-text-box</v-icon>
        {{ $t('termDetails.definition', 'Definition') }}
      </div>
      <div class="text-body-2 text-justify">
        {{ definition }}
      </div>
    </div>

    <!-- Synonyms Section -->
    <div v-if="synonyms && synonyms.length > 0" class="mb-2">
      <div class="text-subtitle-2 text-medium-emphasis mb-1">
        <v-icon size="small" class="mr-1">mdi-tag-multiple</v-icon>
        {{ $t('termDetails.synonyms', 'Synonyms') }}
      </div>
      <div class="d-flex flex-wrap gap-1">
        <v-chip
          v-for="(synonym, index) in synonyms"
          :key="index"
          size="small"
          color="grey-lighten-2"
          label
        >
          {{ synonym }}
        </v-chip>
      </div>
    </div>

    <!-- No Details Available Message -->
    <div v-if="!definition && (!synonyms || synonyms.length === 0)" class="text-caption text-medium-emphasis">
      <v-icon size="small" class="mr-1">mdi-information-outline</v-icon>
      {{ $t('termDetails.noDetails', 'No additional details available for this term.') }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  definition?: string | null
  synonyms?: string[] | null
}

const props = defineProps<Props>()

const hasDetails = computed(() => {
  return Boolean(props.definition || (props.synonyms && props.synonyms.length > 0))
})
</script>

<style scoped>
.term-details {
  background-color: rgba(var(--v-theme-surface-variant), 0.3);
  border-radius: 4px;
}
</style>
```

**File:** `frontend/src/components/ResultsDisplay.vue`

**Changes:**
```vue
<template>
  <!-- Existing results info card -->

  <v-list lines="two" class="rounded-lg mt-2">
    <!-- NEW: Use expansion panels instead of plain list items -->
    <v-expansion-panels
      v-if="settingsStore.showTermDetails"
      variant="accordion"
      class="mb-1"
    >
      <v-expansion-panel
        v-for="(result, index) in responseData.results"
        :key="result.hpo_id"
        class="mb-1"
      >
        <!-- Panel Header (existing result display) -->
        <v-expansion-panel-title
          :class="collectedPhenotypeIds.has(result.hpo_id) ? 'bg-primary-lighten-5' : 'bg-grey-lighten-5'"
        >
          <template #default="{ expanded }">
            <div class="d-flex align-center w-100">
              <!-- Rank Badge -->
              <v-badge :content="index + 1" color="primary" inline class="mr-3" />

              <!-- Term Info -->
              <div class="flex-grow-1">
                <div class="d-flex align-center mb-1">
                  <a
                    :href="`https://hpo.jax.org/browse/term/${result.hpo_id}`"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="hpo-link"
                    @click.stop
                  >
                    <span class="font-weight-bold">{{ result.hpo_id }}</span>
                    <v-icon size="x-small" class="ml-1">mdi-open-in-new</v-icon>
                  </a>
                </div>
                <div class="text-body-2">{{ result.label }}</div>
              </div>

              <!-- Scores -->
              <div class="d-flex align-center gap-1">
                <SimilarityScore
                  :score="result.similarity"
                  type="similarity"
                  :decimals="2"
                  :show-animation="false"
                />
                <SimilarityScore
                  v-if="result.cross_encoder_score"
                  :score="result.cross_encoder_score"
                  type="rerank"
                  :decimals="2"
                  :show-animation="false"
                />
              </div>

              <!-- Expand Icon -->
              <v-icon class="ml-2">
                {{ expanded ? 'mdi-chevron-up' : 'mdi-chevron-down' }}
              </v-icon>
            </div>
          </template>
        </v-expansion-panel-title>

        <!-- Panel Content (NEW: Term Details) -->
        <v-expansion-panel-text>
          <TermDetailsPanel
            :definition="result.definition"
            :synonyms="result.synonyms"
          />
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <!-- Fallback: Regular list items if details disabled -->
    <v-list-item
      v-else
      v-for="(result, index) in responseData.results"
      :key="result.hpo_id"
      class="mb-1 rounded-lg"
      :color="collectedPhenotypeIds.has(result.hpo_id) ? 'primary-lighten-5' : 'grey-lighten-5'"
      border
      density="compact"
    >
      <!-- Existing list item template -->
    </v-list-item>
  </v-list>
</template>

<script setup lang="ts">
import { useSettingsStore } from '@/stores/settings'
import TermDetailsPanel from './TermDetailsPanel.vue'

const settingsStore = useSettingsStore()
</script>
```

#### Step 4.4: Add i18n Translations

**Files:** `frontend/src/locales/*.json`

**Add translations:**
```json
// frontend/src/locales/en.json
{
  "queryInterface": {
    "optionsTitle": "Query Options",
    "showTermDetails": "Show term details (definitions & synonyms)",
    "showTermDetailsHint": "Includes detailed information in search results. May slightly increase response time."
  },
  "termDetails": {
    "definition": "Definition",
    "synonyms": "Synonyms",
    "noDetails": "No additional details available for this term."
  }
}

// frontend/src/locales/de.json
{
  "queryInterface": {
    "optionsTitle": "Abfrageoptionen",
    "showTermDetails": "Begriffsdetails anzeigen (Definitionen & Synonyme)",
    "showTermDetailsHint": "Enthält detaillierte Informationen in den Suchergebnissen. Kann die Antwortzeit leicht erhöhen."
  },
  "termDetails": {
    "definition": "Definition",
    "synonyms": "Synonyme",
    "noDetails": "Keine weiteren Details für diesen Begriff verfügbar."
  }
}

// Add translations for es.json, fr.json, nl.json similarly
```

**Validation:**
```bash
make frontend-i18n-check  # REQUIRED: Validates all locales match
```

---

### Phase 5: Testing & Quality Assurance

**Estimated Time:** 3 hours

#### Step 5.1: Unit Tests Coverage

**Target:** ≥90% coverage for new code

**Test Files:**
- `tests/test_hpo_database.py` - Database batch lookup
- `tests_new/unit/test_details_enrichment.py` - Enrichment logic
- `tests_new/unit/test_api_query_endpoint.py` - API endpoint
- `frontend/src/stores/__tests__/settings.spec.ts` - Settings store
- `frontend/src/components/__tests__/TermDetailsPanel.spec.ts` - Detail component

**Run Tests:**
```bash
# Python tests
make test                    # All tests
pytest tests_new/unit/test_details_enrichment.py -v  # Specific file
make test-cov                # Coverage report

# Frontend tests
make frontend-test           # All tests
make frontend-test-cov       # Coverage report
```

#### Step 5.2: Integration Tests

**Test Scenarios:**
1. ✅ End-to-end CLI query with details
2. ✅ API POST request with `include_details=true`
3. ✅ Frontend toggle → API call → Results display
4. ✅ Performance benchmark (details vs no details)

**Example Integration Test:**
```python
# tests_new/integration/test_query_with_details_e2e.py (NEW)

def test_cli_query_with_details_integration(tmp_path):
    """Test full CLI flow with details enrichment."""
    # Setup test environment
    setup_test_data(tmp_path)

    # Run CLI command
    result = runner.invoke(
        app,
        [
            "query",
            "patient has seizures",
            "--include-details",
            "--data-dir", str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Definition:" in result.stdout
    assert "Synonyms:" in result.stdout


def test_api_query_with_details_integration(test_client):
    """Test full API flow with details enrichment."""
    response = test_client.post(
        "/api/query/",
        json={
            "text": "patient has fever and seizures",
            "num_results": 5,
            "include_details": True,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert "results" in data
    assert len(data["results"]) > 0

    # Verify at least one result has details
    has_definition = any(r.get("definition") for r in data["results"])
    assert has_definition
```

#### Step 5.3: Performance Testing

**Benchmarks:**
```python
# tests/benchmark_details_performance.py (NEW)

import time
import statistics

def benchmark_details_overhead():
    """Measure performance overhead of details enrichment."""

    # Test with 10 results
    results_10 = [
        {"hpo_id": f"HP:00{i:05d}", "label": f"Term {i}", "similarity": 0.9}
        for i in range(1, 11)
    ]

    # Benchmark without details (baseline)
    times_baseline = []
    for _ in range(100):
        start = time.perf_counter()
        # Simulate baseline processing
        _ = results_10.copy()
        times_baseline.append(time.perf_counter() - start)

    # Benchmark with details
    times_with_details = []
    for _ in range(100):
        start = time.perf_counter()
        enriched = enrich_results_with_details(results_10.copy())
        times_with_details.append(time.perf_counter() - start)

    # Calculate stats
    baseline_mean = statistics.mean(times_baseline) * 1000  # ms
    details_mean = statistics.mean(times_with_details) * 1000  # ms
    overhead = details_mean - baseline_mean

    print(f"Baseline (no details): {baseline_mean:.2f}ms")
    print(f"With details: {details_mean:.2f}ms")
    print(f"Overhead: {overhead:.2f}ms")

    # Assert acceptable overhead
    assert overhead < 100, f"Details overhead too high: {overhead}ms"
```

**Expected Results:**
- Baseline API query (10 results): ~150-200ms
- With details enrichment (10 results): ~200-250ms
- Overhead: < 100ms ✅

#### Step 5.4: Code Quality Checks

**Run All Checks:**
```bash
# REQUIRED before commit!
make all                     # Format + lint + test

# Individual checks
make check                   # Ruff format + lint
make typecheck-fast          # mypy type checking
make frontend-lint           # ESLint 9
make frontend-format         # Prettier
make frontend-i18n-check     # Translation validation
```

**Expected Results:**
- ✅ 0 Ruff errors
- ✅ 0 mypy errors
- ✅ 0 ESLint errors
- ✅ All tests passing
- ✅ All locales valid

---

## UI/UX Design

### Design Principles

1. **Progressive Disclosure:** Details hidden by default (expandable on demand)
2. **Clear Affordance:** Obvious expand/collapse indicators
3. **Performance Feedback:** Loading states during API calls
4. **Accessibility:** Keyboard navigation, screen reader support
5. **Mobile-First:** Responsive design for all screen sizes

### Visual Mockup (Text-Based)

```
┌─────────────────────────────────────────────────────────────┐
│ Query: "patient has seizures"                         [🔍]  │
├─────────────────────────────────────────────────────────────┤
│ ⚙️ Query Options                                        [▼] │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ ☑️ Show term details (definitions & synonyms)      │   │
│   │ ℹ️ Includes detailed information in search results │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Results (5 found)                                           │
├─────────────────────────────────────────────────────────────┤
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│ ┃ [1] HP:0001250 🔗                           0.95 [▼]  ┃  │
│ ┃     Seizure                                           ┃  │
│ ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  │
│ ┃ 📄 Definition:                                        ┃  │
│ ┃    A seizure is an intermittent abnormality of       ┃  │
│ ┃    nervous system physiology characterized by...     ┃  │
│ ┃                                                       ┃  │
│ ┃ 🏷️ Synonyms:                                          ┃  │
│ ┃    [Seizures] [Epileptic seizure]                    ┃  │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                             │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ [2] HP:0002119 🔗                           0.87 [▶]  │ │
│ │     Ventriculomegaly                                  │ │
│ └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Interaction Flow

1. **User enables toggle** → State saved to localStorage
2. **User submits query** → API called with `include_details=true`
3. **Results load** → Expansion panels rendered
4. **User clicks panel** → Details revealed with smooth animation
5. **User disables toggle** → Next query uses compact view

### Accessibility Features

- ✅ ARIA labels on all interactive elements
- ✅ Keyboard navigation (`Tab`, `Enter`, `Space`)
- ✅ Screen reader announcements for state changes
- ✅ Sufficient color contrast (WCAG AA)
- ✅ Focus indicators on all controls

---

## Performance Considerations

### Optimization Strategies

1. **Batch Database Queries:** Single query for all term IDs (O(n) not O(n²))
2. **Lazy Expansion:** Details only rendered when panel opened
3. **Response Caching:** Browser caches API responses (standard HTTP)
4. **Conditional Rendering:** Vue v-if skips rendering when details disabled
5. **JSON Serialization:** Pydantic excludes `None` fields (smaller payloads)

### Benchmarks & SLAs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Database lookup (10 terms) | < 50ms | `benchmark_get_terms_by_ids()` |
| API response (10 results, with details) | < 300ms | Integration test |
| Frontend render (10 expansion panels) | < 100ms | Browser DevTools |
| Bundle size increase | < 5KB | Webpack bundle analyzer |

### Monitoring

**Add Performance Logging:**
```python
# phentrieve/retrieval/details_enrichment.py
import time

def enrich_results_with_details(...):
    if not results:
        return results

    start_time = time.perf_counter()

    # ... enrichment logic ...

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        f"Details enrichment completed in {elapsed_ms:.2f}ms "
        f"for {len(results)} results"
    )

    return results
```

---

## Rollback Plan

### If Issues Discovered in Production

**Step 1: Emergency Disable (Frontend)**
```typescript
// frontend/src/stores/settings.ts
// Force disable details feature
const showTermDetails = ref<boolean>(false)
const FEATURE_DISABLED = true // Emergency kill switch

watch(showTermDetails, (newValue) => {
  if (FEATURE_DISABLED) {
    showTermDetails.value = false
    return
  }
  // ... normal logic ...
})
```

**Step 2: Revert API Changes**
```bash
# Revert to previous commit
git revert <commit-sha> --no-edit

# Rebuild and redeploy
make docker-build
docker-compose up -d
```

**Step 3: Database Rollback (Not Needed)**
- No schema changes made ✅
- No data migrations needed ✅
- Database remains compatible ✅

### Rollback Validation

```bash
# Verify API still works
curl -X POST http://localhost:8734/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"text": "patient has fever"}'

# Verify frontend loads
curl http://localhost:5734/

# Run smoke tests
pytest tests/test_api_basic.py -v
```

---

## References

### Best Practices Documentation

- **REST API Design:** [Microsoft Azure API Guidelines](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design)
- **FastAPI/Pydantic:** [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/body-fields/)
- **Vuetify Expansion Panels:** [Official Docs](https://vuetifyjs.com/en/components/expansion-panels/)
- **SQL Injection Prevention:** Parameterized queries (used in implementation)
- **SOLID Principles:** [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

### Related Code

- Database schema: `phentrieve/data_processing/hpo_database.py:32-59`
- API schemas: `api/schemas/query_schemas.py:7-76`
- Retrieval helper: `phentrieve/retrieval/api_helpers.py:13-217`
- Results display: `frontend/src/components/ResultsDisplay.vue`

### GitHub Issue

- **Issue #24:** [HPO Term Details in API Responses](https://github.com/berntpopp/phentrieve/issues/24)

---

## Appendix: Implementation Checklist

### Pre-Implementation
- [ ] Review plan with team
- [ ] Confirm database schema (already optimal ✅)
- [ ] Set up feature branch: `git checkout -b feature/issue-24-hpo-term-details`

### Phase 1: Database Layer
- [ ] Implement `HPODatabase.get_terms_by_ids()`
- [ ] Write unit tests (≥90% coverage)
- [ ] Run performance benchmark (< 50ms for 10 terms)
- [ ] Code review

### Phase 2: CLI Integration
- [ ] Create `details_enrichment.py` module
- [ ] Add `--include-details` CLI flag
- [ ] Update output formatter
- [ ] Manual testing with real data
- [ ] Code review

### Phase 3: API Integration
- [ ] Update `QueryRequest` schema
- [ ] Update `HPOResultItem` schema
- [ ] Update `execute_hpo_retrieval_for_api()`
- [ ] Update `run_hpo_query()` endpoint
- [ ] API integration tests
- [ ] Performance benchmark (< 300ms total)
- [ ] Code review

### Phase 4: Frontend Integration
- [ ] Create `settings.ts` store
- [ ] Add toggle to `QueryInterface.vue`
- [ ] Create `TermDetailsPanel.vue` component
- [ ] Update `ResultsDisplay.vue` with expansion panels
- [ ] Add i18n translations (EN, DE, ES, FR, NL)
- [ ] Validate translations: `make frontend-i18n-check`
- [ ] Manual UI testing (desktop + mobile)
- [ ] Accessibility audit
- [ ] Code review

### Phase 5: Testing & QA
- [ ] Run full test suite: `make test`
- [ ] Run frontend tests: `make frontend-test`
- [ ] Type checking: `make typecheck-fast`
- [ ] Linting: `make check` + `make frontend-lint`
- [ ] E2E integration tests
- [ ] Performance benchmarks
- [ ] Load testing (optional)

### Pre-Commit
- [ ] **CRITICAL:** `make all` (format + lint + test) ✅
- [ ] **CRITICAL:** `make typecheck-fast` ✅
- [ ] **CRITICAL:** `make frontend-i18n-check` ✅
- [ ] All tests passing ✅
- [ ] Update CLAUDE.md with new API parameters
- [ ] Update API docs (OpenAPI examples)

### Deployment
- [ ] Create pull request with detailed description
- [ ] Link to issue #24
- [ ] Request code review
- [ ] Address review comments
- [ ] Merge to main
- [ ] Monitor production logs
- [ ] Update STATUS.md

### Post-Deployment
- [ ] Verify API `/docs` shows new fields
- [ ] User acceptance testing
- [ ] Performance monitoring (first 48 hours)
- [ ] Move plan to `plan/02-completed/`
- [ ] Close issue #24 🎉

---

## Success Metrics (After Deployment)

- ✅ Feature used by ≥20% of API consumers (tracked via logs)
- ✅ No increase in error rate (monitored via logs)
- ✅ P95 response time remains < 500ms
- ✅ Zero bug reports related to details feature (first 2 weeks)
- ✅ Positive user feedback on UI/UX

---

**Plan Status:** Ready for Implementation
**Next Action:** Begin Phase 1 (Database Layer)
**Estimated Total Time:** 14 hours (spread over 2-3 days)
