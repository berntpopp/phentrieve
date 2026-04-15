# HPO Term Details in API Responses - REVISED Plan (v2)

**Status:** Active (Revised after critical review)
**Created:** 2025-11-20
**Revised:** 2025-11-20 (Addressed critical issues from code review)
**Priority:** High
**Related Issue:** [#24](https://github.com/berntpopp/phentrieve/issues/24)
**Review Document:** `HPO-TERM-DETAILS-CRITICAL-REVIEW.md`
**Estimated Complexity:** Medium
**Estimated Time:** 10 hours (reduced from 14)

---

## Changes from v1

**Critical Fixes Applied:**
- ✅ **DRY**: Extracted shared JSON deserialization helper
- ✅ **Antipattern**: Replaced broad exception handling with specific errors
- ✅ **Purity**: Enrichment function returns new objects (no mutation)
- ✅ **KISS**: Removed over-engineered Pinia store, using composable instead
- ✅ **Performance**: Added result count limits with details enabled
- ✅ **UX**: Consistent UI structure (always expansion panels)
- ✅ **Concurrency**: Added connection reuse pattern
- ✅ **Testing**: Reduced coverage target to 80% (realistic)

**Complexity Reduction:**
- 📉 Files: 6 → 3 new files (50% reduction)
- 📉 LOC: ~800 → ~450 lines (44% reduction)
- 📉 Phases: 5 → 3 phases (40% reduction)
- 📉 Time: 14h → 10h (29% faster)

---

## Table of Contents

1. [Objective](#objective)
2. [Success Criteria](#success-criteria)
3. [Implementation Plan](#implementation-plan)
4. [Testing Strategy](#testing-strategy)
5. [Rollback Plan](#rollback-plan)

---

## Objective

Enable API consumers to optionally retrieve HPO term details (definitions and synonyms) through a single request, following DRY, KISS, and SOLID principles.

**Core Requirements:**
- ✅ Add `include_details` parameter to `QueryRequest` (default: `false`)
- ✅ Extend `HPOResultItem` with optional `definition` and `synonyms` fields
- ✅ Fetch details efficiently from SQLite database
- ✅ Maintain backward compatibility
- ✅ Preserve performance (< 300ms for 10 results with details)
- ✅ Follow best practices (no antipatterns, no over-engineering)

---

## Success Criteria

### Backend
- [ ] Database method uses DRY helper for JSON deserialization
- [ ] CLI supports `--include-details` flag
- [ ] API request includes `include_details: Optional[bool] = False`
- [ ] API response includes optional `definition` and `synonyms` fields
- [ ] Result count limited to 20 when details enabled (performance protection)
- [ ] Enrichment function returns new objects (no input mutation)
- [ ] Error handling lets unexpected errors propagate (fail fast)
- [ ] Database connection reused via caching (concurrency safe)
- [ ] All tests pass (80% coverage target)
- [ ] Type checking passes (0 mypy errors)

### Frontend
- [ ] Uses `useLocalStorage` composable (no Pinia store)
- [ ] Options panel includes "Show term details" toggle
- [ ] Consistent expansion panel UI (not conditional structure)
- [ ] Responsive design works on mobile
- [ ] i18n support for all UI text (EN, DE, ES, FR, NL)

### Quality
- [ ] CLAUDE.md updated
- [ ] API docs reflect schema changes
- [ ] Code follows style conventions
- [ ] No antipatterns introduced

---

## Implementation Plan

### Phase 1: Database & Core Logic (4 hours)

#### Step 1.1: Extract DRY Helper in Database Class

**File:** `phentrieve/data_processing/hpo_database.py`

**Fix DRY violation by extracting shared deserialization:**

```python
class HPODatabase:
    # ... existing methods ...

    def _deserialize_term_row(self, row: sqlite3.Row) -> dict[str, Any]:
        """
        Deserialize database row into term dictionary.

        Single source of truth for JSON field deserialization.
        Reused by load_all_terms() and get_terms_by_ids().

        Args:
            row: SQLite Row object with term data

        Returns:
            Dictionary with deserialized JSON fields
        """
        term = dict(row)
        term["synonyms"] = json.loads(term.get("synonyms") or "[]")
        term["comments"] = json.loads(term.get("comments") or "[]")
        return term

    def load_all_terms(self) -> list[dict[str, Any]]:
        """Load all HPO terms from database."""
        conn = self.get_connection()
        cursor = conn.execute(
            """
            SELECT id, label, definition, synonyms, comments
            FROM hpo_terms
            ORDER BY id
            """
        )

        # Use shared helper (DRY)
        terms = [self._deserialize_term_row(row) for row in cursor]

        logger.debug(f"Loaded {len(terms)} HPO terms from database")
        return terms

    def get_terms_by_ids(self, term_ids: list[str]) -> dict[str, dict[str, Any]]:
        """
        Efficiently fetch multiple HPO terms by IDs in a single query.

        Args:
            term_ids: List of HPO term IDs (e.g., ["HP:0001250", "HP:0002119"])

        Returns:
            Dictionary mapping {term_id: term_data} with deserialized JSON fields

        Performance:
            - Single SQL query using IN clause (O(n))
            - Returns empty dict if term_ids is empty (short-circuit)

        Example:
            >>> db = HPODatabase(Path("hpo_data.db"))
            >>> terms = db.get_terms_by_ids(["HP:0001250"])
            >>> terms["HP:0001250"]["definition"]
            "A seizure is an intermittent abnormality..."
        """
        if not term_ids:
            logger.debug("Empty term_ids list, returning empty dict")
            return {}

        conn = self.get_connection()

        # Parameterized query for SQL injection safety
        placeholders = ",".join("?" * len(term_ids))
        query = f"""
            SELECT id, label, definition, synonyms, comments
            FROM hpo_terms
            WHERE id IN ({placeholders})
        """

        cursor = conn.execute(query, term_ids)

        # Use shared helper (DRY)
        terms_map = {
            row["id"]: self._deserialize_term_row(row)
            for row in cursor
        }

        # Log warning for missing terms
        if len(terms_map) < len(term_ids):
            found_ids = set(terms_map.keys())
            missing_ids = set(term_ids) - found_ids
            logger.warning(
                f"Terms not found: {sorted(missing_ids)[:5]}"
                f"{'...' if len(missing_ids) > 5 else ''}"
            )

        logger.debug(f"Fetched {len(terms_map)}/{len(term_ids)} terms")
        return terms_map
```

**Testing:**
```python
# tests/test_hpo_database.py
def test_get_terms_by_ids_batch_lookup(hpo_db_with_data):
    """Test efficient batch lookup."""
    ids = ["HP:0000001", "HP:0000118"]
    result = hpo_db_with_data.get_terms_by_ids(ids)

    assert len(result) == 2
    assert "HP:0000001" in result
    assert isinstance(result["HP:0000001"]["synonyms"], list)

def test_get_terms_by_ids_empty_list():
    """Test empty input returns empty dict."""
    db = HPODatabase(":memory:")
    db.initialize_schema()
    assert db.get_terms_by_ids([]) == {}

def test_deserialize_term_row_consistency(hpo_db_with_data):
    """Ensure load_all_terms and get_terms_by_ids return consistent format."""
    all_terms = {t["id"]: t for t in hpo_db_with_data.load_all_terms()}
    batch_terms = hpo_db_with_data.get_terms_by_ids(list(all_terms.keys())[:5])

    # Same keys should have same structure
    for term_id in batch_terms:
        assert all_terms[term_id].keys() == batch_terms[term_id].keys()
```

---

#### Step 1.2: Create Pure Enrichment Function

**File:** `phentrieve/retrieval/details_enrichment.py` (NEW)

**Key principles:**
- ✅ Returns new objects (no mutation)
- ✅ Specific error handling (no broad exceptions)
- ✅ Fail fast for unexpected errors

```python
"""
HPO term details enrichment utilities.

Provides pure functions for adding definitions and synonyms to query results.
"""

import logging
from pathlib import Path

from phentrieve.config import DEFAULT_HPO_DB_FILENAME
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.utils import get_default_data_dir, resolve_data_path

logger = logging.getLogger(__name__)


class DatabaseNotFoundError(FileNotFoundError):
    """Raised when HPO database file doesn't exist."""
    pass


def enrich_results_with_details(
    results: list[dict],
    data_dir_override: str | None = None,
) -> list[dict]:
    """
    Enrich HPO query results with definitions and synonyms.

    Returns NEW dictionaries with added detail fields. Input is NOT modified.

    Args:
        results: List of result dicts containing at minimum:
                 - hpo_id: str
                 - label: str
        data_dir_override: Optional data directory path override

    Returns:
        NEW list with enriched result dictionaries containing:
        - All original fields (preserved)
        - definition: str | None
        - synonyms: list[str] | None

    Raises:
        DatabaseNotFoundError: If database file doesn't exist (expected at startup)
        sqlite3.Error: On database failures (caller should handle)

    Example:
        >>> results = [{"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95}]
        >>> enriched = enrich_results_with_details(results)
        >>> enriched[0]["definition"]  # New field added
        "A seizure is..."
        >>> enriched[0]["similarity"]  # Original field preserved
        0.95
        >>> results[0].get("definition")  # Original NOT modified
        None
    """
    if not results:
        logger.debug("Empty results list, returning empty list")
        return []

    # Resolve database path
    data_dir = resolve_data_path(
        data_dir_override, "data_dir", get_default_data_dir
    )
    db_path = data_dir / DEFAULT_HPO_DB_FILENAME

    # Check database exists (expected error at startup - handle gracefully)
    if not db_path.exists():
        logger.warning(
            f"Database not found: {db_path}. "
            "Returning results without details. "
            "Run 'phentrieve data prepare' to generate database."
        )
        # Return NEW dicts with None details (preserve original)
        return [
            {**result, "definition": None, "synonyms": None}
            for result in results
        ]

    # Fetch term details (let database errors propagate - unexpected failures)
    hpo_ids = [result["hpo_id"] for result in results]

    db = HPODatabase(db_path)
    try:
        terms_map = db.get_terms_by_ids(hpo_ids)
    finally:
        db.close()  # Always close connection

    # Create NEW enriched dictionaries (pure function - no mutation)
    enriched_results = []
    for result in results:
        hpo_id = result["hpo_id"]
        term_data = terms_map.get(hpo_id, {})

        # Extract details (convert empty strings to None for API clarity)
        definition = term_data.get("definition", "")
        synonyms = term_data.get("synonyms", [])

        # Create NEW dict with all original fields + details
        enriched = {
            **result,  # Spread all original fields
            "definition": definition if definition else None,
            "synonyms": synonyms if synonyms else None,
        }
        enriched_results.append(enriched)

    logger.debug(
        f"Enriched {len(enriched_results)} results with details "
        f"({len(terms_map)} found in database)"
    )

    return enriched_results
```

**Testing:**
```python
# tests_new/unit/test_details_enrichment.py
import pytest
from phentrieve.retrieval.details_enrichment import (
    enrich_results_with_details,
    DatabaseNotFoundError,
)


def test_enrich_returns_new_objects():
    """Ensure enrichment doesn't mutate input (pure function)."""
    original = [{"hpo_id": "HP:0001250", "label": "Seizure"}]
    original_copy = original.copy()

    enriched = enrich_results_with_details(original, data_dir_override="/tmp")

    # Original unchanged
    assert original == original_copy
    assert "definition" not in original[0]

    # New object created
    assert enriched is not original
    assert "definition" in enriched[0]


def test_enrich_empty_results():
    """Test empty input returns empty output."""
    result = enrich_results_with_details([])
    assert result == []


def test_enrich_with_valid_terms(tmp_path, create_test_db):
    """Test successful enrichment."""
    db_path = tmp_path / "hpo_data.db"
    create_test_db(
        db_path,
        terms=[{
            "id": "HP:0001250",
            "label": "Seizure",
            "definition": "A seizure is...",
            "synonyms": '["Seizures", "Epileptic seizure"]',
            "comments": "[]",
        }]
    )

    results = [{"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95}]
    enriched = enrich_results_with_details(results, data_dir_override=str(tmp_path))

    assert len(enriched) == 1
    assert enriched[0]["definition"] == "A seizure is..."
    assert enriched[0]["synonyms"] == ["Seizures", "Epileptic seizure"]
    assert enriched[0]["similarity"] == 0.95  # Original preserved


def test_enrich_without_database(tmp_path):
    """Test graceful handling when database missing."""
    results = [{"hpo_id": "HP:0001250", "label": "Seizure"}]

    # Should not raise, return results with None details
    enriched = enrich_results_with_details(
        results, data_dir_override=str(tmp_path)
    )

    assert enriched[0]["definition"] is None
    assert enriched[0]["synonyms"] is None


def test_enrich_with_db_connection_error(tmp_path, monkeypatch):
    """Test that database errors propagate (fail fast)."""
    def mock_get_connection_error(*args):
        raise sqlite3.OperationalError("Database locked")

    monkeypatch.setattr(HPODatabase, "get_connection", mock_get_connection_error)

    results = [{"hpo_id": "HP:0001250", "label": "Seizure"}]

    # Should raise (not swallow error)
    with pytest.raises(sqlite3.OperationalError, match="Database locked"):
        enrich_results_with_details(results, data_dir_override=str(tmp_path))
```

---

#### Step 1.3: Add Connection Reuse Pattern

**File:** `phentrieve/retrieval/details_enrichment.py`

**Add cached database connection (follows existing `@lru_cache` pattern):**

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_shared_database(db_path_str: str) -> HPODatabase:
    """
    Get shared HPODatabase instance for reuse across requests.

    Thread-safe for concurrent reads (SQLite Row factory).
    Connection is cached and reused to avoid overhead.

    Args:
        db_path_str: String path to database (strings are hashable for LRU cache)

    Returns:
        Shared HPODatabase instance

    Note:
        Call .cache_clear() in tests to reset connection.
    """
    return HPODatabase(db_path_str)


def enrich_results_with_details(
    results: list[dict],
    data_dir_override: str | None = None,
) -> list[dict]:
    """..."""
    # ... validation ...

    # Use shared connection (cached)
    db = get_shared_database(str(db_path))

    # No close() - connection is shared
    terms_map = db.get_terms_by_ids(hpo_ids)

    # ... enrichment logic ...
```

**Testing:**
```python
def test_connection_reuse(tmp_path, create_test_db):
    """Verify database connection is reused across calls."""
    db_path = tmp_path / "hpo_data.db"
    create_test_db(db_path, terms=[...])

    # Clear cache before test
    get_shared_database.cache_clear()

    results = [{"hpo_id": "HP:0001250", "label": "Seizure"}]

    # First call creates connection
    db1 = get_shared_database(str(db_path))

    # Second call reuses connection
    db2 = get_shared_database(str(db_path))

    assert db1 is db2  # Same object!
```

---

#### Step 1.4: Add CLI Flag

**File:** `phentrieve/cli/query_commands.py`

**Simple addition:**

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
        phentrieve query "patient has seizures"
        phentrieve query "patient has fever" --include-details
    """
    # ... existing query logic ...

    # After getting results
    if include_details:
        from phentrieve.retrieval.details_enrichment import (
            enrich_results_with_details,
        )
        results = enrich_results_with_details(results, data_dir_override)

    # Format and display
    format_results_for_cli(results, include_details=include_details)
```

**Update formatter:**

```python
# phentrieve/retrieval/output_formatters.py

def format_results_for_cli(
    results: list[dict],
    include_details: bool = False,
) -> None:
    """Format query results for CLI display."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, result in enumerate(results, 1):
        console.print(f"\n[bold cyan]{i}. {result['hpo_id']}[/bold cyan]")
        console.print(f"   Label: {result['label']}")

        if result.get("similarity"):
            console.print(f"   Similarity: {result['similarity']:.3f}")

        # Show details if included
        if include_details:
            if result.get("definition"):
                console.print(f"\n   [dim]Definition:[/dim]")
                console.print(f"   {result['definition']}")

            if result.get("synonyms"):
                console.print(f"\n   [dim]Synonyms:[/dim]")
                console.print(f"   {', '.join(result['synonyms'])}")
```

---

### Phase 2: API Integration (3 hours)

#### Step 2.1: Update API Schemas with Validation

**File:** `api/schemas/query_schemas.py`

**Add validation to prevent performance issues:**

```python
from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    """Request schema for HPO term query."""

    text: str = Field(..., min_length=1)
    # ... existing fields ...

    # NEW FIELD
    include_details: bool = Field(
        default=False,
        description=(
            "Include HPO term definitions and synonyms. "
            "Limited to 20 results when enabled for performance. "
            "Default: false"
        ),
    )

    @model_validator(mode='after')
    def validate_result_count_with_details(self) -> 'QueryRequest':
        """
        Limit result count when details enabled (performance protection).

        Prevents excessive response sizes and database load.
        """
        if self.include_details and self.num_results > 20:
            raise ValueError(
                "Maximum 20 results allowed when include_details=true. "
                "Requested: {self.num_results}. "
                "Use pagination for larger result sets or disable details."
            )
        return self


class HPOResultItem(BaseModel):
    """Single HPO term result."""

    hpo_id: str
    label: str
    similarity: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    original_rank: Optional[int] = None

    # NEW FIELDS (Optional for backward compatibility)
    definition: Optional[str] = Field(
        default=None,
        description="HPO term definition (when include_details=true)",
    )
    synonyms: Optional[list[str]] = Field(
        default=None,
        description="List of term synonyms (when include_details=true)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.954,
                "definition": "A seizure is an intermittent abnormality...",
                "synonyms": ["Seizures", "Epileptic seizure"],
            }
        }
```

---

#### Step 2.2: Update API Endpoint

**File:** `phentrieve/retrieval/api_helpers.py`

**Add parameter and enrichment call:**

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
    """Execute HPO term retrieval for API requests."""
    # ... existing retrieval logic ...

    # Format base results
    formatted_results = []
    for item in hpo_embeddings_results:
        result_item = {
            "hpo_id": item["hpo_id"],
            "label": item["label"],
            "similarity": item["similarity"],
        }
        # Add scoring fields if available
        if "cross_encoder_score" in item:
            result_item["cross_encoder_score"] = item["cross_encoder_score"]
        if "original_rank" in item:
            result_item["original_rank"] = item["original_rank"]

        formatted_results.append(result_item)

    # Enrich with details if requested
    if include_details:
        from phentrieve.retrieval.details_enrichment import (
            enrich_results_with_details,
        )
        formatted_results = enrich_results_with_details(formatted_results)

    return {
        "query_text_processed": segment_to_process,
        "results": formatted_results,
        "original_query_assertion_status": (
            original_query_assertion_status.value
            if original_query_assertion_status
            else None
        ),
    }
```

**File:** `api/routers/query_router.py`

**Pass flag to helper:**

```python
@router.post("/", response_model=QueryResponse)
async def run_hpo_query(
    request: QueryRequest,
    retriever: DenseRetriever = Depends(get_retriever_for_request),
):
    """Execute HPO term query."""
    # ... existing setup ...

    query_results_dict = await execute_hpo_retrieval_for_api(
        text=request.text,
        language=language_to_use,
        retriever=retriever,
        num_results=request.num_results,
        similarity_threshold=request.similarity_threshold,
        enable_reranker=request.enable_reranker and (cross_encoder_instance is not None),
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

    # ... return response ...
```

---

### Phase 3: Frontend Integration (3 hours)

#### Step 3.1: Simple State Management (No Store!)

**File:** `frontend/src/components/QueryInterface.vue`

**Use `useLocalStorage` composable (KISS principle):**

```vue
<script setup lang="ts">
import { useLocalStorage } from '@vueuse/core'

// Simple reactive localStorage (no Pinia needed!)
const showTermDetails = useLocalStorage('phentrieve_show_term_details', false)
const optionsPanelOpen = ref<number | undefined>(undefined)

const submitQuery = async () => {
  const requestBody = {
    text: queryText.value,
    num_results: numResults.value,
    // ... other parameters ...
    include_details: showTermDetails.value,  // From composable
  }

  const response = await fetch('/api/query/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  })

  // ... handle response ...
}
</script>

<template>
  <v-card>
    <!-- Existing query form -->

    <!-- Options Section -->
    <v-expansion-panels v-model="optionsPanelOpen" class="mt-2">
      <v-expansion-panel>
        <v-expansion-panel-title>
          <v-icon start>mdi-cog</v-icon>
          {{ $t('queryInterface.optionsTitle') }}
        </v-expansion-panel-title>

        <v-expansion-panel-text>
          <v-switch
            v-model="showTermDetails"
            color="primary"
            :label="$t('queryInterface.showTermDetails')"
            :hint="$t('queryInterface.showTermDetailsHint')"
            persistent-hint
          />
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <!-- Submit button -->
  </v-card>
</template>
```

**Benefits:**
- ✅ 10 lines instead of 50+ (Pinia store eliminated)
- ✅ Still reactive
- ✅ Still persists to localStorage
- ✅ Standard Vue 3 pattern
- ✅ One less file to maintain

---

#### Step 3.2: Consistent UI with Expansion Panels

**File:** `frontend/src/components/ResultsDisplay.vue`

**Always use expansion panels (no conditional structure):**

```vue
<script setup lang="ts">
import { useLocalStorage } from '@vueuse/core'

const showTermDetails = useLocalStorage('phentrieve_show_term_details', false)

interface Props {
  responseData: QueryResponse
}

const props = defineProps<Props>()
</script>

<template>
  <!-- Info card (existing) -->

  <!-- ALWAYS use expansion panels (consistent UI) -->
  <v-expansion-panels variant="accordion" class="mt-2">
    <v-expansion-panel
      v-for="(result, index) in responseData.results"
      :key="result.hpo_id"
      class="mb-1"
    >
      <!-- Header: Always visible -->
      <v-expansion-panel-title
        :class="collectedPhenotypeIds.has(result.hpo_id) ? 'bg-primary-lighten-5' : 'bg-grey-lighten-5'"
      >
        <div class="d-flex align-center w-100">
          <v-badge :content="index + 1" color="primary" inline class="mr-3" />

          <div class="flex-grow-1">
            <a
              :href="`https://hpo.jax.org/browse/term/${result.hpo_id}`"
              target="_blank"
              class="hpo-link"
              @click.stop
            >
              <span class="font-weight-bold">{{ result.hpo_id }}</span>
              <v-icon size="x-small" class="ml-1">mdi-open-in-new</v-icon>
            </a>
            <div class="text-body-2 mt-1">{{ result.label }}</div>
          </div>

          <div class="d-flex align-center gap-1">
            <SimilarityScore :score="result.similarity" type="similarity" />
            <SimilarityScore
              v-if="result.cross_encoder_score"
              :score="result.cross_encoder_score"
              type="rerank"
            />
          </div>
        </div>
      </v-expansion-panel-title>

      <!-- Content: Conditionally show details -->
      <v-expansion-panel-text>
        <!-- If details enabled and available -->
        <div v-if="showTermDetails && (result.definition || result.synonyms)" class="term-details pa-3">
          <!-- Definition -->
          <div v-if="result.definition" class="mb-3">
            <div class="text-subtitle-2 text-medium-emphasis mb-1">
              <v-icon size="small" class="mr-1">mdi-text-box</v-icon>
              {{ $t('termDetails.definition') }}
            </div>
            <div class="text-body-2 text-justify">
              {{ result.definition }}
            </div>
          </div>

          <!-- Synonyms -->
          <div v-if="result.synonyms && result.synonyms.length > 0">
            <div class="text-subtitle-2 text-medium-emphasis mb-1">
              <v-icon size="small" class="mr-1">mdi-tag-multiple</v-icon>
              {{ $t('termDetails.synonyms') }}
            </div>
            <div class="d-flex flex-wrap gap-1">
              <v-chip
                v-for="(synonym, idx) in result.synonyms"
                :key="idx"
                size="small"
                color="grey-lighten-2"
                label
              >
                {{ synonym }}
              </v-chip>
            </div>
          </div>
        </div>

        <!-- If details disabled -->
        <div v-else-if="!showTermDetails" class="text-caption text-medium-emphasis pa-3">
          <v-icon size="small" class="mr-1">mdi-information-outline</v-icon>
          {{ $t('termDetails.enableToSee') }}
        </div>

        <!-- If details enabled but none available -->
        <div v-else class="text-caption text-medium-emphasis pa-3">
          <v-icon size="small" class="mr-1">mdi-information-outline</v-icon>
          {{ $t('termDetails.noDetails') }}
        </div>
      </v-expansion-panel-text>
    </v-expansion-panel>
  </v-expansion-panels>
</template>

<style scoped>
.term-details {
  background-color: rgba(var(--v-theme-surface-variant), 0.3);
  border-radius: 4px;
}
</style>
```

**Benefits:**
- ✅ Consistent UI structure (always expansion panels)
- ✅ Smooth transitions (no layout shifts)
- ✅ Better UX (predictable interactions)
- ✅ Setting only controls data visibility, not structure

---

#### Step 3.3: Add i18n Translations

**Files:** `frontend/src/locales/{en,de,es,fr,nl}.json`

```json
// en.json
{
  "queryInterface": {
    "optionsTitle": "Query Options",
    "showTermDetails": "Show term details (definitions & synonyms)",
    "showTermDetailsHint": "Includes detailed information in search results"
  },
  "termDetails": {
    "definition": "Definition",
    "synonyms": "Synonyms",
    "noDetails": "No additional details available for this term",
    "enableToSee": "Enable 'Show term details' to see definitions and synonyms"
  }
}

// de.json (similar structure, German translations)
// es.json, fr.json, nl.json (similar structure, respective translations)
```

**Validation:**
```bash
make frontend-i18n-check  # REQUIRED!
```

---

## Testing Strategy

### Coverage Target: 80% (Realistic)

**Focus Areas:**
- **Critical Paths (100%):** Database queries, API endpoints, enrichment logic
- **Business Logic (80%):** CLI formatting, validation, error handling
- **UI Components (60%):** Key interactions, not every edge case

### Test Suite

**Backend Tests:**
```bash
# Database layer
pytest tests/test_hpo_database.py -v

# Enrichment function
pytest tests_new/unit/test_details_enrichment.py -v

# API endpoint
pytest tests_new/unit/test_api_query_endpoint.py -v

# Coverage
make test-cov
```

**Frontend Tests:**
```bash
# Component tests
make frontend-test

# Coverage
make frontend-test-cov
```

**Integration Tests:**
```bash
# Full flow
pytest tests_new/integration/test_query_with_details.py -v
```

### Performance Smoke Test (Simple)

```python
def test_details_enrichment_performance():
    """Ensure details enrichment is reasonably fast."""
    results = [make_result() for _ in range(10)]

    start = time.time()
    enriched = enrich_results_with_details(results)
    elapsed_ms = (time.time() - start) * 1000

    assert elapsed_ms < 100, f"Too slow: {elapsed_ms}ms"
```

**No complex benchmarking needed** - Simple smoke test is sufficient.

---

## Rollback Plan

### Emergency Disable (Frontend)

```typescript
// frontend/src/components/QueryInterface.vue
// Force disable feature
const FEATURE_DISABLED = true  // Emergency kill switch

const showTermDetails = useLocalStorage(
  'phentrieve_show_term_details',
  FEATURE_DISABLED ? false : false
)
```

### Revert Backend

```bash
# Revert commits
git revert <commit-sha> --no-edit

# Rebuild
make docker-build
docker-compose up -d

# Verify
curl -X POST http://localhost:8734/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}'
```

---

## Pre-Commit Checklist

**MANDATORY checks:**
```bash
make all                      # Format + lint + test
make typecheck-fast           # Type checking
make frontend-i18n-check      # Translation validation
```

**All must pass:**
- [ ] 0 Ruff errors
- [ ] 0 mypy errors
- [ ] 0 ESLint errors
- [ ] All tests passing
- [ ] All locales valid

---

## Summary of Improvements (v1 → v2)

**Critical Fixes:**
1. ✅ **DRY**: JSON deserialization extracted to `_deserialize_term_row()`
2. ✅ **Purity**: Enrichment returns new objects (no mutation)
3. ✅ **Error Handling**: Specific exceptions, fail fast
4. ✅ **KISS**: Removed Pinia store, using `useLocalStorage`
5. ✅ **Performance**: Result count validation (max 20 with details)
6. ✅ **UX**: Consistent expansion panel UI
7. ✅ **Concurrency**: Connection reuse with `@lru_cache`
8. ✅ **Testing**: Realistic 80% coverage target

**Complexity Reduction:**
- 50% fewer new files (3 vs 6)
- 44% less code (~450 vs ~800 LOC)
- 40% fewer phases (3 vs 5)
- 29% faster implementation (10h vs 14h)

**Maintained Strengths:**
- ✅ Backward compatible
- ✅ Zero new dependencies
- ✅ Comprehensive tests
- ✅ Clear documentation
- ✅ Rollback plan

---

**Plan Status:** ✅ Ready for Implementation
**Next Action:** Begin Phase 1 (Database & Core Logic)
