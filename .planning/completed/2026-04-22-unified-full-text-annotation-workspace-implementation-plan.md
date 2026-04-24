# Unified Full-Text Annotation Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current static full-text results display with a unified annotation workspace that serves both `standard` and `llm` backends, adds case-oriented phenopacket workflows, and updates anonymous LLM quota behavior to a configurable default of `5/day`.

**Architecture:** Keep the conversation shell and the existing `POST /api/v1/text/process` backend, but refactor the frontend full-text response into an expandable workspace composed of a document pane, phenotype pane, and a single switchable right sidebar. Add a workspace-scoped Pinia store keyed by conversation turn ID, enrich API metadata for quota/fallback behavior, and route phenopacket export through backend serialization helpers that reuse `phentrieve.phenopackets` and the annotation-sidecar design. Use the CSS Custom Highlight API as the primary highlighting primitive, with an explicit fallback only where required.

**Tech Stack:** FastAPI, Pydantic, Python 3.10, Vue 3, Pinia, Vuetify, Vue I18n, Vitest, pytest, existing Phentrieve phenopacket utilities, CSS Custom Highlight API.

---

## Parallel Execution Strategy

This plan is optimized for maximum safe parallelization across isolated git worktrees. The implementation should not be executed as one long linear branch. Instead, run independent streams in parallel against disjoint write scopes, then merge them into a single integration branch in a fixed order.

**Execution model:**

- Create one integration branch for this feature, for example `feat/unified-full-text-workspace`.
- Create one child branch and one worktree per stream from that integration branch.
- Keep each stream inside its assigned files. If a task needs a file owned by another stream, stop and move that work into the integration stream instead of cross-editing.
- Merge completed streams back into the integration branch one at a time, with verification after each merge group.
- Reserve the shared Vue entrypoints for the integration stream to avoid early merge conflicts.

**Worktree naming recommendation:**

- `ft-backend-quota`
- `ft-backend-phenopacket`
- `ft-frontend-store`
- `ft-frontend-document`
- `ft-frontend-findings`
- `ft-frontend-case`
- `ft-integration`

**Suggested setup commands:**

```bash
git switch -c feat/unified-full-text-workspace
git worktree add .worktrees/ft-backend-quota -b ft-backend-quota
git worktree add .worktrees/ft-backend-phenopacket -b ft-backend-phenopacket
git worktree add .worktrees/ft-frontend-store -b ft-frontend-store
git worktree add .worktrees/ft-frontend-document -b ft-frontend-document
git worktree add .worktrees/ft-frontend-findings -b ft-frontend-findings
git worktree add .worktrees/ft-frontend-case -b ft-frontend-case
git worktree add .worktrees/ft-integration -b ft-integration
```

If `.worktrees/` does not exist locally, create it only after confirming it is ignored by git.

**Branch ownership rule:**

- Stream branches own feature files and stream-local tests.
- The integration branch owns shared entrypoints, final wiring, conflict resolution, and any follow-up edits caused by merge interaction.
- Do not edit `frontend/src/components/ResultsDisplay.vue`, `frontend/src/components/FullTextAnnotationWorkspace.vue`, `frontend/src/components/QueryInterface.vue`, or `frontend/src/services/PhentrieveService.js` outside the integration stream unless a task below explicitly says otherwise.

## Parallel Streams

### Stream A: Backend quota and fallback metadata

**Purpose:** Ship the `5/day` default, quota reset metadata, and opt-in standard fallback behavior without touching phenopacket export.

**Owned files:**

- `api/config.py`
- `api/api.yaml`
- `api/llm_quota.py`
- `api/schemas/text_processing_schemas.py`
- `api/routers/text_processing_router.py`
- `tests/unit/api/test_llm_quota.py`
- `tests/unit/api/test_text_processing_router.py`

**Conflict policy:**

- This stream is the only stream allowed to edit `api/routers/text_processing_router.py`.
- If phenopacket export later needs shared response types, add only additive schema changes here and let Stream B consume them without rewriting this file.

### Stream B: Backend phenopacket export

**Purpose:** Add frontend-facing phenopacket serialization that reuses backend and CLI helpers.

**Owned files:**

- `api/schemas/phenopacket_schemas.py`
- `api/routers/phenopacket_router.py`
- `api/main.py`
- `tests/unit/api/test_phenopacket_router.py`

**Conflict policy:**

- This stream is the only stream allowed to edit `api/main.py`.
- Do not modify text-processing routes or quota logic here.

### Stream C: Frontend workspace store and constants

**Purpose:** Build the per-turn workspace state model so UI streams can consume stable interfaces without fighting over state logic.

**Owned files:**

- `frontend/src/stores/fullTextWorkspace.js`
- `frontend/src/constants/fullTextWorkspace.js`
- `frontend/src/stores/conversation.js`
- `frontend/src/test/stores/fullTextWorkspace.test.js`

**Conflict policy:**

- This stream defines workspace state shape, action names, undo or redo scope, and sidebar mode constants.
- UI streams may read these contracts but should not modify them directly; request integration-stream changes if contract adjustments are needed.

### Stream D: Annotated document pane and inline actions

**Purpose:** Build the document-first reading surface, CSS Custom Highlight integration, honest chunk-vs-span affordances, and selection popover.

**Owned files:**

- `frontend/src/components/AnnotatedDocumentPane.vue`
- `frontend/src/components/AnnotationActionPopover.vue`
- `frontend/src/test/components/AnnotatedDocumentPane.test.js`

**Conflict policy:**

- This stream owns highlight rendering primitives and text-selection behavior.
- Do not wire results-shell expansion or case-sidebar switching here.

### Stream E: Findings pane and annotation inspector

**Purpose:** Build the extracted phenotype review surface and deep-inspection UI.

**Owned files:**

- `frontend/src/components/PhenotypeFindingsPane.vue`
- `frontend/src/components/AnnotationInspectorPanel.vue`
- `frontend/src/test/components/PhenotypeFindingsPane.test.js`

**Conflict policy:**

- This stream owns confidence band presentation and inspector-only numeric details.
- Do not modify the case workflow or collection-panel behavior here.

### Stream F: Case workspace and phenopacket actions

**Purpose:** Replace full-text `HPO Collection` behavior with case-oriented workspace interactions and export affordances.

**Owned files:**

- `frontend/src/components/CaseWorkspacePanel.vue`
- `frontend/src/components/PhenotypeCollectionPanel.vue`
- `frontend/src/test/components/CaseWorkspacePanel.test.js`

**Conflict policy:**

- This stream owns case list behavior, add-all action behavior, and the temporary rename bridge from `HPO Collection` to `Case Workspace`.
- Do not change conversation-turn wiring or service calls here.

### Stream G: Integration and shared entrypoints

**Purpose:** Merge the backend and UI building blocks into the shipped experience.

**Owned files:**

- `frontend/src/components/ResultsDisplay.vue`
- `frontend/src/components/FullTextAnnotationWorkspace.vue`
- `frontend/src/components/QueryInterface.vue`
- `frontend/src/services/PhentrieveService.js`
- `frontend/src/test/components/FullTextAnnotationWorkspace.test.js`
- `frontend/src/test/components/ResultsDisplay.test.js`
- `frontend/src/test/components/QueryInterface.test.js`
- `frontend/src/test/services/PhentrieveService.test.js`

**Conflict policy:**

- This stream is intentionally last for frontend coding work.
- It is the only stream that may resolve contract mismatches between backend responses, store APIs, and component props.

## Merge Order

Merge streams back into the integration branch in this order:

1. Stream A
2. Stream B
3. Stream C
4. Stream D
5. Stream E
6. Stream F
7. Stream G
8. Final verification and cleanup

**Why this order:**

- Backend contracts land before frontend integration depends on them.
- The workspace store lands before higher-level UI composition.
- Document, findings, and case UI streams stay parallel and mostly isolated.
- Shared entrypoint wiring is delayed until component and store contracts have stabilized.

**Suggested merge commands:**

```bash
git switch feat/unified-full-text-workspace
git merge --no-ff ft-backend-quota
git merge --no-ff ft-backend-phenopacket
git merge --no-ff ft-frontend-store
git merge --no-ff ft-frontend-document
git merge --no-ff ft-frontend-findings
git merge --no-ff ft-frontend-case
git merge --no-ff ft-integration
```

## Merge Checkpoints

Run these checks on the integration branch after each checkpoint:

- After Streams A and B merge:
  - `uv run pytest tests/unit/api/test_llm_quota.py tests/unit/api/test_text_processing_router.py tests/unit/api/test_phenopacket_router.py -n 0 -v`
- After Streams C, D, E, and F merge:
  - `make frontend-test`
- After Stream G merges:
  - `make check`
  - `make typecheck-fast`
  - `make test`

If a checkpoint fails, fix it on the integration branch unless the failure is clearly isolated to an unmerged stream.

## Shared-File Guardrails

These files are high-conflict and must remain integration-only:

- `frontend/src/components/ResultsDisplay.vue`
- `frontend/src/components/FullTextAnnotationWorkspace.vue`
- `frontend/src/components/QueryInterface.vue`
- `frontend/src/services/PhentrieveService.js`

These files are backend integration choke points and should remain single-owner:

- `api/routers/text_processing_router.py` owned by Stream A
- `api/main.py` owned by Stream B

If parallel workers discover they need to touch one of these files outside the assigned stream, they should record the required change in their handoff notes and stop instead of creating a merge collision.

## File Map

### Backend API and serialization

- Modify: `api/config.py`
  Change the default anonymous LLM daily limit from `3` to `5`.
- Modify: `api/api.yaml`
  Change the default config value for `PHENTRIEVE_LLM_DAILY_LIMIT` from `3` to `5`.
- Modify: `api/llm_quota.py`
  Expose enough status metadata to compute and return the UTC-day reset timestamp.
- Modify: `api/schemas/text_processing_schemas.py`
  Add typed metadata helpers for fallback/quota fields and export-ready annotation fields where needed.
- Modify: `api/routers/text_processing_router.py`
  Attach richer quota metadata, fallback metadata, and reset timestamps to `meta`.
- Create: `api/schemas/phenopacket_schemas.py`
  Define request/response payloads for phenopacket export from frontend case data.
- Create: `api/routers/phenopacket_router.py`
  Add a serialization endpoint that reuses backend phenopacket helpers.
- Modify: `api/main.py`
  Register the new phenopacket router and request ID middleware if added in Task 2.

### Frontend workspace state and services

- Create: `frontend/src/stores/fullTextWorkspace.js`
  Own per-turn workspace state, active case selection, undo/redo, inspector mode, and fallback banners.
- Modify: `frontend/src/stores/conversation.js`
  Preserve turn IDs and expose helpers needed by the workspace store without mixing state ownership.
- Modify: `frontend/src/services/PhentrieveService.js`
  Parse richer full-text response metadata and add a phenopacket export API call.
- Create: `frontend/src/constants/fullTextWorkspace.js`
  Centralize workspace mode labels, confidence band thresholds, and sidebar mode constants.

### Frontend components

- Modify: `frontend/src/components/ResultsDisplay.vue`
  Replace the current static text-processing block with a workspace entrypoint.
- Create: `frontend/src/components/FullTextAnnotationWorkspace.vue`
  Expand/collapse shell for the unified full-text reader workspace.
- Create: `frontend/src/components/AnnotatedDocumentPane.vue`
  Render chunk text, gutter honesty affordances, selection behavior, and linked highlights.
- Create: `frontend/src/components/PhenotypeFindingsPane.vue`
  Show extracted phenotypes with evidence bands and inspect actions.
- Create: `frontend/src/components/CaseWorkspacePanel.vue`
  Replace full-text `HPO Collection` behavior with case/phenopacket workflows.
- Create: `frontend/src/components/AnnotationInspectorPanel.vue`
  Handle deep editing for one term/span and show numeric scores only here.
- Create: `frontend/src/components/AnnotationActionPopover.vue`
  Show quick actions on click/selection.
- Modify: `frontend/src/components/QueryInterface.vue`
  Wire workspace-aware add behavior and preserve conversation-first flow.
- Modify: `frontend/src/components/PhenotypeCollectionPanel.vue`
  Limit or gate the old collection UI so full-text mode uses the new case workspace.

### Frontend tests

- Create: `frontend/src/test/stores/fullTextWorkspace.test.js`
- Create: `frontend/src/test/components/FullTextAnnotationWorkspace.test.js`
- Create: `frontend/src/test/components/AnnotatedDocumentPane.test.js`
- Create: `frontend/src/test/components/PhenotypeFindingsPane.test.js`
- Create: `frontend/src/test/components/CaseWorkspacePanel.test.js`
- Modify: `frontend/src/test/components/ResultsDisplay.test.js`
- Modify: `frontend/src/test/components/QueryInterface.test.js`
- Modify: `frontend/src/test/services/PhentrieveService.test.js`

### Backend tests

- Create: `tests/unit/api/test_phenopacket_router.py`
- Modify: `tests/unit/api/test_text_processing_router.py`
- Modify: `tests/unit/api/test_llm_quota.py`

### Documentation

- Modify: `.planning/archived/2026-04-21-unified-full-text-annotation-workspace-design.md`
  Only if implementation uncovers contradictions that must be reflected back into the spec.

## Task-to-Stream Mapping

- **Task 1** -> Stream A
- **Task 2** -> Stream B
- **Task 3** -> Stream C
- **Task 4** -> Stream G
- **Task 5** -> Stream D
- **Task 6** -> Stream E
- **Task 7** -> Stream F, plus Stream G for service wiring only
- **Task 8** -> Stream G
- **Task 9** -> Integration branch only

**Execution note:** Tasks 1, 2, 3, 5, 6, and the stream-local portion of 7 can proceed in parallel once the worktrees exist. Task 4 should be deferred until Stream C has stabilized the store contract. Task 8 starts only after Streams A through F have merged.

## Task 1: Enrich the full-text API contract for quota, fallback, and request metadata

**Files:**
- Modify: `api/config.py`
- Modify: `api/api.yaml`
- Modify: `api/llm_quota.py`
- Modify: `api/schemas/text_processing_schemas.py`
- Modify: `api/routers/text_processing_router.py`
- Test: `tests/unit/api/test_llm_quota.py`
- Test: `tests/unit/api/test_text_processing_router.py`

- [ ] **Step 1: Write failing API tests for quota reset metadata and fallback banner inputs**

```python
def test_text_processing_router_returns_localizable_quota_metadata(client, monkeypatch):
    monkeypatch.setattr("api.config.PHENTRIEVE_ENV", "production", raising=False)
    monkeypatch.setattr("api.config.PHENTRIEVE_LLM_DAILY_LIMIT", 5, raising=False)
    monkeypatch.setattr(
        "api.routers.text_processing_router.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": "gpt-5.4-mini",
                "llm_mode": "two_phase",
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Patient had recurrent seizures.",
            "extraction_backend": "llm",
            "llm_model": "gpt-5.4-mini",
        },
    )

    assert response.status_code == 200
    assert response.json()["meta"]["quota_limit"] == 5
    assert response.json()["meta"]["quota_reset_at"]


def test_text_processing_router_can_mark_standard_fallback_after_llm_exhaustion(client, monkeypatch):
    monkeypatch.setattr("api.config.PHENTRIEVE_ENV", "production", raising=False)
    monkeypatch.setattr(
        "api.routers.text_processing_router.check_llm_quota_or_raise",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            QuotaExceededError(
                quota_used=5,
                quota_limit=5,
                quota_remaining=0,
                usage_date_utc="2026-04-22",
            )
        ),
    )
    monkeypatch.setattr(
        "api.routers.text_processing_router.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "standard",
                "fallback_reason": "llm_quota_exhausted",
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Patient had recurrent seizures.",
            "extraction_backend": "llm",
            "llm_model": "gpt-5.4-mini",
        },
        headers={"X-Phentrieve-Allow-Standard-Fallback": "true"},
    )

    assert response.status_code == 200
    assert response.json()["meta"]["fallback_reason"] == "llm_quota_exhausted"
```

- [ ] **Step 2: Run the targeted tests to verify the metadata is missing**

Run:

```bash
uv run pytest tests/unit/api/test_llm_quota.py tests/unit/api/test_text_processing_router.py -n 0 -v
```

Expected: FAIL because the router does not yet emit `quota_reset_at` and does not support explicit standard fallback signaling.

- [ ] **Step 3: Add reset-time helpers and move the default limit to 5**

In `api/llm_quota.py`, add helpers shaped like:

```python
from datetime import UTC, date, datetime, time, timedelta


def quota_reset_at_iso(usage_date_utc: str) -> str:
    current_day = date.fromisoformat(usage_date_utc)
    next_day = current_day + timedelta(days=1)
    return datetime.combine(next_day, time(0, 0, tzinfo=UTC)).isoformat()
```

In `api/config.py` and `api/api.yaml`, change:

```python
PHENTRIEVE_LLM_DAILY_LIMIT: int = int(os.getenv("PHENTRIEVE_LLM_DAILY_LIMIT", "5"))
```

```yaml
PHENTRIEVE_LLM_DAILY_LIMIT: 5
```

- [ ] **Step 4: Extend router metadata and optional fallback handling**

In `api/routers/text_processing_router.py`, update the successful LLM path to attach:

```python
response.meta["quota_limit"] = updated_quota_status.quota_limit
response.meta["quota_remaining"] = updated_quota_status.quota_remaining
response.meta["quota_reset_at"] = quota_reset_at_iso(
    updated_quota_status.usage_date_utc
)
```

Add an opt-in standard fallback branch:

```python
allow_standard_fallback = (
    http_request.headers.get("x-phentrieve-allow-standard-fallback", "").lower()
    == "true"
)

if request.extraction_backend == "llm" and _is_production_environment():
    try:
        quota_status = check_llm_quota_or_raise(http_request)
    except QuotaExceededError as exc:
        if allow_standard_fallback:
            request = request.model_copy(
                update={
                    "extraction_backend": "standard",
                    "llm_model": None,
                    "llm_mode": None,
                }
            )
            forced_standard_fallback = {
                "fallback_reason": "llm_quota_exhausted",
                "llm_quota_limit": exc.quota_limit,
                "llm_quota_reset_at": quota_reset_at_iso(exc.usage_date_utc),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=exc.to_detail(),
            ) from exc
```

After service execution, merge `forced_standard_fallback` into `response.meta` when present.

- [ ] **Step 5: Keep the response schema permissive but document the new metadata keys**

In `api/schemas/text_processing_schemas.py`, add a typed metadata helper model for internal validation and examples:

```python
class TextProcessingMetaAPI(BaseModel):
    extraction_backend: Literal["standard", "llm"]
    quota_limit: int | None = None
    quota_remaining: int | None = None
    quota_reset_at: str | None = None
    fallback_reason: str | None = None
    llm_quota_limit: int | None = None
    llm_quota_reset_at: str | None = None
```

Then keep `TextProcessingResponseAPI.meta` as `dict[str, Any]` for backward compatibility while updating docs/examples to reflect the new keys.

- [ ] **Step 6: Run the targeted tests and router linting**

Run:

```bash
uv run pytest tests/unit/api/test_llm_quota.py tests/unit/api/test_text_processing_router.py -n 0 -v
uv run ruff check api/config.py api/llm_quota.py api/schemas/text_processing_schemas.py api/routers/text_processing_router.py tests/unit/api/test_llm_quota.py tests/unit/api/test_text_processing_router.py
```

Expected: PASS for the targeted tests and no Ruff findings in touched files.

- [ ] **Step 7: Commit**

```bash
git add api/config.py api/api.yaml api/llm_quota.py api/schemas/text_processing_schemas.py api/routers/text_processing_router.py tests/unit/api/test_llm_quota.py tests/unit/api/test_text_processing_router.py
git commit -m "feat: enrich full-text quota and fallback metadata"
```

## Task 2: Add backend phenopacket serialization endpoint for case workspace export

**Files:**
- Create: `api/schemas/phenopacket_schemas.py`
- Create: `api/routers/phenopacket_router.py`
- Modify: `api/main.py`
- Test: `tests/unit/api/test_phenopacket_router.py`

- [ ] **Step 1: Write the failing phenopacket export router tests**

```python
def test_phenopacket_router_exports_bundle_with_optional_sidecar(client):
    response = client.post(
        "/api/v1/phenopackets/export",
        json={
            "case_id": "case-1",
            "case_label": "Case 1",
            "input_text": "Patient had recurrent seizures.",
            "include_annotation_sidecar": True,
            "phenotypes": [
                {
                    "hpo_id": "HP:0001250",
                    "label": "Seizure",
                    "assertion_status": "affirmed",
                    "source_chunk_ids": [1],
                    "text_attributions": [
                        {
                            "chunk_id": 1,
                            "start_char": 8,
                            "end_char": 26,
                            "matched_text_in_chunk": "recurrent seizures",
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "phenopacket_json" in payload
    assert payload["annotation_sidecar"]["phenopacket_id"]
```

- [ ] **Step 2: Run the test to verify the endpoint is missing**

Run:

```bash
uv run pytest tests/unit/api/test_phenopacket_router.py -n 0 -v
```

Expected: FAIL because the router and schema do not exist.

- [ ] **Step 3: Define request/response schemas aligned with the current frontend payload shape**

Create `api/schemas/phenopacket_schemas.py` with models shaped like:

```python
class ExportTextAttributionRequest(BaseModel):
    chunk_id: int
    start_char: int | None = None
    end_char: int | None = None
    matched_text_in_chunk: str | None = None


class ExportPhenotypeRequest(BaseModel):
    hpo_id: str
    label: str
    assertion_status: str = "affirmed"
    source_chunk_ids: list[int] = Field(default_factory=list)
    text_attributions: list[ExportTextAttributionRequest] = Field(default_factory=list)
    confidence_band: Literal["high", "medium", "low"] | None = None


class PhenopacketExportRequest(BaseModel):
    case_id: str
    case_label: str | None = None
    input_text: str | None = None
    include_annotation_sidecar: bool = False
    phenotypes: list[ExportPhenotypeRequest] = Field(default_factory=list)
```

- [ ] **Step 4: Implement the export router using `phentrieve.phenopackets.utils.export_phenopacket_bundle`**

Create `api/routers/phenopacket_router.py` with:

```python
router = APIRouter(prefix="/api/v1/phenopackets", tags=["Phenopackets"])


@router.post("/export")
def export_phenopacket(request: PhenopacketExportRequest) -> dict[str, Any]:
    aggregated_results = [
        {
            "hpo_id": phenotype.hpo_id,
            "name": phenotype.label,
            "status": "negated" if phenotype.assertion_status == "negated" else "affirmed",
            "source_chunk_ids": phenotype.source_chunk_ids,
            "text_attributions": [
                item.model_dump() for item in phenotype.text_attributions
            ],
        }
        for phenotype in request.phenotypes
    ]

    return export_phenopacket_bundle(
        aggregated_results=aggregated_results,
        input_text=request.input_text,
        include_annotation_sidecar=request.include_annotation_sidecar,
    )
```

- [ ] **Step 5: Register the router in `api/main.py`**

Add:

```python
from api.routers.phenopacket_router import router as phenopacket_router
```

and include it with the other routers:

```python
app.include_router(phenopacket_router)
```

- [ ] **Step 6: Run the focused tests**

Run:

```bash
uv run pytest tests/unit/api/test_phenopacket_router.py -n 0 -v
uv run ruff check api/schemas/phenopacket_schemas.py api/routers/phenopacket_router.py api/main.py tests/unit/api/test_phenopacket_router.py
```

Expected: PASS and no Ruff findings.

- [ ] **Step 7: Commit**

```bash
git add api/schemas/phenopacket_schemas.py api/routers/phenopacket_router.py api/main.py tests/unit/api/test_phenopacket_router.py
git commit -m "feat: add phenopacket export api for case workspace"
```

## Task 3: Create a per-turn full-text workspace store and preserve conversation-shell boundaries

**Files:**
- Create: `frontend/src/stores/fullTextWorkspace.js`
- Create: `frontend/src/constants/fullTextWorkspace.js`
- Modify: `frontend/src/stores/conversation.js`
- Test: `frontend/src/test/stores/fullTextWorkspace.test.js`

- [ ] **Step 1: Write failing store tests for per-turn isolation and sidebar mode**

```javascript
import { createPinia, setActivePinia } from 'pinia';
import { describe, it, expect, beforeEach } from 'vitest';
import { useFullTextWorkspaceStore } from '../../stores/fullTextWorkspace';

describe('fullTextWorkspace store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  it('isolates workspace state by conversation turn id', () => {
    const store = useFullTextWorkspaceStore();

    store.initializeTurn('turn-a');
    store.initializeTurn('turn-b');
    store.setSidebarMode('turn-a', 'inspector');

    expect(store.turns['turn-a'].sidebarMode).toBe('inspector');
    expect(store.turns['turn-b'].sidebarMode).toBe('case');
  });

  it('stores quota banners independently from conversation history', () => {
    const store = useFullTextWorkspaceStore();
    store.initializeTurn('turn-a');
    store.setQuotaBanner('turn-a', {
      fallbackReason: 'llm_quota_exhausted',
      quotaResetAt: '2026-04-23T00:00:00+00:00',
    });

    expect(store.turns['turn-a'].quotaBanner.fallbackReason).toBe('llm_quota_exhausted');
  });
});
```

- [ ] **Step 2: Run the store tests to verify the module is missing**

Run:

```bash
cd frontend && npm run test:run -- src/test/stores/fullTextWorkspace.test.js
```

Expected: FAIL because the store and constants file do not exist.

- [ ] **Step 3: Add workspace constants for mode names and confidence bands**

Create `frontend/src/constants/fullTextWorkspace.js`:

```javascript
export const SIDEBAR_MODE_CASE = 'case';
export const SIDEBAR_MODE_INSPECTOR = 'inspector';
export const CONFIDENCE_BANDS = Object.freeze({
  high: { label: 'High', min: 0.85 },
  medium: { label: 'Medium', min: 0.6 },
  low: { label: 'Low', min: 0.0 },
});
```

- [ ] **Step 4: Implement the workspace store with per-turn state**

Create `frontend/src/stores/fullTextWorkspace.js`:

```javascript
import { defineStore } from 'pinia';
import { ref } from 'vue';
import {
  SIDEBAR_MODE_CASE,
  SIDEBAR_MODE_INSPECTOR,
} from '../constants/fullTextWorkspace';

function createEmptyTurnState() {
  return {
    expanded: false,
    sidebarMode: SIDEBAR_MODE_CASE,
    selectedPhenotypeId: null,
    selectedSpanId: null,
    quotaBanner: null,
    activeCaseId: null,
    cases: [],
    undoStack: [],
    redoStack: [],
  };
}

export const useFullTextWorkspaceStore = defineStore('fullTextWorkspace', () => {
  const turns = ref({});

  function initializeTurn(turnId) {
    if (!turns.value[turnId]) {
      turns.value[turnId] = createEmptyTurnState();
    }
  }

  function setSidebarMode(turnId, mode) {
    initializeTurn(turnId);
    turns.value[turnId].sidebarMode = mode;
  }

  function setQuotaBanner(turnId, banner) {
    initializeTurn(turnId);
    turns.value[turnId].quotaBanner = banner;
  }

  return {
    turns,
    initializeTurn,
    setSidebarMode,
    setQuotaBanner,
  };
});
```

- [ ] **Step 5: Keep `conversationStore` focused on chat history**

In `frontend/src/stores/conversation.js`, add a narrow helper instead of moving workspace state into it:

```javascript
function findQueryById(id) {
  return queryHistory.value.find((item) => item.id === id) || null;
}
```

and export it from the store return block. Do not add sidebar or annotation state there.

- [ ] **Step 6: Run the focused frontend tests**

Run:

```bash
cd frontend && npm run test:run -- src/test/stores/fullTextWorkspace.test.js
cd frontend && npm run test:run -- src/test/conversation.test.js
```

Expected: PASS with no regressions in conversation-store behavior.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/stores/fullTextWorkspace.js frontend/src/constants/fullTextWorkspace.js frontend/src/stores/conversation.js frontend/src/test/stores/fullTextWorkspace.test.js
git commit -m "feat: add per-turn full-text workspace store"
```

## Task 4: Replace the current static full-text results block with an expandable workspace shell

**Files:**
- Create: `frontend/src/components/FullTextAnnotationWorkspace.vue`
- Modify: `frontend/src/components/ResultsDisplay.vue`
- Modify: `frontend/src/test/components/ResultsDisplay.test.js`
- Create: `frontend/src/test/components/FullTextAnnotationWorkspace.test.js`

- [ ] **Step 1: Write failing component tests for expansion and quota banner rendering**

```javascript
it('renders text processing results inside an expandable workspace shell', async () => {
  const component = (await import('../../components/ResultsDisplay.vue')).default;
  const wrapper = mount(component, {
    props: {
      resultType: 'textProcess',
      responseData: {
        meta: {
          extraction_backend: 'standard',
          fallback_reason: 'llm_quota_exhausted',
          llm_quota_reset_at: '2026-04-23T00:00:00+00:00',
        },
        processed_chunks: [{ chunk_id: 1, text: 'Patient had recurrent seizures.', status: 'affirmed' }],
        aggregated_hpo_terms: [{ hpo_id: 'HP:0001250', name: 'Seizure', confidence: 0.91, status: 'affirmed', evidence_count: 1, source_chunk_ids: [1], text_attributions: [] }],
      },
    },
    global: {
      plugins: [vuetify, i18n, pinia],
    },
  });

  expect(wrapper.findComponent({ name: 'FullTextAnnotationWorkspace' }).exists()).toBe(true);
  expect(wrapper.text()).toContain('Standard full-text analysis');
});
```

- [ ] **Step 2: Run the tests to verify the shell is missing**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/ResultsDisplay.test.js src/test/components/FullTextAnnotationWorkspace.test.js
```

Expected: FAIL because `FullTextAnnotationWorkspace` does not exist and `ResultsDisplay` still mounts the legacy chunk/term components directly.

- [ ] **Step 3: Create the expandable workspace shell**

Create `frontend/src/components/FullTextAnnotationWorkspace.vue`:

```vue
<template>
  <v-card class="full-text-workspace" rounded="lg" elevation="1">
    <v-card-title class="d-flex justify-space-between align-center">
      <div>
        <div class="text-subtitle-1">Full-text analysis</div>
        <div v-if="bannerText" class="text-caption text-medium-emphasis">
          {{ bannerText }}
        </div>
      </div>
      <v-btn
        variant="text"
        size="small"
        :icon="expanded ? 'mdi-chevron-up' : 'mdi-chevron-down'"
        @click="expanded = !expanded"
      />
    </v-card-title>
    <v-expand-transition>
      <div v-show="expanded" class="workspace-body">
        <slot />
      </div>
    </v-expand-transition>
  </v-card>
</template>
```

- [ ] **Step 4: Mount the workspace shell from `ResultsDisplay.vue`**

Replace the current text-processing block with:

```vue
<FullTextAnnotationWorkspace
  v-else-if="hasTextProcessResults"
  :response-data="responseData"
  :result-type="resultType"
>
  <div class="workspace-grid">
    <ChunkResultsView
      v-if="processedChunks.length > 0"
      ref="chunkResultsView"
      :chunks="processedChunks"
      :highlighted-attributions="highlightedAttributions"
    />
    <AggregatedTermsView
      :terms="responseData.aggregated_hpo_terms"
      :collected-phenotype-ids="collectedPhenotypeIds"
      @add-to-collection="(phenotype, status) => addToCollection(phenotype, status)"
      @highlight-attributions="updateHighlightedAttributions"
      @clear-attributions="clearHighlightedAttributions"
      @scroll-to-chunk="scrollToChunk"
    />
  </div>
</FullTextAnnotationWorkspace>
```

This is temporary scaffolding for later tasks; preserve query-mode rendering untouched.

- [ ] **Step 5: Run the focused component tests**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/ResultsDisplay.test.js src/test/components/FullTextAnnotationWorkspace.test.js
```

Expected: PASS and the legacy text-processing behavior remains covered.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/FullTextAnnotationWorkspace.vue frontend/src/components/ResultsDisplay.vue frontend/src/test/components/ResultsDisplay.test.js frontend/src/test/components/FullTextAnnotationWorkspace.test.js
git commit -m "feat: add expandable full-text workspace shell"
```

## Task 5: Build the annotated document pane with honest chunk-vs-span rendering

**Files:**
- Create: `frontend/src/components/AnnotatedDocumentPane.vue`
- Create: `frontend/src/components/AnnotationActionPopover.vue`
- Create: `frontend/src/test/components/AnnotatedDocumentPane.test.js`

- [ ] **Step 1: Write failing tests for banded evidence rendering and linked highlights**

```javascript
it('renders chunk-only evidence with gutter tint and span evidence with marks', async () => {
  const component = (await import('../../components/AnnotatedDocumentPane.vue')).default;
  const wrapper = mount(component, {
    props: {
      chunks: [
        {
          chunk_id: 1,
          text: 'Patient had recurrent seizures.',
          status: 'affirmed',
          evidence_mode: 'chunk',
        },
        {
          chunk_id: 2,
          text: 'Developmental delay was present.',
          status: 'affirmed',
          evidence_mode: 'span',
          annotations: [
            { id: 'ann-1', start_char: 0, end_char: 19, matched_text_in_chunk: 'Developmental delay' },
          ],
        },
      ],
    },
  });

  expect(wrapper.find('[data-chunk-evidence-mode=\"chunk\"]').exists()).toBe(true);
  expect(wrapper.find('mark[data-annotation-id=\"ann-1\"]').exists()).toBe(true);
});
```

- [ ] **Step 2: Run the test to verify the component is missing**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/AnnotatedDocumentPane.test.js
```

Expected: FAIL because the component does not exist.

- [ ] **Step 3: Create a document pane that prefers CSS Custom Highlight API**

Create `frontend/src/components/AnnotatedDocumentPane.vue` with a setup shaped like:

```vue
<script setup>
import { computed, onMounted, watch } from 'vue';

const props = defineProps({
  chunks: { type: Array, default: () => [] },
  selectedAnnotationIds: { type: Array, default: () => [] },
});

const supportsCustomHighlight =
  typeof CSS !== 'undefined' &&
  typeof Highlight !== 'undefined' &&
  typeof CSS.highlights !== 'undefined';

function applyCustomHighlights(chunkId, element, annotations) {
  if (!supportsCustomHighlight || !element) return;
  const ranges = annotations
    .filter((item) => item.start_char != null && item.end_char != null)
    .map((item) => {
      const range = new Range();
      range.setStart(element.firstChild, item.start_char);
      range.setEnd(element.firstChild, item.end_char);
      return range;
    });
  CSS.highlights.set(`chunk-${chunkId}`, new Highlight(...ranges));
}
</script>
```

For unsupported browsers, fall back to rendering `<mark>` fragments from pre-split segments.

- [ ] **Step 4: Make chunk honesty a visible primitive**

Render each chunk with:

```vue
<article
  :data-chunk-id="chunk.chunk_id"
  :data-chunk-evidence-mode="chunk.evidence_mode || 'chunk'"
  class="annotated-chunk"
>
  <div class="chunk-gutter" :class="`chunk-gutter--${chunk.evidence_mode || 'chunk'}`"></div>
  <div class="chunk-content">
    <p class="chunk-text">
      <template v-if="!needsFallbackMarks(chunk)">
        {{ chunk.text }}
      </template>
      <template v-else>
        <span v-for="segment in buildMarkedSegments(chunk)" :key="segment.key">
          <mark
            v-if="segment.annotationId"
            :data-annotation-id="segment.annotationId"
            :aria-details="`annotation-detail-${segment.annotationId}`"
          >
            {{ segment.text }}
          </mark>
          <span v-else>{{ segment.text }}</span>
        </span>
      </template>
    </p>
  </div>
</article>
```

Use CSS so `chunk-gutter--chunk` is a subdued tint and `mark` carries the true-offset styling.

- [ ] **Step 5: Add a small action popover for selection/click**

Create `frontend/src/components/AnnotationActionPopover.vue`:

```vue
<template>
  <v-menu :model-value="visible" location="top">
    <v-list density="compact">
      <v-list-item title="Inspect" @click="$emit('inspect')" />
      <v-list-item title="Add to case" @click="$emit('add-to-case')" />
      <v-list-item title="Change term" @click="$emit('change-term')" />
      <v-list-item title="Remove annotation" @click="$emit('remove-annotation')" />
    </v-list>
  </v-menu>
</template>
```

- [ ] **Step 6: Run the focused tests**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/AnnotatedDocumentPane.test.js
```

Expected: PASS with both honest chunk rendering and `<mark>` accessibility covered.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/AnnotatedDocumentPane.vue frontend/src/components/AnnotationActionPopover.vue frontend/src/test/components/AnnotatedDocumentPane.test.js
git commit -m "feat: add annotated document pane with honest evidence rendering"
```

## Task 6: Build the phenotype pane and switchable inspector mode

**Files:**
- Create: `frontend/src/components/PhenotypeFindingsPane.vue`
- Create: `frontend/src/components/AnnotationInspectorPanel.vue`
- Create: `frontend/src/test/components/PhenotypeFindingsPane.test.js`
- Modify: `frontend/src/components/FullTextAnnotationWorkspace.vue`

- [ ] **Step 1: Write failing tests for confidence bands and sidebar-mode switching**

```javascript
it('renders confidence bands instead of raw numeric scores in the primary list', async () => {
  const component = (await import('../../components/PhenotypeFindingsPane.vue')).default;
  const wrapper = mount(component, {
    props: {
      terms: [
        { hpo_id: 'HP:0001250', name: 'Seizure', confidence: 0.92, status: 'affirmed', source_chunk_ids: [1] },
      ],
    },
  });

  expect(wrapper.text()).toContain('High');
  expect(wrapper.text()).not.toContain('0.92');
});
```

- [ ] **Step 2: Run the tests to verify the new pane is missing**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/PhenotypeFindingsPane.test.js
```

Expected: FAIL because the pane does not exist.

- [ ] **Step 3: Create the findings pane with inspect-first interactions**

Create `frontend/src/components/PhenotypeFindingsPane.vue`:

```vue
<template>
  <section class="findings-pane">
    <v-list>
      <v-list-item
        v-for="term in terms"
        :key="term.hpo_id"
        @mouseenter="$emit('hover-term', term.hpo_id)"
        @mouseleave="$emit('clear-hover')"
        @click="$emit('inspect-term', term.hpo_id)"
      >
        <template #title>{{ term.name }}</template>
        <template #subtitle>{{ confidenceBand(term.confidence) }}</template>
      </v-list-item>
    </v-list>
  </section>
</template>
```

Use a helper:

```javascript
function confidenceBand(value) {
  if (value >= 0.85) return 'High';
  if (value >= 0.6) return 'Medium';
  return 'Low';
}
```

- [ ] **Step 4: Add a switchable inspector component**

Create `frontend/src/components/AnnotationInspectorPanel.vue` with explicit mode labeling and one-click return:

```vue
<template>
  <aside class="annotation-inspector annotation-inspector--active">
    <div class="panel-header">
      <div class="text-subtitle-2">Annotation Inspector</div>
      <v-btn variant="text" size="small" icon="mdi-arrow-left" @click="$emit('back')" />
    </div>
    <div v-if="selectedTerm">
      <div :id="`annotation-detail-${selectedTerm.hpo_id}`">{{ selectedTerm.name }}</div>
      <div class="text-caption">Confidence: {{ selectedTerm.confidence?.toFixed(2) }}</div>
    </div>
  </aside>
</template>
```

- [ ] **Step 5: Update `FullTextAnnotationWorkspace.vue` to switch the right sidebar explicitly**

Add:

```vue
<CaseWorkspacePanel
  v-if="sidebarMode === 'case'"
  :turn-id="turnId"
  @open-inspector="openInspector"
/>
<AnnotationInspectorPanel
  v-else
  :selected-term="selectedTerm"
  @back="setSidebarMode(turnId, 'case')"
/>
```

Bind `keydown.esc` on the workspace shell so `Esc` returns to case mode.

- [ ] **Step 6: Run the focused tests**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/PhenotypeFindingsPane.test.js src/test/components/FullTextAnnotationWorkspace.test.js
```

Expected: PASS with inspect-first and sidebar-mode-switch rules covered.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/PhenotypeFindingsPane.vue frontend/src/components/AnnotationInspectorPanel.vue frontend/src/components/FullTextAnnotationWorkspace.vue frontend/src/test/components/PhenotypeFindingsPane.test.js frontend/src/test/components/FullTextAnnotationWorkspace.test.js
git commit -m "feat: add phenotype pane and switchable annotation inspector"
```

## Task 7: Replace full-text collection behavior with a case workspace and backend phenopacket export

**Files:**
- Create: `frontend/src/components/CaseWorkspacePanel.vue`
- Modify: `frontend/src/services/PhentrieveService.js`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/composables/usePhenotypeCollection.js`
- Create: `frontend/src/test/components/CaseWorkspacePanel.test.js`
- Modify: `frontend/src/test/services/PhentrieveService.test.js`
- Modify: `frontend/src/test/components/QueryInterface.test.js`

- [ ] **Step 1: Write failing tests for auto-case creation and export API usage**

```javascript
it('creates a case when adding from full-text and no case is selected', async () => {
  const store = useFullTextWorkspaceStore();
  store.initializeTurn('turn-a');

  store.addPhenotypeToActiveCase('turn-a', {
    hpo_id: 'HP:0001250',
    label: 'Seizure',
  });

  expect(store.turns['turn-a'].cases).toHaveLength(1);
  expect(store.turns['turn-a'].activeCaseId).toBeTruthy();
});
```

- [ ] **Step 2: Run the tests to verify case behavior is missing**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/CaseWorkspacePanel.test.js src/test/services/PhentrieveService.test.js src/test/components/QueryInterface.test.js
```

Expected: FAIL because the case workspace component and export API call are missing.

- [ ] **Step 3: Extend the workspace store with case management**

In `frontend/src/stores/fullTextWorkspace.js`, add:

```javascript
function createCaseDraft() {
  return {
    id: crypto.randomUUID(),
    label: `Case ${new Date().toLocaleTimeString()}`,
    phenotypes: [],
    subject: { id: '', sex: null, dateOfBirth: null },
  };
}

function ensureActiveCase(turnId) {
  initializeTurn(turnId);
  const turn = turns.value[turnId];
  if (!turn.activeCaseId) {
    const nextCase = createCaseDraft();
    turn.cases.push(nextCase);
    turn.activeCaseId = nextCase.id;
  }
  return turn.cases.find((item) => item.id === turn.activeCaseId);
}
```

- [ ] **Step 4: Create the case workspace panel**

Create `frontend/src/components/CaseWorkspacePanel.vue`:

```vue
<template>
  <aside class="case-workspace">
    <div class="panel-header">
      <div class="text-subtitle-2">Case Workspace</div>
      <v-btn size="small" variant="text" @click="$emit('create-case')">New case</v-btn>
    </div>
    <v-list>
      <v-list-item
        v-for="item in cases"
        :key="item.id"
        :active="item.id === activeCaseId"
        @click="$emit('select-case', item.id)"
      >
        <template #title>{{ item.label }}</template>
        <template #subtitle>{{ item.phenotypes.length }} phenotypes</template>
      </v-list-item>
    </v-list>
    <v-btn block color="primary" @click="$emit('add-all')">Add all extracted phenotypes</v-btn>
    <v-btn block variant="tonal" @click="$emit('export-case')">Export Phenopacket</v-btn>
  </aside>
</template>
```

- [ ] **Step 5: Route phenopacket export through the backend**

In `frontend/src/services/PhentrieveService.js`, add:

```javascript
async exportPhenopacket(exportData) {
  const response = await axios.post(`${API_URL}/phenopackets/export`, exportData);
  return response.data;
}
```

In `frontend/src/composables/usePhenotypeCollection.js`, replace direct JSON assembly with:

```javascript
const bundle = await PhentrieveService.exportPhenopacket({
  case_id: activeCase.id,
  case_label: activeCase.label,
  input_text: activeCase.inputText,
  include_annotation_sidecar: true,
  phenotypes: activeCase.phenotypes,
});
downloadText(bundle.phenopacket_json, `${activeCase.id}.phenopacket.json`);
if (bundle.annotation_sidecar) {
  downloadJson(bundle.annotation_sidecar, `${activeCase.id}.annotations.json`);
}
```

- [ ] **Step 6: Wire add behavior in `QueryInterface.vue`**

When the current item is a full-text response, route add actions into the active case instead of the old global collection:

```javascript
handleAddPhenotypeFromFullText(turnId, phenotype, status) {
  const workspace = useFullTextWorkspaceStore();
  workspace.addPhenotypeToActiveCase(turnId, {
    ...phenotype,
    assertion_status: status || 'affirmed',
  });
}
```

Preserve current query-mode add behavior for non-full-text results.

- [ ] **Step 7: Run the focused frontend tests**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/CaseWorkspacePanel.test.js src/test/services/PhentrieveService.test.js src/test/components/QueryInterface.test.js
```

Expected: PASS with case creation, case switching, and backend phenopacket export covered.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/CaseWorkspacePanel.vue frontend/src/services/PhentrieveService.js frontend/src/components/QueryInterface.vue frontend/src/composables/usePhenotypeCollection.js frontend/src/test/components/CaseWorkspacePanel.test.js frontend/src/test/services/PhentrieveService.test.js frontend/src/test/components/QueryInterface.test.js
git commit -m "feat: add case workspace and phenopacket export flow"
```

## Task 8: Integrate the new workspace panes end-to-end and finalize fallback UX

**Files:**
- Modify: `frontend/src/components/ResultsDisplay.vue`
- Modify: `frontend/src/components/FullTextAnnotationWorkspace.vue`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/test/components/ResultsDisplay.test.js`
- Modify: `frontend/src/test/components/QueryInterface.test.js`

- [ ] **Step 1: Write failing integration tests for the fallback banner and unified pane composition**

```javascript
it('shows a one-line fallback banner while keeping the full-text workspace layout intact', async () => {
  const component = (await import('../../components/FullTextAnnotationWorkspace.vue')).default;
  const wrapper = mount(component, {
    props: {
      responseData: {
        meta: {
          extraction_backend: 'standard',
          fallback_reason: 'llm_quota_exhausted',
          llm_quota_reset_at: '2026-04-23T00:00:00+00:00',
        },
        processed_chunks: [{ chunk_id: 1, text: 'Patient had recurrent seizures.', status: 'affirmed' }],
        aggregated_hpo_terms: [],
      },
    },
  });

  expect(wrapper.text()).toContain('Richer LLM analysis is unavailable for today');
  expect(wrapper.text()).toContain('resets');
});
```

- [ ] **Step 2: Run the focused tests to verify the banner copy is not wired**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/ResultsDisplay.test.js src/test/components/QueryInterface.test.js
```

Expected: FAIL until the new metadata is translated into the workspace banner and the panes are integrated.

- [ ] **Step 3: Finish composing the workspace**

In `frontend/src/components/FullTextAnnotationWorkspace.vue`, replace temporary legacy slots with:

```vue
<div class="workspace-layout">
  <AnnotatedDocumentPane
    :chunks="workspaceChunks"
    :selected-annotation-ids="selectedAnnotationIds"
    @inspect-annotation="openInspector"
    @selection-action="openActionPopover"
  />
  <PhenotypeFindingsPane
    :terms="workspaceTerms"
    @inspect-term="inspectTerm"
    @hover-term="hoverTerm"
    @clear-hover="clearHover"
  />
</div>
```

Drive the right sidebar from store state and response metadata.

- [ ] **Step 4: Compute the user-local reset banner text**

Add a banner helper:

```javascript
function localResetText(resetAtIso) {
  if (!resetAtIso) return '';
  return new Date(resetAtIso).toLocaleString();
}
```

and render:

```vue
<v-alert
  v-if="responseData.meta?.fallback_reason === 'llm_quota_exhausted'"
  type="info"
  density="compact"
  variant="tonal"
>
  Richer LLM analysis is unavailable for today. Standard full-text analysis is shown instead. LLM access resets {{ localResetText(responseData.meta?.llm_quota_reset_at || responseData.meta?.quota_reset_at) }}.
</v-alert>
```

- [ ] **Step 5: Keep the conversation shell behavior intact**

In `QueryInterface.vue`, continue adding the full-text response to `conversationStore.queryHistory` exactly as now, but initialize the workspace store for text-processing turns after response arrival:

```javascript
if (useTextProcessMode) {
  const workspace = useFullTextWorkspaceStore();
  workspace.initializeTurn(queryId);
}
```

- [ ] **Step 6: Run final focused frontend verification**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/ResultsDisplay.test.js src/test/components/QueryInterface.test.js src/test/components/FullTextAnnotationWorkspace.test.js src/test/components/AnnotatedDocumentPane.test.js src/test/components/PhenotypeFindingsPane.test.js src/test/components/CaseWorkspacePanel.test.js src/test/stores/fullTextWorkspace.test.js src/test/services/PhentrieveService.test.js
```

Expected: PASS with the new workspace composition covered.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/ResultsDisplay.vue frontend/src/components/FullTextAnnotationWorkspace.vue frontend/src/components/QueryInterface.vue frontend/src/test/components/ResultsDisplay.test.js frontend/src/test/components/QueryInterface.test.js
git commit -m "feat: integrate unified full-text annotation workspace"
```

## Task 9: Full verification and documentation alignment

**Files:**
- Modify: `.planning/archived/2026-04-21-unified-full-text-annotation-workspace-design.md` (only if implementation changed the agreed design)

- [ ] **Step 1: Run Python verification for touched backend paths**

Run:

```bash
uv run pytest tests/unit/api/test_llm_quota.py tests/unit/api/test_text_processing_router.py tests/unit/api/test_phenopacket_router.py -n 0 -v
uv run ruff check api/config.py api/llm_quota.py api/schemas/text_processing_schemas.py api/schemas/phenopacket_schemas.py api/routers/text_processing_router.py api/routers/phenopacket_router.py api/main.py
```

Expected: PASS and no Ruff findings.

- [ ] **Step 2: Run frontend verification for touched components and store**

Run:

```bash
cd frontend && npm run test:run -- src/test/components/ResultsDisplay.test.js src/test/components/QueryInterface.test.js src/test/components/FullTextAnnotationWorkspace.test.js src/test/components/AnnotatedDocumentPane.test.js src/test/components/PhenotypeFindingsPane.test.js src/test/components/CaseWorkspacePanel.test.js src/test/stores/fullTextWorkspace.test.js src/test/services/PhentrieveService.test.js
cd frontend && npm run build
```

Expected: PASS and a successful production build.

- [ ] **Step 3: Run project-required checks before claiming completion**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected: PASS. If any unrelated failures appear, document them explicitly before claiming completion.

- [ ] **Step 4: Review the design spec for drift**

If the implementation forced any material behavior change, update:

```markdown
.planning/archived/2026-04-21-unified-full-text-annotation-workspace-design.md
```

with the exact final behavior, especially for:

- sidebar mode switching
- fallback banner content
- phenopacket export payload shape
- workspace state ownership

- [ ] **Step 5: Commit**

```bash
git add .planning/archived/2026-04-21-unified-full-text-annotation-workspace-design.md
git commit -m "docs: align full-text workspace spec with implementation"
```

## Self-Review

### Spec coverage

- unified workspace for both backends: Tasks 4, 5, 6, 8
- conversation shell preserved: Tasks 4 and 8
- case workspace replacing full-text collection behavior: Task 7
- phenopacket export reuse from backend/CLI: Task 2 and Task 7
- quota update to `5/day`: Task 1
- local reset-time fallback UX: Tasks 1 and 8
- CSS Custom Highlight API and chunk-vs-span honesty: Task 5
- accessibility and inspector-only numeric confidence: Tasks 5 and 6
- workspace state isolation by turn ID: Task 3
- monitoring/fallback metadata groundwork: Tasks 1 and 9

### Placeholder scan

- no `TODO` or deferred implementation markers
- no references to undefined files or stores
- each task includes explicit commands and code snippets

### Type and naming consistency

- workspace store name: `fullTextWorkspace`
- sidebar modes: `case` and `inspector`
- export router prefix: `/api/v1/phenopackets/export`
- fallback reason: `llm_quota_exhausted`
