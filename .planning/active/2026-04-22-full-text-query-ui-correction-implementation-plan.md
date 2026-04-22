# Full-Text Query UI Correction Implementation Plan

Status: implemented on branch `feat/unified-full-text-workspace` and committed in
`8881e0d` (`Refine full-text workspace and restore live LLM annotations`).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make query mode and full-text mode feel like one coherent product by
moving the primary mode switch into the search shell, unifying the shared case
and collection behavior, removing the broken legacy full-text layout, and
restoring congruent phenotype presentation.

**Architecture:** Keep the conversation-first shell. The final implementation
keeps the submitted note in the user bubble, keeps phenotypes in the bot
response, uses inline note annotations instead of a large evidence pane, and
reuses shared phenotype card primitives across query and full-text mode. The
work also included shared full-text service fixes for score propagation and a
local API env-loader fix so live LLM requests work in development.

**Tech Stack:** Vue 3, Vuetify 3, Pinia, Vue I18n, Vitest, Playwright, existing FastAPI backend.

---

## File Map

### Primary UI entrypoints

- Modify: `frontend/src/components/QueryInterface.vue`
  - Own the top-level mode switch behavior, auto-switch rules, and suppression of duplicate case-workspace surfaces in full-text flow.
- Modify: `frontend/src/components/AdvancedOptionsPanel.vue`
  - Remove primary-mode responsibility and reduce to secondary options only.
- Modify: `frontend/src/components/ResultsDisplay.vue`
  - Ensure query/full-text rendering uses shared phenotype presentation boundaries.

### Full-text workspace

- Modify: `frontend/src/components/FullTextAnnotationWorkspace.vue`
  - Remove the broken legacy three-column surface and keep full-text behavior
    aligned with the conversation-native layout.
- Modify: `frontend/src/components/CaseWorkspacePanel.vue`
  - Compact hierarchy, consistent shared usage, smaller actions, included-phenotypes preview.
- Modify or suppress: `frontend/src/components/PhenotypeCollectionPanel.vue`
  - Keep only the shared case-workspace path active; remove duplicate full-text behavior.

### Shared phenotype presentation

- Modify: `frontend/src/components/ResultItem.vue`
  - Extract or align the phenotype item visual language.
- Modify: `frontend/src/components/PhenotypeFindingsPane.vue`
  - Reuse the query-mode presentation language rather than bespoke list styling.
- Create: `frontend/src/components/PhenotypeCardRow.vue`
  - Shared presentational primitive for phenotype rows/cards.

### Tests

- Modify: `frontend/src/test/components/QueryInterface.test.js`
- Modify: `frontend/src/test/components/ResultsDisplay.test.js`
- Modify: `frontend/src/test/components/FullTextAnnotationWorkspace.test.js`
- Modify: `frontend/src/test/components/CaseWorkspacePanel.test.js`
- Modify: `frontend/src/test/components/PhenotypeFindingsPane.test.js`
- Create: `frontend/src/test/components/PhenotypeItemBase.test.js` if a shared component is introduced

### Backend and integration support

- Modify: `phentrieve/llm/types.py`
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `phentrieve/text_processing/full_text_service.py`
  - Preserve real score/confidence values in the stable full-text response.
- Modify: `api/run_api_local.py`
  - Load repo-root `.env` as well as `api/local_api_config.env` so local LLM
    full-text requests can use `GEMINI_API_KEY`.

## Execution Rules

- Keep `make dev-api` and `make dev-frontend` running throughout the UI correction pass.
- Work in small slices and stop after each slice for:
  - focused Playwright verification
  - user monkey testing on the live app
- Do not batch all UI corrections into one large refactor.
- Do not commit any API keys or secret configuration.

## Task 1: Search Shell Mode Switch

**Files:**
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/AdvancedOptionsPanel.vue`
- Test: `frontend/src/test/components/QueryInterface.test.js`

- [ ] Add explicit top-level mode state to `QueryInterface.vue` so the primary UX does not depend on advanced settings.
- [ ] Render right-side labeled mode pills in the search shell:
  - `Query`
  - `Full Text`
- [ ] Make the input morph in place:
  - single-line input in query mode
  - multi-line textarea in full-text mode
- [ ] Remove primary mode selection responsibility from `AdvancedOptionsPanel.vue`.
- [ ] Preserve advanced backend options for full-text mode only.
- [ ] Add or update unit tests covering:
  - explicit manual mode switching
  - input shell expansion/contraction
  - advanced-options no longer being the only mode switch path

**Checkpoint verification:**

- Run: `cd frontend && npm run test:run -- src/test/components/QueryInterface.test.js`
- Run a focused Playwright live-app flow to verify:
  - pills render on the right side of the search shell
  - clicking `Full Text` expands to textarea
  - clicking `Query` restores the compact input

## Task 2: Auto-Switch On Long Text

**Files:**
- Modify: `frontend/src/components/QueryInterface.vue`
- Test: `frontend/src/test/components/QueryInterface.test.js`

- [ ] Implement threshold-based auto-switch when long text is typed or pasted in query mode.
- [ ] Show a small inline helper notice when the auto-switch happens.
- [ ] Ensure manual override remains available after auto-switch.
- [ ] Add or update unit tests for:
  - long paste switches to full-text
  - helper notice appears
  - short query text does not auto-switch
  - user can switch back manually

**Checkpoint verification:**

- Run: `cd frontend && npm run test:run -- src/test/components/QueryInterface.test.js`
- Run Playwright flow with:
  - short query remains query mode
  - long clinical note triggers full-text mode and helper notice

## Task 3: Remove Duplicate Full-Text Case Surfaces And Legacy Layout

**Files:**
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/PhenotypeCollectionPanel.vue`
- Modify: `frontend/src/components/ResultsDisplay.vue`
- Test: `frontend/src/test/components/ResultsDisplay.test.js`
- Test: `frontend/src/test/components/QueryInterface.test.js`

- [x] Stop rendering the legacy global `PhenotypeCollectionPanel` as an active
  competing surface during full-text review.
- [x] Remove the broken legacy multi-column full-text layout.
- [x] Keep one shared case and collection interaction path active.
- [x] Ensure query mode still retains shared collection behavior in a consistent
  pattern.
- [x] Remove duplicate full-text case-workspace semantics from the page.
- [ ] Add or update tests asserting:
  - no duplicate case workspace surface is shown for full-text results
  - query mode retains shared case-workspace behavior

**Checkpoint verification:**

- Run: `cd frontend && npm run test:run -- src/test/components/ResultsDisplay.test.js src/test/components/QueryInterface.test.js`
- Run Playwright flow to verify:
  - only one case workspace is visible
  - no legacy floating FAB/drawer competes with the full-text review surface

## Task 4: Shared Add-To-Case And Collection Behavior

**Files:**
- Modify: `frontend/src/components/CaseWorkspacePanel.vue`
- Modify: `frontend/src/components/FullTextAnnotationWorkspace.vue`
- Test: `frontend/src/test/components/CaseWorkspacePanel.test.js`
- Test: `frontend/src/test/components/FullTextAnnotationWorkspace.test.js`

- [x] Keep the shared collection path available from full-text response cards.
- [x] Restore `Add all` for full-text phenotypes.
- [x] Preserve individual add-to-collection actions.
- [x] Keep query and full-text behavior consistent without forcing a dedicated
  full-text case sidebar.
- [x] Add/update tests for add-all and shared collection hooks.

**Checkpoint verification:**

- Run: `cd frontend && npm run test:run -- src/test/components/CaseWorkspacePanel.test.js src/test/components/FullTextAnnotationWorkspace.test.js`
- Run Playwright region screenshots for:
  - case workspace
  - expanded workspace

## Task 5: Shared Phenotype Presentation

**Files:**
- Modify: `frontend/src/components/ResultItem.vue`
- Modify: `frontend/src/components/PhenotypeFindingsPane.vue`
- Possibly create: `frontend/src/components/PhenotypeItemBase.vue`
- Test: `frontend/src/test/components/PhenotypeFindingsPane.test.js`
- Test: `frontend/src/test/components/ResultsDisplay.test.js`
- Test: `frontend/src/test/components/PhenotypeItemBase.test.js` if needed

- [x] Align full-text findings with the query-mode phenotype visual language.
- [x] Extract a shared presentational primitive instead of maintaining separate
  bespoke styles.
- [x] Keep mode-specific metadata additive:
  - query mode: retrieval-oriented data
  - full-text mode: evidence-oriented data
- [x] Ensure spacing, typography, chips, and action placement remain congruent.
- [x] Add or update tests to cover:
  - consistent shared item rendering
  - evidence metadata in full-text without visual divergence

**Checkpoint verification:**

- Run: `cd frontend && npm run test:run -- src/test/components/PhenotypeFindingsPane.test.js src/test/components/ResultsDisplay.test.js`
- Run Playwright visual comparison on:
  - query result item
  - full-text finding item

## Task 6: Note Annotations, Score Propagation, And Error Handling

**Files:**
- Modify: `frontend/src/components/FullTextAnnotationWorkspace.vue`
- Modify: `frontend/src/components/AnnotatedDocumentPane.vue` if needed
- Test: `frontend/src/test/components/FullTextAnnotationWorkspace.test.js`

- [x] Expand the clinical note by default for new full-text turns.
- [x] Hide the truncated preview when the note is expanded.
- [x] Build inline note annotations from chunk offsets and `text_attributions`.
- [x] Link hover state between response phenotypes and annotated note spans.
- [x] Show tooltip text with HPO label and ID on annotated spans.
- [x] Preserve real `confidence` and `score` values through the shared LLM
  full-text service.
- [x] Render full-text failures as real errors instead of a fake `0 findings`
  receipt.
- [x] Fix local API env loading for live LLM full-text development.
- [x] Add/update tests for:
  - default note expansion
  - note preview suppression
  - note span hover linkage
  - tooltip content
  - score propagation
  - error-state rendering

**Checkpoint verification:**

- Run: `cd frontend && npm run test:run -- src/test/components/FullTextAnnotationWorkspace.test.js`
- Run Playwright with the provided NAA10 sample note and verify:
  - one shared case workspace
  - stable document area
  - coherent findings pane

## Final Verification

- [x] Run: `make check`
- [x] Run: `make typecheck-fast`
- [x] Run: `make test`
- [x] Run focused frontend tests during iteration for `QueryInterface`,
  full-text workspace, findings pane, and store behavior
- [x] Verify live local LLM full-text request after API restart:
  - direct POST to `http://127.0.0.1:8734/api/v1/text/process`
  - NAA10 note returned `200`
  - `5` aggregated phenotypes
  - non-zero scores present
