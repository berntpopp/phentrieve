# PR 229 Review Follow-Up Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the responsibility load in `QueryInterface.vue` and `AnnotatedDocumentPane.vue` while preserving the PR 229 full-text behavior and the already-fixed regression coverage.

**Architecture:** Keep the existing product behavior and public component boundaries stable, but move derivation-heavy and browser-specific logic into focused composables and child components. `QueryInterface.vue` remains the top-level conversation/search shell, and `AnnotatedDocumentPane.vue` remains the document rendering and popover boundary.

**Tech Stack:** Vue 3, Vuetify 3, Pinia, Vue I18n, Vitest, existing frontend composable/component patterns.

---

## File Map

### Primary implementation files

- Create: `frontend/src/components/FullTextResponseReceipt.vue`
- Create: `frontend/src/composables/useUserNoteAnnotations.js`
- Create: `frontend/src/composables/useQueryInterfaceController.js`
- Create: `frontend/src/composables/useDocumentAnnotations.js`
- Create: `frontend/src/composables/useCustomHighlightOverlay.js`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/AnnotatedDocumentPane.vue`

### Secondary touch points if needed

- Modify: `frontend/src/components/ResultsDisplay.vue`
- Modify: `frontend/src/components/FullTextAnnotationWorkspace.vue`

### Tests

- Modify: `frontend/src/test/components/QueryInterface.test.js`
- Modify: `frontend/src/test/components/AnnotatedDocumentPane.test.js`
- Create: `frontend/src/test/components/FullTextResponseReceipt.test.js`
- Create: `frontend/src/test/composables/useUserNoteAnnotations.test.js`
- Create: `frontend/src/test/composables/useDocumentAnnotations.test.js`
- Create: `frontend/src/test/composables/useCustomHighlightOverlay.test.js`
- Keep green: `frontend/src/test/components/ResultsDisplay.test.js`
- Keep green: `frontend/src/test/components/PhenotypeFindingsPane.test.js`
- Keep green: `frontend/src/test/components/FullTextAnnotationWorkspace.test.js`

## Execution Rules

- Preserve public event names and payload shapes unless a test proves they are internal-only.
- Extract one seam at a time; do not batch both oversized components into one unreviewable diff.
- Treat the PR 229 regression tests as guardrails, not optional follow-up coverage.
- End with the repo checks required by `AGENTS.md`:
  - `make check`
  - `make typecheck-fast`
  - `make test`

### Task 1: Extract User-Note Annotation Derivation

**Files:**
- Create: `frontend/src/composables/useUserNoteAnnotations.js`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/test/components/QueryInterface.test.js`
- Create: `frontend/src/test/composables/useUserNoteAnnotations.test.js`

- [ ] **Step 1: Write failing composable tests for note range derivation**

```js
import { describe, expect, it } from 'vitest';
import {
  buildUserNoteSegments,
  resolveChunkOffsetsInNote,
  resolveMatchedTextRange,
} from '../../composables/useUserNoteAnnotations';

describe('useUserNoteAnnotations', () => {
  it('maps chunk offsets into note segments', () => {
    const note = 'Patient had seizures. Developmental delay documented.';
    const chunks = [
      { chunk_id: 1, text: 'Patient had seizures.' },
      { chunk_id: 2, text: 'Developmental delay documented.' },
    ];
    const terms = [
      {
        hpo_id: 'HP:0001250',
        name: 'Seizure',
        text_attributions: [{ chunk_id: 1, start_char: 12, end_char: 20 }],
      },
    ];

    const segments = buildUserNoteSegments({ note, chunks, terms, activePhenotypeId: null });
    expect(segments.some((segment) => segment.highlighted)).toBe(true);
  });
});
```

- [ ] **Step 2: Run the new composable test and confirm it fails**

Run: `cd frontend && npm run test:run -- src/test/composables/useUserNoteAnnotations.test.js`
Expected: FAIL because `useUserNoteAnnotations.js` does not exist yet.

- [ ] **Step 3: Create the composable with pure derivation helpers**

```js
export function resolveChunkOffsetsInNote(noteText, chunks) {
  const offsets = new Map();
  let cursor = 0;

  chunks.forEach((chunk) => {
    const chunkText = typeof chunk?.text === 'string' ? chunk.text : '';
    const index = noteText.indexOf(chunkText, cursor);
    if (index >= 0) {
      offsets.set(chunk.chunk_id, index);
      cursor = index + chunkText.length;
    }
  });

  return offsets;
}
```

- [ ] **Step 4: Move the inline note helper logic out of `QueryInterface.vue`**

```js
import {
  buildUserNoteSegments,
  formatDocumentSummaryMeta,
  summarizeDocumentQuery,
} from '../composables/useUserNoteAnnotations';
```

- [ ] **Step 5: Update `QueryInterface.test.js` to validate the component still renders expanded note highlights**

```js
expect(wrapper.find('[data-testid="user-note-expanded"]').exists()).toBe(true);
expect(wrapper.findAll('[data-testid="annotated-note-span"]').length).toBeGreaterThan(0);
```

- [ ] **Step 6: Run focused tests**

Run: `cd frontend && npm run test:run -- src/test/composables/useUserNoteAnnotations.test.js src/test/components/QueryInterface.test.js`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/composables/useUserNoteAnnotations.js \
  frontend/src/components/QueryInterface.vue \
  frontend/src/test/components/QueryInterface.test.js \
  frontend/src/test/composables/useUserNoteAnnotations.test.js
git commit -m "refactor: extract query note annotation helpers"
```

### Task 2: Extract Full-Text Receipt Rendering From QueryInterface

**Files:**
- Create: `frontend/src/components/FullTextResponseReceipt.vue`
- Modify: `frontend/src/components/QueryInterface.vue`
- Create: `frontend/src/test/components/FullTextResponseReceipt.test.js`
- Modify: `frontend/src/test/components/QueryInterface.test.js`

- [ ] **Step 1: Write failing component tests for the receipt surface**

```js
it('emits add-all-to-collection for normalized full-text phenotypes', async () => {
  const wrapper = mount(FullTextResponseReceipt, {
    props: {
      item: {
        id: 'turn-1',
        response: {
          aggregated_hpo_terms: [{ hpo_id: 'HP:0001250', name: 'Seizure', status: 'affirmed' }],
        },
      },
    },
  });

  await wrapper.get('[data-testid="full-text-response-add-all"]').trigger('click');
  expect(wrapper.emitted('add-all-to-collection')).toHaveLength(1);
});
```

- [ ] **Step 2: Run the new receipt tests and confirm they fail**

Run: `cd frontend && npm run test:run -- src/test/components/FullTextResponseReceipt.test.js`
Expected: FAIL because `FullTextResponseReceipt.vue` does not exist yet.

- [ ] **Step 3: Implement the child component using the existing receipt markup**

```vue
<FullTextResponseReceipt
  :item="item"
  :collected-phenotypes="conversationStore.collectedPhenotypes"
  :hovered-phenotype-id="getHoveredNotePhenotype(item.id)"
  @add-to-collection="handleAddToCollection"
  @add-all-to-collection="handleAddAllToCollection"
  @hover-phenotype="setHoveredNotePhenotype(item.id, $event)"
  @clear-hover="clearHoveredNotePhenotype(item.id)"
/>
```

- [ ] **Step 4: Replace the inline receipt block in `QueryInterface.vue` with the child component**

```vue
<FullTextResponseReceipt
  v-else-if="item.type === 'textProcess'"
  :item="item"
  :collected-phenotypes="conversationStore.collectedPhenotypes"
  :hovered-phenotype-id="getHoveredNotePhenotype(item.id)"
  @add-to-collection="handleAddToCollection"
  @add-all-to-collection="handleAddAllToCollection"
  @hover-phenotype="setHoveredNotePhenotype(item.id, $event)"
  @clear-hover="clearHoveredNotePhenotype(item.id)"
/>
```

- [ ] **Step 5: Extend `QueryInterface.test.js` to assert the new component is used**

```js
expect(wrapper.findComponent({ name: 'FullTextResponseReceipt' }).exists()).toBe(true);
```

- [ ] **Step 6: Run focused tests**

Run: `cd frontend && npm run test:run -- src/test/components/FullTextResponseReceipt.test.js src/test/components/QueryInterface.test.js`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/FullTextResponseReceipt.vue \
  frontend/src/components/QueryInterface.vue \
  frontend/src/test/components/FullTextResponseReceipt.test.js \
  frontend/src/test/components/QueryInterface.test.js
git commit -m "refactor: extract full-text response receipt"
```

### Task 3: Extract QueryInterface Bootstrap And Submit Orchestration

**Files:**
- Create: `frontend/src/composables/useQueryInterfaceController.js`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/test/components/QueryInterface.test.js`

- [ ] **Step 1: Write failing characterization tests around the controller-owned behavior**

```js
it('loads available models through the controller', async () => {
  const wrapper = await mountQueryInterface();
  expect(wrapper.vm.availableModels).toEqual(
    expect.arrayContaining([expect.objectContaining({ value: 'test-model' })])
  );
});
```

- [ ] **Step 2: Implement the controller composable by moving bootstrap and submit logic out of component methods**

```js
export function useQueryInterfaceController(deps) {
  async function fetchAvailableModels() {
    const config = await deps.service.getConfigInfo();
    return config.available_embedding_models.map((model) => ({
      text: model.id.split('/').pop(),
      value: model.id,
    }));
  }

  return {
    fetchAvailableModels,
    applyUrlParametersAndAutoSubmit,
    submitQuery,
  };
}
```

- [ ] **Step 3: Replace in-component method bodies with controller calls**

```js
const controller = useQueryInterfaceController({
  service: PhentrieveService,
  router: this.$router,
  route: this.$route,
  conversationStore: this.conversationStore,
});
```

- [ ] **Step 4: Keep existing component tests green and add one test that URL hydration still works**

```js
expect(wrapper.vm.forceEndpointMode).toBe('textProcess');
expect(wrapper.vm.showAdvancedOptions).toBe(true);
```

- [ ] **Step 5: Run focused tests**

Run: `cd frontend && npm run test:run -- src/test/components/QueryInterface.test.js src/test/components/QueryInterface.vuetify-registration.test.js`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/composables/useQueryInterfaceController.js \
  frontend/src/components/QueryInterface.vue \
  frontend/src/test/components/QueryInterface.test.js
git commit -m "refactor: extract query interface controller"
```

### Task 4: Extract Pure Document Annotation Helpers

**Files:**
- Create: `frontend/src/composables/useDocumentAnnotations.js`
- Modify: `frontend/src/components/AnnotatedDocumentPane.vue`
- Create: `frontend/src/test/composables/useDocumentAnnotations.test.js`
- Modify: `frontend/src/test/components/AnnotatedDocumentPane.test.js`

- [ ] **Step 1: Write failing tests for annotation normalization and fallback segmentation**

```js
import {
  buildMarkedSegments,
  getSpanAnnotations,
  needsFallbackMarks,
} from '../../composables/useDocumentAnnotations';

it('normalizes span annotations and merges adjacent fallback segments', () => {
  const chunk = {
    chunk_id: 1,
    text: 'Developmental delay was present.',
    evidence_mode: 'span',
    annotations: [{ start_char: 0, end_char: 19, matched_text_in_chunk: 'Developmental delay' }],
  };

  expect(getSpanAnnotations(chunk)).toHaveLength(1);
  expect(buildMarkedSegments(chunk, new Set())).toHaveLength(2);
});
```

- [ ] **Step 2: Run the new helper tests and confirm they fail**

Run: `cd frontend && npm run test:run -- src/test/composables/useDocumentAnnotations.test.js`
Expected: FAIL because `useDocumentAnnotations.js` does not exist yet.

- [ ] **Step 3: Implement pure helper exports by moving non-DOM logic out of the component**

```js
export function getSpanAnnotations(chunk) {
  if ((chunk.evidence_mode || 'chunk') !== 'span') {
    return [];
  }

  return (Array.isArray(chunk.annotations) ? chunk.annotations : []).filter(
    (item) => item.start_char != null && item.end_char != null
  );
}
```

- [ ] **Step 4: Update `AnnotatedDocumentPane.vue` to consume the helper module**

```js
import {
  buildMarkedSegments,
  getChunkAnnotationDetails,
  getSpanAnnotations,
  needsFallbackMarks,
} from '../composables/useDocumentAnnotations';
```

- [ ] **Step 5: Keep component-level tests focused on rendering outcomes rather than helper internals**

```js
expect(wrapper.findAll('mark[data-annotation-id]').length).toBeGreaterThan(0);
expect(wrapper.text()).toContain('Developmental delay');
```

- [ ] **Step 6: Run focused tests**

Run: `cd frontend && npm run test:run -- src/test/composables/useDocumentAnnotations.test.js src/test/components/AnnotatedDocumentPane.test.js`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/composables/useDocumentAnnotations.js \
  frontend/src/components/AnnotatedDocumentPane.vue \
  frontend/src/test/composables/useDocumentAnnotations.test.js \
  frontend/src/test/components/AnnotatedDocumentPane.test.js
git commit -m "refactor: extract document annotation helpers"
```

### Task 5: Extract Custom Highlight Overlay Lifecycle

**Files:**
- Create: `frontend/src/composables/useCustomHighlightOverlay.js`
- Modify: `frontend/src/components/AnnotatedDocumentPane.vue`
- Create: `frontend/src/test/composables/useCustomHighlightOverlay.test.js`
- Modify: `frontend/src/test/components/AnnotatedDocumentPane.test.js`

- [ ] **Step 1: Write failing tests for highlight overlay setup and unsupported-environment fallback**

```js
it('returns no-op highlight state when CSS highlights are unavailable', () => {
  const overlay = useCustomHighlightOverlay({ chunks: [], selectedAnnotationIds: [] });
  expect(overlay.supportsCustomHighlight).toBe(false);
});
```

- [ ] **Step 2: Run the new overlay tests and confirm they fail**

Run: `cd frontend && npm run test:run -- src/test/composables/useCustomHighlightOverlay.test.js`
Expected: FAIL because `useCustomHighlightOverlay.js` does not exist yet.

- [ ] **Step 3: Move range building, hitbox generation, style lifecycle, and refresh scheduling into the composable**

```js
export function useCustomHighlightOverlay({ chunks, selectedAnnotationIds, rootElement }) {
  function syncCustomHighlights() {}
  function clearCustomHighlights() {}
  function getAnchorTarget(anchor) {}

  return {
    supportsCustomHighlight,
    customHighlightHitboxes,
    syncCustomHighlights,
    clearCustomHighlights,
    getAnchorTarget,
  };
}
```

- [ ] **Step 4: Reduce `AnnotatedDocumentPane.vue` to rendering, popover state, and UI event translation**

```js
const overlay = useCustomHighlightOverlay({
  chunks: toRef(props, 'chunks'),
  selectedAnnotationIds: toRef(props, 'selectedAnnotationIds'),
  rootElement,
});
```

- [ ] **Step 5: Keep existing component tests for popover clearing, custom-highlight synchronization, and fallback marks passing**

```js
expect(wrapper.find('.popover-probe').attributes('data-visible')).toBe('true');
expect(set).toHaveBeenCalled();
```

- [ ] **Step 6: Run focused tests**

Run: `cd frontend && npm run test:run -- src/test/composables/useCustomHighlightOverlay.test.js src/test/components/AnnotatedDocumentPane.test.js`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/composables/useCustomHighlightOverlay.js \
  frontend/src/components/AnnotatedDocumentPane.vue \
  frontend/src/test/composables/useCustomHighlightOverlay.test.js \
  frontend/src/test/components/AnnotatedDocumentPane.test.js
git commit -m "refactor: extract custom highlight overlay logic"
```

### Task 6: Regression Sweep For PR 229 Behavior

**Files:**
- Modify only if test fallout shows a real integration mismatch:
  - `frontend/src/components/ResultsDisplay.vue`
  - `frontend/src/components/PhenotypeFindingsPane.vue`
  - `frontend/src/components/FullTextAnnotationWorkspace.vue`
- Verify:
  - `frontend/src/test/components/ResultsDisplay.test.js`
  - `frontend/src/test/components/PhenotypeFindingsPane.test.js`
  - `frontend/src/test/components/FullTextAnnotationWorkspace.test.js`

- [ ] **Step 1: Run the PR 229 regression suite**

Run: `cd frontend && npm run test:run -- src/test/components/ResultsDisplay.test.js src/test/components/PhenotypeFindingsPane.test.js src/test/components/FullTextAnnotationWorkspace.test.js`
Expected: PASS

- [ ] **Step 2: Fix only integration regressions introduced by extraction**

```js
defineEmits(['add-all-to-collection']);
```

- [ ] **Step 3: Re-run the regression suite**

Run: `cd frontend && npm run test:run -- src/test/components/ResultsDisplay.test.js src/test/components/PhenotypeFindingsPane.test.js src/test/components/FullTextAnnotationWorkspace.test.js`
Expected: PASS

- [ ] **Step 4: Commit if changes were required**

```bash
git add frontend/src/components/ResultsDisplay.vue \
  frontend/src/components/PhenotypeFindingsPane.vue \
  frontend/src/components/FullTextAnnotationWorkspace.vue
git commit -m "fix: preserve full-text regression behavior after refactor"
```

### Task 7: Final Verification

**Files:**
- Verify the full frontend and repo state; no planned code changes in this task.

- [ ] **Step 1: Run targeted frontend test sweep**

Run: `cd frontend && npm run test:run -- src/test/components/QueryInterface.test.js src/test/components/QueryInterface.vuetify-registration.test.js src/test/components/AnnotatedDocumentPane.test.js src/test/components/FullTextResponseReceipt.test.js src/test/composables/useUserNoteAnnotations.test.js src/test/composables/useDocumentAnnotations.test.js src/test/composables/useCustomHighlightOverlay.test.js`
Expected: PASS

- [ ] **Step 2: Run repo-required checks**

Run: `make check`
Expected: PASS

- [ ] **Step 3: Run fast type checks**

Run: `make typecheck-fast`
Expected: PASS

- [ ] **Step 4: Run the main test suite**

Run: `make test`
Expected: PASS

- [ ] **Step 5: Record any residual warnings explicitly if they are pre-existing and unchanged**

```md
- Existing warning remains: [exact warning text]
- Verified unchanged by this refactor: yes/no
```

- [ ] **Step 6: Final commit**

```bash
git add frontend/src/components frontend/src/composables frontend/src/test .planning
git commit -m "refactor: modularize full-text query and annotation panes"
```

## Self-Review Notes

- Spec coverage: the plan addresses both remaining review findings and keeps the
  fixed PR 229 regressions as explicit guardrails.
- Placeholder scan: file paths, commands, and verification points are concrete;
  optional integration edits are constrained to named files.
- Type consistency: composable and component names match the approved design
  document and the task sequence builds from pure helpers to integration.
