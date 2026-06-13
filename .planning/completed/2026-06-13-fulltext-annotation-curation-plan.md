# Full-Text Annotation Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a curator interactively change/remove/add/revert HPO annotations on the live full-text note (re-query via single-term search), with persisted auto/manual provenance and synced findings; and delete the orphaned full-text annotation workspace cluster.

**Architecture:** Frontend-only. A persisted Pinia store (`fullTextCuration`, keyed by turnId) holds a note-relative annotation model seeded from the API response. A composable (`useFullTextCuration`) derives note segments + findings and runs re-queries through the existing `/api/v1/query/`. A new `FullTextNoteCurator` container hosts the existing highlighted note (`query/FullTextWorkspace.vue`, made interactive), the reused `AnnotationActionPopover`, and a new `HpoTermPickerDialog`. `FullTextResponseReceipt` reads curated findings.

**Tech Stack:** Vue 3 (script setup for new units; QueryInterface stays Options API), Vuetify 3, Pinia 3 + pinia-plugin-persistedstate, Vue I18n, Vitest + Vue Test Utils, Playwright, Lighthouse.

**Reference facts:** `PhentrieveService.queryHpo(payload)` POSTs `{text, model_name, language, num_results, similarity_threshold, query_assertion_language, detect_query_assertion, include_details}` to `/api/v1/query/` and returns `QueryResponse` `{query_text_received, language_detected, model_used_for_retrieval, query_assertion_status, results: HPOResultItem[]}`; `HPOResultItem = {hpo_id, label, similarity, definition, synonyms, component_scores}`. Conversation store persists `queryHistory` (items: `{id, timestamp, type, query, redactedQuery, response, loading, error}`) and `collectedPhenotypes` (`{hpo_id, label, assertion_status, added_at}`); `maxHistoryLength=50`. Locales: en/de/fr/es/nl. Existing i18n: `annotatedDocumentPane.actions.{inspect,addToCase,changeTerm,removeAnnotation}`.

---

## Phase A — Remove the orphaned workspace cluster (do first; keep suite green)

### Task A1: Drop FullTextAnnotationWorkspace from ResultsDisplay + dead chunk-highlight code

**Files:**
- Modify: `frontend/src/components/ResultsDisplay.vue`
- Test: `frontend/src/test/components/ResultsDisplay.test.js`

- [ ] **Step 1:** In `ResultsDisplay.vue` remove the template branch that renders `<FullTextAnnotationWorkspace>` (the `<div v-else-if="hasTextProcessResults">...</div>` block, ~lines 84-91). Leave the subsequent `v-else-if` error/empty branches intact so a textProcess error still renders the `v-else-if="error"` alert.
- [ ] **Step 2:** Remove the import (`import FullTextAnnotationWorkspace from './FullTextAnnotationWorkspace.vue';`) and its `components` registration entry.
- [ ] **Step 3:** Remove now-dead members: computed `hasTextProcessResults`; data `highlightedAttributions`; methods `updateHighlightedAttributions`, `highlightAttributions`, `clearHighlights`, `clearHighlightedAttributions`, `getHighlightedChunkSegments`. Grep the file first to confirm none are referenced in the remaining template: `grep -nE "hasTextProcessResults|highlightedAttributions|highlightAttributions|getHighlightedChunkSegments|clearHighlights|updateHighlightedAttributions" frontend/src/components/ResultsDisplay.vue` — only the definitions should remain; delete them.
- [ ] **Step 4:** In `ResultsDisplay.test.js` remove the three tests that stub/assert `FullTextAnnotationWorkspace` (the `stubs: { FullTextAnnotationWorkspace: ... }` mounts and `findComponent({ name: 'FullTextAnnotationWorkspace' })` assertions).
- [ ] **Step 5:** Run: `cd frontend && npx vitest run src/test/components/ResultsDisplay.test.js` — Expected: PASS.
- [ ] **Step 6:** Commit:
```bash
git add frontend/src/components/ResultsDisplay.vue frontend/src/test/components/ResultsDisplay.test.js
git commit -m "refactor(frontend): drop orphaned FullTextAnnotationWorkspace branch from ResultsDisplay"
```

### Task A2: Remove fullTextWorkspaceStore wiring from QueryInterface

**Files:**
- Modify: `frontend/src/components/QueryInterface.vue`
- Test: `frontend/src/test/components/QueryInterface.test.js`

- [ ] **Step 1:** Remove `import { useFullTextWorkspaceStore } from '../stores/fullTextWorkspace';` and the test-only exposure (`const fullTextWorkspaceStore = useFullTextWorkspaceStore();` and the two `fullTextWorkspaceStore` references in the returned/exposed object).
- [ ] **Step 2:** Grep `grep -n "fullTextWorkspaceStore\|useFullTextWorkspaceStore" frontend/src/components/QueryInterface.vue` — Expected: no matches.
- [ ] **Step 3:** In `QueryInterface.test.js` remove any assertion referencing `fullTextWorkspaceStore`.
- [ ] **Step 4:** Run: `cd frontend && npx vitest run src/test/components/QueryInterface.test.js` — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/components/QueryInterface.vue frontend/src/test/components/QueryInterface.test.js
git commit -m "refactor(frontend): remove test-only fullTextWorkspace store wiring from QueryInterface"
```

### Task A3: Delete the orphan files and their tests

**Files (delete):** `frontend/src/components/FullTextAnnotationWorkspace.vue`, `frontend/src/components/AnnotatedDocumentPane.vue`, `frontend/src/components/AnnotationInspectorPanel.vue`, `frontend/src/components/PhenotypeFindingsPane.vue`, `frontend/src/composables/useDocumentAnnotations.js`, `frontend/src/composables/useCustomHighlightOverlay.js`, `frontend/src/stores/fullTextWorkspace.js`, `frontend/src/constants/fullTextWorkspace.js`, `frontend/src/utils/annotationInspector.js` and the matching test files under `frontend/src/test/{components,composables,stores}/`.

- [ ] **Step 1:** Confirm no live refs remain: `grep -rnE "FullTextAnnotationWorkspace|AnnotatedDocumentPane|AnnotationInspectorPanel|PhenotypeFindingsPane|useDocumentAnnotations|useCustomHighlightOverlay|stores/fullTextWorkspace|constants/fullTextWorkspace|annotationInspector" frontend/src --include=*.vue --include=*.js | grep -v "/test/" | grep -v "\.test\."` — Expected: no matches (AnnotationActionPopover is NOT in this list; it stays).
- [ ] **Step 2:** Delete the source files:
```bash
cd frontend && git rm \
  src/components/FullTextAnnotationWorkspace.vue \
  src/components/AnnotatedDocumentPane.vue \
  src/components/AnnotationInspectorPanel.vue \
  src/components/PhenotypeFindingsPane.vue \
  src/composables/useDocumentAnnotations.js \
  src/composables/useCustomHighlightOverlay.js \
  src/stores/fullTextWorkspace.js \
  src/constants/fullTextWorkspace.js \
  src/utils/annotationInspector.js
```
- [ ] **Step 3:** Delete the matching tests (use `git rm` for those that exist; confirm names first with `ls frontend/src/test/components frontend/src/test/composables frontend/src/test/stores`):
```bash
cd frontend && git rm \
  src/test/components/FullTextAnnotationWorkspace.test.js \
  src/test/components/AnnotatedDocumentPane.test.js \
  src/test/components/PhenotypeFindingsPane.test.js \
  src/test/composables/useDocumentAnnotations.test.js \
  src/test/composables/useCustomHighlightOverlay.test.js \
  src/test/stores/fullTextWorkspace.test.js
```
- [ ] **Step 4:** Run full frontend suite + lint + build: `cd frontend && npm run test -- --run && npm run lint && npm run build` (or `make frontend-test-ci && make frontend-lint && make frontend-build`). Expected: all green; build emits no missing-import errors.
- [ ] **Step 5:** Commit:
```bash
git add -A
git commit -m "refactor(frontend): remove orphaned full-text annotation workspace cluster (~2.3k LOC)"
```

---

## Phase B — Pure annotation model (extend useUserNoteAnnotations.js)

The Annotation type (note-relative): `{ id, hpoId, label, status: 'affirmed'|'negated', spans: [{start,end,text}], origin: 'auto'|'manual', confidence: number|null, replacedFrom?: {hpoId,label} }`.

### Task B1: seedAnnotationsFromResponse

**Files:**
- Modify: `frontend/src/composables/useUserNoteAnnotations.js`
- Test: `frontend/src/test/composables/useUserNoteAnnotations.test.js`

- [ ] **Step 1: Write failing test** (append to the existing test file):
```js
import { seedAnnotationsFromResponse } from '../../composables/useUserNoteAnnotations';

describe('seedAnnotationsFromResponse', () => {
  const note = 'She has microcephaly and feeding problems.';
  const response = {
    processed_chunks: [{ chunk_id: 1, text: 'She has microcephaly and feeding problems.' }],
    aggregated_hpo_terms: [
      { hpo_id: 'HP:0000252', name: 'Microcephaly', status: 'present', confidence: 1,
        text_attributions: [{ chunk_id: 1, start_char: 8, end_char: 20, matched_text_in_chunk: 'microcephaly' }] },
      { hpo_id: 'HP:0011968', name: 'Feeding difficulties', status: 'present', confidence: 0.9,
        text_attributions: [{ chunk_id: 1, start_char: 25, end_char: 41, matched_text_in_chunk: 'feeding problems' }] },
    ],
  };

  it('builds note-relative auto annotations with origin auto', () => {
    const result = seedAnnotationsFromResponse({ note, response });
    expect(result).toHaveLength(2);
    const micro = result.find((a) => a.hpoId === 'HP:0000252');
    expect(micro.label).toBe('Microcephaly');
    expect(micro.origin).toBe('auto');
    expect(micro.status).toBe('affirmed');
    expect(micro.spans[0]).toMatchObject({ start: 8, end: 20, text: 'microcephaly' });
  });

  it('maps absent/present status to negated/affirmed', () => {
    const r = seedAnnotationsFromResponse({ note, response: {
      ...response,
      aggregated_hpo_terms: [{ ...response.aggregated_hpo_terms[0], status: 'absent' }],
    }});
    expect(r[0].status).toBe('negated');
  });

  it('falls back to matched-text search when offsets do not resolve', () => {
    const r = seedAnnotationsFromResponse({ note, response: {
      processed_chunks: [{ chunk_id: 9, text: 'unrelated' }],
      aggregated_hpo_terms: [{ hpo_id: 'HP:0000252', name: 'Microcephaly', status: 'present', confidence: 1,
        text_attributions: [{ chunk_id: 9, start_char: 0, end_char: 0, matched_text_in_chunk: 'microcephaly' }] }],
    }});
    expect(r[0].spans[0]).toMatchObject({ start: 8, end: 20 });
  });
});
```
- [ ] **Step 2:** Run `cd frontend && npx vitest run src/test/composables/useUserNoteAnnotations.test.js -t seedAnnotationsFromResponse` — Expected: FAIL (not exported).
- [ ] **Step 3: Implement** in `useUserNoteAnnotations.js`. Reuse existing `resolveChunkOffsetsInNote` and `resolveMatchedTextRange`. Add a status normalizer and the seeder:
```js
export function normalizeAnnotationStatus(status) {
  if (status === 'absent' || status === 'negated') return 'negated';
  if (status === 'uncertain') return 'uncertain';
  if (status === 'unknown') return 'unknown';
  return 'affirmed';
}

export function seedAnnotationsFromResponse({ note, response }) {
  const noteText = typeof note === 'string' ? note : '';
  const chunks = Array.isArray(response?.processed_chunks) ? response.processed_chunks : [];
  const terms = Array.isArray(response?.aggregated_hpo_terms) ? response.aggregated_hpo_terms : [];
  if (!noteText || terms.length === 0) return [];
  const chunkOffsets = resolveChunkOffsetsInNote(noteText, chunks);

  return terms
    .filter((t) => t && typeof t.hpo_id === 'string' && typeof t.name === 'string')
    .map((term, index) => {
      const spans = (Array.isArray(term.text_attributions) ? term.text_attributions : [])
        .map((attr) => {
          const base =
            chunkOffsets.get(attr?.chunk_id) ??
            chunkOffsets.get(String(attr?.chunk_id)) ??
            chunkOffsets.get(Number(attr?.chunk_id));
          if (base != null) {
            const start = Math.max(0, Math.min(base + Math.max(0, attr.start_char ?? 0), noteText.length));
            const end = Math.max(0, Math.min(base + Math.max(0, attr.end_char ?? 0), noteText.length));
            if (end > start) return { start, end, text: noteText.slice(start, end) };
          }
          const resolved = resolveMatchedTextRange(noteText, attr?.matched_text_in_chunk);
          return resolved ? { ...resolved, text: noteText.slice(resolved.start, resolved.end) } : null;
        })
        .filter(Boolean)
        .sort((a, b) => a.start - b.start);
      if (spans.length === 0) return null;
      const confidence = typeof term.confidence === 'number' ? term.confidence : null;
      return {
        id: `auto-${term.hpo_id}-${index}`,
        hpoId: term.hpo_id,
        label: term.name,
        status: normalizeAnnotationStatus(term.status),
        spans,
        origin: 'auto',
        confidence,
      };
    })
    .filter(Boolean);
}
```
- [ ] **Step 4:** Run the test again — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/composables/useUserNoteAnnotations.js frontend/src/test/composables/useUserNoteAnnotations.test.js
git commit -m "feat(frontend): seed note-relative annotation model from full-text response"
```

### Task B2: buildSegmentsFromAnnotations + deriveFindingsFromAnnotations

**Files:**
- Modify: `frontend/src/composables/useUserNoteAnnotations.js`
- Test: `frontend/src/test/composables/useUserNoteAnnotations.test.js`

- [ ] **Step 1: Write failing tests:**
```js
import { buildSegmentsFromAnnotations, deriveFindingsFromAnnotations } from '../../composables/useUserNoteAnnotations';

describe('buildSegmentsFromAnnotations', () => {
  const note = 'She has microcephaly and feeding problems.';
  const annotations = [
    { id: 'a1', hpoId: 'HP:0000252', label: 'Microcephaly', status: 'affirmed', origin: 'auto', spans: [{ start: 8, end: 20, text: 'microcephaly' }] },
    { id: 'a2', hpoId: 'HP:0011968', label: 'Feeding difficulties', status: 'affirmed', origin: 'manual', spans: [{ start: 25, end: 41, text: 'feeding problems' }] },
  ];
  it('emits highlighted + plain segments in order with termIds and annotationIds', () => {
    const segs = buildSegmentsFromAnnotations(note, annotations);
    const highlighted = segs.filter((s) => s.highlighted);
    expect(highlighted).toHaveLength(2);
    expect(highlighted[0].termIds).toContain('HP:0000252');
    expect(highlighted[0].annotationIds).toContain('a1');
    expect(segs.map((s) => s.text).join('')).toBe(note);
  });
  it('merges overlapping spans and unions termIds', () => {
    const overlap = [
      { id: 'x', hpoId: 'HP:1', label: 'A', status: 'affirmed', origin: 'auto', spans: [{ start: 8, end: 16, text: 'microcep' }] },
      { id: 'y', hpoId: 'HP:2', label: 'B', status: 'affirmed', origin: 'auto', spans: [{ start: 12, end: 20, text: 'cephaly' }] },
    ];
    const segs = buildSegmentsFromAnnotations(note, overlap).filter((s) => s.highlighted);
    expect(segs).toHaveLength(1);
    expect(segs[0].termIds).toEqual(expect.arrayContaining(['HP:1', 'HP:2']));
  });
});

describe('deriveFindingsFromAnnotations', () => {
  it('dedupes by hpoId, keeps origin, drops nothing when spans exist', () => {
    const findings = deriveFindingsFromAnnotations([
      { id: 'a1', hpoId: 'HP:1', label: 'A', status: 'affirmed', origin: 'auto', confidence: 1, spans: [{ start: 0, end: 1, text: 'x' }] },
      { id: 'a2', hpoId: 'HP:1', label: 'A', status: 'affirmed', origin: 'manual', confidence: null, spans: [{ start: 2, end: 3, text: 'y' }] },
      { id: 'a3', hpoId: 'HP:2', label: 'B', status: 'negated', origin: 'manual', confidence: null, spans: [{ start: 4, end: 5, text: 'z' }] },
    ]);
    expect(findings).toHaveLength(2);
    const a = findings.find((f) => f.hpo_id === 'HP:1');
    expect(a.origin).toBe('manual'); // any manual span marks the term manual
  });
});
```
- [ ] **Step 2:** Run `npx vitest run src/test/composables/useUserNoteAnnotations.test.js -t "buildSegmentsFromAnnotations|deriveFindingsFromAnnotations"` — Expected: FAIL.
- [ ] **Step 3: Implement** (reuse the existing merge logic from `buildUserNoteSegments`):
```js
export function buildSegmentsFromAnnotations(noteText, annotations) {
  const text = typeof noteText === 'string' ? noteText : '';
  const labelById = new Map();
  const ranges = [];
  (Array.isArray(annotations) ? annotations : []).forEach((ann) => {
    labelById.set(ann.hpoId, ann.label);
    (ann.spans || []).forEach((s) => {
      if (s && s.end > s.start) ranges.push({ start: s.start, end: s.end, termIds: [ann.hpoId], annotationIds: [ann.id] });
    });
  });
  if (ranges.length === 0) return [{ key: 'plain-note', text, highlighted: false }];
  ranges.sort((l, r) => l.start - r.start);
  const merged = [];
  for (const range of ranges) {
    const prev = merged[merged.length - 1];
    if (prev && range.start <= prev.end) {
      prev.end = Math.max(prev.end, range.end);
      prev.termIds = [...new Set([...prev.termIds, ...range.termIds])];
      prev.annotationIds = [...new Set([...prev.annotationIds, ...range.annotationIds])];
    } else {
      merged.push({ ...range });
    }
  }
  const segments = [];
  let cursor = 0;
  merged.forEach((range, index) => {
    if (range.start > cursor) segments.push({ key: `plain-${index}-${cursor}`, text: text.slice(cursor, range.start), highlighted: false });
    segments.push({
      key: `mark-${index}-${range.start}`,
      text: text.slice(range.start, range.end),
      highlighted: true,
      termIds: range.termIds,
      annotationIds: range.annotationIds,
      tooltip: range.termIds.map((id) => `${labelById.get(id) || id} (${id})`).join(', '),
    });
    cursor = range.end;
  });
  if (cursor < text.length) segments.push({ key: `plain-tail-${cursor}`, text: text.slice(cursor), highlighted: false });
  return segments;
}

export function deriveFindingsFromAnnotations(annotations) {
  const byTerm = new Map();
  (Array.isArray(annotations) ? annotations : []).forEach((ann) => {
    if (!ann.spans || ann.spans.length === 0) return;
    const existing = byTerm.get(ann.hpoId);
    if (!existing) {
      byTerm.set(ann.hpoId, {
        hpo_id: ann.hpoId, name: ann.label, label: ann.label,
        status: ann.status, confidence: ann.confidence ?? null, origin: ann.origin,
      });
    } else if (ann.origin === 'manual') {
      existing.origin = 'manual';
    }
  });
  return [...byTerm.values()];
}
```
- [ ] **Step 4:** Run the tests — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/composables/useUserNoteAnnotations.js frontend/src/test/composables/useUserNoteAnnotations.test.js
git commit -m "feat(frontend): derive note segments and findings from annotation model"
```

---

## Phase C — Curation store

### Task C1: stores/fullTextCuration.js (persisted, keyed by turnId)

**Files:**
- Create: `frontend/src/stores/fullTextCuration.js`
- Test: `frontend/src/test/stores/fullTextCuration.test.js`

- [ ] **Step 1: Write failing test:**
```js
import { setActivePinia, createPinia } from 'pinia';
import { beforeEach, describe, it, expect } from 'vitest';
import { useFullTextCurationStore } from '../../stores/fullTextCuration';

const seedAnn = () => [
  { id: 'auto-HP:1-0', hpoId: 'HP:1', label: 'A', status: 'affirmed', origin: 'auto', confidence: 1, spans: [{ start: 0, end: 3, text: 'aaa' }] },
];

describe('fullTextCuration store', () => {
  beforeEach(() => setActivePinia(createPinia()));

  it('seeds a turn once (idempotent) and reads annotations', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.seedTurn('t1', []); // ignored, already seeded
    expect(s.annotationsForTurn('t1')).toHaveLength(1);
  });

  it('removeAnnotation drops it and supports undo', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.removeAnnotation('t1', 'auto-HP:1-0');
    expect(s.annotationsForTurn('t1')).toHaveLength(0);
    s.undo('t1');
    expect(s.annotationsForTurn('t1')).toHaveLength(1);
  });

  it('replaceTerm records replacedFrom and flips origin to manual', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.replaceTerm('t1', 'auto-HP:1-0', { hpoId: 'HP:9', label: 'Z', status: 'negated' });
    const ann = s.annotationsForTurn('t1')[0];
    expect(ann.hpoId).toBe('HP:9');
    expect(ann.origin).toBe('manual');
    expect(ann.replacedFrom).toMatchObject({ hpoId: 'HP:1', label: 'A' });
  });

  it('addManual appends a manual annotation', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', []);
    s.addManual('t1', { start: 5, end: 9, text: 'cccc' }, { hpoId: 'HP:7', label: 'M', status: 'affirmed' });
    const ann = s.annotationsForTurn('t1')[0];
    expect(ann.origin).toBe('manual');
    expect(ann.spans[0]).toMatchObject({ start: 5, end: 9 });
  });

  it('revert restores an auto annotation to its seeded value', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.replaceTerm('t1', 'auto-HP:1-0', { hpoId: 'HP:9', label: 'Z', status: 'affirmed' });
    s.revert('t1', 'auto-HP:1-0');
    expect(s.annotationsForTurn('t1')[0].hpoId).toBe('HP:1');
    expect(s.annotationsForTurn('t1')[0].origin).toBe('auto');
  });

  it('dropTurn clears state for evicted history items', () => {
    const s = useFullTextCurationStore();
    s.seedTurn('t1', seedAnn());
    s.dropTurn('t1');
    expect(s.isSeeded('t1')).toBe(false);
  });
});
```
- [ ] **Step 2:** Run `cd frontend && npx vitest run src/test/stores/fullTextCuration.test.js` — Expected: FAIL.
- [ ] **Step 3: Implement** (use the captured clone helper pattern; keep a per-turn `seed` snapshot for revert; persist via plugin):
```js
import { defineStore } from 'pinia';
import { ref } from 'vue';

function clone(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

export const useFullTextCurationStore = defineStore(
  'fullTextCuration',
  () => {
    // turnId -> { seeded, seed: Annotation[], annotations: Annotation[], undoStack: Annotation[][] }
    const turns = ref({});
    let manualSeq = 0;

    function ensure(turnId) {
      if (!turns.value[turnId]) {
        turns.value[turnId] = { seeded: false, seed: [], annotations: [], undoStack: [] };
      }
      return turns.value[turnId];
    }
    function pushUndo(turn) {
      turn.undoStack.push(clone(turn.annotations));
      if (turn.undoStack.length > 50) turn.undoStack.shift();
    }

    function seedTurn(turnId, annotations) {
      const turn = ensure(turnId);
      if (turn.seeded) return;
      turn.seed = clone(annotations) || [];
      turn.annotations = clone(annotations) || [];
      turn.seeded = true;
    }
    function isSeeded(turnId) {
      return Boolean(turns.value[turnId]?.seeded);
    }
    function annotationsForTurn(turnId) {
      return turns.value[turnId]?.annotations ?? [];
    }
    function removeAnnotation(turnId, id) {
      const turn = ensure(turnId);
      pushUndo(turn);
      turn.annotations = turn.annotations.filter((a) => a.id !== id);
    }
    function replaceTerm(turnId, id, term) {
      const turn = ensure(turnId);
      pushUndo(turn);
      turn.annotations = turn.annotations.map((a) => {
        if (a.id !== id) return a;
        const replacedFrom = a.replacedFrom || { hpoId: a.hpoId, label: a.label };
        return { ...a, hpoId: term.hpoId, label: term.label, status: term.status || a.status, origin: 'manual', replacedFrom };
      });
    }
    function addManual(turnId, span, term) {
      const turn = ensure(turnId);
      pushUndo(turn);
      manualSeq += 1;
      turn.annotations = [
        ...turn.annotations,
        { id: `manual-${term.hpoId}-${manualSeq}`, hpoId: term.hpoId, label: term.label,
          status: term.status || 'affirmed', spans: [clone(span)], origin: 'manual', confidence: null },
      ];
    }
    function revert(turnId, id) {
      const turn = ensure(turnId);
      const original = turn.seed.find((a) => a.id === id);
      if (!original) return;
      pushUndo(turn);
      turn.annotations = turn.annotations.map((a) => (a.id === id ? clone(original) : a));
    }
    function undo(turnId) {
      const turn = turns.value[turnId];
      if (!turn || turn.undoStack.length === 0) return;
      turn.annotations = turn.undoStack.pop();
    }
    function dropTurn(turnId) {
      delete turns.value[turnId];
    }

    return { turns, seedTurn, isSeeded, annotationsForTurn, removeAnnotation, replaceTerm, addManual, revert, undo, dropTurn };
  },
  {
    persist: {
      key: 'phentrieve-fulltext-curation',
      storage: localStorage,
      pick: ['turns'],
    },
  }
);
```
- [ ] **Step 4:** Run the test — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/stores/fullTextCuration.js frontend/src/test/stores/fullTextCuration.test.js
git commit -m "feat(frontend): persisted full-text curation store keyed by turn"
```

---

## Phase D — Orchestration composable

### Task D1: composables/useFullTextCuration.js

**Files:**
- Create: `frontend/src/composables/useFullTextCuration.js`
- Test: `frontend/src/test/composables/useFullTextCuration.test.js`

Exposes for a `turnId`: `ensureSeeded(item, noteText)`, reactive `segments`, `findings`, and `requery(text, options)` -> `HPOResultItem[]`. The component owns popover/dialog visual state; the composable owns data + queries.

- [ ] **Step 1: Write failing test** (mock the service):
```js
import { setActivePinia, createPinia } from 'pinia';
import { beforeEach, describe, it, expect, vi } from 'vitest';

vi.mock('../../services/PhentrieveService', () => ({
  default: { queryHpo: vi.fn(async () => ({ query_assertion_status: 'affirmed', results: [{ hpo_id: 'HP:9', label: 'Z', similarity: 0.8 }] })) },
}));
import PhentrieveService from '../../services/PhentrieveService';
import { useFullTextCuration } from '../../composables/useFullTextCuration';

const item = {
  id: 't1',
  response: {
    processed_chunks: [{ chunk_id: 1, text: 'She has microcephaly.' }],
    aggregated_hpo_terms: [{ hpo_id: 'HP:0000252', name: 'Microcephaly', status: 'present', confidence: 1,
      text_attributions: [{ chunk_id: 1, start_char: 8, end_char: 20, matched_text_in_chunk: 'microcephaly' }] }],
  },
};

describe('useFullTextCuration', () => {
  beforeEach(() => { setActivePinia(createPinia()); PhentrieveService.queryHpo.mockClear(); });

  it('seeds and exposes segments + findings', () => {
    const c = useFullTextCuration('t1');
    c.ensureSeeded(item, 'She has microcephaly.');
    expect(c.findings.value).toHaveLength(1);
    expect(c.segments.value.some((s) => s.highlighted)).toBe(true);
  });

  it('requery calls the service and returns candidates with assertion', async () => {
    const c = useFullTextCuration('t1');
    const res = await c.requery('microcephaly', { model_name: 'm', language: 'en', num_results: 8, similarity_threshold: 0.1 });
    expect(PhentrieveService.queryHpo).toHaveBeenCalledWith(expect.objectContaining({ text: 'microcephaly', include_details: true, num_results: 8 }));
    expect(res.assertion).toBe('affirmed');
    expect(res.results[0].hpo_id).toBe('HP:9');
  });

  it('replace + remove + add mutate findings', () => {
    const c = useFullTextCuration('t1');
    c.ensureSeeded(item, 'She has microcephaly.');
    const annId = c.annotations.value[0].id;
    c.replace(annId, { hpo_id: 'HP:9', label: 'Z' }, 'negated');
    expect(c.findings.value[0].hpo_id).toBe('HP:9');
    c.remove(annId);
    expect(c.findings.value).toHaveLength(0);
    c.addManual({ start: 0, end: 3, text: 'She' }, { hpo_id: 'HP:7', label: 'M' }, 'affirmed');
    expect(c.findings.value[0].hpo_id).toBe('HP:7');
  });
});
```
- [ ] **Step 2:** Run `cd frontend && npx vitest run src/test/composables/useFullTextCuration.test.js` — Expected: FAIL.
- [ ] **Step 3: Implement:**
```js
import { computed } from 'vue';
import PhentrieveService from '../services/PhentrieveService';
import { useFullTextCurationStore } from '../stores/fullTextCuration';
import {
  seedAnnotationsFromResponse,
  buildSegmentsFromAnnotations,
  deriveFindingsFromAnnotations,
} from './useUserNoteAnnotations';

export function useFullTextCuration(turnId) {
  const store = useFullTextCurationStore();

  function ensureSeeded(item, noteText) {
    if (store.isSeeded(turnId)) return;
    store.seedTurn(turnId, seedAnnotationsFromResponse({ note: noteText, response: item?.response }));
  }

  const annotations = computed(() => store.annotationsForTurn(turnId));
  const findings = computed(() => deriveFindingsFromAnnotations(annotations.value));
  const noteTextRef = { value: '' };
  const segments = computed(() => buildSegmentsFromAnnotations(noteTextRef.value, annotations.value));

  function setNoteText(text) { noteTextRef.value = typeof text === 'string' ? text : ''; }

  async function requery(text, options = {}) {
    const payload = {
      text,
      model_name: options.model_name,
      language: options.language ?? null,
      num_results: options.num_results ?? 8,
      similarity_threshold: options.similarity_threshold ?? 0.1,
      query_assertion_language: options.language ?? null,
      detect_query_assertion: true,
      include_details: true,
    };
    const data = await PhentrieveService.queryHpo(payload);
    return { assertion: data?.query_assertion_status ?? null, results: Array.isArray(data?.results) ? data.results : [] };
  }

  function replace(annotationId, term, assertion) {
    store.replaceTerm(turnId, annotationId, { hpoId: term.hpo_id, label: term.label, status: assertion });
  }
  function remove(annotationId) { store.removeAnnotation(turnId, annotationId); }
  function revert(annotationId) { store.revert(turnId, annotationId); }
  function undo() { store.undo(turnId); }
  function addManual(span, term, assertion) {
    store.addManual(turnId, span, { hpoId: term.hpo_id, label: term.label, status: assertion });
  }

  return { ensureSeeded, setNoteText, annotations, findings, segments, requery, replace, remove, revert, undo, addManual };
}
```
NOTE: `segments` must be reactive to note text. Implement `noteTextRef` as a real `ref` (import `ref`), not a plain object — update the implementation to `const noteTextRef = ref('')` and `noteTextRef.value` accordingly. (The test sets note text via `ensureSeeded`'s noteText; have `ensureSeeded` also call `setNoteText(noteText)`.)
- [ ] **Step 4:** Adjust `ensureSeeded` to call `setNoteText(noteText)` and switch `noteTextRef` to a `ref('')`. Run the test — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/composables/useFullTextCuration.js frontend/src/test/composables/useFullTextCuration.test.js
git commit -m "feat(frontend): useFullTextCuration composable (derive + requery + mutate)"
```

---

## Phase E — Components

### Task E1: HpoTermPickerDialog.vue

**Files:**
- Create: `frontend/src/components/HpoTermPickerDialog.vue`
- Test: `frontend/src/test/components/HpoTermPickerDialog.test.js`

Props: `modelValue:Boolean`, `mode:'replace'|'add'`, `spanText:String`, `candidates:Array<HPOResultItem>`, `loading:Boolean`, `assertion:String`. Emits: `update:modelValue`, `requery(text)`, `submit({term, assertion})`, `cancel`. UI: title (Change term / Annotate selection), a search `v-text-field` prefilled with `spanText` (debounced -> emit `requery`), a scrollable result list (`v-list` as listbox; each item shows `hpo_id`, `label`, score, truncated definition; selectable), an affirmed/negated `v-btn-toggle`, empty state when `candidates.length===0 && !loading`, footer `Cancel`/`Replace|Add` (disabled until a selection). `aria-labelledby` on the dialog card title; results have `role="option"`, list `role="listbox"`.

- [ ] **Step 1: Write failing test:**
```js
import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import HpoTermPickerDialog from '../../components/HpoTermPickerDialog.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en }, missing: () => '' });

function mountDialog(props = {}) {
  return mount(HpoTermPickerDialog, {
    attachTo: document.body,
    props: { modelValue: true, mode: 'replace', spanText: 'normal birth weight', candidates: [], loading: false, assertion: 'affirmed', ...props },
    global: { plugins: [vuetify, i18n] },
  });
}

describe('HpoTermPickerDialog', () => {
  it('shows an empty state when no candidates and not loading', () => {
    const w = mountDialog({ candidates: [] });
    expect(document.body.textContent).toMatch(/no matching|no results/i);
  });
  it('emits submit with the selected term and assertion', async () => {
    const w = mountDialog({ candidates: [{ hpo_id: 'HP:0001518', label: 'Small for gestational age', similarity: 0.74 }] });
    await w.vm.$nextTick();
    document.querySelector('[data-testid="hpo-candidate"]').click();
    await w.vm.$nextTick();
    document.querySelector('[data-testid="hpo-picker-submit"]').click();
    expect(w.emitted('submit')[0][0]).toMatchObject({ term: { hpo_id: 'HP:0001518' }, assertion: 'affirmed' });
  });
});
```
- [ ] **Step 2:** Run `cd frontend && npx vitest run src/test/components/HpoTermPickerDialog.test.js` — Expected: FAIL.
- [ ] **Step 3: Implement** the component (`<script setup>`, Vuetify `v-dialog`/`v-card`/`v-text-field`/`v-list`/`v-btn-toggle`/`v-btn`). Use `data-testid="hpo-candidate"`, `data-testid="hpo-picker-submit"`, `data-testid="hpo-picker-cancel"`. Debounce the search field (200ms) before emitting `requery`. i18n keys under a new `hpoTermPicker` namespace (Task F3).
- [ ] **Step 4:** Run the test — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/components/HpoTermPickerDialog.vue frontend/src/test/components/HpoTermPickerDialog.test.js
git commit -m "feat(frontend): HPO term picker dialog for curation"
```

### Task E2: Extend AnnotationActionPopover (selection mode + revert)

**Files:**
- Modify: `frontend/src/components/AnnotationActionPopover.vue`
- Test: `frontend/src/test/components/AnnotationActionPopover.test.js` (create)

- [ ] **Step 1: Write failing test:** mount with `mode='annotation'` asserts Change term + Remove visible and clicking emits `change-term`/`remove-annotation`; mount with `mode='selection'` asserts a single `annotate-selection` action; `canRevert=true` shows a `revert` action.
- [ ] **Step 2:** Run it — Expected: FAIL.
- [ ] **Step 3: Implement:** add props `mode:'annotation'|'selection'` (default 'annotation') and `canRevert:Boolean`. In `selection` mode render only an `Annotate selection` item emitting `annotate-selection`. In `annotation` mode render Change term, Remove, Add to collection, and (when `canRevert`) Revert (emitting `revert`). Add events `annotate-selection`, `revert` to `defineEmits`. Add i18n keys `annotatedDocumentPane.actions.annotateSelection`, `annotatedDocumentPane.actions.revert`, `addToCollection` (Task F3) — keep existing keys.
- [ ] **Step 4:** Run it — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/components/AnnotationActionPopover.vue frontend/src/test/components/AnnotationActionPopover.test.js
git commit -m "feat(frontend): popover gains selection mode and revert action"
```

### Task E3: Make note marks interactive in query/FullTextWorkspace.vue

**Files:**
- Modify: `frontend/src/components/query/FullTextWorkspace.vue`
- Test: `frontend/src/test/components/FullTextWorkspace.test.js`

- [ ] **Step 1: Write failing test:** expanded note with a highlighted segment carrying `annotationIds`. Assert the mark has `role="button"` and `aria-haspopup="menu"`; a `click` emits `span-activate` with `{ annotationIds, termIds, rect }`; pressing `Enter` (keydown) also emits `span-activate`; a `contextmenu` event emits `span-activate` and calls `preventDefault`.
- [ ] **Step 2:** Run it — Expected: FAIL.
- [ ] **Step 3: Implement:** on the `<mark>` add `role="button"`, `aria-haspopup="menu"`, keep `tabindex=0`; add handlers `@click`, `@contextmenu.prevent`, `@keydown.enter.prevent`, `@keydown.space.prevent` that emit `span-activate` with `{ annotationIds: segment.annotationIds, termIds: segment.termIds, rect: $event.currentTarget.getBoundingClientRect() }`. Add a `@mouseup` on the expanded container that, when there is a non-collapsed selection not starting/ending inside a mark, emits `text-select` with `{ text, range }` (use `window.getSelection()`). Update `aria-label` to `Edit annotation: <tooltip>`. Add the two emits to `emits:[...]`. Keep hover/tooltip behavior intact.
- [ ] **Step 4:** Run it — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/components/query/FullTextWorkspace.vue frontend/src/test/components/FullTextWorkspace.test.js
git commit -m "feat(frontend): make highlighted note spans interactive (click/right-click/keyboard + selection)"
```

### Task E4: FullTextNoteCurator.vue container

**Files:**
- Create: `frontend/src/components/FullTextNoteCurator.vue`
- Test: `frontend/src/test/components/FullTextNoteCurator.test.js`

Props: `item:Object`, `noteText:String`, `expanded:Boolean`, `activePhenotypeId:String`, `queryOptions:Object` (model/language/threshold/numResults). Emits: `toggle`, `hover`, `clear-hover`, `add-to-collection`. Internals: `useFullTextCuration(item.id)`; on setup `ensureSeeded(item, noteText)`; renders `FullTextWorkspace` (`:segments`/`:summary`/`:meta`/`:expanded`/`:active-phenotype-id`, forwards `toggle`/`hover`/`clear-hover`, handles `span-activate`/`text-select`), `AnnotationActionPopover` (target/anchor from the span rect; actions wired to composable + dialog), `HpoTermPickerDialog`, and a `v-snackbar` for "Removed — Undo". `summary`=`summarizeDocumentQuery(noteText)`, `meta`=`formatDocumentSummaryMeta(noteText)`.

- [ ] **Step 1: Write failing test:** mount with `item` having one auto annotation; assert it renders a highlighted span (via the real FullTextWorkspace child, `expanded:true`). Simulate `span-activate` from the child -> popover visible. Stub the dialog and emit `submit` -> assert composable replaced the term (findings reflect the new id; verify by spying on the store or asserting emitted state). Simulate child `remove` flow -> snackbar shown with undo; click undo -> annotation restored. (Use the real store via Pinia; mock `PhentrieveService`.)
- [ ] **Step 2:** Run it — Expected: FAIL.
- [ ] **Step 3: Implement** the container. Map popover actions: `change-term` -> open dialog mode 'replace' with `requery(spanText)`; `annotate-selection` -> dialog mode 'add'; `remove-annotation` -> `remove()` + show undo snackbar; `revert` -> `revert()`; `add-to-collection` -> emit `add-to-collection` with `{ hpo_id, label, assertion_status }` for the active annotation's term. Dialog `requery` event -> call composable `requery` and feed candidates/loading. Dialog `submit` -> `replace()` (replace mode) or `addManual()` (add mode).
- [ ] **Step 4:** Run it — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/components/FullTextNoteCurator.vue frontend/src/test/components/FullTextNoteCurator.test.js
git commit -m "feat(frontend): FullTextNoteCurator container wiring popover + dialog + store"
```

---

## Phase F — Integration + i18n

### Task F1: FullTextResponseReceipt reads curated findings + badge

**Files:**
- Modify: `frontend/src/components/FullTextResponseReceipt.vue`
- Test: `frontend/src/test/components/FullTextResponseReceipt.test.js`

- [ ] **Step 1: Write failing test:** with Pinia active, seed the curation store for the item's id with a manual annotation; mount the receipt; assert it shows that term and a `manual` badge (`data-testid="finding-origin-badge"`), and that adding all emits the curated set.
- [ ] **Step 2:** Run it — Expected: FAIL.
- [ ] **Step 3: Implement:** replace the `phenotypes` computed source with `useFullTextCuration(props.item.id)` -> `ensureSeeded(props.item, props.item.query??...)` then `findings`. (Note text for the receipt: pass a new `noteText` prop from QueryInterface to keep seeding identical; default to `item.query`.) Render a small `v-chip` badge per finding showing origin when `origin==='manual'`. Keep `mapTextProcessPhenotypeToResult`/add-to-collection behavior. Add `noteText` prop.
- [ ] **Step 4:** Run it — Expected: PASS.
- [ ] **Step 5:** Commit:
```bash
git add frontend/src/components/FullTextResponseReceipt.vue frontend/src/test/components/FullTextResponseReceipt.test.js
git commit -m "feat(frontend): findings receipt reflects curated annotations with origin badge"
```

### Task F2: Wire QueryInterface to FullTextNoteCurator

**Files:**
- Modify: `frontend/src/components/QueryInterface.vue`
- Test: `frontend/src/test/components/QueryInterface.test.js`

- [ ] **Step 1:** Replace the `<FullTextWorkspace .../>` usage (user bubble) with:
```vue
<FullTextNoteCurator
  :item="item"
  :note-text="getHistoryDisplayQuery(item)"
  :expanded="isUserNoteExpanded(item.id)"
  :active-phenotype-id="getHoveredNotePhenotype(item.id)"
  :query-options="curationQueryOptions"
  @toggle="toggleUserNote(item.id)"
  @hover="handleAnnotatedTextHover(item.id, $event)"
  @clear-hover="clearHoveredNotePhenotype(item.id)"
  @add-to-collection="handleAddToCollection"
/>
```
- [ ] **Step 2:** Add `:note-text="getHistoryDisplayQuery(item)"` to the existing `<FullTextResponseReceipt>` usage. Replace the `FullTextWorkspace` import/registration with `FullTextNoteCurator`. Add a `curationQueryOptions` computed returning `{ model_name: this.selectedModel, language: this.selectedLanguage, num_results: 8, similarity_threshold: this.similarityThreshold }` (reuse existing reactive option refs already present in the component).
- [ ] **Step 3:** Remove the now-unused local `buildUserNoteSegments` method and any imports from `useUserNoteAnnotations` that are no longer referenced in QueryInterface (let `npm run lint` flag unused — remove what it reports). Keep `getHistoryDisplayQuery`.
- [ ] **Step 4:** Add a history-eviction hook: where `queryHistory` is trimmed in the conversation store (or in QueryInterface on add), call `fullTextCurationStore.dropTurn(evictedId)` for any removed item. Minimal approach: in the conversation store's add path, after trimming, return evicted ids; or simpler — in QueryInterface watch `conversationStore.queryHistory` ids and drop curation for ids no longer present. Implement the watch in QueryInterface (`watch` on mapped ids; on removal call the curation store `dropTurn`).
- [ ] **Step 5:** Run `cd frontend && npx vitest run src/test/components/QueryInterface.test.js && npm run lint` — Expected: PASS + no lint errors.
- [ ] **Step 6:** Commit:
```bash
git add frontend/src/components/QueryInterface.vue frontend/src/test/components/QueryInterface.test.js
git commit -m "feat(frontend): mount FullTextNoteCurator in the full-text conversation turn"
```

### Task F3: i18n keys (en + de/fr/es/nl) and parity

**Files:**
- Modify: `frontend/src/locales/{en,de,fr,es,nl}.json`

- [ ] **Step 1:** Add to each locale under `annotatedDocumentPane.actions`: `annotateSelection`, `revert`, `addToCollection`; and a new top-level `hpoTermPicker` block with keys: `changeTitle`, `addTitle`, `searchLabel`, `assertionAffirmed`, `assertionNegated`, `replace`, `add`, `cancel`, `empty`, `loading`, `scoreLabel`. English values are authoritative; translate de/fr/es/nl (dispatch parallel translator subagents or translate inline).
- [ ] **Step 2:** Run `make frontend-i18n-check` — Expected: parity PASS (no missing/extra keys across locales).
- [ ] **Step 3:** Commit:
```bash
git add frontend/src/locales
git commit -m "i18n(frontend): curation menu + HPO term picker strings for all locales"
```

---

## Phase G — Verification

### Task G1: Full frontend gates

- [ ] **Step 1:** `make ci-frontend` — Expected: lint + format + i18n + tests PASS.
- [ ] **Step 2:** `make frontend-test-ci` and `make frontend-build-ci` — Expected: PASS, production build succeeds.
- [ ] **Step 3:** Fix any failures, re-run, then commit fixes if any.

### Task G2: Docker rebuild + Playwright E2E + Lighthouse

- [ ] **Step 1:** `make docker-build && make docker-up` (rebuild the stack at `http://localhost:8080/`).
- [ ] **Step 2:** Playwright: submit a clinical note in Full Text mode; expand the note; (a) right-click "normal birth weight" -> Remove -> assert the span and its finding disappear and Undo restores; (b) click "microcephaly" -> Change term -> dialog -> pick another term -> Replace -> assert the note + findings reflect the new term + a `manual` badge; (c) select unhighlighted text -> Annotate selection -> dialog -> pick a term -> Add -> assert a new highlighted span + finding; (d) reload the page -> assert curation persisted. Capture screenshots for each.
- [ ] **Step 3:** Re-run Lighthouse on `http://localhost:8080/`; assert accessibility >= 95 and `aria-tooltip-name` no longer failing.
- [ ] **Step 4:** `make docker-down` (leave stack running if the user wants to inspect — confirm). Record results in `.planning/analysis/2026-06-13-fulltext-annotation-curation-verification.md`.

---

## Self-Review

- **Spec coverage:** Change term (E1/E4/D1), Remove (C1/E4), Annotate selection (E3/E4), Revert (C1/E4), provenance+persistence (C1), findings sync (F1), collection/export (unchanged path via F2), a11y/tooltip fix (E3), orphan removal (A1-A3), testing (every task), edge cases (B1 fallback, C1 dropTurn, dialog empty state E1). Covered.
- **Type consistency:** Annotation shape `{id,hpoId,label,status,spans[{start,end,text}],origin,confidence,replacedFrom}` is consistent across B1/B2/C1/D1. Findings shape `{hpo_id,name,label,status,confidence,origin}` consistent B2/D1/F1. Composable API (`ensureSeeded,setNoteText,annotations,findings,segments,requery,replace,remove,revert,undo,addManual`) consistent D1/E4/F1.
- **Note on D1:** `noteTextRef` must be a real Vue `ref` for reactivity (flagged in Task D1 Step 3 note).
