# Full-Text Annotation Curation - Design

- Status: Approved (design), pending implementation
- Date: 2026-06-13
- Area: `frontend/` (Vue 3 / Vuetify / Pinia). No backend changes.
- Related: builds on the live full-text workspace shipped in PR #277
  (`query/FullTextWorkspace.vue` + `FullTextResponseReceipt.vue`).

## 1. Problem and Goal

In full-text mode the clinical note is rendered with phenotype spans
highlighted (`query/FullTextWorkspace.vue`) and a synced findings list
(`FullTextResponseReceipt.vue`). The highlights are display-only: there is no
way to correct the automatic annotations. Auto-extraction produces obvious
false positives (e.g. `normal birth weight` highlighted as a phenotype) and
misses spans a curator would want to add.

Goal: let a curator interactively correct the automatic annotations directly on
the highlighted note:

1. **Change term** - re-run a highlighted span's text through single-term query
   and replace the mapped HPO term with a chosen candidate.
2. **Remove** - delete a wrong auto-annotation and drop its term from findings.
3. **Annotate selection** - select previously-unhighlighted text, query it, and
   add a new manual annotation.
4. **Revert** - restore a curated annotation to its original automatic value.

Edits are per-turn, **persisted**, and **provenance-tracked** (auto vs manual,
original term retained on replace). Findings list, collection, and exports
reflect curation.

A second, explicitly-requested goal: **remove the orphaned full-text annotation
workspace cluster** (~2,300 lines) that was superseded by the PR #277 path and
is no longer reachable in the live UI.

## 2. Decisions (locked)

| Decision | Choice |
|----------|--------|
| Build target | Extend the live note path; reuse `AnnotationActionPopover`; keep findings synced. Do **not** revive the orphaned workspace. |
| Curation state | Tracked + persisted (auto vs manual provenance, original term on replace, revert-to-auto, exports honor curation). |
| Open trigger | Click + right-click (`contextmenu`, `preventDefault`) + keyboard (Enter/Space on focusable span). |
| Candidate picker | Full dialog: search field, scrollable ranked results with score + definition/synonyms, assertion toggle, Replace/Add action. |
| Backend | None. Reuse `POST /api/v1/query/` via `PhentrieveService.queryHpo`. |

## 3. Architecture (frontend-only)

```
QueryInterface (user bubble)            QueryInterface (bot bubble)
  +- FullTextNoteCurator (NEW)            +- FullTextResponseReceipt (MODIFIED)
       +- FullTextWorkspace (MODIFIED:         reads curated findings + badges
       |    emits span-activate / text-select)
       +- AnnotationActionPopover (REUSED, wired; + "Annotate selection" mode)
       +- HpoTermPickerDialog (NEW, full dialog)
                              |
            both read/write   v
            stores/fullTextCuration.js (NEW; persisted; keyed by turnId)
            composables/useFullTextCuration.js (NEW; orchestration)
            composables/useUserNoteAnnotations.js (EXTENDED; seed + derive)
```

New units (each small, single-purpose, independently testable):

- **`stores/fullTextCuration.js`** (Pinia, persisted via
  `pinia-plugin-persistedstate`, keyed by `turnId`). Holds the curated
  annotation model per turn plus a per-turn op stack for undo. Seeded from
  `item.response` the first time a turn is viewed. Borrows the turn-keying /
  deep-clone / stack-assertion helper *patterns* from the deleted
  `stores/fullTextWorkspace.js` (purpose-built; not a rename).
- **`composables/useFullTextCuration.js`** - the single orchestration unit for a
  given `turnId`. Exposes derived `noteSegments`, `findings`, popover/dialog
  state, and `requery(text)` (wraps `PhentrieveService.queryHpo`, reusing the
  turn's model/language/threshold). Keeps `QueryInterface` (>800 lines) from
  growing.
- **`components/FullTextNoteCurator.vue`** - container; renders the note,
  popover, and dialog; wires note interaction events to composable handlers.
- **`components/HpoTermPickerDialog.vue`** - presentational full dialog (search,
  ranked results, assertion toggle, Replace/Add).
- **`composables/useUserNoteAnnotations.js`** (extend) - add pure functions to
  (a) seed a note-relative annotation model from an API response and (b) derive
  segments from the annotation model. Existing `buildUserNoteSegments` logic is
  reused/refactored to consume the model.

## 4. Curation data model

Everything normalizes to **note-relative character offsets** so automatic and
manual annotations are uniform.

```js
annotation = {
  id,                                  // stable; `${origin}-${hpoId}-${seq}`
  hpoId, label,
  status: 'affirmed' | 'negated',
  spans: [{ start, end, text }],       // note-relative offsets
  origin: 'auto' | 'manual',
  confidence,                          // auto only (number) or null
  replacedFrom?: { hpoId, label },     // set when Change-term mutates an auto term
}
```

Seeding (`origin: 'auto'`): for each `aggregated_hpo_terms[].text_attributions`
entry, convert chunk-relative offsets to note offsets using the existing
`resolveChunkOffsetsInNote` (fallback: `resolveMatchedTextRange`). The raw API
response stays immutable - curation is a derived overlay, which gives clean
revert and provenance.

Per-turn store shape:

```js
turnState = {
  seeded: boolean,
  annotations: Annotation[],
  ops: Op[],          // for undo; { type, payload, before }
}
```

Operations (store actions): `seedFromResponse(turnId, response, noteText)`,
`removeAnnotation(turnId, id)`, `replaceTerm(turnId, id, term)`,
`addManual(turnId, span, term)`, `revert(turnId, id)`, `undo(turnId)`.

Derivations:

- **Note segments**: from `annotations[].spans` (merge overlaps; preserve
  multi-term tooltips), unchanged rendering contract for
  `query/FullTextWorkspace.vue`.
- **Findings**: dedupe `annotations` by `hpoId` -> one finding per term,
  carrying `origin` for the badge; removing a term's last span removes the
  finding.

## 5. Interaction flows

**Span menu (existing annotation).** Click / right-click / Enter on a span ->
`AnnotationActionPopover` with: *Change term*, *Remove annotation*, *Revert*
(only when `origin === 'manual'` or `replacedFrom` present), *Add to
collection*. Right-click calls `preventDefault` to suppress the native menu.

**Change term.** Opens `HpoTermPickerDialog` pre-seeded with the span text;
runs `requery`; shows ranked candidates; Replace -> `replaceTerm`. Original term
recorded in `replacedFrom`; `origin` becomes `manual`.

**Fresh selection.** `mouseup` with a non-empty selection that does not start/end
inside an existing mark -> popover with a single *Annotate selection* ->
dialog in *add* mode -> Add -> `addManual` with the selection's note-relative
offsets.

**Remove.** Drops the targeted span; if it was the term's last span the finding
disappears. Show a snackbar with **Undo** (-> `undo`).

**Assertion.** New/replaced annotations default their status from the query
response `query_assertion_status`; the dialog has an affirmed/negated toggle.

## 6. Findings / collection / export integration

- `FullTextResponseReceipt.vue`: source `phenotypes` from the curated findings
  (composable) instead of `item.response.aggregated_hpo_terms`. Add a small
  `auto` / `manual` badge per finding. Hover-sync and add-to-collection
  behavior unchanged.
- Collection and phenopacket/text export already read `collectedPhenotypes`
  (conversation store). Curated terms flow into the collection through the
  existing add-to-collection path, so exports honor curation with **no export
  code changes**.

## 7. Accessibility (and Lighthouse cleanup)

Baseline (measured 2026-06-13, desktop): Performance 57 (FCP 4.7s, LCP 5.5s;
TBT 0, CLS 0), Accessibility 95, Best-practices 100, SEO 100. Accessibility
failures: `aria-tooltip-name`, `heading-order`, `label-content-name-mismatch`.

In scope:

- Interactive marks become `role="button"`, `aria-haspopup="menu"`, accessible
  name `Edit annotation: <label>`; this **fixes `aria-tooltip-name`** in this
  area (tooltip activator gets a name).
- Popover and dialog: focus trap, `Esc` to close, focus returns to the
  originating span. Dialog labelled (`aria-labelledby`), results as a listbox.
- Keyboard: spans already `tabindex=0`; add Enter/Space to open the menu.
- Opportunistically fix `heading-order` / `label-content-name-mismatch` if they
  occur in touched files.

Out of scope: the Performance score / unused-JS bundle work (separate concern,
recorded as a follow-up, not bundled here).

## 8. Edge cases

- Overlapping spans: merge, retain multi-term tooltip text.
- Selection overlapping an existing mark: treat as re-annotate (offer dialog on
  the overlapping annotation rather than creating a malformed manual span).
- Query returns zero results: dialog shows an empty state; original annotation
  unchanged.
- Persisted curation re-hydrates on reload; segments/findings re-derive. If the
  note text cannot be matched (offset seeding fails), fall back to plain text
  (existing behavior) and skip curation for that turn.
- History eviction (conversation `maxHistoryLength`): drop the turn's curation
  state to avoid orphaned store entries.

## 9. Orphan removal (explicitly requested)

The orphaned full-text annotation workspace cluster - superseded by the PR #277
path and reachable only through `ResultsDisplay`'s `hasTextProcessResults`
branch (itself effectively dead because successful textProcess turns render
`FullTextResponseReceipt`, not `ResultsDisplay`).

**Delete (source):**

- `components/FullTextAnnotationWorkspace.vue`
- `components/AnnotatedDocumentPane.vue`
- `components/AnnotationInspectorPanel.vue`
- `components/PhenotypeFindingsPane.vue`
- `composables/useDocumentAnnotations.js`
- `composables/useCustomHighlightOverlay.js`
- `stores/fullTextWorkspace.js`
- `constants/fullTextWorkspace.js`
- `utils/annotationInspector.js`

**Delete (tests):** the matching files under `test/components/`,
`test/composables/`, `test/stores/` for each deleted unit.

**Edit:**

- `components/ResultsDisplay.vue`: remove the `FullTextAnnotationWorkspace`
  import/registration/template branch, the `hasTextProcessResults` computed, and
  the now-dead chunk-highlight `data`/methods
  (`highlightedAttributions`, `updateHighlightedAttributions`,
  `highlightAttributions`, `clearHighlights`, `clearHighlightedAttributions`,
  `getHighlightedChunkSegments`). The textProcess error case then falls through
  to the existing `v-else-if="error"` alert.
- `components/QueryInterface.vue`: remove the `useFullTextWorkspaceStore` import
  and its test-only exposure.
- `test/components/ResultsDisplay.test.js`, `test/components/QueryInterface.test.js`:
  drop assertions tied to removed code.

**Keep:** `components/AnnotationActionPopover.vue` (re-homed onto
`FullTextNoteCurator`).

The executor must re-run a reference grep before each deletion and confirm no
live (non-test) references remain.

## 10. Testing

- **Unit (Vitest)**: note-relative offset seeding + merge; each curation op
  (remove / replaceTerm / addManual / revert / undo); segment + findings
  derivation; store persistence + hydration; composable `requery` with a mocked
  `PhentrieveService`.
- **Component (Vue Test Utils)**: `HpoTermPickerDialog` (search, select,
  Replace/Add, empty state, assertion toggle); `FullTextNoteCurator` (menu via
  click / contextmenu / keyboard; replace / remove / add / revert; snackbar
  undo); updated `FullTextWorkspace` (emits) and `FullTextResponseReceipt`
  (curated findings + badge).
- **i18n**: add keys for new strings under existing `annotatedDocumentPane`/new
  namespace; `make frontend-i18n-check` parity for de/fr/es/nl.
- **Quality gates**: `make ci-frontend`, `make frontend-test-ci`,
  `make frontend-build-ci`; full `make ci-local` + `make security-python` before
  finishing.
- **E2E / manual**: after `make docker-build` + `make docker-up`, Playwright
  reproduces all four stories on `http://localhost:8080/` and a Lighthouse
  re-run confirms accessibility is not regressed (target: `aria-tooltip-name`
  resolved).

## 11. Out of scope (YAGNI)

- No new backend endpoint.
- No multi-note "cases" workspace (the deleted store's `cases` model is dropped).
- No reviving `AnnotatedDocumentPane` / chunk-pane layout.
- No bulk re-annotation or batch operations.
- No Performance/bundle optimization (tracked separately).
