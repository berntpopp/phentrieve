# PR 229 Review Follow-Up Modularization Design

Date: 2026-04-23
Source review: `.planning/analysis/pr-229-deep-review-report.md`

## Goal

Address the remaining review concerns from PR 229 by reducing the practical
responsibility load in the two oversized frontend components without changing
the already-corrected full-text behavior:

- keep the `ResultsDisplay` bulk-add passthrough fix intact
- keep normalized assertion status behavior intact
- keep the unified full-text workspace behavior intact
- make the next round of frontend changes easier to reason about and test

## Scope

This follow-up covers only the structural debt called out in the review:

1. `frontend/src/components/QueryInterface.vue`
2. `frontend/src/components/AnnotatedDocumentPane.vue`

It also covers the supporting test updates needed to lock in extracted
boundaries.

This follow-up does not reopen backend work, API behavior, or the already-fixed
review regressions unless a refactor exposes an existing coupling that must be
moved to preserve behavior.

## Problem Summary

### `QueryInterface.vue`

`QueryInterface.vue` currently mixes too many jobs:

- search-shell rendering and mode switching
- query submission orchestration
- URL hydration and auto-submit behavior
- conversation rendering
- full-text receipt rendering
- user-note highlight segment construction
- phenotype collection/export wiring
- snackbar/error presentation

The issue is not only line count. The component is acting as renderer,
controller, and formatter at the same time. That makes targeted changes risky
because UI tweaks and behavioral logic share the same file and state surface.

### `AnnotatedDocumentPane.vue`

`AnnotatedDocumentPane.vue` currently mixes:

- rendered chunk markup
- fallback-mark segmentation
- annotation normalization
- custom highlight CSS lifecycle
- DOM range construction
- hitbox geometry generation
- popover anchoring and selection handling
- resize/layout refresh coordination

This produces a component that is difficult to test in focused ways because the
DOM rendering path and the browser-specific highlight machinery are tightly
interleaved.

## Constraints

- Keep the shipped user-visible behavior stable.
- Stay within established Vue 3, Vuetify, Pinia, and Vitest patterns already
  used in the repo.
- Prefer extraction over rewrite.
- Keep the refactor scoped to the review findings; do not redesign the entire
  full-text workflow.
- Treat `<600` lines as a directional target, not a reason to force unnatural
  fragmentation.

## Chosen Approach

Use targeted decomposition around natural seams that already exist in the code:

1. Move presentation-only or derivation-heavy logic into focused child
   components and composables.
2. Keep `QueryInterface.vue` as the top-level wiring component for the search
   shell and conversation store.
3. Keep `AnnotatedDocumentPane.vue` as the rendered document host and popover
   boundary.
4. Extract pure data-shaping logic separately from DOM- and browser-dependent
   highlight logic so tests can cover each layer independently.

This is preferred over a large rewrite because the review identified
maintainability debt, not a broken architecture. The existing behavior is
mostly correct; the problem is concentration of responsibilities.

## Design

### `QueryInterface.vue` target shape

`QueryInterface.vue` should retain only top-level orchestration concerns:

- search shell state
- store/composable integration
- submission entrypoints
- high-level conversation loop
- collection panel wiring

The following responsibilities should move out:

#### 1. Full-text receipt rendering

Create a dedicated child component for the bot-side full-text receipt and
phenotype list, for example:

- `frontend/src/components/FullTextResponseReceipt.vue`

This component should own:

- receipt header text
- receipt metadata display
- add-all button for full-text phenotypes
- list rendering of response phenotypes
- hover interactions for response-side phenotype rows

It should receive normalized inputs and emit narrow events such as:

- `add-to-collection`
- `add-all-to-collection`
- `hover-phenotype`
- `clear-hover`

`QueryInterface.vue` should stop embedding this receipt markup inline.

#### 2. User-note annotation derivation

Extract the note-segmentation and attribution-merging logic into a composable or
utility, for example:

- `frontend/src/composables/useUserNoteAnnotations.js`

This module should own:

- chunk offset resolution inside the submitted note
- fallback matched-text resolution
- merged highlight range construction
- user-note segment generation for rendering
- hover-aware segment activation

The extraction should make the logic testable without mounting the full
`QueryInterface.vue` component.

#### 3. Query submission and URL hydration logic

Extract the route/bootstrap and submission orchestration that currently lives in
methods/watchers into a focused composable, for example:

- `frontend/src/composables/useQueryInterfaceController.js`

This module should own:

- model loading bootstrap
- URL query hydration
- auto-submit decision logic
- mode auto-switch handling
- submit payload assembly for query vs full-text mode

`QueryInterface.vue` should still call the composable and bind the returned
state/actions, but should stop carrying the full orchestration body inline.

### `AnnotatedDocumentPane.vue` target shape

`AnnotatedDocumentPane.vue` should remain the visible pane component and event
boundary, but internal mechanics should be split into two layers.

#### 1. Annotation data and fallback-mark derivation

Extract pure annotation helpers into a composable or utility, for example:

- `frontend/src/composables/useDocumentAnnotations.js`

This module should own:

- annotation normalization and ID fallback
- span-annotation filtering
- chunk annotation detail generation
- fallback-mark segmentation
- selected-annotation decoration metadata

This layer should be pure or nearly pure so it can be unit-tested without DOM
or `CSS.highlights`.

#### 2. Custom highlight and hitbox lifecycle

Extract browser-specific highlight logic into a composable, for example:

- `frontend/src/composables/useCustomHighlightOverlay.js`

This module should own:

- highlight-name generation
- style element lifecycle
- DOM text-node collection
- range construction
- hitbox geometry generation
- layout refresh scheduling
- resize observer integration
- popover target refresh support

`AnnotatedDocumentPane.vue` should use the composable and keep only:

- chunk rendering
- popover visibility state
- event handlers that translate UI events into popover actions
- integration between rendered marks and the highlight composable

### Component boundary rules

- Child components should not reach into Pinia stores directly unless they
  already follow an established store-owning pattern.
- Composables should return plain state and actions; avoid hidden global state.
- Pure derivation helpers must not depend on `document`, `window`, or Vue
  component instance APIs.
- Browser-only highlight code must guard unsupported environments explicitly so
  fallback rendering remains intact.

## File Plan

### New frontend modules

- Create: `frontend/src/components/FullTextResponseReceipt.vue`
- Create: `frontend/src/composables/useUserNoteAnnotations.js`
- Create: `frontend/src/composables/useQueryInterfaceController.js`
- Create: `frontend/src/composables/useDocumentAnnotations.js`
- Create: `frontend/src/composables/useCustomHighlightOverlay.js`

Exact names may change slightly during implementation if adjacent code suggests
better alignment with existing naming, but the split of responsibilities should
remain the same.

### Existing files to modify

- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/AnnotatedDocumentPane.vue`
- Modify: `frontend/src/components/FullTextAnnotationWorkspace.vue` only if a
  prop/event interface must be tightened after extraction
- Modify: `frontend/src/components/ResultsDisplay.vue` only if receipt
  extraction changes how text-process responses are rendered

### Tests to add or update

- Modify: `frontend/src/test/components/QueryInterface.test.js`
- Modify: `frontend/src/test/components/AnnotatedDocumentPane.test.js`
- Create: `frontend/src/test/components/FullTextResponseReceipt.test.js`
- Create: `frontend/src/test/composables/useUserNoteAnnotations.test.js`
- Create: `frontend/src/test/composables/useDocumentAnnotations.test.js`
- Create: `frontend/src/test/composables/useCustomHighlightOverlay.test.js`
- Keep existing regression coverage in:
  - `frontend/src/test/components/ResultsDisplay.test.js`
  - `frontend/src/test/components/PhenotypeFindingsPane.test.js`

## Testing Strategy

### Preserve current review fixes

Keep the existing regression tests for:

- `ResultsDisplay` re-emitting `add-all-to-collection`
- `PhenotypeFindingsPane` assertion-status normalization

These are the direct protections for the bugs already fixed from the review.

### Add focused unit coverage for extracted logic

New tests should verify:

- note-range derivation from processed chunks and `text_attributions`
- matched-text fallback range resolution when offsets are unavailable
- merged segment construction for user-note rendering
- annotation normalization and fallback-mark segmentation
- custom-highlight range and hitbox generation under supported environments
- safe no-op behavior when custom highlight APIs are unavailable

### Keep integration coverage at the component boundary

`QueryInterface.test.js` and `AnnotatedDocumentPane.test.js` should still prove
that:

- the extracted modules are wired together correctly
- hover and add-to-collection events still flow through the same public
  component boundaries
- the popover and fallback rendering behavior remain stable

## Non-Goals

- Rewriting `QueryInterface.vue` to `<script setup>`
- Redesigning the search shell or unified full-text UX
- Changing store schemas
- Changing API payload shapes
- Refactoring unrelated large files outside the two review findings

## Risks And Mitigations

### Risk: behavior drift during extraction

Mitigation:

- extract one seam at a time
- keep current regression tests green before and after each slice
- avoid renaming public events or payload shapes unless strictly necessary

### Risk: browser-only highlight logic becomes harder to exercise

Mitigation:

- separate pure range/hitbox helpers from lifecycle wrappers where possible
- keep DOM-dependent tests focused and synthetic
- preserve fallback-mark rendering as the compatibility baseline

### Risk: line-count chasing creates worse abstractions

Mitigation:

- optimize for SRP and testability first
- use line count only as a pressure signal
- do not split code that naturally belongs together

## Success Criteria

This follow-up is complete when:

- `QueryInterface.vue` no longer embeds full-text receipt markup and no longer
  owns note-annotation derivation inline
- `AnnotatedDocumentPane.vue` no longer embeds the full custom-highlight and
  fallback-annotation implementation body inline
- extracted logic has dedicated tests
- existing review-fix regressions remain covered
- repo checks required by `AGENTS.md` pass

## Recommended Execution Order

1. Extract and test user-note annotation derivation from `QueryInterface.vue`
2. Extract and test `FullTextResponseReceipt.vue`
3. Extract and test query-interface controller/bootstrap logic
4. Extract pure document-annotation helpers from `AnnotatedDocumentPane.vue`
5. Extract custom-highlight overlay lifecycle
6. Run full frontend and repo verification
