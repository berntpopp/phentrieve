# Full-text Annotation Hover + Tooltip Fix — Design Spec

- Date: 2026-06-13
- Branch: `fix/fulltext-annotation-hover-tooltip`
- Status: Approved (design decisions confirmed with user)

## Problem

In Full Text mode, the submitted clinical note is rendered with HPO phenotype
mentions highlighted as `<mark>` spans, each wrapped in a Vuetify `v-tooltip`.
Two defects were reproduced and quantified live on `http://localhost:8080/`
using Playwright with the reporter's clinical text:

### Bug 1 — Hover collapses all highlights to one ("doesn't reset")
- Baseline: 22 annotated spans highlighted.
- On hovering any single span: total highlighted spans drops to **1** — every
  other annotation reverts to plain text. On leaving, it restores to 22.
- Root cause: `buildUserNoteSegments()`
  (`frontend/src/composables/useUserNoteAnnotations.js:95`) filters the term
  list down to only `activePhenotypeId`:
  ```js
  .filter((term) => !activePhenotypeId || term?.hpo_id === activePhenotypeId)
  ```
  The segment list is recomputed reactively on hover
  (`QueryInterface.vue:636` passes `activePhenotypeId` into it), so hover both
  (a) hides all non-hovered annotations and (b) re-creates the hovered `<mark>`
  DOM node (its `key` changes from `mark-${j}-…` to `mark-0-…`). The destroyed
  node never fires `mouseleave`, which is the "doesn't reset / one stays
  highlighted" perception.

### Bug 2 — Tooltip contrast is unreadable
- The tooltip renders on Vuetify's **default dark-grey** `#424242` while its
  inner text stays near-black/blue:
  - label `rgba(15,23,42,0.92)` on `#424242` → **1.72:1**
  - eyebrow `rgba(37,99,235,0.92)` on `#424242` → **1.83:1**
  - WCAG 2.1 SC 1.4.3 requires **4.5:1** for normal text.
- Root cause: the custom light background lives in a **scoped** `:deep()` rule
  (`FullTextWorkspace.vue:143`). Scoped `:deep(.annotated-note-tooltip)`
  compiles to `[data-v-xxx] .annotated-note-tooltip`, but the tooltip content
  is **teleported to `<body>`**, outside the scoped ancestor, so the rule never
  matches and Vuetify's default dark surface wins. The inner `__eyebrow` /
  `__label` color rules *do* apply (those divs are in the component template and
  carry the scope attribute) — producing dark-on-dark text.

## Approved Decisions

1. **Hover behavior:** keep ALL annotations highlighted; emphasize the hovered
   phenotype's mention(s). (No dimming of others.)
2. **Tooltip style:** theme-aware light surface using Vuetify theme tokens —
   readable at WCAG AA in light AND dark mode.
3. **Extra scope:** keyboard/focus a11y; audit legacy `AnnotatedDocumentPane.vue`;
   add regression tests.

## Solution

### Live path (the rendering at `localhost:8080`)
Components: `QueryInterface.vue` → `components/query/FullTextWorkspace.vue`,
backed by `composables/useUserNoteAnnotations.js`.

- **Bug 1 fix:** Remove the `activePhenotypeId` filter and parameter from
  `buildUserNoteSegments()`. Segments become a pure function of
  `(note, chunks, terms)` and are therefore **stable across hovers** — all
  spans stay rendered, no DOM re-creation, `mouseleave` always fires. The
  hovered phenotype's emphasis is driven purely by the existing
  `annotated-note-span--active` class bound to the `activePhenotypeId` prop
  (`segment.termIds.includes(activePhenotypeId)`), which is a class toggle, not
  a structural change. `QueryInterface.vue` stops passing `activePhenotypeId`
  into the builder (the prop still flows to the child for the emphasis class).
  Note↔findings cross-highlight already works via shared
  `hoveredTextProcessPhenotypes` state and is preserved.

- **Bug 2 fix:** Move the tooltip surface styling out of scoped `:deep()` into a
  **global** (unscoped) `<style>` rule targeting
  `.v-overlay__content.annotated-note-tooltip`, using:
  - background `rgb(var(--v-theme-surface))`
  - text `rgb(var(--v-theme-on-surface))`
  - eyebrow `rgb(var(--v-theme-primary))`
  - border `rgba(var(--v-theme-on-surface), 0.12)`
  This resolves correctly inside the teleported overlay (theme vars cascade from
  the app theme root) and adapts to light/dark. Target ≥ 4.5:1.

- **A11y (WCAG 1.4.13 + keyboard parity):** make each annotated `<mark>`
  focusable (`tabindex="0"`, `role="button"`, `aria-label` describing the linked
  phenotype). The Vuetify `v-tooltip` already opens on focus for bound
  activators; ensure focus emphasis matches hover emphasis and the tooltip is
  dismissible (Escape / blur). No change to the dismissible/hoverable defaults.

### Legacy path (audit + align only)
`components/AnnotatedDocumentPane.vue` (+ `AnnotationActionPopover.vue`), reached
via `ResultsDisplay.vue` → `FullTextAnnotationWorkspace.vue`. On inspection this
path uses the CSS Custom Highlight API with a click popover already styled via
theme tokens (`--v-theme-surface` / `--v-theme-on-surface`), so it does NOT have
the contrast bug or the collapse-filter. Action: verify it does not regress and
apply only minimal contrast/focus alignment if a concrete gap is found — no
rewrite.

## Tests

- `useUserNoteAnnotations.test.js`: with multiple terms AND a non-null
  `activePhenotypeId`, ALL mentions remain `highlighted` (no collapse); segment
  structure is independent of `activePhenotypeId`.
- `FullTextWorkspace.test.js`: rendering N segments yields N `<mark>` elements
  regardless of `activePhenotypeId`; `--active` toggles only on spans whose
  `termIds` include the active id; highlighted marks are focusable and expose an
  `aria-label`; tooltip uses the `annotated-note-tooltip` content-class.
- Live Playwright re-verification: hovering keeps 22 spans at 22; measured
  tooltip contrast ≥ 4.5:1; keyboard focus opens a readable tooltip.

## Verification Gate

- `make ci-frontend` (lint + format:check + unit tests + i18n where relevant)
- `make frontend-test-ci` and `make frontend-build-ci` (GitHub Actions parity)
- Live Playwright re-check on the running dev stack (Vite dev server against the
  running API), capturing before/after evidence.

## Implementation Plan (task order)

1. (TDD) Add failing tests for hover-stability in `useUserNoteAnnotations.test.js`.
2. Remove filter + `activePhenotypeId` param from `buildUserNoteSegments()`.
3. Update `QueryInterface.vue` builder call (drop `activePhenotypeId`).
4. `FullTextWorkspace.vue`: global theme-aware tooltip styling; a11y on marks;
   confirm `--active` emphasis treatment.
5. (TDD) Extend `FullTextWorkspace.test.js` for all-marks-stable + a11y + tooltip class.
6. Audit legacy `AnnotatedDocumentPane.vue` / `AnnotationActionPopover.vue`; align if needed.
7. Run CI-parity frontend gate; fix until green.
8. Live Playwright + contrast re-verification; capture evidence.

## Verification Results (2026-06-13, live on `localhost:8080`)

- Bug 1: hovering a span now keeps **all** marks highlighted (23/23, 45/45 across
  multiple notes); `--active` emphasis applies to only the hovered phenotype;
  highlight resets to 0 active on leave; `<mark>` keys are stable across hovers.
- Bug 2: tooltip now renders on a theme surface — measured contrast **21:1**
  (label, black on white) and **5.61:1** (eyebrow, primary on white), both above
  WCAG AA 4.5:1 (was 1.72:1 / 1.83:1).
- A11y: marks are focusable (`tabindex=0`) with `aria-label`; tooltip opens on
  keyboard focus (`open-on-focus`); focus applies the same emphasis as hover.
  axe-core (wcag2a/2aa/21a/21aa) on the annotated note region: **0 violations**.
  Fixed an adjacent `button-name` gap (note expand/collapse toggle now named).
- Legacy `AnnotatedDocumentPane.vue` / `AnnotationActionPopover.vue`: audited —
  uses the CSS Custom Highlight API + a theme-token light v-menu popover and
  `sr-only` detail spans; it has neither bug, so it was left unchanged (avoid
  unrelated refactors).
- Tests: full frontend suite green (324 passed). `make ci-frontend` green
  (ESLint, Prettier, tests, build). Python `make typecheck-fast` green (no
  Python changes).
- Local-dev LLM quota disabled via `docker-compose.dev.yml`
  (`PHENTRIEVE_ENV=development`); full-text annotation now succeeds where it
  previously returned the daily-limit message.

### Known pre-existing finding (out of scope)
- axe `aria-tooltip-name` flags 118 nodes app-wide — these are **other**
  Vuetify `v-tooltip` usages (footer icons, log viewer, mode pills), **not** the
  annotated-note tooltips (0 of the 118). Tracked for a separate a11y pass.

## Out of Scope

- Changing the highlight color identity of the marks (kept; optionally migrated
  to theme tokens for dark-mode parity as a minor consistency tweak).
- Backend / annotation pipeline changes.
- The separate local-dev LLM quota change (already applied in
  `docker-compose.dev.yml`).
