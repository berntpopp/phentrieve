# Full-Text Annotation Curation - Verification

- Date: 2026-06-14
- Spec: `specs/2026-06-13-fulltext-annotation-curation-design.md`
- Plan: `active/2026-06-13-fulltext-annotation-curation-plan.md`
- Branch: `feat/fulltext-annotation-curation`
- Stack tested: Docker dev stack rebuilt from this branch, frontend at
  `http://localhost:8080/` (API at `:8001`, LLM extraction backend).

## Outcome

All four user stories verified end-to-end via Playwright on the rebuilt Docker
stack. Two defects surfaced during E2E were fixed and re-verified. The orphaned
workspace cluster (~2,300 LOC, 9 source + 6 test files) was removed.

## Automated gates

- `make ci-frontend`: ESLint, Prettier `format:check`, i18n validation, Vitest,
  and production build all green.
- Vitest: 298 tests across 31 files pass (was 261 before; net +37 after removing
  6 orphan test files and adding feature + regression tests).
- i18n: 349 keys congruent across en/de/fr/es/nl; all `{param}` placeholders
  match.
- Changes are frontend-only; no Python sources touched.

## E2E (Playwright, localhost:8080, LLM backend)

Test note: "The patient is a 4-year-old boy with global developmental delay and
frequent seizures. He has muscular hypotonia, microcephaly, and bilateral
sensorineural hearing loss. He was born at term with a normal birth weight.
There was no hepatomegaly on examination."

Full-text analysis returned 7 findings and 7 highlighted note spans (incl.
Negated chips for Hepatomegaly and the SGA mapping of "normal birth weight").

1. **Remove + Undo** - Right/left-clicking the "normal birth weight" span ->
   popover (Change term / Add to collection / Remove annotation; no Revert for an
   auto term) -> Remove dropped the span and its finding (7->6). A second remove
   showed the snackbar "Annotation removed  Undo"; clicking Undo restored it
   (5->6). PASS.
2. **Change term (live re-query)** - "microcephaly" -> Change term -> dialog
   re-queried `/api/v1/query/` and returned 8 ranked candidates with scores +
   definitions -> selected "Primary microcephaly" -> Replace. Finding #3 became
   `HP:0011451 Primary microcephaly` with a **Manual** badge; provenance
   (`replacedFrom` = Microcephaly) recorded. PASS.
3. **Annotate selection** - Selected the now-unhighlighted "normal birth weight"
   -> selection popover showed only "Annotate selection" -> dialog (title
   "Annotate selection", action "Add") re-queried and returned candidates ->
   added "Intrauterine growth retardation". New highlighted span + finding #7
   with a **Manual** badge (2 manual total). PASS.
4. **Persistence** - Page reload: 7 curated findings persisted (both Manual
   badges intact). The clinical note is intentionally not persisted (stored as
   `[redacted]`), so highlights do not survive reload by design; the note
   degrades to clean "[redacted]" plain text with **zero** broken marks. PASS.

Interaction triggers confirmed: left-click, right-click (contextmenu), and
keyboard (Enter) all open the menu; spans expose `role="button"` +
`aria-haspopup="menu"`.

## Defects found during E2E and fixed

1. **Empty annotation set locked in during loading** - The note curator mounts
   while the turn is still loading (`response: null`) and seeded an empty model,
   so findings stayed at 0 after the API returned. Fix: gate seeding on response
   presence and seed via an immediate watcher in both the curator and the
   receipt. Regression test added.
2. **Broken empty marks after reload** - The note text is redacted on reload but
   the curation store persists span offsets; slicing "[redacted]" produced empty
   `<mark>` nodes. Fix: only render a highlight when the stored span text still
   matches the note at that offset; otherwise degrade to plain text. Regression
   test added.
3. **Popover header showed a raw i18n key** - `annotatedDocumentPane.actions.title`
   never existed (the orphaned component referenced it but it was never shown).
   Added the key across all five locales; header now reads "Annotation tools".

## Lighthouse (desktop, localhost:8080)

| Category | Before | After |
|----------|--------|-------|
| Performance | 57 | 57 |
| Accessibility | 95 | 95 |
| Best practices | 100 | 100 |
| SEO | 100 | 100 |

No regression. The three accessibility findings are **pre-existing and unrelated
to this feature** (Lighthouse audits the empty landing page, where the curation
UI is not rendered):

- `aria-tooltip-name` (13 nodes): generic Vuetify `v-tooltip` overlay containers
  used app-wide (footer icons, chat avatars) render `role="tooltip"` without an
  accessible name. Not the annotated-note tooltip.
- `label-content-name-mismatch` (1 node): the existing "Open advanced options"
  settings button.
- `heading-order`: landing-page heading structure.

Correction to the spec: this feature does not resolve `aria-tooltip-name` (that
failure originates from app-wide Vuetify tooltips, not the annotation tooltip).
The feature's own accessibility (button-role spans, `aria-haspopup`, keyboard
activation, focus-returning menu/dialog) is sound and adds no new violations.

## Follow-ups (out of scope)

- App-wide Vuetify `v-tooltip` accessible-name fix (`aria-tooltip-name`).
- Landing-page `heading-order` and the settings-button label mismatch.
- Performance: ~116 KiB unused JS / ~44 KiB unused CSS on first load (FCP 4.7s).
- Optional: a findings-list affordance to remove/curate terms that have no note
  span (currently only span-bearing annotations are curatable from the note).
