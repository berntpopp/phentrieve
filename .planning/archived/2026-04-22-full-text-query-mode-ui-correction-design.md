# Full-Text And Query Mode UI Correction Design

Date: 2026-04-22

## Goal

Correct the current search and full-text review UX so that:

- query mode and full-text mode feel like one coherent product
- the primary mode switch is visible and fast to use
- long text entry can auto-switch into full-text mode without surprising the user
- the `Case Workspace` behaves as one shared system across both modes
- phenotype presentation is visually congruent across query results and full-text findings
- the team can iterate quickly with Playwright-driven UI checks instead of long manual redesign loops

This started as a UI and interaction correction spec. The shipped implementation
also required a small shared-service/backend integration correction so live LLM
full-text requests could carry real scores and the local API could read the
repo-root `.env` during development.

## Implemented Outcome

The current implementation differs in a few important ways from the original
design draft:

- the user-submitted clinical note remains in the user bubble, not in a
  dedicated document pane
- the bot response remains the phenotype-results surface and reuses the shared
  phenotype card design from query mode
- note-side evidence is expressed as inline annotated spans with hover linkage
  rather than a large standalone evidence panel
- the legacy three-column full-text review layout was removed in favor of a
  tighter conversation-native layout
- the shared full-text service now preserves real `confidence`, `score`, and
  `reranker_score` values so the frontend can render non-zero confidence
  consistently
- local development required `api/run_api_local.py` to load both
  `api/local_api_config.env` and the repo-root `.env`, because the live LLM path
  depends on `GEMINI_API_KEY`

## Final UX Shape

### Search shell

Implemented as designed:

- visible `Query` and `Full Text` mode pills in the main shell
- long text auto-switches to `Full Text`
- a helper notice explains the auto-switch
- advanced options remain secondary

### Clinical note presentation

Implemented shape:

- the submitted note is rendered in the user bubble as a compact summary
- new full-text turns expand the clinical note by default
- when expanded, the truncated preview is hidden
- inline note annotations are built from `text_attributions` and matched chunk
  offsets
- hover on a phenotype card highlights matching note spans
- hover on an annotated note span highlights the linked phenotype card
- annotated spans expose tooltip text with HPO label and ID

This replaced the earlier idea of a large dedicated `Document pane` in the main
response layout.

### Full-text result presentation

Implemented shape:

- the bot response shows a compact receipt:
  - `Full-text analysis ready`
  - findings count
- extracted phenotypes remain in the bot response, not in the user note bubble
- phenotype cards reuse the shared query-mode visual language via shared card
  primitives
- full-text mode adds note-linkage behavior and add-all actions without
  diverging into a separate card style

### Shared phenotype presentation

Implemented with a shared primitive:

- `PhenotypeCardRow.vue` provides the shared base presentation
- `ResultItem.vue` uses the shared row for query and full-text response cards
- `PhenotypeFindingsPane.vue` uses the same card family instead of legacy custom
  full-text tiles

### Case and collection behavior

Implemented behavior:

- the phenotype response remains in the conversation flow
- the shared collection/case path is preserved
- full-text response cards support individual add-to-collection actions
- full-text responses also expose `Add all`

### Full-text evidence behavior

The original large `Evidence in note` review block was intentionally removed.
The shipped interaction is:

- evidence stays in the note text itself
- evidence is shown as inline annotations
- evidence discovery happens through hover linkage, not through a second large
  review panel

This is smaller, more legible, and consistent with the conversation-first
interface.

## Backend/Integration Adjustments

The shipped UI depended on two non-visual fixes:

### Score propagation

The shared LLM full-text adapter had been normalizing scores to zero. The final
implementation preserves:

- `confidence`
- `score`
- `reranker_score`

from the LLM pipeline into the stable full-text response. This is now the
source for the frontend confidence display.

### Local API env loading

During local development the live LLM full-text path failed with `500` because
`run_api_local.py` only loaded `api/local_api_config.env`, while
`GEMINI_API_KEY` lived in the repo-root `.env`.

The implementation now loads both env files for local runs so default LLM
full-text requests work in the development environment.

## Error-State Correction

One additional regression surfaced during implementation:

- failed full-text requests were being rendered as a fake success receipt with
  `0 findings`

The final implementation now renders true error states instead of showing a
false empty-success receipt.

## Problems To Correct

### Hidden primary mode switch

The current switch between query mode and full-text mode is buried inside advanced options. This makes a primary workflow decision feel secondary and causes poor discoverability.

### Conflicting workspace systems

The current UI exposes both:

- a legacy global `PhenotypeCollectionPanel`
- a new in-workspace `CaseWorkspacePanel`

This creates duplicate mental models, duplicate labels, and duplicate right-side surfaces.

### Full-text result is still too card-bound

The conversation shell should remain, but the full-text result should expand into a deliberate review workspace rather than behaving like a cramped chat-card detail panel.

### Phenotype visuals regressed

Query mode and full-text mode currently present phenotypes with materially different visual patterns. This breaks continuity and strongly suggests duplicated implementation rather than shared UI primitives.

### Iteration speed is too low

The current workflow is too dependent on manual inspection and verbal interpretation. Design correction must become faster and more deterministic.

## Product Principles

### One product, two modes

`Query` and `Full Text` are two ways of using the same product, not two unrelated interfaces.

Rules:

- keep one shared search shell
- keep one shared `Case Workspace`
- keep one shared phenotype-item visual language
- let only the input behavior and review surface change by mode

### Make the important choice obvious

The query/full-text mode choice is a primary action and must be visible in the search shell itself.

### Preserve conversation-first flow

The conversation shell remains the top-level interaction model. Full-text review expands under the active result rather than moving the user to a separate application screen.

### DRY, KISS, SOLID

The implementation should reduce duplication, not add another parallel layer of UI.

Rules:

- one search shell component family
- one case workspace system
- one phenotype-item design language
- small, composable components with clear ownership

## Recommended UX Direction

## Search Shell

The search shell remains the central entry surface for both modes.

### Layout

Desktop search shell structure:

1. Main input area in the center
2. Mode pills on the right edge of the shell
3. Advanced-options affordance adjacent to the mode pills
4. Search/submit button at the far right end

### Mode switch

Use short labeled pills with icons:

- `Query` with a search icon
- `Full Text` with a document icon

This is preferred over icon-only mode switching because it is significantly clearer while still feeling compact and modern.

### Input behavior

In `Query` mode:

- input is a single-line field
- optimized for short phenotype or syndrome queries

In `Full Text` mode:

- the same shell expands into a multi-line textarea
- optimized for note-like clinical text

### Auto-switch behavior

If the user is in `Query` mode and pastes or types text above the configured threshold:

- the UI auto-switches into `Full Text`
- the input expands into the multi-line form
- a small inline helper appears:
  - `Switched to Full Text for longer clinical text`

This preserves speed while making the behavior understandable.

### Manual override

The user can always switch back manually. Auto-switching should never lock the mode.

## Result Surface

### Query mode

Query mode continues to show concise query results below the search shell, using the shared phenotype presentation primitives.

### Full-text mode

Full-text mode expands the active response into a review workspace below the conversation item.

This workspace should feel wider and more intentional than a standard result card, while still clearly belonging to the conversation flow.

## Full-Text Workspace Layout

### Desktop structure

The expanded full-text result should use three coordinated regions:

1. `Document pane`
2. `Findings pane`
3. `Case Workspace`

Recommended width balance:

- document pane: 50-55%
- findings pane: 25-30%
- case workspace: 20-25%

### Document pane

The document pane is primary.

Responsibilities:

- render the clinical note
- show subtle evidence highlights
- preserve reading flow
- support evidence inspection and selection

If the backend does not return chunked document content for a valid full-text result, the UI must still provide a clear document-area fallback rather than collapsing the primary pane.

### Findings pane

The findings pane lists extracted phenotypes using the same core phenotype-item visual language as query mode.

Full-text adds:

- evidence metadata
- source/evidence summaries
- selection/highlight linkage

It should not introduce a new, unrelated phenotype card style.

### Case Workspace

The `Case Workspace` is the right-side utility rail for both modes.

It should not be replaced by a second drawer or a second sidebar system during full-text review.

## Case Workspace Design

### Role

The `Case Workspace` is a shared case-assembly surface, not a separate mini-application.

It should feel:

- stable
- compact
- quiet
- always understandable

### Structure

1. Header
- title: `Case Workspace`
- active case context
- small `New case` action

2. Case switcher
- compact case list
- active case clearly indicated
- case counts visible but subdued

3. Active case summary
- selected case name
- phenotype count

4. Actions
- primary: `Add all`
- secondary: `Export phenopacket`

5. Included phenotypes preview
- compact list of phenotypes already added to the active case

### Visual hierarchy

The case workspace must not use oversized block-button treatment that overwhelms the review flow.

Rules:

- one clear primary action
- secondary actions visually quieter
- compact cards or rows for included items
- avoid dashboard-style CTA stacks

## Phenotype Presentation

### Shared visual language

Query-mode phenotypes and full-text findings should share one design language.

Shared elements:

- phenotype label
- HPO identifier
- assertion/status chip
- add/remove affordance positioning
- spacing and typography rhythm

Mode-specific additions:

- query mode can emphasize retrieval confidence
- full-text mode can emphasize evidence type and document linkage

### Implementation direction

Refactor toward shared phenotype primitives instead of separate bespoke patterns.

Preferred outcome:

- a shared phenotype row/card base component
- query wrapper for query-specific metadata
- full-text wrapper for evidence-specific metadata

This keeps behavior modular and avoids divergence.

## Legacy Surface Removal

The legacy global `PhenotypeCollectionPanel` and its floating action button must not remain active in full-text review mode.

Desired rule:

- one shared case workspace system
- no duplicate `Case Workspace` labels
- no second competing right-side drawer during full-text review

If a page-level right sidebar shell is retained across the product, its content model must remain the same shared `Case Workspace` instead of a legacy bridge component.

## Rapid Iteration Strategy

### Objective

UI correction should move quickly with feedback loops shorter than full manual review cycles.

### Playwright role

Playwright should be used as an iteration tool, not only as final regression testing.

Use it for:

- repeatable interaction flows
- layout verification
- screenshot-based region checks
- fast comparison between UI revisions

### Golden-path test coverage

Create one stable golden-path UI flow covering:

1. toggle `Query` and `Full Text`
2. auto-switch to `Full Text` on long paste
3. submit a representative note
4. verify one shared `Case Workspace`
5. verify search shell state
6. verify phenotype presentation consistency

### Screenshot regions

Use targeted screenshot assertions for:

- search shell
- query result phenotype item
- full-text findings item
- case workspace
- expanded full-text workspace

Avoid relying only on full-page screenshots, which are more brittle and slower to interpret.

### Implementation slices

Make changes in small, reviewable slices:

1. search shell mode pills and mode-switch behavior
2. auto-switch behavior and inline notice
3. removal or suppression of legacy full-text collection drawer behavior
4. case workspace compaction and hierarchy correction
5. shared phenotype primitive refactor
6. full-text workspace layout polish

Each slice should be validated with Playwright before moving to the next.

## Environment Configuration Guidance

### Local development

The LLM API key must remain server-side only.

Rules:

- never expose it through `VITE_*`
- never commit it into tracked files
- local dev may load it from an ignored env source into the API process

### Docker development

Development Docker may use an env file for convenience, provided it remains ignored and API-only.

### Production

Production should prefer Docker secrets over plain environment variables for LLM API credentials.

The API container should be the only service granted access.

## Component Direction

Recommended refactor boundaries:

- `QueryInterface` owns the shared search shell and top-level mode state
- a dedicated search-shell subcomponent should own the input/pill control presentation
- `CaseWorkspacePanel` should become the shared right-side case surface
- `PhenotypeCollectionPanel` should be retired or reduced to a compatibility wrapper during migration
- phenotype display should move toward shared item primitives

## Success Criteria

The correction is successful when:

- users can immediately find and understand the query/full-text mode choice
- long clinical text naturally moves into full-text mode with clear feedback
- only one `Case Workspace` system is visible and active
- query-mode and full-text phenotypes feel visually related
- the full-text review surface feels intentional, readable, and modern
- Playwright can validate the core interaction and layout path quickly

## Open Questions Resolved

- Mode switch location: right side of the search field
- Control style: short labeled pills with icons
- Long-text behavior: auto-switch to full text with a small inline notice
- Case workspace principle: shared across both modes
