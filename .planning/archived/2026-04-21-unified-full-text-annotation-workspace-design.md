# Unified Full-Text Annotation Workspace Design

Date: 2026-04-21

## Goal

Redesign Phentrieve full-text mode into a single unified annotation workspace that:

- preserves the existing conversation-first product feel
- uses one shared UI for both `standard` and `llm` backends
- keeps chunk-based fallback fully usable when LLM quota is exhausted
- upgrades the result experience from a static chunk/results display into an interactive document review and curation surface
- replaces the current full-text `HPO Collection` behavior with a case-oriented phenopacket workspace

This design also updates the anonymous free production quota from `3` successful LLM full-text requests per UTC day to `5`.

This is primarily a frontend and interaction design layered onto the already shipped full-text backend. Backend work remains limited to response metadata, serialization reuse, quota configuration, and provenance/state support needed by the workspace.

## Scope

This design covers:

- frontend UX and interaction model for full-text mode
- API and response-contract implications needed to support the workspace
- case/phenopacket workflow changes in the right sidebar
- quota and fallback UX
- monitoring and instrumentation requirements for the new surface

This design does not cover:

- user authentication, registration, or sessions
- billing or paid quotas
- a separate top-level “classic” full-text UI
- redesign of single-term query mode
- full free-form annotation-tool complexity on day one

Auth is intentionally deferred. If identity is added later, the preferred direction is a self-hosted solution such as SuperTokens, but this is out of scope for the current design.

## Current Product Findings

### Existing surface already supports both backends

The codebase already exposes LLM full-text mode end to end:

- CLI full-text uses the shared full-text service in `phentrieve/text_processing/full_text_service.py`
- API full-text uses `POST /api/v1/text/process`
- frontend advanced options already surface `extractionBackend = standard | llm` plus `llmModel` and `llmMode`

Relevant files:

- `api/routers/text_processing_router.py`
- `api/schemas/text_processing_schemas.py`
- `api/llm_quota.py`
- `frontend/src/components/QueryInterface.vue`
- `frontend/src/components/AdvancedOptionsPanel.vue`
- `frontend/src/components/ResultsDisplay.vue`
- `frontend/src/components/ChunkResultsView.vue`
- `frontend/src/components/AggregatedTermsView.vue`
- `frontend/src/services/PhentrieveService.js`

Related design work that this spec depends on:

- `.planning/archived/2026-04-19-phenopacket-v2-and-annotation-sidecar-design.md`

The sidecar format from that design is the intended provenance read/write model for this workspace. This spec should not define a parallel provenance contract.

### Existing UX strengths

- the conversation shell is simple and already liked by the user
- the main landing surface is clean and visually restrained
- backend switching already exists behind one full-text mode
- the product already has a right-side collection workflow

### Existing UX weaknesses

- full-text results are split into separate chunk and aggregated-term surfaces that feel loosely coupled
- LLM mode is framed as an advanced setting rather than a richer evidence-review experience
- highlighting is functional but visually basic
- the right-side `HPO Collection` is too flat and term-centric for note-level full-text work
- editability is limited and does not support annotation-style review

## Product Direction

### One full-text product, not multiple full-text UIs

Phentrieve should have one full-text user experience.

- `standard` remains available and fully usable
- `llm` enriches the same workspace with richer evidence/provenance
- backend differences are expressed through evidence quality and metadata, not through a different user-facing product mode

The old chunk-based display should not survive as a separate “classic mode”. Compatibility should remain at the API/data-contract level, not as a separate visual experience.

### Keep the conversation shell

The top-level interaction model should remain conversation-first.

- users still submit notes from the current conversation UI
- full-text responses still appear in conversation history
- the full-text response expands into a richer annotation workspace inside that flow

This preserves the product’s current feel while fixing the weak part: the result review surface.

### Use a hybrid expanded workspace

The recommended pattern is a hybrid conversation plus expandable workspace model:

- user submits a note in conversation
- the response appears in conversation
- opening or expanding that response reveals a full-text annotation workspace

This is preferable to:

- a dedicated separate screen, which breaks the current product feel
- a static embedded card, which is too cramped for richer interactions

## Layout

### Desktop structure

The expanded full-text workspace should have three coordinated areas:

1. `Document pane`
- primary visual surface
- displays the note in chunked form
- chunk boundaries remain visible but subdued
- annotations appear inline in the text

2. `Phenotype pane`
- secondary pane inside the main workspace
- lists extracted HPO terms with counts, status, and evidence summary
- linked directly to annotations in the document pane

3. `Right sidebar`
- only one persistent right sidebar
- default mode: case / phenopacket workspace
- edit mode: annotation inspector

There should be no permanent left sidebar.

### Mobile and narrow layouts

- document remains primary
- phenotype pane can stack/collapse responsively
- right sidebar behavior should convert to a bottom sheet, push view, or compact inspector pattern rather than a desktop-style floating modal
- annotation popovers on touch should trigger from `selectionchange` with a short delay, not directly on `mouseup`, so they can coexist with the native long-press selection menu

## Interaction Model

### Core interaction principles

- inspect first
- edit second
- preserve document context during editing
- avoid interruption-heavy modal flows

### Three levels of interaction

1. `Hover linkage`
- hover a phenotype term: highlight linked spans in the note
- hover a span: highlight linked phenotype term

2. `Inline quick actions`
- click span, click term, or select text
- show a small contextual popover near the interaction target
- keep actions short and immediate

3. `Deep edit`
- switch the right sidebar into annotation-inspector mode
- show richer editing and provenance controls there

### Sidebar mode policy

The switch between `Case Workspace` and `Annotation Inspector` is a known mode-slip risk and must be made explicit in the UI.

Rules:

- the sidebar header label must change clearly between modes
- the sidebar surface should have a subtle but distinct tint per mode
- the sidebar must never auto-flip on hover
- the sidebar only switches into inspector mode from an explicit user action such as `Inspect`
- `Esc` returns from inspector mode to case-workspace mode
- returning to case-workspace mode must be possible in one click

### Modal policy

Modals should not be the main editing mechanism.

Use modals only for:

- destructive or global actions
- discarding all edits
- deleting a case
- other high-risk confirmations

Normal annotation work should remain inline or in the switchable right sidebar.

## Annotation Design

### Visual language recommendation

Phentrieve should look like a calm clinical note reader with linked phenotype evidence, not like a generic NLP labeling tool.

Recommended annotation style:

- subtle text highlights first
- restrained color system
- one strong active-selection accent
- muted semantic colors for assertion/state
- quiet chunk boundaries
- chips only where they provide real signal
- chunk-only evidence uses a subdued chunk gutter tint
- true character-offset evidence uses inline `<mark>`-style highlighting
- primary surfaces use banded confidence labels (`high`, `medium`, `low`) instead of raw numeric confidence scores

Avoid:

- rainbow entity palettes
- dense inline labels everywhere
- annotation-lab visual clutter
- fake precision when the backend only has chunk-level evidence

### Render primitive

The workspace should use the CSS Custom Highlight API as the primary span-highlighting primitive.

Reasons:

- it avoids destructive DOM span-wrapping for ordinary highlights
- it preserves copy/paste and selection behavior better
- it behaves better with overlapping annotation layers

If Safari versions older than `17.2` must be supported, use a nested-`<span>` fallback only for unsupported browsers and keep that fallback constrained to the minimum necessary visual states.

### Unified annotation model

Both backends should feed the same conceptual workspace model:

- annotation span
- linked HPO term
- assertion / status
- provenance
- case membership

### Accessibility

Annotated evidence spans should use an ARIA Annotations-style approach:

- render true inline evidence with `<mark>`
- connect the `<mark>` to the phenotype detail node in the phenotype pane or inspector using `aria-details`

Do not rely on `aria-label` on `<mark>` as the primary accessibility mechanism for annotation meaning.

### Standard backend behavior

Standard mode should remain a first-class full-text path:

- chunk-based evidence is primary
- phrase-level attribution is used where available
- if exact spans are weak, the UI should show honest local evidence instead of fabricating exact offsets
- chunk-only evidence must be rendered with the subdued chunk gutter tint rather than inline fine-grained highlights

This path must remain fully usable when LLM quota is exhausted.

### LLM backend behavior

LLM mode should enrich the same UI:

- richer phrase/span evidence
- better provenance
- finer mapping/remapping support
- more precise note-term linkage when available

The user should not experience this as a separate product.

When true character offsets are available, the UI may render inline `<mark>`-style highlights in the note while keeping chunk gutters visible but secondary.

### Editing actions

Default click behavior should be `inspect`, not immediate deep edit.

Available actions across the system:

- inspect linked evidence
- add phenotype
- add to case
- remove annotation
- change mapped HPO term
- change assertion/status
- search a different HPO term for a selected region

The system should also preserve user-curated state distinctly from model-generated state.

Primary surfaces should use banded confidence labels only. Raw numeric scores belong in the inspector pane and developer-facing tooling, not in the main note or phenotype pane.

## Case Workspace

### Replace `HPO Collection` in full-text mode

In full-text mode, the current right-side `HPO Collection` should become a `Case Workspace`.

Each case represents:

- one note / document
- one phenotype curation context
- one phenopacket draft

### Multiple cases per session

The system should support multiple cases in one session.

The user clarified that this means:

- multiple different patient notes/documents
- not splitting one single note into multiple subjects

### Case-creation behavior

Adding from conversation should remain lightweight:

- if no case exists, create one
- if no case is selected, create one
- if a case is selected, add there

### Case workspace contents

The right sidebar in case-workspace mode should contain:

- current case selector
- list of session cases
- current case summary
- phenotype count
- source-note status
- export action(s)

Frontend phenopacket export is new in this design; backend serialization logic is reused from the CLI and sidecar-related provenance should align with `.planning/archived/2026-04-19-phenopacket-v2-and-annotation-sidecar-design.md`.

Prominent actions:

- add selected phenotype
- add all extracted phenotypes
- review staged additions
- export phenopacket
- create new case
- switch case

To reduce rename risk during the alpha transition, the user-facing label can change to `Case Workspace` while internal IDs/slugs and migration paths continue to treat the current collection model as the compatibility baseline for one release.

### Sidebar mode switching

There should never be two competing right-side panels.

Right sidebar modes:

- `Case Workspace`
- `Annotation Inspector`

Deep editing temporarily switches the sidebar from workspace mode into inspector mode, then allows returning to the workspace.

## Result-Screen Behavior

### Primary focus

The full-text result view should be text-first:

- note/chunks are visually primary
- extracted phenotype list is secondary but always available on desktop
- clicking terms highlights note evidence
- users review a document, not just a list of extracted results

### Expanded response behavior

The full-text result in conversation should:

- start compact enough to fit conversation rhythm
- expand into the richer workspace for review and edits
- preserve conversational continuity while allowing serious note curation

### Phenotype pane behavior

The extracted phenotype pane should remain visible on desktop with behavior broadly similar to the current app:

- present by default on larger screens
- responsive/collapsible on smaller screens

The improvement should come from richer interactions, not from changing the visibility model users already understand.

### Workspace state isolation

Annotation workspace state must be isolated from chat history state.

Rules:

- workspace state lives in a dedicated Pinia store scoped to the full-text workspace
- state is keyed by conversation turn ID
- state persists across collapse/expand of that turn’s workspace
- state is independent of `conversationStore.queryHistory`
- undo/redo scope is per turn

## Quota, Fallbacks, And Trust

### Quota update

Anonymous production LLM full-text quota should change from:

- `3/day`

to:

- `5/day`

Count successful LLM full-text requests per UTC day, consistent with the current quota model.

The limit should remain environment-configurable through the existing server-side configuration (`PHENTRIEVE_LLM_DAILY_LIMIT`) so the value can be tuned without code changes or redeploying a different product shape.

### LLM fallback UX

When LLM quota is exhausted:

- silently fall back to standard full-text analysis for the current action
- show a one-line banner explaining that richer LLM analysis is unavailable for today
- include the reset time in the user’s local timezone
- keep the same full-text workspace UI
- do not present the event as a blocking error
- do not eject the user into a visibly worse or different product mode

### Trust rules

The UI should make it easy to answer:

- why is this phenotype here?
- where in the note did it come from?
- how precise is that evidence?
- did the user edit it?

Confidence and provenance should be available on demand, not visually dominant everywhere.

The system must distinguish clearly between:

- model-generated annotation
- user-edited annotation
- unresolved/manual mapping
- chunk-level evidence vs span-level evidence

The UI must operationalize chunk-vs-span honesty visually:

- chunk-only evidence = subdued chunk gutter tint
- true offset evidence = inline `<mark>`-style highlight

## Monitoring And Observability

Because the new surface introduces richer review/edit workflows, add instrumentation for:

- full-text request type (`standard` vs `llm`)
- LLM quota checks and exhaustion
- fallback-to-standard events
- “add all phenotypes” actions
- annotation edits
- remap actions
- manual search/replacement actions
- export actions

Logging requirements:

- structured events
- request correlation / request IDs
- no sensitive raw prompt logging by default
- avoid logging unnecessary full clinical text in user-facing telemetry paths

The current client-side log viewer is useful for development, but the production monitoring design should favor server-side structured events and aggregate metrics.

## Technical Implications

### Frontend

Likely refactor direction:

- keep `QueryInterface.vue` as the conversation shell
- keep `ResultsDisplay.vue` as the result entry point
- replace the current split chunk/results full-text rendering with a unified workspace component, e.g.:
  - `FullTextAnnotationWorkspace.vue`

Likely supporting components:

- `AnnotatedDocumentPane.vue`
- `PhenotypeFindingsPane.vue`
- `CaseWorkspacePanel.vue`
- `AnnotationInspectorPanel.vue`
- `AnnotationActionPopover.vue`

### State management

No existing Pinia store currently owns full-text workspace annotation state.

The current `stores/` tree contains conversation, log, queryPreferences, and disclaimer concerns only. This redesign therefore requires a new workspace-scoped store responsible for:

- per-turn annotation state
- active selection
- hover/focus linkage state
- turn-scoped undo/redo
- sidebar mode (`case` vs `inspector`)
- local edits before export/commit
- fallback-mode and quota banner state for the active turn

### API

Keep the current single endpoint:

- `POST /api/v1/text/process`

Continue using the shared response shape:

- `meta`
- `processed_chunks`
- `aggregated_hpo_terms`

Extend only as needed to support:

- richer evidence metadata
- user-curation state if edits become persisted
- clearer standard-vs-LLM provenance and fallback metadata
- updated quota metadata reflecting the new `5/day` production limit

### Data-contract rule

Maintain one stable full-text contract and let both backends populate it honestly.

If exact spans are unavailable, the backend must not synthesize false precision just to satisfy the workspace.

## Design Principles

- one full-text UX
- conversation-first shell preserved
- note/chunks are primary
- phenotypes are linked, inspectable, and editable
- standard and LLM share the same workspace
- annotation editing uses inline popovers and one switchable right sidebar
- no second persistent right-side panel
- no permanent left sidebar
- modals only for destructive/global actions
- trust over spectacle

## External UX References

These sources informed the interaction direction:

- Hypothesis for host-page highlight and sidebar linkage:
  - https://web.hypothes.is/help/overview-of-the-hypothesis-system/
  - https://web.hypothes.is/help/annotation-basics/
- tagtog for readable overlapping annotations and direct annotation actions:
  - https://docs.tagtog.com/webeditor.html
- INCEpTION for suggestion review without interrupting central annotation work:
  - https://inception-project.github.io/releases/39.2/docs/user-guide.html
- MedCATTrainer for biomedical concept inspection and linked text/concept workflows:
  - https://aclanthology.org/D19-3024.pdf
- brat for annotation primitives, automatic-tool integration, and always-saved edits:
  - https://brat.nlplab.org/features.html
- CKEditor for inline-vs-sidebar annotation display modes:
  - https://ckeditor.com/docs/ckeditor5/latest/features/collaboration/annotations/annotations-display-mode.html
- SAP Fiori supporting pane guidance for contextual secondary panes:
  - https://www.sap.com/design-system/fiori-design-android/v25-4/layouts/supporting-pane/supporting-pane-overview

## Summary Recommendation

Phentrieve should evolve full-text mode into a single annotation-centric document reader that:

- stays inside the conversation shell
- expands into a hybrid review workspace
- keeps chunks usable and visible
- lets `standard` and `llm` power the same UI
- upgrades the right panel into a case/phenopacket workspace
- supports richer evidence review and curation without turning the app into a dense annotation lab

This is the best fit for the current alpha stage: stronger than the current UI, simpler than a full annotation platform, and aligned with both the current codebase and the user’s product goals.
