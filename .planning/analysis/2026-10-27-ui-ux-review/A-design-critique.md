# Phentrieve -- Assessment A: Design Critique (Operate mode)

Reviewer stance: senior product designer, unanchored (detector not run). Evidence: live app at
localhost:5734 (Vite dev) + API :8734, headless Chrome via playwright-core. Screenshots in
`shots-design/` (same scratchpad). Source read under `frontend/src`.

## 1. Specificity verdict

**Mostly generic Material/Vuetify template with a chatbot skin, with two pockets of real authorship.**

- Generic: default Vuetify blue primary + unrelated teal FAB (`color="secondary"`), grey-lighten-4 page,
  chat bubbles with a "U" avatar and a robot avatar (`QueryInterface.vue:73-74, 109-110`), uppercase
  Material buttons ("I UNDERSTAND AND AGREE", "ADD ALL", "SKIP TOUR"), hero = logo + centered blurb
  + pill search bar. Swap the copy and this could be any "AI-powered X" demo.
- Authored for phenotyping: (a) the full-text note curator -- the source note with inline highlights of
  matched spans next to findings with a `Negated` chip is exactly the right mental model for HPO curation
  (`FullTextNoteCurator.vue`, `FullTextResponseReceipt.vue`); (b) Phenopacket export with subject fields
  in the Case Workspace. The `/connect` MCP page is also clearly authored and better written than the home.
- The chat metaphor actively fights the domain: curation is a review/accept task over a *list*, not a
  conversation. "U" and a robot avatar add no information and cost 44px of width per turn.
- Typography is unintentional: Vuetify asks for `Roboto` (never loaded; `index.html` and `main.js` load
  no text font) so body falls back to the OS sans, while inputs/bubbles use the `system-ui` stack from
  `critical.css:22-27`. On Linux this renders two visibly different faces on the same screen
  (welcome blurb vs. "Phenotype query"/note bubble -- see `03-home-desktop-light.png`, `32-standard-result.png`).

## 2. Nielsen heuristics (0 = absent, 4 = excellent)

| # | Heuristic | Score | Key issue |
|---|-----------|:----:|-----------|
| 1 | Visibility of system status | 2 | Query loading is a bare 24px spinner in an empty bubble (`QueryInterface.vue:115`); full-text LLM run has no stage/progress/ETA. API status hidden in a footer dot. Positive: "28 findings across 14 document sections" receipt. |
| 2 | Match with real world | 2 | Result rows lead with `HP:0001250` in bold and put the clinical label second (`PhenotypeCardRow.vue:26-33`); clinicians scan by label. "Similarity Threshold", "Embedding Model", "Two-phase extraction", "Chunk retrieval threshold" are engineer vocabulary. German query "kurze Statur" is reported as `Language: en` (`ResultsDisplay.vue:27-33`). |
| 3 | User control and freedom | 1 | "Clear Collection" wipes all curated terms + subject info with no confirm or undo (`usePhenotypeCollection.js:89-95`, button `PhenotypeCollectionPanel.vue:270-278`) while the far less costly logo-reset *does* confirm (`HomeView.vue:34`). Tutorial ignores Escape (verified: backdrop still present after Esc). Log viewer drawer covers the footer toggle that opened it. |
| 4 | Consistency and standards | 1 | Mode set in two places (pill switch AND "Processing Mode" dropdown in Settings, `AdvancedOptionsPanel.vue:162-166`); two "settings" (content "Settings" pill vs footer gear = *Conversation* settings). `Negated` is orange/warning in results (`ResultItem.vue:147`) but pink in the workspace (`PhenotypeCollectionPanel.vue:83`). Score colour scale green->blue->orange (`SimilarityScore.vue:160-166`) -- blue is not on any confidence scale. Hard-coded English in an i18n app: "Query", "Full Text", "Settings" (`query/QueryModeControls.vue:15,27,48`), "Clinical note"/"Paste or type a clinical note" (`query/QueryForm.vue:33,39`), "Full-text analysis ready"/"Add all" (`FullTextResponseReceipt.vue:3,17`). |
| 5 | Error prevention | 1 | "Add all" bulk-accepts all 28 candidates incl. redundant/wrong ones (Megalencephaly, Macrocephaly at birth, Auriculitis for "ear infections") with one click (`FullTextResponseReceipt.vue:7-18`). 1-char query "a" returns 10 results scored 0.59-0.66 with no low-signal warning. Enter in the clinical-note textarea submits (`query/QueryForm.vue:36`, `.enter.prevent` also eats Shift+Enter), so a typed multi-paragraph note is impossible. Default full-text backend is LLM, which fails hard when no provider key is configured. |
| 6 | Recognition rather than recall | 1 | ~13 icon-only footer controls (scale, check, language, gear, hand, ?, log, GitHub, docs, license, API, robot, info). Meaning only on hover tooltip; touch users get nothing. Collection toggle icons show the *action* (red X on an Affirmed row, green check on a Negated row, `PhenotypeCollectionPanel.vue:107-113`) so they read as the opposite state. |
| 7 | Flexibility and efficiency | 2 | Enter-to-submit, URL query params, MCP/API for power users. But no keyboard accept/reject in results, no multi-select, no "add only affirmed / above threshold", no batch input. Tab order puts "Reset conversation" (logo) before the search field. |
| 8 | Aesthetic and minimalist design | 2 | Home is calm, but 3-line marketing blurb sits above the only task; result cards are 70px tall for two lines of text (5 results per desktop viewport); debug-grade Log Viewer, Conversation Settings ("Storage Used: 44 bytes"), Version dialog are exposed to every visitor. |
| 9 | Help users recognize/recover from errors | 1 | Full-text 503 renders as a saturated red slab "The service is temporarily unavailable... (Status: 503)" with no retry and no "switch to standard extraction" path even though that works. Empty submit: silent (button disabled, no hint). |
| 10 | Help and documentation | 2 | Tutorial (8 steps), FAQ, docs link, /connect exist. FAQ is two collapsed accordions in heavy black boxes; tutorial copy is generic ("Click this button to start the AI-powered analysis") and the step counter wraps ("1 /\n8"). |
| | **Total** | **15/40** | |

## 3. Cognitive load

Decision points with > 4 visible options:
- **Footer: 13 controls**, no labels, no grouping except one divider before API/MCP (`App.vue:11-249`). Legal, i18n, help, dev tooling, and status are all equal-weight blue glyphs. On 390px the row overflows (footer scrollWidth 612 vs 390): API, MCP and status/info are cut off and unreachable (`11-mobile-home.png`).
- **Settings panel (full-text mode): 7-9 fields** -- Embedding Model, Similarity Threshold, Language, Include Details, Processing Mode, Extraction Backend, LLM Model, LLM Mode (+ chunking fields in standard backend). `AdvancedOptionsPanel.vue` has 34 form controls in total. "Processing Mode" appears twice (section header + field label).
- **Result row: 4 targets** (HPO-ID external link, details toggle, score chip, add). Acceptable, but the external link is the biggest, boldest target -- the wrong primary.
- **Full-text result: 28 findings, flat list** with near-duplicate siblings and negated items interleaved by score. The user must make 28 accept/reject decisions or one blind "Add all".
- Case Workspace: 28 rows x (toggle + delete), then Export JSON / Export Text / **Clear Collection** stacked with equal width directly under each other.

Mode switch discoverability: the Query / Full Text pill is right-aligned, small (size="small"), *below* the input, and there is also an "Automatic (by text length)" mode hidden in Settings plus an auto-switch notice. A first-time user pasting a note into the single-line query field will not see the switch before submitting. It should be the first thing above or inside the input, left-aligned, with an explanatory placeholder per mode.

## 4. Emotional journey

1. **0-300 ms**: the disclaimer modal fades in over a *blank* page (`01-load-0ms.png`), then the app shell pops in behind it at ~300 ms -- a visible layout pop under a scrim. First impression = legal wall.
2. **Disclaimer**: persistent modal, 4 bullet limitations + "User Responsibility", single shouty uppercase CTA. Necessary, but delivered before the user knows what the tool does. Tone: defensive.
3. **Home**: calm, empty, slightly generic. The welcome sentence re-explains the product the disclaimer just explained.
4. **First query** ("seizures"): fast (results at <150 ms) and correct -- the peak moment. Score chips with a star at 1.00 feel trustworthy.
5. **Nonsense query** ("a"): ten confident-looking orange 0.6x chips -- trust erodes silently.
6. **Full text (default LLM)**: red error slab -- the valley; the note is cleared from the input and there is no path forward unless the user knows to dig into Settings > Extraction Backend.
7. **Full text (standard)**: genuinely delightful highlight view... then 28 findings and an "Add all" that will poison a phenopacket. Ends with export buttons beside an unguarded "Clear".

## 5. Three strengths

1. **Note-with-highlights curator** -- matched spans highlighted in the source note, `Negated` chips on findings, hover linkage. This is the product's soul and the most domain-specific surface.
2. **Speed and correctness of the short-query path** -- sub-150 ms, sensible ranking, German input mapped correctly, scores visible, one-click add.
3. **Honest framing and privacy affordances** -- research-use disclaimer, LLM-provider warning in Settings, PII review dialog, read-only MCP with clear explanation on `/connect`.

## 6. Priority issues

**P0-1. Bulk "Add all" with no curation guard.** `FullTextResponseReceipt.vue:7-18`.
28 findings (near-duplicates, wrong mappings, negated) go into the phenopacket in one click. Fix: replace with "Add N affirmed" (excluding negated/uncertain and collapsing ancestor/descendant duplicates), show a confirm sheet listing what will be added with per-row checkboxes pre-ticked only above a threshold; group findings by note span ("macrocephaly" -> Macrocephaly / Megalencephaly / Macrocephaly at birth) so the user picks one per span.

**P0-2. Destructive "Clear Collection" with no confirm/undo, adjacent to Export.** `PhenotypeCollectionPanel.vue:270-278`, `usePhenotypeCollection.js:89-95`.
Fix: snackbar with Undo (keep the previous array for 10 s) or reuse the HomeView reset dialog pattern; move Clear out of the export stack (overflow menu or text button at the top-right of the drawer).

**P1-3. Full-text default fails hard; error is a dead end.** Default `extractionBackend: 'llm'` (`AdvancedOptionsPanel.vue:540,556`); error rendered via `ResultsDisplay` (`QueryInterface.vue:117-126`).
Fix: if the LLM backend is unavailable (capabilities endpoint) default to standard; on 503 render a neutral inline alert with "Retry" and "Run with standard extraction" actions, and keep the note in the input.

**P1-4. Mode switch is hard to find and duplicated.** `query/QueryModeControls.vue:4-29`, `AdvancedOptionsPanel.vue:162-166`.
Fix: move Query / Full text segmented control above the input, left-aligned, labelled ("Look up a term" / "Extract from a note"); remove the Processing Mode dropdown (or keep "Automatic" as the only setting); show per-mode placeholder examples ("e.g. short stature", "Paste a clinical note...").

**P1-5. Footer icon wall; mobile overflow hides API/MCP/status.** `App.vue:11-249`.
Fix: keep 3-4 items in the footer (Disclaimer, Help, Language, status dot); move GitHub/Docs/License/API/MCP to a labelled "About & integrations" menu; move Log Viewer and Conversation Settings behind a dev flag or into the version dialog. Make the footer wrap or collapse to an overflow menu under 600px.

**P1-6. Result row hierarchy inverted.** `PhenotypeCardRow.vue:18-34`.
Fix: label first in `text-subtitle-1` high emphasis, HPO ID second in monospace caption with the external-link icon; tighten row to ~48px (two-line list density) so 10 results fit a viewport; put the add button at the left or make the whole row toggle.

**P2-7. Inconsistent assertion/score colour semantics.** `ResultItem.vue:147`, `PhenotypeCollectionPanel.vue:83,107-113`, `SimilarityScore.vue:160-166`.
Fix: one token map: affirmed = neutral/green, negated = one colour everywhere (with strikethrough or "NOT" prefix, not colour alone), uncertain = grey. Score: single-hue ramp (or plain numerals + a low-confidence warning icon) instead of green/blue/orange. Collection toggle: segmented "Present | Absent" control that shows state, not action.

**P2-8. No low-signal guard on short queries.** Fix: below ~3 characters or when top score < ~0.7, show "No strong match -- try a clinical description" above results; dim weak rows.

**P2-9. Untranslated UI strings and wrong language chip.** `query/QueryModeControls.vue:15,27,48`, `query/QueryForm.vue:33,39`, `FullTextResponseReceipt.vue:3,17`; `ResultsDisplay.vue:27-33` shows "Language: en" for German input. Fix: move strings to `locales/*.json`; show the *selected* language and a "detected: de" hint only when it differs.

**P3-10. Typeface mismatch and no dark theme.** Roboto requested but not loaded; inputs use `system-ui` (`critical.css:22-27`). `App.vue:2` hard-codes `theme="light"`, `plugins/vuetify.js:132-134` defines only light, and `prefers-color-scheme: dark` is ignored (`13/14-*darkscheme.png` identical to light) although some components ship dark-mode CSS (`SimilarityScore.vue:451`). Fix: pick one font stack app-wide (system-ui is fine; set it on `.v-application`); define a dark theme and follow the OS preference.

Also: Textarea Enter submits (`query/QueryForm.vue:36`) -- use Ctrl/Cmd+Enter in full-text mode. Tutorial: add Esc to close (`TutorialOverlay.vue`, no keydown handler). Disclaimer: show the shell first, avoid the blank->pop on load (`01-load-0ms.png` vs `01-load-300ms.png`).

## 7. Persona red flags

- **Clinical geneticist in a hurry**: must read past a modal, then a blurb, then find the small mode pill; bold HPO IDs instead of labels slow scanning; "Add all" is tempting and wrong; no keyboard accept/reject.
- **Researcher batch-curating**: one note at a time, chat history is the only record (max history 50, hidden in a footer gear dialog); no per-span grouping or dedup; Clear without undo can destroy a session's work; no CSV/TSV export of findings with source spans.
- **First-time visitor**: disclaimer before value; 13 unlabeled icons; Settings exposes embedding model names and thresholds; default full-text run errors out (when LLM unconfigured). Tutorial helps but is generic.
- **Screen-reader user**: H1 is read as "Phen [ otype re ] trieve" (`HomeView.vue:19-23`); only one heading on the page -- results/workspace have no headings to navigate; logo is a `role="button"` reset that is the 2nd tab stop; query field aria-label "Clinical text input field" in Query mode is misleading; icon-only state toggles in the workspace rely on colour.
- **Mobile user (390px)**: footer cut off (API/MCP/status unreachable), FAB overlaps the add button of result rows (`12-mobile-results.png`), score chip wraps under the label pushing the add button down, settings panel runs under the fixed footer.

## 8. Flicker / layout jumps observed

- 0 ms: blank grey page + fading disclaimer; ~300 ms: app shell appears behind the scrim (layout pop under modal).
- Query results appear in <150 ms with no skeleton; the turn is inserted at the top of history pushing previous turns down (no scroll anchoring announcement).
- Toggling Full Text grows the input from 60px to ~130px and pushes the controls row down 72px; the settings panel expands inline (no transition) and pushes the conversation.
- Settings icon swaps glyph (tune-variant -> cog) when open -- a meaningless change.

## 9. Provocative questions

1. If the product's value is "review the extracted phenotypes against the note", why is the primary layout a chat transcript instead of a two-pane note | findings workbench?
2. Should "Add all" exist at all in a tool whose disclaimer says results require independent review?
3. Who is the Settings panel for? If clinicians never change the embedding model, why is it the second thing they see after the mode switch?
4. Why does a research-only demo show a log viewer, storage bytes, and version build info to every visitor but hide the API/MCP links on mobile?
5. Could the disclaimer be an inline, dismissible banner tied to the first submit, so the first screen shows the product instead of a legal wall?
6. What would a "done" state look like -- is the end of a session an exported phenopacket, and if so why is export tucked behind an unlabeled FAB?
