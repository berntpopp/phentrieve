# B - Technical Audit: Phentrieve frontend (Vue 3.5 + Vuetify 3.13.7)

Date: 2026-10-27. Target: production bundle (`CI=1 vite build` into scratch `dist-prod`,
served by a gzip + SPA-fallback + `/api` proxy static server on :5799, API on :8734).
Repo untouched (CI=1 skips the visualizer that would write `frontend/dist/stats.html`).

Artifacts (all under the scratchpad):
- Lighthouse JSON+HTML: `lighthouse/{home,faq,connect}-{mobile,desktop}.report.{json,html}`
- axe: `axe/axe-results.json`, state screenshots `axe/*.png`
- Filmstrips: `filmstrip/light-4x/`, `filmstrip/dark-4x/`, `filmstrip/screencast-slow4g/`,
  montages `filmstrip/*-strip.png`
- Scripts: `scripts/{serve.mjs,lh.sh,axe.cjs,flicker.cjs,screencast.cjs,measure.cjs,contrast.cjs}`

## Audit health score

| Dimension | Score | Key evidence |
|---|---|---|
| Accessibility | 2/4 | LH 95-96, but 13-29 nameless `role=tooltip` nodes on every page, 2.5.3 label-in-name failures, FAQ uses wrong tab/tablist ARIA with `aria-label="undefined category"`, score chips 2.8:1 / 3.1:1 contrast (axe missed: "incomplete"), mobile footer actions unreachable, 26 hard-coded English aria-labels |
| Performance | 2/4 | Desktop 98, mobile 68-70: FCP 4.7 s, LCP 5.0-5.4 s. 394 KB icon font (all ~7k MDI glyphs for 93 used), 47 KB render-blocking Vuetify CSS 95% unused, blank page until JS mounts |
| Responsive | 2/4 | No page-level horizontal scroll, but footer row is 612 px wide and clipped at 320/390 px (6-8 actions invisible); `/connect` code blocks overflow; disclaimer CTA 146-370 px below the fold; FAB overlaps result rows |
| Theming | 1/4 | Vuetify default palette (no custom theme), `theme="light"` hard-coded in `App.vue:2`, OS dark preference ignored; 117 raw hex/rgba literals vs 29 `--v-theme-*` uses |
| Implementation integrity | 2/4 | `critical.css` is bundled (not inlined) so it does nothing; stale `index.html` hints/version; icon "optimizer" keeps 971 icon rules and the full font; hard-coded "Settings" text; reduced-motion handled in 1 file |
| **Total** | **9/20** | "Needs work": functional and fast on desktop, but mobile load, mobile footer, ARIA correctness and theming need a pass |

## 1. Impeccable detector

`impeccable detect --json frontend/src` -> exit 2, **2 findings** (both `frontend/src/components/SimilarityScore.vue`, imported by AggregatedTermsView, ChunkResultsView, ResultItem):

| # | Rule | Location | Verdict |
|---|---|---|---|
| 1 | `overused-font` (slop, warning) | SimilarityScore.vue:249 `font-family: 'Roboto Mono', monospace` | **Partial false positive.** Roboto Mono is never loaded (no webfont in the app), so it only resolves if installed locally; otherwise system monospace. Not a design-slop issue; it is a dead/inconsistent declaration. Fix: use `ui-monospace, SFMono-Regular, Menlo, monospace` + `font-variant-numeric: tabular-nums`. |
| 2 | `layout-transition` (quality, warning) | SimilarityScore.vue:207 (inline `transition: width 0.6s`), also :216 `left 0.6s` | **True positive, low impact.** Width/left animate inside the score tooltip scale only (`.scale-fill` at :58). Use `transform: scaleX()` with `transform-origin: left`. |

Detector misses (found manually): `HomeView.vue:167,184` animate `max-width` on the logo (layout property); `TutorialOverlay.vue:282,287` `transition: all`; `FAQView.vue:220` `transition: all`.

## 2. Lighthouse (production build, Lighthouse simulated throttling)

| Page | FF | Perf | A11y | BP | SEO | FCP | LCP | TBT | CLS | SI |
|---|---|---|---|---|---|---|---|---|---|---|
| / | mobile | **68** | 95 | 100 | 100 | 4.7 s | 5.4 s | 0 ms | 0 | 4.7 s |
| / | desktop | 98 | 95 | 100 | 100 | 0.8 s | 1.0 s | 0 ms | 0 | 0.8 s |
| /faq | mobile | **70** | 95 | 100 | 100 | 4.7 s | 5.0 s | 0 ms | 0 | 4.7 s |
| /faq | desktop | 98 | 95 | 100 | 100 | 0.8 s | 0.9 s | 0 ms | 0 | 0.8 s |
| /connect | mobile | **70** | 96 | 100 | 100 | 4.7 s | 5.0 s | 0 ms | 0 | 4.7 s |
| /connect | desktop | 98 | 96 | 100 | 100 | 0.8 s | 0.9 s | 0 ms | 0 | 0.8 s |

Note: the disclaimer modal shows on every first load, so LCP is the dialog text in all runs.

Transfer (home, mobile, gzip): **749 KB total** = Font 394 KB, JS 289 KB, CSS 60 KB.
Largest: `materialdesignicons-webfont.woff2` 394 KB; `vuetify.js` 114 KB; `HomeView.js` 66 KB;
`components.js` 50 KB; `index.js` 47 KB; `vuetify.css` 47 KB; `vendor.js` 12 KB; `index.css` 8 KB.
(/faq: 681 KB; raw build: vuetify.css 680 KB uncompressed.)

- Unused JS: vuetify.js 50/114 KB, HomeView.js 29/65 KB, components.js 27/50 KB (106 KB on /).
  `components` chunk (DisclaimerDialog+LogViewer+ResultsDisplay, vite.config.js manual group) loads on /faq and /connect too.
- Unused CSS: vuetify.css **44 of 46 KB** unused (full `vuetify/styles` incl. all utility classes, main.js:16).
- Render-blocking: vuetify.css (est. 1,501 ms), index.css (451 ms), components.css (301 ms); total est. savings 2.3-2.55 s mobile.
- Fonts: only the MDI icon font; `font-display` audit passes (swap) but swap produces tofu squares (see 4). Text uses system-ui stack, so no text font swap.
- bf-cache: fails ("Internal error", likely the open persistent dialog/websocket; minor).
- Failing a11y audits: `aria-tooltip-name`, `heading-order` (disclaimer h3 with no h2), `label-content-name-mismatch`.

## 3. axe-core 4.x (WCAG 2.0/2.1/2.2 A+AA + best-practice), 1280x900

| State | Violations (impact, count) |
|---|---|
| / with disclaimer | aria-tooltip-name (serious, 16); label-content-name-mismatch (serious, 2: `.search-controls-row__settings`, disclaimer `.v-btn--slim`); heading-order (moderate, 1: `#disclaimer-content h3`) |
| / dismissed (light) | aria-tooltip-name (serious, 17); label-content-name-mismatch (serious, 1: `.search-controls-row__settings`) |
| / settings panel open | aria-tooltip-name (21); label-content-name-mismatch (1) |
| / after "seizures" query | aria-tooltip-name (29); label-content-name-mismatch (1) |
| / dark (`prefers-color-scheme: dark`) | identical to light: the app ignores dark preference (`App.vue:2 theme="light"`) |
| /faq | aria-tooltip-name (13); label-content-name-mismatch (2) + **nested-interactive (serious, 2)**: `.v-expansion-panel.faq-panel[role="tab"]` |
| /connect | aria-tooltip-name (13); **scrollable-region-focusable (serious, 2)**: `.snippet-wrap > .snippet` (McpAccess.vue:139) |

No `color-contrast` violations reported, but axe returned "incomplete" (background overlapped) for the results area. Manual check of computed colors:
- Score chips `bg-success` white on #4CAF50 = **2.78:1**, `bg-info` white on #2196F3 = **3.12:1** (14px/600 text needs 4.5:1). SimilarityScore.vue:162-163 -> Vuetify default `success`/`info`.
- "Affirmed" chip text #4CAF50 on tinted white ~2.6:1.

Root causes:
- **Tooltips**: Vuetify puts `role="tooltip"` on the `.v-overlay` root; the repo passes names via `:content-props="{ 'aria-label': ... }"` (App.vue:17,41,71 and 30 `v-tooltip`s total), which lands on `.v-overlay__content`, not the role node. Empty overlay roots persist after first render.
- **Label mismatch**: `QueryModeControls.vue:39` aria-label "Open advanced options" vs visible "Settings" (:48, also hard-coded English); `DisclaimerDialog.vue:59` "Accept disclaimer and continue..." vs visible "I Understand and Agree".
- **FAQ**: `FAQView.vue:52-58` forces `role="tablist"`/`role="tab"` on an accordion and `:aria-label="category.title + ' category'"` where `category.title` does not exist -> literal **"undefined category"**; `:69` `'Question: ' + qa.question` same pattern (titles come from i18n keys at :61/:72).

## 4. Flicker / CLS / long frames

Measured with PerformanceObserver (`layout-shift`, `long-animation-frame`, `longtask`), 390x844 mobile, 4x CPU throttle, through first query result:

- **CLS = 0.000** for load and query (only one 0.0014 shift with `hadRecentInput`, `DIV.d-flex.align-center`). Results render without skeleton->content jumps (`filmstrip/light-4x-query-strip.png`).
- Long frames: boot LoAF 435-507 ms (blocking 385-456 ms) in `index.js` + `vuetify.js`; HomeView mount 166-221 ms; post-query 130-160 ms frames in `components.js`/`vuetify.js` (result list render). TBT 0 under Lighthouse's model, but 4x-CPU boot is ~0.5 s of main-thread block.
- **Blank-until-mount**: screencast on slow 4G (150 ms RTT, 1.6 Mbps) + 4x CPU (`filmstrip/screencast-slow4g-strip.png`): frames 0-1386 ms pure white; 2270 ms grey shell with no text; 2337 ms dialog fading in. FCP 2272 ms. `#app` is empty in `index.html` - there is no static shell.
- **Icon pop-in (tofu)**: icon font finishes at 4768 ms; from 2.3 s to 4.8 s every icon renders as an empty square (dialog warning icon, footer row, account icon - frames 02420-03513 vs 04811). Caused by swap on a 394 KB icon font.
- **Late route CSS**: `HomeView.css` arrives at 2518 ms, after FCP (2272 ms) - masked by the modal on first visit; may cause a restyle on repeat visits when the dialog is absent.
- **Theme flash**: none possible - there is no dark theme; `prefers-color-scheme: dark` renders light (`filmstrip/dark-4x/` identical to light). Minor white (#fff) -> #f5f5f5 body change at mount.
- `critical.css` (src/critical.css) is imported from `main.js:7`, so it is bundled into `index.css` and is render-blocking like everything else; it never reaches first paint early. Its `@font-face { font-family:'Material Design Icons'; src: local(...) }` is a no-op duplicate.

## 5. Static / runtime checks

- **Footer at 320/390 px** (App.vue:11-250): footer content `scrollWidth` 612 px in a 320/390 px footer; overflow clipped, no wrap/menu. At 320 px visible: disclaimer, language, settings, tutorial, half the FAQ icon. Hidden: log viewer(390 shows), GitHub, docs, license, API docs, **MCP/Connect**, version/status. (`axe/home-320.png`)
- **Touch targets < 44 px at 390 px**: footer icon buttons 40x40 (13 buttons, `size="small"`), language selector 63x28, account button 28x28, search submit 34x34, mode pills 90x28 / 105x28, Settings 106x34, HPO ID links 112x24 (10 per result list), /connect copy buttons 32x32, "COPY" 79x28, "REST API DOCS" 142x28, FAQ back 173x36.
- **Horizontal overflow**: no document scroll at 320/390/768. `/connect` `<code>` lines extend to 600 px inside cards at 320/390 (scrollable but not keyboard-focusable).
- **Disclaimer on mobile**: CTA at y=1170 (320) / 946 (390) in an 800 px viewport; persistent modal with its primary action below the fold.
- **FAB** (case workspace) overlaps the "+ add" and score area of result rows at 390 px (`filmstrip/light-4x-query-strip.png`).
- **prefers-reduced-motion**: only `SimilarityScore.vue:470`. Unguarded: `HomeView.vue:118` infinite pulse, `ConversationSkeleton.vue:191` infinite shimmer, `ChunkResultsView.vue:211` flash, logo `max-width` transitions, Vuetify transitions globally.
- **Icon-only buttons without names**: none found (all have aria-label). But 26 aria-labels are hard-coded English in 9 files (App.vue, DisclaimerDialog, FAQView, TutorialOverlay, LanguageSwitcher, QueryModeControls, ConversationSkeleton, McpAccess, LogViewer) - wrong for the German/other locales.
- **Headings**: one H1 per page. Home H1 (`HomeView.vue:19-22`) exposes "Phen[otype re]trieve" to AT (hidden spans use max-width/opacity, not display:none). Disclaimer h3 without h2.
- **lang**: `<html lang="en">` static; check that LanguageSwitcher updates `document.documentElement.lang` (not verified).
- **Hard-coded colors**: 117 hex/rgba literals (top: SimilarityScore 34, FullTextWorkspace 19, QueryInterface 15, FAQView 10, ConversationSkeleton 9) vs 29 `var(--v-theme-*)` uses; `plugins/vuetify.js:132` defines no colors.
- **index.html**: preconnect to fonts.googleapis.com (no Google Fonts used), dns-prefetch to api.phentrieve... (prod uses relative `/api`), JSON-LD `softwareVersion "0.15.0"` stale, comment "No modulepreload" contradicted by the build.

## Issue list

### P0
None (no blocking breakage; core query flow works on all tested widths).

### P1
1. **Footer actions unreachable on phones** - `frontend/src/App.vue:11` (v-footer, 13 icon buttons in one non-wrapping row). Fix: below `sm`, collapse secondary links (GitHub, docs, license, API, MCP, version) into a single `v-menu` "More" button, or `flex-wrap: wrap` + 2 rows; keep 44 px targets.
2. **Score chip / status contrast 2.6-3.1:1** - `frontend/src/plugins/vuetify.js:132` (no theme colors), used by `components/SimilarityScore.vue:162-163`. Fix: define theme `colors: { success: '#2E7D32', info: '#1565C0', ... }` (>= 4.5:1 with white) instead of per-component overrides.
3. **Mobile FCP 4.7 s / 394 KB icon font + tofu pop-in** - `frontend/src/main.js:17` (`@mdi/font` css) + `plugins/vuetify.js` `iconsets/mdi`. Fix: switch to `vuetify/iconsets/mdi-svg` + `@mdi/js` named imports (only the ~93 used icons ship, no font, no tofu); delete `vite-icon-optimizer.js` (it keeps 971 rules and never subsets the font). Interim: `pyftsubset` the woff2 to used codepoints and set `font-display: block`.
4. **Nameless tooltips (axe serious, 13-29/page)** - `App.vue:17,41,71` and all 30 `<v-tooltip>` using `:content-props="{'aria-label'}"`. Fix: bind `:aria-label` on `v-tooltip` itself (falls through to the `.v-overlay` root that carries `role="tooltip"`), or set a global `defaults.VTooltip` and verify with axe; tooltips duplicating the button's aria-label can drop the role entirely.
5. **FAQ broken ARIA** - `views/FAQView.vue:52-58,69`. Fix: remove `role="tablist"`, `role="tab"`, `role="listitem"` and the custom aria-labels (they read "undefined category"/"Question: undefined"); Vuetify's panel title button already has `aria-expanded`.

### P2
6. **Render-blocking 47 KB Vuetify CSS, 95% unused** - `main.js:16` `import 'vuetify/styles'`. Fix: use `vite-plugin-vuetify` with `styles: { configFile }` / `'sass'` and disable unused utility groups (`$utilities: false` per group) so only used component CSS ships.
7. **No static shell; white screen until JS** - `frontend/index.html:47` (`<div id="app"></div>`), `src/critical.css` bundled via `main.js:7`. Fix: move the critical rules into an inline `<style>` in index.html and render a lightweight header/search skeleton inside `#app` (replaced on mount); set `body{background:#f5f5f5}` inline.
8. **Label-in-name mismatch (WCAG 2.5.3)** - `components/query/QueryModeControls.vue:39` and `:48`; `components/DisclaimerDialog.vue:59`. Fix: drop the aria-labels (visible text suffices) or start them with the visible text; i18n the hard-coded "Settings".
9. **Touch targets < 44 px** - footer `App.vue` `size="small"` buttons (40x40), account button 28x28, `QueryModeControls` pills 28 px tall, `McpAccess.vue` copy buttons 32x32, result HPO links 24 px tall. Fix: `size="default"`/`min-height: 44px` on coarse pointers (`@media (pointer: coarse)`).
10. **Disclaimer CTA below the fold on mobile** - `DisclaimerDialog.vue:2` (`max-width="600"`, content scrolls). Fix: `scrollable` dialog with sticky `v-card-actions`, or `fullscreen` on `xs`.
11. **Reduced motion ignored** - `views/HomeView.vue:118`, `components/ConversationSkeleton.vue:191`, `components/ChunkResultsView.vue:211`, `components/TutorialOverlay.vue:282`. Fix: one global `@media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms!important; animation-iteration-count:1!important; transition-duration:.01ms!important } }` in the app stylesheet.
12. **/connect snippets not keyboard scrollable** - `views/McpAccess.vue:139`. Fix: `tabindex="0"` + `role="region"` + aria-label on `.snippet`.
13. **Hard-coded English aria-labels (26 in 9 files)** - e.g. `App.vue:28,81,102,121,143,163,183,205,223,240`. Fix: route through `$t()`.
14. **Garbled H1 accessible name** - `views/HomeView.vue:19-22`. Fix: `aria-label="Phentrieve"` on the h1 and `aria-hidden="true"` on the bracket/hidden spans.

### P3
15. **No dark theme; theme forced light** - `App.vue:2`, `plugins/vuetify.js:132`. Fix (if desired): add a `dark` theme and follow `prefers-color-scheme`; required before reducing the 117 raw color literals to theme tokens (start with SimilarityScore.vue and FullTextWorkspace.vue).
16. **Layout-property transitions** - `SimilarityScore.vue:207,216` (width/left), `HomeView.vue:167,184` (max-width), `transition: all` at `TutorialOverlay.vue:282,287`, `FAQView.vue:220`, `SimilarityScore.vue:235`. Fix: transform/opacity and named properties.
17. **Dead/stale head hints** - `index.html:8-9` (unused Google Fonts preconnect, prod API dns-prefetch), JSON-LD `softwareVersion "0.15.0"`, stale "No modulepreload" comment. Fix: remove hints; inject version at build.
18. **Roboto Mono never loaded** - `SimilarityScore.vue:249`. Fix: system monospace stack + `tabular-nums`.
19. **`components` manual chunk loads on every route** - `vite.config.js` codeSplitting group (DisclaimerDialog/LogViewer/ResultsDisplay). Fix: split ResultsDisplay out so /faq and /connect don't pay ~50 KB.
20. **Disclaimer heading order** - `DisclaimerDialog.vue` h3 under dialog title. Fix: use h2 or make the title an h2.
21. **FAB overlaps result rows on 390 px** - `components/CaseWorkspacePanel.vue` open button. Fix: add bottom padding to the results list equal to FAB height + 16 px on `xs`.

Not checked: real-device performance, screen reader walkthrough, Full Text mode / PII dialog / auth views, keyboard focus order and focus trap in drawers (closed drawers had 0 focusable elements), LanguageSwitcher updating `lang`.
