# Phentrieve UI - Test Protocol, Monkey Test and Responsiveness Report (C)

Date: 2026-10-27. Target: Vite dev frontend `http://localhost:5734` (Frontend 0.18.4) against API `http://localhost:8734` (API 0.18.7, CLI 0.28.7, model BioLORD-2023-M).
Tooling: Node + playwright-core driving headless `/usr/bin/google-chrome`, one fresh browser context per case (disclaimer pre-acknowledged through storageState unless noted), off-origin requests blocked.
Scripts: `$S/tools/qa-lib.mjs`, `qa-protocol.mjs`, `qa-monkey.mjs`, `qa-perf.mjs`, plus `qa-dbg*.mjs` for verification. Raw results: `$S/qa/*.json`. Screenshots: `$S/shots-qa/`.
(`$S` = `/tmp/claude-1000/-home-bernt-popp-development-phentrieve/b91ab283-17da-4e9f-aefe-1ec0819e9d26/scratchpad`)

**Totals: 59 cases, 38 PASS, 21 FAIL (5 P1, 8 P2, 8 P3, 0 P0).** Monkey: about 4,150 random actions over 3 seeds, desktop and mobile, with 0 uncaught exceptions, 0 Vue warnings, 0 unhandled rejections and 0 genuinely stuck overlays.

## 1. Brainstorm - what to test in a clinical phenotype-term search tool

Comparable tools are Phenomizer, Doc2Hpo, PhenoTips, Monarch search, ClinPhen and PhenoTagger. Their usual failure classes:

- **Retrieval correctness**: exact labels, synonyms, lay terms, misspellings, nonsense input with confident-looking garbage, and score calibration.
- **Multilingual input**: DE/ES/FR/NL text, diacritics and compound nouns. Which language does the system *assume*: the UI locale, auto-detection, or an explicit choice?
- **Negation and assertion**: "no X", "denies X", "X ruled out", "kein X", "pas de X". Negation must (a) be flagged and (b) not distort which concept is retrieved. Negated findings must export as `excluded` in phenopackets.
- **Long notes / full text**: 10k to 50k characters, size limits, PII scrubbing, LLM back-end availability, multi-line editing.
- **Input robustness / security**: emoji, CJK, zero-width and bidi characters, HTML/script injection in queries and deep links, whitespace-only input, empty submit.
- **Concurrency**: rapid Enter, double-click, clicks during loading, auto-submit races.
- **Navigation**: deep links (`?text=`), parameter validation, back/forward, unknown routes, refresh mid-request, re-submit on reload.
- **Curation workflow**: add/remove/toggle assertion, duplicates, persistence across reload (localStorage), destructive actions (clear/reset), and export to Phenopacket JSON and text.
- **i18n**: switching language mid-session, hard-coded strings, `<html lang>`, persistence.
- **Theming**: dark mode / `prefers-color-scheme`.
- **Keyboard and accessibility**: Tab order, skip link, Enter submit, Esc closing dialogs, focus trapping, focus return, accessible names.
- **Responsive layout**: 390px viewport, horizontal overflow, clipped controls, drawers wider than the viewport.
- **Network resilience**: Slow 3G loading feedback, API down, 4xx/5xx mapping, offline, hung requests (timeout or cancel).
- **Perceived performance**: time to loader, time to results, layout shift and flicker.

## 2. Protocol (executed)

`Console / network` lists the per-case console errors and warnings, Vue warnings, page errors and failed or >=400 requests that were captured. Rows marked *(verified)* in the bug list were re-checked with dedicated debug scripts, because two harness bugs produced false results in the first run: newest-first history order, and a locator that hit the wrong drawer. Those rows have been corrected.

| ID | Area | Steps | Expected | Actual | Result | Sev | Evidence | Console / network (abridged) |
|---|---|---|---|---|---|---|---|---|
| Q01 | Query | Type "seizures", press Enter | Top hit HP:0001250 Seizure; spinner then results | 10 ids, top HP:0001250 | **PASS** | - | Q01.png | - |
| Q02 | Query | Type "short stature", click search button | Results with HP:0004322 in top 3 | top3 HP:0004322,HP:0003502,HP:0003508 | **PASS** | - | Q02.png | - |
| Q03 | Query | Input whitespace-only "   " then Enter / inspect button | No request; search button disabled | requests=0, button disabled=true | **PASS** | - | Q03.png | - |
| Q04 | Query | Query "xqzvbnm" (nonsense) | Graceful "no results"/low-score display, no error | Model: BioLORD-2023-M Language: en Assertion: Affirmed 1 HP:0001417 X-linked inheritance 0.58 2 HP:0000473 Torticollis 0.54 3 HP:6000011 Guilt 0.51 4 HP:0001084 | **PASS** | - | Q04.png | - |
| L01 | Multilingual | UI in English (query language defaults to UI locale "en"); query "Krampfanfälle" | HP:0001250 in top 5 | sent language=en; top5 HP:0001250,HP:0011145,HP:0011097,HP:0033348,HP:0033349 | **PASS** | - | L01.png | - |
| L02 | Multilingual | UI in English (query language defaults to UI locale "en"); query "convulsiones" | HP:0001250 in top 5 | sent language=en; top5 HP:0001250,HP:0011145,HP:0033348,HP:0011097,HP:0007359 | **PASS** | - | L02.png | - |
| L03 | Multilingual | UI in English (query language defaults to UI locale "en"); query "crises d'épilepsie" | HP:0001250 in top 5 | sent language=en; top5 HP:0001250,HP:0011145,HP:0011097,HP:0033348,HP:0033349 | **PASS** | - | L03.png | - |
| L04 | Multilingual | UI in English (query language defaults to UI locale "en"); query "Kleinwuchs" | HP:0004322 in top 5 | sent language=en; top5 HP:0004322,HP:0003510,HP:0001510,HP:0003502,HP:0003508 | **PASS** | - | L04.png | - |
| L05 | Multilingual | UI in English (default); query German negation "kein Krampfanfall" | Assertion Negated (language should follow the text, not the UI locale) | Assertion chip "Affirmed" (UI en sends language=en + query_assertion_language=en); top-1 HP:0002121 absence seizure | **FAIL** | P1 | L05.png | - |
| N01 | Negation | query "no seizures" | Assertion chip shows Negated; seizure term retrieved | negation visible=true; top=HP:0002121; lang sent=en | **PASS** | - | N01.png | - |
| N02 | Negation | Switch UI to de; query "kein Krampfanfall" | Assertion chip shows Negated; seizure term retrieved | negation visible=true; top=HP:0002121; lang sent=de | **PASS** | - | N02.png | - |
| N03 | Negation | Switch UI to fr; query "pas de convulsions" | Assertion chip shows Negated; seizure term retrieved | negation visible=true; top=HP:0002121; lang sent=fr | **PASS** | - | N03.png | - |
| N04 | Negation | query "seizures were ruled out" | Assertion chip shows Negated; seizure term retrieved | negation visible=false; top=HP:0002121; lang sent=en | **FAIL** | P2 | N04.png | - |
| N05 | Negation | Query "no seizures"; check top-1 term | Top-1 is HP:0001250 Seizure (negated), not an unrelated seizure subtype | top1=HP:0002121 (Generalized non-motor (absence) seizure); HP:0001250 rank=absent | **FAIL** | P1 | N05.png | - |
| S01 | Input robustness | Query with emoji/unicode "Krampf 😀 ✓ ß 中文 \u200b" | No crash; results or graceful message; text echoed intact | echo="U  Krampf 😀 ✓ ß 中文 ​"; bubble=Model: BioLORD-2023-M Language: en Assertion: Affirmed 1 HP:0002380 Fasciculatio | **PASS** | - | S01.png | - |
| S02 | Security | Query `<img src=x onerror=alert(1)><script>alert(2)</script>` | Rendered as literal text, no JS dialog, no injected img | injected img=0, dialogs=0 | **PASS** | - | S02.png | - |
| S03 | Security | Deep link /?text=<svg onload=alert(1)> | Literal text, no dialog | dialogs=0; svg in DOM=0 | **PASS** | - | S03.png | - |
| S04 | Input robustness | Paste 50,000-char note (repeated clinical sentence), observe mode, submit | Auto-switch to Full Text; UI stays responsive; either processed or clear size-limit message | fill 41ms; mode=fulltext; requests=1; piiDialog=false; outcome="The service is temporarily unavailable. Please try again later. (Status: 503)" (note: same 503 LLM-config failure as S06; the 50k payload itself was accepted, no freeze, fill 41ms) | **PASS** | - | S04-50k.png | error: Failed to load resource: the server responded with a status of 503 (Service Unavailable); 503 POST http://localhost:5734/api/v1/text/process |
| S05 | Input robustness | Full Text mode: type line1, Shift+Enter, line2 | Newline inserted (multi-line clinical notes must be typeable) | Shift+Enter submitted the note ("line one") instead of inserting a newline; textarea cleared | **FAIL** | P1 | S05.png | error: Failed to load resource: the server responded with a status of 503 (Service Unavailable); 503 POST http://localhost:5734/api/v1/text/process |
| S06 | Full text / LLM | Full Text: "Patient has seizures. No hypotonia." -> Enter | Extraction result OR clear actionable error (LLM key missing) | Generic red alert "The service is temporarily unavailable. Please try again later. (Status: 503)". Real cause (API detail): "Unable to evaluate LLM quota state. Verify PHENTRIEVE_LLM_QUOTA_DB_PATH..." -- detail discarded, no hint to switch to the Standard backend | **FAIL** | P2 | S06-fulltext.png | error: Failed to load resource: the server responded with a status of 503 (Service Unavailable); 503 POST http://localhost:5734/api/v1/text/process |
| R01 | Concurrency | Fill "seizures", press Enter 5x in 50ms | Exactly 1 request and 1 history entry | requests=1, history=1 | **PASS** | - | R01.png | - |
| R02 | Concurrency | Fill "seizures", double-click search button | 1 request | requests=1, history=1 | **PASS** | - | R02.png | - |
| R03 | Concurrency | Deep link /?text=seizures (text stays in box during auto-submit, API delayed 2s) then click search button twice while loading | 1 request (button must be inert while loading) | requests=1, history=1 | **PASS** | - | R03.png | - |
| D01 | Deep link | Open /?text=short%20stature | Auto-submits; results shown; autoSubmit param handled | history=1, url=/?text=short%20stature, top=HP:0004322 | **PASS** | - | D01.png | - |
| D02 | Deep link | Open /?text=seizures&autoSubmit=false | Text prefilled, no submit | value=seizures, history=0 | **PASS** | - | D02.png | - |
| D03 | Deep link | Open /?text=seizures&forceEndpointMode=bogus&threshold=7 | Invalid params ignored/sanitized, query still runs | Query ran (1 request), but forceEndpointMode=bogus is stored unvalidated: neither "Query" nor "Full Text" toggle is shown active | **FAIL** | P3 | D03-bogus-mode.png | - |
| D04 | Navigation | Home -> query -> FAQ (footer) -> Back -> Forward -> Back | History preserved, no errors | history after back=1, after fwd/back=1 | **PASS** | - | D04.png | - |
| D05 | Navigation | Open unknown route /does-not-exist | A 404/not-found page or redirect home | Blank main area (only footer + hidden log drawer); Vue Router warns "No match found"; no 404 page / redirect | **FAIL** | P3 | D05-404.png | warning: [VUE_ROUTER_R0004] No match found for location with path "/does-not-exist" ├▶ fix: Add a route matching this path or check for typos in the location. ╰▶ see: https://router.vuejs.org/guide/essentials/dynamic-mat |
| D06 | Refresh | Submit query with API delayed 4s, reload page mid-request | After reload: no eternal spinner; turn shows error/retry or is dropped | spinners after reload+5s=1; history=1; stored={'queryHistory':[{'id':'cd14054d-58d7-4470-b853-7549b52660c5','timestamp':'2026-10-27T21:37:52.102Z','loading':true,'response':null,'error':null,'type':'query', | **FAIL** | P1 | D06-reload-midrequest.png | - |
| D07 | Deep link | Open /?text=seizures, wait for result, reload page | Reload does not re-submit (no duplicate turn) | url after submit=/?text=seizures; history after reload=2 | **FAIL** | P3 | D07.png | - |
| C01 | Collection | Query seizures, click + on first result | Panel opens with HP:0001250; + becomes check/disabled | panel has id=true; add btn disabled=true | **PASS** | - | C01-collection.png | - |
| C02 | Collection | Add same term from two different queries ("seizures", "seizure") | No duplicate in collection | second add btn disabled=true; collected=1 | **PASS** | - | C02.png | - |
| C03 | Persistence | Add term, reload page | History + collection restored from localStorage | collected=1, history=1, results ids visible=10 | **PASS** | - | C03.png | - |
| C04 | Collection | Toggle assertion on collected term, then remove it | Assertion flips; removal empties list | assertion after toggle=negated; count after remove=0 | **PASS** | - | C04.png | - |
| C05 | Export | Collect 2 terms, set subject id "P1", Export as Phenopacket | Valid JSON phenopacket download with 2 phenotypicFeatures | After working around C09/C10 (close drawer via scrim): phentrieve_phenopacket_<ts>.json, subject.id=P1, 2 features HP:0001250,HP:0004322, schema 2.0.0. Harness run itself FAILED because the open drawer blocks the result list. | **PASS** | - | C05.png | - |
| C06 | Export | Collect 1 term, Export as Text | Text download containing HP id | file=hpo_phenotypes.txt 118B has id=true | **PASS** | - | C06.png | - |
| C07 | Collection | Collect 2 terms, click "Clear Collection" | Confirmation before destroying curated work (or undo) | confirm dialog=0; undo snackbar=0; remaining=0 | **FAIL** | P2 | C07.png | - |
| C08 | Export | Phenopacket subject DOB "2030-99-99" then export | Validation message or sanitized DOB | Native date input rejected "2030-99-99" (browser sanitized); export succeeded without invalid DOB | **PASS** | - | C08.png | warning: The specified value "2030-99-99" does not conform to the required format, "yyyy-MM-dd". |
| C09 | Collection | Add a term (drawer auto-opens), click the drawer close "X" at top-right | Drawer closes | Click intercepted: fixed AccountButton (z-index 2000, top:8px right:12px) sits on top of the close X; Playwright: "<i mdi-account-outline> from <div class=account-button> intercepts pointer events". Escape also does not close the drawer; only a scrim click does | **FAIL** | P1 | C01-collection.png | - |
| C10 | Collection | 1440px: add a term, then try to add a second term from the next query | Can keep adding terms while the workspace is open | Temporary drawer (400px + scrim) covers the results column and its + buttons; each additional term requires closing the drawer first | **FAIL** | P2 | C01-collection.png | - |
| I01 | i18n | Run query (en), switch UI language to Deutsch via switcher | UI strings switch; history kept; subsequent queries use de | history=2; query language sent after switch=en; input label=""; html lang=en | **FAIL** | P2 | I01-de.png | - |
| I02 | i18n | Set lang=de, reload | German persists across reload | German UI strings persist after reload (phentrieve-lang=de) | **PASS** | - | I02.png | - |
| T01 | Theme | Emulate prefers-color-scheme: dark; load | Dark theme honoured (or explicit toggle exists) | app bg=rgb(255, 255, 255); v-app theme class=v-theme--light | **FAIL** | P3 | T01-dark.png | - |
| K01 | Keyboard | Fresh load, press Tab x8 and record focus order | First stop is skip-link or query input; logical order | Sign in or register > Reset conversation > Clinical text input field > Query > Full Text > Open advanced options > Open Case Workspace Panel > View disclaimer information | **FAIL** | P3 | K01.png | - |
| K02 | Keyboard | Tab to query input, type, Enter, then Tab to first result + button, press Enter | Query submitted and term added via keyboard only | reached add-btn=true; collected=1 | **PASS** | - | K02.png | - |
| K03 | Keyboard | Open Version dialog (footer info) with keyboard, Tab x10, Esc | Focus trapped inside dialog; Esc closes; focus returns to opener | focus escaped dialog 0/10 tabs; open after Esc=0; focus after close="null" | **FAIL** | P2 | K03.png | - |
| K04 | Keyboard | Activate logo (Reset) with Enter, press Esc | Reset dialog opens; Esc closes (or Cancel focused) | opened=1; still open after Esc=1; focused="Reset Application? T" | **FAIL** | P3 | K04.png | - |
| K05 | Keyboard | Fresh profile: disclaimer dialog; Tab x6; Esc | Focus trapped in disclaimer; Esc does not bypass legal ack (persistent) | focus outside 0/6; still open after Esc=true | **PASS** | - | K05.png | - |
| K06 | Keyboard | Open tutorial (footer), press Esc | Tutorial overlay closes with Esc | tutorial still visible after Esc=true | **FAIL** | P3 | K06-tutorial.png | - |
| A01 | A11y | Inspect disclaimer accept button accessible name | Accessible name contains visible text (WCAG 2.5.3) | Visible "I UNDERSTAND AND AGREE", aria-label "Accept disclaimer and continue to application" | **FAIL** | P3 | x-01-first.png | - |
| M01 | Mobile | 390x844 viewport: load, run query, add term, open panel | No horizontal overflow; controls reachable; footer not covering results | No page-level h-overflow, but footer scrollWidth 612 > 390 and is clipped: Docs, License, API, MCP, Version/status buttons are off-screen at x=387..572 and unreachable by touch | **FAIL** | P2 | M01-mobile-results.png, M01-mobile-panel.png | - |
| NW01 | Slow network | CDP Slow-3G (400kbps, 400ms RTT); submit "seizures" | Loading indicator within 100ms; results eventually | loader at +45ms; results at +575ms | **PASS** | - | NW01.png | - |
| NW02 | API down | Abort all /api/v1 requests; load app and submit | Clear error alert; status indicator shows disconnected; no uncaught errors | msg="Could not connect to the server. Please check your internet connection and try again."; uncaught=0 | **PASS** | - | NW02-api-down.png | error: Failed to fetch versions from API: AxiosError: Network Error     at XMLHttpRequest.handleError (http://localhost:5734/node_modules/.vite/deps/axios.js?v=9e81197b:2756:16)     at Axios$1.request (http://localhost:5 |
| NW03 | API error | API returns 500 {"detail":"boom"} for /query | Readable error message, can retry | "A server error occurred. Please try again later. (Status: 500)"; retry after unroute returned results (first harness run misread the oldest bubble) | **PASS** | - | NW03.png | error: Failed to load resource: the server responded with a status of 500 (Internal Server Error); 500 POST http://localhost:5734/api/v1/query/ |
| NW04 | API error | API returns 422 validation error array for /query | Readable message, not "[object Object]" | msg="The input was rejected. Check it and note the size limits: 50,000 characters for full text, 10,000 for a single query. (Status: 422)" | **PASS** | - | NW04-422.png | error: Failed to load resource: the server responded with a status of 422 (Unprocessable Entity); 422 POST http://localhost:5734/api/v1/query/ |
| NW05 | Offline | context.setOffline(true); submit query | Offline-specific or network error message, no spinner stuck | msg="Could not connect to the server. Please check your internet connection and try again."; spinner=0 | **PASS** | - | NW05-offline.png | error: Failed to load resource: net::ERR_INTERNET_DISCONNECTED; POST http://localhost:5734/api/v1/query/ net::ERR_INTERNET_DISCONNECTED |
| NW06 | API timeout | API never responds for /query (hang 70s) | Request times out with message; or cancel available | still spinning after 65s=true; cancel button=0 | **FAIL** | P2 | NW06.png | - |
| O01 | Routes | Open /faq and /connect directly | Pages render, no console errors | faq chars=1277, connect chars=4473, errors=0 | **PASS** | - | O01.png | - |
| O02 | Links | Inspect HPO external link on result | target=_blank with rel=noopener | {"href":"https://hpo.jax.org/browse/term/HP:0001250","target":"_blank","rel":"noopener noreferrer"} | **PASS** | - | O02.png | - |
## 3. Monkey test (seeded gremlin)

Script: `qa-monkey.mjs <seed> <secondsPerViewport>`. It uses a mulberry32 PRNG. Action mix:

- 45% click on a random visible interactive element (or a blank point)
- 20% typing from a corpus with clinical words, emoji, CJK, `‮` bidi, HTML, SQL-ish text and 300-character strings, half of it into the query box, often followed by Enter
- 20% key presses (Tab, Shift+Tab, Enter, Escape, Space, arrows, Backspace)
- 10% wheel scroll
- 5% viewport toggles (desktop / mobile / tablet)

It ran 110 s at 1440x900 and then 110 s at 390x844. After every step it probes for: uncaught exceptions (`pageerror`), console errors, `[Vue warn]`, `unhandledrejection` (an init-script hook), JS dialogs, horizontal overflow (`scrollWidth > innerWidth`), and overlays still open after 40 steps plus an Escape (stuck overlay).

Guards: off-origin requests are aborted and popups are closed. `POST /api/v1/auth/*` is stubbed with 400 so no junk accounts get created, and `/api/v1/text/*` is stubbed with 503 so no paid LLM calls are made. Because of that, the 503 console errors below are self-inflicted.

| Seed | Steps | Query submits | Uncaught exc. | Vue warn | Unhandled rej. | Console errors | H-overflow | "Stuck" overlay | Popups (external links, closed) |
|---|---|---|---|---|---|---|---|---|---|
| 1337 | 1423 | 7 | 0 | 0 | 0 | 0 | 0 | 3x disclaimer (first step 46) | 10 |
| 4242 | 1419 | 4 | 0 | 0 | 0 | 0 | 0 | 1x disclaimer (step 329) | 23 |
| 20261027 | 1303 | 35 | 0 | 0 | 0 | 33x 503 (stubbed `/text/process`) | 0 | 3x disclaimer (step 156) | 19 |

Interpretation:
- The "stuck overlay" hits are the **Research Use Disclaimer**. The gremlin re-opened it from the footer, and it is `persistent` by design (Esc does not close it, the Accept button does). That is expected, not a bug.
- No crash, Vue warning or unhandled rejection was found in about 4,150 random actions. The app is robust under chaotic input.
- The page-level overflow probe was clean, but it **missed** the clipped footer at 390px (bug B-09). The footer overflows inside a clipping ancestor, so `document.scrollWidth` stays at 390. That gap was found by the deterministic case M01.
- Reproduce with `node $S/tools/qa-monkey.mjs <seed> 110`. Per-step logs are in `$S/qa/monkey-<seed>-steps.log`, events with the last 5 actions in `$S/qa/monkey-<seed>.json`.

## 4. Responsiveness

Measured on the local stack: the API on the same host, about 20 to 90 ms per query. Script `qa-perf.mjs` uses a MutationObserver from the Enter keypress, a `layout-shift` PerformanceObserver, and a CDP screencast filmstrip.

Navigation: DOMContentLoaded 178 ms, load 210 ms, FCP 212 ms.

| Query | Loader visible | Results in DOM | Results painted | Layout shift sum (all hadRecentInput) |
|---|---|---|---|---|
| seizures (1st) | 28 ms | 88 ms | 100 ms | 0 |
| short stature | 23 ms | 99 ms | 107 ms | 0.182 |
| intellectual disability | 22 ms | 90 ms | 99 ms | 0.182 |
| Krampfanfaelle | 33 ms | 103 ms | 112 ms | 0.182 |
| no hypotonia | 26 ms | 96 ms | 107 ms | 0.182 |

- **Loading indicator within 100 ms: yes, every time** (22 to 33 ms). Under CDP Slow-3G (case NW01) the loader appeared at 45 ms and results arrived at 575 ms.
- **Flicker**: the filmstrips (`shots-qa/perf-filmstrip-q1.jpg`, `perf-filmstrip-q2.jpg`, 100 ms buckets) show a clean swap with no blank frame and no spinner flash at local latency.
- **Layout shift**: every query after the first shifts the previous turn (`.mb-4`) by 0.029 when it is pushed down 365->481px, then by 0.153 when it jumps 481->0 (`scrollToTop` on the conversation container). It is excluded from CLS because of `hadRecentInput`, so it is not a Core Web Vitals problem. It is still a visible jump of the previous answer, which the user can lose if they had scrolled to compare (P3, B-17).

## 5. Bug list (ranked)

### P1

**B-01 Negated query retrieves the wrong concept (top-1 is "absence seizure").** *(verified via UI and API)*
- Repro: query `no seizures`, `denies seizures` or `kein Krampfanfall`.
- Result: top-1 is `HP:0002121 Generalized non-motor (absence) seizure` at 0.92 to 0.97. `HP:0001250 Seizure` is not in the top 10. The negation cue ("no"/"absence") is embedded together with the concept, so a curator who clicks + collects the wrong term (case N05).
- Cause: `phentrieve/retrieval/api_helpers.py:183` embeds `segment_to_process = text.strip()`, the raw negated text.
- Fix: when the assertion detector reports negated or normal, embed the text with the cue span removed. The detector `details` already locate the cue. Alternatively, run retrieval on both and prefer the cue-stripped result.

**B-02 German or French negation is reported as "Affirmed" when the UI is in English.** *(verified)*
- Repro: UI en (the default), query `kein Krampfanfall`.
- Result: chip says Affirmed and + adds it as present. It says Negated only when the UI locale is `de` (cases L05 vs N02). The UI locale is pinned as the query language, which disables the API's auto-detection.
- Cause: `frontend/src/components/QueryInterface.vue:495` sets `selectedLanguage = currentLocale`, which is then sent as `language` and `query_assertion_language` at `frontend/src/composables/useQueryInterfaceController.js:262,265`.
- Fix: default `selectedLanguage` to `null` (Auto) and send `query_assertion_language` only when the user picked one explicitly. Also show the detected language in the response header, which already exists. Caveat: an API call with `language: null` also returned Affirmed, because auto-detection on a 2-word German query falls back to en. The detector should also try the other supported languages' negation cues when detection confidence is low.

**B-03 Reloading during an in-flight request leaves an eternal spinner.** *(verified, D06)*
- Repro: submit with a slow API and reload before the response arrives.
- Result: the persisted turn has `loading:true` and spins forever after reload ("[redacted]" bubble plus spinner).
- Cause: `frontend/src/stores/conversation.js:71-80` `sanitizeHydratedQueryHistory` does not reset `loading`.
- Fix: map `loading: false` and set an `error` such as "Request interrupted - resubmit" for any `loading && !response` item.

**B-04 A multi-line clinical note can't be typed: Shift+Enter submits.** *(verified, S05)*
- Repro: Full Text mode, type a line, press Shift+Enter.
- Result: the note is submitted and the textarea cleared.
- Cause: `frontend/src/components/query/QueryForm.vue:36` uses `@keydown.enter.prevent` on the textarea, which also catches Shift+Enter.
- Fix: on the textarea use `@keydown.enter.exact.prevent` (or submit only on Ctrl/Cmd+Enter) so Shift+Enter inserts a newline.

**B-05 The Case Workspace close button can't be clicked.** *(verified, C09)*
- Repro: add a term (the drawer auto-opens) and click the X at the top right.
- Result: the click lands on the fixed account icon instead. Esc does not close the drawer either. Only a scrim click works, which is not discoverable.
- Cause: `frontend/src/components/auth/AccountButton.vue:137-141` (`position: fixed; top: 8px; right: 12px; z-index: 2000`) overlaps the close button at `frontend/src/components/PhenotypeCollectionPanel.vue:47-53`.
- Fix: hide or offset the account button while a right drawer is open (or lower its z-index below the overlay layer), and make the drawer close on Esc.

### P2

**B-06 Clear Collection destroys curated work instantly.** No confirmation and no undo (C07).
- Code: `frontend/src/composables/usePhenotypeCollection.js:89-93`, button in `PhenotypeCollectionPanel.vue` (~270).
- Fix: add a confirm dialog like the existing reset dialog, or an undo snackbar.

**B-07 Requests have no client timeout or cancel.** A hung API leaves a spinner for more than 65 s with no way out (NW06).
- Code: `frontend/src/services/apiClient.js:28` calls `axios.create` without `timeout`.
- Fix: set a timeout (for example 30 s for query, longer for text/process) and add an AbortController-backed Cancel.

**B-08 The full-text LLM failure message is misleading.** *(S06)* This is how the UI communicates a missing LLM configuration locally.
- The API returns 503 with detail "Unable to evaluate LLM quota state. Verify PHENTRIEVE_LLM_QUOTA_DB_PATH...". The UI shows "The service is temporarily unavailable. Please try again later." That is wrong guidance: retrying never helps, and it doesn't suggest the working Standard extraction back-end.
- Code: `frontend/src/services/PhentrieveService.js:348-350` maps every 503 to `errors.api.serviceUnavailable`.
- Fix: surface `detail` (or a distinct `llm_unavailable` error code) and offer "Retry with Standard extraction".

**B-09 At 390px, 5 footer controls are off-screen and unreachable.** Docs, License, API, MCP and Version/API-status sit at x=387 to 572; the footer scrollWidth is 612 (M01).
- Code: `frontend/src/App.vue:11` `<v-footer class="d-flex justify-space-between">` with no wrap or overflow menu.
- Fix: `flex-wrap`, horizontal scroll, or collapse the secondary links into an overflow menu below `sm`.

**B-10 The workspace drawer covers the results' + buttons on desktop.** Collecting N terms requires N open/close cycles (C10).
- Code: `PhenotypeCollectionPanel.vue:34-36` (`location="right" width="400" temporary`). Combined with B-05, users get stuck.
- Fix: use a non-temporary (push/rail) drawer at >= lg, or stop auto-opening on add (`QueryInterface.vue` `handleAddToCollection` sets `showCollectionPanel = true`) and show a badge or snackbar instead.

**B-11 Language switch is incomplete.** After switching to Deutsch (I01):
- The query language stays `en`: `selectedLanguage` is only set in `created()` (`QueryInterface.vue:495`) and never watches `$i18n.locale`.
- `<html lang>` stays `en`.
- Labels are hard-coded in English: `QueryForm.vue:33,39,59` ("Paste or type a clinical note", "Clinical note", "Phenotype query") and `QueryModeControls.vue:15,27,48` ("Query", "Full Text", "Settings").
- Fix: watch the locale (or default to Auto per B-02), move the strings to locale files, and set `document.documentElement.lang` in `LanguageSwitcher.vue` and `i18n.js`.

**B-12 Focus is not returned after closing the Version dialog.** Focus lands on `<body>` (K03).
- Code: `frontend/src/App.vue:262`. The dialog is opened through `v-model` without an `activator`, so Vuetify can't restore focus.
- Fix: use the `activator` slot or prop, or focus the opener on `@after-leave`. The same pattern likely affects the other v-model dialogs.

**B-13 "seizures were ruled out" is not detected as negated (N04).** It shows Affirmed.
- Fix: add post-negation cues ("ruled out", "was excluded", "negative for") to the assertion detector (`phentrieve/text_processing/assertion_detection`).

### P3

**B-14 Unknown routes render a blank page** (D05). `frontend/src/router/index.js:7` has no catch-all. Add `{ path: '/:pathMatch(.*)*', redirect: '/' }` or a NotFound view.

**B-15 The deep link is re-submitted on every reload** (D07). `?text=` stays in the URL after auto-submit, which adds a duplicate turn. `useQueryInterfaceController.js:47` only strips `autoSubmit`, and only for manual submits. Fix: after an auto-submit, `router.replace` without `text`/`autoSubmit`.

**B-16 Unvalidated `forceEndpointMode` deep-link parameter** (D03). Any string is accepted (`useQueryInterfaceController.js:116-118`), leaving neither mode pill active. Fix: whitelist `query` and `textProcess`. Validate `chunkingStrategy` the same way.

**B-17 The previous answer jumps on each new query** (layout shift 0.18 from `scrollToTop`). Fix: insert new turns without forcing a scroll jump when the user has scrolled, or animate.

**B-18 No dark mode.** `frontend/src/App.vue:2` hard-codes `theme="light"` and `prefers-color-scheme: dark` is ignored (T01).

**B-19 Esc doesn't close dialogs:**
- Reset dialog (`HomeView.vue:34`, `persistent`) (K04).
- Guided tutorial (`TutorialOverlay.vue` has no Esc handler) (K06).

Fix: drop `persistent` on the non-legal reset dialog and add a keydown Esc handler, and Esc-to-skip on the tutorial.

**B-20 No skip link; the first Tab stop is the account button** (K01). Fix: add a "Skip to search" link and consider autofocusing the query input.

**B-21 Accessible name doesn't match the visible label** (A01, WCAG 2.5.3). `frontend/src/components/DisclaimerDialog.vue:59` has aria-label "Accept disclaimer and continue..." while the button reads "I understand and agree". Fix: drop the aria-label or include the visible text.

**Observation (not counted):** nonsense input (`xqzvbnm`) returns ten results at 0.47 to 0.58 ("X-linked inheritance", "Guilt") with no "no good match" hint. Consider a minimum-score empty state.

## 6. What passed (highlights)

- **Core search**: EN/DE/ES/FR lay terms all find the right top-5 concept, even with language pinned to en.
- **Input validation**: whitespace-only input is blocked.
- **Injection**: payloads in the query and in deep links render inert.
- **Long notes**: the 50k note auto-switches to Full Text with no UI freeze (fill 41 ms).
- **Concurrency**: rapid Enter, double-click and auto-submit-while-loading each produced exactly 1 request.
- **Navigation and persistence**: back/forward and reload persistence work.
- **Collection**: duplicate add is prevented; assertion toggle and remove work.
- **Export**: Phenopacket v2.0.0 export is valid with subject id and features; text export works.
- **Error handling**: API-down, 500, 422 and offline give readable messages with no stuck spinner, and retry works.
- **Other**: the HPO link uses `target=_blank rel=noopener noreferrer`; /faq and /connect render.

## 7. Limitations

- Full-text LLM extraction could not be exercised successfully because of the local quota-DB/LLM misconfiguration. Only the error path was tested.
- In the monkey run, `/text/*` and auth POSTs were stubbed.
- Timings are against a local API. Production latency will be dominated by the network and model.
- Screen-reader output was not tested; only focus and naming were checked programmatically.
