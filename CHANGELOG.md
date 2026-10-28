# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to pre-1.0 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
conventions: breaking changes bump the minor version during the 0.x series
(`major_on_zero = false` in `[tool.semantic_release]`).

Each release has three independent component versions which are all bumped
together:

- `phentrieve` — Python CLI/library (root `pyproject.toml`)
- `phentrieve-api` — FastAPI backend (`api/pyproject.toml`)
- `phentrieve-frontend` — Vue.js frontend (`frontend/package.json`)

---

## [Unreleased]

## [0.30.0] — 2026-10-28 (CLI 0.30.0 / API 0.20.0 / Frontend 0.20.0)

LLM phase-1 axes contract (prompt v3.3.0: negation scope, uncertain, other-person
findings, temporality), evidence offsets computed in code, bounded and capped phase-1
retries, REST LLM-error fallback, non-blocking `/query`, bundle integrity checks, and a
batch of API, retrieval and frontend fixes from the 2026-10-27 review.

Behaviour changes: `GET /api/v1/health/` returns `"status": "healthy"` (was `"ok"`);
Chroma query failures raise instead of returning empty results;
`PROMPT_BEHAVIOR_VERSION` is `5`, so earlier LLM benchmark checkpoints do not match.

### Changed

- **The API LLM text path reuses the cached dense retriever** (#420) and its
  model for semantic chunking, instead of opening a new Chroma collection per
  request. `run_llm_backend` now forwards a supplied `retriever` to the
  pipeline's tool executor.
- **Removed the text router's re-exported private service aliases** and
  `_process_text_internal` (#428); tests patch `api.services.text_processing_*`.
- **Query schema uses the shared strategy types** (#427). `QueryRequest`
  validates `aggregation_strategy` against the `AggregationStrategy` enum and
  `query_assertion_preference` against one shared `AssertionPreference` alias
  (`phentrieve.config`); the router casts and hand-copied `Literal`s are gone
  and `benchmark compare-vectors` derives its valid strategies from the enum.
  The published request JSON schema and accepted values are unchanged.
- **API warms the default Chroma index at startup** (#419). The lifespan runs
  one throwaway query after preloading the model and retriever, so the first
  real request no longer pays for loading the index. A failed warm-up is
  logged and startup continues.
- **`QueryInterface.vue` uses `<script setup>`** (#430). The
  `getContext()`/`getState()`/`setState()` bridge is gone:
  `useQueryInterfaceController` now takes the component's refs, the
  conversation store, the PII review flow and the route/router directly, and has
  its own unit tests with plain refs. Behaviour is unchanged.
- **One term-similarity implementation for REST and MCP**
  ([#415](https://github.com/berntpopp/phentrieve/issues/415)). The similarity
  router and the MCP `phentrieve_compare_hpo_terms` tool both call
  `phentrieve.evaluation.term_similarity.compute_term_similarity` and only
  translate its errors. Response shapes are unchanged. Differences: labels in
  both now come from the same `hpo_data.db` as the graph data; MCP reports
  missing ontology data as `temporarily_unavailable` instead of `not_found`;
  the REST 404/400/500 error details now serialize (`formula_used` was an enum
  and failed JSON encoding).

- **Phase-1 prompt v3.3.0 with the axes contract** (#379, #375). The prompt
  decides experiencer, assertion and temporality in independent steps: a
  negation cue applies to every item of the coordinated list it introduces (also
  across chunk lines), `uncertain` is defined with multilingual cues, absent and
  uncertain findings are extracted as exhaustively as present ones, and
  non-phenotype information (onset, inheritance, age, measurement) always gets
  category Other. Findings of another person who is not a relative
  (experiencer `other`) are returned in a new `other_person_findings` list on
  the pipeline result, REST response and MCP extract tools, and are never
  exported into a phenopacket. Every term gets a `temporality`
  (`current` | `historical` | `unknown`); unexpected values are coerced to
  `unknown`, and an `unknown` twin of a term with a known temporality is merged
  into it. The phase-1 schema no longer asks for character offsets.
  `PROMPT_BEHAVIOR_VERSION` is bumped to `4`.
- **One helper for located chunks and the gaps between them** (#401). The
  "chunk has a document position" check and the walk over the text cleanup
  dropped between chunks were written several times (evidence offsets, full-text
  adapter, phase-1 prompt rendering, pipeline offset attachment). They now live
  in `phentrieve.llm.evidence_offsets`, and a rendered chunk's evidence search
  region is exactly the source text its phase-1 line shows (before, it extended
  over the whole gap to the next chunk).
- **One `needs_semantic_model()` check for chunking pipelines** (#429). The
  "does this chunking pipeline need a sentence-transformer model" loop that was
  copy-pasted in `text process`, `text chunk` and `text interactive` now lives
  in `phentrieve.text_processing.config_resolver.needs_semantic_model`.
- **One health route: `GET /api/v1/health`** (#433). It now answers directly
  (the Docker health checks used to get a 307 to `/api/v1/health/`) and
  returns `{"status": "healthy", "service", "version", "timestamp"}`, the shape
  `/api/v1/system/health` already returned; the old body was
  `{"status": "ok"}`. `/api/v1/health/` and `/api/v1/system/health` stay as
  undocumented aliases with the same payload, and the frontend connection
  poll now calls `/api/v1/health`.
- **Batched embedding and spaCy calls in standard text processing** (#410).
  The sliding-window splitter encodes the windows of all segments in one
  `model.encode` call instead of one call per segment, and assertion detection
  parses all chunks of a document with one `nlp.pipe` call
  (`AssertionDetector.detect_batch`, `DependencyAssertionDetector.detect_doc`)
  instead of one `nlp()` call per chunk. Output is unchanged.

### Fixed

- **`/api/v1/query` no longer blocks the event loop** (#405).
  `execute_hpo_retrieval_for_api` is now synchronous and both the REST router
  and the MCP `search_hpo_terms` adapter run it via `run_in_threadpool`, so one
  query no longer stalls health checks, MCP and text processing.
- **`GET /api/v1/query` returns 422 for invalid parameters and uses the
  requested model's retriever** (#408). The retriever dependency binds
  `model_name` directly (no more un-awaited coroutine or stray `?model=`
  parameter), `QueryRequest` validation errors become 422 instead of 500, and
  the retriever-mismatch re-initialization and second model-policy resolution
  are removed.
- **API model caches no longer expire hourly or race** (#416). The SBERT model,
  retriever, status and lock caches are plain dicts (bounded by the model
  allowlist) read once with `.get()`, a failed model load is retried after a
  30 s cooldown instead of sticking for an hour, and `cleanup_model_caches`
  also clears the embedding model registry so models are actually freed.
- **Other-person findings get correct highlight offsets** (#394). The full-text
  adapter mapped their document spans without the document text, so offsets
  drifted by the whitespace that chunk cleanup collapsed.
- **Full-text attribution offsets on non-ASCII text** (#411). The fallback
  search for evidence inside a chunk ran on the lowered chunk text, whose length
  can differ ("İ" lowers to two code points), shifting the offsets. It now
  searches the original text case-insensitively. (The other points of #411 do
  not apply on this branch: attribution offsets are computed in code and
  clipped to the chunk, not taken from the model.)
- **LLM evidence offsets** (#397):
  - A `chunk_span` fallback (evidence not found in the cited chunks) is no
    longer shown as an exact highlight of the whole chunk. The adapter searches
    the cited chunks for the evidence text and adds no attribution when it is
    absent, as before the evidence-offset rework; the frontend then falls back
    to the label. The evidence record keeps `offset_basis="chunk_span"`.
  - Soft hyphens (U+00AD) and words hyphenated at a line break
    (`"Hämo-\nglobin"`), common in text pasted from PDFs, no longer prevent a
    match.
  - A chunk that follows a chunk the text pipeline could not locate no longer
    repeats that chunk's text in its phase-1 line, so the model cannot cite it
    for the unlocated finding. This changes the phase-1 prompt content for such
    notes; `PROMPT_BEHAVIOR_VERSION` is bumped to `5`.
  - Each search region is normalized once per document instead of once per
    finding.
- **Phenopacket export honours every negation synonym and `excluded: true`**
  (#407). `denied`, `no`, `negative` and `excluded` now export as excluded
  features (they previously normalized to `unknown`, i.e. present); an explicit
  `excluded: true` wins over any `status`/`assertion` string; and the MCP
  export reads the extractor's `excluded` flag.

- **MCP response budget keeps the patient's findings** (#396). The extract
  tools now trim `other_person_findings` first, then `family_history_findings`,
  and `aggregated_hpo_terms` only if still over budget, so a large secondary
  list no longer pushes the patient's own terms out of the response.
- **Frontend health poll and response logging are cheaper** (#431). The API
  health poll skips its interval tick while the tab is hidden, and API response
  logs read the size from `Content-Length` instead of `JSON.stringify`-ing every
  response body.

- **ESLint no longer lints the generated `frontend/coverage/` directory** (#432).

- **Full-text "normal" findings are no longer collected as present** (#404).
  The frontend mapped a `status: "normal"` term (and any term the API flags
  `excluded: true`) to `affirmed`, so "add to collection" exported ruled-out
  findings as present phenotypes. They now map to `negated` when the curation
  model is seeded and in the findings receipt.
- **Long notes no longer silently return zero HPO terms**
  ([#402](https://github.com/berntpopp/phentrieve/issues/402)).
  `DenseRetriever.query_batch` sent every chunk embedding to Chroma in one
  query, which exceeded SQLite's bound-variable limit ("too many SQL
  variables") for notes of a few hundred chunks; the error was logged and every
  chunk got empty results, so the API answered HTTP 200 with no terms. Chroma is
  now queried in slices of `CHROMA_QUERY_BATCH_SIZE` (64) embeddings. A failed
  index query raises `RetrievalError` instead of returning empty results
  (`query_batch_multi_vector` no longer swallows errors either, and an invalid
  aggregation strategy raises `ValueError`), so API, CLI and MCP callers report
  an error.

- **Bundle verification no longer accepts a corrupted SQLite index**
  ([#398](https://github.com/berntpopp/phentrieve/issues/398)). The
  lock-row-tolerant "stable" `indexes/` checksum hashes the SQLite dump, which
  never reads index pages. It is now accepted only if `PRAGMA integrity_check`
  reports `ok`. The verified inventory (and so the benchmark retrieval-asset
  `content_sha256`) records the checksum computed from the data rather than the
  manifest hash: for a directory holding `chroma.sqlite3` this is its stable
  checksum, which does not change when Chroma adds lock rows. Benchmark
  checkpoints written before this change therefore do not resume.

- **Similarity recovers once HPO data is prepared**
  ([#417](https://github.com/berntpopp/phentrieve/issues/417)). A missing or
  empty HPO database was cached as empty graph data (and the REST router cached
  an empty label map), so similarity answered 503 until the process restarted.
  Failed graph loads are no longer cached and labels are looked up per request.

- **LLM evidence offsets are computed in code from the cited chunks.** The
  model reported `start_char`/`end_char` relative to a chunk, so evidence spans
  did not point into the document. The pipeline now locates `evidence_text`
  inside the cited chunks (tolerating typography, whitespace, letter case and
  Unicode composition) and records how the span was found in a new
  `offset_basis` (`evidence_in_chunk`, `evidence_in_chunk_hull`, `chunk_span`,
  `unresolved`). A hull search only spans runs of adjacent cited chunks, and
  chunks the text pipeline could not locate (`-1` positions) are ignored. The
  full-text adapter maps the document offsets into each chunk's cleaned text for
  `text_attributions`. Paraphrased evidence now gets a whole-chunk attribution
  (`chunk_span`) where it previously had none.

- **Grounded phase-1 extraction sees negation cues that chunk cleanup removed**
  ([#375](https://github.com/berntpopp/phentrieve/issues/375)). `FinalChunkCleaner`
  strips leading words such as `no`/`not`, and assertion detection is disabled
  for grounded chunks, so the extraction prompt showed e.g. `hepatosplenomegaly`
  for "There was no hepatosplenomegaly" and the model extracted it as present.
  Grounded chunks now carry a `prompt_text` that prefixes the source text dropped
  since the previous chunk ("There was no hepatosplenomegaly"). It is used only
  in the phase-1 chunk index (ungrouped, precomputed and fallback groups), with
  whitespace runs and line breaks in the restored text collapsed so each chunk
  stays on one index line. Chunk `text` and offsets, retrieval, phase-2 mapping
  context and API output are unchanged. `PROMPT_BEHAVIOR_VERSION` is bumped to
  `3` so earlier LLM benchmark checkpoints do not match new runs. Limitation: a
  cue that scopes over a coordinated list is restored only for the chunk it
  directly precedes ("no plantar or palmar pits, facial abnormalities or
  macrocephaly" restores `no` for `plantar` but not for the later items).

- **Runaway phase-1 generations no longer burn the full output budget repeatedly**
  ([#375](https://github.com/berntpopp/phentrieve/issues/375)). A degenerate
  response (e.g. a repeated `\n` escape inside `evidence_text`) previously ran to
  the fixed 65,536-token ceiling, was retried identically and then re-sent as a
  single large group, costing four maximal requests before small groups ran.
  Grounded phase-1 requests now get a chunk-proportional output budget
  (`max(8192, 1024 + 256 × chunks)`, capped at 65,536). Structured retries are
  skipped when a response was truncated at a budget the retry cannot raise.
  Grounded phase-1 always passes its budget explicitly, so the budget is also
  the retry ceiling: a truncated grounded phase-1 response is never retried with
  a larger budget (it was not before either, at 65,536) and recovery comes from
  the grouped fallback. After such a truncation, `grouped_large` is skipped when
  it would resend all chunks as one group, and extraction falls back directly to
  `grouped_small`. Reasoning models count hidden reasoning tokens against the
  same budget; the 8,192-token floor was derived from a non-reasoning model.
- **Grounded Phase-1 prompt no longer sends an empty clinical-text block.** In
  grounded and grouped extraction the chunk index is the note, but the user
  prompt still rendered an empty `UNTRUSTED_CLINICAL_TEXT` block while the system
  prompt asked for extraction "from the full note". Grounded requests now use a
  dedicated chunk-index-only `grounded_user_prompt_template`, and the system
  prompt names both note sources: the chunk index, or the clinical text block
  when the chunk index is empty (legacy whole-document mode), with chunk_ids
  required only when the index is not empty (prompt `v3.1.0` → `v3.2.0`).
  Custom templates without the new key keep the previous rendering. The prompt
  identity hashes the grounded template, and `PROMPT_BEHAVIOR_VERSION` is bumped
  to `2`, so earlier LLM benchmark checkpoints no longer match new runs.
- **Other-person findings are fully accounted for** (#401). The LLM benchmark
  assertion distribution counts `other_person_findings` next to the family
  count, and the other-person resolution path records phase timings, the
  phase-2b LLM trace, `local_fallbacks` and `llm_mapped_phrases` under
  `other_person_*` keys through the same helper as the family path. The dead
  span-overlap guard and bounds merge in phase-1 deduplication are removed
  (phase-1 items carry no offsets until after deduplication).
- **A truncated phase-1 group is no longer dropped silently** (#395). A
  `grouped_small` group (the last fallback level) that truncates at its scaled
  output budget is retried once at a fixed 65,536-token ceiling
  (`DEFAULT_GROUNDED_PHASE1_TRUNCATION_RETRY_OUTPUT_TOKENS`); first attempts
  keep the scaled budget, so runaway generations are still cut early. At most
  two such retries run per request
  (`DEFAULT_GROUNDED_PHASE1_MAX_TRUNCATION_RETRIES`), so a model that runs away
  in every group of a long note cannot cost one ceiling call per group. A group
  that still fails is counted in `phase1_unrecovered_groups`, and REST
  responses (`meta`) and the MCP `phentrieve_extract_hpo_terms_llm` tool
  (`_meta`) then carry `incomplete: true` with a short `incomplete_reason`.
  Retries are traced per group (`truncation_retry`) and counted in
  `phase1_truncation_retries`. The frontend does not display the flag yet.
- **REST `allow_standard_fallback` now also covers LLM backend errors** (#418),
  as on MCP: the request reruns on the standard backend with
  `meta.fallback_reason = "llm_backend_error"` instead of returning a 500, and
  the failed LLM attempt's quota reservation is released.
### Removed

- **`scripts/validate_config_sync.py` and `make config-validate`** (#426). The
  script still parsed `QueryInterface.vue` after the defaults moved to
  `frontend/src/constants/defaults.js`, so it always failed;
  `tests/unit/profiles/test_frontend_constant_parity.py` covers the same parity.
  `AdvancedOptionsPanel.vue` now takes its prop defaults from
  `constants/defaults.js`, fixing its `similarityThreshold` default (0.5 vs 0.3).

- **Unused evaluation code** (#414). Deleted the unreferenced
  `phentrieve.evaluation` modules `assertion_metrics`, `document_aggregation`,
  `hierarchical_matching`, `threshold_optimization`, `full_text_runner` and
  `semantic_metrics` (with its tests), the unused `result_analyzer` dataframe
  helpers, the `ExtractionReporter.plot_*` methods, and the dead helpers
  `utils.get_index_dir`, `utils.load_german_translation_text`,
  `config.get_sliding_window_config`,
  `evaluation.runner.compare_models_with_statistics`,
  `query_orchestrator._format_structured_results_to_text_display`,
  `llm.types.LLMTermsResult` and `details_enrichment.DatabaseNotFoundError`.

### Deprecated

- **No-op `phentrieve text process` options** (#414).
  `--cross-language-hpo-retrieval`/`--xlhpo`, `--use-cached-index` and
  `--chunk-confidence`/`-cc` never had an effect. They are now hidden from
  `--help` and print a deprecation warning when passed; they will be removed in
  a future release.

## [0.29.1] — 2026-10-28 (CLI 0.29.1 / API 0.19.0 / Frontend 0.19.1)

Case workspace labels follow the UI language.

### Fixed

- **The case workspace drawer's title and labels are translated through the
  locale files.** They came from a hard-coded per-language table (German spelled
  without umlauts, e.g. "oeffnen") that overrode the "HPO collection" strings.
  "Case workspace" now names the drawer in all five languages; "collection"
  stays for the phenotype list inside it.

## [0.29.0] — 2026-10-28 (CLI 0.29.0 / API 0.19.0 / Frontend 0.19.0)

UI/UX review: correct negated queries, cancellable requests, dark mode, a
faster frontend and research-only copy in all five languages.

### Fixed

- **Negated queries find the right term.** "no seizures" embedded the cue and
  ranked absence seizure first, with Seizure outside the top 10. Negated queries
  now retrieve on the negated concept and stay marked negated (API and CLI).
- **German negation in an English UI** was read as affirmed, because the UI
  language was sent as the query language. Queries now default to auto-detect;
  for short queries, assertion detection tries all five languages. Hyphenated
  compounds ("non-progressive ataxia", "no-reflow") are no longer negations.
- **Negated (amber) and manually edited (violet) note highlights** never
  rendered: a base `mark` selector outranked them.
- **Search spinner** was dimmed to 26% and invisible in dark mode.
- Interrupted requests restored after a reload no longer spin forever; failed
  full-text submits restore the note; unknown routes redirect home;
  `<html lang>` follows the UI language; accessibility fixes (tooltip labels,
  FAQ ARIA roles, WCAG AA status colours).

### Added

- **Dark mode** following the system setting, with a light/dark/system toggle
  next to sign-in.
- **Cancel and timeouts:** requests time out (60 s query, 120 s full text) and
  can be cancelled. The API stops starting paid LLM calls when the client
  disconnects; a cancel before any LLM call is refunded, after one it counts
  like a timeout.
- "Add all" asks for a confirming click; "Copy present HPO IDs"; two-step clear;
  logo tooltip for starting over.
- The LLM full-text path auto-detects the text language instead of assuming
  English.

### Changed

- **Frontend icons are inline SVG** (`@mdi/js`) instead of the icon font: page
  weight 750 -> 364 KiB, mobile Lighthouse performance 68 -> 84.
- **Research-only wording in every language**, including the page title and
  search description; no diagnostic claims. English is sentence case with
  actionable error messages; de/es/fr/nl were reviewed against the official HPO
  translations and caught up with the research-use rewrite (#245).

## [0.28.7] — 2026-10-27 (CLI 0.28.7 / API 0.18.7 / Frontend 0.18.4)

Smaller API image: CPU-only torch (#446).

### Changed

- **The API image ships CPU-only torch: 6.9 GB -> 2.8 GB.** The deployment has
  no GPU, but the image installed the locked CUDA build of torch plus triton and
  15 `nvidia-*` wheels (~4.6 GB). The build now installs the CPU build of the
  same locked torch version (2.13.0+cpu). `pyproject.toml` and `uv.lock` are
  unchanged, so local and benchmark installs keep CUDA.
- **CI:** the API image smoke test fails if CUDA wheels or non-CPU torch reach
  the image. The PR image job no longer frees runner disk space, which took
  4.5 minutes of a 12-minute job.

## [0.28.6] — 2026-10-27 (CLI 0.28.6 / API 0.18.6 / Frontend 0.18.4)

Dependency security patch and CI follow-ups to 0.28.5 (#436).

### Fixed

- **dnspython 2.9.0** (CVE-2026-114605), pulled in via email-validator. The
  advisory was published after 0.28.5 and failed the Security Scanning workflow
  on `main`. The floor is enforced by the dependency policy test.

### Changed

- **CI runs on every PR**, including docs-only ones. The `main` ruleset now
  requires the "CI Summary" check, and the `pull_request` `paths-ignore` meant
  docs-only PRs never reported it. `Detect Changes` still skips the heavy jobs.
- The frontend image is rebuilt and scanned when
  `frontend/nginx-security-headers.conf` changes.

## [0.28.5] — 2026-10-27 (CLI 0.28.5 / API 0.18.5 / Frontend 0.18.4)

Security hardening and regression fixes from the v0.28.4 code review.

### Fixed

- **API image ships the `phentrieve` CLI again** (#403): v0.28.4 installed only
  the dependencies, so `phentrieve data prepare` / `index build` failed inside
  the container and `python -m phentrieve` raised `PackageNotFoundError`.
- **Auth session handling:** re-registering a pending account replaces its
  password and revokes its sessions; email verification revokes earlier
  sessions; refresh-token rotation is atomic and a reused token revokes all of
  the user's sessions (30 s grace for concurrent tabs); access tokens issued
  before a password change are rejected; an expired lockout no longer re-locks
  on the next typo.
- **Login responses are uniform:** unknown, locked and wrong-password logins
  all run bcrypt and return the same 401 (locked accounts previously got 429).
- **Passwords over 72 bytes** are rejected with a 422 instead of a 500, and no
  longer consume the reset token (#412).
- **LLM quota** is reserved before the LLM call and released on failure, so
  concurrent requests can no longer exceed the daily limit. MCP callers are
  counted per client instead of sharing one global bucket, and MCP honours
  `PHENTRIEVE_LLM_QUOTA_ENFORCE`. Quota helpers are shared (#434).
- **nginx:** security headers (CSP, framing, nosniff) now reach `index.html`
  and every location; the real client IP is used behind the reverse proxy;
  credential and email endpoints are rate-limited per client.
- **Dependency advisories:** authlib 1.9.x (CVE-2026-96760), msgpack 1.2.4
  (CVE-2026-114613); stale torch ignore removed (#413).

### Added

- **Input limits:** 50,000 characters for full text and MCP text arguments,
  10,000 for queries; the frontend explains a 422 (#406).
- **`PHENTRIEVE_MAX_CONCURRENT_INFERENCE`** (default 2) caps concurrent
  full-text inference across REST and MCP (#406).
- **`PHENTRIEVE_LLM_QUOTA_PEPPER`**: HMAC key for quota subject hashes (falls
  back to `PHENTRIEVE_AUTH_JWT_SECRET`, then a per-process key).
- **`PHENTRIEVE_AUTH_REFRESH_REUSE_GRACE_SECONDS`** (default 30).
- **CI smoke-runs the API image** (#409); the full Python matrix runs on
  dependency changes (#424); tests for the MCP validation unwrap (#423).

### Changed

- **Production with auth enabled refuses to start** unless
  `PHENTRIEVE_AUTH_JWT_SECRET` has at least 32 characters and
  `PHENTRIEVE_EMAIL_BACKEND=smtp`.
- **Quota counters reset once** on upgrade because subject hashes changed.
- `langdetect` and `de_core_news_sm` are locked in the `api` extra (#421);
  `make security-python` mirrors the CI pip-audit invocation (#413).
- The missing-bundle-manifest error says how to recover (#422); phase-1 LLM
  traces keep the underlying exception type (#425).

## [0.28.4] — 2026-10-27 (CLI 0.28.4 / API 0.18.4 / Frontend 0.18.3)

Security patch release: clears every open Dependabot and Trivy alert, makes the
API image reproducible from `uv.lock`, and hardens the Docker and CI/CD supply
chain (#392). No functional changes.

### Fixed

- **Python dependency advisories (~60 Dependabot alerts):** pillow 13.0.0
  (GHSA-5x8w-8jj5-q52h, GHSA-w889-685w-c8fh and ~40 more), virtualenv 21.15.0,
  filelock 4.2.4, fsspec 2026.9.0 (GHSA-27vj-qcqg-25rc), opentelemetry-api
  1.45.1, certifi 2026.10.23 (GHSA-6q48-6c7w-9gw2), setuptools 84.2.1
  (GHSA-xmj3-9gf7-h52w), fastmcp 3.4.10 (GHSA-7xgj-8392-5q4r) and mcp 1.30.0
  (GHSA-84m7-p3x7-pcfv). Floors are pinned in `tool.uv.constraint-dependencies`
  and the `mcp` extra, and enforced by policy tests.
- **Frontend advisories:** vite 8.3.5, vitest 5.0.5, happy-dom 21.0.4
  (GHSA-834v-8h66-3h79), follow-redirects 1.16.2; `npm audit` reports 0.
- **API image OS advisories (40 Trivy alerts):** the runtime layer now
  force-upgrades `openssl-provider-legacy` and `libpcre2-8-0` alongside OpenSSL.
- **MCP argument validation:** FastMCP 3.4.10 wraps pydantic `ValidationError`;
  the middleware unwraps it so bad arguments still return the
  `validation_failed` envelope.

### Changed

- **API image installs from `uv.lock`** (`uv sync --frozen`, hash-verified)
  instead of an ad-hoc `pip install`, so it ships exactly the versions CI tests.
- **Base images pinned by digest:** Python 3.11.18, Node 24 LTS, nginx 1.30
  stable on Alpine 3.24 (Node 20 and Alpine 3.20 are end-of-life).
- **Docker build context is an allowlist**; untracked env files and virtualenvs
  can no longer reach an image layer.
- **CI/CD:** actions pinned to commit SHAs; read-only default token
  permissions; draft PRs skipped. Lint and typecheck gate the tests. PRs test
  the Python 3.11 floor, and the full 3.11-3.13 matrix runs weekly. Image build
  and Trivy scan (blocking on fixable HIGH/CRITICAL) run on PRs after tests.
  Faster security scans; injection-safe data-bundle workflow.
- `anthropic` optional dependency range widened to `<3.0.0`.

## [0.28.3] — 2026-10-01 (CLI 0.28.3 / API 0.18.3 / Frontend 0.18.2)

Consolidated Dependabot version updates and security fixes. No functional
changes.

### Fixed

- **Security vulnerabilities in Python dependencies (Dependabot alerts #142-#163):**
  - Raised the direct `urllib3` floor from 2.7.0 to `>=2.8.0`, resolving
    GHSA-8988-9cw3-xx77, GHSA-vxq7-64xx-v4gw and GHSA-gh4c-6fx4-qh6g.
  - Upgraded transitive `pyjwt` 2.13.0 -> 2.15.1 (via `mcp[crypto]`), resolving
    GHSA-ffc3-869f-jxw9 (critical), GHSA-42vr-xj54-vc7v, GHSA-gvp8-978c-rx2q and
    the remaining 2.14.0-fixed advisories.
  - Upgraded transitive `oauthlib` 3.3.1 -> 4.0.0 (via `chromadb` ->
    `kubernetes`), resolving GHSA-xpv3-w29h-x7cv and GHSA-hj66-6f7g-4r5v.
  - Upgraded transitive dev-only `virtualenv` 21.3.3 -> 21.14.2 (via
    `pre-commit`), resolving GHSA-p58f-9548-mpm2, GHSA-x78j-v8h9-3j2q,
    GHSA-94p9-xgh2-xp45 and GHSA-9h9j-4vrj-gf7g.
  - Pinned `pyjwt>=2.15.0`, `oauthlib>=4.0.0` and `virtualenv>=21.7.13` in
    `tool.uv.constraint-dependencies` and enforced all four floors in
    `tests/unit/test_dependency_security_policy.py`.
- **Security vulnerabilities in frontend dev dependencies (Dependabot alerts
  #139-#141):** upgraded transitive `brace-expansion` to 1.1.21, 2.1.7 and
  5.0.12, resolving GHSA-q2hr-2g5m-vwhr.

### Changed

- **Frontend dependencies:** `vue` 3.5.43 (#362), `vite` 8.3.1 (#364),
  `vuetify` 3.13.5 (#365), and the minor/patch group (#366):
  `google-protobuf` (^4.0.3), `libphonenumber-js` (^1.13.14), `vue-i18n`
  (^11.4.12), `@vitejs/plugin-vue` (^6.0.9), `@vue/test-utils` (^2.5.1),
  `eslint` (^10.11.0), `eslint-plugin-vue` (^10.11.1), `prettier` (^3.9.9).

## [0.28.2] — 2026-09-18 (CLI 0.28.2 / API 0.18.2 / Frontend 0.18.1)

Consolidated dependency and security updates across frontend, backend, and
GitHub Actions tooling.

### Fixed

- **Vite 8.3 native config loader warnings:** Added explicit `.js` extensions to
  local config imports in `frontend/vite.config.js` and `frontend/vite-icon-optimizer.js`.
- **Security vulnerabilities in Python dependencies:**
  - Bumped `mkdocs-material` from 9.7.6 to >=9.7.7, resolving CVE-2026-73295
    (Dependabot alert #132).
  - Pinned constraint `anyio>=4.14.2` in `tool.uv.constraint-dependencies` and
    enforced via `tests/unit/test_dependency_security_policy.py`, resolving
    CVE-2026-63374 / CVE-2026-64847.
  - Added `types-cachetools>=5.5.0` to dev dependencies to satisfy type checking.

### Changed

- **Frontend & Dev Dependencies:**
  - Upgraded `vitest`, `@vitest/coverage-v8`, and `@vitest/ui` to 5.0.1 in tandem,
    resolving peer dependency deadlocks across split Dependabot PRs (#351, #352, #353).
  - Configured `vitest-updates` group in `.github/dependabot.yml` to bundle future
    Vitest ecosystem updates into a single PR.
  - Upgraded `vite` to 8.3.0 (#356) and `vuetify` to 3.13.4 (#357).
  - Upgraded minor/patch frontend dependencies (#358): `libphonenumber-js` (^1.13.13),
    `vue-router` (^5.3.1), `autoprefixer` (^10.5.6), `eslint` (^10.10.0),
    `eslint-plugin-vue` (^10.11.0), `happy-dom` (^20.14.3), `postcss` (^8.5.28).
- **Backend & CI Tooling:**
  - Expanded `fastmcp` requirement from `<4.0.0,>=3.2.0` to `>=3.2.0,<5.0.0` (#354).
  - Updated `hadolint/hadolint-action` from 3.4.0 to 3.5.0 in `.github/workflows/docker-publish.yml` (#355).

## [0.28.1] — 2026-09-02 (CLI 0.28.1 / API 0.18.1 / Frontend 0.18.0)

Security patch closing the last open Trivy code-scanning alert on the published
API image. The frontend is unchanged.

### Fixed

- **setuptools floors pinned to the CVE-2026-59890 patched release (83.0.0).**
  The published API image kept shipping setuptools 82.0.1 even when rebuilt,
  because the `api/Dockerfile` `pip install --upgrade "setuptools>=78.1.1"`
  floor was non-binding. It resolves to whatever release is newest when
  BuildKit materialises that layer, and because the `RUN` string never changed,
  the registry build cache replayed the layer verbatim and froze the original
  resolution. The `pyproject.toml` `[build-system]` floor was ruled out as the
  source: `pip install .` resolves it inside pip's isolated build environment
  and never touches `/opt/venv`. CVE-2026-59890, PYSEC-2026-3447 and
  GHSA-h35f-9h28-mq5c are aliases of a single advisory, so the dev-group floor
  was already correct. All three floors now sit at `83.0.0`.

### Added

- **Floor-alignment policy test** in
  `tests/unit/test_dependency_security_policy.py` ties the build-system floor
  and the `api/Dockerfile` floor to the dev-group floor. They cannot drift
  apart again, and any future bump must edit the Dockerfile line, which busts
  the cached layer.
- **Container base image audit** recorded in
  `.planning/analysis/2026-09-02-setuptools-floor-drift-and-base-image-audit.md`.
  No base image was bumped. The audit notes that Node 20 (end-of-life
  2026-04-30) and Alpine 3.20 (end-of-life 2026-04-01) both back the frontend
  build, that bumping Node would not change the scanner result because both
  tags carry identical npm-bundled findings, and that both published images
  currently scan clean because the Dockerfiles patch at build time.

## [0.28.0] — 2026-09-02 (CLI 0.28.0 / API 0.18.0 / Frontend 0.18.0)

Dependency consolidation, the Transformers 5.x migration, and reproducible
benchmark identities. Removing the JinaBert model unblocked the patched
Transformers major and, in the process, surfaced a silent embedding-corruption
bug in the GTE model. Ships alongside HPO data release `hpo-v2026-06-23-r2`.

### Added

- **Reproducible LLM benchmark identities and manifest schema v2.** LLM runs now
  record separate source, semantic input, gold-label, assertion-projection,
  prompt-bundle, resolved model/provider, retrieval-asset, execution, and
  scoring identities. Schema-v2 manifests inventory every owned file with its
  SHA-256 digest while retaining direct schema-v1 role aliases.
- **`--llm-seed` and `--evaluation-hpo-version` benchmark declarations.** The
  resolved seed is bound to execution and the expected evaluation HPO version
  must match the installed retrieval bundle.

### Changed

- **LLM checkpoint reuse now fails closed.** A checkpoint must match both the
  execution/scoring fingerprints and the complete output-affecting benchmark
  configuration, including the normalized runtime-package source digest.
  Package version, Git commit, and dirty state remain descriptive provenance.
  Missing, non-object, and pre-identity checkpoints are not resumed implicitly.
- **LLM overwrite validation is non-destructive.** Existing artifacts are kept
  until the preserved checkpoint passes compatibility checks; incompatible
  runs must use a new run id or be removed deliberately before a fresh run.
- **Benchmark identity contracts are runtime-faithful.** Assertion aliases are
  canonicalized, scoring and producer-source contracts are versioned, provider
  endpoints exclude credentials, and persisted provider failures use stable
  public error codes instead of raw exception text.
- **Retrieval assets are verified before LLM execution.** Required database and
  index checksums, safe relative paths, and bundle contents are validated; the
  manifest's pinned model revision, trust/code revision, vector mode, and index
  path are passed directly into the runtime.
- **LLM artifacts publish as immutable generations.** The root manifest switches
  atomically only after a complete hashed generation exists. Discovery rejects
  malformed, escaping, linked, or checksum-mismatched artifacts, and resume
  prefers the checkpoint referenced by the committed manifest. Publication
  retains the active and one rollback generation, pruning older snapshots.
- Dependency ceilings widened to match upstream majors: `sentence-transformers<7.0.0`,
  `anthropic<2.0.0`, `openai<4.0.0`, `mcp<3.0.0`.
- Frontend dependencies updated, including `pinia` 3 -> 4, `vite` 8.1.4 -> 8.2.2,
  `vuetify` 3.12.9 -> 3.13.2, `vue` 3.5.39 -> 3.5.42, and `eslint` 10.7 -> 10.9.
- GitHub Actions bumped: `actions/setup-python` v6 -> v7, `actions/setup-node`
  v6 -> v7, `hadolint/hadolint-action` v3.3.0 -> v3.4.0.

### Removed

- **`jinaai/jina-embeddings-v2-base-de` is no longer a supported embedding model.**
  Its pinned remote implementation imports
  `transformers.pytorch_utils.find_pruneable_heads_and_indices`, a symbol removed in
  Transformers 5.x, so the model could not load on any release carrying the fixes for
  GHSA-29pf-2h5f-8g72 and GHSA-fgcw-684q-jj6r. It was never the default, ranked below
  the default BioLORD model on the German benchmark (HR@1 0.222 vs 0.333), and was the
  only remaining blocker for the patched Transformers major. The data-release contract
  now declares eight models; `Alibaba-NLP/gte-multilingual-base` remains the only
  `trust_remote_code` model and is fully compatible with Transformers 5.x. Existing
  bundles are unaffected, but new HPO data releases no longer ship a `jina-de` index.
  German text is best served by the default BioLORD model or
  `T-Systems-onsite/cross-en-de-roberta-sentence-transformer`.

### Security

- **Bump `mcp[cli]` floor to `>=1.28.1`.** Resolves `pip-audit` findings
  CVE-2026-52869 and CVE-2026-52870 (fixed in 1.27.2) and CVE-2026-59950
  (fixed in 1.28.1) in the pinned `mcp` dependency.
- **Transformers upgraded to the patched 5.x major** (4.57.6 -> 5.16.1), resolving
  GHSA-29pf-2h5f-8g72, GHSA-fgcw-684q-jj6r, GHSA-69w3-r845-3855, and the matching
  PYSEC advisories. The four `pip-audit --ignore-vuln` suppressions and two
  Dependency Review allowlist entries they required were removed.
- **MCP SDK floor raised to 1.28.1**, resolving GHSA-hvrp-rf83-w775 and
  GHSA-jpw9-pfvf-9f58 (1.27.2) and GHSA-vj7q-gjh5-988w (1.28.1).
- **New transitive security floors** pinned in `[tool.uv] constraint-dependencies`:
  `cryptography>=50.0.0` (GHSA-g6cj-pr64-35w5), `pyasn1>=0.6.4` (GHSA-m4p7-r5rc-7g4j,
  GHSA-8ppf-4f7h-5ppj, GHSA-hm4w-wwcw-mr6r), `pymdown-extensions>=11.0.1`
  (GHSA-gm37-52c6-37mw, GHSA-9xwg-3r6f-jcx2), and `pip>=26.2` (PYSEC-2026-3721).
- **Frontend advisories cleared** (`npm audit`: 0 vulnerabilities): `nanoid`
  (GHSA-2v37-7h3g-55p8, GHSA-28wg-ghj8-5hjv), `js-yaml` (GHSA-5p4m-2wfm-xmqj),
  `postcss` (GHSA-fxqj-rqcc-2cmp), and `brace-expansion` (GHSA-3jxr-9vmj-r5cp and
  follow-ups).
- **Code scanning**: fixed a log-injection sink in
  `api/services/text_processing_context.py` where the request-controlled assertion
  `preference` and `language` reached a log record unsanitized, and removed a dead
  `_INDEX_DOCUMENT_COUNTS` global in `phentrieve/data_processing/release_contract.py`.
- The three open chromadb advisories (GHSA-36p7-vc44-83pf, GHSA-2wm9-hf6c-p5cr,
  GHSA-xph7-9rjv-w5fr) all require the ChromaDB HTTP server, multi-tenant
  authorization, or `SimpleRBACAuthorizationProvider`. Phentrieve only constructs an
  embedded `chromadb.PersistentClient`, so they are documented as not applicable
  rather than silently ignored.

## [0.26.0] — 2026-07-13 (CLI 0.26.0 / API 0.17.0 / Frontend 0.17.1)

Reproducible benchmark result storage. Retrieval, extraction, and LLM benchmarks
now write to one collision-free run directory per run, described by a manifest
that records the run's identity, dataset checksum, configuration, and the
machine-readable location of every artifact it produced. The API is unchanged.

### Added

- **Shared benchmark result store** (`phentrieve/benchmark/result_store.py`).
  Runs land in `results/<type>/<dataset>/<model>/<run-id>/` with a `manifest.json`,
  a `summary.json`, and analysis-first `terms.jsonl` / `cases.jsonl` artifacts;
  extraction additionally emits `diagnostics/chunks.jsonl` with every retrieved
  candidate and the stage at which it was filtered. Previous flat output formats
  are preserved under `legacy/` for the migration period.
- **`--run-id` / `--overwrite` / `--output-dir`** on the retrieval, extraction, and
  LLM benchmark commands, for stable run names and explicit reuse. The LLM
  benchmark's `--output-path`, `--checkpoint-path`, and `--artifacts-dir` are
  deprecated but still honoured.
- **CSC/GSC LLM benchmark support** and recursive discovery of nested runs by
  comparison and visualization commands.

### Changed

- **Checkpoint reuse is bound to the inputs that determine its contents** —
  dataset checksum, accounting configuration, resolved provider/model/base URL,
  and the contents of the prompt templates the run loads. An edited prompt
  template can no longer be resumed into an existing run, which would have merged
  old-prompt and new-prompt document outputs into one set of metrics.
- **Retrieval benchmarks record non-evaluable cases explicitly** rather than
  dropping them, and every row of `cases.jsonl` is keyed on the same 0-based
  `case_id` that `terms.jsonl` joins on.
- **Frontend dependencies** bumped (vite, eslint, prettier, postcss, terser,
  libphonenumber-js, @vitest/coverage-v8).

### Fixed

- **Overwritten runs no longer inherit artifacts they did not produce.** Reusing a
  run directory left the previous run's files in place, and the manifest — which
  decides what to advertise from what exists on disk — published them as the new
  run's own output.
- **`benchmark compare` no longer lists extraction and LLM runs as retrieval
  models.** All three benchmark types write a `summary` artifact into the shared
  results root, so the dense-retrieval comparison table had been picking them up
  and scoring them `MRR (Dense) = 0`.
- **LLM run status is derived from per-case outcomes.** A run in which every
  document failed had still been recorded in its manifest as `complete`.

## [0.25.0] — 2026-07-04 (CLI 0.25.0 / API 0.17.0 / Frontend 0.17.0)

LLM extraction contract v2 (Phase 2, blocks B0–B3): the model's assertion and
experiencer axes become load-bearing, family-history findings are partitioned
out of the proband set, ruled-out qualifiers surface as excluded findings, and a
canonical `absent → excluded` export vocabulary closes the long-standing
"absent silently exports as present" trap. Includes a deep review pass that
root-caused and eliminated a precision regression and hardened the benchmark.

### Added

- **Family-history findings** are partitioned from proband terms and surfaced
  separately through the service, REST, and MCP; the Phenopacket export is guarded
  by experiencer so family mentions never leak into the proband packet.
- **`negated_qualifier → excluded` findings.** An "X without Y" phrase emits a
  generated excluded finding for Y (gated by the similarity floor), surfaced
  through REST and MCP.
- **Ontology guard.** Resolved terms known to sit outside Phenotypic abnormality
  (HP:0000118) — clinical modifiers/course such as "Mild", "Spatial pattern",
  "Slowly progressive" — are dropped from the output. Fails open on unknown
  ancestry so a real finding is never removed.
- **Benchmark `assertion_distribution` report** so the negation/family axes the
  contract produces (which present-only scoring projects away) stay visible,
  including a `documents_counted` / `documents_total` under-count guard.

### Changed

- **A `normal` finding now exports as excluded** across every surface (REST, MCP,
  Phenopacket, Vue). A normalcy verdict is a ruled-out abnormality; this matches
  the LLM backend and closes the normal-exports-as-present class.
- **Extraction benchmark is multi-vector aware** with committed strict +
  present-only baselines as a regression fence; `assert-no-regression` now fails
  closed on a missing gated metric or a missing `scoring_mode`.

### Fixed

- **Phase-2 mapping precision regression** eliminated at its root: dropped the
  unused `experiencer`/`assertion` keys that perturbed the mapping LLM toward
  modifier nodes, and stopped a `normal` category being promoted to `present`.
  Restores the benchmark to baseline exactly (recall unchanged).
- **CLI Phenopacket export** of `status`/`excluded`-keyed aggregated terms now
  exports ruled-out findings as excluded instead of defaulting to present.
- **Ontology-guard cache** no longer latches the fail-open empty map for the
  process lifetime when the HPO database is briefly unavailable.

## [0.24.1] — 2026-07-03 (CLI 0.24.1 / API 0.16.1 / Frontend 0.16.1)

Dependency, security, and code-scanning maintenance — consolidates the open
Dependabot updates and clears the outstanding GitHub security surface. Also ships
the previously-unreleased frontend footer / MCP-connect work under one unified tag.

### Added

- **Footer shortcuts for the API and MCP.** The bottom bar gains two
  developer-integration icons in the lower right, set off by a divider and
  placed just before the live status indicator: `mdi-api` opens the live OpenAPI
  (Swagger UI) docs in a new tab (URL derived from the configured API base), and
  `mdi-robot-outline` links to the new MCP connection page. Tooltips are
  localized via `app.footer.apiTooltip` / `app.footer.mcpTooltip` across all five
  locales.
- **"Connect an AI Agent (MCP)" page (`/connect`).** A new view
  (`views/McpAccess.vue`) explaining how to attach Phentrieve's public,
  read-only MCP server to Claude (web & desktop), ChatGPT developer mode, Claude
  Code, and Codex CLI, with copy-to-clipboard recipes, the read-only tool
  inventory, and the research-use disclaimer. The server address is derived from
  `window.location.origin` (overridable via `VITE_MCP_URL`). The route is
  `/connect`, not `/mcp`, because the MCP transport is proxied same-origin under
  the `/mcp` prefix. Includes a "Back to home" affordance mirroring the FAQ view.

### Changed

- **Dependencies — security & maintenance (consolidates Dependabot #294, #296,
  #297, #299, #300).** Python: `starlette` 1.0.1→1.3.1, `cryptography`
  46.0.7→49.0.0, `msgpack` 1.1.2→1.2.1, `python-multipart` 0.0.29→0.0.32,
  `pydantic-settings` 2.13.1→2.14.2, with the `fastapi` floor raised to 0.137.2
  and enforced security floors added under `[tool.uv]`. Frontend: refreshed to
  latest minor/patch (`vue` 3.5.39, `vite` 8.1.3, `form-data` 4.0.6, plus the
  12-package group; `npm audit` clean). CI: `actions/checkout`→v7,
  `actions/cache`→v6. `typer` is held at `<0.26.0` — 0.26 vendors Click and
  breaks the lazy CLI groups, so Dependabot #298 was declined.

### Fixed

- **CodeQL code-scanning findings.** Resolved `py/empty-except`
  (`api/mcp/service_adapters.py`), `py/import-and-import-from` (`api/main.py`),
  and `py/ineffectual-statement` (`api/auth/email.py`).
- **Anonymous `/auth/refresh` 403 console error.** Silent session restore on app
  start now skips the refresh request entirely when the readable `csrf_token`
  cookie is absent (no session to restore, and the request would 403 anyway),
  eliminating a guaranteed console error on every anonymous visit. A stale
  persisted user is cleared without a network call.

### Security

- Cleared 12 Dependabot advisories: `starlette`, `cryptography`, `msgpack`,
  `python-multipart`, `pydantic-settings` (pip) and `form-data` (npm). The
  `chromadb` CRITICAL advisory (GHSA-f4j7-r4q5-qw2c) is documented as **not
  applicable** — the pre-auth code injection is in Chroma's HTTP server mode
  only, phentrieve uses an embedded `PersistentClient` exclusively, and no
  patched release exists above 1.5.9 (which is also pinned for data-bundle
  compatibility).

## [0.24.0] — 2026-06-14 (CLI 0.24.0 / API 0.16.0 / Frontend 0.15.0)

MCP server stabilization: every quality dimension lifted toward >9/10, traced to
`.planning/analysis/2026-06-14-mcp-stabilization-plan.md`. Schema/descriptor
changes intentionally roll `capabilities_version` (the warm-client signal).

### Added

- **Orthogonal experiencer/assertion axes + negation qualifier (LLM-2).** The
  Phase-1 grounded/non-grounded schemas gain `experiencer`/`assertion` enums and
  a `negated_qualifier`, declared before the legacy `category` label so the model
  reasons about them first; `negated_qualifier` ("X without Y") is threaded to the
  aggregated term. The two_phase/en prompt adds a negation-scope rule and
  few-shots (prompt `v3.0.0` → `v3.1.0`).
- **Live diagnostics (D3).** `diagnostics` probes the real caches and reports
  `loaded | loading | cold | error` for `embedding_model`/`vector_index` instead
  of a constant `"lazy"`.
- **`text_attributions` is always an array (D1).** Empty `[]` now means a
  semantically retrieved term with no literal span (kept present at compact).

### Changed

- **Band-based `no_high_confidence_match` (B1).** A threshold-emptied result set
  is correctly flagged (was the inverted `false`).
- **`chunk_text` lazy-loads the embedding model (B2),** so all 7 advertised
  strategies work; a genuine load failure is `temporarily_unavailable` (not
  `invalid_input`).
- **LLM extraction no longer mislabels relatives (LLM-1).** `family_history` is
  dropped from the actionable set and aggregation keys on
  `(term_id, experiencer, assertion)`, removing the self-contradictory
  present+negated pair; the coarse chunk-status override is dropped for the LLM
  path so "X without Y" keeps X present.
- **One canonical phenopacket (R1).** `export_phenopacket` returns the parsed
  `phenopacket` object as the single form; the `phenopacket_json` blob is gated to
  `standard`/`full`. The object is kept whole at every mode.
- **Leaner payloads.** The extract→export pre-fill is capped/compacted at
  minimal/compact (R2); deterministic `num_results_per_chunk` defaults to 3 so the
  exact child term is not lost (Q1); response synonyms are capped to 10 with a
  `synonyms_truncated` count (R3).
- **Accurate error recovery (D4).** `not_found`/`ambiguous_query` →
  `recovery_action: resolve_identifier` with a `search` next-command.
- **`compare` honesty (D2).** `lca_details` `ic_proxy` is renamed
  `normalized_depth` (it is `depth/max_depth`, a structural proxy, not corpus IC).
- **Alias-policy docs (B4)** state that alias forwarding is client-dependent and
  canonical names are always safe.

### Fixed

- **Value-level blank-text envelope (B3).** Blank/whitespace `text` returns the
  validator's reason, not the unknown-argument-name template with
  `allowed_values`=parameter-names.
- **Concurrent-validation race.** `LLMPhenotype`'s forward reference is resolved
  eagerly (`model_rebuild`) so parallel grouped phase-1 cannot race the lazy
  rebuild; the `tests/unit/api` package shadowing of the top-level `api/` package
  is fixed via a proper test-package hierarchy so the full suite collects.

## [0.23.2] — 2026-06-14 (CLI 0.23.2 / API 0.15.3 / Frontend 0.14.1)

### Added

- **MCP `compare_hpo_terms` now explains its score.** At `response_mode`
  `standard`/`full` the tool returns `lca_details`: the most informative common
  ancestor (id + label + depth), each term's depth and information-content proxy,
  and the subsumer path length; `minimal`/`compact` keep the lean payload. The
  tool previously ignored `response_mode` entirely (assessment D5).
- **MCP `export_phenopacket` returns a native phenopacket object.** A
  `phenopacket` JSON object is returned alongside the existing
  `phenopacket_json` string (kept for backwards compatibility), so consumers no
  longer have to `JSON.parse` an escaped blob (assessment D4, per MCP
  structured-output guidance).
- **MCP `search_hpo_terms` low-confidence signalling.** Each hit carries a
  `confidence_band` (high ≥ 0.7, moderate ≥ 0.4, else low) and the response sets
  `no_high_confidence_match` when the top hit is below the high floor, so a
  nearest-neighbour match for out-of-domain text is not mistaken for a real one
  (assessment D6). The default `similarity_threshold` is unchanged.
- **MCP best-effort startup warmup** loads the embedding model + vector index
  before serving so the first diagnostics/extract call is warm; failures are
  logged and never block startup (assessment D9).
- **MCP capability descriptor documents the cache-key contract.** A
  `cache_contract` field clarifies that `capabilities_version` is the canonical
  warm-cache key (echoed in `_meta`, stable across `details`) while
  `descriptor_hash` is the per-descriptor content hash (assessment D8).

### Fixed

- **MCP deterministic extractor no longer over-negates co-located phenotypes.**
  Negation is now applied per match by overlapping the matched phrase span with
  the computed negated scope, so an affirmed concept beside a negated one
  ("severe intellectual disability **without** regression") is no longer marked
  negated; cue-stripped single-concept chunks ("no X") stay negated (assessment
  D1, the highest-severity finding).
- **MCP deterministic extractor surfaces co-occurring findings.** The chunker now
  splits multi-concept spans on progression markers ("progressing to", "evolving
  into", "followed by", …), so distinct phenotypes ("hypotonia progressing to
  hypertonia") each surface as their own best-match chunk without raising
  `num_results_per_chunk` (assessment D2); degenerate function-word chunks (e.g.
  a bare "due") are dropped (assessment D3).
- **MCP extraction payload shape.** Projected aggregated terms now carry a
  uniform `text_attributions` key (empty list when none) and drop null padding,
  so the schema is consistent and `full` mode is not bloated (assessment D7/D13).
- **MCP `export_phenopacket` provenance is honest.** Client-supplied phenotypes
  are marked `match_method: client_supplied` / `source_mode: unknown` instead of
  fabricating `legacy_dict`/`chunk`; caller-supplied provenance is preserved
  (assessment D11).

- **Optional user accounts with a higher full-text quota.** Visitors can
  register and sign in via an unobtrusive top-right account control; a
  **verified** account raises the daily full-text (LLM) quota from the anonymous
  IP limit (`PHENTRIEVE_LLM_DAILY_LIMIT`) to a per-account limit
  (`PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT`, default 20). The feature is off by
  default (`PHENTRIEVE_AUTH_ENABLED=false`) and adds no new Python runtime
  dependencies.
  - Backend (`api/auth/`): SQLite user store, bcrypt hashing, JWT access token +
    rotating HttpOnly refresh cookie with double-submit CSRF, console/SMTP email
    backends, email verification (verify-to-upgrade), password reset, and login
    lockout. New `GET /api/v1/text/quota` reports the caller's tier without
    consuming quota.
  - Frontend: Pinia auth store (token in memory, silent refresh on load), axios
    refresh-retry interceptor, `AccountButton` + `AuthDialog`, email-link
    verify/reset views, and `auth.*` i18n for all five locales.
  - Optional pre-verified dev account seeded at startup via
    `PHENTRIEVE_AUTH_SEED_EMAIL` / `PHENTRIEVE_AUTH_SEED_PASSWORD`.
  - Docs: `deployment/authentication.md` (setup, tier table, SMTP for
    production) plus quota notes in the API and frontend usage guides.

### Fixed

- **MCP deterministic extractor no longer asserts negated phenotypes.** Assertion
  detection now runs over the chunk's restored within-sentence context, so
  prepositional negation (`no X`, `not X`, `does not have X`) is honoured instead
  of being lost when the chunk cleaner strips the leading cue — eliminating
  false-positive affirmed phenotypes (the highest-severity defect). Negated
  findings are emitted with `assertion_status: negated` so `excluded: true`
  phenopacket features can be built.
- **MCP `export_phenopacket` consumes extractor output directly.** It accepts both
  `{hpo_id, label, assertion}` and the raw `{id, name, assertion_status}` shapes,
  preserves a per-item `score` as the evidence confidence (no more
  `confidence: 0.0000`), and returns a typed `validation_failed` with a
  did-you-mean mapping hint instead of leaking a raw `KeyError`.
- **MCP `capabilities_version` is stable across `details` expansion**, so it always
  matches the value echoed in `_meta` and the warm-cache contract holds; the
  per-descriptor content hash is exposed separately as `descriptor_hash`.
- **MCP input validation**: empty/whitespace queries are rejected with
  `validation_failed`; `chunk_text` `strategy` is an enumerated value validated
  against the known strategy set (listed in capabilities); `next_commands` no
  longer emit non-executable `<placeholder>` free-text steps.
- **Phenopacket provenance** records `phentrieve-core <version>` to disambiguate
  the extraction-library version from the MCP server version.

### Changed

- **MCP extraction output is normalized for the agent consumer (MCP boundary
  only; REST API and frontend contracts unchanged).** Deterministic extraction
  defaults to the single best match per phrase (was up to 10 mutually exclusive
  siblings); aggregated terms carry one `score` and one chunk-index scheme
  (`chunk_ids` + `top_evidence_chunk_id`) instead of four duplicate score fields
  and dual 0-/1-based schemes; empty-match chunks are dropped unless
  `include_unmatched_chunks=true`; `hpo_matches: []` is always serialized.
- **`recommended_citation` is emitted in every response mode** (including
  compact/minimal) and embeds the HPO release version; `include_details` defaults
  to `false` and is honoured at compact verbosity when explicitly set.
- **MCP observability** now surfaces per-phase LLM timing
  (`_meta.observability.phase_timings`) and a `latency_profile` in capabilities.

## [0.23.1] — 2026-06-14 (CLI 0.23.1 / API 0.14.0 / Frontend 0.13.1)

### Changed

- **Frontend build toolchain migrated to vite 8 / vue-router 5.1.** Records and
  verifies the toolchain migration tracked in #273. The repo was held on the
  pinned vite 6 toolchain because `vue-router` 5.1.0 declares
  `peerOptional vite "^7.0.0 || ^8.0.0"`, which made `npm ci` fail with
  `ERESOLVE` (held back in #270/#271). The dependency bumps landed in 0.23.0
  (vite 6 → 8 via rolldown, `vue-router` 5.0.7 → 5.1); this release closes out
  the migration: the production build — including `vite-plugin-compression`
  brotli output and the bundle visualizer — and the full 306-test Vitest suite
  are green under vite 8, and the Docker/CI Node version (`20.19`) satisfies
  vite's `^20.19.0 || >=22.12.0` requirement. The `vue-router` dependabot ignore
  block added in #271 has been removed so normal updates resume. Closes #273.

### Added

- **Interactive full-text annotation curation.** On the highlighted clinical
  note you can now curate the automatic annotations directly: left-click,
  right-click, or keyboard-activate a highlighted phrase to open a menu, then
  **change the term** (re-runs the phrase through single-term query and offers
  ranked HPO candidates in a picker dialog with definitions, scores, and an
  affirmed/negated toggle), **remove** the annotation (with an Undo snackbar),
  **add to collection**, or **revert** a curated term to its original value.
  Select previously-unhighlighted text to **annotate a new span**. Edits are
  per-turn, persisted, and provenance-tracked — findings carry an auto/manual
  badge and manually curated spans render in a distinct colour in the note.
- **Auto-locate terms by label.** When the extractor returns a term without
  usable text attributions but its label appears verbatim in the note, the term
  is now highlighted instead of only appearing in the findings list.

### Changed

- Removed the orphaned full-text annotation workspace cluster (~2,300 lines: the
  superseded `FullTextAnnotationWorkspace`, `AnnotatedDocumentPane`,
  `AnnotationInspectorPanel`, `PhenotypeFindingsPane`, `useDocumentAnnotations`,
  `useCustomHighlightOverlay`, the `fullTextWorkspace` store/constants, and the
  `annotationInspector` util) plus dead chunk-highlight code in `ResultsDisplay`.

### Fixed

- Manual annotation of a selection that includes surrounding whitespace (typical
  of a mouse drag) now highlights correctly — the selection text and its
  character offsets are trimmed together so the span stays aligned to the note.
- The affirmed/negated choice in the term picker now applies; it previously had
  no effect in the production build, so manual annotations were always affirmed.

## [0.22.1] — 2026-06-13 (CLI 0.22.1 / API 0.13.0 / Frontend 0.12.1)

### Fixed

- **Full-text annotation hover no longer collapses the highlights.** Hovering a
  single annotated phrase previously filtered the note down to that one
  phenotype, so every other highlight disappeared and the re-rendered `<mark>`
  could leave a highlight "stuck". Annotated segments are now a pure function of
  the note, chunks, and terms, so all mentions stay highlighted and the hovered
  phenotype is emphasised purely via a CSS class — no annotations collapse and
  the highlight always resets on leave.
- **Phenotype tooltip is now readable.** The tooltip rendered near-black text on
  Vuetify's default dark surface (~1.7:1 contrast) because the custom surface
  style sat in a scoped `:deep()` rule that could not reach the body-teleported
  overlay. It now uses a global, theme-aware surface (`surface` / `on-surface` /
  `primary` tokens) that meets WCAG AA (≈21:1 label, ≈5.6:1 eyebrow) in both
  light and dark themes.

### Added

- **Keyboard accessibility for annotated spans.** Annotated mentions are now
  focusable (`tabindex`, `aria-label`), reveal their tooltip on keyboard focus
  (`open-on-focus`), and show the same emphasis on focus as on hover (WCAG
  1.4.13). The clinical-note expand/collapse toggle gained an accessible name.

### Changed

- **Local dev stack no longer enforces the public LLM daily quota.** The quota
  gate only fires when `PHENTRIEVE_ENV=production`; `docker-compose.dev.yml` now
  runs the dev API as `development`, so full-text LLM extraction works locally
  without hitting the daily limit. Set `PHENTRIEVE_ENV=production` in `.env` to
  exercise production quota behaviour.

## [0.22.0] — 2026-06-13 (CLI 0.22.0 / API 0.13.0 / Frontend 0.12.0)

### Changed

- **MCP server modernized to the FastMCP v3 "Gen-3" house style (breaking).**
  Tools are renamed from dotted (`phentrieve.extract_hpo_terms`) to
  underscore-namespaced (`phentrieve_extract_hpo_terms`). Every tool now returns
  a Family B envelope — `success` plus a `_meta` block (`tool`, `request_id`,
  `elapsed_ms`, `response_mode`, `capabilities_version`,
  `unsafe_for_clinical_use`, `next_commands`) — or a structured error
  (`error_code`, `retryable`, `recovery_action`). A new `response_mode`
  (`minimal | compact | standard | full`, default `compact`) controls token
  cost. Clients depending on the previous response shapes or tool names must
  update.
- **MCP transport is Streamable HTTP only.** The stdio transport (the broken
  `phentrieve-mcp` entry point) and the legacy `fastapi-mcp` OpenAPI bridge have
  been removed. `phentrieve mcp serve` no longer takes `--http` (HTTP is the only
  mode). The `fastapi-mcp` dependency is replaced by `fastmcp>=3.2`.

### Added

- New MCP tools: `phentrieve_export_phenopacket` (GA4GH Phenopacket v2 export),
  `phentrieve_chunk_text` (chunk-only), and `phentrieve_diagnostics` (subsystem
  health + recent errors). `phentrieve_get_server_capabilities` is renamed to
  `phentrieve_get_capabilities` and now reports a content-hashed
  `capabilities_version` and `descriptor_chars`.
- MCP discoverability: `phentrieve://schema/overview` and
  `phentrieve://schema/tool-guide` markdown resources, richer server
  `instructions`, `readOnlyHint` tool annotations, structured output schemas,
  argument-alias normalization with did-you-mean validation errors, and
  `next_commands` workflow hints on every response.
- MCP research-use acknowledgement parity: extraction tools require
  `research_use_acknowledged=true` when the server runs in public-hosted or
  research-ack mode, mirroring the REST `X-Research-Ack` gate.

## [0.20.0] — 2026-04-30 (CLI 0.20.0 / API 0.11.0 / Frontend 0.10.0)

### Changed

- Public REST and MCP LLM extraction now use a server-owned
  `gemini/gemini-3.1-flash-lite-preview` target. Public clients should omit
  `llm_model`, `llm_provider`, and `llm_base_url`; those request fields are
  rejected by the public API/MCP policy.
- Frontend text-processing requests no longer send `llm_model` for the public
  LLM backend, and frontend logs report LLM target selection as server-owned.

### Documentation

- Documented the public REST/MCP prompt-injection defense-in-depth posture,
  research-only limitation, and warning not to submit identifiable patient data
  to public deployments.

## [0.19.0] — 2026-04-26 (CLI 0.19.0 / API 0.10.0 / Frontend 0.9.0)

### Added

- **`--profile NAME`** flag on `query`, `text process`, `text interactive` (issue #28).
  Apply a named profile from `phentrieve.yaml` to preset CLI options. See
  [Configuration Profiles](docs/user-guide/configuration-profiles.md). Both root
  placement (`phentrieve --profile X cmd`) and per-command placement
  (`phentrieve cmd --profile X`) work; subcommand-level wins on conflict.
- **`phentrieve config`** subcommand group with `list-profiles`, `show`,
  `validate`, `path` subcommands.
- **`--show-resolved-config`** debug flag on every command. Prints resolved
  option values with source labels (profile/yaml/const/commandline) to stderr
  before running.
- **`PHENTRIEVE_PROFILE`** environment variable, equivalent to `--profile`.
- New `phentrieve.yaml` sections: `profiles:` and `extraction:`.
- Built-in profiles: `default` (API-matching strict defaults) and `interactive`
  (legacy `text interactive` defaults).
- **Adaptive re-chunking** (issue #148): an opt-in mechanism that detects
  per-chunk retrieval quality and, when poor, subdivides the chunk into
  sentence-bounded sub-chunks, re-queries them, and merges results. Enable
  on the CLI via `phentrieve text process FILE --adaptive-rechunking` with
  three threshold flags (`--adaptive-rechunking-quality-threshold`,
  `--adaptive-rechunking-margin-threshold`,
  `--adaptive-rechunking-max-depth`), or in `phentrieve.yaml` under the new
  `extraction.adaptive_rechunking:`
  block (also surfaced as `Profile.adaptive_rechunking` for per-profile
  overrides). The API `/text/process` request schema gains an
  `adaptive_rechunking` field (request-time override), and the response's
  `meta` block gains an `adaptive_rechunking` summary
  (`triggered_chunks`, `subdivided_chunks`, `recursion_depth`, etc.) when
  the feature runs. See
  [docs/user-guide/adaptive-rechunking.md](docs/user-guide/adaptive-rechunking.md).

### Fixed

- **`phentrieve text interactive`** now uses config-driven defaults (issue #171).
  Previously hardcoded `language="en"`, `chunk_retrieval_threshold=0.3`,
  `aggregated_term_confidence=0.35`, `num_results=5` are now read from the new
  built-in `interactive` profile (preserving prior behavior — no migration
  needed). Pass `--profile default` to switch to API-matching strict defaults.
- **Frontend `DEFAULT_SIMILARITY_THRESHOLD`** aligned with the API: `0.5` → `0.3`.
  This is a behavior change for users who relied on the frontend's stricter
  cutoff. To recover, pass an explicit threshold in the UI.

### Changed

- The previously-documented `--config-profile` flag (which was never
  implemented) is replaced by the now-real `--profile`.
  `docs/user-guide/configuration-profiles.md` is rewritten in place to reflect
  the actual design.
- **`orchestrate_hpo_extraction` return type**: now returns an
  `OrchestrationResult` dataclass instead of a plain tuple. Backward
  compatible — legacy 2-tuple unpacking via `__iter__` continues to work
  for all existing call sites. New attribute `raw_query_results` exposes
  the unfiltered top-K from `query_batch` for callers (e.g. adaptive
  re-chunking) that need scores below `chunk_retrieval_threshold`.

## [0.18.2] — 2026-04-25

**Component versions**: phentrieve `0.18.2`, phentrieve-api `0.9.3`, phentrieve-frontend `0.8.3`

Patch release for release-bundle workflow stabilization.

### Fixed

- fixed data bundle creation for minimal database-only bundles
- aligned the release data bundle workflow with the repository data directory
  used by `phentrieve data prepare`

## [0.18.1] — 2026-04-25

**Component versions**: phentrieve `0.18.1`, phentrieve-api `0.9.2`, phentrieve-frontend `0.8.2`

Patch release for the deprecated second-stage ranking removal.

### Changed

- removed the obsolete second-stage ranking implementation, public options,
  stale benchmark artifacts, and related documentation/frontend copy
- added a tracked-source guard test for removed ranking terminology

### Benchmarks

- `gemini-3.1-flash-lite-preview` full 10-doc GeneReviews run:
  - micro precision `0.820`
  - micro recall `0.806`
  - micro F1 `0.813`

## [0.15.0] — 2026-04-17

**Component versions**: phentrieve `0.15.0`, phentrieve-api `0.8.2`, phentrieve-frontend `0.7.3`

Post-merge stabilization and release follow-up.

### Fixed

- repaired the GitHub Pages documentation deployment by aligning the docs
  workflow Python version with the project baseline and fixing a strict MkDocs
  warning caused by an invalid repository-root link in the test documentation

### Release

- bumped the Phentrieve CLI/library minor version to `0.15.0`

## [0.14.0] — 2026-04-17

**Component versions**: phentrieve `0.14.0`, phentrieve-api `0.8.2`, phentrieve-frontend `0.7.0`

Focused LLM CLI stabilization and provider-quality release.

### Changed

- default LLM model changed to `gemini-3.1-flash-lite-preview` after full-run
  GeneReviews benchmarking
- benchmark observability now records richer Gemini usage and cost accounting,
  including thought and cached-token usage
- Gemini retries now cover transient transport/network failures so long-running
  CLI and benchmark executions do not fail on a single connectivity blip
- grounded routing, grouped Phase-1 extraction, and shared mapping prompt
  behavior were stabilized after regression investigation and reruns
- invalid trusted-proxy CIDR values are no longer logged in clear text

### Benchmarks

- `gemini-3.1-flash-lite-preview` full 10-doc GeneReviews run:
  - wall clock `109.640s`
  - micro precision `0.8291`
  - micro recall `0.8186`
  - micro F1 `0.8238`
  - estimated cost `$0.0546`

## [0.13.0] — 2026-04-10

**Component versions**: phentrieve `0.13.0`, phentrieve-api `0.8.0`, phentrieve-frontend `0.7.0`

A 90-commit refactor-and-perf release that closes the 2026-04-09 code quality
review, removes a deprecated second-stage ranking layer, and speeds up the full
CI pipeline by ~17% on warm caches.

### ⚠ BREAKING CHANGES

- **Deprecated second-stage ranking removed** across every layer (CLI, API,
  frontend, config, tests, docs). It was unvalidated against benchmark data and
  added significant complexity without measured accuracy gains.
- **`sys.path.append`** in `api/main.py` removed. The API must now be installed
  via `pip install -e .` / `uv sync` before running `uvicorn api.main:app` or
  the module import will fail. Docker images already did this correctly.

### ✨ Features

- **`ErrorResponse` API contract**: every 4xx/5xx response from the Phentrieve
  API now conforms to a single `ErrorResponse` Pydantic schema
  (`status_code`, `error`, `detail`, `request_id`). Three exception handlers
  registered in `create_app()`:
  - `StarletteHTTPException` — routing 404s + explicit `HTTPException` raises
  - `RequestValidationError` — Pydantic validation failures (422)
  - `Exception` — catch-all 500 with server-side traceback logging and a
    non-leaky `internal_server_error` payload
  - Dict and list detail payloads are preserved through the handler (the
    similarity router returns structured error dicts on 404 that clients
    can parse directly).

### ♻ Refactoring

- **Backend query orchestration**: `phentrieve/retrieval/query_orchestrator.py`
  reduced from 1057 → 715 LOC (−32%). Extracted shared retrieval and formatting
  flow.
  Extracted `InteractiveState` to its own module with proper instance attributes.
  Extracted `convert_multi_vector_to_chromadb_format()` to `phentrieve/retrieval/utils.py`.
- **HPO extraction orchestrator**: `phentrieve/text_processing/hpo_extraction_orchestrator.py`
  reduced from 298 → 103 LOC via decomposition into 4 private helpers in
  `_hpo_extraction_helpers.py`: `process_chunk_matches`, `load_term_details`,
  `build_evidence_map`, `aggregate_and_rank`. 9 characterization tests lock
  the public behavior.
- **Chunker pipeline**: `_create_chunkers()` in `phentrieve/text_processing/pipeline.py`
  reduced from 158 → ~37 LOC via a registry-based factory (`_chunker_registry.py`).
  Dead duplicate sliding_window branch removed.
- **Language resource loading**: `FinalChunkCleaner.__init__` DRY'd up —
  three near-identical word-list loading blocks collapsed to calls on a new
  `_load_language_word_list()` helper.
- **API application factory**: `create_app()` pattern introduced; MCP mount
  moved from import-time to `lifespan` startup; `similarity_router` HPO graph
  loads lazily via `@lru_cache` on first use (was eager at import).
- **Model caches hardened**: `LOADED_SBERT_MODELS`, `LOADED_RETRIEVERS`,
  `MODEL_LOADING_STATUS`, `MODEL_LOAD_LOCKS` replaced with bounded
  `cachetools.TTLCache` instances protected by a `threading.Lock`. Race
  condition between `in` check and indexed access fixed with `.get()` /
  `.pop(default)` patterns. `cleanup_model_caches()` wired into lifespan
  shutdown — cancels in-flight loads via `asyncio.shield` + `gather` before
  clearing.
- **SBERT loader inlined**: `_load_model_with_status_tracking()` was a 2-caller
  abstraction; after model-loading simplification the only remaining call site
  was inlined into `_load_sbert_in_background()` and the `is_sbert` dead
  parameter dropped.
- **Frontend mega-components decomposed**:
  - `QueryInterface.vue` 1483 → 801 LOC (−46%)
  - `ResultsDisplay.vue` 1079 → 634 LOC (−41%)
  - `App.vue` ~600 → 570 LOC (−5%)
  - 6 new composables: `useAdvancedOptions`, `usePhenotypeCollection`,
    `useFileDownload` (eliminates 3× download duplication), `useVersionCheck`,
    plus query/text-processing composables
  - 5 new sub-components: `AdvancedOptionsPanel`, `PhenotypeCollectionPanel`,
    `ResultItem`, `ChunkResultsView`, `AggregatedTermsView`
- **Frontend store consistency**: `queryPreferences` store migrated from
  Options API to setup store — all 4 Pinia stores now use the same pattern.
- **Hardcoded constants**: thresholds extracted to `frontend/src/constants/defaults.js`,
  URLs to `frontend/src/constants/urls.js`. 7+ `document.querySelector` calls
  replaced with `data-tutorial-step` attributes and reactive refs.
- **Dead code removed**: `phentrieve/visualization/` module deleted;
  `useDisclaimer.js` composable deleted; 6 unused chunker class imports removed
  from `pipeline.py`.

### ⚡ Performance

**CI pipeline (Tier 1 + Tier 2 speedup — 17 tasks, plan at
`.planning/active/CI-SPEEDUP-PLAN-2026-04-10.md`):**

- **`pytest-xdist -n auto`** enabled in `pyproject.toml` addopts →
  `make test` 25.9s → 18.5s (−29%), 927 tests parallelized across CPU cores
- **mypy cache** persisted across CI runs via `actions/cache@v4`
- **`COVERAGE_CORE=sysmon`** on Python 3.12+ (~53% faster coverage
  instrumentation per Trail of Bits benchmark; branch coverage not enabled
  so the sysmon gap is a non-issue)
- **Vite esbuild minifier** replaces terser (30-90× faster, ~0.5–2%
  compression loss) → frontend build 4.0s → 2.1s (−47%). `console` and
  `debugger` statements still stripped in production via `esbuild.drop`
  gated on `mode === 'production'` (dev server preserves them).
- **Brotli compression + bundle visualizer** gated behind `!process.env.CI`
  to skip deploy-time work in CI
- **ESLint `--cache`** with `--cache-strategy content`, cached in CI
- **Prettier `--cache`** with `--cache-strategy content`
- **Workflow concurrency groups** with `cancel-in-progress: true` on
  `pull_request` events only (main branch pushes run to completion)
- **`astral-sh/setup-uv@v7`** with `enable-cache: true` replaces the
  3-step `setup-python + install uv + actions/cache` pattern
- **Vitest `pool: 'threads'`** + `test:ci` script with `--no-coverage` for
  PR runs (main branch still uploads coverage to Codecov)
- **`timeout-minutes`** on every job (caps runaway cost at ~2× normal)
- **`fail-fast: false`** on Python matrix so cross-version regressions
  aren't masked by the first failure
- **`HF_HOME` cache** for SBERT (`paraphrase-MiniLM-L3-v2`, `BioLORD-2023-M`)
  model downloads in integration tests
- **`make ci-local`** meta-target reproduces full CI on one command
  (format-check, lint, typecheck, test-ci, frontend-lint, format-check,
  test-ci, build-ci in the same order as CI)

**Net CI impact**: full wall-clock 5.1 min → 4.3 min on warm caches (−17%)

### 🐛 Bug Fixes

- **Phenopacket DOB field**: frontend export composable was writing the
  subject's date of birth to `subject.timeAtLastEncounter` (semantically
  incorrect per Phenopacket v2). Fixed to populate `subject.dateOfBirth`.
- **`ErrorResponse.detail`** widened to `str | dict[str, Any] | list[Any]`
  so structured error payloads (e.g., the similarity router's 404 dict)
  pass through unchanged instead of being stringified.
- **Phenopacket composable**: replaced blocking `alert()` with a thrown
  `Error` so the consuming component can present an accessible snackbar.
- **TTLCache race conditions**: all get/set/del on model caches now
  wrapped in a single `_cache_lock`.
- **Ruff `target-version`**: aligned with project minimum (`py39` → `py310`).
- **Frontend download composable**: try/finally + `setTimeout(() => URL.revokeObjectURL(url), 0)`
  for Safari/WebKit URL revoke race fix.

### 🧪 Testing

- **Coverage threshold re-enabled** at 40% baseline; actual coverage now
  **55.81%** (was ~45%).
- **`norecursedirs` fix in `pyproject.toml`**: stray `phentrieve`/`api`
  entries were matching at every directory level and silently excluding
  `tests/unit/api/`, hiding 29 broken tests. Removing them exposed the
  failures, the bugs got fixed, and the test count jumped from 796 → **927**.
- **+44 Python characterization tests** pinning behavior through the
  `query_orchestrator`, `api/dependencies`, `api/main`, `hpo_extraction_orchestrator`,
  and `_create_chunkers` refactors.
- **+39 frontend tests** covering composables, stores, and components.
- **`@pytest.mark.parametrize` consolidation** across assertion_detection
  and hpo_parser tests.
- **Centralized fixtures**: `mock_sbert_model` fixture now lives in a
  shared conftest.
- **18 zero-assertion tests** fixed with meaningful assertions.
- **15 previously unmarked test files** now have `@pytest.mark.unit`.

### 📚 Documentation

- Pre-commit hooks (`ruff-check`, `ruff-format`, trailing whitespace,
  end-of-file, YAML, merge-conflicts, large-files) added via
  `.pre-commit-config.yaml`.
- TTLCache vs `@lru_cache` caching decisions documented in
  `docs/architecture/caching.md`.
- Review dashboard at `.planning/analysis/CODE-QUALITY-REVIEW-2026-04-09.md`
  updated with post-refactor status (all Critical findings closed).
- Phase 2 plan (`.planning/completed/2026-04-10-code-quality-phase-2.md`)
  and CI speedup plan (`.planning/active/CI-SPEEDUP-PLAN-2026-04-10.md`)
  saved for future reference.
- Stale deprecated-ranking references removed from docs and config.
- `AGENTS.md` notes the `pytest -n 0` escape hatch for single-threaded
  debugging when the default `-n auto` causes issues.

### 🧹 Chore

- `terser` devDependency removed from `frontend/package.json` after the
  Vite minifier switch (orphaned optional peer dependency).
- `Makefile` gains 6 new CI-parity targets: `format-check`, `test-ci`,
  `frontend-build-ci`, `frontend-format-check`, `frontend-test-ci`, `ci-local`.

---

## [0.12.1] — 2025-11-xx

Previous release — see git history for details. Notable work:
configuration resolver refactoring, HPO graph caching via `@lru_cache`,
SQLite migration for HPO data, model caching optimization, benchmark data
reorganization.

---

[0.13.0]: https://github.com/berntpopp/phentrieve/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/berntpopp/phentrieve/releases/tag/v0.12.1
