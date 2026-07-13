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
