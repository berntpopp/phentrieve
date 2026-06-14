# Phentrieve MCP Server — Assessment & Stabilization Plan

**Date:** 2026-06-14
**Author:** LLM-consumer assessment + senior-MCP-tester campaign + parallel codebase root-cause analysis
**Server under test:** local instance, `phentrieve-core 0.23.2` / MCP `0.15.3`, HPO `v2026-02-16`, LLM backend `gemini-3.1-flash-lite` (two-phase `whole_document_grounded`)
**Goal:** Ground every finding in the actual code and current best practices (Google Gemini structured output, Anthropic MCP tool design, clinical-NLP assertion/experiencer detection), then propose concrete, minimal changes that lift **every** quality dimension above 9/10.

> Status: analysis + proposals only. No code was modified to produce this document.

---

## 0. Executive summary

The Phentrieve MCP server is an unusually well-engineered protocol surface: discoverability, observability, caching, and GA4GH export fidelity are already reference-quality. Three classes of issue keep it off a consistent >9/10:

1. **Two functional correctness bugs** that cause a well-behaved client to draw a wrong conclusion or lose capability:
   - `no_high_confidence_match` returns `false` on a threshold-emptied result set (**B1**).
   - `chunk_text` hard-fails for 6 of its 7 advertised strategies instead of lazy-loading the embedding model that `search`/`extract` already load on demand (**B2**).
2. **LLM extraction-quality defects** (model/prompt layer, but fixable in-repo): negation mis-scoping ("X without Y" negates X) and missing experiencer modelling (family-history mentions become proband findings, producing a self-contradictory `present`+`negated` pair on one HPO id).
3. **A token-redundancy cluster** that multiplies the same term data up to four times across one extract→export response (**R1/R2**), plus several consistency/polish gaps (**B3, B4, D1–D4, Q1, R3**).

Every item below is traced to an exact `file:line`, has a minimal fix, and is mapped to the quality dimension it lifts. **None of the fixes are interdependent**; they can land as separate minimal commits. Several touch the capabilities descriptor or output schemas — by the server's own cache-key contract that intentionally rolls `capabilities_version`, which is the correct signal to warm clients.

### Score trajectory (MCP-only; model/prompt excluded per scoring rule)

| Dimension | Now | After plan | Lever |
|---|:--:|:--:|---|
| Discoverability | 9 | 10 | B4 (alias docs), honest `chunk_strategies` (B2) |
| Token efficiency | 7 | 9–10 | R1, R2, R3 |
| Speed / latency | 9 | 9 | unchanged (already strong) |
| Observability | 10 | 10 | maintained; D3 makes diagnostics live |
| Caching contract | 9 | 10 | maintained; version rolls only on intentional schema change |
| Schema ergonomics / errors | 8 | 10 | B3, D4 |
| Safety / compliance | 9 | 9–10 | maintained |
| **Per-tool: `chunk_text`** | **5** | **9** | B2 |
| **Per-tool: `diagnostics`** | **7** | **9** | D3 |
| **Overall (MCP-only)** | **9** | **>9.5** | full P0+P1 |

Extraction *accuracy* (a model/prompt property, scored separately) moves from ~6 → ~9 once the experiencer/negation work (LLM-1/LLM-2) lands.

---

## 1. Methodology & scope

- **Live test campaign:** 16 calls against the local server exercising all 8 tools — happy paths, error paths (`not_found`, `invalid_input`, `validation_failed`), all four `response_mode`s, both `compare` formulas, the two canonical round-trips (search→compare, extract→export), and the argument-alias contract.
- **Parallel codebase root-cause analysis:** four independent read-only agents — (A) architecture map, (B) functional-bug root cause, (C) token/consistency root cause, (D) LLM-pipeline + best-practices research. All claims below cite `file:line`.
- **Best-practices research:** official Google Gemini structured-output docs, Anthropic "writing tools for agents" guidance, and the clinical-NLP assertion/experiencer literature (ConText/NegEx, DEEPEN, modern assertion-detection LLMs).
- **Scope boundary:** the protocol surface and in-repo extraction pipeline. Retrieval-model accuracy (e.g. the "ASD → Autistic behavior" acronym collision) is noted but treated as embedding-model behavior, not a server defect.

---

## 2. Assessment A — LLM-consumer UX rating

### 2.1 Full experience (includes model/prompt output quality)

| Dimension | Score | Basis |
|---|:--:|---|
| Discoverability | 9/10 | `get_capabilities` + per-tool `next_tools`/`do_not_use_for` + pre-filled `next_commands` make routing unambiguous |
| Token efficiency | 8/10 | `response_mode` with explicit `char_budgets`; redundancy in `full` mode |
| Speed / latency | 8/10 | search ~31 ms, capabilities 0.6 ms; LLM extract ~13 s but fully disclosed |
| Observability | 10/10 | `request_id`, per-call `elapsed_ms`, `phase_timings`, match-method counts, token I/O |
| Output correctness | 6/10 | strong retrieval; assertion-scoping errors in the LLM path |
| Schema ergonomics / errors | 8/10 | alias layer + did-you-mean + enumerated `error_codes` |
| Safety / compliance | 9/10 | consistent `unsafe_for_clinical_use`, citation contract, prompt-injection guidance |
| **Overall** | **8/10** | exceptional protocol; correctness is the limiter |

### 2.2 MCP-only (model/prompt stripped — the fair score for the *server*)

| Dimension | Score | Why not 10 |
|---|:--:|---|
| Discoverability | 9/10 | — (reference quality) |
| Token efficiency | 7/10 | `full` payloads duplicate term data (R1/R2) |
| Speed / latency (server overhead) | 9/10 | sub-ms to ~350 ms for deterministic ops |
| Observability | 10/10 | best-in-class |
| Caching contract | 9/10 | `capabilities_version` cleanly separated from `descriptor_hash` |
| Schema ergonomics / errors | 8/10 | blank-text envelope misclassified (B3); alias/`additionalProperties` asymmetry (B4) |
| Safety / compliance | 9/10 | thorough |
| **Overall (MCP-only)** | **9/10** | payload shaping is the one contained weakness |

---

## 3. Assessment B — Senior-tester report

### 3.1 Coverage & per-tool verdict

| Tool | Cases run | Verdict | Score |
|---|---|:--:|:--:|
| `get_capabilities` | descriptor + `sample_calls`/`aliases` | ✅ reference-quality | 10 |
| `compare_hpo_terms` | related / identity / formula switch / not_found | ✅ clean & rich | 9 |
| `search_hpo_terms` | phrase / threshold boundary / nonsense / alias | ⚠️ flag bug (B1) | 8 |
| `extract_hpo_terms` (det.) | happy / negation / k=1 vs k=5 / minimal / blank | ⚠️ default + envelope (Q1/B3) | 8 |
| `extract_hpo_terms_llm` | full abstract | ✅ protocol; ⚠️ accuracy (LLM-1/2) | 8 |
| `export_phenopacket` | round-trip + subject + sidecar | ⚠️ double-serialized (R1) | 8 |
| `diagnostics` | health snapshot | ⚠️ stale lazy state (D3) | 7 |
| `chunk_text` | simple / detailed / sliding_window | ❌ 1 of 7 strategies usable (B2) | 5 |
| **Overall (as tested)** | | | **8** |

### 3.2 What is genuinely strong (must be preserved)

- **Structured error envelopes** — `error_code`, `message`, `retryable`, `recovery_action`, `field`, `next_commands` — consistent and machine-actionable.
- **`export_phenopacket`** — correct GA4GH v2: negated → `"excluded": true`, ECO evidence codes, pinned HPO version, intended-use disclaimer embedded, provenance sidecar.
- **`compare`** — identity = 1.0, full LCA/MICA detail, working formula switch, `next_commands` that suggests the alternate formula.
- **Deterministic `extract`** — auditable `assertion_details` (dependency parser + keyword negation scopes); correctly negated "no microcephaly".
- **`response_mode` budgets honored**; `minimal` is genuinely lean.
- **`capabilities_version` stable across all 16 calls** — a reliable warm-cache key.

---

## 4. Root-cause analysis (grounded)

Architecture entrypoints for reference: MCP server built in `api/mcp/facade.py:59` (`create_phentrieve_mcp`), tool bodies wrapped by `run_mcp_tool` (`api/mcp/envelope.py:158`); tools in `api/mcp/tools/{retrieval,similarity,phenopacket,discovery}.py`; service adapters in `api/mcp/service_adapters.py`; response shaping in `api/mcp/shaping.py`; `next_commands` in `api/mcp/next_commands.py`; capabilities/aliases in `api/mcp/capabilities.py` + `api/mcp/arg_help.py`; LLM pipeline in `phentrieve/llm/pipeline.py` with prompts under `phentrieve/llm/prompts/templates/two_phase/`.

### P0 — Functional correctness

#### B1 — `no_high_confidence_match` is `false` on threshold-emptied results
- **Root cause:** `api/mcp/confidence.py:42` — `out["no_high_confidence_match"] = bool(results) and top < HIGH_FLOOR`. The `bool(results)` guard short-circuits to `False` on an empty set, which is semantically inverted (an empty set is the *strongest* "no high-confidence match" signal).
- **Why:** `similarity_threshold` is applied inside retrieval (`phentrieve/retrieval/api_helpers.py:44,83`) before the confidence layer. At `0.99`, gibberish hits (~0.55) are filtered out, the adapter returns `{"results": []}`, and `bool([]) and …` → `False`. At the default threshold the hits survive, so the flag is correctly `True` — exactly the asymmetry observed.
- **Fix (band-based, future-proof against truncation):**
  ```python
  out["no_high_confidence_match"] = not any(
      r.get("confidence_band") == "high" for r in results
  )
  ```
- **Risk:** Low. Only changes the two currently-wrong cases. Grep `tests/` for an assertion pinning `false` on empty results — it would encode the bug.

#### B2 — `chunk_text` hard-fails model-dependent strategies instead of lazy-loading
- **Root cause:** `api/mcp/service_adapters.py:481` builds the pipeline with `sbert_model_for_semantic_chunking=None`; any strategy whose config contains a `sliding_window` stage raises `ValueError` in `phentrieve/text_processing/_chunker_registry.py:55-59`, caught at `service_adapters.py:485-491` → `invalid_input`.
- **Premise correction:** `detailed` genuinely needs the model — `DETAILED_CHUNKING_CONFIG` (`phentrieve/config.py:165-178`) includes a `sliding_window` stage. **6 of 7 strategies require the model; only `simple` is model-free.** The schema enum and `capabilities.chunk_strategies` (`api/mcp/capabilities.py:134`) advertise all 7 with no model-dependency marker, so the surface over-promises.
- **Why the inconsistency:** `extract` lazy-loads via `phentrieve/text_processing/full_text_service.py:615-616` (`load_embedding_model`, a thread-safe cached singleton — the same instance search/extract warm). `chunk_text` simply never calls it.
- **Fix (recommended — option a, restore parity):** detect the need from the resolved config and lazy-load the cached singleton.
  ```python
  needs_model = any(isinstance(s, dict) and s.get("type") == "sliding_window" for s in config)
  sbert_model = load_embedding_model(DEFAULT_MODEL) if needs_model else None
  pipeline = TextProcessingPipeline(..., sbert_model_for_semantic_chunking=sbert_model)
  ```
  Narrow the broad `except` so a true load failure surfaces as `temporarily_unavailable`/`internal_error`, not `invalid_input`.
- **Fix (option b — honest surface, if `chunk_text` must stay zero-cost):** make `capabilities.chunk_strategies` a `{model_free:[...], model_dependent:[...]}` map, mark the `ChunkStrategy` field doc (`api/mcp/tools/_common.py:42-44`), and have the error enumerate the model-free strategy precisely. Rolls `capabilities_version` (intended).
- **Recommendation:** option (a) — matches the documented lazy-load latency contract and removes a real functional gap. Lifts `chunk_text` 5 → 9.

#### LLM-1 / LLM-2 — assertion-scoping & experiencer (model/prompt layer, fixable in-repo)
- **Single underlying cause:** the Phase-1 schema collapses *assertion* (present/absent) and *experiencer* (proband/family) into one `category` enum (`phentrieve/llm/types.py:90-96`; prompt `phentrieve/llm/prompts/templates/two_phase/en.yaml:15-20`), and there is **no per-phrase negation-scope signal**.
- **"X without Y" bug:** no negation-scope guidance in the rules (`en.yaml:22-32`); additionally the LLM path's coarse override `_project_llm_term_status_from_chunks` (`phentrieve/text_processing/full_text_service.py:355-370`, applied `:480-484`) flips an LLM `present` to `negated` whenever *all* source chunks are NegEx-negated (`phentrieve/llm/preprocessing.py:21-50`), **without** the span-overlap check the standard path already has (`phentrieve/text_processing/_hpo_extraction_helpers.py:182-209`).
- **Family-history bug:** `family_history` is in `ACTIONABLE_CATEGORIES` (`phentrieve/llm/pipeline_phase1.py:14`), so it is retrieved and mapped to a proband HPO term, then folded to `affirmed`/`negated` at the MCP boundary (`api/mcp/service_adapters.py:380`). De-dup is keyed only on `(term_id, assertion)` (`pipeline_phase2.py:339`, `api/mcp/projection.py:22`), so one HPO id can appear as both `present` and `negated`.
- **Fix (schema):** split `category` into orthogonal enums `experiencer ∈ {proband, family_history, other}` and `assertion ∈ {present, absent, uncertain}`, plus an optional `negated_qualifier` for "X without Y". Declare fields in reasoning order (evidence → experiencer → assertion → qualifier → phrase) since the Gemini schema is generated from Pydantic `model_json_schema()` (`phentrieve/llm/providers/base.py:247-252`) and Gemini emits keys in schema order.
- **Fix (prompt):** add a two-step rule ("decide experiencer, then assertion, independently"), a negation-scope rule ("a cue negates only the noun phrase it directly modifies; in 'X without Y', X is present and only Y absent"), and 2–3 few-shot examples covering "X without Y" and "no family history of Z". Bump prompt `version`.
- **Fix (pipeline):** (1) remove `"family_history"` from `ACTIONABLE_CATEGORIES` so relatives' mentions never become proband terms (minimal, highest-confidence), OR carry `experiencer` to output and project family-history into a separate list; (2) change the aggregation/dedup key to `(term_id, experiencer, assertion)`; (3) for the LLM backend, trust the per-phrase `assertion` (drop the coarse chunk-status override) or gate it through span overlap; (4) `_coerce_export_phenotype` must not fold `family_history` into a proband `PhenotypicFeature`.

### P1 — Token efficiency & DX

#### R1 — `export_phenopacket` emits the packet twice
- **Root cause:** `api/mcp/service_adapters.py:429-442` adds a parsed `phenopacket` object alongside the declared `phenopacket_json` string (comment marks it intentional "backwards compatibility"). `PHENOPACKET_SCHEMA` (`api/mcp/schemas.py:55-58`) only declares `phenopacket_json` — `phenopacket` survives solely because output schemas are `additionalProperties:true`.
- **Fix:** make the parsed object canonical; add `phenopacket_json` to `shaping._DETAIL_FIELDS` so the serialized blob appears only at `standard`/`full`. GA4GH consumers want one canonical document. Gate rather than delete to avoid breaking a client reading `phenopacket_json`.

#### R2 — `next_commands` re-serializes the full term list, even at `minimal`
- **Root cause:** `api/mcp/next_commands.py:51-73` (`after_extract`) maps up to 25 aggregated terms into 4-field objects for the export pre-fill. `_meta` is exempt from shaping (`api/mcp/shaping.py:117-118`; `enforce_budget` only trims the named list field), so the pre-fill survives `minimal`.
- **Fix:** pass `mode` into `after_extract`; cap at ~5 terms under `minimal`/`compact` and keep `{hpo_id, assertion, score}` (still directly executable via `_coerce_export_phenotype`; keep `score` to avoid 0.0-confidence export). Do **not** revert to a free-text placeholder (a prior defect).
- **Cross-cutting:** R1 + R2 + a populated D1 multiply the *same* term data across one response (`aggregated_hpo_terms` + pre-fill + phenopacket(+json)). Fixing R1 and R2 together yields the largest token win for the common extract→export path.

#### B3 — blank `text` returns the unknown-argument-NAME envelope
- **Root cause:** `_reject_blank_text` (`api/mcp/tools/_common.py:10-14`) raises a `ValueError` that pydantic wraps as a `value_error` on a *valid* arg; `describe_constraints` (`api/mcp/arg_help.py:92-119`) only understands enum/numeric bounds, returns `None` for `minLength`, so `build_arg_error_envelope` (`api/mcp/envelope.py:250-268`) falls into the *name-error* template — emitting `allowed_values` = parameter names and discarding the real message.
- **Fix:** thread the validator's message through `middleware.py._error_result` and add a value-message branch to `build_arg_error_envelope` that returns `Invalid value for argument 'text' …: text must not be empty/whitespace.` with the signature `hint` but **no** `allowed_values=param-names`.

#### D4 — `recovery_action: "reformulate_input"` for `not_found`
- **Root cause:** `api/mcp/envelope.py:49-59` maps `not_found → reformulate_input`. A bogus-but-well-formed HPO id should be *resolved*, not reformulated (a compare call has no free text).
- **Fix:** map `not_found` (and `ambiguous_query`) → `resolve_identifier`, and point error `next_commands` at `phentrieve_search_hpo_terms` instead of the generic capabilities/diagnostics. New enum value is schema-safe (`recovery_action` is free-form `_STR`); grep tests switching on the old set.

#### D3 — `diagnostics` reports static `"lazy"` even when warm
- **Root cause:** `api/mcp/service_adapters.py:520-525` hardcodes `embedding_model:"lazy"`, `vector_index:"lazy"`, `llm_backend:"configured"`; only `ontology_data` is probed.
- **Fix:** probe the live caches in `api/dependencies.py` (`LOADED_SBERT_MODELS`, `MODEL_LOADING_STATUS`, `LOADED_RETRIEVERS`) and report `loaded|loading|cold|error`. Membership tests trigger no load and need no lock. Wrap in try/except (diagnostics must never raise). Lifts `diagnostics` 7 → 9 and makes cold-start predictable for clients.

### P2 — Consistency & polish

#### D1 — `text_attributions` inconsistently present (`[]` / missing / populated)
- **Root cause:** literal label/synonym matching (`phentrieve/retrieval/text_attribution.py:18-112`) yields `[]` for semantically-retrieved terms with no verbatim span; LLM path drops un-offsettable records (`full_text_service.py:309-352`). At `compact`, empty detail fields are dropped (`shaping.py:33,82-88`) so the key vanishes — hence "missing".
- **Fix:** adopt one contract — *always present as an array; empty = semantic match, no literal span.* Add `text_attributions` to `shaping._ALWAYS_KEEP_EMPTY` (currently only `("hpo_matches",)`, `shaping.py:55`) and document the empty-means-semantic semantics in `api/mcp/projection.py`.

#### D2 — `compare.ic_proxy` is normalized depth, not corpus IC
- **Root cause:** `api/mcp/service_adapters.py:286-289` computes `ic_proxy = depth / max_depth`. With `max_depth==14`, every depth-7 leaf yields exactly `0.5` (verified against `data/hpo_data.db`: 4194 terms at depth 7). Real Resnik IC (`-log p(t)`) needs a corpus the bundle doesn't ship (`HPODatabase` exposes only structural data).
- **Fix (recommended — honest relabel):** rename the key to `normalized_depth` (or add `ic_proxy_method: "normalized_depth"` on `lca_details`) and change the `compare` description (`api/mcp/tools/similarity.py:31-38`) to "normalized ontology depth (structural proxy, not corpus information content)". Optional follow-up: compute Seco intrinsic IC from the already-loaded `ancestors` map (cache it) — strictly better than raw depth, still not corpus IC.

#### R3 — `include_details=true` dumps uncapped synonym lists
- **Root cause:** `_DETAIL_KEEP=("definition","synonyms")` (`api/mcp/tools/retrieval.py:50`); synonyms sourced uncapped (`phentrieve/text_processing/_hpo_extraction_helpers.py:311`; `phentrieve/retrieval/api_helpers.py:286-289`).
- **Fix:** cap synonyms in the *response* projection (`api/mcp/projection.py:44`) to N (e.g. 10) with a `synonyms_truncated` count — never cap the list used for attribution matching. Optionally split `include_synonyms` from `include_details`.

#### Q1 — deterministic `extract` default `num_results_per_chunk=1`
- **Root cause:** `api/mcp/tools/_common.py:20` (`DEFAULT_EXTRACT_NUM_RESULTS=1`); the LLM tool defaults to 10. With k=1, a diluted/long chunk's top-1 can be a *parent* term (`HP:0011994`, 0.88), so the exact child (`HP:0001631`, 0.99997 in isolation) never enters the evidence map and is lost — a *wrong* top term, not merely fewer.
- **Fix:** raise the default to 3 (`DEFAULT_EXTRACT_NUM_RESULTS=3`). Aggregation already prunes by `chunk_retrieval_threshold` (0.7) and ranks/de-dups, so the final payload grows only modestly while precision improves. Minimum acceptable alternative: document the dilution risk in the tool description.

#### B4 — alias contract is client-dependent and mis-documented
- **Root cause:** input schemas declare `additionalProperties:false`, and `capabilities.argument_alias_policy` (`api/mcp/capabilities.py:158-168`) claims strict clients reject aliases pre-server. In practice the Claude Code client passed `limit` through and the server applied `limit→num_results` (`api/mcp/middleware.py:66`, disclosed via `_meta.argument_aliases_applied`). Behavior is therefore client-dependent.
- **Fix:** reconcile the docs — state that alias rewriting depends on client strictness and that canonical names are always safe; consider whether to keep advertising aliases given `additionalProperties:false`. Documentation-only; lifts discoverability honesty to 10.

---

## 5. Best-practices alignment

### Google Gemini structured output
- **Enum every categorical field** to prevent invalid classifications → make `experiencer`/`assertion` enums, not a free-text `category`. (https://ai.google.dev/gemini-api/docs/structured-output)
- **Property order controls generation order; put reasoning before labels.** Schema is generated from Pydantic field order (`phentrieve/llm/providers/base.py:247-252`) → declare evidence → experiencer → assertion → qualifier → phrase. (https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/)
- **Use `description` fields heavily; mark required.** → directive descriptions on the new fields; `experiencer`/`assertion` required. (Gemini docs)
- **Schema validity ≠ semantic validity — validate downstream.** → keep span-overlap negation refinement as a server-side gate, don't trust the label blindly. (Gemini docs)
- Server already uses `response_mime_type="application/json"` + JSON schema (`phentrieve/llm/providers/gemini.py:152,166-167`) — additions will be enforced; no provider change needed.

### Anthropic MCP tool design (https://www.anthropic.com/engineering/writing-tools-for-agents)
- **Return contextually meaningful, non-conflated fields** → split `assertion` from `experiencer` so the agent can reason instead of decoding an overloaded `category`.
- **Actionable signals over opaque coercion** → label family-history explicitly rather than silently folding to `affirmed`.
- **`response_format`/verbosity enum for token control** → already implemented as `response_mode`; keep experiencer detail out of `minimal`/`compact`.
- **Consolidate; keep results token-efficient** → R1/R2 and `(term_id, experiencer, assertion)` keying remove duplicate/contradictory term copies.

### Clinical NLP (assertion + experiencer)
- **ConText/NegEx separate Negation, Temporality, and Experiencer (patient|family|other)** — the exact axis separation Phentrieve currently lacks. (http://toolfinder.chpc.utah.edu/content/contextnegex)
- **NegEx mis-scopes negation in complex sentences; DEEPEN adds dependency awareness** — the "without regression" failure mode. (https://pmc.ncbi.nlm.nih.gov/articles/PMC5863758/)
- **Modern assertion-detection models use explicit Present/Absent/Hypothetical + family experiencer labels** — supports explicit schema fields + few-shot over a downstream rule engine. (https://arxiv.org/html/2503.17425v1)

---

## 6. Stabilization roadmap to >9/10

Ordered by impact; each item is independent.

**P0 — correctness (ship first):**
1. **B1** — fix `no_high_confidence_match` (band-based). _search 8→9._
2. **B2** — lazy-load model in `chunk_text` (option a). _chunk_text 5→9._
3. **LLM-1** — drop `family_history` from `ACTIONABLE_CATEGORIES` + key dedup on `(term_id, experiencer, assertion)`. _kills the contradictory pair._
4. **LLM-2** — add `experiencer`/`assertion`/`negated_qualifier` to the Phase-1 schema, update prompt + few-shots, stop the coarse chunk-status override for the LLM path. _accuracy ~6→9._

**P1 — token efficiency & DX:**
5. **R1** — single canonical phenopacket form; gate `phenopacket_json` to standard/full.
6. **R2** — cap/compact the `after_extract` pre-fill; suppress under minimal/compact.
7. **B3** — value-level validation envelope for blank `text`.
8. **D4** — accurate `recovery_action` + `search` next-command for `not_found`.
9. **D3** — live subsystem probing in `diagnostics`.

**P2 — consistency & polish:**
10. **D1** — `text_attributions` always-present contract (`_ALWAYS_KEEP_EMPTY`).
11. **D2** — relabel `ic_proxy` → `normalized_depth` (+ honest description).
12. **R3** — synonym cap (and/or split `include_synonyms`).
13. **Q1** — `DEFAULT_EXTRACT_NUM_RESULTS = 3`.
14. **B4** — reconcile alias docs with real client-dependent behavior.

**Capabilities-version note:** items B2(b), D2, R3(split), and any schema-shape change intentionally roll `capabilities_version` — the correct warm-client signal per `api/mcp/capabilities.py:190-197`.

---

## 7. Verification plan

Per repo `AGENTS.md` and the team's CI-parity expectations, each change ships with coverage-improving tests for the touched code:

- **Required gates:** `make check`, `make typecheck-fast`, `make test`; before push, `make ci-local` + `make security-python`.
- **New/updated unit tests (under `tests/`, no `tests_new/`):**
  - B1 — `no_high_confidence_match=true` for empty results AND for a non-high top hit; `false` only when a `high`-band hit exists.
  - B2 — `chunk_text(strategy="sliding_window"|"detailed")` returns chunks after lazy-load; first cold call works; cached on subsequent.
  - B3 — blank/whitespace `text` → `validation_failed` with a value-level message and **no** `allowed_values`=param-names.
  - D3 — diagnostics reports `loaded` after a search/warmup and `cold` before.
  - D4 — `not_found` envelope carries `recovery_action="resolve_identifier"` and a `search` next-command.
  - R1 — compact/standard responses contain exactly one canonical phenopacket form.
  - R2 — `minimal` extract response's `next_commands` pre-fill is capped and free of the full 25-term copy.
  - D1 — `text_attributions` present as `[]` at compact for a semantic-only match.
  - LLM-1/2 — golden cases: "severe intellectual disability without regression" → ID present, qualifier=regression; "no familial history for long QT" → experiencer=family_history (dropped/flagged), no proband `present`+`negated` pair on `HP:0001657`.
  - Drift guards: update `chunk_strategies`/enum tests and any `capabilities_version`-pinned snapshot to reflect intentional rolls.

---

## 8. Appendix

### 8.1 Finding → file:line index
| ID | Dimension | Primary site |
|---|---|---|
| B1 | search correctness | `api/mcp/confidence.py:42` |
| B2 | chunk_text capability | `api/mcp/service_adapters.py:481`; `phentrieve/config.py:165-178`; `_chunker_registry.py:55-59` |
| B3 | error ergonomics | `api/mcp/tools/_common.py:10-14`; `api/mcp/arg_help.py:92-119`; `api/mcp/envelope.py:250-268` |
| B4 | discoverability/docs | `api/mcp/capabilities.py:158-168`; `api/mcp/middleware.py:66` |
| R1 | token efficiency | `api/mcp/service_adapters.py:429-442`; `api/mcp/schemas.py:55-58` |
| R2 | token efficiency | `api/mcp/next_commands.py:51-73`; `api/mcp/shaping.py:117-118` |
| R3 | token efficiency | `api/mcp/tools/retrieval.py:50`; `_hpo_extraction_helpers.py:311` |
| D1 | consistency | `phentrieve/retrieval/text_attribution.py:18-112`; `api/mcp/shaping.py:33,55` |
| D2 | compare transparency | `api/mcp/service_adapters.py:286-289` |
| D3 | observability | `api/mcp/service_adapters.py:520-525`; `api/dependencies.py:28-33` |
| D4 | error ergonomics | `api/mcp/envelope.py:49-59` |
| Q1 | extract precision | `api/mcp/tools/_common.py:20`; `api/mcp/tools/retrieval.py:146` |
| LLM-1/2 | extraction accuracy | `phentrieve/llm/types.py:90-96`; `prompts/templates/two_phase/en.yaml`; `pipeline_phase1.py:14`; `full_text_service.py:355-370,480-484` |

### 8.2 Sources
- Gemini structured output — https://ai.google.dev/gemini-api/docs/structured-output
- Gemini implicit property ordering — https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/
- Anthropic, Writing effective tools for AI agents — https://www.anthropic.com/engineering/writing-tools-for-agents
- ConText/NegEx — http://toolfinder.chpc.utah.edu/content/contextnegex
- DEEPEN dependency-aware negation — https://pmc.ncbi.nlm.nih.gov/articles/PMC5863758/
- Beyond Negation Detection (assertion-detection LLMs) — https://arxiv.org/html/2503.17425v1
