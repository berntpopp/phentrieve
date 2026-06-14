# Phentrieve MCP Hardening -- Design Spec

- Date: 2026-06-14
- Author: MCP engineering pass (brainstormed from the LLM-consumer evaluation)
- Source evaluation: `.planning/analysis/2026-06-14-phentrieve-mcp-evaluation.md`
- Goal: raise every evaluation dimension to >= 9.5/10 in line with current
  Anthropic MCP / tool-use guidance and Google Gemini function-calling guidance.

## 1. Scope and decisions

Three confirmed product decisions shape this spec:

1. **Schema/token cleanup lands at the MCP boundary** (a projection layer in
   `api/mcp/`), not in the shared `full_text_service`. The redundant score
   fields and dual chunk-index schemes are produced by the shared service that
   also powers the REST API (`AggregatedHPOTermAPI`) and the Vue curation
   frontend (`AggregatedTermsView.vue` reads `max_score_from_evidence` and
   `confidence`). Normalizing only for the MCP consumer keeps the REST/frontend
   contract intact.
2. **LLM (Gemini) negation fix is in scope but benchmark-gated.** Prompt changes
   have regressed mapping benchmarks before (PR #261). The fix ships only if a
   before/after benchmark shows no mapping-accuracy regression; otherwise the LLM
   change alone is reverted.
3. **Verification includes a live re-test.** Reproduce each defect against the
   running MCP, apply fixes, restart `make mcp-serve-http`, and re-run the tool
   matrix, in addition to unit/integration tests and `make ci-local`.

### 1.1 Layering principle

- **Correctness -> shared pipeline (`phentrieve/`).** Negation must be correct
  for every consumer. C1, H2, and L5-at-source live here.
- **Contract and token shape -> MCP boundary (`api/mcp/`).** A thin, additive
  projection normalizes the schema for the MCP consumer only: H1 default, H3,
  M2, M3, M4, T1, L5-serialization, L7-collision, M1, M5, L4, L8, L1, L2.
- **LLM negation -> `phentrieve/llm/`** (benchmark-gated): the deterministic-LLM
  negation-scope fix (evaluation L7 and the LLM half of C1).

### 1.2 Best-practices grounding (researched 2026-06-14)

- **Token efficiency:** Anthropic "Writing effective tools for agents"
  (2025-09-11) and "Code execution with MCP" (2025-11-04): return high-signal
  data only, avoid duplicate/low-signal fields and empty placeholders, use a
  verbosity enum (our `response_mode` is the analogue), pair truncation with
  steering text. Directly supports dropping the 4 duplicate scores and ~30 empty
  chunks.
- **Errors:** MCP spec (2025-11-25) -- execution errors returned as results the
  model can self-correct from; never leak raw exception strings; show a
  correctly-formatted example. Supports M3 and L1.
- **Enums / `additionalProperties`:** MCP does not define a "strict" mode;
  `additionalProperties:false` is Anthropic-provider strict-mode behavior, not an
  MCP requirement. Argument aliases are an anti-pattern (Anthropic advises one
  unambiguous canonical name). Supports L4 and the L8 "document, don't expand"
  posture.
- **Capabilities versioning:** the content-hash capabilities tool is a *custom*
  convention; the MCP spec mechanism is `tools/list_changed` (+ draft TTL hints).
  M1 fix = stable base hash + honest documentation.
- **outputSchema:** when declared, servers MUST conform and clients SHOULD
  validate; trimming is safe only for non-required fields or permissive schemas.
  Our `EXTRACT_SCHEMA` is permissive (`additionalProperties:true`), so the
  projection is spec-safe.
- **Gemini negation:** for *linguistic* errors (scope/polarity) the fix is
  few-shot contrastive examples + an in-schema reasoning step ordered before the
  enum, not tighter schema constraints. `propertyOrdering` honoring varies by
  `google-genai` SDK version and must be verified. Temperature 0.0 is fine.

## 2. Workstreams

Each item: defect id, root cause (file:line), fix, layer, dimension moved.

### WS1 -- Deterministic extractor correctness (Correctness 6 -> 9.5)

**C1 (CRITICAL) -- negation scope lost.**
- Root cause: `phentrieve/text_processing/pipeline.py:265-267` calls
  `assertion_detector.detect(cleaned_final_chunk)`; `FinalChunkCleaner`
  (`chunkers.py:317-394`, `leading_cleanup_words.json`) strips leading negation
  cues, and the chunk offset (`spans.find_span_in_text`) jumps past the cue.
- Fix: run assertion detection over the **source-sentence span taken from the
  original text** (located via the chunk's char offsets, expanded to sentence
  boundaries), passing the cleaned chunk as the target concept. Keep the cleaned
  text as the retrieval query and display text. Do not strip negation triggers
  from the span fed to the detector.
- Layer: shared.
- Tests: `"There is no nystagmus"` -> negated; `"She does not have ataxia"` ->
  negated; `"patient denies headache"` -> negated; **`"severe intellectual
  disability without regression"` -> intellectual disability affirmed**,
  regression negated. Add to `tests/unit/text_processing/`.

**H2 (HIGH) -- negated findings dropped.**
- Root cause: to be pinned by a failing test. `process_chunk_matches`
  (`_hpo_extraction_helpers.py:82-88`) always emits a `matches` key and
  `aggregate_and_rank` keeps negated terms via the status counter, so the most
  likely drop sites are (a) negated chunks skipped before retrieval in the
  orchestrator, or (b) response-mode dropping empty lists. Write the failing test
  first, then fix at the identified site.
- Fix: ensure negated chunks still retrieve and emit matches with
  `assertion_status: negated`, so `excluded: true` phenopacket features can be
  built deterministically.
- Layer: shared (retrieval/aggregation) + MCP (serialization, see L5).
- Tests: `"patient denies headache"` yields a Headache term with
  `assertion_status: negated` in both `processed_chunks` and
  `aggregated_hpo_terms`.

**H1 (HIGH) -- candidate explosion.**
- Root cause: MCP extract tools pass `num_results_per_chunk=DEFAULT_NUM_RESULTS`
  (shared with search, where many results is correct) and never collapse;
  `process_chunk_matches` already supports `top_term_per_chunk`
  (`_hpo_extraction_helpers.py:79`).
- Fix: introduce `DEFAULT_EXTRACT_NUM_RESULTS = 1` used by both extract tools
  (best match per phrase). Keep `num_results_per_chunk` exposed so raising it
  surfaces sibling candidates. Update the tool descriptions to say "candidates
  per phrase (default 1 = best match)." Do not change `search`'s default.
- Layer: MCP.
- Tests: `"The patient had seizures."` -> exactly 1 aggregated term (Seizure);
  `num_results_per_chunk=5` -> up to 5.

### WS2 -- Lossless, pipeable export (Output/schema 5 -> 9.5)

**H3 (HIGH) -- export loses confidence (`0.0000`).**
- Root cause: `api/mcp/next_commands.py:30-36` omits `score`;
  `api/mcp/service_adapters.py:302-310` builds `ExportPhenotypeRequest` without
  score; `phentrieve/phenopackets/utils.py:351` defaults `confidence` to 0.0.
- Fix: include `score` in `after_extract` phenotype objects; add a `score`/
  `confidence` field to `ExportPhenotypeRequest`; thread it into the evidence
  reference so the phenopacket records the real retrieval confidence.
- Layer: MCP.
- Tests: piping `aggregated_hpo_terms` into export yields a non-zero confidence
  in the evidence reference.

**M2 (MEDIUM) -- extract->export key mismatch.**
- Root cause: `service_adapters.py:304` accesses `p["hpo_id"]` directly; extractor
  emits `{id, name, assertion_status}`.
- Fix: accept both key sets via `.get()` fallbacks (`hpo_id` or `id`; `label` or
  `name`; `assertion` or `assertion_status` or `status`). Raw
  `aggregated_hpo_terms` must pipe directly.
- Layer: MCP.
- Tests: `export_phenopacket(phenotypes=[{id,name,assertion_status,score}])`
  succeeds.

**M3 (MEDIUM) -- raw KeyError leak.**
- Root cause: `api/mcp/envelope.py:103,109,123` maps `KeyError` to
  `invalid_input` with `str(exc)` = bare repr (`"'hpo_id'"`).
- Fix: with M2's `.get()`, no KeyError escapes; additionally raise an explicit
  `McpToolError("validation_failed", "phenotypes[i] missing 'hpo_id' (got keys:
  ...); map id->hpo_id, name->label, assertion_status->assertion")` when neither
  id key is present. Ensure no bare exception repr is ever the message.
- Layer: MCP.
- Tests: malformed phenotype -> `validation_failed` with actionable message.

**M4 (MEDIUM) -- redundant schema / dual indexing.**
- Root cause: `full_text_service._adapt_aggregated_terms` (213-250) emits four
  score copies (`score`/`avg_score`/`confidence`/`max_score_from_evidence`) and
  two index schemes (`chunks`+`top_evidence_chunk_idx` 0-based vs
  `source_chunk_ids`+`top_evidence_chunk_id` 1-based).
- Fix: new `api/mcp/projection.py` `project_aggregated_terms_for_mcp()` that, for
  the MCP consumer only, keeps a single `score` (max evidence score), a single
  chunk-reference scheme (`chunk_id` 1-based + `top_evidence_chunk_id`), and
  consistent offsets in `evidence_records` (populate from `text_attributions` or
  drop null `start_char`/`end_char`). Shared service output is untouched.
- Layer: MCP.
- Tests: projected term has exactly one score field and one index scheme.

**T1 -- token efficiency.**
- Root cause: `full`-mode carries empty `processed_chunks` and duplicate scores;
  `next_commands` re-emits term data already in `aggregated_hpo_terms`.
- Fix: `project_processed_chunks_for_mcp()` drops empty-match chunks unless
  `include_unmatched_chunks=true`; score dedup via M4; keep `next_commands`
  export payload slim (`hpo_id, label, assertion, score`, capped) and do not echo
  full term objects.
- Layer: MCP.
- Tests: `full`-mode payload contains no empty chunks and no duplicate score
  fields; char-count regression assertion.

**L5 -- `hpo_matches` omitted on negated/empty chunks.**
- Root cause: `apply_response_mode`/`_shape_item` drops empty values in
  compact/standard.
- Fix: always serialize `hpo_matches: []` (whitelist the key against empty-drop
  in the MCP projection/shaping).
- Layer: MCP.
- Tests: a chunk with no matches still has `hpo_matches: []`.

**L7-collision -- duplicate `hpo_id` (present + negated).**
- Fix: document that aggregated terms are keyed by `(hpo_id, assertion)`; in the
  projection, keep both entries but ensure the pair is explicit so a consumer
  keying by `hpo_id` alone does not silently collapse them. (The LLM
  over-negation that creates the spurious pair is addressed in WS6.)
- Layer: MCP.

**L6 -- provenance version mismatch.**
- Root cause: `phenopackets/utils.py:487` writes `createdBy: phentrieve
  {core_version}` (0.23.1) while the MCP server reports 0.15.1.
- Fix: write both versions, e.g. `Phentrieve (core X.Y.Z, mcp-server A.B.C)`;
  thread the server version into export meta.
- Layer: MCP + shared (signature).

### WS3 -- Cache contract and discoverability (Discoverability 8 -> 9.5)

**M1 -- unstable `capabilities_version`.**
- Root cause: `api/mcp/capabilities.py:183-201` stamps the detailed descriptor's
  own hash into `capabilities_version`, but `_meta` always uses the base hash.
- Fix: `_cached_descriptor` always sets `capabilities_version` to the **base**
  hash (`capabilities_version()`), so base and detailed both match `_meta`. Add a
  separate `descriptor_hash` for the actual serialized content if a client wants
  it. Document in capabilities that `capabilities_version` is a custom warm-cache
  convention and `tools/list_changed` is the spec mechanism.
- Layer: MCP.
- Tests: base and `details=[...]` responses both report the same
  `capabilities_version`, equal to `_meta.capabilities_version`.

**L4 -- `strategy` has no enum.**
- Root cause: `retrieval.py:245` `strategy: str | None`.
- Fix: `ChunkStrategy = Literal[...valid strategies...]` in
  `api/mcp/tools/_common.py` (sourced from
  `text_processing/config_resolver.py:195-203`); reject unknown values with the
  valid list; enumerate valid strategies in capabilities.
- Layer: MCP.
- Tests: unknown strategy -> `validation_failed` with allowed_values; valid
  strategies listed in capabilities.

**M5 -- `include_details=true` is a no-op in compact.**
- Root cause: details only surface at standard/full while the default is
  `include_details=true` + `response_mode=compact`.
- Fix: default `include_details=false`; when explicitly `true`, keep
  `definition`/`synonyms` for matched terms even in compact (honor the flag).
  Document the response-mode detail floor.
- Layer: MCP.
- Tests: `include_details=true` in compact returns definitions; default compact
  does not.

**L8 -- alias reachability under strict schemas.**
- Fix: document in the capabilities `argument_aliases` section that canonical
  names are authoritative and aliases are a convenience for non-strict clients
  (validating/strict clients must use canonical names). Do not advertise aliases
  as a guaranteed feature. Verify the actual generated `inputSchema`
  `additionalProperties` value and state it accurately.
- Layer: MCP (docs/capabilities).

**Cold-start hint.** Add "tools may load on demand; start with
`phentrieve_get_capabilities`" to the server instructions.

### WS4 -- Error handling polish (8 -> 9.5)

**L1 -- empty/whitespace query accepted.**
- Root cause: `api/mcp/tools/_common.py:14-21` `TextArg` has no `min_length`.
- Fix: add `min_length=1` and reject whitespace-only text with
  `validation_failed`.
- Layer: MCP.
- Tests: `search("")` and `search("   ")` -> `validation_failed`.

**L2 -- non-executable `next_commands` placeholders.**
- Root cause: `next_commands.py:25` (`text="<surrounding clinical text>"`) and
  `:49` (`text="<related phenotype phrase>"`).
- Fix: emit executable steps (e.g., after compare, search the resolved label of a
  compared term) or omit the placeholder entirely.
- Layer: MCP.
- Tests: no `next_commands` entry contains an unfilled `<...>` placeholder.

**M3** -- see WS2.

### WS5 -- Observability, speed, safety (9 -> 9.5 each)

- **Observability:** add per-phase timing (`chunking_ms`, `phase1_ms`,
  `phase2_ms`, `phase2b_ms`) to the LLM `observability` block using
  `time.perf_counter()` around pipeline phases. Additive fields; safe for all
  consumers. (`full_text_service.py` LLM path + `phentrieve/llm`.)
- **Speed:** best-effort warmup of the lazy embedding model + vector index on MCP
  server startup (or a warmup action); document expected LLM-path latency.
  Diagnostics currently report both `lazy`.
- **Safety/citations:** emit `recommended_citation` in **all** response modes
  (remove the standard/full gate in `tools/retrieval.py:44-46`) and in the export
  tool; embed the HPO release version in the citation string
  (`api/mcp/resources.py`).

### WS6 -- LLM negation scope (Correctness, benchmark-gated)

- Root cause: `phentrieve/llm/prompts/templates/two_phase/en.yaml` (v3.0.0) lacks
  negation-scope guidance; assertion polarity is produced without an explicit
  scope-reasoning step, so "X without Y" over-negates X.
- Fix:
  1. System-instruction scope rule: the negation cue belongs only to the concept
     it directly modifies; in "X without Y", X is affirmed and only Y is negated;
     a modifier-level negation (e.g. "non-progressive", "without regression")
     never negates the head finding.
  2. Add a scope-reasoning field plus `negation_cue` and `cue_target`, ordered
     **before** the assertion enum, in the controlled-generation schema
     (`phentrieve/llm/providers/gemini.py` / response schema builder). Verify
     `propertyOrdering` actually holds in the installed `google-genai`; if not,
     name fields so alphabetical order still puts reasoning first, or raise
     `thinking_level` as the model-native reasoning path.
  3. Add contrastive negation few-shots to the phase-1 template: the
     "severe intellectual disability without regression" pair (head present),
     "intellectual disability with developmental regression" (cue absent),
     "no intellectual disability" (true head negation), "seizures without fever",
     "non-progressive ataxia", "mother had epilepsy" (family history),
     "possible mild hearing loss" (uncertain). Bump prompt v3.0.0 -> v3.1.0.
- Gate: run the existing benchmark before/after. If mapping accuracy regresses,
  revert only the LLM change (the WS1-WS5 fixes stand on their own).
- Tests: a focused regression eval over the few-shot cases plus variants.

## 3. New/changed files (anticipated)

- `phentrieve/text_processing/pipeline.py` -- assertion detection over source
  sentence span (C1).
- `phentrieve/text_processing/_hpo_extraction_helpers.py` /
  `hpo_extraction_orchestrator.py` -- ensure negated retrieval/emission (H2).
- `api/mcp/projection.py` (new) -- MCP-only schema projection (M4, T1, L5, L7).
- `api/mcp/tools/retrieval.py` -- extract default 1, strategy enum, citation in
  all modes, include_details default/honor, projection wiring.
- `api/mcp/tools/_common.py` -- `TextArg` min_length, `ChunkStrategy` enum,
  `DEFAULT_EXTRACT_NUM_RESULTS`.
- `api/mcp/next_commands.py` -- score in export payload, executable placeholders.
- `api/mcp/service_adapters.py` -- export key aliasing + score threading (M2/H3).
- `api/mcp/capabilities.py` -- stable base hash + docs (M1), strategy enum, alias
  docs (L4/L8).
- `api/mcp/envelope.py` -- typed validation message (M3).
- `api/mcp/resources.py` -- citation string + HPO version.
- `api/mcp/server.py` / facade -- warmup, cold-start instruction hint.
- `phentrieve/phenopackets/utils.py` / `export_models.py` -- score in evidence
  ref, dual-version createdBy (H3/L6).
- `phentrieve/llm/prompts/templates/two_phase/en.yaml` + response schema -- WS6.
- Tests under `tests/` mirroring each fix.

## 4. Testing and verification plan

1. **TDD:** each fix gets a failing test first under `tests/` (unit/integration).
2. **Gates:** `make check`, `make typecheck-fast`, `make test`, then
   `make ci-local` and `make security-python` before any push.
3. **Frontend safety:** since correctness fixes touch the shared pipeline, run
   `make frontend-test-ci` to confirm the curation UI contract is unaffected.
4. **Benchmark gate (WS6):** before/after mapping benchmark; revert LLM change on
   regression.
5. **Live re-test:** reproduce each evaluation defect against the running MCP,
   apply fixes, restart `make mcp-serve-http`, re-run the ~20-call matrix, and
   record results in `.planning/analysis/2026-06-14-phentrieve-mcp-hardening-
   verification.md` with an updated scorecard.
6. **Worktree:** all work in a git worktree; one logical commit per workstream;
   version bumps (core + MCP server) + CHANGELOG at the end.

## 5. Definition of done (>= 9.5 everywhere)

- No false-positive affirmed phenotype from common negation constructions (C1,
  WS6).
- One phenotype per phrase by default; candidate lists opt-in (H1).
- `aggregated_hpo_terms` pipes into `export_phenopacket` without remapping and
  preserves `score` (M2, H3).
- `full`-mode payload has no empty chunks and no duplicate score fields (T1, M4).
- `capabilities_version` is stable and matches `_meta` across `details` (M1).
- Every error is a typed envelope with an actionable message (M3, L1, L4).
- Single chunk-index scheme; consistent offsets; no silent id collisions (M4, L5,
  L7).
- `recommended_citation` present in all modes with HPO version (safety).
- Per-phase timing on the LLM path (observability).
- All Python + frontend gates green; live matrix re-run confirms the scorecard.
