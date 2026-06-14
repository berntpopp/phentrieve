# Phentrieve MCP -- Assessment Remediation Design (to >= 9.5/10)

- Date: 2026-06-14
- Author: Claude (Opus 4.8), acting as senior MCP engineer
- Source assessment: `.planning/analysis/2026-06-14-phentrieve-mcp-assessment.md`
- Prior work: PR #288 (`.planning/completed/2026-06-14-phentrieve-mcp-hardening-plan.md`,
  verification `.planning/analysis/2026-06-14-phentrieve-mcp-hardening-verification.md`)
- Status: APPROVED to plan + execute (decisions captured below)

## 1. Why this exists (and how it differs from PR #288)

PR #288 raised the MCP server to ~9.4 and fixed a large defect set (C1 under-negation,
H1/H2/H3, M1-M5, T1, L1-L8, citations-in-all-modes, per-phase LLM timing). The new
consumer assessment (which **supersedes** the prior evaluation) was run against a server
that still exhibits a distinct set of defects -- some that PR #288 *explicitly punted* and
some it never scoped. This design closes exactly those remaining gaps.

Critical distinction: the assessment's **D1 is NOT the same bug as PR #288's C1**.

- **C1 (fixed):** *under*-negation. The chunk cleaner stripped a leading cue ("no", "not")
  so the detector reported AFFIRMED for an explicitly negated finding. Fixed by detecting
  over the restored within-sentence context.
- **D1 (this design):** *over*-negation. The negated scope is computed correctly (e.g.
  `keyword_negated_scopes: ["without: regression"]`) but the single **chunk-level** status
  is then applied to **every** match in the chunk -- so "Severe intellectual disability" in
  "severe intellectual disability without regression" is wrongly marked negated. The
  hardening verification report concedes this: *"the deterministic tool is screening-grade
  ... the 'without' case is handled in the LLM path."* The assessment rejects that as the
  highest-value remaining fix.

### 1.1 Current-state delta (verified in code, 2026-06-14)

| Assessment defect | State on `main` now | Evidence |
|---|---|---|
| D1 over-negation (deterministic) | UNFIXED (punted as "screening-grade") | `_hpo_extraction_helpers.py:66-71` applies one chunk status to all matches |
| D1 over-negation (LLM prompt) | REVERTED (regressed Gemini benchmark) | commit `add7a16` reverted by `6672e79` |
| D2 default num_results=1 drops co-findings | BY DESIGN (PR #288 H1) -- conflict | `api/mcp/tools/_common.py:20` `DEFAULT_EXTRACT_NUM_RESULTS = 1` |
| D3 precision / bare-token chunks | UNFIXED | sub-sentence splitter emits "due"/"walk"; incidental "sedation" wins |
| D4 phenopacket double-encoded | UNFIXED | `phenopacket_router.py:151` `json.dumps(...)`; schema `phenopacket_json: str` |
| D5 compare ignores response_mode | UNFIXED | `service_adapters.py:268-274` returns 4 fields only |
| D6 search no low-confidence signal | UNFIXED | no band/flag on results |
| D7 full-mode triplication residual | PARTIAL | projection dedups; `next_commands.after_extract` still re-embeds list |
| D8 capabilities hash contract | MOSTLY FIXED (M1) | needs one-line contract note |
| D9 diagnostics latency | PARTIAL (warmup deferred) | timing added, no startup warmup |
| D11 client-supplied provenance | PARTIAL (L6) | needs explicit `client_supplied`/`unknown` |
| D12 createdBy split | FIXED (L6) | dual version string |
| D13 inconsistent text_attributions | UNFIXED | present on some aggregated entries only |

## 2. Decisions (locked)

1. **Defer LLM-D1.** Everything shipped here is locally verifiable; no Gemini budget. The
   LLM negation-scope prompt change stays out (it regressed the mapping benchmark and the
   LLM tool already outperforms deterministic on negation). Documented as a known limit.
2. **Shared pipeline for correctness.** D1/D2/D3 go in `phentrieve/` so REST + the Vue
   curation frontend benefit; frontend CI is re-run as a blast-radius check. Output-shape
   fixes (D4/D5/D7/D13) stay at the `api/mcp/` boundary. D4 native phenopacket is **additive**
   (keep `phenopacket_json` string for back-compat).
3. **Keep `num_results_per_chunk` default = 1.** Solve D2 by splitting multi-concept chunks
   so each distinct phenotype is its own best-match chunk; the existing parameter remains the
   opt-in to surface sibling candidates. Do **not** reverse PR #288's H1.

## 3. Best-practices grounding (current docs)

- **MCP spec (2025-06-18) + SEP-1624:** return real JSON in `structuredContent`, not
  stringified JSON ("at worst semantically duplicated output"); a serialized string may be
  kept in `content` only for backwards compatibility. -> validates **D4 additive native object**.
- **Anthropic, "Writing effective tools for AI agents":** signal uncertainty and empty
  results explicitly; `response_format` concise/detailed (concise ~= 1/3 tokens); avoid
  redundant/low-signal fields; consistent naming; high-signal context; actionable errors.
  -> validates **D5 (honor response_mode), D6 (confidence signal), D7/D13 (no redundancy)**.
- **RAG retrieval guidance:** below-threshold/low-confidence retrievals should trigger
  graceful abstention/degradation rather than silent acceptance. -> validates **D6**.

Sources:
- https://modelcontextprotocol.io/specification/draft/server/tools
- https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1624
- https://www.anthropic.com/engineering/writing-tools-for-agents

## 4. Architecture

Reuse the established 3-layer split:

```
phentrieve/ (shared)   -> correctness: D1 span-level negation, D2 multi-concept split, D3 chunk hygiene
api/mcp/ (boundary)    -> shape/signal: D4 native phenopacket, D5 compare richness, D6 confidence,
                          D7 trim, D13 attribution consistency, D8 doc, D11 provenance
api/mcp/server lifespan -> D9 best-effort warmup
(deferred)             -> LLM-D1 prompt work
```

## 5. Workstream A -- Deterministic correctness (shared `phentrieve/`)

### A1. D1 -- span-level negation (CRITICAL)

**Goal:** a match is `negated` only when its matched-phrase span overlaps a computed
negated-scope span; otherwise it keeps the non-negated status. C1 behaviour ("no X" negates
X) must be preserved.

**Data available:** `_extract_scope` (`assertion_detection.py:601-693`) already computes
`cue_index`, `cue_end`, `scope_start`, `scope_end` char offsets within the detection text
but returns only the joined scope *string*. `get_text_attributions` (`text_attribution.py:18-112`)
returns each match's `start_char`/`end_char` **relative to the chunk text**. The C1 change
(`pipeline.py:247-294`) gives `assertion_context_start`, `start_char`, `end_char` in the
`original_text` frame and the `assertion_input` text actually fed to `detect()`.

**Algorithm:**
1. Detector change (additive): also return `negated_scope_spans: list[tuple[int,int]]` (and
   `normal_scope_spans`) in `assertion_details`, in the **`assertion_input` frame**. Build
   them from the offsets `_extract_scope`/`_detect_with_context_rules` already compute.
   Keep the existing `keyword_negated_scopes` text for observability/back-compat.
2. Pipeline change: when detection ran on restored context, translate each scope span from
   the `assertion_input` frame to the **chunk-text frame**:
   `chunk_offset = assertion_context_start + span_offset - start_char`. Carry the translated
   spans on the chunk record (e.g. `negated_spans`). Spans wholly before offset 0 (a leading
   cue living in the restored prefix) are clamped to start at 0 -- they legitimately scope the
   whole chunk (the C1 case).
3. Per-match decision: in the evidence/aggregation layer where `text_attributions` are
   computed (`_hpo_extraction_helpers.build_evidence_map`), set each match's
   `assertion_status` to NEGATED iff any attribution span overlaps any `negated_spans` entry;
   otherwise AFFIRMED (or NORMAL if normal-scope overlap). Replace the blanket
   `assertion_statuses[chunk_idx]` propagation in `process_chunk_matches` with this
   span-aware result. When no attribution span is found (degenerate), fall back to the
   chunk-level status (preserves C1 for cue-stripped single-concept chunks).
4. Aggregation: `aggregate_and_rank` already votes per chunk status; ensure it votes on the
   per-match span-aware status.

**Regression fixtures (exact assessment strings):**
- "severe intellectual disability without regression" -> ID **affirmed**, regression **negated**
- "does not walk or speak" -> walking/speech findings **present** (not negated as gait/mutism)
- "respiratory arrest under propofol sedation ... could not be performed" -> arrest **affirmed**
- carried: "There is no nystagmus" -> negated; "She does not have ataxia" -> negated;
  "patient denies headache" -> negated (C1 cases stay green)

### A2. D2 -- co-findings via multi-concept chunk splitting

**Goal:** distinct co-occurring phenotypes in one span each surface as their own best-match
chunk, without raising the default cap.

**Change:** extend the conjunction/transition splitting in
`phentrieve/text_processing/chunkers.py` to split on intra-clause transition markers that
signal a new concept -- "progressing to", "with", "without", and arrow/"->" forms -- in the
default `sliding_window_punct_conj_cleaned` strategy. Keep `num_results_per_chunk` default
= 1 as the opt-in to surface sibling candidates; document the cap in the tool description.

**Fixture:** "floppy infant with initial hypotonia progressing to hypertonia of the
extremities" -> Hypotonia AND (Limb) hypertonia both surface.

### A3. D3 -- chunk hygiene + precision

**Changes:**
- Drop degenerate chunks: a chunk that is only stopwords/function words or a single short
  token ("due", bare "walk") is not emitted as a retrieval chunk. Add a guard in
  `FinalChunkCleaner` (`chunkers.py`).
- The finer A2 splitting lets the salient phenotype win over incidental context
  ("respiratory arrest" instead of "propofol sedation"). No bespoke re-rank.
- Over-specific default mappings (microcephaly -> Secondary microcephaly) are an
  embedding/retrieval property; **documented as a known limitation**, not over-fit here.

## 6. Workstream B -- Output shape & schema (`api/mcp/` boundary)

### B1. D4 -- native phenopacket object (additive)
At the MCP `export_phenopacket` boundary, parse `phenopacket_json` back to an object and add
a `phenopacket` field (real JSON object). Keep `phenopacket_json` (string) for back-compat
per MCP guidance. REST schema unchanged.

### B2. D5 -- compare honours response_mode
Enrich `compare_hpo_terms_service` so at `standard`/`full` it returns: `mica` (id + label),
`lca_depth`, per-term `depth` and IC-proxy (`depth/max_depth`), and path lengths to the MICA
-- all from `find_lowest_common_ancestor` + `load_hpo_graph_data` (`metrics.py:255-399`).
`minimal`/`compact` stay the lean 4-field payload. Wire `apply_response_mode` to actually
gate these fields.

### B3. D7 -- stop residual triplication
Keep `next_commands.after_extract` as the slim, self-contained executable copy-forward
(hpo_id/label/assertion/score, capped) -- it must stay executable, so this is the correct
minimal form (documented). Ensure `processed_chunks` is opt-in beyond matched evidence so
`full` mode does not re-serialise the aggregated findings. Confirm null/default padding is
dropped by the projection (`start_char:null`, `invalid_chunk_reference_count:0`).

### B4. D13 -- consistent text_attributions
`project_aggregated_terms_for_mcp` guarantees a `text_attributions` key on every aggregated
term (empty list when none), so the schema is uniform across records.

### B5. D11 -- honest provenance for client-supplied phenotypes
When phenotypes are supplied directly to `export_phenopacket` (not produced by an extractor
run), mark provenance `client_supplied` / `match_method: unknown` instead of defaulting to
`legacy_dict` / `source_mode: chunk`.

## 7. Workstream C -- Confidence & discoverability (`api/mcp/`)

### C1. D6 -- low-confidence signal
Add to `search_hpo_terms` (and extract chunk matches) a `confidence_band`
(`high` >= 0.7, `moderate` >= floor, `low` otherwise) and a top-level
`no_high_confidence_match: true` when the top score is below the high floor (~0.7).
Non-breaking; default `similarity_threshold` is **unchanged** (signal, don't silently
re-filter). Document the band semantics in the tool description + capabilities.

### C2. D8 -- cache-key contract note
Add a one-line note to the capabilities body: `capabilities_version` is the canonical
warm-cache key (echoed in `_meta`, stable across `details`); `descriptor_hash` is the
per-descriptor content hash for the expanded view; `tools/list_changed` is the MCP-spec
change signal.

## 8. Workstream D -- Speed & polish

### D1w. D9 -- real startup warmup
Add a best-effort warmup in the MCP server lifespan (`api/mcp/server.py` / facade): load the
embedding model + vector index so `diagnostics` and first extract are warm. Guard with
try/except + a log line; never block or fail startup. (PR #288 deferred this.)

## 9. Out of scope

- LLM-D1 negation-scope prompt/few-shot work (deferred decision).
- Streaming the ~14.7s LLM extract (documented latency profile already shipped).
- Over-specific embedding mappings (documented limitation).
- Any REST schema or frontend behaviour change beyond what shared-pipeline correctness
  naturally produces (frontend CI must stay green).

## 10. Testing strategy

- TDD per fix; single-threaded runs for clarity (`-n 0 -q`).
- New tests under `tests/` only (respect `tests/unit/api/mcp/__init__.py` parity; never add
  package dirs without `__init__.py`).
- D1: unit tests on the span-aware assertion result + orchestrator integration tests using
  the exact assessment fixtures (Section 5 A1).
- D2/D3: chunker unit tests for the multi-concept splits and degenerate-chunk drops.
- D4/D5/D6/D11/D13: MCP-boundary unit tests (object shape, response_mode gating, band/flag,
  provenance, attribution presence).
- Gates: `make check`, `make typecheck-fast`, `make test`, then `make ci-local` +
  `make security-python` + `make frontend-test-ci` (blast-radius), per the CI-parity rule.
- Live MCP re-test of the full assessment matrix; write
  `.planning/analysis/2026-06-14-phentrieve-mcp-assessment-remediation-verification.md` with
  before/after evidence and an updated scorecard.
- Add coverage tests for any touched lines lacking them.

## 11. Definition of done (>= 9.5 everywhere targeted by the assessment)

- No present finding is mis-negated: negation applies only to matches overlapping a negated
  scope (D1); the C1 cases stay green.
- Multi-finding spans return all distinct co-occurring phenotypes by default (D2/D3); no
  degenerate bare-token chunks.
- `export_phenopacket` returns a native `phenopacket` object (string kept for compat) (D4).
- `compare_hpo_terms` returns MICA/IC/labels/path at `standard`/`full` and honours every
  mode (D5).
- `search` flags low-confidence / no-high-confidence-match (D6).
- `full`-mode extract carries no triplicated or null-padded data; `text_attributions`
  present on every aggregated term (D7/D13).
- Capabilities documents one canonical cache key (D8); client-supplied provenance is honest
  (D11); diagnostics/first-extract are warm (D9).
- Full local + frontend gates green; live re-test confirms each item; verification report +
  scorecard land; versions bumped + CHANGELOG updated; PR opened (not self-merged).

## 12. Defect -> workstream coverage

D1 -> A1. D2 -> A2. D3 -> A3. D4 -> B1. D5 -> B2. D7 -> B3. D13 -> B4. D11 -> B5.
D6 -> C1. D8 -> C2. D9 -> D1w. D12 -> already fixed (verify). LLM-D1 -> deferred.
