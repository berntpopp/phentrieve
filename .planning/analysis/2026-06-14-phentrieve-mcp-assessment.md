# Phentrieve MCP Server -- LLM-Consumer Assessment and Remediation Roadmap

- Date: 2026-06-14
- Evaluator: Claude (Opus 4.8), acting first as an LLM consumer, then as a senior MCP tester
- Target: local Phentrieve MCP (`http://localhost:8001/mcp/`)
- Versions observed this session:
  - MCP server `0.15.1`, core `phentrieve-core 0.23.1`, HPO `v2026-02-16`
  - internal extraction LLM `gemini-3.1-flash-lite`, extraction prompt `v3.1.0+v4.1.0`
  - transport `streamable_http`, endpoint `/mcp`
  - `capabilities_version sha256:d8098d3720af5a42`, `descriptor_hash sha256:c8ffebae756ebac7`
- Method: live tool calls across all 8 tools (happy-path, edge, error, response-mode,
  cross-tool pipeline, deterministic-vs-LLM comparison). Every finding below is reproduced
  from real responses captured this session.
- Scope: consumer-facing behaviour and contract quality. Research use only; the server is
  not clinical decision support.

This document consolidates the two assessments produced in session -- an LLM
user-experience rating (Part A) and a senior-tester defect register (Part B) -- then maps
each gap to current Anthropic (MCP / tool-use) and Google (Gemini function-calling) best
practices (Part C) and lays out the concrete changes needed to reach a consistent `>= 9/10`
across every dimension (Part D).

It supersedes the prior same-dated evaluation. Findings from that prior session that were
not reproduced here, or that this session refined, are reconciled in Section 8 so no
knowledge is lost.

---

## 1. Executive summary

Phentrieve is an above-average MCP citizen. Its observability, self-describing tool
signatures, `next_commands` chaining, structured/typed error envelopes, and GA4GH
conformance are genuinely strong and should be preserved. The gap to a uniform `>= 9` is
concentrated in three areas, all fixable:

1. **Extraction correctness.** Both extractors share one negation defect: the negated scope
   is computed correctly but then applied to HPO matches that fall outside that scope. This
   is the single highest-value fix.
2. **Payload shape.** `export_phenopacket` double-encodes its main output as a JSON string;
   extract `full` mode triplicates the findings; `compare_hpo_terms` ignores `response_mode`
   entirely.
3. **Confidence signalling.** `search_hpo_terms` returns nearest-neighbour hits for pure
   gibberish with no low-confidence flag.

The plumbing (errors, observability, safety, discoverability) is reference-quality; the
weak link is what the extraction tools put in the context window and how accurate the
assertion polarity is.

### 1.1 Consolidated scorecard (current vs. target)

| Dimension              | Current | Target | Primary blockers to `>= 9` |
|------------------------|:------:|:------:|----------------------------|
| Discoverability        |   9    |  9.5   | D8 (hash contract), D10 (chunk strategy parity) |
| Token efficiency       |   6    |  9     | D7 (full-mode triplication + padding), D4 (double-encode) |
| Speed                  |   7    |  9     | D9 (diagnostics latency), LLM extract 14.7 s, no streaming |
| Observability          |  10    |  10    | keep as-is |
| Output / schema design |   6    |  9     | D4 (double-encode), D5 (compare ignores mode), D11/D13 (provenance + attribution consistency) |
| Correctness            |   6    |  9     | D1 (negation scope), D2 (per-chunk=1 drops co-findings), D3 (mapping/precision) |
| Safety & citations     |   9    |  9.5   | citation in every mode incl. `minimal` |
| Error handling         |   9    |  9.5   | maintain typed envelopes across every tool |
| **Overall**            | **7.5**|  9+    | resolve D1 + D2 + D3 + D4 + D5 + D6 |

---

## 2. Tools exercised this session

| Tool | Cases run | Verdict (score) |
|------|-----------|-----------------|
| `phentrieve_get_capabilities` | base + `details=[sample_calls, argument_aliases]` | strong; hash-contract nit (9.5) |
| `phentrieve_diagnostics` | health probe | pass; slow (8) |
| `phentrieve_search_hpo_terms` | German phrase, gibberish, mode variants | strong; no low-confidence signal (8.5) |
| `phentrieve_chunk_text` | clinical paragraph, `simple` strategy | pass; strategy parity nit (8) |
| `phentrieve_extract_hpo_terms` (deterministic) | negation-heavy excerpt, `include_unmatched_chunks` | weakest; correctness (6) |
| `phentrieve_extract_hpo_terms_llm` | full case report | best matches; shared negation bug, latency, bloat (6.5) |
| `phentrieve_compare_hpo_terms` | related, unrelated, invalid id, both formulas, standard+full | correct ordering + great error path; `response_mode` inert (7) |
| `phentrieve_export_phenopacket` | subject metadata, negated->excluded, sidecar | spec-faithful; double-encoded output (8.5) |

---

## 3. Part A -- LLM-consumer UX evaluation (per dimension)

### 3.1 Keep these (genuine strengths)

- **Self-describing tool signatures.** Each description embeds the call signature, enum
  constraints, examples, and a `do_not_use_for`. Correct calls require no trial-and-error.
- **`next_commands` chaining.** After extraction, `_meta.next_commands` returns a populated
  `phentrieve_export_phenopacket` payload -- a copy-forward that collapses a multi-step
  workflow into one step. `compare` even self-suggests the alternate formula.
- **Observability (10/10).** Every call carries `_meta` (`request_id`, `elapsed_ms`,
  `capabilities_version`, `unsafe_for_clinical_use`, `response_mode`). The LLM extractor adds
  a rich `observability` block: per-phase timings, local vs. llm-mapped phrase counts,
  unresolved/deferred counts, token I/O, provider/model/prompt_version.
- **Structured error envelopes.** `success:false`, typed `error_code`, `retryable`,
  `recovery_action`, and `next_commands` pointing at recovery tools.
- **GA4GH conformance.** `excluded:true` for negated features, ECO evidence codes, HPO
  resource/version block, research-use disclaimer embedded in the bundle, linked annotation
  sidecar.
- **Safety contract.** `unsafe_for_clinical_use:true` on every response,
  `research_use_acknowledged` gating on extraction, `recommended_citation` verbatim.

### 3.2 Where it loses points

- **Token efficiency (6).** `full`-mode extraction serialises the findings three times
  (`processed_chunks`, `aggregated_hpo_terms`, `_meta.next_commands.phenotypes`) and pads
  every record with null/zero fields (`start_char:null`, `end_char:null`,
  `invalid_chunk_reference_count:0`). Good `response_mode` knob, bloated `full` payload.
- **Output / schema design (6).** `export_phenopacket` returns its core artifact as an
  escaped JSON string; `compare_hpo_terms` ignores `response_mode`; deterministic
  `aggregated_hpo_terms` carry `text_attributions` on some entries but not others.
- **Correctness (6).** Both extractors mis-apply negation scope (Section 4, D1); the
  deterministic path additionally drops co-occurring findings and mis-maps incidental
  context.
- **Speed (7).** Search is excellent (~40 ms warm); the LLM extract is ~14.7 s with no
  streaming or progress, and `diagnostics` is a slow 1.75 s health check.

---

## 4. Part B -- Defect register (severity-ranked, with this-session evidence)

Severity reflects impact on a consumer that pipes Phentrieve output into a phenopacket
without human review.

### CRITICAL

**D1 -- Negation scope is computed correctly but applied to the wrong HPO span (both
extractors).**
The deterministic extractor's own `assertion_details` is the smoking gun:

```
chunk: "developmental course showed severe intellectual disability without regression"
keyword_negated_scopes:      ["without: regression"]
dependency_negated_concepts: ["without -> regression"]
final_status: "negated"
-> HPO match "Severe intellectual disability" (HP:0010864) tagged assertion: "negated"
```

The negation is correctly scoped to *regression*, but the **chunk-level** `negated` status
is then applied to *every* match in the chunk, including "intellectual disability", which is
not in the negated span. The patient demonstrably *has* severe ID. The LLM extractor shares
this assertion layer and produced three analogous mis-negations on the full case report:
- "severe intellectual disability without regression" -> ID marked negated (present in reality)
- "could sit ... does not walk or speak" -> Motor delay marked negated (delay is present)
- "respiratory arrest under propofol sedation ... could not be performed" -> arrest negated
  (it occurred)

Fix: only flip a match to `negated` when the matched phrase span overlaps a computed
negated-scope span. The required data already exists in `assertion_details`.

### HIGH

**D2 -- Deterministic `num_results_per_chunk` defaults to 1 and silently drops co-occurring
findings.**
The chunk "floppy infant with initial hypotonia progressing to hypertonia of the
extremities" returned **only** `HP:0008935 Generalized neonatal hypotonia`; the co-present
hypertonia was dropped. The LLM extractor caught both (Hypotonia + Limb hypertonia). A
one-finding-per-sentence default is wrong for clinical prose.
Fix: raise the default (1 -> ~3) or detect multi-concept chunks; document the cap.

**D3 -- Deterministic precision / span-selection problems.**
- "respiratory arrest under propofol sedation" -> `HP:6000883 Recent history of sedation by
  propofol infusion` (0.75): grabbed the incidental sedation, missed the actual phenotype.
- "does not walk or speak" was over-chunked to bare tokens `walk` / `speak` ->
  `Waddling gait` (negated) and `Mutism`.
- "microcephaly" -> `HP:0005484 Secondary microcephaly` (over-specific vs. the LLM path's
  generic `HP:0000252 Microcephaly`).
- The sub-sentence splitter emitted `"due"` as a standalone chunk.
Fix: prefer the salient phenotype over incidental context; avoid bare-token chunks; revisit
over-specific mappings.

**D4 -- `export_phenopacket` double-encodes its primary output.**
`phenopacket_json` is returned as an escaped JSON **string** (`"{\n \"id\": ...}"`), forcing
a client-side `JSON.parse` and roughly doubling tokens via `\"` / `\n` escaping.
Fix: return a native JSON object (or gate the string form behind a flag).

### MEDIUM

**D5 -- `compare_hpo_terms` ignores `response_mode` entirely.**
`standard` and `full` returned the *identical* minimal payload:
`{term1_id, term2_id, formula_used, similarity_score}`. No MICA / common ancestor, no IC
values, no path length, not even the term labels. (Ordering itself is sound: related pair
HP:0000787 vs HP:0004724 = 0.63 hybrid / 0.5 resnik; unrelated HP:0000787 vs HP:0001631 =
0.096.)
Fix: at `standard`/`full`, return the MICA, per-term IC, term labels, and subsumer path so
the score is explainable.

**D6 -- `search_hpo_terms` has no low-confidence signal.**
Pure gibberish ("wuggle frobnicate zxcvbnm splorptin quibblewax") returned 5 hits at
0.48-0.59 -- above the 0.3 default `similarity_threshold` -- with nothing marking them as
junk. A naive caller would treat `Deuteranomaly 0.59` as a real match.
Fix: add a `confidence_band` / `no_high_confidence_match` flag when the top score is below a
floor (~0.7); consider raising the default threshold.

**D7 -- Extract `full`-mode payload is bloated.**
Findings are serialised three times (`processed_chunks`, `aggregated_hpo_terms`,
`_meta.next_commands.phenotypes`) and every record is padded with null/zero fields.
Fix: drop empty-match `processed_chunks` (or gate behind `include_unmatched_chunks`), make
`next_commands` reference the aggregated list instead of re-embedding it, and omit
null/default fields.

**D8 -- Capabilities hash contract is ambiguous.**
The base `get_capabilities` returns `descriptor_hash sha256:c8ffebae...` alongside
`capabilities_version sha256:d8098d37...` -- two different hashes -- and the warm-cache
contract says to compare `_meta.capabilities_version`. It is unclear what `descriptor_hash`
is for or whether it is stable across `details` expansion.
Fix: document a single canonical cache key, ensure it is stable across `details`, and either
explain or drop `descriptor_hash`.

### LOW

- **D9** -- `diagnostics` took 1.75 s while reporting `embedding_model: lazy` and
  `vector_index: lazy` (not loaded); the cost is likely an un-cached LLM-backend probe.
  Cache/parallelise probes or document the cost.
- **D10** -- `chunk_text` `simple` strategy does not mirror the sub-sentence chunking the
  extractors use internally (which produced `"due"`), so it cannot preview extract
  segmentation. Document or expose the extractor's strategy.
- **D11** -- `export_phenopacket` fabricated provenance as `match_method: "legacy_dict"` /
  `source_mode: "chunk"` for hand-supplied phenotypes. Mark client-supplied provenance as
  `client_supplied` / `unknown`.
- **D12** -- Version split: the bundle's `createdBy` is `phentrieve-core 0.23.1` while the
  server reports `0.15.1`. Cosmetic; document the core-vs-server split.
- **D13** -- Deterministic `aggregated_hpo_terms` carry `text_attributions` on some entries
  (exact matches) but omit them on others -- a schema inconsistency across records.

---

## 5. Part C -- Best-practices alignment

### 5.1 Anthropic MCP / tool-use guidance

| Practice | Status | Notes |
|----------|:------:|-------|
| Descriptions prescriptive about WHEN to call | PASS | `do_not_use_for`, `next_tools`, `canonical_workflow` are exemplary |
| Enums for fixed-value parameters | PASS | `response_mode`, `formula`, and `strategy` are all enumerated in their schemas |
| Focused tool set (avoid sprawl) | PASS | 8 well-scoped tools |
| Structured/typed errors, not string-matching | PASS | `not_found` envelope is textbook (typed code, retryable, recovery_action) |
| Do not dump intermediate data into the context | PARTIAL | `full`-mode triplication + padding (D7); double-encoded phenopacket (D4) |
| Honor declared response controls consistently | FAIL | `compare_hpo_terms` ignores `response_mode` (D5) |
| Stable cache key / version signal | PARTIAL | two hashes, unclear contract (D8) |
| Deterministic, parseable output | PARTIAL | phenopacket returned as an escaped string (D4) |
| Signal low-confidence / no-result clearly | FAIL | gibberish accepted silently (D6) |

### 5.2 Google Gemini function-calling guidance (internal extraction LLM)

The LLM extractor uses `gemini-3.1-flash-lite` (`whole_document_grounded`, two-phase). The
shared negation defect (D1) and over-negation point to post-processing and prompt/decoding
choices, not retrieval.

| Practice | Status | Notes |
|----------|:------:|-------|
| Detailed function + parameter descriptions | PASS | Surface descriptions are strong |
| Strong typing / enums on the function surface | PASS | Flat, enum-constrained schemas |
| Controlled generation (response schema) for extraction | RECOMMEND | Make assertion polarity a typed enum field with an explicit scope span, not free text |
| Low temperature for deterministic extraction | RECOMMEND | Verify the internal call pins low/zero temperature for reproducibility |
| Few-shot for hard linguistic cases | RECOMMEND | Add negation-scope few-shots: "X without Y" must NOT negate X; "no X / not X / does not X / denies X" must negate X |
| Keep parameters flat and simple | PASS | Flat schemas throughout |

---

## 6. Part D -- Remediation roadmap to `>= 9/10`

Grouped into prioritized workstreams; each lists the dimension it lifts.

### Workstream 1 -- Extraction correctness (Correctness 6 -> 9)

1. **D1 (CRITICAL)**: intersect the computed negated-scope span with each matched HPO
   phrase span; only set `negated` when they overlap. Drive both the deterministic and LLM
   assertion layers from this rule. Add regression fixtures from this session:
   - "severe intellectual disability without regression" -> ID **affirmed**, regression **negated**
   - "does not walk or speak" -> motor/speech findings **present**
   - "respiratory arrest under propofol sedation" -> respiratory arrest **affirmed**
   - plus the carried-over cases from Section 8: "There is no nystagmus", "She does not have
     ataxia", "patient denies headache".
2. **D2**: raise deterministic `num_results_per_chunk` default (1 -> ~3) or auto-detect
   multi-concept chunks; document the cap so dropped co-findings are never silent.
3. **D3**: improve span selection (salient phenotype over incidental context), eliminate
   bare-token chunks, and revisit over-specific default mappings.
4. **LLM path**: add Gemini negation-scope few-shots, enforce a typed assertion enum via
   controlled generation, and verify a low decoding temperature.

### Workstream 2 -- Output shape and token efficiency (Output/schema 6 -> 9; Token 6 -> 9)

5. **D4**: return `phenopacket_json` as a native JSON object (or add an inline flag); stop
   double-encoding.
6. **D7**: in `full` mode drop empty-match `processed_chunks` (or gate behind
   `include_unmatched_chunks`), make `next_commands` reference the aggregated list rather
   than re-embedding it, and omit null/default fields.
7. **D5**: make `compare_hpo_terms` honor `response_mode` -- return MICA, per-term IC, term
   labels, and subsumer path at `standard`/`full`.
8. **D13**: emit `text_attributions` consistently on every `aggregated_hpo_terms` entry.
9. **D11**: mark client-supplied phenotype provenance as `client_supplied` / `unknown`
   instead of defaulting to `legacy_dict`.

### Workstream 3 -- Confidence signalling and discoverability (Discoverability 9 -> 9.5)

10. **D6**: add a `confidence_band` / `no_high_confidence_match` flag to `search` (and to
    extract chunk matches) when the top score is below a floor; consider raising the default
    `similarity_threshold`.
11. **D8**: document a single canonical cache key, ensure it is stable across `details`, and
    clarify or drop `descriptor_hash`.
12. **D10**: document or expose the extractors' internal chunk strategy so `chunk_text` can
    reproduce extract segmentation.

### Workstream 4 -- Speed (7 -> 9)

13. **D9**: cache/parallelise `diagnostics` probes.
14. LLM extract latency: stream progress or document the ~14.7 s cost; optionally warm the
    lazy embedding model and vector index at startup or via a warmup call.

### Workstream 5 -- Safety, citations, polish (9 -> 9.5)

15. Emit `recommended_citation` in every response mode (confirm `minimal`).
16. **D12**: reconcile or document the `createdBy` core-vs-server version split.
17. Maintain typed error envelopes uniformly across all eight tools.

### 6.1 Definition of done (`>= 9` everywhere)

- No present finding is mis-negated by any common negation construction; negation applies
  only to matches overlapping a negated scope (D1).
- Multi-finding sentences return all distinct co-occurring phenotypes by default (D2, D3).
- `phenopacket_json` is a native object; `full`-mode extract carries no triplicated or
  padded data (D4, D7).
- `compare_hpo_terms` returns MICA / IC / labels at `standard`/`full` (D5).
- `search` flags low-confidence / no-match results (D6).
- One canonical, stable capabilities cache key (D8).
- Every tool honors `response_mode`; every error is a typed, actionable envelope.

---

## 7. Appendix -- selected evidence (this session)

- D1: `phentrieve_extract_hpo_terms` (request `c83161139ffc`) chunk 2 `assertion_details`
  reported `keyword_negated_scopes: ["without: regression"]` yet tagged
  `HP:0010864 Severe intellectual disability` as `negated`. The LLM run
  (request `9eb44a3fbed0`) negated severe ID, Motor delay, and Respiratory arrest.
- D2: same deterministic run, chunk 6 ("...hypotonia progressing to hypertonia...") returned
  only `HP:0008935 Generalized neonatal hypotonia`; the LLM run returned both
  `HP:0001252 Hypotonia` and `HP:0002509 Limb hypertonia`.
- D3: chunk 9 "respiratory arrest under propofol sedation" ->
  `HP:6000883 Recent history of sedation by propofol infusion`; chunk 8 text was `"due"`.
- D4: `phentrieve_export_phenopacket` (request `49e773531ce9`) returned `phenopacket_json`
  as an escaped JSON string; `score 0.99` was preserved into the evidence reference
  ("Phentrieve retrieval confidence: 0.9900") when supplied.
- D5: `phentrieve_compare_hpo_terms` `response_mode="standard"` (request `16e4c69dd730`) and
  `response_mode="full"` (request `3900abd724fa`) returned identical minimal payloads.
- D6: `phentrieve_search_hpo_terms("wuggle frobnicate zxcvbnm splorptin quibblewax")`
  (request `d6347048b2f0`) returned 5 hits at 0.48-0.59 with no low-confidence flag.
- D8: base `get_capabilities` (request `3091696bd67d`) returned
  `descriptor_hash sha256:c8ffebae756ebac7` alongside
  `capabilities_version sha256:d8098d3720af5a42`.
- D9: `phentrieve_diagnostics` (request `72819e95408d`) `elapsed_ms 1755` with
  `embedding_model: lazy`, `vector_index: lazy`.
- Error path: `phentrieve_compare_hpo_terms("HP:9999999", "HP:0000787")` (request
  `1e9803cebd20`) returned a clean `not_found` envelope (`retryable:false`,
  `recovery_action: reformulate_input`) in 0.28 ms -- the exemplar to keep.

---

## 8. Reconciliation with the prior (removed) same-dated evaluation

The previous file recorded findings from an earlier session at extraction prompt `v3.0.0`.
This session ran at `v3.1.0` and refined or did not reproduce some of them. Carried forward
for re-verification:

- **Refined**: the negation defect was previously hypothesised as "the chunker strips the
  leading negation cue" (false positives on "no X" / "not X"). This session shows a sharper
  mechanism (D1): the cue is *retained and scoped correctly*, but chunk-level status is
  applied to non-scoped matches. Both manifestations should be covered by the same fix and
  the same fixtures.
- **Not reproduced this session**: the prior "export emits confidence 0.0000" finding -- when
  a `score` was supplied, export preserved it (0.99). The underlying cause (the suggested
  `next_commands` phenotype shape omits `score`) should still be confirmed and fixed so the
  copy-forward path is lossless.
- **Re-verify**: prior findings on (a) extract->export key-shape acceptance, (b) empty/
  whitespace query handling, (c) the `capabilities_version` changing under `details`
  expansion, and (d) the `include_details` no-op at `compact`. The current schema advertises
  accepting the raw extractor shape, but it was not re-tested here.

All behaviour above is from the local instance at the versions in the header and may change
as the server is updated; re-run the test matrix after applying fixes.
