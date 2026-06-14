# Phentrieve MCP Server -- LLM-Consumer Evaluation and Test Report

- Date: 2026-06-14
- Evaluator: Claude (Opus 4.8), acting as an LLM consumer and senior MCP tester
- Target: local Phentrieve MCP (`http://localhost:8001/mcp/`)
- Versions observed: MCP server `0.15.1`, core `phentrieve 0.23.1`, HPO `v2026-02-16`,
  internal extraction LLM `gemini-3.1-flash-lite`, extraction prompt `v3.0.0+v4.1.0`
- Method: ~20 live tool calls across all 8 tools (happy-path, edge, error, response-mode,
  cross-tool pipeline). All findings are reproduced from real responses.
- Scope: consumer-facing behavior and contract quality. Research use only; the server is
  not clinical decision support.

This document consolidates two assessments produced in session -- an LLM user-experience
rating and a senior-tester defect register -- and adds a remediation roadmap mapping each
gap to current Anthropic (MCP / tool-use) and Google (Gemini function-calling) best
practices, with the concrete changes needed to reach a consistent >= 9/10 across every
dimension.

---

## 1. Executive summary

Phentrieve is an above-average MCP citizen. Its observability, self-describing tool
signatures, `next_commands` chaining, and structured error envelopes are genuinely strong
and should be preserved. It loses points in three places that are all fixable:

1. The **deterministic extractor** (`phentrieve_extract_hpo_terms`) mishandles negation and
   has very low precision -- the single highest-value fix.
2. The **export handoff** loses retrieval confidence and cannot consume extractor output
   without manual key remapping.
3. **Token efficiency** in `full` mode is poor (padding + field redundancy), and the
   **capabilities-version cache contract** is self-inconsistent.

The LLM extractor (`phentrieve_extract_hpo_terms_llm`) and the retrieval/ontology core are
solid; the deterministic extractor is the weak link.

### 1.1 Scorecard (current vs. target)

| Dimension              | Current | Target | Primary blockers to >= 9 |
|------------------------|:------:|:------:|--------------------------|
| Discoverability        |   8    |  9.5   | M1 (version hash), L4 (strategy enum), M5 (details no-op) |
| Token efficiency       |   5    |  9     | T1 full-mode padding, H1 candidate explosion |
| Speed                  |   8    |  9     | warm lazy model/index; document/stream LLM latency |
| Observability          |   9    |  9.5   | per-phase timing on LLM path |
| Output / schema design |   5    |  9     | M4 dual index + score dup, M2 key mismatch, L5 missing key |
| Correctness            |   6    |  9     | C1 negation scope, H1 precision, H2 dropped negatives, LLM over-negation |
| Safety & citations     |   9    |  9.5   | emit `recommended_citation` in all modes |
| Error handling         |   8    |  9.5   | M3 raw KeyError, L1 empty-query, L4 unknown strategy |
| **Overall**            | **7**  | **9+** | resolve C1 + H1 + H2 + H3 + M1-M5 |

---

## 2. Tools exercised

| Tool | Cases run | Verdict |
|------|-----------|---------|
| `phentrieve_get_capabilities` | base, `details=[sample_calls, argument_aliases]` | version-hash bug |
| `phentrieve_diagnostics` | health | pass |
| `phentrieve_search_hpo_terms` | compact/minimal/full, empty, nonsense+threshold | pass (1 minor) |
| `phentrieve_compare_hpo_terms` | hybrid/resnik, identical, not_found, distant, full | pass (2 minor) |
| `phentrieve_chunk_text` | happy, whitespace, bad strategy | pass (1 minor) |
| `phentrieve_extract_hpo_terms` (deterministic) | happy, 3 negation patterns, ack=false | FAIL (correctness) |
| `phentrieve_extract_hpo_terms_llm` | full real clinical note | over-negation (1 case) |
| `phentrieve_export_phenopacket` | full bundle + subject + sidecar, raw-shape | valid output; data-loss bug |

---

## 3. Part A -- UX evaluation (per dimension)

### 3.1 What is genuinely good (keep these)

- **Self-describing tool signatures.** Each description embeds the full call signature plus
  enum constraints and examples. Correct calls can be constructed without trial-and-error.
- **`next_commands` chaining.** After extraction, `_meta.next_commands` returns a populated
  `phentrieve_export_phenopacket` payload -- a copy-forward that turns a multi-step workflow
  into one step.
- **Observability.** `_meta` on every call (`request_id`, `elapsed_ms`, `capabilities_version`,
  `unsafe_for_clinical_use`, `response_mode`), plus the LLM tool's rich `observability` block
  (local vs. llm-mapped phrase counts, unresolved phrases, phase counters, token I/O,
  provider/model/prompt_version).
- **`truncated` field.** Explicit `{field, returned, total}` -- no silent truncation.
- **GA4GH conformance.** `excluded: true` for negated features, ECO evidence codes, HPO
  resource/version block, research-use disclaimer baked into the bundle.
- **Safety contract.** `unsafe_for_clinical_use: true` on every response,
  `research_use_acknowledged` gating, `recommended_citation` verbatim.

### 3.2 Where it loses points

- **Token efficiency (5).** `full`-mode extraction carries ~30 empty `processed_chunks` and
  triplicated score fields; `next_commands` re-emits the full phenotype list already present
  in `aggregated_hpo_terms`. Good `response_mode` knob, bloated `full` payload.
- **Output/schema design (5).** Dual chunk-index schemes, quadruple-redundant score fields,
  null offsets in `evidence_records` while `text_attributions` carries real offsets, and a
  duplicate `hpo_id` (present + negated) that collides for any consumer keying by id.
- **Correctness (6).** Deterministic extractor misses `no X`/`not X` negation and dumps the
  whole candidate list per phrase; LLM extractor over-negates one construction.

---

## 4. Part B -- Defect register (severity-ranked)

Severity reflects impact on a consumer that feeds Phentrieve output into a phenopacket
unsupervised.

### CRITICAL

**C1 -- Deterministic extractor silently misses `no X` / `not X` negation -> false-positive
phenotypes.**
The phrase chunker strips the leading negation cue before assertion detection, so polarity is
lost.

| Input | Chunk text scored | Result | Correct? |
|-------|-------------------|--------|:--------:|
| "There is no nystagmus" | `nystagmus` (offset 41) | Nystagmus **affirmed** | no |
| "She does not have ataxia" | `ataxia` (offset 70) | Ataxia **affirmed** | no |
| "patient denies headache" | `patient denies headache` | negated (dropped) | yes |

Verb-form negation ("denies") survives; prepositional negation ("no", "does not have") does
not, because the chunk start offset jumps past the negation token. This asserts phenotypes the
patient explicitly lacks -- the worst failure class for an annotation tool.
Fix: run assertion detection over the original sentence span, not the trimmed phrase; never
strip negation triggers when selecting chunk boundaries.

### HIGH

**H1 -- No per-phrase best-match selection; the whole top-N candidate list is emitted as
separate affirmed phenotypes.**
"The patient had seizures." -> 10 phenotypes (Seizure, Symptomatic seizures, Focal-onset
seizure, Epileptic aura, Non-epileptic seizure, ...). Three sentences -> 40 aggregated terms.
These are mutually exclusive siblings, not co-occurring findings.
Fix: collapse to top-1 (or score-gap-thresholded) per phrase; move the rest under an optional
`candidates` field. The LLM tool already does this (1 term/phrase).

**H2 -- Deterministic extractor drops negated findings entirely.**
Negated chunks return no `hpo_matches` key, so "denies headache" yields no Headache term --
not even an excluded one. You cannot build `excluded: true` phenopacket features from this
tool, although `export_phenopacket` supports them and the LLM tool emits them.
Fix: emit negated matches with `assertion_status: negated`, matching the LLM path.

**H3 -- Export loses retrieval confidence -> every phenopacket reads `confidence: 0.0000`.**
The `next_commands` phenotype shape (`{hpo_id, label, assertion}`) omits `score`, so the
emitted evidence reference is `"Phentrieve retrieval confidence: 0.0000, Rank: 1"`. The score
exists upstream but is discarded at the handoff.
Fix: include `score` in the suggested phenotype objects and map it into the evidence reference.

### MEDIUM

**M1 -- `capabilities_version` breaks its own warm-cache contract.**
Base `get_capabilities` -> `sha256:46eb...` (matches `_meta`). With
`details=[sample_calls, argument_aliases]` -> `sha256:a8cb...`, while `_meta.capabilities_version`
stays `46eb...`. The contract says "compare it to `_meta.capabilities_version` to skip
re-fetching", but a client that cached the detailed descriptor mismatches forever and
re-fetches on every call.
Fix: compute the version over a canonical base independent of `details`, or document that only
the base hash is comparable.

**M2 -- extract -> export shape mismatch blocks direct piping.**
Extract emits `{id, name, assertion_status}`; export requires `{hpo_id, label, assertion}`.
Feeding raw `aggregated_hpo_terms` fails, although the schema says "Hand it the
aggregated_hpo_terms from an extract call".
Fix: accept both key sets in export (alias `id->hpo_id`, `name->label`,
`assertion_status->assertion`), consistent with the advertised argument-alias policy.

**M3 -- Raw KeyError leaks as the error message.**
A missing key returns `error_code: invalid_input, message: "'hpo_id'"` (a bare Python
KeyError repr), whereas capabilities promise `validation_failed` with a did-you-mean.
Fix: `validation_failed: phenotypes[0] missing 'hpo_id' (got id, name, assertion_status); map id->hpo_id, name->label, assertion_status->assertion`.

**M4 -- `aggregated_hpo_terms` schema is redundant and a footgun.**
Two parallel index schemes: `chunks` / `top_evidence_chunk_idx` (0-based) vs.
`source_chunk_ids` / `top_evidence_chunk_id` (1-based `chunk_id`). And
`score == avg_score == confidence == max_score_from_evidence` (four copies).
Fix: keep one chunk-reference scheme (the `chunk_id`) and one score field.

**M5 -- `include_details=true` is a silent no-op at the default `response_mode`.**
Defaults are `include_details=true` + `response_mode=compact`, but definitions/synonyms only
appear at `standard`/`full`. The default call advertises details and returns none.
Fix: honor `include_details` in compact, or default it to `false` and document the floor.

### LOW

- **L1** -- Empty/whitespace query (`search ""`) returns `success: true, results: []` instead
  of `validation_failed`. Garbage-in silently accepted.
- **L2** -- `compare` `next_commands` emits a non-executable placeholder
  `{"text": "<related phenotype phrase>"}`; fill it or omit.
- **L3** -- `compare` `response_mode=full` adds only `recommended_citation` -- no IC / MICA /
  subsumer breakdown. Full mode under-delivers for this tool.
- **L4** -- `strategy` (chunk_text) has no enum; unknown values are not rejected with a valid
  list (a garbage string was treated as a "semantic" strategy). Valid strategies are not
  enumerated in capabilities.
- **L5** -- Negated chunks omit the `hpo_matches` key entirely instead of `[]` -- schema
  inconsistency across chunk objects.
- **L6** -- Provenance version mismatch: exporter writes `createdBy: phentrieve 0.23.1` while
  the server reports `0.15.1`.
- **L7 (LLM tool)** -- Over-negates "severe intellectual disability **without** regression"
  (marks the ID negated); duplicate `hpo_id` present+negated collides on key; `full`-mode
  payload bloat.
- **L8** -- The advertised argument-alias feature (`query`->`text`, `limit`->`num_results`,
  etc.) is unreachable by a strict MCP client: the tool input schemas declare
  `additionalProperties: false` and require the canonical name, so a schema-validating caller
  cannot send an alias. Aliases only help non-validating callers. Either document this clearly
  or relax the schema for aliased fields.

---

## 5. Part C -- Best-practices alignment

### 5.1 Anthropic MCP / tool-use guidance

| Practice | Status | Notes |
|----------|:------:|-------|
| Descriptions prescriptive about WHEN to call, not just what | PASS | `do_not_use_for`, `next_tools`, canonical_workflow are exemplary |
| Enums for fixed-value parameters | PARTIAL | `response_mode`/`formula` good; `strategy` has no enum (L4) |
| Focused tool set (avoid sprawl) | PASS | 8 well-scoped tools |
| Structured/typed errors, not string-matching | PASS (1 leak) | Strong envelopes; M3 leaks a raw KeyError |
| Do not dump intermediate data into the context window | FAIL | full-mode empty chunks + score dup + next_commands echo (T1); candidate explosion (H1) |
| Stable cache key / version signal | FAIL | M1 -- version hash unstable across `details` |
| Deterministic, parseable output | PARTIAL | M4 dual indexing and id collisions invite consumer bugs |

### 5.2 Google Gemini function-calling guidance (for the internal extraction LLM)

The LLM extractor uses `gemini-3.1-flash-lite`. The over-negation in L7 and the general
negation fragility point to prompt/decoding choices.

| Practice | Status | Notes |
|----------|:------:|-------|
| Detailed function + parameter descriptions | PASS | Surface descriptions are strong |
| Strong typing / enums on the function surface | PARTIAL | Same `strategy` gap (L4) |
| Controlled generation (response schema) for extraction | RECOMMEND | Enforce a response schema on the Gemini call so assertion polarity is a typed enum field, not free text |
| Low temperature for deterministic extraction | RECOMMEND | Verify the internal call pins low temperature; extraction should be reproducible |
| Few-shot for hard linguistic cases | RECOMMEND | Add negation-scope few-shots, especially "X without Y" (do not negate X) and "no X / not X / denies X" (negate X) -- directly targets C1 and L7 |
| Keep parameters flat and simple | PASS | Flat schemas throughout |

---

## 6. Part D -- Remediation roadmap to >= 9/10

Grouped into prioritized workstreams. Each dimension's path to >= 9 is listed.

### Workstream 1 -- Deterministic extractor correctness (Correctness 6 -> 9)

1. **C1**: detect assertions on the original sentence span; never strip negation cues during
   chunk-boundary selection. Add tests for `no X`, `not X`, `does not have X`, `denies X`,
   `without X` (the last must NOT negate X).
2. **H1**: select top-1 (or score-gap-thresholded) candidate per phrase; expose the rest under
   an optional `candidates` field, off by default.
3. **H2**: emit negated matches with `assertion_status: negated` so `excluded` phenopacket
   features can be built deterministically.
4. **L7 / LLM**: add Gemini few-shot examples for negation scope; enforce a typed
   assertion enum via controlled generation; fix the "without <other feature>" over-negation.

### Workstream 2 -- Lossless, pipeable export (Output/schema 5 -> 9; part of Correctness)

5. **H3**: carry `score` in the suggested phenotype objects and into the evidence reference
   (no more `0.0000`).
6. **M2**: accept both phenotype key shapes in `export_phenopacket` (alias
   `id/name/assertion_status` -> `hpo_id/label/assertion`).
7. **M4**: collapse to one chunk-reference scheme (`chunk_id`) and one score field; populate
   offsets consistently (or drop the null `start_char`/`end_char` in `evidence_records`).
8. **id-collision (L7)**: namespace duplicate `hpo_id` by assertion scope (subject vs.
   family_history), or merge into one entry with a per-evidence assertion list.
9. **L5**: emit `hpo_matches: []` on negated chunks rather than omitting the key.
10. **L6**: reconcile the `createdBy` version with the server version (or document the
    core-vs-server split explicitly in the bundle).

### Workstream 3 -- Token efficiency (5 -> 9)

11. **T1**: in `full` mode, drop empty-match `processed_chunks` (or gate behind
    `include_unmatched_chunks`); collapse the four score copies to one; make `next_commands`
    `ids_only` or omit it when the same data already sits in `aggregated_hpo_terms`.
12. **H1** also cuts tokens dramatically (40 terms -> ~3 for the test input).

### Workstream 4 -- Cache contract and discoverability (Discoverability 8 -> 9.5)

13. **M1**: make `capabilities_version` stable across `details` expansion, or document that
    only the base hash is comparable to `_meta.capabilities_version`.
14. **L4**: give `strategy` an enum and reject unknown values with the valid list; enumerate
    valid strategies in capabilities.
15. **M5**: honor `include_details` in compact, or default it to `false` and document the
    response-mode floor for details.
16. **L8**: document (or relax) the argument-alias reachability limitation under strict
    `additionalProperties: false` schemas.
17. Add a one-line "tools load on demand; start with `phentrieve_get_capabilities`" hint to
    the server instructions for cold agents.

### Workstream 5 -- Error handling polish (8 -> 9.5)

18. **M3**: replace raw KeyError reprs with descriptive `validation_failed` + did-you-mean.
19. **L1**: reject empty/whitespace queries with `validation_failed`.
20. **L2**: fill or omit the non-executable `next_commands` placeholder on `compare`.

### Workstream 6 -- Observability, speed, safety (9 -> 9.5 each)

21. **Observability**: add per-phase timing to the LLM extract `_meta` (chunking vs. phase-1
    vs. phase-2b) so the ~13s latency is explainable.
22. **Speed**: optionally warm the lazy embedding model and vector index (diagnostics showed
    both `lazy`) at startup or via a warmup call; document expected LLM-path latency and/or
    stream progress.
23. **Safety & citations**: emit `recommended_citation` in all response modes (currently only
    standard/full) and include the HPO release version in the citation string.

### Definition of done (>= 9 everywhere)

- No false-positive affirmed phenotype from any common negation construction (C1, L7).
- One phenotype per phrase by default; candidate lists opt-in (H1).
- `aggregated_hpo_terms` pipes into `export_phenopacket` without remapping and preserves
  `score` (M2, H3).
- `full`-mode payload contains no empty chunks and no duplicated score fields (T1).
- `capabilities_version` is stable and the warm-cache contract holds (M1).
- Every error is a typed envelope with an actionable message (M3, L1, L4).
- Single chunk-index scheme; consistent offsets; no id collisions (M4, L5, L7).

---

## 7. Appendix -- selected evidence

- C1: `phentrieve_extract_hpo_terms("There is no nystagmus. She does not have ataxia.")`
  returned Nystagmus and Ataxia as `affirmed`; chunk texts were `nystagmus` (start_char 41)
  and `ataxia` (start_char 70), i.e. the negation cue was stripped.
- H1: `phentrieve_extract_hpo_terms("The patient had seizures.")` -> 10 aggregated terms.
- H3: `phentrieve_export_phenopacket` evidence reference read
  `"Phentrieve retrieval confidence: 0.0000, Rank: 1"` for hand-supplied phenotypes lacking
  `score`.
- M1: `get_capabilities()` base -> `capabilities_version: sha256:46eb4ea0bfb20474`
  (matches `_meta`); `get_capabilities(details=[sample_calls, argument_aliases])` ->
  `sha256:a8cb0b8784b97567` while `_meta` stayed `46eb...`.
- M2: `export_phenopacket(phenotypes=[{id, name, assertion_status}])` ->
  `error_code: invalid_input, message: "'hpo_id'"`.
- M5: `search_hpo_terms("seizures", response_mode="full")` included `definition` + `synonyms`;
  the same call in `compact` (default) did not, despite `include_details=true` default.
- L7: LLM extraction of a clinical note marked
  "severe intellectual disability without regression" as `negated`.

All behavior above is from the local instance at the versions listed in the header and may
change as the server is updated; re-run the test matrix after applying fixes.
