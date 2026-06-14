# Phentrieve MCP Hardening -- Verification Report

- Date: 2026-06-14
- Branch: worktree-feat+mcp-hardening (off main)
- Spec: `.planning/specs/2026-06-14-phentrieve-mcp-hardening-design.md`
- Plan: `.planning/active/2026-06-14-phentrieve-mcp-hardening-plan.md`
- Baseline (before): `.planning/analysis/2026-06-14-mcp-baseline.md`
- Method: TDD per fix; full local gates; live re-test through the real FastMCP
  HTTP stack (worktree server on :8734 via fastmcp.Client). The Docker server on
  :8001 (old code) was left untouched and provided the "before" snapshot.

## 1. Gate results (all green)

| Gate | Result |
|------|:------:|
| `make check` (ruff format + lint) | PASS |
| `make typecheck-fast` (mypy, 167 files) | PASS (no issues) |
| `make test` (unit + integration) | PASS (1827 passed, 43 skipped) |
| `make ci-local` (Python quality + frontend CI: format-check, lint, typecheck, test, build, i18n) | PASS (EXIT 0) |
| `make security-python` (bandit) | PASS (EXIT 0; 6 pre-existing Low findings, none in changed MCP code) |
| `make frontend-test-ci` | PASS (317 tests) -- shared-pipeline changes did not break the curation UI |

Note: 3 LLM integration tests flaked once under xdist parallelism (prompt-loader
cache + parallel ordering); they pass deterministically in isolation and the full
suite passed on re-run. Pre-existing flakiness class, not introduced here.

## 2. Defect-by-defect before/after (live, full FastMCP stack)

| ID | Before (Docker :8001, old code) | After (worktree :8734, new code) |
|----|----------------------------------|----------------------------------|
| **C1** | "There is no nystagmus. She does not have ataxia." -> Nystagmus **affirmed**, Ataxia **affirmed** (false positives) | -> Nystagmus **negated**, Ataxia **negated** |
| **H1** | "The patient had seizures." -> **10** aggregated sibling terms | -> **1** term (Seizure) |
| **H2** | negated findings not surfaced (no excluded features) | prepositional-negation findings emit as `assertion_status: negated` (excluded features buildable) |
| **H3** | export evidence reads `confidence: 0.0000` | score preserved: evidence reads `0.9100` (no `0.0000`) |
| **M1** | base `46eb...` vs details `a8cb...`; `_meta` stayed base -> warm cache breaks | base == details == `_meta` (`sha256:d8098d37...`); `descriptor_hash` separate |
| **M2** | export of `{id,name,assertion_status}` -> error | success; raw aggregated_hpo_terms pipe unchanged |
| **M3** | `error_code: invalid_input, message: "'hpo_id'"` (raw KeyError) | `validation_failed: phenotypes[0] missing 'hpo_id' (got keys: ['name']); map id->hpo_id ...` |
| **M4** | term carries 4 score copies + dual chunk-index schemes | one `score`, one scheme (`chunk_ids` + `top_evidence_chunk_id`); keys: assertion, chunk_ids, evidence_count, hpo_id, label, rank, score, top_evidence_chunk_id |
| **M5** | `include_details=true` no-op in compact | default `false`; honored in compact when explicitly set |
| **T1** | full-mode empty chunks + score dup + term echo | empty chunks dropped (opt-in), score dedup, slim next_commands |
| **L1** | `search "   "` -> `success: true, results: []` | `validation_failed` |
| **L2** | `next_commands` `text="<surrounding clinical text>"` / `"<related phenotype phrase>"` | executable: export single hit / cross-check compare with alternate formula |
| **L4** | bogus strategy silently treated as a strategy | `validation_failed: ... must be one of: simple, detailed, semantic, ...`; strategies listed in capabilities |
| **L5** | negated/empty chunk omits `hpo_matches` key | `hpo_matches: []` always present |
| **L6** | `createdBy: phentrieve 0.23.1` vs server 0.15.1 (confusing) | `createdBy: phentrieve-core <version>` (disambiguated) |
| **L8** | aliases advertised as accepted | capabilities states input schemas are additionalProperties:false; strict clients must use canonical names |
| **safety** | citation only in standard/full | `recommended_citation` in all modes (incl. compact/minimal) + export; embeds HPO release version when the data root is configured (e.g. `Human Phenotype Ontology (HPO v2026-02-16), ...`) |
| **observability** | no per-phase LLM timing | `_meta.observability.phase_timings` surfaced; `latency_profile` in capabilities |
| **L7 (LLM)** | over-negates "severe intellectual disability without regression" | prompt v3.1.0 adds a negation-scope rule + contrastive few-shot (BENCHMARK-GATED, see sec. 4) |

## 3. Scorecard (target vs achieved)

| Dimension | Before | Target | After | Basis |
|-----------|:-----:|:-----:|:-----:|-------|
| Discoverability | 8 | 9.5 | 9.5 | M1 stable hash, L4 enum + capabilities listing, M5 fixed, L8 documented, cold-start hint |
| Token efficiency | 5 | 9 | 9.5 | H1 (10->1), M4 score/index dedup, T1 empty-chunk drop, slim next_commands |
| Speed | 8 | 9 | 9 | phase_timings make latency explainable; latency_profile documented (warmup deferred as documented) |
| Observability | 9 | 9.5 | 9.5 | per-phase LLM timing surfaced |
| Output / schema design | 5 | 9 | 9.5 | single score + single index, normalized keys, hpo_matches always present, lossless pipeable export |
| Correctness | 6 | 9 | 9 | C1 negation fixed + emitted (H2), H1 best-match; LLM L7 fix gated; deterministic single-chunk "without" remains a screening-tool limitation |
| Safety & citations | 9 | 9.5 | 9.5 | citation in all modes + HPO version; research-use posture intact |
| Error handling | 8 | 9.5 | 9.5 | M3 typed did-you-mean, L1 blank rejected, L4 enum errors |
| **Overall** | **7** | **9+** | **9.4** | every dimension >= 9; bounded by the gated LLM item + the deterministic "without" edge |

## 4. Honest caveats

1. **LLM negation fix (L7) is benchmark-gated and not behaviorally validated here.**
   The Gemini key lives in the Docker container, not the worktree env, so the
   mapping benchmark could not run locally. The change is isolated in one commit
   (`feat(llm): negation-scope rule + contrastive few-shot ...`) with static
   contract tests. Before merging it, run the mapping benchmark
   (`phentrieve/benchmark/extraction_benchmark.py`) and revert THAT commit alone
   if accuracy regresses (prompt changes regressed in PR #261).
2. **Deterministic "X without Y" single-chunk over-negation** is not fully fixed
   for the standard extractor: when the chunker keeps "X without Y" as one chunk,
   the chunk-level assertion still marks it negated. The CRITICAL false-positive
   class (no X / not X / does not have X) IS fixed. The "without" case is handled
   in the LLM path (the precision tool); the deterministic tool is screening-grade.
3. **Verb-negation retrieval dilution** ("patient denies headache" as one chunk)
   can keep the negated concept below the retrieval threshold; the LLM tool
   handles such concept extraction. Prepositional-negation findings DO surface as
   negated (verified).
4. **Citation HPO version** requires the data root to be resolvable
   (`PHENTRIEVE_DATA_ROOT_DIR`, set in the Docker deployment); otherwise it
   degrades to `(HPO)` without the version.
5. **Schema/token changes are MCP-boundary only.** The shared service, REST
   `AggregatedHPOTermAPI`, and the Vue curation frontend are unchanged
   (frontend CI green). Correctness fixes (negation) are in the shared pipeline so
   all consumers benefit.

## 5. To deploy and re-confirm on :8001

The Docker `phentrieve-phentrieve_api-1` container still runs the old code. Rebuild
it from this branch (`make docker-build && make docker-up`) to serve the fixes on
:8001, then re-run the registered-client matrix. All fixes were validated on the
worktree server (:8734) through the identical FastMCP HTTP stack.
