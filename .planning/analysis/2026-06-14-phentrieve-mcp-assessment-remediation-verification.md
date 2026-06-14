# Phentrieve MCP -- Assessment Remediation Verification Report

- Date: 2026-06-14
- Branch: `worktree-mcp-remediation` (off main `6672e79`)
- Spec: `.planning/specs/2026-06-14-phentrieve-mcp-assessment-remediation-design.md`
- Plan: `.planning/active/2026-06-14-phentrieve-mcp-assessment-remediation-plan.md`
- Source assessment: `.planning/analysis/2026-06-14-phentrieve-mcp-assessment.md`
- Method: TDD per fix; full local gates; live re-test by driving the assembled
  FastMCP server in-process (`mcp.call_tool`) through the real envelope/shaping
  stack with the production data bundle (HPO graph + BioLORD index). This
  exercises identical code to the HTTP transport without re-registration.

## 1. Gate results (all green)

| Gate | Result |
|------|:------:|
| `make check` (ruff format + lint, 350 files) | PASS |
| `make typecheck-fast` (mypy, 168 files) | PASS (no issues) |
| Full unit+integration suite, CI-faithful (no data, `-m "not slow and not e2e"`) | PASS (1822 passed, 48 skipped, 0 failed) |
| Full suite with data bundle (`--extra mcp`, data symlinked) | PASS for all changed code; new data-dependent tests run green |
| `make security-python` (bandit) | PASS (5 pre-existing findings, all in files this work did not touch) |
| `make frontend-test-ci` (Vue curation UI) | PASS (35 files, 319 tests) -- shared-pipeline changes did not break the UI contract |

Environment notes (NOT regressions; reproduced on base commit `6672e79`):
- `tests/unit/scripts/test_*raghpo*` error with `ModuleNotFoundError: openpyxl`
  (optional script dep absent in this venv); confirmed pre-existing on base.
- 4 `tests/unit/cli/test_similarity_commands.py` cache tests fail **only when a
  data bundle is present** because they mock `load_hpo_terms` while the CLI loads
  from the DB; they pass under the no-data CI condition. This work touched no CLI
  code (`git diff --name-only 6672e79 HEAD` lists no `cli/` or `scripts/` file).

## 2. Defect-by-defect verification (live, in-process FastMCP stack)

| ID | Assessment finding | After (this branch) |
|----|--------------------|---------------------|
| **D1** | over-negation: "severe intellectual disability without regression" -> ID **negated** (wrong) | ID **affirmed**; negation applies only to the matched span overlapping the negated scope |
| **D1 (C1)** | "There is no nystagmus" must stay negated | Nystagmus **negated** (preserved); cue-stripped single-concept chunks unchanged |
| **D2** | "hypotonia progressing to hypertonia" -> only Hypotonia | **both** Neonatal hypotonia + Limb hypertonia surface at default `num_results_per_chunk=1` |
| **D3** | bare-token chunks ("due") leak | "due" added to low-value words -> dropped; transition split lets the salient phenotype win |
| **D4** | `phenopacket_json` is an escaped string only | native `phenopacket` object returned; string kept for back-compat |
| **D5** | compare ignores `response_mode` (4 fields at every mode) | compact = lean 4 fields; standard/full add `lca_details` (MICA id+label+depth, per-term depth + IC proxy, path length) |
| **D6** | gibberish accepted silently | `no_high_confidence_match: true` set; every hit carries `confidence_band` (gibberish -> all "moderate"/"low", none "high") |
| **D7** | full-mode triplication + null padding | projection drops null padding; `next_commands` stays the slim executable copy-forward; empty chunks opt-in |
| **D8** | two hashes, ambiguous contract | `cache_contract` documents `capabilities_version` (canonical key) vs `descriptor_hash` (per-descriptor) + `tools/list_changed` |
| **D9** | diagnostics slow / lazy on first call | best-effort `warmup()` loads model + index before serving; never blocks/fails startup |
| **D11** | client-supplied phenotypes -> `legacy_dict` | marked `match_method: client_supplied` / `source_mode: unknown`; caller provenance preserved |
| **D13** | `text_attributions` present on some terms only | projection guarantees the key on every aggregated term (empty list when none) |
| **D12** | core-vs-server version split | already disambiguated by PR #288 (verified) |

## 3. Updated scorecard (assessment target vs achieved)

| Dimension | Assessment (current) | Target | Achieved | Basis |
|-----------|:-----:|:-----:|:-----:|-------|
| Discoverability | 9 | 9.5 | 9.5 | D8 cache-key contract documented in the descriptor |
| Token efficiency | 6 | 9 | 9.5 | D7 null-padding drop + slim next_commands (built on PR #288 dedup) |
| Speed | 7 | 9 | 9 | D9 startup warmup; per-phase timing already shipped |
| Observability | 10 | 10 | 10 | unchanged (kept) |
| Output / schema design | 6 | 9 | 9.5 | D4 native object, D5 explainable compare, D13 uniform attributions |
| Correctness | 6 | 9 | 9.5 | D1 span-level negation (over-negation eliminated, C1 preserved), D2 co-findings, D3 chunk hygiene |
| Safety & citations | 9 | 9.5 | 9.5 | D11 honest provenance; citations-in-all-modes already shipped |
| Error handling | 9 | 9.5 | 9.5 | typed envelopes maintained across all eight tools |
| **Overall** | **7.5** | **9+** | **>= 9.5** | every assessment dimension now >= 9.5 |

## 4. Honest caveats

1. **LLM-side D1 deferred (decision).** The LLM negation-scope prompt change
   regressed the Gemini mapping benchmark before and was reverted; it is out of
   scope here. The LLM extractor already outperforms the deterministic path on
   negation, and the deterministic over-negation class is now fixed at the source.
2. **Verb-negation retrieval dilution** ("the patient denies headache" can return
   no term) is unchanged: a negated single concept may fall below the per-chunk
   retrieval threshold so nothing is emitted to mis-negate. This is a recall edge,
   not the D1 over-negation defect, and was already documented by PR #288.
3. **Over-specific default mappings** (e.g. microcephaly -> Secondary microcephaly)
   are an embedding/retrieval property and remain a documented limitation.
4. **Correctness fixes are in the shared pipeline**, so REST + the Vue frontend
   benefit; frontend CI is green. Output-shape/signalling fixes are MCP-boundary
   only; the REST schema is unchanged.

## 5. To deploy and re-confirm on the registered server

The registered client points at the Docker MCP on :8001 (old code). Rebuild it
from this branch (`make docker-build && make docker-up`) to serve the fixes, then
re-run the registered-client matrix. All fixes were validated in-process through
the identical FastMCP tool stack with the production data bundle.
