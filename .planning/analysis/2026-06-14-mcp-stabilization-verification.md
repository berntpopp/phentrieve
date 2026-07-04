# Phentrieve MCP Server -- Stabilization Deep Re-Verification (2026-07-03)

- Date: 2026-07-03
- Type: verification record, backfilled alongside the design spec and completed
  plan for PR #291 -- see `.planning/specs/2026-06-14-mcp-stabilization-design.md`
  and `.planning/completed/2026-06-14-mcp-stabilization-plan.md`
- Subject: PR #291, "MCP stabilization plan", merged as v0.24.0 (2026-06-14)
- Method: four code-grounded passes over current `main` (2026-07-03), re-checking
  every finding against the file:line sites in the source analysis
  (`.planning/analysis/2026-06-14-mcp-stabilization-plan.md`)
- Result: all 14 findings verified SHIPPED and test-pinned, including LLM-1/LLM-2
  -- which two earlier 06-14 verification passes had incorrectly recorded as
  "deferred". This record is the corrected, up-to-date status.

## Verified status: all 14 findings shipped

| ID | Shipping commit | Verified against current code |
|---|---|---|
| B1 | `e3ba664` | `api/mcp/confidence.py` uses the band-based check (`not any(... == "high" ...)`), not the inverted `bool(results) and top < HIGH_FLOOR` guard. |
| B2 | `b00a2ff` | `api/mcp/service_adapters.py`'s `chunk_text` path lazy-loads the cached embedding model when the resolved config needs a `sliding_window` stage. |
| LLM-1 | `566cb97` | `family_history` is absent from `ACTIONABLE_CATEGORIES` (`phentrieve/llm/pipeline_phase1.py`); dedup keys on `(term_id, experiencer, assertion)`. |
| LLM-2 | `6c30502` | Phase-1 schema carries `experiencer`/`assertion`/`negated_qualifier`; `two_phase/en.yaml` is at prompt `v3.1.0` with the negation-scope rule + few-shots. |
| R1 | `a96a7ec`, `c2645ae` | `export_phenopacket` returns one canonical `phenopacket` object; `phenopacket_json` is gated to `standard`/`full`. |
| R2 | `ad94514` | `after_extract`'s next-commands pre-fill is capped/compacted under `minimal`/`compact`. |
| B3 | `bf1ba2d` | Blank/whitespace `text` returns a value-level envelope with the real validator message. |
| D4 | `b8f82fc` | `not_found`/`ambiguous_query` map to `recovery_action: resolve_identifier` with a search next-command. |
| D3 | `666f5d5` | `diagnostics` probes `LOADED_SBERT_MODELS` / `MODEL_LOADING_STATUS` / `LOADED_RETRIEVERS` and reports live state. |
| D1 | `02104d2` | `text_attributions` is in `shaping._ALWAYS_KEEP_EMPTY`; always present as an array. |
| D2 | `4f6bca1` | `compare`'s `ic_proxy` is renamed `normalized_depth` with an honest structural-proxy description. |
| R3 | `7645bad` | Response-projection synonyms are capped to 10 with a `synonyms_truncated` count. |
| Q1 | `2bdc378` | Deterministic `extract`'s `DEFAULT_EXTRACT_NUM_RESULTS` is 3. |
| B4 | `62bc14e` | Alias-policy docs state client-dependent behavior in `api/mcp/capabilities.py`. |

Shipped release: v0.24.0 (CLI 0.24.0 / API 0.16.0 / Frontend 0.15.0), 2026-06-14.
Currently released: v0.24.1 (CLI 0.24.1 / API 0.16.1 / Frontend 0.16.1), 2026-07-03
-- a dependency/security/frontend-footer maintenance release layered on top; it did
not touch or regress any of the 14 findings above.

## Why the earlier "LLM-1/LLM-2 deferred" record was wrong

Two verification documents committed on 2026-06-14 (before PR #291 finished
landing all commits) recorded LLM-1 and LLM-2 as deferred to a follow-up. The
2026-07-03 deep re-verification re-read the current code directly (not the
earlier docs) and confirmed both commits (`566cb97`, `6c30502`) are present on
`main` and their behavior is test-pinned. This record supersedes that stale
"deferred" status; no further action was needed for LLM-1/LLM-2 themselves.

## Residual behaviors left on the table (these seeded the v2 effort)

PR #291 fixed the acute LLM defects (removed the deterministic-detector override
on the LLM path; stopped the contradictory present+negated pair on one id) but did
**not** deliver a full assertion/experiencer contract. Three residuals remain,
each confirmed against current code as of 2026-07-03:

1. **Assertion is advisory, not load-bearing.** The LLM schema carries
   `experiencer`/`assertion` axes, but Phase 2 re-derives polarity from the legacy
   `category` enum (`phentrieve/llm/pipeline_phase2.py:273-279`) instead of
   consuming the model's own `assertion` value. The axis exists but does not
   drive behavior.
2. **Family-history mentions are silently dropped**, not surfaced. They are
   removed from the actionable set before mapping (`phentrieve/llm/pipeline.py:
   340-343`) and dropped again at MCP export (`api/mcp/service_adapters.py:
   381-384`). This correctly prevents the contradictory pair (LLM-1's fix), but
   pedigree information vanishes with no trace instead of surfacing in a
   dedicated list.
3. **`negated_qualifier` ("X without Y") is metadata-only.** The qualifier phrase
   is captured as a string (`phentrieve/text_processing/full_text_service.py:
   457-463`) but Y is never mapped to an HPO id, so it is not a
   machine-actionable exclusion.

## Issue #289 status

Issue #289 (LLM assertion polarity / "X without Y" / family-history) remains
**open**, and correctly so. PR #291 reduced the acute symptoms described above but
did not deliver the full contract (load-bearing assertion, family surfaced instead
of dropped, qualifier mapped to an excluded term). That full contract is the
`extraction-contract-v2` effort (see
`.planning/specs/2026-07-03-extraction-contract-v2-and-finalization-design.md`);
issue #289 stays open until that effort's Phase 3 closes it with `Closes #289`.

## Planning-hygiene gap also found and remediated

Separately from the code-level residuals, this re-verification found PR #291 had
no spec / completed-plan / verification artifacts in `.planning/` -- only its
source analysis doc (`.planning/analysis/2026-06-14-mcp-stabilization-plan.md`),
committed roughly three weeks after the PR merged. This gap is what the companion
design spec and completed plan (backfilled 2026-07-03) close, restoring parity
with the earlier hardening (PR #288) and remediation efforts, which each already
had all three artifact types.

## Gate

This is a documentation-only verification record; no application code changed.
Per repo `AGENTS.md`, `make check` / `make typecheck-fast` / `make test` remain
green (trivially, for docs) and are re-run for parity before this document's
commit lands.
