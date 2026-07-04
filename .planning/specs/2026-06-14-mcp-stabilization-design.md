# Phentrieve MCP Server -- Stabilization Design (retroactive)

- Date: 2026-06-14
- Type: retroactive design spec (written 2026-07-03, backfilled to close the
  planning-hygiene gap noted by
  `.planning/specs/2026-07-03-extraction-contract-v2-and-finalization-design.md`
  section 1)
- Source analysis: `.planning/analysis/2026-06-14-mcp-stabilization-plan.md`
  ("Phentrieve MCP Server -- Assessment & Stabilization Plan", 2026-06-14)
- Shipped by: PR #291
- Status: SHIPPED in PR #291 (v0.24.0)

## Why this exists

PR #291 implemented all 14 findings from the 2026-06-14 assessment/stabilization
analysis in one atomic, test-first commit series, but the `.planning/` tree never
recorded a design spec, a completed-plan execution record, or a verification
report for it -- unlike the earlier hardening (PR #288) and remediation efforts,
which each have all three. This document backfills the missing design spec so
PR #291 has the same planning-tree parity. The companion completed-plan and
verification documents are backfilled alongside it (see
`.planning/completed/2026-06-14-mcp-stabilization-plan.md` and
`.planning/analysis/2026-06-14-mcp-stabilization-verification.md`).

The underlying assessment found the Phentrieve MCP server already reference-quality
on discoverability, observability, and caching, but held below a consistent >9/10 by
two functional-correctness bugs (B1, B2), LLM extraction-quality defects rooted in a
conflated `category` enum (LLM-1, LLM-2), a token-redundancy cluster (R1, R2, R3),
and several consistency/polish gaps (B3, B4, D1-D4, Q1). Every finding was traced to
an exact `file:line` in the source analysis with a minimal, independent fix -- none
of the 14 fixes were interdependent, so they shipped as 14 separate atomic commits
(plus incidental test-infra fixes surfaced while running the full suite).

## Goal

Lift every MCP-only quality dimension (discoverability, token efficiency, latency,
observability, caching contract, schema ergonomics, safety/compliance) toward or
above 9/10, and lift LLM extraction accuracy from ~6/10 to ~9/10, without touching
anything outside the traced `file:line` sites and without introducing interdependent
changes that would force a single large commit.

## Decisions (locked, per the source analysis section 6 roadmap)

1. Fix functional-correctness bugs first (P0): B1, B2, then the LLM extraction
   defects LLM-1/LLM-2, since LLM-1/LLM-2 share one root cause (the conflated
   `category` enum) and gate the largest accuracy jump.
2. Token efficiency and DX next (P1): R1, R2, B3, D4, D3.
3. Consistency and polish last (P2): D1, D2, R3, Q1, B4.
4. Schema/descriptor changes intentionally roll `capabilities_version` -- this is
   the correct warm-client signal per the server's own cache-key contract, not a
   side effect to avoid.
5. Each finding ships as one minimal, independently testable commit; coverage-
   improving tests accompany every fix (per repo `AGENTS.md` discipline).

## The 14 findings (ordered by the roadmap in analysis section 6)

| ID | Dimension | One-line fix |
|---|---|---|
| B1 | search correctness | Band-based `no_high_confidence_match`: flag `true` whenever no result has a `high` confidence band, instead of short-circuiting to `false` on an empty (threshold-emptied) result set. |
| B2 | `chunk_text` capability | Lazy-load the cached embedding model singleton when the resolved chunking config needs a `sliding_window` stage, so all 7 advertised strategies work instead of only `simple`. |
| LLM-1 | extraction accuracy | Drop `family_history` from `ACTIONABLE_CATEGORIES` and key term dedup on `(term_id, experiencer, assertion)`, eliminating the self-contradictory present+negated pair on one HPO id. |
| LLM-2 | extraction accuracy | Add orthogonal `experiencer`/`assertion`/`negated_qualifier` fields to the Phase-1 schema (declared before the legacy `category` label so Gemini reasons about them first), update the prompt with a negation-scope rule + few-shots, and stop the coarse chunk-status override for the LLM path. |
| R1 | token efficiency | Make the parsed `phenopacket` object the single canonical form; gate the serialized `phenopacket_json` string to `standard`/`full` response modes. |
| R2 | token efficiency | Cap/compact the `after_extract` next-commands pre-fill under `minimal`/`compact` response modes instead of re-serializing the full term list into the shaping-exempt `_meta`. |
| B3 | error ergonomics | Give blank/whitespace `text` a value-level validation envelope (the real message) instead of misrouting it through the unknown-argument-name template with `allowed_values` set to parameter names. |
| D4 | error ergonomics | Map `not_found`/`ambiguous_query` to `recovery_action: resolve_identifier` with a `search` next-command, instead of the generic `reformulate_input`. |
| D3 | observability | Have `diagnostics` probe the live model/index caches and report `loaded \| loading \| cold \| error`, instead of hardcoding `"lazy"`. |
| D1 | consistency | Make `text_attributions` always present as an array (`[]` = semantic-only match, no literal span), added to the shaping `_ALWAYS_KEEP_EMPTY` set. |
| D2 | compare transparency | Rename `compare`'s `ic_proxy` to `normalized_depth` and update its description to say it is a structural proxy (`depth/max_depth`), not corpus information content. |
| R3 | token efficiency | Cap synonyms returned in the response projection to 10, with a `synonyms_truncated` count, without capping the list used internally for attribution matching. |
| Q1 | extract precision | Raise deterministic `extract`'s default `num_results_per_chunk` from 1 to 3 so a diluted top-1 parent match cannot silently drop the correct child term. |
| B4 | discoverability/docs | Reconcile the alias-policy documentation with observed client-dependent behavior (argument aliasing depends on client strictness; canonical names are always safe). |

## Scope boundary

- In scope: the MCP protocol surface (`api/mcp/`) and the in-repo LLM extraction
  pipeline (`phentrieve/llm/`) as traced by the source analysis.
- Out of scope (per the source analysis section 1 "Scope boundary"): retrieval-model
  embedding accuracy (e.g. acronym collisions), which is embedding-model behavior,
  not a server or pipeline defect.

## Gate

Per repo `AGENTS.md`: `make check`, `make typecheck-fast`, `make test` on every
commit; `make ci-local` + `make security-python` before push. The PR body records
`make test` stable across 10 runs (1880 passed) after eliminating flakes surfaced
by running the full suite.
