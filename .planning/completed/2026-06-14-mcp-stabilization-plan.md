# Phentrieve MCP Server -- Stabilization Execution Record (retroactive)

- Date: 2026-06-14
- Type: retroactive completed-plan (written 2026-07-03, backfilled alongside the
  design spec and verification doc -- see
  `.planning/specs/2026-06-14-mcp-stabilization-design.md`)
- Source analysis: `.planning/analysis/2026-06-14-mcp-stabilization-plan.md`
- Shipped by: PR #291 ("Implements the MCP stabilization plan end-to-end")
- Status: COMPLETE, merged as v0.24.0 (CLI 0.24.0 / API 0.16.0 / Frontend 0.15.0),
  2026-06-14 per `CHANGELOG.md`

## Provenance of this mapping

The finding-to-commit mapping below is copied verbatim from the PR #291 body
(`gh pr view 291 --json body`, fetched 2026-07-03) and cross-checked against
`git log --oneline` in this repository -- every commit short hash listed below was
confirmed present on the branch history. This satisfies the "completed execution
plan" artifact that PR #291 shipped without (the hardening PR #288 and remediation
efforts each have one; #291 did not, until this backfill).

## Execution record: one finding = one atomic, test-first commit

### P0 -- correctness

| ID | Commit | One-line |
|---|---|---|
| B1 | `e3ba664` | Band-based `no_high_confidence_match`: a threshold-emptied result set now correctly reads `true` (was the inverted `false`). |
| B2 | `b00a2ff` | `chunk_text` lazy-loads the cached embedding model (6/7 strategies need it); a load failure is `temporarily_unavailable`, not `invalid_input`. |
| LLM-1 | `566cb97` | Drop `family_history` from `ACTIONABLE_CATEGORIES` + experiencer-keyed dedup `(term_id, experiencer, assertion)` + export drops family-history -- kills the contradictory present+negated pair on one id. |
| LLM-2 | `6c30502` | Additive `experiencer`/`assertion`/`negated_qualifier` axes on the Phase-1 schema, threaded to the term; drop the coarse chunk-status override for the LLM path; prompt negation-scope rule + few-shots (prompt v3.0.0 -> v3.1.0). |

### P1 -- token efficiency & DX

| ID | Commit | One-line |
|---|---|---|
| R1 | `a96a7ec` + `c2645ae` | Single canonical `phenopacket` object; `phenopacket_json` gated to standard/full; object kept whole at minimal. |
| R2 | `ad94514` | Cap/compact the `after_extract` export pre-fill under minimal/compact (was duplicating 25 terms into exempt `_meta`). |
| B3 | `bf1ba2d` | Blank `text` -> value-level envelope (no `allowed_values`=param-names). |
| D4 | `b8f82fc` | `not_found`/`ambiguous_query` -> `recovery_action: resolve_identifier` + a search next-command. |
| D3 | `666f5d5` | Diagnostics probes live caches (loaded\|loading\|cold\|error). |

### P2 -- consistency & polish

| ID | Commit | One-line |
|---|---|---|
| D1 | `02104d2` | `text_attributions` always an array (empty = semantic). |
| D2 | `4f6bca1` | `ic_proxy` -> `normalized_depth` (honest structural proxy). |
| R3 | `7645bad` | Cap response synonyms to 10 (+`synonyms_truncated`). |
| Q1 | `2bdc378` | Deterministic extract `num_results_per_chunk` default 1 -> 3. |
| B4 | `62bc14e` | Reconcile alias-policy docs with client-dependent behavior. |

### Test-infra (incidental, pre-existing flakes surfaced by running the full suite)

| Commit | One-line |
|---|---|
| `24fc639` | Fix `tests/unit/api` package shadowing via a proper test-package hierarchy (reverts an interim importlib approach `480ef55` that caused duplicate-module classes). |
| `0589391` | Resolve `LLMPhenotype` forward ref eagerly (`model_rebuild`) -- fixes an intermittent pydantic validation race under parallel grouped phase-1. |
| `a71b416` | Pin grouped phase-1 to 1 worker in the FIFO-fake integration tests. |

## Test plan (from the PR body, as executed)

- [x] `make check`, `make typecheck-fast`, `make test` (1880 passed)
- [x] `make ci-local` (Python quality + frontend), `make security-python`
- [x] `make test` stable 10/10 runs (flakes eliminated)
- [ ] Live MCP re-test of the LLM acceptance cases (qualifier=regression;
      family-history dropped) -- needs a live Gemini call; deterministic
      acceptance (B1/B2/B3/D3/D4 + LLM-1/2 pipeline) is covered by unit/
      integration tests. This box stayed unchecked in the PR; the 2026-07-03
      deep re-verification (see the companion verification doc) confirms it was
      later covered.

## Notes

Schema/descriptor changes intentionally rolled `capabilities_version` (B4); the
hash is computed dynamically, with no pinned fixture to update.
