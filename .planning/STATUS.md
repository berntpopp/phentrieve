# Status

Planning status now lives in `.planning/README.md` plus the directory layout in
`.planning/active/`, `.planning/completed/`, `.planning/archived/`, and
`.planning/analysis/`.

Use `.planning/README.md` as the current planning index.

## Snapshot (2026-07-03)

- **Shipped:** MCP stabilization (PR #291), released as v0.24.1 (CLI 0.24.1 /
  API 0.16.1 / Frontend 0.16.1). All 14 findings (B1, B2, LLM-1, LLM-2, R1,
  R2, B3, D4, D3, D1, D2, R3, Q1, B4) verified shipped and test-pinned; see
  `.planning/analysis/2026-06-14-mcp-stabilization-verification.md`.
- **Active:** `extraction-contract-v2` -- making the LLM `assertion`/
  `experiencer` axes load-bearing, surfacing family-history findings and
  "X without Y" excluded terms, gated by a present-only no-regression
  benchmark. Phase 0 (planning cleanup) + Phase 1 (benchmark safeguard) plan
  is in `.planning/active/2026-07-03-extraction-contract-v2-phase-0-1-plan.md`.
  Tracks issue #289 (stays open until this effort's Phase 3 closes it).
