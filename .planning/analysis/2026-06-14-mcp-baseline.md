# MCP Hardening -- Live "Before" Baseline

- Date: 2026-06-14
- Target: live Docker MCP at http://localhost:8001/mcp/ (server 0.15.1, pre-fix)
- Purpose: capture reproductions of each defect before applying fixes, for
  before/after comparison in the verification report.

## C1 -- negation scope lost (CRITICAL)

`extract("There is no nystagmus. She does not have ataxia.")` ->
- chunk 1 `"nystagmus"` start_char=12, `status: affirmed` (WRONG; "no" stripped)
- chunk 2 `"ataxia"` start_char=41, `status: affirmed` (WRONG; "does not have" stripped)
- 20 aggregated terms, all affirmed.

## H1 -- candidate explosion (HIGH)

`extract("The patient had seizures.")` -> 10 aggregated terms:
Seizure, Symptomatic seizures, Focal-onset seizure, Focal cognitive seizure,
Seizure cluster, Epileptic aura, Thinking-induced seizure, Cognitive epileptic
aura, Focal motor seizure with version, Non-epileptic seizure.
(num_results_per_chunk default = 10.)

## H2 + L5 -- negated findings dropped (HIGH)

`extract("The patient denies headache.")` ->
- chunk 1 `status: negated`, NO `hpo_matches` key (dropped by compact shaping).
- `aggregated_hpo_terms: []` (num_aggregated_hpo_terms=0).
- No Headache term emitted, not even excluded. `excluded:true` features cannot be
  built from this tool.

## M2 + M3 -- export key mismatch + raw KeyError (MEDIUM)

`export_phenopacket(case_id="BASELINE-TEST", phenotypes=[{id, name,
assertion_status}])` ->
`{"success":false,"error_code":"invalid_input","message":"'hpo_id'"}`
(bare Python KeyError repr; promised validation_failed + did-you-mean).

## H3 + T1 -- export confidence lost + next_commands bloat

The nystagmus extract `_meta.next_commands` re-emits all 20 phenotypes as
`{hpo_id, label, assertion}` with NO `score` -> phenopacket evidence reads
`confidence: 0.0000`. The full term list is duplicated from aggregated_hpo_terms.

## M1 -- capabilities_version

All `_meta.capabilities_version` = `sha256:46eb4ea0bfb20474` (base hash). The
detailed descriptor reports a different hash in its body (to be re-confirmed at
verification).

## Schema redundancy (M4) observed in compact aggregated terms

Each term carries `score == avg_score == confidence == max_score_from_evidence`
and dual index schemes `chunks`/`top_evidence_chunk_idx` (0-based) +
`source_chunk_ids`/`top_evidence_chunk_id` (1-based).
