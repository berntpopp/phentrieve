# LLM Evidence Validation Enriched Mapping PR Regression

Date: 2026-05-25

PR: https://github.com/berntpopp/phentrieve/pull/261

Status: superseded, not recommended for merge

## Intent

The PR intended to make the two-phase LLM pipeline more source-faithful and
auditable:

- Validate Phase 1 evidence before Phase 2 mapping.
- Keep Phase 1 extracted phrases faithful to source text.
- Move conservative phrase normalization and abbreviation expansion into Phase
  2 retrieval-query preparation.
- Enrich Phase 2 mapping prompts with compact HPO candidate metadata.
- Add compatible trace and metadata fields for debugging.
- Avoid reranking, hybrid retrieval, and new retrieval subsystems.

## Decision

Do not merge PR #261. Close it as superseded by benchmark regression evidence.

The source-faithful extraction direction is still useful, but this PR changes
too much at once and regresses mapping quality. Future work should restart from
`main` as smaller PRs with same-command A/B gates before broadening scope.

## Same-Command Focused A/B

Run date: 2026-05-25

Command shape:

- Dataset: CSC
- Documents: `CSC_91`, `CSC_71`, `CSC_18`, `CSC_107`, `CSC_4`, `CSC_85`
- Model: `gemini-3.1-flash-lite`
- Seed: `123`
- Same real index/data setup for both branches
- Outputs were written to `/tmp/phentrieve-bench-compare`

Compared commits:

- `main`: `ad984b5`
- PR branch: `5cd3fce`

Strict ID micro metrics:

| Branch | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `main` | 53 | 18 | 20 | 0.746 | 0.726 | 0.736 |
| PR #261 | 43 | 28 | 30 | 0.606 | 0.589 | 0.597 |

Per-document F1:

| Document | `main` F1 | PR #261 F1 | Delta |
|---|---:|---:|---:|
| `CSC_91` | 0.556 | 0.556 | 0.000 |
| `CSC_71` | 0.762 | 0.560 | -0.202 |
| `CSC_18` | 0.737 | 0.632 | -0.105 |
| `CSC_107` | 0.824 | 0.632 | -0.192 |
| `CSC_4` | 0.696 | 0.500 | -0.196 |
| `CSC_85` | 0.783 | 0.651 | -0.132 |

## Root Cause Summary

The PR made Phase 1 more source-faithful. That caused Phase 1 to emit raw source
phrases such as abbreviations and short modifiers where prior behavior often
emitted normalized clinical noun phrases. Examples observed during debugging:

- `GTC` instead of `generalized tonic-clonic seizures`
- `unilateral` instead of `unilateral seizures`
- `BWGS` instead of `Bland-White-Garland syndrome`
- `throbbing` instead of `throbbing headache`
- `sparse axillary` instead of `sparse axillary hairs`

Phase 2 did not recover enough of that lost clinical normalization. Focused
fixes improved selected cases, especially `CSC_91`, but the branch still
regressed five of the six focused documents against current `main`.

## Follow-Up Recommendation

Restart from `main` and split the concept into smaller, independently gated
changes:

1. Add focused same-command benchmark smoke tests or scripted gates for the
   known regression documents.
2. Land pure observability/trace additions separately if they are behavior
   neutral.
3. Land evidence validation separately with explicit non-regression criteria.
4. Land abbreviation and context-head query behavior separately, with exact
   case coverage for `GTC`, `BWGS`, `PJS`, headache modifiers, seizure
   laterality, and hair/tumor context.
5. Require same-model focused A/B to be non-regressive before running full CSC
   or GeneReviews gates.
