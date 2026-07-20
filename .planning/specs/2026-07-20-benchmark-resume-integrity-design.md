# Benchmark Resume Integrity Design

## Context

PR #322 adds separate benchmark execution and scoring fingerprints, exact
schema-v2 artifact inventories, and stricter provenance. A merge-readiness
review found that the live checkpoint loader validates the new fingerprints
instead of, rather than in addition to, the established checkpoint
configuration. It also found that `--overwrite` clears run artifacts before
checkpoint compatibility is known.

## Decision

Keep the current CLI surface and treat `--overwrite` as permission to resume an
explicit run directory. Do not add a destructive `--restart` option. The LLM
command must build identities and validate the preserved checkpoint before it
clears any existing artifacts. An incompatible run remains byte-for-byte
unchanged and the error tells the operator to choose a new run id or remove the
old run deliberately.

Checkpoint reuse requires both layers to match:

1. execution and scoring fingerprints validate the scientific identity split;
2. the complete checkpoint configuration validates output-affecting settings
   such as trace capture, accounting, ontology metrics, dataset selection, and
   prompt overrides.

Dataset identities explicitly carry their schema version and a hash of the
effective assertion projection. The CLI passes the actual dataset projection
mapping used by the benchmark so a projection-code change changes the scoring
fingerprint.

The existing optional `evaluation_hpo_version` helper parameter becomes the
documented `--evaluation-hpo-version` CLI option. If omitted, the installed
retrieval bundle version remains the default. If supplied, it must match the
bundle and therefore acts as an operator assertion rather than invented
provenance.

## Publication And Compatibility

Partial LLM runs publish schema v2 directly. The redundant transient schema-v1
write is removed. Schema-v1 singleton aliases remain in the v2 artifact map for
existing consumers; documentation states that aliases are lookup entries and
inventory consumers must deduplicate by path.

The two illustrative identity JSON files are removed because no test consumes
them and they no longer match the implemented contracts. Executable unit tests
remain the contract examples.

## Documentation

The benchmarking guide will document:

- the installed retrieval-manifest prerequisite;
- evaluation/retrieval HPO matching and the new CLI option;
- source, input, gold, projection, execution, and scoring identities;
- LLM manifest v2 versus retrieval/extraction manifest v1;
- resume behavior, old-checkpoint rejection, and safe remediation;
- v1 artifact aliases and path-based deduplication.

`CHANGELOG.md` will record the new contracts and the intentional checkpoint
compatibility change under `Unreleased`.

## Verification

Regression tests must first fail on the current branch for complete checkpoint
validation and non-destructive mismatch handling. Focused unit suites, `make
check`, `make typecheck-fast`, and `make test` must pass before the PR is
updated.
