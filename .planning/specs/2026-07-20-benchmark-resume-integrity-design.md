# Benchmark Identity And Resume Integrity Design

## Context

PR #322 adds benchmark execution and scoring fingerprints, retrieval and
producer provenance, schema-v2 artifact inventories, and fail-closed checkpoint
resume. An adversarial review confirmed the overall direction but found gaps
where the published identity can diverge from runtime behavior, a compatible
overwrite can temporarily unpublish an existing run, malformed or symlinked
storage can escape expected boundaries, and secrets can reach persisted output.

This revision closes every review item, including the pre-existing inconsistent
normalization of the gold assertion value `absent`.

## Goals

- Bind benchmark identities to the effective inputs, scoring rules, producer
  source, and verified retrieval content actually used at runtime.
- Keep resume fail-closed while allowing identical installed source to resume
  across a git checkout and a packaged installation.
- Ensure the canonical manifest always references one complete immutable run
  snapshot.
- Make checkpoint rejection strictly zero-mutation and prevent failed overwrite
  attempts from changing the active manifest or its published generation.
- Prevent endpoint credentials and raw provider failures from reaching persisted
  benchmark artifacts.
- Apply one assertion normalization and projection contract to every dataset
  format and to both gold and predicted assertions.

## Non-Goals

- Do not add a destructive restart option.
- Do not silently migrate or resume checkpoints created under the older identity
  schemas.
- Do not claim that arbitrary unknown secrets embedded in URL paths can be
  detected heuristically.
- Do not change frontend behavior or benchmark metric formulas beyond making the
  existing ontology configuration identity-bearing.

## Versioned Identity Contracts

The dataset, assertion-projection, run-fingerprint, and producer-source
descriptors receive explicit new schema versions. A checkpoint using an older or
missing schema is rejected with a migration message before filesystem mutation.
Existing result manifests remain readable.

Execution and scoring fingerprints are conservative scientific identities:

- the execution fingerprint includes selected inputs and order, prompt bundle,
  resolved provider request, verified retrieval content, and producer source;
- the scoring fingerprint includes canonical gold records, the complete
  assertion/projection algorithm, ontology scoring configuration, evaluation HPO
  version when ontology-aware scoring is active, and producer source.

Producer commit, dirty state, and provenance status remain descriptive metadata.
Checkpoint matching compares the producer source digest rather than the entire
metadata object.

## Provider And Endpoint Identity

The resolver owns base-URL canonicalization. It trims whitespace, treats blank as
absent, validates the URL, rejects fragments, and removes behaviorally irrelevant
trailing path slashes. Provider constructors consume the canonical value without
performing a second normalization. Providers that do not support a base URL
reject it instead of recording an unused value.

One endpoint-identity function produces both the safe display and behavioral
digest. It removes userinfo, redacts the resolved API key and percent-encoded
forms, and redacts values of explicitly sensitive query keys. Legitimate path and
non-secret query routing remains identity-bearing. Unknown path credentials are
unsupported and documented as such.

Persisted failures contain a stable public error code and safe message. Raw
provider exception strings remain available only through exception chaining and
runtime logging; they are never copied into checkpoints, summaries, predictions,
manifests, or legacy JSON.

## Producer Source Identity

Producer identity hashes a defined inventory of runtime package files. Relative
paths use POSIX separators, text newlines are normalized, and caches, bytecode,
build metadata, tests, and documentation are excluded. Runtime-relevant untracked
package files are included. The same source checkout and built installation must
produce the same digest.

Git provenance accepts complete 40-character SHA-1 and 64-character SHA-256
object IDs. Unknown formats continue to fail closed. Git status does not decide
compatibility; an edited source file changes the source digest directly.

## Assertion And Scoring Contract

All gold and predicted assertions pass through one canonical normalizer:

- missing or blank -> `PRESENT`;
- `present`, `affirmed`, and `normal` -> `PRESENT`;
- `absent` and `negated` -> `ABSENT`;
- `uncertain` -> `UNCERTAIN`;
- `family_history` -> `FAMILY_HISTORY`.

Input normalization uses `strip().casefold()`. Unknown explicit gold assertions
raise instead of silently becoming `PRESENT`. JSON document dictionaries,
tuple/list annotations, directory datasets, and ID-only JSON lists follow the
same rules.

Projection occurs once after normalization and uses the effective source dataset
for both gold and predictions. The aggregate `all` descriptor contains the
per-source projection table for CSC, GSC, GeneReviews, GSC_plus, and ID_68. The
unreachable `positive_hpo_present_v1` fallback is removed; callers must use the
shared effective projection descriptor.

The scoring descriptor records ontology enabled state, semantic floor,
similarity formula, resolved evaluation HPO version, mapping precedence,
normalization mode, fallback behavior, and default ID-only assertion. Inactive
ontology tuning values are retained conservatively so checkpoint configuration
changes never collapse to one identity.

## Retrieval Content And Runtime Binding

The installed bundle is verified before identity construction. Verification is
fail-closed for missing database or index inventory, empty or incomplete checksum
maps, missing files or directories, absolute or parent-traversing keys, escaping
symlinks, and checksum mismatches. Directory hashing uses sorted POSIX-relative
paths and contents so Windows and POSIX produce the same digest.

The retrieval identity uses a canonical verified-content digest rather than only
the raw manifest bytes. Runtime receives the same manifest model name, revision,
code revision, trust-remote-code setting, single/multi-vector mode, and explicit
index directory. Floating `DEFAULT_MODEL`, default data paths, and the default
`multi_vector=True` setting cannot override the published identity.

## Immutable Artifact Publication

Layout resolution for the LLM resume path is read-only. Checkpoint type, schema,
fingerprints, and complete configuration are validated before `mkdir`, temporary
files, locks, deletion, or legacy-directory creation.

Each write attempt creates an immutable generation below the run directory. The
generation contains checkpoint, summary, metrics, predictions, traces, cases,
terms, diagnostics, and their complete hashed inventory. The existing root
manifest remains authoritative while the generation is built.

Publication writes the new manifest to a unique same-directory temporary file
and atomically replaces only `manifest.json` as the commit point. Therefore the
canonical manifest always references either the previous complete generation or
the new complete generation. Failed provider construction, warmup, inference,
artifact generation, hashing, or manifest replacement leaves the previous
manifest and all referenced files valid. An interrupted attempt may leave an
unreferenced generation that a later boundary-validated cleanup can remove.

Root-level artifact names remain best-effort compatibility aliases refreshed
after manifest commit. Resume resolves the active checkpoint from the manifest,
with the old fixed `checkpoint.json` path as a legacy fallback. Old generations
are retained until safe boundary-validated garbage collection can prove they are
not referenced by the active manifest.

## Filesystem Boundaries And Discovery

Before any creation or mutation, the resolved run path must remain below the
resolved results root. A final run directory or parent component that is a
symlink, Windows junction, or other reparse point is rejected. The boundary is
revalidated immediately before mutation.

Artifact discovery validates that the parsed manifest root and `artifacts` value
are mappings. A malformed manifest is skipped independently and cannot abort
discovery of other valid runs.

## Compatibility

- Old identity checkpoints are intentionally not resumable and remain unchanged.
- Existing schema-v1 and fixed-root schema-v2 result layouts remain discoverable.
- Manifest role aliases remain lookup entries; consumers deduplicate by path.
- Direct-path consumers continue to receive root compatibility aliases, while
  the manifest and immutable generation are authoritative.
- The documented migration is to choose a new run ID or deliberately remove the
  incompatible run.

## Verification

Each defect is reproduced by a failing test before implementation. Focused tests
cover URL canonicalization and secret canaries, scoring and projection identity,
gold normalization across formats, aggregate per-source projection, retrieval
inventory validation and runtime configuration, producer hashing across dirty
and installed sources, malformed manifests, symlink/junction boundaries,
zero-mutation rejection, and failure injection before manifest publication.

Required completion checks are:

- `make check`
- `make typecheck-fast`
- `make test`
- `make ci-python-quality`
- `make ci-python-compat PYTHON=3.12`
- `make ci-python-compat PYTHON=3.13`

Frontend parity checks are not required because this remediation changes only
Python benchmark, retrieval, storage, and documentation code.
