# Phenopacket V2 Compatibility And Annotation Sidecar Design

- Date: 2026-04-19
- Status: Draft for review
- Scope: Python Phenopacket export paths for CLI, API-facing serialization helpers, and LLM/full-text provenance export design
- Out of scope: Frontend display changes, non-Phenopacket output formats, external registry publication, and cross-language SDK generation

## 1. Goal

Revise the current Phenopacket export implementation so that the primary exported
Phenopacket artifact is standards-compatible with GA4GH Phenopacket Schema v2,
while adding an optional linked annotation sidecar format that preserves the
full-text provenance and confidence details used by Phentrieve's current and
planned LLM/full-text pipelines.

The design must satisfy both of these constraints at the same time:

- The default Phenopacket output must remain a single strict Phenopacket JSON
  artifact that can be consumed directly by downstream Phenopacket tools.
- Phentrieve must have a standards-respecting way to preserve richer annotation
  details such as spans, evidence text, confidence, and provenance without
  pretending that those details are native fields of the Phenopacket standard.

## 2. Current State

The current `phentrieve/phenopackets/utils.py` implementation is intended as a
Phenopacket v2 exporter, but it has several standards and architecture issues.

### 2.1 What works today

The current exporter correctly models several standard concepts:

- HPO terms are exported as `PhenotypicFeature.type`.
- Negated/absent findings are represented using `PhenotypicFeature.excluded`.
- An ECO evidence code is attached via `PhenotypicFeature.evidence`.
- General provenance such as Phentrieve version, HPO version, and model names is
  carried in `MetaData` and `externalReferences`.

### 2.2 What is problematic today

The current exporter also overloads several fields in ways that are weak or not
ideal for standards-compatible exchange:

- `phenopacket_schema_version` is currently emitted as `2.0.2`, while the
  official documentation describes `2.0` as the supported schema version string.
- Start/end character offsets, assertion labels, and source text are encoded as
  free text inside `ExternalReference.description`.
- Chunk-level retrieval confidence is encoded only inside descriptive strings,
  not as machine-readable data.
- The exporter mixes two distinct concepts:
  - strict Phenopacket exchange
  - Phentrieve-specific annotation provenance
- The current branch in `phentrieve-bench` with richer LLM annotation objects
  confirms the need for structured provenance, but its export path still flattens
  that information into legacy Phenopacket helpers rather than a clean,
  standards-aligned contract.

## 3. Standards Constraints

The official Phenopacket Schema v2 documentation establishes a few hard
boundaries relevant to this design.

### 3.1 Phenotypic feature model

`PhenotypicFeature` is the standard home for the phenotype assertion itself.

It cleanly supports:

- phenotype identity via `type`
- presence/absence via `excluded`
- ontology-backed `severity`
- ontology-backed `modifiers`
- optional onset/resolution
- evidence references

It does not define native fields for:

- arbitrary confidence scores
- exact supporting text snippets
- text span offsets
- retrieval chunk ids
- provider/pipeline-specific provenance

### 3.2 Evidence model

`Evidence` is intentionally minimal:

- `evidence_code`
- optional `reference`

This is suitable for saying how the assertion was supported, but it is not a
full annotation payload container.

### 3.3 ExternalReference model

`ExternalReference` has only:

- `id`
- `reference`
- `description`

This can point to external material or provide a short description, but it is
not a robust schema extension mechanism.

### 3.4 Compatibility requirement

If Phentrieve emits a JSON object that adds custom top-level fields to the
Phenopacket message, strict downstream parsers may reject it unless configured
to ignore unknown fields. Therefore, if direct drop-in Phenopacket compatibility
matters, the primary Phenopacket JSON artifact should not include custom
non-standard top-level fields.

### 3.5 Phenopacket schema version string

The current exporter emits `phenopacket_schema_version="2.0.2"`, but the
official documentation page for `MetaData` describes the schema version field
using `2.0` as the v2 identifier. Because documentation and generated bindings
can diverge in patch-level display, implementation should first verify the exact
accepted value against the Phenopackets Python/protobuf behavior used by this
repository and then standardize on that value consistently.

For planning purposes, the intended correction is:

- stop using an arbitrary exporter-local string without verification
- anchor the emitted value to one explicit source of truth
- add a parsing/round-trip validation test so the chosen value is proven
  acceptable to the actual Phenopackets library in use

## 4. Design Principles

- The default Phenopacket artifact must be strict and standard-compatible.
- Standard fields should only carry semantics the Phenopacket schema actually
  models.
- Full provenance must remain machine-readable, but it should live outside the
  strict Phenopacket payload when it exceeds the standard's field model.
- The richer provenance payload should be generic and annotation-oriented,
  not tightly coupled to one internal pipeline implementation.
- The Phenopacket and sidecar must be linkable through stable identifiers.
- Backward compatibility should be preserved where feasible, but standards
  correctness takes precedence over preserving weak legacy encodings forever.

## 5. User-Facing Behavior

### 5.1 Default behavior

Default Phenopacket export remains a single strict Phenopacket JSON artifact.

For example, when the CLI requests Phenopacket output, the primary output remains
one file or one JSON object that is itself a valid Phenopacket.

This preserves expected behavior and interoperability.

### 5.2 Optional sidecar behavior

Add an explicit opt-in export mode for a linked annotation sidecar.

When enabled, export produces:

- a strict Phenopacket JSON artifact
- a separate annotation sidecar JSON artifact

The sidecar is intended for consumers that need rich provenance such as spans,
certainty, confidence, and retrieval/LLM trace details.

### 5.3 Naming behavior

If the export target is file-based, the recommended naming convention is:

- `<stem>.phenopacket.json`
- `<stem>.annotations.json`

If the export target is stdout, the default Phenopacket-only behavior remains
available, while paired export should require either:

- an output directory, or
- explicit dual-path flags

This avoids ambiguous multi-document stdout behavior.

## 6. Data Model Design

### 6.1 Strict Phenopacket projection

The Phenopacket artifact should contain only standard-compatible content.

Per phenotype:

- `type`: HPO term id and label
- `excluded`: `true` when the finding is absent/negated
- `evidence`: ECO code plus reference to provenance source
- `modifiers`: only when represented by appropriate ontology-backed modifier
  terms and the semantics are clear

General metadata:

- `created`
- `created_by`
- `resources`
- standard references to HPO
- optional standard external references to linked artifacts

### 6.2 Annotation sidecar model

The sidecar should be a generic annotation bundle, not a raw dump of current
internal objects.

The sidecar must itself have an explicit JSON Schema checked into the repository,
for example under:

- `phentrieve/phenopackets/schemas/phenotype_annotation_bundle_v1.schema.json`

This schema becomes the source of truth for `schema_version`.

Recommended top-level structure:

```json
{
  "schema_version": "1.0.0",
  "artifact_type": "phenotype_annotation_bundle",
  "generated_by": {
    "tool": "phentrieve",
    "version": "0.16.0"
  },
  "phenopacket_id": "...",
  "annotations": [
    {
      "annotation_id": "ann-0001",
      "phenotypic_feature_index": 0,
      "hpo_id": "HP:0001250",
      "label": "Seizure",
      "assertion": "affirmed",
      "certainty": "confirmed",
      "confidence": 0.91,
      "evidence_text": "recurrent seizures",
      "spans": [
        {
          "start_char": 123,
          "end_char": 141,
          "text": "recurrent seizures"
        }
      ],
      "chunk_refs": [4],
      "provenance": {
        "source_mode": "two_phase",
        "match_method": "llm_mapping",
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite-preview"
      }
    }
  ]
}
```

Required sidecar concepts:

- stable annotation id
- stable reference back to the exported phenotypic feature by array position
  using `phenotypic_feature_index`
- HPO id and label
- assertion
- confidence if available
- zero or more spans
- provenance block

Optional sidecar concepts:

- certainty
- evidence text
- chunk references
- source mode
- match method
- provider/model metadata

Normative sidecar rules:

- `confidence`, when present, is a numeric score in `[0.0, 1.0]`
- `confidence` is an application score and is not assumed to be a calibrated
  probability unless the producing pipeline explicitly documents that guarantee
- `assertion` values are exactly: `affirmed`, `negated`, `uncertain`,
  `family_history`, `unknown`
- multiple annotation entries may share the same `phenotypic_feature_index` when
  one exported PhenotypicFeature is supported by multiple spans or provenance
  records

### 6.3 Assertion and certainty semantics

The design should distinguish:

- assertion: `affirmed`, `negated`, `uncertain`, `family_history`, `unknown`
- certainty: optional semantic assessment distinct from numeric score

The strict Phenopacket projection should map only what the standard models well:

- negated/absent -> `excluded=true`
- affirmed/present -> `excluded=false`

Other assertion states should remain preserved in the sidecar unless and until a
clear ontology-backed mapping is adopted.

### 6.4 Feature linkage

Because `PhenotypicFeature` itself does not expose a general id field in the JSON
representation, linkage must avoid overloading fields that are meant to carry
external identifiers.

Recommended linkage strategy:

- the sidecar references exported features by deterministic array position using
  `phenotypic_feature_index`
- the ordering of `phenotypic_features` in the Phenopacket export becomes part of
  the paired-artifact contract for sidecar-enabled exports
- if an external link is desired from the Phenopacket to the sidecar itself, use
  a standard `File` or `ExternalReference` pointing to the sidecar artifact, not
  a local pseudo-CURIE stuffed into `ExternalReference.id`

This keeps the Phenopacket strict while making sidecar-to-feature linkage stable
and machine-readable.

## 7. Linking Strategy

### 7.1 Phenopacket to sidecar reference

The strict Phenopacket should reference the sidecar using a standard field.

Preferred mechanism:

- use a `File` reference attached at the Phenopacket scope when file-based export
  is used

Fallback mechanism:

- use an `ExternalReference` in `MetaData` that points to the sidecar path or URI

The file-based approach is cleaner when a real companion file is emitted.

### 7.2 Sidecar to Phenopacket reference

The sidecar should include:

- the exported `phenopacket_id`
- generated feature references

This makes the link bidirectional and robust for downstream consumers.

## 8. Revising The Current Exporter

### 8.1 Standards corrections

The current exporter should be revised to:

- emit the verified schema version value chosen per Section 3.5
- stop treating `ExternalReference.description` as the authoritative structured
  home for spans and provenance
- preserve `excluded` behavior for absent findings
- keep HPO resource metadata standards-aligned

### 8.2 Transitional descriptive strings

Phase 1 of the rollout should not create information loss. Until the sidecar is
implemented, the exporter may continue to include short descriptive strings in
`ExternalReference.description` as a transitional compatibility measure for span,
assertion, and confidence hints.

However, during this phase:

- those strings are explicitly transitional and non-authoritative
- no existing information currently available in exports should disappear
- the authoritative machine-readable home for richer provenance only shifts once
  sidecar support lands

Phase 2 may then reduce or simplify the descriptive strings after sidecar export
exists and tests prove there is no regression in information availability.

### 8.3 Input contracts

The exporter should stop relying on ad hoc dict variants that disagree on key
names such as:

- `id` vs `hpo_id`
- `name` vs `term_name`
- `confidence` vs `score`

Instead, the export path should normalize internal results into one explicit
intermediate export model before generating:

- the strict Phenopacket projection
- the optional annotation sidecar

## 9. Internal Architecture

### 9.1 Export projection layer

Introduce an explicit export-normalization layer between pipeline outputs and
serialized Phenopacket artifacts.

Conceptually:

- internal retrieval/LLM objects -> normalized export annotations ->
  strict Phenopacket projection + optional sidecar projection

This prevents serialization code from having to understand every internal result
shape directly.

### 9.2 Shared export types

Add typed internal export models for:

- normalized phenotype export record
- normalized span record
- sidecar bundle record

These should live near the Phenopacket export code or in a small dedicated export
module, rather than inside provider-specific LLM code.

One concrete normalized record should be part of the implementation contract to
avoid ambiguity during planning and execution. For example:

```python
@dataclass(frozen=True)
class NormalizedPhenotypeExportRecord:
    hpo_id: str
    label: str
    assertion: Literal[
        "affirmed", "negated", "uncertain", "family_history", "unknown"
    ]
    confidence: float | None
    evidence_text: str | None
    spans: list[NormalizedSpan]
    chunk_refs: list[int]
    source_mode: str | None
    match_method: str | None

@dataclass(frozen=True)
class NormalizedSpan:
    start_char: int
    end_char: int
    text: str | None = None
```

All current export inputs should be normalized into this shape before any
Phenopacket or sidecar serialization occurs.

## 10. Testing Strategy

Add or revise tests for:

- strict Phenopacket export remains valid JSON for the Phenopacket message shape
- exported schema version value matches the verified accepted value chosen in
  implementation
- absent findings map to `excluded=true`
- present findings do not set `excluded=true`
- optional sidecar export writes a second artifact with stable linkage ids
- feature-to-sidecar linking is deterministic and round-trippable
- existing chunk/full-text export paths preserve spans and assertion in the
  sidecar when available
- legacy aggregated export paths still work, with explicit normalization
- malformed ad hoc result dicts fail clearly during normalization
- Phenopacket JSON round-trips through actual Phenopacket protobuf parsing with
  strict unknown-field handling
- sidecar JSON validates against its checked-in JSON Schema

## 11. Rollout Plan

### Phase 1

- Correct current Phenopacket exporter for standards compatibility
- Introduce normalized export model
- Preserve current single-file Phenopacket default behavior
- Add tests for compatibility corrections

### Phase 2

- Add optional annotation sidecar export
- Add stable linkage between Phenopacket and sidecar
- Thread richer LLM/full-text provenance into the normalized export model
- Add tests for paired artifact export

### Phase 3

- Consider promoting sidecar export in benchmark or research workflows where
  provenance fidelity is especially important
- Optionally document the sidecar schema for external consumers

## 12. Risks

- Overloading standard fields further would create misleading outputs that look
  interoperable but are not semantically clean.
- A sidecar without stable linkage ids would be difficult to use reliably.
- Multiple legacy internal result shapes can make export brittle unless
  normalized first.
- Assertion/certainty semantics can drift if they are not clearly separated.

## 13. Sources

- PhenotypicFeature documentation:
  https://phenopacket-schema.readthedocs.io/en/latest/phenotype.html
- Evidence documentation:
  https://phenopacket-schema.readthedocs.io/en/latest/evidence.html
- ExternalReference documentation:
  https://phenopacket-schema.readthedocs.io/en/latest/externalreference.html
- MetaData documentation:
  https://phenopacket-schema.readthedocs.io/en/latest/metadata.html
- File documentation:
  https://phenopacket-schema.readthedocs.io/en/latest/file.html
- Python usage notes:
  https://phenopacket-schema.readthedocs.io/en/latest/python.html
- Export/import guidance:
  https://phenopacket-schema.readthedocs.io/en/master/java-export.html
