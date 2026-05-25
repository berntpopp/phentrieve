# LLM Evidence Validation And Enriched Mapping Design

> Status: Superseded on 2026-05-25. PR #261 was closed because a same-model,
> same-document focused A/B against current `main` showed a strict-ID mapping
> regression. See
> `.planning/analysis/2026-05-25-llm-evidence-validation-enriched-mapping-pr-regression.md`.

## Goal

Implement roadmap items 5-7 from
`.planning/analysis/2026-05-23-phentrieve-rag-prompting-literature-report.md`:
validate phase-1 LLM evidence before phase-2 mapping, enrich phase-2 mapping
payloads with compact HPO detail fields, and keep phase 1 source-faithful by
moving rewrite behavior into phase-2 retrieval-query preparation.

## Source Analysis

The current two-phase LLM pipeline already has the right high-level shape:

1. Phase 1 extracts grounded phenotype phrases with chunk IDs and evidence.
2. Phase 2A retrieves HPO candidates.
3. Phase 2B resolves local or structured LLM mappings.

The report and update identify three gaps in that flow:

- Phase-1 records can reach phase 2 even when chunk IDs are invalid, evidence is
  not grounded in referenced chunk text, or offsets cannot support highlighting.
- Mapping prompt payloads include candidate IDs, labels, scores, retrieval
  query, matched text, and matched component, but not definitions or matched
  synonyms even though `phentrieve/retrieval/details_enrichment.py` can already
  provide definitions and synonyms.
- The phase-1 prompt asks for source-faithful extraction, but one few-shot
  example rewrites source text. The current post-phase-1 expansion helper also
  rewrites `XLID` into `X-linked intellectual disability`, which belongs in
  retrieval-query preparation rather than the extracted phrase.

The update also warns not to add P1+ work directly into the 1700-line
`phentrieve/llm/pipeline.py`. In the live tree, `pipeline.py` is still 1761
lines and still owns phase-2A candidate retrieval orchestration, so this spec
includes a small mechanical phase-2A extraction prerequisite.

## Scope

In scope:

- Add a narrow `phentrieve/llm/phase2a.py` extraction for phase-2A retrieval
  orchestration before adding new validation and enrichment behavior.
- Add a pure phase-1 evidence validator and wire it after phase-1 expansion and
  deduplication, before actionable filtering and candidate retrieval.
- Persist validation results in `LLMMeta.trace` and benchmark prediction traces.
- Add phase-count observability for evidence kept, dropped, repaired, and
  downgraded.
- Enrich phase-2 mapping candidate payloads with compact definitions and
  matched synonyms when available.
- Keep matched component and matched text in mapping payloads.
- Include compact ontology context only if it is already available through an
  existing helper. The current implementation has no parent-label helper, so no
  new ontology-context subsystem is added.
- Move abbreviation and conservative phrase canonicalization into
  `prepare_retrieval_queries(...)`.
- Update phase-1 and mapping prompts/examples so they no longer teach
  source-unfaithful extraction or broad mappings when a specific candidate is
  present.
- Add focused tests and A/B benchmark gates.

Out of scope:

- Reranking, cross-encoder reranking, LLM-as-judge reranking, and generic
  reranking.
- Hybrid lexical/dense retrieval.
- New retrieval subsystems or new candidate-generation channels.
- Evidence-aware aggregate confidence changes.
- Frontend UI changes.
- Public response shape changes, except compatible metadata and trace additions.
- New HPO parent/ancestor label enrichment unless an existing helper already
  provides it during implementation.

## Design Summary

Use a scoped split-plus-validate approach:

1. Extract phase-2A retrieval orchestration from `pipeline.py` into
   `phentrieve/llm/phase2a.py` with no behavior change.
2. Add `phentrieve/llm/evidence_validation.py` as a pure validator over
   extracted dicts and grounded chunk dicts.
3. Wire validation into `TwoPhaseLLMPipeline.run(...)` immediately after
   `_expand_combined_phase1_extractions(...)` and
   `_deduplicate_phase1_extractions(...)`, before actionable filtering.
4. Extend `pipeline_phase2.prepare_retrieval_queries(...)` to emit:
   - the original source phrase;
   - a conservative canonical noun-phrase variant for supported patterns;
   - an abbreviation-expanded variant using the existing
     `PHENOTYPE_ABBREVIATIONS` map;
   - the existing unit-stripped variant.
5. Extend `pipeline_phase2.compact_mapping_item(...)` to enrich candidates with
   safe definitions and matched synonyms while preserving existing compact
   fields.
6. Update prompt YAML examples and tests to lock in the intended behavior.

## Components

### Phase-2A Extraction Prerequisite

Create `phentrieve/llm/phase2a.py`.

The new module owns phase-2A retrieval orchestration:

- deduplicate actionable phase-1 records by downstream phrase/category key;
- prepare retrieval-query variants for each unique actionable record;
- call the existing `tool_executor.query_batch_hpo_terms(...)`;
- normalize both direct `candidates` results and Chroma-style
  `metadatas`/`similarities` results;
- merge candidates by HPO ID, keeping the highest score and its retrieval
  query;
- attach grounded context, chunk IDs, evidence text, and offsets to each item.

`pipeline.py` keeps a thin `_retrieve_candidates(...)` compatibility wrapper
that delegates to `phase2a.retrieve_candidates(...)`. This keeps existing tests
that exercise private pipeline methods working while moving new phase-2A
behavior out of the oversized file.

### Evidence Validator

Create `phentrieve/llm/evidence_validation.py`.

Public API:

```python
@dataclass
class EvidenceValidationReport:
    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    repairs: list[dict[str, Any]]
    status: str = "validated"


def validate_phase1_evidence(
    *,
    extracted: list[dict[str, Any]],
    grounded_chunks: list[dict[str, Any]],
    fuzzy_threshold: float = 90.0,
) -> EvidenceValidationReport:
    ...
```

Validation rules:

- If `grounded_chunks` is empty, return all extracted records unchanged with
  `status="skipped_no_grounded_chunks"` to preserve legacy ungrounded behavior.
- Convert chunk IDs to integers and drop records with no chunk IDs using reason
  `empty_chunk_ids`.
- Drop records that reference chunk IDs absent from `grounded_chunks` using
  reason `unknown_chunk_id`.
- Use the concatenated referenced chunk text as a chunk-level grounding target
  for multi-chunk evidence.
- Require `evidence_text` or `phrase` to be non-empty.
- If `evidence_text` has a word-boundary exact match in referenced chunk text
  after case normalization, keep the record.
- If `evidence_text` is missing but `phrase` is grounded, repair
  `evidence_text` to the phrase and record an `evidence_text_repair`.
- If `evidence_text` is not exact but a high-confidence fuzzy match clears the
  threshold, keep the record at chunk-level precision, clear offsets, and record
  a `fuzzy_evidence_downgrade`.
- If evidence cannot be grounded, drop the record with reason
  `evidence_not_grounded`.
- If a single-chunk record has offsets that match the evidence span in a
  supported coordinate mode, keep them.
- If a single-chunk record has missing or mismatched offsets but the evidence
  has an exact located span, repair them.
- If a multi-chunk record is grounded, clear offsets and record
  `multi_chunk_offset_downgrade`; joined chunk text proves chunk-level grounding
  but does not define an offset coordinate frame used by downstream consumers.
- If offsets are out of bounds or ambiguous, set `start_char` and `end_char` to
  `None`, keep chunk-level evidence, and record `offset_downgrade`.

Offset handling:

- The validator accepts chunk-local offsets when they slice the referenced
  single chunk text to the evidence string.
- The validator also accepts document-absolute offsets when every referenced
  single chunk has `start_char`/`end_char` and the absolute slice maps inside
  that chunk.
- Repaired offsets use chunk-local coordinates relative to the referenced
  single chunk text. The trace records the repair kind so downstream consumers
  can distinguish repaired span precision from original LLM offsets.
- Multi-chunk spans are kept at chunk-level precision with offsets cleared.
  The validator may use joined text to decide that evidence is grounded across
  adjacent chunks, but it must not emit joined-string offsets.

The validator does not mutate input records.

### Validation Trace

`TwoPhaseLLMPipeline.run(...)` adds validation after phase-1 deduplication and
before actionable filtering.

Trace shape:

```json
{
  "phase1_evidence_validation": {
    "status": "validated",
    "kept_count": 3,
    "dropped_count": 1,
    "repair_count": 2,
    "downgraded_count": 1,
    "dropped": [
      {"phrase": "invented finding", "reason": "evidence_not_grounded"}
    ],
    "repairs": [
      {"phrase": "recurrent seizures", "kind": "offset_repair"}
    ]
  }
}
```

The trace stores compact summaries, not full note text. Benchmark prediction
records already copy `pipeline_result.meta.trace`, so this field becomes part
of benchmark/debug traces without adding a new benchmark artifact type.

Phase counts added:

- `phase1_evidence_kept`
- `phase1_evidence_dropped`
- `phase1_evidence_repaired`
- `phase1_evidence_downgraded`
- `phase1_validated_phrases`

`_build_observability_counts(...)` already includes all `phase_counts`, so these
counts flow into benchmark prediction metadata.

For benchmark comparability, the existing `extracted_phrases` count remains the
post-expansion, post-dedup, pre-validation count. The new
`phase1_validated_phrases` count records how many extracted records remain after
validation.

### Source-Faithful Phase 1

Update `phentrieve/llm/prompts/templates/two_phase/en.yaml`:

- bump version from `v3.0.0` to `v3.1.0`;
- add extraction rules requiring every phrase to be a verbatim substring of
  `evidence_text` and every `evidence_text` to be a verbatim substring of a
  referenced chunk;
- state that phase 2 computes retrieval variants and phase 1 must not rewrite;
- replace the lactate dehydrogenase and urine-output example so both `phrase`
  and `evidence_text` are source substrings.

The prompt must no longer tell phase 1 to emit a normalized clinical phrase
for indirectly described abnormalities. Phase 1 can still extract the source
evidence that signals an abnormality; phase 2 is responsible for query variants.

Update `pipeline_phase1.expand_combined_phase1_extractions(...)` so it no
longer rewrites abbreviations into expanded phrases. Abbreviation expansion
moves to `pipeline_phase2.prepare_retrieval_queries(...)`. Keep only clear
source-substring split behavior. Source-unfaithful shared-head rewrites must
not run in phase 1.

### Phase-2 Retrieval-Query Preparation

Extend `prepare_retrieval_queries(...)` in `phentrieve/llm/pipeline_phase2.py`.

Variant order:

1. Original normalized phrase.
2. Existing unit-stripped phrase when different.
3. Conservative canonical noun-phrase variant when a supported pattern applies.
4. Abbreviation-expanded variant when the original phrase matches
   `PHENOTYPE_ABBREVIATIONS`.

The helper must deduplicate variants while preserving order.

Supported canonicalization is intentionally narrow:

- `"lactate dehydrogenase was markedly elevated"` ->
  `"elevated lactate dehydrogenase"`.
- `"urine output remained low"` -> `"low urine output"`.
- Similar patterns are allowed only when a single abnormal adjective or
  participle moves in front of a noun phrase without dropping anatomy,
  laterality, severity, or morphology tokens.

The helper must not add broad hand-written paraphrases such as
`"tongue biting"` -> `"self biting"`.

### Enriched Mapping Payloads

Extend `compact_mapping_item(...)` in `phentrieve/llm/pipeline_phase2.py`.

Candidate payload fields remain compact and stable:

- `id`
- `term`
- `retrieval_score`
- `retrieval_query`, when present
- `matched_text`, when present
- `matched_component`, when present
- `definition`, when available
- `matched_synonym`, when available

Definition handling:

- Use `enrich_results_with_details(...)` from
  `phentrieve/retrieval/details_enrichment.py`.
- Truncate definitions at a word boundary with a default limit of 240
  characters.
- Use ASCII `...` for truncation.
- If the HPO database is missing, rely on the existing helper's graceful
  `None` details.
- If enrichment raises an unexpected database error during mapping payload
  construction, log and continue without enrichment so the old mapping behavior
  remains available.

Matched synonym handling:

- If `matched_component == "synonym"` and `matched_text` is present, emit
  `matched_synonym` with that matched text.
- Otherwise, if enriched synonyms contain a case-insensitive exact match for
  `matched_text`, emit that synonym.
- Do not emit an arbitrary synonym merely because a synonym list exists.

Ontology context:

- No new parent-label or ancestor-path lookup is added in this milestone.
- Context review found no existing helper that returns compact parent labels
  for this payload path, so `parent_labels` and ancestor paths remain out of
  scope for this milestone.

### Mapping Prompt Updates

Update:

- `phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml`
- `phentrieve/llm/prompts/templates/two_phase/en_mapping_batch.yaml`

Changes:

- bump both mapping prompt versions from `v4.1.0` to `v4.2.0`;
- document optional `definition`, `matched_synonym`, `matched_component`, and
  `matched_text` fields;
- state that retrieval score is a hint and definitions/synonyms plus grounded
  context decide semantic equivalence;
- prefer the most specific evidence-supported candidate;
- return `null` when no candidate is supported;
- update examples so `frequent falls` maps to `HP:0002359` when that candidate
  is present instead of the broader `Difficulty walking`.

### Benchmark Metric

Add a benchmark-facing observability metric:

- `phase2_mapping_prompt_tokens_per_request`

The value is computed when phase-2B LLM mapping runs:

```text
round(mapping_prompt_tokens / mapping_request_count)
```

If no phase-2B LLM request runs, the value is `0`. This supports the report's
token-growth gate for enriched payloads.

## Data Flow

The new grounded two-phase flow is:

1. Phase 1 emits source-faithful phrase/evidence records.
2. Existing phase-1 split/dedup logic keeps only source-substring splits.
3. Evidence validation drops ungrounded records, repairs exact offsets, and
   downgrades weak spans to chunk-level precision.
4. Actionable filtering runs on validated records.
5. Phase 2A prepares retrieval-query variants from validated source phrases and
   retrieves candidates.
6. Phase 2B local matching runs unchanged.
7. Phase 2B LLM mapping receives enriched compact candidate payloads for
   unresolved items.
8. Public `LLMExtractionResult.terms` stays compatible.
9. `LLMMeta.trace` and benchmark prediction traces include validation and token
   observability.

## Compatibility

Public response compatibility:

- `LLMExtractionResult.terms` and `LLMPhenotype` fields remain unchanged.
- Invalid phase-1 records may disappear before mapping; this is an intentional
  correctness change.
- Offsets may be set to `None` when they cannot be validated. This is a
  precision downgrade, not a response-shape change.
- Trace and benchmark metadata gain compatible additive fields.

Legacy ungrounded compatibility:

- When `grounded_chunks=[]`, validation is skipped and records are preserved.
  This avoids breaking current callers that still use the two-phase pipeline
  without grounded chunks.

Prompt compatibility:

- Prompt versions bump so benchmark artifacts can identify the behavior change.

## Error Handling

- Missing chunk IDs or unknown chunk IDs drop the record.
- Empty phrase/evidence drops the record.
- Ungrounded evidence below the fuzzy threshold drops the record.
- Fuzzy-grounded evidence clears offsets and records a downgrade.
- Offset repair failures clear offsets and keep chunk-level evidence only when
  the evidence text itself is grounded.
- Enrichment database missing: continue with no definitions or synonyms.
- Enrichment database error: log and continue with old compact payload fields.
- Phase-2A extraction prerequisite must be behavior-preserving; failures there
  are treated as regressions.

## Testing Strategy

Focused unit tests:

- `tests/unit/llm/test_phase2a.py`
  - phase-2A extraction preserves current retrieval-query expansion,
    candidate merging, and grounded context.
- `tests/unit/llm/test_evidence_validation.py`
  - drops missing chunk IDs;
  - drops evidence that is not grounded;
  - repairs missing evidence from grounded phrase;
  - repairs exact offsets;
  - downgrades invalid offsets to `None`;
  - keeps fuzzy evidence only at chunk precision;
  - skips validation when no grounded chunks exist.
- `tests/unit/llm/test_pipeline.py`
  - invalid phase-1 records are dropped before retrieval;
  - validation trace and phase counts are persisted;
  - benchmark-facing phase counts include mapping prompt tokens per request.
- `tests/unit/llm/test_two_phase.py`
  - phase-1 expansion no longer expands abbreviations;
  - retrieval queries include abbreviation expansion;
  - retrieval queries include conservative canonical variants;
  - retrieval queries do not invent broad paraphrases.
- `tests/unit/llm/test_prompts.py`
  - phase-1 examples are source-substring faithful;
  - mapping prompts expose enriched fields and version bumps;
  - mapping examples prefer `HP:0002359` for frequent falls when present.
- `tests/unit/test_llm_benchmark.py`
  - benchmark prediction trace preserves `phase1_evidence_validation`;
  - observability includes evidence validation counts and
    `phase2_mapping_prompt_tokens_per_request`.

Existing integration tests under `tests/integration/llm/` must continue to
pass. Some tests that asserted source-unfaithful expansion behavior must be
updated to the new source-faithful contract.

## Benchmark Gates

Required local checks before completion:

- `make check`
- `make typecheck-fast`
- `make test`

A/B gates for LLM/full-text mapping:

- Run a fixed small GeneReviews smoke set before and after implementation when
  provider credentials or a local provider are available:

```bash
uv run phentrieve benchmark llm \
  --test-file tests/data/en/phenobert \
  --dataset GeneReviews \
  --doc-id GeneReviews_NBK1277 \
  --doc-id GeneReviews_NBK148668 \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash \
  --llm-seed 1 \
  --ontology-aware-metrics \
  --capture-phase1-debug \
  --output-path .planning/analysis/llm-evidence-validation-gene-reviews-before.json
```

Repeat with `...-after.json` after implementation.

- Run a fixed CSC smoke set when runtime allows:

```bash
uv run phentrieve benchmark llm \
  --test-file tests/data/en/raghpo_paper \
  --dataset CSC \
  --doc-id CSC_1 \
  --doc-id CSC_2 \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash \
  --llm-seed 1 \
  --ontology-aware-metrics \
  --capture-phase1-debug \
  --output-path .planning/analysis/llm-evidence-validation-csc-after.json
```

Benchmark acceptance:

- failed document count does not increase;
- strict and ontology-aware soft/partial F1 do not regress on the smoke set;
- percentage of predictions with chunk IDs does not decrease;
- percentage of valid evidence spans increases or all invalid spans are
  honestly downgraded to chunk-level evidence;
- `phase2_mapping_prompt_tokens_per_request` grows by no more than 25% versus
  the before artifact;
- trace contains `phase1_evidence_validation` for every completed grounded
  document.

Skipped-benchmark criteria:

- no configured provider credentials and no reachable local provider;
- expected runtime is not acceptable for the current execution window;
- required HPO data or ontology graph artifacts are missing;
- provider quota/rate limit blocks the run.

When skipped, the implementation record must name the skipped gate and the
specific reason.

## Acceptance Criteria

- Phase 1 remains source-faithful: extracted phrases are source substrings, and
  phase-1 prompt examples do not teach rewriting.
- Server-side validation runs before phase 2 for grounded inputs.
- Invalid records are dropped with compact trace metadata.
- Offsets are repaired when exact and downgraded when weak or ambiguous.
- Validation trace fields persist into benchmark prediction records.
- Phase 2 retrieval queries include original, conservative canonical,
  abbreviation-expanded, and unit-stripped variants without adding a new
  retrieval subsystem.
- Mapping payloads include compact definitions and matched synonyms when
  available.
- Mapping prompts teach evidence-supported specificity and abstention.
- Public result shape remains compatible.
- No reranking or hybrid lexical/dense retrieval is introduced.

## Self-Review

- Placeholder scan: no placeholders or deferred decisions remain.
- Consistency check: phase-1 extraction stays source-faithful, and all rewrite
  behavior is assigned to phase-2 retrieval-query preparation.
- Scope check: reranking, hybrid retrieval, and new ontology-context systems are
  explicitly excluded.
- Ambiguity check: offset repair and downgrade behavior is defined, including
  chunk-local and document-absolute validation modes.
