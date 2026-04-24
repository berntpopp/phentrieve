# Grounded Whole-Note LLM CLI Design

Date: 2026-04-16
Branch: `feat/llm-full-text-lean-v1`
Scope: CLI and benchmark maturity for the LLM full-text path
Out of scope: API hardening, frontend integration

## Goal

Mature the LLM full-text workflow in the CLI without regressing the main
strength of the current branch: the phase-1 LLM still sees the whole note and
can reason over document-level context for negation, family history, temporal
scope, and cross-sentence phenotype interpretation.

The design must remove the current fragile provenance model, where the pipeline
extracts from the whole note and later reconstructs local context with naive
sentence splitting.

## Problem Statement

The current LLM path in `feat/llm-full-text-lean-v1` has three structural
problems:

1. The full note is sent directly to phase 1, but no shared chunk/span
   structure is created up front.
2. Phase 2 relies on `_find_original_sentence()` and `text.split(".")` to
   recover local context after extraction.
3. Provenance is too weak for debugging, benchmark diagnosis, and user trust in
   CLI output.

The standard backend already has a richer text-processing stack with
multilingual chunking, conjunction handling, and optional position tracking. The
LLM backend should use that stack as a grounding layer, but not as a replacement
for whole-note reasoning.

## Design Principles

- Keep whole-note LLM extraction as the core phase-1 strategy.
- Use the existing shared chunking pipeline to ground extraction outputs in real
  chunk/span provenance.
- Avoid isolated chunk-only extraction as the default architecture.
- Keep phase-1 extraction source-faithful. Normalization belongs in later
  mapping steps.
- Prefer structured output over free-text parsing whenever Gemini supports it.
- Make provenance first-class in both structured results and logs.
- Optimize for CLI and benchmark maturity first. API and frontend concerns stay
  out of this PR except where shared code naturally benefits.

## Proposed Architecture

### Overview

The new LLM CLI path becomes a grounded whole-note pipeline:

1. Preprocess the note with the shared text-processing pipeline to build an
   ordered chunk/span view of the note.
2. Send the whole note to phase 1 together with compact chunk anchor metadata.
3. Require phase 1 to return phenotype extractions with source-faithful text and
   anchors into the shared chunk structure.
4. Run retrieval per extracted phrase.
5. Run phase-2 mapping against anchored local context from the shared chunk
   structure, optionally including neighboring chunk context.
6. Preserve rich evidence and provenance through aggregation, logging, CLI
   output, and benchmark traces.

### Why This Architecture

This design keeps the LLM's document-level reasoning intact while removing the
current heuristic sentence reconstruction. It also aligns the LLM path with the
repo's existing multilingual chunking infrastructure without forcing a risky
chunk-only extraction design.

## Components

### 1. Grounding Preprocessor

The CLI LLM path will run a shared preprocessing step before phase 1.

Inputs:

- raw clinical text
- language
- chunking configuration, defaulting to the shared repo defaults

Outputs:

- ordered grounded chunks
- `chunk_id`
- `text`
- `start_char` and `end_char` when resolvable
- optional source tracking metadata already available from the text pipeline

This preprocessing output is a grounding layer, not the final extraction unit.

### 2. Whole-Note Phase 1 Extraction

Phase 1 keeps the whole note as the primary text input.

The LLM prompt receives:

- the full note
- language
- compact chunk anchor metadata derived from the grounding preprocessor

The chunk anchor metadata should be minimal enough to avoid unnecessary token
growth. The prompt should not duplicate the entire note multiple times. If full
chunk text is included for anchor resolution, it should appear only once.

### 3. Anchored Phase-1 Output Contract

Phase 1 structured output should evolve from:

- `phrase`
- `category`

to a grounded schema such as:

- `phrase`
- `category`
- `chunk_ids`
- optional `evidence_text`
- optional `start_char`
- optional `end_char`

Minimum acceptance requirement:

- every actionable extracted phenotype must reference at least one real
  `chunk_id`

Preferred requirement:

- each extraction includes `evidence_text` matching source wording

This keeps extraction source-faithful and auditable.

### 4. Retrieval

Retrieval still runs per extracted phrase. No change in overall stage
responsibility.

However, the retrieved candidate record must now carry forward:

- phrase
- category/assertion source
- anchor metadata from phase 1
- local chunk text
- optional neighboring chunk text

### 5. Structured Phase-2 Mapping

Phase 2 mapping should move from free-text parsing to structured output.

Mapping inputs should include:

- original extracted phrase
- category
- anchored local chunk context
- optional neighboring chunk context
- candidate HPO IDs and labels
- optional definitions and a small synonym list if already available cheaply

The mapping stage should not use reconstructed sentence strings from ad hoc text
splitting.

### 6. Deduplication and Aggregation

Deduplication should no longer collapse only on `term_id`.

The internal representation should preserve at least:

- `term_id`
- assertion/category
- evidence list
- provenance list

API-style compact views may still be derived later, but the CLI and benchmark
path should retain richer internal evidence so that clinically meaningful
variants are not discarded during extraction.

## Logging And Provenance

The logging system is part of the design.

Provenance must exist in two aligned forms.

### Structured Provenance In Results

Each resolved phenotype should preserve:

- extracted phrase
- assertion/category
- `chunk_id`
- optional neighboring chunk IDs
- `start_char` and `end_char` when resolvable
- `evidence_text`
- retrieval candidates considered
- selected HPO ID
- match method (`local`, `llm_mapping`, `fallback`)

### Operational Logging

Logs should make the pipeline diagnosable end-to-end.

Minimum log coverage:

- chunking summary: language, strategy, chunk count
- token preflight summary
- phase-1 extraction count and anchor-resolution statistics
- retrieval candidate-set counts and sizes
- phase-2 local-match and LLM-match outcomes
- explicit phase failures
- deduplication decisions when variants are merged or retained

Benchmark traces and CLI debug mode should emit richer provenance than normal
interactive CLI output.

## Error Handling

### Phase 1 Failures

Phase-1 extraction failures must no longer silently turn into empty predictions.

Behavior:

- benchmark mode: record explicit document failure and phase-level error data
- CLI mode: raise or surface a typed pipeline failure with clear log output
- optional retries may exist, but retries must not erase the fact that failure
  occurred

### Anchor Resolution Failures

If an extraction cannot be resolved cleanly to a chunk/span:

- prefer keeping the extraction with degraded provenance rather than silently
  dropping it
- mark provenance status explicitly in logs and trace metadata
- fall back to chunk-level rather than guessed sentence-level context

### Oversize Notes

Use token preflight before phase 1.

Oversize behavior should be explicit and deterministic:

- warn in CLI output and logs
- optionally downgrade to a bounded strategy if implemented
- never silently truncate in a way that hides lost context

## CLI Behavior

The CLI should remain usable for both ordinary interactive use and benchmarking.

### Normal CLI Output

Default output should stay compact and readable.

It should include enough context to answer:

- what HPO term was selected
- what evidence phrase supported it
- what assertion/category applied

### Debug CLI Output

Debug or trace output should expose:

- chunk IDs and chunk text previews
- evidence anchors
- whether a term was locally resolved or mapped by the LLM
- candidate lists when requested
- explicit failure and fallback states

### Benchmark Artifacts

Benchmark artifacts should capture:

- phase-level counts and timings
- token usage
- request counts
- anchor-resolution metrics
- per-term provenance
- failure counters

## Configuration

This PR should introduce an internal LLM preprocessing/grounding mode rather
than replacing behavior blindly.

Recommended internal modes:

- `whole_document_legacy`
- `whole_document_grounded`

CLI default should move to `whole_document_grounded` only after parity or
improvement is verified on benchmark runs in this branch.

This switch does not need to become a user-facing public API in this PR, but it
should exist internally for safe comparison and rollback.

## Testing Strategy

Required tests for this design:

1. shared grounding preprocessor works for the LLM CLI path
2. multilingual grounding at minimum for English and German
3. phase-1 outputs contain anchors that resolve to real chunks/spans
4. phase-2 mapping consumes anchored context, not reconstructed sentence text
5. phase-1 failures are surfaced explicitly
6. duplicate `term_id` values with different assertions/evidence are preserved
7. long-note token preflight behavior is deterministic
8. benchmark traces contain the expected provenance fields

## Risks And Mitigations

### Risk 1: Losing document-level context

Mitigation:

- whole-note phase 1 remains the default
- chunking is grounding, not the main extraction input

### Risk 2: Token bloat from adding chunk metadata

Mitigation:

- use compact anchor metadata
- avoid repeating note text and chunk text unnecessarily
- use token preflight and logging

### Risk 3: Regression from changing too many moving parts

Mitigation:

- stage the rollout behind an internal mode switch
- compare grounded and legacy behavior during benchmark validation

### Risk 4: Provenance anchors are imperfect on messy notes

Mitigation:

- allow degraded-but-explicit provenance
- never replace failed grounding with naive `text.split(".")`

## Explicit Non-Goals For This PR

- public API model allowlisting and quota redesign
- frontend exposure of provenance
- final API response redesign beyond shared internal improvements
- batch API integration for large offline evaluation
- replacing whole-note phase 1 with chunk-only extraction

## Recommended Implementation Order

1. Add grounding preprocessor for the CLI LLM path using the shared chunking
   pipeline.
2. Add anchored phase-1 schema and prompt changes.
3. Remove `_find_original_sentence()` and sentence reconstruction logic.
4. Move phase-2 mapping to structured output with anchored context.
5. Preserve richer evidence/assertion variants in internal aggregation.
6. Add provenance-aware logging and benchmark trace fields.
7. Add token preflight and explicit oversize handling.
8. Benchmark grounded vs legacy behavior and switch CLI default if validated.

## Success Criteria

This PR is successful if:

- the CLI LLM path still uses whole-note reasoning
- provenance is real and traceable end-to-end
- naive sentence reconstruction is gone
- benchmark and debug output clearly explain how each term was produced
- the branch is materially more mature for CLI and benchmark use even before API
  and frontend integration work begins
