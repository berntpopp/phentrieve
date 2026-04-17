# Expert ML Engineering Review: LLM Full-Text Pipeline (PR #216)

Date: 2026-04-16
PR: `berntpopp/phentrieve#216`
Branch: `feat/llm-full-text-lean-v1`
Scope: accuracy, multilingual robustness, latency, and cost

## Executive Summary

The PR introduces a lean LLM full-text extraction path built around a
two-phase pipeline:

1. structured phenotype phrase extraction with Gemini
2. vector retrieval of candidate HPO terms
3. local matching plus a second LLM mapping step for unresolved phrases

That architecture is reasonable for a v1. The main issue is not that the design
is fundamentally wrong, but that the current implementation underuses the
codebase's existing multilingual chunking machinery and over-relies on a
single whole-document LLM pass. The reported GeneReviews benchmark quality
problem is consistent with that implementation choice.

The strongest near-term improvements are:

- reuse the repo's language-aware chunking pipeline before LLM extraction
- stop treating phase-1 failures as empty predictions
- preserve assertion/evidence variants instead of deduplicating by HPO id alone
- tighten cost controls with server-owned model allowlists and token preflight
- move the phase-2 mapping step to structured output

## Assessment Of External Review Feedback

I checked the Claude Code review against the current branch and would separate
it into three buckets:

- Correct and worth keeping:
  - double-query fallback cost in `provider.py`
  - per-call Gemini client construction cost
  - assertion-dropping deduplication
  - module-level prompt-loader mutation in benchmark code
  - hierarchy-aware benchmark concerns
- Correct direction, but incomplete because it ignores existing multilingual
  infrastructure:
  - naive sentence splitting is a real issue, but the right fix is not "add a
    better English sentence tokenizer" in isolation. The right fix is to reuse
    the repo's existing multilingual chunking pipeline.
- I do not agree as stated:
  - "Phase 1 prompt lacks HPO-aware guidance" if that means normalizing or
    expanding phrases during extraction

That last point matters. Phase 1 is an extraction stage, not a normalization
stage. Preserving source wording is the safer default because changing
abbreviations or rewriting phrases can alter clinical meaning and damage
provenance. Abbreviation expansion, terminology normalization, and
HPO-oriented rewriting belong in Phase 2B or another dedicated normalization
step, where the task is candidate selection rather than source-faithful
mention extraction.

## Existing Codebase Strengths Relevant To Multilingual Splitting

The repo already has useful infrastructure that the new LLM path is not
reusing yet:

- `phentrieve/text_processing/chunkers.py`
  - `SentenceChunker` uses `pysbd` and falls back to regex if segmentation fails
  - `ConjunctionChunker` loads language-specific conjunction resources
  - `SlidingWindowSemanticSplitter` supports language-aware semantic splitting
- `phentrieve/text_processing/resource_loader.py`
  - supports bundled and user-overridden language resources
- `phentrieve/text_processing/config_resolver.py`
  - supports multiple chunking strategies already exposed in CLI and API
- `tests/integration/test_chunking_pipeline_integration.py`
  - includes multilingual chunking coverage, including German conjunction cases

This is important because the current LLM implementation introduces its own
naive sentence/context reconstruction inside the LLM pipeline instead of
building on the more mature segmentation path already present in the project.

## Primary Findings

### 1. High: the LLM path bypasses the repo's multilingual chunking stack

Current behavior:

- `run_llm_backend()` in `phentrieve/text_processing/full_text_service.py`
  sends the entire note directly into `TwoPhaseLLMPipeline`
- `TwoPhaseLLMPipeline` then reconstructs context internally rather than
  consuming chunked spans from `TextProcessingPipeline`

Why this matters:

- Whole-document extraction makes prompt length, latency, and token cost scale
  directly with note length.
- It also removes a natural opportunity to isolate phenotypes by local context.
- For multilingual support, it duplicates logic that the repo already handles
  more carefully via `SentenceChunker`, `ConjunctionChunker`, and configurable
  chunking strategies.

Recommendation:

- Run the LLM backend on chunked or sectioned text rather than raw whole-note
  input.
- Reuse the existing chunking pipeline as the default segmentation layer for
  LLM extraction, with language passed through explicitly.
- Aggregate term predictions across chunks after extraction.

Expected impact:

- better recall on long notes
- lower per-request prompt size
- cleaner multilingual behavior

### 2. High: context reconstruction inside the LLM pipeline is not robust across languages

Current behavior:

- `_find_original_sentence()` in `phentrieve/llm/pipeline.py` splits on `"."`
  and selects the sentence with maximal token overlap

Problems:

- breaks on abbreviations, enumerations, decimals, bullets, headings, and
  fragmented note text
- is strongly English-biased
- discards punctuation and formatting cues that matter for negation and
  phenotype scope
- is inconsistent with the repo's existing `SentenceChunker` based on `pysbd`

Why multilingual support changes the recommendation:

- sentence boundary rules are not language-neutral
- conjunction and negation behavior varies by language
- many clinical notes include shorthand, list formatting, or mixed-language
  content where naive splitting is especially fragile

Recommendation:

- remove the internal `"."`-based sentence reconstruction from the LLM path
- pass real chunk/sentence spans produced by the shared chunking pipeline
- preserve original casing and exact surface form in mapping prompts
- if chunk-local context is needed, pass the source chunk plus optional
  neighboring chunk, not a reconstructed sentence string

### 3. High: phase-1 extraction failures are silently converted into empty predictions

Current behavior:

- `_extract_phase1_phenotypes()` in `phentrieve/llm/pipeline.py` catches any
  exception from structured extraction and returns `([], {})`

Problems:

- provider failures
- malformed structured output
- timeout/truncation behavior
- schema mismatch

all collapse into "no phenotypes found".

That makes benchmark diagnosis unreliable and can severely understate recall.

Recommendation:

- in benchmark mode, record explicit error status and fail the document loudly
- in product mode, surface a typed pipeline error or retry path instead of
  returning an empty extraction
- add per-phase error counters to benchmark output

### 4. High: deduplication drops clinically relevant variants

Current behavior:

- `_deduplicate_terms()` in `phentrieve/llm/pipeline.py` keeps only the first
  occurrence per `term_id`

Problems:

- present and negated variants of the same HPO term collapse together
- patient and family-history contexts can be lost
- multiple evidence spans are discarded

Recommendation:

- deduplicate by at least `(term_id, assertion)`
- retain evidence lists and provenance
- if API payload needs a compact form, derive that from a richer internal
  structure rather than discarding data during extraction

### 5. High: the public API exposes cost risk because the LLM model is caller-controlled

Current behavior:

- `TextProcessingRequest` accepts arbitrary `llm_model`
- the LLM provider factory accepts any Gemini model string compatible with its
  guardrails
- quota is request-count based, not token- or tier-based

Problems:

- cost can vary substantially across models for the same quota unit
- long notes can consume much larger prompt budgets without any preflight check
- failures before `record_success()` do not consume quota

Recommendation:

- replace free-form public `llm_model` selection with a server-owned allowlist
- use Gemini token counting before inference for admission control and logging
- meter public usage by token budget or priced tier, not only by successful
  request count

### 6. Medium: phase-2 mapping still uses free-text completion where structured output is a better fit

Current behavior:

- phase 1 uses `run_structured_prompt()`
- phase 2 mapping uses `provider.complete()` and then parses IDs from free text

Problems:

- more brittle parsing
- more fallback logic
- more ambiguity around partial or malformed outputs

Google's structured-output guidance explicitly recommends strong typing,
descriptions, enums where applicable, and application-level validation for
extraction tasks.

Recommendation:

- move phase-2 mapping to structured output
- define single-item and batch schemas explicitly
- use property descriptions to constrain what each field means

### 7. Medium: the candidate payload is too thin for difficult normalization cases

Current behavior:

- phase-2 mapping prompt receives candidate id and term label

Problems:

- similar HPO terms are hard to disambiguate without definitions or synonyms
- abbreviation-heavy phrases are difficult to map precisely
- multilingual phrasing worsens label-only normalization

Recommendation:

- include HPO definition where available
- include a small synonym list where available
- preserve the original extracted phrase
- optionally add a separate normalized or expanded phrase for mapping only
- add language field to mapping payload so prompt wording can adapt when needed

### 7a. Medium: normalization guidance belongs in Phase 2B, not Phase 1

This is the main point where I would revise the external review.

What I agree with:

- abbreviations and informal clinical phrasing are hard to normalize against HPO
- the pipeline likely needs stronger normalization help somewhere

What I do not agree with:

- Phase 1 should not be asked to rewrite extracted mentions into preferred HPO
  wording as part of extraction

Why:

- phase 1 is responsible for faithful mention extraction from source text
- rewriting "DCM", "ASD", or "has trouble learning" during extraction can blur
  provenance and may change meaning
- the current prompt instruction to preserve wording is appropriate for an
  extraction stage

Better recommendation:

- keep Phase 1 source-faithful
- in Phase 2B, pass both:
  - the original extracted phrase
  - an optional expanded/normalized form for mapping assistance
- if abbreviation expansion is added, keep it explicit and auditable rather
  than silently replacing the original mention

### 8. Medium: the current local matching logic is English-leaning and morphologically shallow

Current behavior:

- `_normalize_token()` removes a trailing `s` for tokens longer than 3 chars
- local match heuristics rely on token overlap, cleaned text equality, substring
  match, and token-sort similarity

Problems:

- English plural handling is incomplete
- non-English morphology is not addressed
- false positives and false negatives are both likely for languages with richer
  inflection

Recommendation:

- keep local matching as a cheap fast path, but make it conservative
- avoid pretending it is multilingual normalization
- if stronger local normalization is needed, make it language-aware and opt-in
- do not rely on naive stemming for languages beyond English

### 9. Medium: benchmark execution is online and sequential when the provider supports cheaper asynchronous evaluation modes

Current behavior:

- `phentrieve/benchmark/llm_benchmark.py` iterates documents sequentially
- each document runs synchronous online requests

Recommendation:

- keep synchronous mode for smoke tests and debugging
- add an offline benchmark path using Gemini Batch API for large evaluations
- reserve the online path for interactive or low-latency use cases

This is mainly a benchmark cost/throughput issue rather than a product issue.

## Multilingual Segmentation Recommendations

This is the area that most needs to be adjusted from the earlier review.

The right recommendation is not "add a better English sentence splitter". The
right recommendation is:

1. use the repo's existing language-aware segmentation stack as the default
   pre-LLM stage
2. preserve original text spans and chunk ids through the LLM pipeline
3. make any additional splitting resource-driven and language-specific

### Suggested design for the LLM path

#### Option A: section/chunk-first extraction

- run `TextProcessingPipeline` with a default strategy such as
  `sliding_window_punct_conj_cleaned`
- send each chunk independently to phase-1 extraction
- map and aggregate chunk-level results

Why this is the best default:

- leverages existing multilingual code
- keeps prompt windows small
- provides chunk provenance naturally
- improves robustness on long and mixed-format notes

#### Option B: sentence-group extraction

- run `SentenceChunker` plus conjunction/punctuation splitting
- group resulting segments into bounded token windows
- extract per group

This is lighter-weight and may be enough if full chunking is too aggressive.

#### Option C: whole-document extraction with cached context

- retain current whole-document phase-1 design
- add token counting, context caching, and explicit oversize handling

This is the least attractive option for multilingual and long-note robustness.

### Concrete repo-aligned changes

- Replace `_find_original_sentence()` with chunk/span provenance from the shared
  text-processing pipeline.
- Introduce an internal LLM-preprocessing config that can select:
  - `whole_document`
  - `sentence_groups`
  - `shared_chunk_pipeline`
- Default to `shared_chunk_pipeline` for benchmarks and API.
- Reuse existing language resources for conjunction handling rather than adding
  parallel per-language rules inside the LLM package.
- Add multilingual integration tests for the LLM path, not just the standard
  chunking pipeline.

## Best Practices From Current Docs

### Gemini structured output

Relevant guidance from Google:

- use clear `description` fields in schemas
- use strong typing and enums where possible
- always validate model output in application code

Implication here:

- phase-2 mapping should be structured
- phase-1 schema should explicitly describe phrase preservation and category use

### Gemini token counting

Google exposes `models.countTokens`, which is directly relevant here.

Implication here:

- estimate prompt size before inference
- reject or reroute oversized requests early
- log token budgets in benchmark output before sending requests

### Gemini long-context guidance

Google notes that unnecessary tokens should be avoided, latency grows with
longer inputs, and caching helps when reusing repeated context.

Implication here:

- avoid whole-document prompting where chunking is available
- cache repeated prompt scaffolding where possible

### Gemini Batch API

Google documents Batch API specifically for large-scale, non-urgent workloads,
including evaluations, at reduced cost relative to standard synchronous calls.

Implication here:

- corpus benchmarking should gain a batch mode

### Unicode/ICU sentence boundary handling

ICU boundary analysis and Unicode text-boundary guidance exist precisely
because sentence and word boundaries are not reliably handled by ad hoc regex
across languages.

Implication here:

- do not build new naive sentence splitting inside the LLM package
- prefer standardized segmentation layers or the repo's existing `pysbd` based
  path plus language resources

## Prioritized Recommendations

### Tier 1: highest impact on quality

1. Route LLM extraction through the shared multilingual chunking pipeline.
2. Remove `text.split(".")` context reconstruction from `phentrieve/llm/pipeline.py`.
3. Stop converting phase-1 exceptions into empty predictions.
4. Preserve `(term_id, assertion)` variants and evidence spans.
5. Move phase-2 mapping to structured output.

### Tier 2: highest impact on cost and speed

6. Add token counting preflight for API and benchmark paths.
7. Make public model choice server-owned, not caller-owned.
8. Reuse a persistent Gemini client instead of constructing one per request.
9. Add benchmark batch mode using Gemini Batch API.
10. Add caching for repeated prompt scaffolding and repeated phrase mappings
    where safe.

### Tier 3: multilingual robustness

11. Make LLM extraction consume chunk ids and original spans from the shared
    chunking path.
12. Expand LLM integration tests to German and at least one additional supported
    language path.
13. Keep language-specific conjunction and segmentation rules centralized in the
    existing text-processing resources.
14. Avoid introducing English-only normalization heuristics as if they were
    multilingual.

## Open Questions For Follow-Up Work

- Which existing chunking strategy gives the best accuracy/cost tradeoff for
  LLM extraction: `simple`, `semantic`, or
  `sliding_window_punct_conj_cleaned`?
- Is chunk-level extraction enough, or do some phenotypes require neighboring
  chunk context?
- How much of the current benchmark loss comes from:
  - phase-1 extraction recall
  - retrieval candidate recall
  - phase-2 mapping precision
- Should multilingual support remain "language passed in by caller", or should
  the LLM path also support auto-detection plus per-chunk language handling?

## Suggested Immediate Follow-Up PR Scope

Keep the next PR narrow:

1. integrate shared chunking into the LLM path
2. preserve chunk/span provenance
3. convert phase-2 mapping to structured output
4. add token counting and model allowlisting
5. add multilingual LLM-path tests

That would materially improve quality while staying consistent with the rest of
the codebase.

## Implementation Update: Grounded Whole-Document Fixes

An implementation pass was completed after this review to stabilize the new
grounded whole-document mode and measure its remaining performance gap against
legacy whole-document extraction.

### What changed

The following changes were implemented and verified:

- grounded phase-1 prompt construction now uses the grounded chunk index
  directly instead of sending both the raw full note and a duplicated chunk
  index
- grounded phase-1 now uses a higher output-token ceiling
  (`DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS = 65536`) instead of sharing the
  lower legacy phase-1 budget
- Gemini token preflight now uses the SDK `count_tokens` API instead of a
  character-count heuristic
- retry output-token escalation is now capped against the provider maximum and
  the grounded phase-1 maximum
- additional debug logging was added for:
  - grounded phase-1 request shape
  - prompt token counts
  - chunk count
  - prompt character size
  - grounded chunk index size

Files changed:

- `phentrieve/llm/config.py`
- `phentrieve/llm/pipeline.py`
- `phentrieve/llm/provider.py`
- `phentrieve/text_processing/full_text_service.py`
- `tests/unit/llm/test_prompts.py`
- `tests/unit/llm/test_provider.py`

### Why these fixes were necessary

The grounded mode regression was not primarily caused by the local chunking
pipeline. The critical failure mode was output-side:

- grounded phase-1 produced a larger structured response than legacy phase-1
- the prompt duplicated note content by including both raw note text and the
  chunk index
- the existing phase-1 output budget was too low for some grounded documents
- token preflight was too approximate to explain or guard the real request
  shape

The clearest failure case was `GeneReviews_NBK1277`, which failed
deterministically in grounded phase-1 with `finish_reason=MAX_TOKENS` and
truncated structured JSON before the fix.

### Verified benchmark impact

Three relevant benchmark artifacts were compared:

- broken grounded run:
  `results/llm/pr216_grounded_benchmark.json`
- fixed grounded run:
  `results/llm/pr216_grounded_benchmark_fix1.json`
- legacy baseline:
  `results/llm/llm_benchmark_20260416T_full_genereviews_phrasefix.json`

#### Full benchmark results

Broken grounded:

- `9/10` completed, `1` failed
- wall clock: `884.05s`
- avg/case: `88.40s`
- prompt tokens: `68,769`
- completion tokens: `21,647`
- total tokens: `90,416`
- API calls: `50`
- assertion-aware micro F1: `0.7500`
- assertion-aware macro F1: `0.7601`

Fixed grounded:

- `10/10` completed, `0` failed
- wall clock: `744.10s`
- avg/case: `74.41s`
- prompt tokens: `64,056`
- completion tokens: `24,539`
- total tokens: `88,595`
- API calls: `49`
- assertion-aware micro F1: `0.7682`
- assertion-aware macro F1: `0.7780`

Legacy baseline:

- `10/10` completed, `0` failed
- wall clock: `668.23s`
- avg/case: `66.82s`
- prompt tokens: `66,088`
- completion tokens: `14,531`
- total tokens: `80,619`
- API calls: `49`
- assertion-aware micro F1: `0.7629`
- assertion-aware macro F1: `0.7918`

#### Interpretation

The implementation pass fixed the hard regression:

- the grounded benchmark no longer fails on `NBK1277`
- total runtime improved by about `140s` compared with the broken grounded run
- API calls returned to the legacy-level count (`49`)
- micro F1 improved over both the broken grounded run and the legacy baseline

However, grounded mode still has a measurable residual overhead versus legacy:

- about `+75.9s` wall-clock over the full 10-document benchmark
- much higher completion-token usage than legacy
- lower macro recall and macro F1 than legacy despite improved micro F1

That pattern suggests the current grounded mode is now stable, but it is still
paying for denser structured extraction and additional LLM reasoning in later
mapping stages.

### Document-level observations

The original grounded failure case now succeeds in the full benchmark:

- `GeneReviews_NBK1277`
  - broken grounded: failed in phase 1
  - fixed grounded: `90.63s`, `phase1=54.05s`, `phase2b_llm=35.21s`, `5`
    requests
  - legacy: `68.51s`, `phase1=19.89s`, `phase2b_llm=47.80s`, `5` requests

Notable improvements vs the broken grounded run:

- `GeneReviews_NBK1257`: `82.68s -> 45.21s`
- `GeneReviews_NBK1379`: `126.87s -> 78.22s`
- `GeneReviews_NBK532447`: `148.99s -> 106.99s`

Slowest documents in the fixed grounded run:

- `GeneReviews_NBK532447`: `106.99s`
- `GeneReviews_NBK1277`: `90.63s`
- `GeneReviews_NBK550349`: `87.07s`

The remaining slowdown is still dominated by remote Gemini time in `phase1`
and `phase2b_llm`, not by local retrieval or chunk generation.

### Verification state

Targeted unit coverage added for the new behavior:

- grounded phase-1 prompt omits duplicated full-note text when chunk grounding
  is present
- Gemini token preflight uses the SDK token counting API

Focused validation completed:

- targeted unit suites for prompts, provider, pipeline, and full-text service
- isolated reruns for `GeneReviews_NBK1277` in grounded and legacy modes
- full 10-document grounded benchmark rerun
- `make check`
- `make typecheck-fresh`
- `make test`

### Task 9 outcome

The internal shared-chunk refactor is now functionally complete and benchmarked.
The final Task 9 work also fixed two verification regressions discovered late:

- legacy phase-1 schema selection was incorrectly using grounded structured
  output requirements, which broke `whole_document_legacy` runs
- Gemini `count_tokens` was being called with unsupported `systemInstruction`
  config during grouped preprocessing

Those were corrected by:

- selecting `LLMExtractedPhenotypes` for non-grounded legacy phase-1 requests
- using Gemini `countTokens` against the effective rendered prompt content
  rather than the unsupported `systemInstruction` field
- updating CLI and pipeline tests so non-grounded paths do not incorrectly
  expect grounded provenance unless grounded chunks are actually supplied

Final repo verification completed cleanly:

- `make check`
- `make typecheck-fresh`
- `make test`

### Final benchmark comparison

Benchmarks compared:

- legacy baseline:
  `results/llm/llm_benchmark_20260416T_full_genereviews_phrasefix.json`
- fixed grounded pre-refactor:
  `results/llm/pr216_grounded_benchmark_fix1.json`
- post-refactor grounded:
  `results/llm/pr216_grounded_chunkrefactor.json`
- post-refactor legacy control:
  `results/llm/pr216_legacy_control_postrefactor.json`

Key result:

- the shared-chunk refactor is correct and stable, but it did not meet the
  latency target

Measured outcomes:

- fixed grounded pre-refactor:
  - wall clock `744.10s`
  - avg/case `74.41s`
  - micro F1 `0.7682`
  - macro F1 `0.7780`
  - weighted F1 `0.7666`
  - tokens `88,595`
  - API calls `49`
- post-refactor grounded:
  - wall clock `1051.52s`
  - avg/case `105.15s`
  - micro F1 `0.7631`
  - macro F1 `0.7746`
  - weighted F1 `0.7620`
  - tokens `91,892`
  - API calls `60`
- post-refactor legacy control:
  - wall clock `321.23s`
  - avg/case `32.12s`
  - micro F1 `0.6364`
  - macro F1 `0.4644`
  - weighted F1 `0.5456`
  - tokens `40,262`
  - API calls `30`

Interpretation:

- the grouped grounded path preserved benchmark quality roughly near the prior
  grounded run, but slightly lower on micro, macro, and weighted F1
- the grouped grounded path became substantially slower and more expensive
- the legacy control is now much faster and cheaper, but its quality is too low
  to treat as a viable production fallback

Slowest post-refactor grounded documents:

1. `GeneReviews_NBK321516` `149.50s`
2. `GeneReviews_NBK532447` `130.96s`
3. `GeneReviews_NBK550349` `126.65s`

For those slowest cases, the dominant time remained remote Gemini work:

- `NBK321516`: `phase2b_llm` dominated (`78.18s`), with large `phase1`
  contribution (`46.91s`)
- `NBK532447`: `phase2b_llm` dominated (`71.84s`)
- `NBK550349`: `phase1` dominated (`70.76s`), with `phase2b_llm` still large
  (`35.29s`)

Across the 10-doc slice, the dominant phase split was even:

- grouped grounded: `phase1` dominant in 5 docs, `phase2b_llm` dominant in 5
  docs
- legacy control: `phase1` dominant in 5 docs, `phase2b_llm` dominant in 5
  docs

So the original diagnosis still holds:

- local retrieval is not the bottleneck
- the refactor shifted too much cost into multi-group phase-1 and repeated
  phase-2B mapping calls
- simply "using the shared chunk pipeline" is not enough to improve latency

### Refined next steps

The next optimization work should not add arbitrary extraction caps. The
stability problem was fixed without them, and the remaining gap is now more
clearly attributable to structured-output volume and downstream LLM mapping
cost.

The highest-signal next steps are:

1. reduce phase-2B LLM workload before further architectural chunking changes
2. keep grounded provenance benefits while lowering request fanout
3. preserve the new logging so future regressions are diagnosable from
   artifacts

Concrete follow-up candidates:

1. reduce grouped request fanout by merging only when the token budget gain is
   real, not merely available
2. move more unresolved phrase decisions out of `phase2b_llm` via stronger
   local or retrieval-side narrowing
3. deduplicate unresolved mappings more aggressively across neighboring groups
4. make phase-2 structured outputs more compact if current mapping schemas are
   verbose
5. keep per-document token and phase breakdown summaries in benchmark output so
   regressions are visible without log scraping
6. investigate persistent Gemini client reuse and other provider-side latency
   reductions

This narrows the original review conclusion:

- the grounding design is now operationally viable
- the biggest remaining opportunity is not "fix sentence splitting first"
- the next meaningful speed win is likely in mapping-time LLM usage, not only
  in phase-1 extraction
- multilingual chunking integration is now implemented internally, but the next
  optimization step must focus on request fanout and mapping-time LLM cost

## Sources

Codebase references:

- `phentrieve/llm/pipeline.py`
- `phentrieve/llm/provider.py`
- `phentrieve/text_processing/full_text_service.py`
- `phentrieve/text_processing/chunkers.py`
- `phentrieve/text_processing/config_resolver.py`
- `phentrieve/text_processing/resource_loader.py`
- `tests/integration/test_chunking_pipeline_integration.py`

External references:

- Google Gemini structured output docs: https://ai.google.dev/gemini-api/docs/structured-output
- Google Gemini long-context docs: https://ai.google.dev/gemini-api/docs/long-context
- Google Gemini Batch API docs: https://ai.google.dev/gemini-api/docs/batch-api
- Google Gemini token counting docs: https://ai.google.dev/api/tokens
- ICU BreakIterator / boundary analysis docs: https://unicode-org.github.io/icu-docs/apidoc/dev/icu4c/classicu_1_1BreakIterator.html
- Unicode Text Segmentation (UAX #29): https://www.unicode.org/reports/tr29/tr29-29.html
