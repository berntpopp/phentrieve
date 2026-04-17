# Comparative Review: `feat/llm-full-text-lean-v1`

**Date:** 2026-04-16
**Scope:** Phentrieve branch `feat/llm-full-text-lean-v1` reviewed against
phentrieve-bench branches `feature/llm-annotation-system` and
`feature/agentic-judge-mode`, and against the 2024–2026 external literature
(RAG-HPO, PheNormGPT, AutoPCR, DeepRare, FastHPOCR, PhenoBERT family).
**Method:** Four parallel review passes — one per target — synthesised here.
**Non-negotiable constraint:** Phentrieve is a **multilingual** system (English,
German, and future languages through BioLORD multilingual embeddings plus
per-language chunkers). Every recommendation below carries a transferability
verdict; anything that silently assumes English is called out.

---

## 1. Current state of `feat/llm-full-text-lean-v1`

### Architecture summary

Whole-note clinical text → `TextProcessingPipeline` produces grounded chunks
(language-aware) → **Phase 1** grouped or whole-note extraction via Gemini
structured output, anchored to chunk IDs (`phentrieve/llm/pipeline.py`,
~1075 LOC) → **Phase 2A** batch dense retrieval of HPO candidates per
extracted phrase → **Phase 2B** optional LLM mapping with neighbour-chunk
context when local heuristics do not resolve a phrase. Pydantic schemas
throughout (`LLMGroundedExtractedPhenotypes`,
`LLMBatchMappingSelections`), typed failures (`LLMPipelinePhaseError`),
token preflight through `provider.count_tokens()`, assertion-aware dedup
keyed on `(term_id, assertion)`.

### Strengths that must be preserved in any refactor

- **Chunk-anchored provenance with multilingual chunking.** Evidence records
  carry `chunk_ids`, `evidence_text`, and span positions. This is a genuine
  lead over RAG-HPO (English, sentence-reconstructed) and over both bench
  branches (no real span provenance).
- **Typed failure surface + per-group trace.** `LLMPipelinePhaseError` plus
  structured trace dictionaries make the pipeline debuggable. The bench
  annotator path silently truncates on iteration caps and is much harder to
  audit.
- **Token-budgeted grouping.** `build_extraction_groups`
  (`phentrieve/llm/preprocessing.py:64–135`) respects a real token budget
  rather than character heuristics, and preserves neighbour overlap.
- **Assertion/category-aware extraction** with Abnormal/Normal/Suspected/
  Family_History mapped to present/negated/uncertain.
- **CLI + benchmark parity.** Both paths use `TwoPhaseLLMPipeline`; an
  internal `whole_document_legacy` vs `whole_document_grounded` switch
  allows safe A/B comparison.

### Weaknesses ranked by impact

All file:line anchors come from the current branch tip.

| # | Issue | Anchor | Category |
|---|---|---|---|
| 1 | No phrase dedup before Phase-2A retrieval / Phase-2B mapping; overlapping groups re-query identical strings | `phentrieve/llm/pipeline.py:535–591`, `:749–1064` | tokens + speed |
| 2 | `provider.count_tokens()` invoked synchronously per group during preflight (roughly 50–100 ms RTT × N groups) | `phentrieve/llm/preprocessing.py:87,105`; `phentrieve/llm/provider.py:241` | speed |
| 3 | Phase-1 groups run strictly serially; no `asyncio` or thread pool despite pure I/O | `phentrieve/llm/pipeline.py:470–514` | speed |
| 4 | Phase-2B prompt interleaves static instructions with per-item payload, breaking Gemini implicit prefix caching | `phentrieve/llm/pipeline.py:276` | tokens |
| 5 | Neighbour chunks (±1) always included in grounded context, inflating every Phase-2B prompt | `phentrieve/llm/pipeline.py:613–627` | tokens |
| 6 | `ValueError` on any single chunk exceeding 30 k tokens kills the whole run | `phentrieve/llm/preprocessing.py:114–120` | correctness |
| 7 | `_resolve_with_mapping_prompt()` is a 300+-line state machine with no per-batch retry or timeout granularity | `phentrieve/llm/pipeline.py:749–1064` | hygiene |
| 8 | Many `dict[str, Any]` crossings on retrieval and grounded-context boundaries | pipeline signatures | hygiene |
| 9 | `tests/unit/test_llm_benchmark.py:14–19` globally mocks `_build_grounded_chunks`, so preprocessing + benchmark integration is never exercised | tests | tests |
| 10 | Oversize-chunk branch, `count_tokens` failure branch, and static helpers `_extract_first_result_list` / `_build_grounded_context` are untested | tests | tests |

### Cost model (grounded mode, 10–50 chunks per note)

- Phase 1: roughly 5 k–31 k input tokens + 300–1000 output, 1–5 calls.
- Phase 2A: pure dense retrieval, no LLM cost.
- Phase 2B: 2 k–10 k input + 200–500 output, 0–5 calls.
- Wall-clock today: roughly 15–30 s per note.
- Measured redundancy: 500–3000 tokens per note and 200–500 ms wasted on
  items 1, 2, 4, 5 above.

### Test-coverage gaps worth fixing alongside any PR

- Single chunk > token budget branch (`preprocessing.py:102–120`).
- `count_tokens` failure modes (API error, invalid model).
- Integration test that runs the benchmark path without mocking
  `_build_grounded_chunks` on at least a two-document subset.
- `_extract_first_result_list` / `_build_grounded_context` unit tests.
- Multilingual preprocessing unit coverage for German chunking (integration
  exists in `test_grounded_pipeline_integration.py`, unit does not).

---

## 2. What to steal from phentrieve-bench `feature/llm-annotation-system`

### Summary

The bench branch ships five annotation modes (DIRECT, TOOL_TERM, TOOL_TEXT,
RETRIEVAL_ONLY, TWO_PHASE) plus a combined post-processor. The
**two-phase** mode reaches F1 0.768 at the lowest cost on the 10-doc
GeneReviews benchmark ($0.049, 15.7 F1/$), and the **tool-term** mode
reaches F1 0.762 at the fastest wall-clock (10.3 s/doc).

### Ideas ranked by transferability to our multilingual pipeline

#### A. Four-strategy deterministic local matcher
`/phentrieve/llm/annotation/two_phase.py:269–450`
Exact token-set → normalized exact phrase → substring with word-boundary
guard → RapidFuzz ≥80. Resolves roughly 57 % of phrases with zero LLM cost
in the bench benchmark.

**Multilingual verdict: adopt only with an explicit language gate.**

- Exact and normalized-exact matching work in any language *if* the HPO
  label index for that language is populated. HPO has official translations
  (including a German one), but coverage is uneven — many HPO terms do not
  have a German label yet.
- Substring matching with a word-boundary guard requires a word-boundary
  definition. Word-based languages (en, de, fr, it, es) are fine.
  Space-less / agglutinative languages (ja, zh) need a tokenizer-specific
  boundary; do not ship substring matching for those without one.
- RapidFuzz ≥80 is an English-tuned threshold. It must be recalibrated per
  language on a held-out set, or disabled outside English until we have
  data. German compound nouns especially will confound any fixed threshold.

**Recommended adoption**: implement the matcher as a language-aware
strategy chain. For each language, configure which strategies are enabled
and what threshold they use; default to disabled for any language without a
calibrated threshold. Never hard-code 0.80.

#### B. Modifier term filter
`/phentrieve/llm/annotation/tool_guided.py:30–109`
77-ID frozenset covering onset (`HP:0003623` Neonatal onset, etc.) and
inheritance (`HP:0000006` AD, etc.) terms, stripped post-annotation.

**Multilingual verdict: fully transferable. Adopt as-is.** HPO IDs are
language-neutral; this is a pure ID-based filter. Copy-paste the frozenset
and wire it into the post-processing step.

#### C. Granular `TimingEvent` + `TokenUsage` in every result
`/phentrieve/llm/types.py:78–98, 162–242`
Per-phase (`label`, `duration_seconds`, `category ∈ {llm, tool, postprocess}`)
timing plus cumulative prompt/completion/api_call counters.

**Multilingual verdict: fully transferable. Adopt.** Observability is
language-agnostic and directly enables the `$/note` reporting that the
literature is missing (see §4).

#### D. Hallucination filter
`/phentrieve/llm/annotation/tool_guided.py:259–338`
Drop any predicted HPO ID that is absent from every tool / retrieval result
in the trace.

**Multilingual verdict: fully transferable. Adopt as a safety floor.** It
is a set-difference on IDs; language does not enter.

### Anti-patterns not to copy

- Unbounded tool-calling loops (bench caps at 8; still too high).
- Top-200 retrieval without reranking (bench found F1 0.463 at top-50,
  improved to 0.768 only with hybrid selection).
- Dumping 262–914 candidates into a flat prompt table with no lexical
  prefilter.

---

## 3. What to steal from phentrieve-bench `feature/agentic-judge-mode`

### Summary

Two-pass architecture: an **Annotator** LLM selects candidates from the
retrieval pipeline; an independent **Judge** LLM validates and corrects
errors. Achieves F1 0.777 micro on the 10-doc benchmark, narrowing the gap
to RAG-HPO (0.790) to 1.3 pp. 46 API calls vs the two-phase 120, but 4× the
dollar cost ($0.209 vs $0.049).

### Ideas ranked by transferability

#### E. Asymmetric cost language in the mapping prompt
`en_judge_conservative.yaml:28–36` — "DEFAULT is KEEP; 90 % confidence to
REMOVE, 60 % confidence to KEEP." Experiment 4 in the bench report: +0.72 pp
F1 at zero API cost overhead.

**Multilingual verdict: adopt, but it is a per-language prompt-engineering
commitment, not a one-off.** Every language's mapping prompt template must
gain an equivalent stanza. Do not machine-translate this: the numeric
thresholds and the "default action is KEEP" framing need to land naturally
in each language or they will not influence model behaviour. Budget a
native-speaker review per language we ship.

#### F. Pass `retrieval_score` into the mapping prompt
`agentic_judge.py:884–901`. Experiment 1: +0.88 pp F1 and −4.8 % tokens
because the judge can anchor on a numeric confidence signal.

**Multilingual verdict: fully transferable. Adopt.** Scores are numeric; the
phrasing around them (e.g. "Score ≥ 0.8: require clear counter-evidence to
remove") joins the per-language prompt translation burden, but the data
plumbing is identical across languages.

#### G. Selective judge only on low-confidence deferred phrases
Hybrid proposal from the bench review: run the full judge only on phrases
where Phase-2A/deterministic-matcher confidence is below τ (e.g. 0.7).
Estimated cost: $0.054 / 10 docs vs $0.049 two-phase / $0.209 full judge,
with ~+0.5 pp F1.

**Multilingual verdict: adopt, but τ must be per-language.** The confidence
distribution coming out of dense retrieval depends on how well BioLORD
indexes a given language's clinical text. German clinical text on a BioLORD
multilingual model typically has a flatter score distribution than English.
A single global τ will over-defer or under-defer in non-English paths.
Store τ in the benchmark config per language, tune it on a held-out set per
language, and fall back to "always defer" until a language has enough
labelled data to tune τ.

### Red flags observed in bench — do not replicate

- Stacking the conservative judge prompt with a 3-judge minority-veto
  ensemble collapsed precision by 10.1 pp on NBK1379. Never stack
  over-retention mechanisms.
- Iteration-cap blowups on the agentic loop (`max_iterations=8`). We have
  no tool-calling loop today and should not add one without per-iteration
  telemetry and a hard dollar budget.
- Self-consistency sampling at temperature 0.3 adds 3× cost with unclear
  benefit; do not ship.

---

## 4. What to learn from the external literature (RAG-HPO, PheNormGPT, AutoPCR, DeepRare, FastHPOCR)

### Headline takeaways

1. **The two-phase architecture is correct.** RAG-HPO reports Llama-3 70 B
   collapses from F1 0.80 with retrieval to F1 0.12 without on the same
   112-case set. The retrieval step is load-bearing, not decorative
   (Genome Medicine 2025).
2. **Nobody publishes `$/note`.** RAG-HPO, PheNormGPT, AutoPCR, DeepRare,
   and Hier 2024 all omit dollar cost. Phentrieve is positioned to be the
   first; that requires idea C above.
3. **Benchmarks do not transfer.** FastHPOCR reports F1 0.85 on GSC+ and
   F1 0.48 on the RAG-HPO real-reports set. Our benchmark story must
   include real case reports, not only curated corpora.
4. **BioLORD is already the default HPO embedder in 2025 agentic stacks**
   (DeepRare, AutoPCR). Our choice is mainstream.
5. **15–45 s wall-clock per note is the literature norm.** We are in range;
   there is headroom.

### Ideas ranked by transferability

#### H. Top-k candidate payload with full metadata
AutoPCR (arXiv 2507.19315) ships candidates with label + synonyms +
definition + UMLS xrefs, achieving 79.25 % average F1.

**Multilingual verdict: adopt with fallback rules.** HPO definitions exist
in English, with partial translations for a handful of languages. For
non-English query paths, ship the English definition as fallback but label
it as such in the prompt so the model knows it is cross-lingual context.
Synonyms ship per-language when available, else English. The prompt
template must express this honestly in each language.

#### I. Confidence-gated routing (AutoPCR's τ₁ = 0.95 / τ₂ = 0.85)
HIGH-confidence predictions bypass the LLM; LOW-confidence are dropped;
only the middle band is routed to the mapping LLM.

**Multilingual verdict: the concept transfers; the thresholds do not.**
The τ values from AutoPCR are calibrated on their English eval set with
SapBERT scores. Phentrieve uses BioLORD multilingual. Ship the routing
mechanism with per-language τ configs, and default to "always route to
LLM" (τ₁=∞, τ₂=−∞) in any language not yet calibrated. Same discipline as
idea G.

#### J. Explicit phrase rewriting before retrieval
DeepRare (arXiv 2506.20430) adds a "modify phenotype names via LLM
prompting" step between extraction and BioLORD normalization, with a
cosine ≥ 0.8 threshold.

**Multilingual verdict: partially transferable.** An LLM can rewrite in any
language, but rewriting within-language keeps the retrieval step honest;
cross-language rewriting (rewrite German clinical phrase to English before
embedding) leaks model-specific assumptions into recall. Adopt only
same-language rewriting. Do not use it as a cheap substitute for
multilingual retrieval coverage.

### Caveats on literature claims

- FastHPOCR on GSC+ vs real case reports (F1 0.85 vs 0.48) is the reminder
  that the community historically over-reports. Do not benchmark only on
  GSC+-style corpora.
- Hier 2024 (GPT-4 direct prompting) reaches only normalization accuracy
  0.579 — confirming that LLM-alone without retrieval is not competitive.

---

## 5. Top five concrete PRs, ranked by impact/effort and multilingual safety

Each item is scoped to one PR, with a transferability verdict. Items 1, 2,
3 are strictly language-neutral; items 4 and 5 require per-language care.

### PR 1 — Dedup phrases before Phase-2A retrieval and Phase-2B mapping

**Files:** `phentrieve/llm/pipeline.py` (`_retrieve_candidates`,
`_resolve_with_mapping_prompt`), new helper; unit tests in
`tests/unit/llm/test_pipeline.py`.
**Effort:** 3–4 h.
**Impact:** 30–50 % fewer retrieval and mapping calls on overlap-heavy
notes; no quality change.
**Multilingual verdict:** Fully transferable. Use Unicode normalization
(NFKC + casefold) in the dedup key; never lowercase naively (breaks
German eszett and Turkish dotted-i). Do not dedup across languages — dedup
within `(language, normalized_phrase, category)`.
**Risk:** Low.

### PR 2 — Parallelize Phase-1 group calls + replace tight `count_tokens` loop with a heuristic

**Files:** `phentrieve/llm/pipeline.py:470–514`,
`phentrieve/llm/preprocessing.py:64–135`, `phentrieve/llm/provider.py`.
**Effort:** 1–2 days incl. thread-safety for the provider.
**Impact:** Roughly 2× wall-clock reduction on multi-group notes;
200–500 ms per note saved from preflight.
**Multilingual verdict:** Fully transferable. The `len(text)/4` token
heuristic is mis-calibrated for non-Latin scripts (CJK ≈ 1 token per 1–2
chars, not 4); either keep the real `count_tokens` call on boundary groups
only, or pick a language-aware divisor. Ship a per-language map; default
to calling the real tokenizer when unsure.
**Risk:** Medium. Add a regression test comparing heuristic vs real
tokenizer on a German and (future) Japanese sample if we add CJK.

### PR 3 — Emit `TimingEvent` and `TokenUsage` per phase in the result

**Files:** `phentrieve/llm/types.py`, `phentrieve/llm/pipeline.py`,
`phentrieve/llm/full_text_service.py`, benchmark runner.
**Effort:** 1 day.
**Impact:** Unlocks `$/note` reporting and per-phase SLOs. Prerequisite
for defensible publication numbers.
**Multilingual verdict:** Fully transferable. Instrumentation is
language-neutral.
**Risk:** Low.

### PR 4 — Mapping-prompt refactor: stable prefix + inject `retrieval_score` + KEEP-default language

Combines external idea E + F with local weakness 4 (prefix-cache layout).

**Files:** `phentrieve/llm/pipeline.py:276` (payload construction),
`phentrieve/llm/prompts/templates/two_phase/*_mapping*.yaml` for every
supported language.
**Effort:** 1 day code; additional per-language prompt review (estimate
2–3 h per language for a native speaker).
**Impact:** +0.7–1.5 pp F1 (bench Exp1 + Exp4 ≈ +1.6 pp combined; conservative
estimate after language drift). Plus prompt-cache savings on Gemini.
**Multilingual verdict:** The data plumbing (stable prefix, retrieval-score
field) is language-neutral. The asymmetric-cost framing ("KEEP by default,
90 % to remove, 60 % to keep") must be rewritten natively per language.
Do **not** machine-translate the stanza — commit to native-speaker
review as part of the PR definition of done. Block the PR on German
coverage at minimum since we already ship German.
**Risk:** Medium — prompt phrasing is empirical. Ship behind a benchmark
A/B for at least English and German before merging.

### PR 5 — Port the deterministic local matcher as a Phase-2A→2B pre-filter, language-gated

Combines bench idea A with external idea I (confidence routing).

**Files:** new module `phentrieve/llm/annotation/local_matcher.py`,
wire-up in `phentrieve/llm/pipeline.py`, per-language config.
**Effort:** 2–3 days incl. per-language threshold calibration harness.
**Impact:** Resolves a large fraction of phrases without any LLM call in
Phase 2B — bench reports ~57 % on English. Expect lower in German until
synonym coverage improves.
**Multilingual verdict:** Highest caution of the five.

- Ship the matcher as a chain of strategies with a per-language config
  saying which strategies run and at what threshold.
- **English default:** exact → normalized-exact → substring → RapidFuzz ≥ 80.
- **German default:** exact → normalized-exact → substring only. Disable
  RapidFuzz until calibrated; German compound nouns will false-positive
  with any fixed ratio threshold.
- **Every other language:** exact → normalized-exact only. Everything else
  off by default.
- The confidence threshold τ that decides whether to defer to Phase 2B is
  per-language. Default τ = "always defer to LLM" for unfamiliar
  languages.
- Add a calibration harness that, given a labelled dev set, sweeps
  thresholds per language and reports precision/recall/cost.
- Never introduce a language-blind global threshold.

**Risk:** Medium-high. This PR is the one that can silently degrade
non-English performance if shipped without the per-language discipline.
The PR is gated on (a) a calibration harness, (b) a German benchmark set,
and (c) a regression test that fails if a non-English language is routed
through English thresholds.

### Not in the top five, intentionally

- **Selective judge on deferred phrases (idea G)**: parked until PR 5 lands
  the confidence routing and we have per-language τ values. The judge is
  only useful if "low confidence" is a well-defined concept in that
  language.
- **Phrase rewriting (idea J)**: parked. Only worth doing once we see
  concrete low-recall cases that look like surface-form mismatches; not a
  speculative addition.

---

## 6. Benchmarking recommendations

The current `tests/unit/test_llm_benchmark.py:14–19` over-mocks and does not
represent the real pipeline. Alongside the PRs above:

- **Add a real-notes eval set.** Use the RAG-HPO 112-case corpus for
  English. For German, build or source at least a 20–30 case set from
  public case reports; without this, every claim about non-English
  performance is hand-waving.
- **Report P/R/F1 both ID-only and assertion-inclusive.** The bench
  branch already does this via `ExtractionMetrics` / `CorpusMetrics`.
- **Publish `$/10-docs` with model, prompt version, language, and
  wall-clock.** Nobody else does.
- **Break out Phase-1 phrase recall separately from end-to-end F1.** The
  literature conflates them; we can do better. It is also the metric most
  sensitive to language-specific Phase-1 quality.
- **Run the benchmark per language.** A single aggregated number hides the
  multilingual regressions that matter.

---

## 7. What to tell reviewers / stakeholders

- `feat/llm-full-text-lean-v1` has the right architecture. The ceiling is
  not in the shape; it is in dedup, parallelism, prompt layout, and
  observability — all fixable with low-risk PRs.
- The bench branches explore the design space further, and several of
  their ideas are worth adopting — but most of them were tuned on English
  and need per-language care before they go into our multilingual
  pipeline.
- RAG-HPO is the closest external comparator; its F1 0.80 on real case
  reports is the bar we should measure against, on English first.
- We are one of the few groups positioned to publish multilingual numbers
  and per-note dollar cost. That is a defensible differentiator. Item 3
  (TimingEvent + TokenUsage) is the unlock.

---

## 8. File anchors reproduced for convenience

Phentrieve branch `feat/llm-full-text-lean-v1`:

- `phentrieve/llm/pipeline.py:276` — Phase-2B payload construction
- `phentrieve/llm/pipeline.py:470–514` — Phase-1 group loop
- `phentrieve/llm/pipeline.py:535–591` — `_retrieve_candidates`
- `phentrieve/llm/pipeline.py:613–627` — `_build_grounded_context`
- `phentrieve/llm/pipeline.py:749–1064` — `_resolve_with_mapping_prompt`
- `phentrieve/llm/preprocessing.py:64–135` — `build_extraction_groups`
- `phentrieve/llm/provider.py:241–260` — `count_tokens`
- `phentrieve/llm/full_text_service.py:408–428` — preflight + warning path
- `tests/unit/test_llm_benchmark.py:14–19` — the over-mocking that masks
  real preprocessing
- `.planning/specs/2026-04-15-llm-full-text-lean-v1-design.md`
- `.planning/specs/2026-04-16-llm-cli-grounded-whole-note-design.md`
- `.planning/active/2026-04-16-llm-cli-grounded-whole-note-implementation-plan.md`
- `.planning/active/2026-04-16-llm-shared-chunk-pipeline-internal-refactor-plan.md`

phentrieve-bench `feature/llm-annotation-system`:

- `phentrieve/llm/annotation/two_phase.py:269–450` — 4-strategy matcher
- `phentrieve/llm/annotation/tool_guided.py:30–109` — modifier-term filter
- `phentrieve/llm/annotation/tool_guided.py:259–338` — hallucination filter
- `phentrieve/llm/types.py:78–98, 162–242` — TimingEvent / TokenUsage

phentrieve-bench `feature/agentic-judge-mode`:

- `phentrieve/llm/annotation/agentic_judge.py:884–901` — retrieval score
  pass-through
- `phentrieve/llm/prompts/templates/.../en_judge_conservative.yaml:28–36` —
  asymmetric KEEP-default framing
- `EXPERIMENT-REPORT.md` Exp1, Exp3, Exp4, triple-stack analysis

External:

- RAG-HPO — <https://pmc.ncbi.nlm.nih.gov/articles/PMC11643181/>,
  <https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-025-01521-w>,
  <https://github.com/PoseyPod/RAG-HPO>
- PheNormGPT — <https://pmc.ncbi.nlm.nih.gov/articles/PMC11498178/>
- AutoPCR — <https://arxiv.org/html/2507.19315>
- DeepRare — <https://arxiv.org/html/2506.20430v1>
- FastHPOCR —
  <https://academic.oup.com/bioinformatics/article/40/7/btae406/7698025>
- Hier 2024 GPT-4 normalization — <https://arxiv.org/html/2408.01214v2>

---

## 9. 2026-04-17 update: branch strategy and revised highest-ROI PR order

### Current benchmark reality on this branch

Since the original comparative review, `feat/llm-full-text-lean-v1` has
completed a full grounded whole-note hardening pass, the shared-chunk
internal refactor, and a follow-up regression-fix pass for routing and
grounded whole-note fallback behavior. The benchmark state is now:

| Variant | Wall clock | API calls | Prompt tokens | Completion tokens | Total tokens | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy baseline (`llm_benchmark_20260416T_full_genereviews_phrasefix.json`) | 668.23 s | 49 | 66,088 | 14,531 | 80,619 | 0.7629 | 0.7918 |
| Grounded fixed pre-refactor (`pr216_grounded_benchmark_fix1.json`) | 744.10 s | 49 | 64,056 | 24,539 | 88,595 | 0.7682 | 0.7780 |
| Grounded shared-chunk refactor (`pr216_grounded_chunkrefactor.json`) | 1051.52 s | 60 | 68,686 | 23,206 | 91,892 | 0.7631 | 0.7746 |
| Grounded routing + concurrency regression (`pr216_grounded_routing_phase1_parallel.json`) | 1158.07 s | 83 | 84,880 | 23,343 | 108,223 | 0.7288 | 0.7404 |
| Grounded routing + concurrency fixed (`pr216_grounded_routing_phase1_parallel_fix4.json`) | 899.95 s | 55 | 75,562 | 19,834 | 95,396 | 0.7764 | 0.7870 |

Interpretation:

- The grounded path is no longer broken. The `NBK1277` truncation failure was
  fixed and the CLI/benchmark path is operationally stable.
- The routing/concurrency regression was real, but it has now been corrected.
  The fixed rerun beats the shared-chunk refactor on all headline metrics and
  exceeds the grounded fixed baseline on micro F1.
- The remaining bottleneck is no longer the accidental grouped whole-note path.
  It is residual remote LLM work in Phase 2B and still-heavy grounded context /
  prompt structure costs.

This materially changes the priority order. Another large internal
architecture refactor is not the next highest-value move. The next gains are
in mapping prompt/payload structure, adaptive context, and remaining token /
call reduction.

### Branch strategy

Do **not** split this work into a new feature branch yet. Keep landing the next
optimization sequence onto `feat/llm-full-text-lean-v1` until the CLI path is
mature enough to merge to `main`.

Rationale:

- We still need stable A/B comparison against the current grounded fixed
  baseline and the legacy control.
- The main open question is performance maturation, not product surface shape.
- Splitting prematurely would fragment benchmark history and make it harder to
  tell whether a change helped or hurt.

### Revised next-PR ranking by practical ROI

The original top-five list remains directionally sound, but some items are now
partially implemented and should be reframed as optimization PRs rather than
first-time feature PRs.

#### PR A — Confidence-gated Phase-2B routing

**Why it is now the top priority:** the shared-chunk refactor increased API
calls from 49 to 60. That is the cleanest measurable regression signal.

**Concrete scope:**

- Use retrieval score plus local-match confidence to classify phrases into:
  accept locally, defer to mapping LLM, or drop.
- Keep thresholds per language.
- Start conservatively:
  - English: allow score-gated local acceptance.
  - German: exact and normalized-exact first; defer more aggressively until
    calibrated.
- Log routing counts in benchmark output.

**Expected effect:**

- Speed: high
- Token cost: high
- Accuracy: likely positive or neutral if thresholds are conservative

**Status vs prior review:** implemented in a conservative form. The current
pipeline now routes some high-confidence English matches locally, preserves a
more conservative German path, logs routing counts in benchmark artifacts, and
has recovered the worst regression. The remaining work is calibration, not
first-time implementation.

#### PR B — Parallelize Phase-1 grouped extraction

**Why it is next:** grouped Phase-1 still runs serially, so multi-group notes
pay additive remote latency.

**Concrete scope:**

- Introduce bounded concurrency for group extraction calls.
- Preserve deterministic merge order and current failure accounting.
- Keep provider concurrency configurable and safe for Gemini quotas.

**Expected effect:**

- Speed: very high
- Token cost: none
- Accuracy: neutral

**Status vs prior review:** implemented with bounded concurrency and
thread-local provider accounting. One important correction was required after
the first landing: singleton whole-note cases must stay on the original
grounded whole-note path, otherwise prompt-shape drift hurts both speed and
quality. Multi-group notes now use the concurrent grouped path; the current
10-doc GeneReviews set mostly exercises the whole-note fallback rather than
true multi-group concurrency.

#### PR C — Mapping prompt restructure plus retrieval score

**Why it matters:** current Phase-2B remains expensive and is still the most
promising place to recover quality without adding calls.

**Concrete scope:**

- Refactor the mapping prompt into a stable instruction prefix plus compact
  per-item payload.
- Pass `retrieval_score` explicitly.
- Add the asymmetric KEEP-default decision framing in language-specific prompt
  templates.
- Benchmark English and German before making the prompt default.

**Expected effect:**

- Speed: medium
- Token cost: medium
- Accuracy: medium to high

**Status vs prior review:** still open. Retrieval scores are present in
candidate data but are not yet part of a redesigned stable-prefix mapping
contract, and the KEEP-default language has not yet been benchmarked as a
prompt-level change.

#### PR D — Adaptive grounded context

**Why it matters:** current grounded context always includes neighbor chunks,
inflating every Phase-2B request even when the primary chunk is sufficient.

**Concrete scope:**

- Default to primary chunk only.
- Add neighbors only for ambiguous contexts, uncertain assertion handling, or
  low-confidence retrieval cases.
- Emit observability counters for primary-only vs expanded-context calls.

**Expected effect:**

- Speed: medium
- Token cost: medium to high
- Accuracy: neutral to positive if the expansion heuristic is careful

**Status vs prior review:** still open. Neighbor context is still always on.

#### PR E — Heuristic token preflight with real-token fallback

**Why it matters:** `count_tokens()` calls are still in the extraction-group
builder and add synchronous latency with no quality upside.

**Concrete scope:**

- Use a heuristic token estimator for ordinary groups.
- Only call the real provider tokenizer near configured boundaries.
- Keep language-aware fallback rules so non-Latin scripts do not silently
  overflow.

**Expected effect:**

- Speed: low to medium
- Token cost: none
- Accuracy: neutral

**Status vs prior review:** partially implemented. The worst case has been
removed: the code now does a single whole-note budget check first and only
builds extraction groups when the note actually exceeds the grounded prompt
budget. However, this is still real provider token counting rather than a
heuristic estimator with near-boundary fallback, so there is still speed left
to recover here.

### Items from the original top-five that are already present

These should no longer be treated as separate first-wave PRs:

- **Phase-2A / Phase-2B phrase dedup** is already substantially present via
  `unique_actionable` and `unique_unresolved` in `phentrieve/llm/pipeline.py`.
- **Phase-1 extraction dedup and merge** is already present via
  `_deduplicate_phase1_extractions()`.
- **Deterministic local matching** is already implemented in the pipeline.
- **Hallucination floor by candidate-ID filtering** is already implemented.

The correct next move is to optimize and calibrate these mechanisms, not to
rebuild them from scratch.

### Concrete merge gates for `feat/llm-full-text-lean-v1`

Do not merge this branch to `main` until all of the following are true:

1. Grounded CLI path regains or exceeds the current grounded fixed baseline on
   English:
   - micro F1 >= 0.7682
   - no benchmark failures
   - **Current status:** met (`0.7764`, no failures)
2. Wall-clock materially improves from the current shared-chunk regression:
   - target <= 800 s on the 10-doc GeneReviews benchmark
   - stretch target <= 744 s to match the best grounded baseline
   - **Current status:** improved but not yet met (`899.95 s`)
3. Total token usage materially improves from the current shared-chunk run:
   - target <= 88,595 total tokens
   - **Current status:** improved but not yet met (`95,396`)
4. API calls return to baseline territory:
   - target <= 49 calls on the 10-doc benchmark
   - **Current status:** improved but not yet met (`55`)
5. German path has a defended non-regression story:
   - either a benchmark set or a documented holdout evaluation with explicit
     limitations
   - **Current status:** not yet met
6. Repo verification stays green on every landing step:
   - `make check`
   - `make typecheck-fast`
   - `make test`
   - **Current status:** met for the latest landing, with the usual `dmypy`
     crash/restart caveat in `make typecheck-fast`

### Recommended execution order on this branch

Land the next optimization sequence on `feat/llm-full-text-lean-v1` in this
order:

1. Mapping prompt restructure with retrieval score and KEEP-default framing
2. Adaptive grounded context
3. Token preflight heuristic
4. Per-language calibration harness for routing thresholds
5. German non-regression evaluation set / documented holdout

This is the best order if the objective is a mature CLI path with better speed,
lower cost, and no quality backslide.

---

## 6. 2026-04-17 follow-up: shared mapping prompt landed, but cost regressed

### What was tested

The branch now includes a shared English mapping prompt for all languages with
an explicit `{language}` instruction and `retrieval_score` in the compact
mapping payload.

Benchmark artifact:

- `results/llm/pr216_shared_mapping_prompt.json`

### Measured outcome vs the last fixed grounded baseline

Compared against `pr216_grounded_routing_phase1_parallel_fix4.json`:

- **Quality improved**: micro-F1 `0.7764 -> 0.7846` (`+0.0083`)
- **Wall-clock regressed slightly**: `899.95 s -> 912.65 s`
- **API calls were flat**: `55 -> 55`
- **Total tokens regressed heavily**: `95,396 -> 141,582` (`+46,186`)

Compared against the older chunkrefactor baseline:

- micro-F1 `0.7631 -> 0.7846`
- wall-clock `1051.52 s -> 912.65 s`
- API calls `60 -> 55`

So PR C in its current form is a **quality win but a cost regression**.

### Root-cause diagnosis from local traces

The regression is dominated by **prompt-token growth**, not by more
completions:

- `GeneReviews_NBK321516`: prompt tokens `11,564 -> 20,659`, completions
  `3,473 -> 3,141`
- `GeneReviews_NBK532447`: prompt tokens `10,468 -> 17,148`, completions
  `2,959 -> 3,085`
- `GeneReviews_NBK169825`: prompt tokens `8,982 -> 15,133`, completions
  `2,540 -> 2,540`

The mapping templates grew by **547 characters each** in the landed change,
and those extra instructions/examples are paid repeatedly across the many
Phase-2B mapping calls in a benchmark run. That matches the per-document token
delta: roughly **500-1000 extra total tokens per API call**, with API-call
count mostly unchanged.

This means the shared prompt itself is not inherently the problem; the issue is
that the current version adds too much repeated prefix material per call.

### External guidance that changes the recommendation

Official Gemini docs now point to a sharper optimization strategy:

- Google documents that Gemini 2.5 Flash has **dynamic thinking enabled by
  default** and that `thinkingBudget = 0` is supported on 2.5 Flash if you want
  to disable it. That makes thinking a valid optimization lever, but not a
  blanket recommendation.
  Source: <https://ai.google.dev/gemini-api/docs/thinking>
- Google also documents that Gemini API pricing bills **output price including
  thinking tokens**, so thought-token expansion is a real cost driver, not just
  an observability detail.
  Source: <https://ai.google.dev/gemini-api/docs/pricing>
- Google recommends using **`usage_metadata` from `generateContent`** for actual
  token accounting, while `countTokens` is for input sizing and planning. That
  means cost/debugging should be driven by post-response usage, not by preflight
  token counts alone.
  Source: <https://ai.google.dev/api/tokens>
- The structured-output guide emphasizes **clear schema descriptions, strong
  typing, prompt clarity, and app-side validation**, which supports moving some
  repeated prompt guidance into schema descriptions instead of paying for it in
  every Phase-2B prompt prefix.
  Source: <https://ai.google.dev/gemini-api/docs/structured-output>
- The caching guide says Gemini 2.5 models have **implicit caching** by default;
  cache hits are helped by putting large common content at the beginning of the
  prompt and reusing similar prefixes close together. Cache-hit counts are
  surfaced in `usage_metadata`.
  Source: <https://ai.google.dev/gemini-api/docs/caching/>
- The Vertex/Google Cloud inference reference documents `seed` as **best effort
  only** and explicitly says deterministic output is not guaranteed even with a
  fixed seed.
  Source: <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference>

### New live evidence from this branch

Follow-up instrumentation and real Gemini runs materially changed the picture:

- A seeded, costed single-document benchmark (`GeneReviews_NBK321516`,
  `gemini-2.5-flash`, `seed=123`) reported:
  - `prompt_tokens = 19,142`
  - `completion_tokens = 3,188`
  - `thoughts_tokens = 20,324`
  - `cached_content_tokens = 0`
  - `total_tokens = 42,654`
  - estimated cost `= $0.0645`
- On that sample, **thought tokens exceeded visible completion tokens**, so
  output-side reasoning is currently the dominant cost bucket.
- The same sample also showed that **same-seed runs still varied** in tokens and
  F1, matching Google's "best effort" wording for reproducibility.
- Most importantly, the live experiment that paired prompt compression with
  `thinkingBudget=0` caused a Phase-2 structured call to fail with
  `finish_reason=RECITATION`, followed by
  `RuntimeError: Gemini returned no structured response payload`. Prompt
  compression alone completed; the instability appeared only once thinking was
  forcibly disabled.

This changes the recommendation materially: the docs make `thinkingBudget=0`
available, but our workflow evidence says **blanket thinking-off is not safe for
structured clinical mapping on this branch today**.

### Revised recommendation

Do **not** revert the shared-language prompt idea. Keep the quality gain, but
optimize with a measurement-first policy:

1. **Trim the repeated mapping prefix aggressively**
   - Collapse the language contract to one short sentence.
   - Keep `retrieval_score` in the runtime payload, but remove it from verbose
     few-shot examples if it is not needed for the example to teach the task.
   - Reduce the number and verbosity of few-shot examples; keep only the
     highest-signal example per prompt if quality holds.
2. **Move guidance from prose into schema where possible**
   - Use response-schema descriptions to carry some behavioral guidance instead
     of repeating it in the prompt text.
3. **Instrument thought-token usage per phase before any new thinking change**
   - Persist `thoughts_tokens` and `cached_content_tokens` per request and per
     benchmark artifact, then attribute them to Phase 1 vs Phase 2B.
   - The next decision is not "should we disable thinking globally?" but
     "which exact call types actually earn their thought tokens?"
4. **Treat thinking control as a targeted experiment, not a default**
   - Only re-test `thinkingBudget=0` or reduced budgets on narrowly scoped
     calls, with live structured-output verification and RECITATION/no-payload
     monitoring.
   - The likely safe candidates are simpler extraction or high-confidence
     mapping paths, not the full low-confidence grounded Phase-2B path.
5. **Instrument and then exploit cache effectiveness**
   - Surface cache-hit metadata from `usage_metadata` if the SDK exposes it.
   - For Gemini 2.5 Flash, implicit caching only becomes relevant once requests
     have at least a 1,024-token shared prefix; keep stable common prefixes at
     the beginning of prompts and send similar requests close together.
   - If Phase-2B prefixes consistently clear that threshold and are reused
     heavily, explicit caching becomes worth evaluating.
6. **Keep adaptive grounded context next**
   - Primary-only context remains the next best branch-local optimization once
     repeated prefix cost is reduced.

### Updated next-step order

Given the new evidence, the recommended order on this branch is now:

1. Shared mapping prompt compression, schema tightening, and phase-level token
   attribution
2. Adaptive grounded context
3. Selective thinking-budget experiments on specific call types only
4. Cache-hit optimization / possible explicit caching for repeated prefixes
5. Token preflight heuristic
6. Per-language calibration harness for routing thresholds
7. German non-regression evaluation set / documented holdout
