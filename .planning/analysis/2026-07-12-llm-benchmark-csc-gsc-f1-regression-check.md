# LLM Benchmark CSC/GSC F1 Regression Check

Date: 2026-07-12

## Trigger

After running `phentrieve benchmark llm` end-to-end against the RAG-HPO `CSC`
and `GSC` datasets for two models (Llama 3.1 70B and Gemini 3.1 Flash Lite via
OpenRouter), the measured micro F1 scores were:

| Model | Dataset | Documents | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Llama 3.1 70B | CSC | 112/112 | 0.644 | 0.619 | 0.631 |
| Llama 3.1 70B | GSC | 114/114 | 0.767 | 0.580 | 0.660 |
| Gemini 3.1 Flash Lite | CSC | 112/112 | 0.742 | 0.687 | 0.713 |
| Gemini 3.1 Flash Lite | GSC | 114/114 | 0.810 | 0.698 | 0.750 |

The user recalled prior runs having F1 around 0.8 and asked whether these
numbers represent a regression. This document records the investigation into
that question: what prior numbers actually exist in this repository's git
history and planning tree, what changed in the scoring pipeline since then,
and whether CSC/GSC were ever benchmarked against an LLM for real before this
run.

## Method

1. Searched `.planning/` (specs, completed, archived, analysis, active,
   drafts) for any previously recorded LLM benchmark F1 numbers on CSC or GSC.
2. Distinguished, for every "F1 0.8"-shaped number found, whether it was a
   number this codebase actually measured, or a number quoted from the
   RAG-HPO paper itself as an external target/comparator.
3. Reviewed git history for commits touching HPO extraction scoring
   semantics, assertion/negation handling, present-only vs. strict scoring
   mode, and the LLM benchmark pipeline itself
   (`phentrieve/benchmark/llm_benchmark.py`,
   `phentrieve/evaluation/extraction_metrics.py`,
   `phentrieve/benchmark/data_loader.py`), to see whether any of them made
   CSC/GSC scoring mechanically stricter after any historical baseline was
   measured.
4. Checked whether `phentrieve benchmark llm --dataset CSC/GSC` was ever
   actually run for real (not just planned/scripted ad hoc) before this pass.

## Finding 1: The "F1 0.80" figure is the RAG-HPO paper's own number, not ours

Two independent citations in the planning tree quote 0.80 as the RAG-HPO
paper's reported number, not a number this codebase produced:

- `.planning/archived/2026-04-16-llm-lean-v1-comparative-review.md:226-229`:

  > "The two-phase architecture is correct. RAG-HPO reports Llama-3 70B
  > collapses from F1 0.80 with retrieval to F1 0.12 without on the same
  > 112-case set. The retrieval step is load-bearing, not decorative (Genome
  > Medicine 2025)."

  Note "112-case set" — that is CSC's document count, confirming this is the
  paper's own CSC-shaped evaluation, not something this repository ran. The
  0.80 figure is *with* retrieval (their full RAG pipeline, analogous to
  ours); 0.12 is their retrieval-ablated control.

- Same file, line 427:

  > "RAG-HPO is the closest external comparator; its F1 0.80 on real case
  > reports is the bar we should measure against, on English first."

  This explicitly frames 0.80 as an external target to measure *against*,
  not a number we had already reached.

- `.planning/archived/pre-convention/HPO-EXTRACTION-BENCHMARKING-FRAMEWORK.md:1011-1014`,
  under a "References" section listing external literature:

  > "RAG-HPO (2024) — medRxiv preprint on LLM-based phenotyping — 120 case
  > report benchmark, F1=0.80"

  (This file also separately cites FastHPOCR's own paper reporting "F1 0.85
  on GSC+" at lines 233-234/279 of the comparative review — GSC+ is
  PhenoBERT's corpus, a different dataset from RAG-HPO's "GSC" that we
  benchmarked. Easy to conflate by name alone; confirmed distinct datasets.)

Our own internal ~0.77-0.85 F1 numbers appearing elsewhere in that same
comparative-review document (lines 679, 735, 742, 896, 901) are all on a
10-document **GeneReviews** benchmark — a small, separate, curated corpus,
not CSC or GSC.

## Finding 2: The one real historical CSC baseline is 0.724, not 0.8 — and matches today closely

`.planning/analysis/2026-05-25-llm-evidence-validation-enriched-mapping-pr-regression.md`
(added 2026-06-13 in commit `4a73e9b`, "docs(planning): record csc68
benchmark watch item") records a full-CSC run on `main` with
`gemini-3.1-flash-lite`, under "Current Main Baseline And CSC_68 Watch Item"
(lines 87-90):

| Dataset | Cases | Model | Strict micro P | Strict micro R | Strict micro F1 | Status |
|---|---:|---|---:|---:|---:|---|
| GeneReviews | 10 | `gemini-3.1-flash-lite` | 0.840 | 0.797 | 0.818 | matches the local stable run |
| CSC | 112 | `gemini-3.1-flash-lite` | 0.747 | 0.702 | **0.724** | slightly above the older local CSC run |

This is the only real, measured, full-CSC LLM benchmark number recorded
anywhere in this repository's history. Comparing directly to today's run of
the same model on the same dataset:

| Run | Precision | Recall | F1 |
|---|---:|---:|---:|
| 2026-05-25 baseline (`gemini-3.1-flash-lite`, CSC, 112 docs) | 0.747 | 0.702 | 0.724 |
| 2026-07-11 today (`gemini-3.1-flash-lite`, CSC, 112 docs) | 0.742 | 0.687 | 0.713 |
| Delta | -0.005 | -0.015 | **-0.011** |

An F1 delta of ~0.01 is well within normal run-to-run variance for a
non-deterministic LLM pipeline (temperature 0 does not guarantee
bit-for-bit-identical outputs across provider-side model updates, retrieval
index state, or OpenRouter routing), and is far short of what a genuine
scoring or pipeline regression would produce (contrast with the real PR #261
regression documented in the same file, which dropped CSC F1 from 0.736 to
0.597 — a 0.14 drop, over 10x larger).

No real measured number for GSC, and no real measured number for
Llama-3.1-70B on either dataset, exists anywhere in `.planning/` or git
history. There is nothing for today's Llama 3.1 70B / GSC numbers to
regress *from* — these are first-ever measurements, not re-measurements.

## Finding 3: The CSC/GSC scoring pipeline has not changed since the 0.724 baseline

- The dataset-specific gold/prediction assertion projection that scopes
  CSC/GSC scoring to present-only findings
  (`DATASET_ASSERTION_PROJECTION` in `phentrieve/benchmark/llm_benchmark.py`)
  was introduced in commit `e3713e4` ("fix: address review feedback on
  sidecar benchmark PR", 2026-04-20) and has not changed since. It was
  already active when the 2026-05-25 baseline of 0.724 was measured, so it
  is not a new source of stricter scoring for today's run.
- The "extraction contract v2" effort added `normalize_for_scoring(mode=...)`
  with a `present-only` mode alongside the existing default (`strict`, commit
  `f65af2c`, 2026-07-03, "feat(benchmark): add normalize_for_scoring
  present-only projection helper"). The default `strict` mode is defined as
  an identity transform (returns input unchanged), so this addition did not
  alter default scoring behavior for either the 2026-05-25 baseline run or
  today's runs, both of which use the default mode.
- No other commit touching `phentrieve/benchmark/llm_benchmark.py`,
  `phentrieve/evaluation/extraction_metrics.py`, or
  `phentrieve/benchmark/data_loader.py` changes what counts as a
  gold/predicted match for CSC or GSC between 2026-05-25 and today.

## Finding 4: CSC/GSC support in `phentrieve benchmark llm` itself is brand new

Commits `568d247` ("fix: label RAG-HPO benchmark datasets correctly") and
`b232911` ("docs: document llm benchmark result-store layout and CSC/GSC
support") — both dated 2026-07-10, the day before this run — finalized the
CLI's dataset naming and documentation for CSC/GSC in the `benchmark llm`
path. Prior CSC evidence (the 2026-05-25 baseline and the PR #261 A/B) came
from focused ad hoc script-based comparisons against the same underlying
two-phase pipeline, not from long-standing `phentrieve benchmark llm
--dataset CSC` CLI usage. This is the first time the full CSC and GSC
datasets were run end-to-end through the now-unified `benchmark llm` CLI
path and result-store structure (documented in
`docs/user-guide/benchmarking-guide.md`).

## Conclusion

No evidence of a regression. The one real prior measured baseline (Gemini
3.1 Flash Lite on CSC: F1 0.724, 2026-05-25) is closely matched by today's
result on the same model and dataset (F1 0.713, delta -0.011), and the
scoring pipeline has not changed in the interim in any way that would affect
CSC/GSC. The "~0.8" figure the user recalled traces to the RAG-HPO paper's
own reported Llama-3 70B number (quoted repeatedly in this repo's planning
docs as an external target to measure against), not a number this codebase
ever produced. GSC and Llama-3.1-70B-on-CSC/GSC have no prior real
measurement to compare against at all — today's runs are first
measurements, not re-measurements, so "regression" does not apply to them.

## Follow-up note (data quality, not a regression)

The 2026-05-25 baseline analysis separately flags `CSC_68` as a data-quality
watch item: its ten gold HPO labels appear ungrounded in the available
source text (strict F1 0.000 for that single document, TP 0 / FP 7 / FN 10),
distinct from four other CSC cases (`CSC_22`, `CSC_25`, `CSC_44`, `CSC_89`)
excluded during dataset conversion for having no released annotations at
all. `CSC_68` remains in the benchmark by decision; it depresses the
micro-average CSC F1 slightly for every model run against the full 112-case
set, including both of today's runs. This is unrelated to whether a
regression occurred, but is relevant context for interpreting the absolute
CSC F1 level (~0.63-0.75) against the paper's 0.80.
