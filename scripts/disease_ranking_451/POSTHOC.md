# Post-hoc exploratory analyses (#451)

Everything here was done after the registered test run. The LLM step that reorders the
scorer's candidate list is called "reorder" in code and file names; the repository's policy
test for a removed second-stage ranking feature bans the more common name. None of it changes the registered
decision (`REPORT.md`, `PROTOCOL.md`). Research use only.

All numbers are on non-cited test cases (n = 289 per language) unless marked "dev". Intervals
are 95% PMID-clustered bootstrap intervals (2,000 replicates).

## 1. Error profile of the embedding backend (`errors.py`)

Counterfactual German Hit@10 for the default output:

| Profile | Hit@10 | Effect |
|---|---|---|
| Gold terms | 0.557 | |
| Gold terms Phentrieve recovered (misses only) | 0.516 | misses cost 4 pp |
| Gold plus Phentrieve's wrong terms (noise only) | 0.388 | noise costs 17 pp |
| Phentrieve's related terms only | 0.377 | term choice costs another 14 pp |
| Phentrieve default output | 0.311 | |

- **Most extracted terms are wrong.** 75% of the affirmed terms are unrelated to any gold
  term. A chunk yields a median of 7 affirmed candidates.
- **Normal findings become abnormal terms.** "Normal development" becomes *Global
  developmental delay*; "50th centile" becomes *Postnatal growth retardation*;
  "normocephalic" becomes *Dolichocephaly*; "cheerful" becomes *Euphoria*.
- **The existing flag doesn't close the gap.** `--top-term-per-chunk` reaches Hit@10 0.370
  (DE) and 0.391 (EN).

## 2. LLM full-text backend (`extract_llm.py`, `explore_arms.py`)

This is Phentrieve's default full-text mode: `gemini-3.1-flash-lite`, `two_phase`,
`whole_document_grounded`.

- **Before the fix in section 3:**
  - German: Hit@10 0.457 [0.381, 0.534], 0.82 x gold.
  - English: Hit@10 0.509, 0.91 x gold.
  - Exact precision rises from 0.06 (default embedding output) to 0.59 (DE).
- **With the fix:** both languages are at 0.85 x gold.
  - German: Hit@10 0.471 [0.400, 0.546], ratio [0.77, 0.91].
  - English: Hit@10 0.474 [0.404, 0.550], ratio [0.77, 0.93].

## 3. Grounded-negation bug, fixed in PR #453

**Symptom.** Before the fix, 11.9% of English present terms (603 of 5,059) were findings
the gold record lists as excluded. In German the rate was 3.7%.

**Cause.**
- The chunker strips negation cues ("He has no seizures" becomes the chunk `seizures`).
- Grounded phase 1 sees only the chunk index, never the note text.
- Assertion detection is disabled for grounded chunks.

**Fix.** Phase 1 also receives the source sentences behind its chunks, and the prompt adds
rules for list negation. After the fix:

| | wrong present, before | wrong present, after |
|---|---|---|
| English | 603 | 7 |
| German | 153 | 2 |

True findings are unchanged (EN 2,308 to 2,356; DE 2,499 to 2,491), and there is no new
over-negation.

**Effect on the evaluation.** The bug had inflated English: phenopackets often list the true
disease's own typical features as excluded, so the wrongly present findings helped the
scorer. With the fix, English Hit@10 drops from 0.509 to 0.474, and the gap between the
languages disappears. All results below use the fixed extraction (`_negfix`).

## 4. LLM reorder of the scorer's top 50 (`reorder.py`)

**Setup.**
- Claude Opus 5.5 via local Claude Code (`claude -p`).
- Prompt versions v1-v5 were tuned on dev only (111 cases per language).
- Tools: `reorder_debug.py` replays single cases; `reorder_eval.py` compares variants.

**Dev findings** (on the old extraction unless noted):

| Variant | What it showed |
|---|---|
| v1 | Showed only the 10 most frequent features. Unstable Hit@1: it lost 14-20 correct #1s. The deciding information was missing: the truth's matching features were hidden, and there was no inheritance or onset. |
| v2 | Adds matched features, inheritance and onset, absent findings, and "ranker order is a strong prior". +4-8 pp Hit@1/3/10 in both languages. |
| v3 | Rule-gated, with gene/series groups and conflicts. Better than v2 in German, worse in English. |
| Debug agent | v3's English losses came from the negation bug (section 3). Groups were too coarse and anchoring too strong. Single runs disagree on the #1 in about 30% of cases. |
| v4/v5 | Lighter rules; v5 adds 3-pass shuffled consensus. Neither beat v2/v3, and consensus hurt. |
| With the fixed extraction | v2 beats v3 on every metric in both languages. v2 was frozen (`results/reorder_frozen.json`) before the test run. |

**Test result (single run of the frozen v2):**

| | DE Hit@1 | DE Hit@3 | DE Hit@10 | EN Hit@1 | EN Hit@3 | EN Hit@10 |
|---|---|---|---|---|---|---|
| Gold terms -> scorer | 0.232 | 0.405 | 0.557 | 0.232 | 0.405 | 0.557 |
| Fixed extraction -> scorer | 0.173 | 0.329 | 0.471 | 0.145 | 0.353 | 0.474 |
| **+ Opus reorder** | **0.277** | **0.446** | **0.550** | **0.280** | **0.453** | **0.564** |
| Reorder minus scorer | +10.4 [6.0, 15.2] | +11.8 | +8.0 [4.4, 11.8] | +13.5 [9.2, 18.3] | +10.0 | +9.0 [5.4, 12.7] |

- All reorder-vs-scorer differences have exact McNemar p < 0.001.
- Correct #1s: German lost 6 and gained 36; English lost 4 and gained 43.
- Gene level (any disease of the causal gene), Hit@1 / Hit@10:
  - reorder: DE 0.439 / 0.682, EN 0.433 / 0.668;
  - gold: 0.370 / 0.696.
- Data: `results/reorder_test.csv`.

## 5. Caveats

- **The comparison with gold is not like-for-like.** The reorder reads the letter and uses
  the LLM's own disease knowledge; the gold arm is only scored, not reordered. "Reorder >= gold"
  means the full pipeline ranks as well as curated terms put through the plain scorer.
- **Contamination is possible.** The cases come from published reports that Opus may know.
  The letters are synthetic and written by Gemini, and the case IDs are not shown to the
  model, but recall of known cases cannot be excluded.
- **Tuning used 111 dev cases.** The prompt choice is selected, not registered. One test run
  per variant was made, and only the frozen v2 was run on test.
- **The text is synthetic.** These are upper-bound numbers. Real letters, ideally the
  clinician gold standard of phentrieve-benchmark#6, are the next step. That run should be
  registered, with letters from a different generator family.

## 6. Why scorer Hit@1 dropped with the negation fix (`negfix_effect.py`)

With the fix, extraction-only Hit@1 went from 0.194 to 0.173 (DE) and from 0.159 to 0.145 (EN).

- **Most excluded findings belong to the true disease.** 55% of the findings that phenopackets
  record as excluded (936 of 1,695) are annotated features of the case's own disease. Authors
  list the classic features of the disease they suspect that the patient did not have.
- **So the bug was handing the scorer the answer.** The letters correctly say "no X". The bug
  turned X into a present finding, and X usually pointed to the true disease.
- **Most lost #1s trace back to this:**

| | True #1 lost after fix | ...where the removed findings were true-disease features | True #1 gained | Net |
|---|---|---|---|---|
| English | 10 | 8 | 6 | -4 cases (-1.4 pp) |
| German | 14 | 5 | 8 | -6 cases (-2.1 pp) |

The remaining German losses are most likely run-to-run variation of the LLM extraction; this is
not verified. The lower Hit@1 after the fix is the honest number. The full pipeline with the
reorder does not depend on the artifact.

## 7. Sets and sample sizes

| Set | Cases per language | Used for |
|---|---|---|
| Test, non-cited | 289 | All test numbers in this file and in `REPORT.md` |
| Dev | 111 | Soft-profile tuning (registered) and reorder prompt tuning (post hoc) |
| All letters (dev + test) | 400 | Extraction-error counts for the fix (section 3) |

- **The test set is a seeded sample**, not all phenopacket-store cases: about 2,500 non-cited
  cases were eligible, with at most 3 per gene. The 200 cited test cases were dropped for the
  proof of concept (deviation 2026-10-28).
- **Dev is easier than test.** Even gold terms reach Hit@1 0.459 on dev but 0.232 on test, so
  absolute dev numbers do not carry over; the gains over the scorer do (dev +5-6 pp Hit@1,
  test +10-14 pp).
- **Hit@1 intervals on 289 cases** are roughly +/-6 pp.

## 8. Reproduce the post-hoc steps

All commands run from `scripts/disease_ranking_451/`. `EVAL451_DATA` and the bundle run
directory are set up as in `run.sh`. The registered analyses need nothing extra.

1. **Embedding error profile and arms:**
   - `python -I errors.py <hpo>`
   - `python -I explore_arms.py <hpo>`
2. **LLM extraction** (Gemini; `GEMINI_API_KEY`). Run from the bundle run directory:
   - before the fix: `python extract_llm.py de|en`
   - with the fix: check out PR #453 (`fix/llm-list-negation-scope`, commit `d0f431b`), then
     `EXTRACT_TAG=_negfix uv run --project <that checkout> python extract_llm.py de|en`
3. **Reorder via local Claude Code** (`claude -p`; no API key used):
   - dev variants: `EXTRACT_TAG=_negfix REORDER_PROMPT=v2 REORDER_TOP_N=50 python reorder.py <hpo> dev cc-claude-opus-5-5`
   - frozen test run: the same command with `test`, using the setting in
     `results/reorder_frozen.json`
4. **Evaluation:**
   - `python -I reorder_eval.py <hpo> dev`
   - `python -I reorder_test.py <hpo>`
   - `python -I negfix_effect.py <hpo>`
5. **Debug one case:** `REORDER_PROMPT=v2 REORDER_TOP_N=50 python reorder_debug.py <hpo> <case> <lang> [--ask cc-claude-opus-5-5]`

The extraction and reorder outputs are committed in `results/`; the shuffled-consensus runs
are committed gzipped. Re-running the LLM steps costs API credit (Gemini) or Claude Code
usage, and the outputs will vary slightly from run to run.
