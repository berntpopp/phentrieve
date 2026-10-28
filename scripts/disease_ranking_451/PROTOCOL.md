# Protocol: does extracted HPO rank the right disease? (issue #451)

Registered 2026-10-28, before any arm was run on the test split, on the dev split, or on any
generated letter. At registration time only these steps had run:
- the scorer port and its conformance test against VarLens;
- the case selection (`cases.jsonl`, `results/attrition.csv`).
No disease rank of any arm had been computed.

This file is not edited after registration. Results are appended under "Results", and
deviations go in a dated "Deviations" section.

## Question and hypotheses

Do the HPO terms that Phentrieve extracts from clinical letters rank the true rare disease about
as well as the curated (gold) HPO terms of the same patient?

- **H1 (primary, decides go/no-go).** On German letters of non-cited test cases, the best
  Phentrieve arm reaches disease Hit@10 >= 0.8 x gold Hit@10.
- **H2 (soft profiles).** On the same set, Hit@10 of soft-k minus top-1 is >= +2.0 pp, and the
  lower bound of its 95% CI is > 0.
- **H3 (descriptive).** The gap to gold is reported separately for German and English. No claim
  is made that one language is better than the other.

## Decision rules

- **Go** for `phentrieve match` if H1 holds on the point estimate. The CI of the ratio is
  reported next to it.
- **Keep soft profiles** if H2 holds. Otherwise the default output is used.
- Otherwise: **no-go or rework**, with the reason, read off the diagnostics below.

## Pinned inputs

All SHA256 hashes are in `manifest.json`.

- **HPO v2026-09-01:** `hp.json`, `phenotype.hpoa`, `genes_to_phenotype.txt`.
- **Phentrieve data bundle `hpo-v2026-09-01-r1`:** from `berntpopp/phentrieve-data`, file
  `phentrieve-data-v2026-09-01-biolord-multivec.tar.gz`. Model `FremyCompany/BioLORD-2023-M`,
  multi-vector index.
- **phenopacket-store 0.1.27:** `all_phenopackets.zip`.
- **German HPO labels and synonyms:** `obophenotype/hpo-translations` at commit `735ab35`
  (`babelon/hp-de.babelon.tsv`, `hp-de.synonyms.tsv`). Used only for the leakage check.
- **Phentrieve code:** the commit that adds this protocol. Library code is unchanged from
  `main` @ `02f8fea`.
- **Letter generator:** Gemini `gemini-3.7-flash` (version `3.7-flash-08-2026`), temperature 0,
  seed = 451 + attempt. It is called through Phentrieve's `GeminiStructuredOutputProvider`.
  The full prompt is in `generate.py`.

## Scorer (frozen)

The scorer is a port of VarLens phenotype scoring v2 at its shipped setting
(`VarLens/.planning/specs/2026-10-14-hpo-phenotype-scoring-v2.md`, §3 and §7):

- tau = 1;
- background correction off;
- exclusions `conflictsOnly`, so excluded terms never change the score;
- alpha_f 0.25, f0 1.0, gamma 0.55.

IC and frequencies are built exactly as VarLens's DB builder (builder v6) builds them.
`tests/unit/scripts/test_disease_ranking_scorer.py` checks the port against VarLens's own
output on 30 phenopacket profiles, 5 of them with key weights. Scores match to 1e-5 (observed
maximum deviation 2.2e-16) and ranks match exactly.

The ranking universe is the 9,162 gene-linked diseases of the index. A gene's score is the
maximum over its diseases.

## Cases

Source: phenopacket-store 0.1.27. A case is eligible when:

- it has exactly one diagnosed gene and at least one observed HPO term;
- one of its diagnoses is a gene-linked disease of the index;
- it has a source PMID.

A case is **cited** when `phenotype.hpoa` cites one of its PMIDs for one of its diagnoses
(VarLens rule). All other cases are **non-cited**.

Sample (seed 451):

1. Shuffle the eligible cases.
2. Take non-cited cases first, up to 400, then fill with cited cases to 600 in total. A gene
   contributes at most 3 cases overall.

Split: 20% of the sampled genes form the **dev** split (seed 452); the rest form the **test**
split. Result: 600 cases, 332 genes and 427 PMIDs. Dev has 60 non-cited and 51 cited cases;
test has 289 non-cited and 200 cited. Attrition is in `results/attrition.csv`.

## Letters

For each case, one German and one English referral letter (doctor to doctor, about 150-300
words). The generator receives sex, age at last encounter, disease onset, and the labels of
the observed and excluded terms. It is instructed to:

- paraphrase findings as clinicians write them, never copying an HPO label verbatim;
- write excluded terms as natural negations;
- add distractors (normal findings, family history, medication, social details);
- never name a diagnosis, gene or disease.

**Regeneration.** A letter is regenerated (seed + 1, at most 3 attempts) when it contains any
of:

- the gene symbol (case-sensitive, whole word);
- the disease ID;
- the disease label or the part of the label before the first comma or semicolon
  (case-insensitive).

Letters that still fail after 3 attempts are kept, flagged and counted.

**Leakage check.** For each letter we compute the share of gold observed terms (phenotype
subtree) whose label or any synonym appears verbatim (case-insensitive, whole words):

- English letters: the English label and synonyms from `hp.json`;
- German letters: the German label and synonyms, plus the English label and synonyms.

We report the distribution per language. The authors read 10 letters per language, and 2 are
quoted in the report.

## Arms (paired, same cases)

1. **Gold:** the observed phenopacket terms, weight 1.
2. **Top-1:** the default `phentrieve text process` output:
   - standard backend, BioLORD multi-vector;
   - chunking `sliding_window_punct_conj_cleaned`, chunk threshold 0.7, 10 results per chunk,
     minimum aggregated confidence 0.75;
   - `dependency` assertion, no adaptive rechunking.

   Terms whose aggregated status is `affirmed` enter with weight 1. The issue calls this arm
   "top-1". It is Phentrieve's default hard output, which can hold more than one term per
   chunk.
3. **Soft-k:** from the same run, the per-chunk candidates (`processed_chunks[].hpo_matches`,
   already >= 0.7). Only matches whose per-match assertion is `affirmed` are kept, and the
   top k by score are taken in each chunk. Each chunk has unit weight, shared among its
   candidates by either:
   - `raw`: s_i / sum_j s_j, or
   - `softmax(T)`: exp(s_i / T) / sum_j exp(s_j / T).

   A term found in several chunks takes its maximum weight. These weights replace w_p in the
   scorer's forward score; nothing else changes.

The run language is set explicitly (`de` or `en`). A case whose arm yields no usable term gets
the tie rank (N + 1) / 2.

**Tuning (dev only).** The grid is k in {1, 3, 5} x weighting in {raw, softmax T = 0.05, 0.1,
0.2}, 12 settings in total.

- Pick the setting with the highest disease Hit@10 on all dev cases, German and English letters
  pooled.
- Ties are broken by the higher MRR, then the smaller k, then the order raw < 0.05 < 0.1 < 0.2.
- The chosen setting is frozen and written to `results/frozen.json` before the test split is
  scored, once.

## Metrics and statistics

**Disease level:** Hit@1, Hit@3, Hit@10, median rank and MRR. Ties take the average rank.
**Gene level:** Hit@10, secondary.

All metrics are reported for:

- each language and each arm;
- the subsets non-cited test (primary), cited test and all test.

**Intervals.** Paired bootstrap with 2,000 replicates, clustered by the first PMID, seed 451,
95% percentile intervals. Intervals are given for each metric, for arm differences, and for
the ratio best arm / gold.

**Tests.** Exact McNemar (two-sided binomial on the discordant pairs) for paired Hit@k
differences, for top-1 vs gold, soft-k vs gold and soft-k vs top-1.

**Best Phentrieve arm.** Whichever of top-1 and soft-k has the higher Hit@10 on the primary
set.

**Diagnostics** (they explain the gap; they do not decide anything):

- Term precision and recall of each Phentrieve arm against gold. For soft-k, the candidates
  with weight > 0 count.
  - Exact: canonical IDs equal.
  - Ancestor credit: one term is an ancestor-or-self of the other within the phenotype
    subtree, with HP:0000118 excluded.
- Hit@10 by tertile of the verbatim-label rate, per language and arm.
- Runtime per case, for the Phentrieve call and for scoring.
- **Sanity check.** Gold profiles from the RareBench cohorts (MME, LIRICAL, HMS, RAMEDIS) are
  run through the scorer to confirm that it gives plausible ranks. This is not an endpoint.

## Limitations

The letters are synthetic. They give an upper bound for real letters and do not replace them.
The generator may also have seen the case reports during training (contamination).

## Reproduction

`scripts/disease_ranking_451/run.sh` reruns everything from the pinned inputs. Letters are
cached in `letters/`, so a rerun does not call the LLM unless a letter is missing.

## Results

(appended after the test run)
