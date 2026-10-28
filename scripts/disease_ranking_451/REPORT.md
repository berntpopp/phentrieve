# Does extracted HPO rank the right disease? Proof of concept (#451)

**Registered verdict: no-go, but it covers only Phentrieve's embedding backend.** The
default `text process` backend reached 0.65 x gold Hit@10 against the 0.8 bar.

**Exploratory follow-up (post hoc, RUO; details in `POSTHOC.md`).** The full pipeline below
matches the curated gold terms on non-cited test cases (n = 289 per language):

1. Phentrieve LLM full-text extraction, with the grounded-negation fix of PR #453;
2. the VarLens scorer;
3. a Claude Opus 5.5 reorder of the scorer's top 50 (prompt frozen on dev).

| Non-cited test | DE Hit@1 | DE Hit@10 | EN Hit@1 | EN Hit@10 |
|---|---|---|---|---|
| Gold terms -> scorer | 0.232 | 0.557 | 0.232 | 0.557 |
| LLM extraction (fixed) -> scorer | 0.173 | 0.471 | 0.145 | 0.474 |
| **+ Opus reorder** | **0.277** [0.22, 0.35] | **0.550** [0.48, 0.62] | **0.280** [0.22, 0.35] | **0.564** [0.50, 0.64] |

What these rows compare: the reorder reads the letter and uses the LLM's own disease
knowledge, while the gold arm is not reordered. "Matches gold" therefore means the full
pipeline ranks as well as curated terms put through the plain scorer, not that the
extraction is perfect. The LLM may also have seen the published case reports.

Setup:

- Synthetic referral letters for 400 phenopacket-store cases. The primary set is 289
  non-cited test cases.
- HPO v2026-09-01, and Phentrieve's default pipeline with BioLORD (multi-vector).
- Diseases are ranked by a port of VarLens phenotype scoring v2 at its shipped setting. The
  port is bit-exact against VarLens on 30 reference profiles.
- Registered protocol and deviations: `PROTOCOL.md`.

## Answers

1. **How far does the rank fall?** On German letters, Hit@10 drops from 0.557 (gold terms) to
   0.311 with Phentrieve's default output: -24.6 pp [-31.1, -18.3]. The median rank of the
   true disease goes from 6 to 39.
2. **Do soft profiles help?** Tuning on dev chose k = 1, i.e. only the best affirmed candidate
   per chunk instead of every candidate above the thresholds. This gains about 5 pp in both
   languages:
   - German: +4.8 pp [-0.3, +10.3]. The registered rule (CI lower bound > 0) is missed
     narrowly.
   - English: +5.9 pp [+0.4, +11.6].

   Weighting several candidates per span (k = 3 or 5) did not help on dev.
3. **German vs English?** The gap to gold is similar: best arm / gold Hit@10 is 0.65 [0.56,
   0.74] in German and 0.69 [0.59, 0.80] in English. The ranges overlap, so we draw no
   ranking between the languages.

**Go/no-go.** The registered threshold was best arm >= 0.8 x gold Hit@10 (German,
non-cited). The result is 0.65, so this is a **no-go**.

## Main table (non-cited test, n = 289 per language)

Intervals are 95% PMID-clustered bootstrap intervals (2,000 replicates).

| Lang | Arm | Hit@1 | Hit@3 | Hit@10 | MRR | Median rank | Gene Hit@10 | Terms |
|---|---|---|---|---|---|---|---|---|
| de | gold | 0.23 [0.18, 0.29] | 0.41 [0.34, 0.47] | **0.56** [0.49, 0.63] | 0.35 | 6 | 0.64 | 8 |
| de | top-1 (default) | 0.06 [0.03, 0.09] | 0.17 [0.13, 0.23] | 0.31 [0.25, 0.38] | 0.15 | 39 | 0.43 | 90 |
| de | soft, k = 1 | 0.11 [0.07, 0.15] | 0.23 [0.17, 0.28] | **0.36** [0.30, 0.43] | 0.20 | 29 | 0.47 | 23 |
| en | top-1 (default) | 0.06 [0.03, 0.09] | 0.16 [0.12, 0.21] | 0.33 [0.26, 0.39] | 0.14 | 37 | 0.46 | 91 |
| en | soft, k = 1 | 0.12 [0.08, 0.16] | 0.23 [0.17, 0.28] | 0.38 [0.31, 0.46] | 0.20 | 28 | 0.50 | 25 |

The gold arm does not depend on language. Every Phentrieve-vs-gold difference has an exact
McNemar p < 0.001.

## Why the gap: precision, not recall

German test letters, mean per case:

| Arm | Precision exact / with ancestor credit | Recall exact / with ancestor credit |
|---|---|---|
| top-1 | 0.06 / 0.25 | 0.60 / 0.83 |
| soft | 0.16 / 0.30 | 0.34 / 0.65 |

Phentrieve finds most of the gold findings. But it returns about 90 affirmed terms per
letter: several near-synonyms per chunk, plus terms matched from normal findings and
distractors. The scorer's forward score is an IC-weighted average over all patient terms, so
the noise dilutes the true signal. VarLens already saw this weakness of tau = 1 on noisy
profiles (its spec, §7). Keeping one candidate per chunk removes part of the noise and
recovers about 5 pp.

**Rework before rebuilding (cheap: letters and extractions are cached; see the error profile
in `POSTHOC.md`).**

1. Fix precision first. Options:
   - one term per span by default;
   - a higher chunk threshold;
   - drop chunks that assert normality;
   - collapse candidates that are ancestors or siblings of each other.
2. Rerun this harness unchanged on the dev split, then once on test with a newly registered
   threshold.
3. Optionally test the scorer's tau = 0 on noisy profiles. That would change the frozen
   scorer, so it needs its own protocol.

## Data, attrition and leakage

**Attrition:** 10,377 phenopackets → 10,305 single-gene with >= 1 observed term → 10,126
whose diagnosis is a gene-linked disease of the index. Of these, 2,503 are non-cited and
7,623 cited. The seeded sample (at most 3 per gene, 332 genes, 427 PMIDs) has 600 cases.
For the PoC we dropped the 200 cited test cases, leaving 400 (dev 111, test 289 non-cited).
See `results/attrition.csv`.

**Letters:** Gemini `gemini-3.7-flash`, temperature 0, seed 451. Median 222 words. Two
letters were regenerated, and none names the gene, the disease ID or the disease label.

**Leakage** (share of gold labels or synonyms appearing verbatim in the letter):

- German: median 0.06, p75 0.20, p90 0.31; 49% of letters have none.
- English: median 0.10, p75 0.20, p90 0.38; 42% have none.

German labels cover 97% of gold mentions. Hit@10 by leakage tertile shows no strong
inflation in German:

- top-1: 0.30 / 0.30 / 0.33
- soft: 0.34 / 0.35 / 0.39

In English, soft rises from 0.32 to 0.46 across the tertiles, so some English wording echoes
labels.

**Example letters** (excerpts; the full text is in `letters/`):

> *de, myopathy case:* "Die Patientin berichtet über seit der Adoleszenz … bestehende
> rezidivierende Schmerzen im Bereich der Muskulatur … Laborchemisch zeigte sich bei
> mehreren Kontrollen eine deutliche Erhöhung der Serum-Kreatinkinase mit Werten um 600 U/l.
> Körperlich fällt eine ausgeprägte Hohlfußbildung an beiden Füßen auf. Eine Hypertrophie
> der Wadenmuskulatur liegt hingegen nicht vor …"

> *en, neuropathy case:* "manual testing revealed distinct bilateral weakness isolated to
> her long finger extensors … Sensory testing demonstrated blunted light touch appreciation
> over her distal fingers and toes … her deep tendon reflexes were completely normal …
> neurophysiological testing demonstrated marked slowing across both motor and sensory
> conduction velocities"

We read 10 letters per language. Each covered every gold finding in clinical wording, with
natural negations and normal distractors. Unlisted abnormal findings were rare: in the pilot
they led to a tightened prompt, and afterwards they appeared occasionally as a presenting
complaint.

**Scorer sanity check** (RareBench gold profiles, Hit@10): MME 0.85, LIRICAL 0.75, HMS 0.56,
RAMEDIS 0.39.

## Limitations

- **Synthetic text is an upper bound.** Generated letters are cleaner and more complete than
  real ones, so these results do not replace an evaluation on real letters. A clinician
  gold standard (phentrieve-benchmark#6) would close that gap.
- **Possible contamination.** The generator may have seen the published case reports. The
  non-cited restriction only addresses HPOA curation from the same paper.
- **One generator and one embedding model.** Results may differ with other LLMs or
  Phentrieve models.
- **The "top-1" arm is the default output,** which allows several terms per chunk. Soft
  with k = 1 is the literal top-1-per-chunk.
- **Protocol deviations** (all made before any rank was computed): tightened generator
  prompt, sample cut to 400, three truncated letters regenerated.

**Reproduce:** `EVAL451_DATA=<dir> scripts/disease_ranking_451/run.sh`. Inputs are pinned by
SHA256 in `manifest.json`.
