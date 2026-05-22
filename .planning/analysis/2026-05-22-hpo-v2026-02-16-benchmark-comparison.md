# HPO v2026-02-16 Benchmark Comparison

## Build Inputs

- Previous HPO version: v2025-11-24
- New HPO version: v2026-02-16
- Model: FremyCompany/BioLORD-2023-M
- Single-vector bundle: phentrieve-data-v2026-02-16-biolord.tar.gz
- Multivector bundle: phentrieve-data-v2026-02-16-biolord-multivec.tar.gz
- Release artifact directory: dist/hpo-v2026-02-16
- Build mode: RTX-local fallback after completing workflow/runbook fixes
- Bundle count: 19 tar.gz files, covering minimal DB-only, 9 single-vector bundles, and 9 multivector bundles
- New active term count: 19389
- Previous active term count: 19393
- New BioLORD single-vector document count: 19389
- New BioLORD multivector document count: 60503

## Results

| Dataset | Index | Previous | New | Delta | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 570terms_german, MRR@1 | BioLORD multivector | 0.7526 | 0.8386 | +0.0860 | Previous artifact used v2025-11-24 with all_weighted aggregation; new run uses v2026-02-16 with label_synonyms_max. |
| 570terms_german, MRR@3 | BioLORD multivector | 0.8222 | 0.8839 | +0.0617 | Computed from per-case reciprocal ranks with ranks greater than K set to 0. |
| 570terms_german, MRR@5 | BioLORD multivector | 0.8299 | 0.8873 | +0.0574 | Same 570-case German benchmark. |
| 570terms_german, MRR@10 | BioLORD multivector | 0.8346 | 0.8898 | +0.0552 | New multivector MRR overall is 0.8907. |
| 570terms_german, Hit@1 | BioLORD multivector | 0.7526 | 0.8386 | +0.0860 | Top-1 exact hit rate improved. |
| 570terms_german, Hit@3 | BioLORD multivector | 0.9035 | 0.9404 | +0.0368 | Top-3 exact hit rate improved. |
| 570terms_german, Hit@5 | BioLORD multivector | 0.9368 | 0.9544 | +0.0175 | Top-5 exact hit rate improved. |
| 570terms_german, Hit@10 | BioLORD multivector | 0.9719 | 0.9737 | +0.0018 | Top-10 exact hit rate held steady. |
| 570terms_german, MaxOntSim@10 | BioLORD multivector | 0.9854 | 0.9868 | +0.0014 | Ontology-near retrieval remains high. |
| 570terms_german, runtime | BioLORD multivector | n/a | 16.94s | n/a | Timed locally with /usr/bin/time against extracted release bundle. |
| 570terms_german, MRR@1 | BioLORD single-vector | n/a | 0.5561 | n/a | No matching previous single-vector 570-case artifact was present under data/results. |
| 570terms_german, MRR@3 | BioLORD single-vector | n/a | 0.6640 | n/a | New single-vector MRR overall is 0.6940. |
| 570terms_german, MRR@5 | BioLORD single-vector | n/a | 0.6829 | n/a | New single-vector benchmark ran against extracted bundle. |
| 570terms_german, MRR@10 | BioLORD single-vector | n/a | 0.6915 | n/a | New single-vector top-10 hit rate is 0.9404. |
| 570terms_german, Hit@1 | BioLORD single-vector | n/a | 0.5561 | n/a | 570 cases. |
| 570terms_german, Hit@3 | BioLORD single-vector | n/a | 0.7947 | n/a | 570 cases. |
| 570terms_german, Hit@5 | BioLORD single-vector | n/a | 0.8754 | n/a | 570 cases. |
| 570terms_german, Hit@10 | BioLORD single-vector | n/a | 0.9404 | n/a | 570 cases. |
| 570terms_german, MaxOntSim@10 | BioLORD single-vector | n/a | 0.9744 | n/a | Ontology-near retrieval remains high. |
| 570terms_german, runtime | BioLORD single-vector | n/a | 13.20s | n/a | Timed locally with /usr/bin/time against extracted release bundle. |

## Single-Word German Subset

The single-word subset was derived from tests/data/benchmarks/german/570terms_german.json by keeping one-token German query strings. It contains 340 cases.

| Dataset | Index | MRR@1 | MRR@3 | MRR@5 | MRR@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MaxOntSim@10 | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-word German | BioLORD single-vector | 0.5618 | 0.6667 | 0.6852 | 0.6921 | 0.5618 | 0.8000 | 0.8794 | 0.9324 | 0.9679 | 11.71s |
| single-word German | BioLORD multivector | 0.8529 | 0.8848 | 0.8897 | 0.8924 | 0.8529 | 0.9235 | 0.9441 | 0.9647 | 0.9818 | 14.03s |

## LLM GeneReviews Multivector Run

This run used the extracted BioLORD multivector bundle, the repo default LLM model, and the 10 GeneReviews documents in tests/data/en/phenobert/GeneReviews.

| Dataset | Index | LLM provider | LLM model | Cases | Strict micro F1 | Soft micro F1 | Partial micro F1 | Runtime | Tokens | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GeneReviews | BioLORD multivector | gemini | gemini-3.1-flash-lite | 10 | 0.8164 | 0.8828 | 0.9357 | 80.12s | 124352 | Used two_phase, whole_document_grounded, seed 123, ontology-aware metrics. |

## Bundle Validation

- Minimal bundle extracted with checksum verification; manifest HPO version is v2026-02-16 with 19389 active terms and 555 obsolete terms.
- BioLORD single-vector bundle extracted with checksum verification; manifest HPO version is v2026-02-16, multi_vector is false, dimension is 768.
- BioLORD multivector bundle extracted with checksum verification; manifest HPO version is v2026-02-16, multi_vector is true, dimension is 768.
- Single-vector query sanity check for "Nierensteine" returned Urolithiasis and Nephrolithiasis in the top two results.
- Multivector query sanity check for "Nierensteine" returned Nephrolithiasis as the top result.

## Release Verification

- GitHub release: https://github.com/berntpopp/phentrieve/releases/tag/data-v2026-02-16
- Release title: Pre-built HPO Data Bundles (HPO v2026-02-16)
- Published: 2026-05-22T20:48:33Z
- Uploaded assets: 21 total: 19 tar.gz bundles, SHA256SUMS.txt, and SHA256SUMS-multivec.txt.
- CLI listing verified data-v2026-02-16 and showed minimal, BioLORD single-vector, BioLORD multivector, and the remaining model single/multivector assets.
- CLI single-vector download verified phentrieve-data-v2026-02-16-biolord.tar.gz, extracted checksums successfully, and installed HPO v2026-02-16 with 19389 active terms.
- CLI multivector download verified phentrieve-data-v2026-02-16-biolord-multivec.tar.gz, extracted checksums successfully, and installed HPO v2026-02-16 with 19389 active terms.
- `phentrieve data status` verified the downloaded single-vector bundle at .runs/download-verify-v2026-02-16/single with hpo_data.db and Chroma indexes present.
- `phentrieve data status` verified the downloaded multivector bundle at .runs/download-verify-v2026-02-16/multi with hpo_data.db and Chroma indexes present.

## Decision

Publish. The new bundles load, extracted checksums verify, direct retrieval queries work, and the benchmark results do not indicate broken retrieval. The BioLORD multivector 570-case benchmark improves over the previous local v2025-11-24 artifact across MRR@1/3/5/10 and Hit@1/3/5/10. The active term count changed by four terms, which is consistent with an ontology refresh rather than an index construction failure.
