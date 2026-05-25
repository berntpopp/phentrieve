# Phentrieve Retrieval And Prompting Improvement Report

Date: 2026-05-23
Scope: retrieval, full-text RAG, LLM prompting, and evaluation strategy
Status: research analysis with follow-up outcomes; full-text parity completed,
LLM evidence/enriched mapping attempt superseded after benchmark regression

## Executive Summary

Phentrieve already has a strong foundation: multilingual HPO retrieval,
multi-vector indexing, chunked full-text processing, assertion detection,
ontology-aware evaluation, and a two-phase LLM backend. The best improvement
path is not a rewrite. It is to make the strongest retrieval path available
consistently across direct query and full-text extraction, then tighten
evidence validation and mapping prompts.

The highest-value opportunities are:

1. Make full-text standard extraction use the same multi-vector aggregation
   behavior as direct query.
2. Add a hybrid retrieval cascade: lexical/dictionary candidates, current dense
   multi-vector candidates, and optional synthetic clinical phrase embeddings.
3. Add an optional HPO-specific reranker for ambiguous top-K candidates, not a
   generic reranker.
4. Replace average-only full-text aggregation with max-score, evidence-count,
   assertion, and ontology-aware scoring.
5. Validate LLM evidence before phase 2 mapping: chunk IDs, evidence text,
   offsets, and assertion category consistency.
6. Enrich mapping payloads with definitions, synonyms, matched component, and
   ontology context.
7. Keep phase 1 source-faithful and move normalization/rewrite behavior into
   phase 2.
8. Require A/B benchmark gates for every retrieval or prompt change.

## Update 2026-05-25: What Was Tried And What Changed

Follow-up work executed the report's first implementation candidates, with
different outcomes:

| Item | Outcome | Planning evidence |
| --- | --- | --- |
| Full-text multi-vector parity | Completed and moved to completed planning records. | [`../completed/2026-05-24-full-text-multi-vector-parity-plan.md`](../completed/2026-05-24-full-text-multi-vector-parity-plan.md), [`../completed/2026-05-24-full-text-multi-vector-parity-design.md`](../completed/2026-05-24-full-text-multi-vector-parity-design.md) |
| LLM evidence validation + enriched mapping | Attempted in PR #261, then closed because focused same-command A/B showed regression versus `main`. | [`2026-05-25-llm-evidence-validation-enriched-mapping-pr-regression.md`](2026-05-25-llm-evidence-validation-enriched-mapping-pr-regression.md) |
| Original combined implementation plan/spec | Archived as superseded; the direction remains useful, but the combined PR is not mergeable. | [`../archived/2026-05-25-llm-evidence-validation-enriched-mapping-plan.md`](../archived/2026-05-25-llm-evidence-validation-enriched-mapping-plan.md), [`../archived/2026-05-25-llm-evidence-validation-enriched-mapping-design.md`](../archived/2026-05-25-llm-evidence-validation-enriched-mapping-design.md) |

The focused same-command comparison used the same six CSC documents, model,
seed, and index on `main` and PR #261:

| Branch | Commit | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `main` | `ad984b5` | 53 | 18 | 20 | 0.746 | 0.726 | 0.736 |
| PR #261 | `5cd3fce` | 43 | 28 | 30 | 0.606 | 0.589 | 0.597 |

Per-document F1 changed as follows:

| Document | `main` F1 | PR #261 F1 |
| --- | ---: | ---: |
| `CSC_91` | 0.556 | 0.556 |
| `CSC_71` | 0.762 | 0.560 |
| `CSC_18` | 0.737 | 0.632 |
| `CSC_107` | 0.824 | 0.632 |
| `CSC_4` | 0.696 | 0.500 |
| `CSC_85` | 0.783 | 0.651 |

What PR #261 tried:

- Extracted phase 2A into a scoped module as a prerequisite.
- Added Phase 1 evidence validation and trace metadata for chunk IDs, evidence
  text, offsets, and validation outcomes.
- Added enriched Phase 2 candidate context such as definitions, synonyms, and
  matched metadata.
- Kept the implementation inside the existing retrieval architecture: no
  generic reranking, no LLM-as-judge reranking, no hybrid lexical/dense
  retrieval, and no new retrieval subsystem.
- Added targeted fixes after debugging for abbreviation/context handling, but
  the final focused A/B still regressed.

Observed regression mechanism:

- The implementation made Phase 1 more source-faithful, which improved
  provenance but shifted extracted phrases toward raw abbreviations and local
  modifiers such as `GTC`, `unilateral`, `BWGS`, `throbbing`, and
  `sparse axillary`.
- The pre-existing `main` behavior often emitted normalized clinical phrases
  such as `generalized tonic-clonic seizures` or `Dravet syndrome`, which made
  candidate retrieval easier.
- Phase 2 did not recover enough normalized concepts from the source-faithful
  phrases. The result was exact-ID loss and new false positives, especially in
  abbreviation-heavy and modifier-heavy cases.

Revised recommendation after the experiment:

- Keep the full-text multi-vector parity work completed.
- Do not merge or continue PR #261 wholesale.
- Restart from `main` with smaller behavior-gated PRs:
  observability-only trace fields first if behavior-neutral, then evidence
  validation, then Phase 2 abbreviation/context query expansion.
- Test the risky cases before full benchmarks: `GTC`, `BWGS`, `PJS`, seizure
  laterality, headache modifiers, sparse hair/axillary findings, and tumor
  context.
- Require same-command focused A/B before full GeneReviews/CSC runs.

## Research Method

PubTator-Link MCP endpoint used:

- `https://pubtator-link.genefoundry.org/mcp`
- Server reported as `pubtator-link` version `3.2.4`.
- Transport: streamable HTTP.
- Review ID: `phentrieve-rag-hpo-improvement-2026-05-22`.

The broad one-shot `pubtator_ground_question` queries were too constrained and
returned no PMIDs, so the workflow switched to shorter PubTator searches,
manual corpus selection, review evidence indexing, review index inspection, and
review-scoped retrieval.

Indexed corpus:

- PMID:33471061, PhenoTagger
- PMID:35351638, PhenoRerank
- PMID:38913850, FastHPOCR
- PMID:38264716, PhenoBCBERT and PhenoGPT
- PMID:40826123, RAG-HPO
- PMID:40103736, simplified retriever for LLM phenotype normalization
- PMID:39720417, synthetic case reports and embedding-based HPO retrieval
- PMID:38297371, GPT models for phenotype concept recognition
- PMID:39812777, systematic review/meta-analysis of biomedical RAG
- PMID:41191926, DiscHPO continuous/discontinuous phenotype extraction and
  normalization

PubTator review index coverage:

- 10 sources indexed successfully.
- 9 sources had full-text coverage.
- 1 source had abstract-only coverage.
- 1,405 prepared passages.
- 388,580 indexed characters.
- 0 failed sources.

Parallel codebase exploration covered:

- retrieval/indexing architecture
- prompting/LLM/full-text workflows
- evaluation, prior planning, and benchmark safeguards

## Current Architecture Findings

### Retrieval And Indexing

HPO source data is SQLite-backed. Indexing builds ChromaDB collections from HPO
terms. Single-vector documents concatenate label, synonyms, definition, and
comments into one text per term. Multi-vector documents create separate
documents for each label, synonym, and definition component in
[`phentrieve/data_processing/multi_vector_document_creator.py`](../../phentrieve/data_processing/multi_vector_document_creator.py).

Direct query already supports multi-vector aggregation through
[`DenseRetriever.query_multi_vector`](../../phentrieve/retrieval/dense_retriever.py).
That method queries more raw component hits, groups by HPO ID, and aggregates
component scores with configurable strategies.

The README reports a large multi-vector retrieval gain on the 570-term German
benchmark:

| Retrieval Mode | MRR | Hit@1 | Hit@10 | Ont Sim@1 |
| --- | ---: | ---: | ---: | ---: |
| Single-vector | 0.695 | 55.8% | 94.0% | 79.9% |
| Multi-vector `all_max` | 0.892 | 84.0% | 97.4% | 91.9% |

The 2026-05-22 HPO data release analysis also shows strong BioLORD multivector
performance on the 570 German benchmark: MRR@10 `0.8898`, Hit@10 `0.9737`, and
MaxOntSim@10 `0.9868`.

### Full-Text Standard Extraction

Standard full-text extraction uses the text processing pipeline to chunk text
and detect assertion status, then calls
[`orchestrate_hpo_extraction`](../../phentrieve/text_processing/hpo_extraction_orchestrator.py).

Classification: this is an implementation bug or incomplete follow-through from
the multi-vector update, not merely a future enhancement. The code can connect
standard full-text extraction to a multi-vector Chroma collection, but the
standard full-text retrieval path still treats returned rows as raw component
hits instead of applying HPO-level multi-vector aggregation. As a result,
standard full-text mode can diverge from direct query mode for the same model
and index.

The key mismatch: standard full-text extraction calls raw
`DenseRetriever.query_batch()` for chunk retrieval. When connected to a
multi-vector index, this can return component-level hits, but it does not apply
the same HPO-level aggregation used by direct query.

This is separate from the LLM backend. The LLM two-phase path performs phrase
candidate retrieval through `ToolExecutor`, which already has a multi-vector
branch. The gap identified here is specifically in `extraction_backend =
"standard"`, where full text is chunked and mapped without LLM phrase
extraction.

Current aggregation collapses chunk evidence by HPO ID and uses average score
for confidence/filtering, with max score recorded separately:

- `confidence = avg_score`
- filter uses `avg_score`
- sort uses `avg_score`, then evidence count

This can suppress a term with one excellent local match if other weaker
evidence lowers the average.

### LLM Full-Text Pipeline

The active public LLM backend is `two_phase`:

1. phase 1 extracts source-faithful phenotype phrases with chunk IDs and
   evidence text
2. phase 2A retrieves candidate HPO terms
3. phase 2B resolves exact candidate selection locally or with structured LLM
   mapping

The phase 1 prompt in
[`phentrieve/llm/prompts/templates/two_phase/en.yaml`](../../phentrieve/llm/prompts/templates/two_phase/en.yaml)
has the right broad shape: preserve source wording, require chunk IDs, categorize
abnormal/normal/suspected/family history/other, and emit JSON only.

The structured schema in
[`phentrieve/llm/types.py`](../../phentrieve/llm/types.py) includes chunk IDs,
evidence text, and optional offsets, but the system can do more validation
before phase 2.

Phase 2 mapping payloads are compact and currently include:

- primary chunk text
- neighboring chunk text
- phrase
- category
- candidates with id, term, retrieval score, retrieval query, matched text, and
  matched component when present

They do not consistently include HPO definitions, top synonyms, parent labels,
or ontology context.

## Literature Findings

### Hybrid Candidate Generation Is The Strongest Pattern

PhenoTagger combines dictionary matching with machine learning. It uses HPO
concepts and synonyms to build a dictionary, creates weakly supervised training
data, then combines dictionary and ML predictions. The paper frames automatic
phenotype recognition as difficult because dictionary methods are precise but
can miss unseen variants.

FastHPOCR intentionally returns to a dictionary-based approach, arguing that
the HPO concept recognition lifecycle benefits from immediate ontology refresh
and efficient re-analysis. It handles lexical variability with clusters of
morphologically equivalent tokens and reports 10,000 abstracts processed in
5 seconds.

RAG-HPO adds a retrieval-augmented LLM workflow: extract phenotypic phrases,
retrieve semantically similar HPO-linked phrases from a dynamic vector database
of more than 54,000 mapped phenotype phrases, then return best matches to the
LLM for final assignment.

Implication for Phentrieve: the best candidate generator is probably not dense
only. It should combine lexical/dictionary candidates, current BioLORD
multi-vector candidates, and optionally synthetic clinical phrase embeddings.

### Reranking Helps When It Is HPO-Specific

PhenoRerank uses HPO terms, definitions, comments, and sentence context to
filter candidate concepts produced by base annotators. It reports F1 gains over
base methods and removal of many false positives. The important architectural
point is that it reranks candidate concept pairs using ontology descriptive
data and local sentence context.

DiscHPO uses a two-stage normalization architecture: a sentence-transformer
biencoder for candidate generation and a cross-encoder reranker for selecting
the best HPO concept. It also notes that exact span extraction is not always
required for successful normalization if the partial mention captures the
essential concept information.

Implication for Phentrieve: a reranker should be optional, HPO-aware, and
limited to ambiguous candidate sets. A generic cross-encoder reranker is less
attractive than a phenotype-normalization reranker trained or prompted around
HPO labels, definitions, synonyms, and local evidence.

### LLMs Need Retrieval, Constraints, And Cost Controls

The GPT phenotype concept recognition evaluation found strong results only when
the task was constrained with prior knowledge of candidate ontology terms. It
also warned about non-determinism, cost, and lack of concordance across runs.

The simplified retriever paper reports GPT-4o phenotype normalization accuracy
improving from 62% without retriever augmentation to 85% with candidate
retrieval.

The biomedical RAG systematic review found RAG improved LLM performance overall
compared with baseline LLMs and recommends system-level, knowledge-level, and
integration-level enhancements.

Implication for Phentrieve: the current two-phase architecture is directionally
right. The next gains should come from stronger retrieval context, stricter
schema/evidence validation, deterministic abstention behavior, and benchmarked
cost/quality gates.

### Ontology-Aware Evaluation Is Necessary

RAG-HPO reports that many exact-match false positives were broader ancestor
terms rather than unrelated hallucinations. This mirrors Phentrieve's prior
planning analysis: strict exact F1 can understate semantic quality when
predictions are ontology-near.

Phentrieve already has ontology-aware metrics and recent GeneReviews results:

- strict micro F1: `0.8164`
- soft micro F1: `0.8828`
- partial micro F1: `0.9357`
- tokens: `124352`
- runtime: `80.12s`

Implication for Phentrieve: exact-match metrics remain necessary for
comparability, but implementation decisions should be judged with strict, soft,
partial, failed-document count, span coverage, and cost together.

## Recommended Roadmap

### 1. Full-Text Multi-Vector Parity

Problem:

Direct query benefits from multi-vector aggregation. Standard full-text
extraction does not apply the same aggregation semantics per chunk. This should
be treated as a correctness bug in the standard backend after multi-vector
support, because users selecting or defaulting to a multi-vector index should
get consistent HPO-level aggregation across query and full-text standard modes.

Where it happens:

- `run_standard_backend()` dispatches the standard full-text path.
- `orchestrate_hpo_extraction()` performs per-chunk retrieval.
- That function calls raw `retriever.query_batch()` instead of a multi-vector
  HPO-level batch aggregation helper.
- The LLM backend is not the primary affected path.

Design direction:

- Add a chunk-batch multi-vector retrieval helper.
- For each chunk, retrieve enough component hits.
- Aggregate by HPO ID using the same strategies as direct query.
- Convert aggregated HPO results back into the existing chunk result contract.
- Preserve current `query_batch()` behavior for single-vector indexes.

Expected benefit:

- Standard full-text mode should benefit from the same label/synonym/definition
  matching improvements seen in direct retrieval benchmarks.

Risks:

- More raw hits per chunk may increase latency.
- Thresholds calibrated for raw component scores may need retuning.

Benchmark gate:

- GeneReviews full-text strict/soft/partial F1.
- PhenoBERT subsets.
- Runtime per document.
- 570 German retrieval benchmark must not regress.

### 2. Hybrid Candidate Generator

Problem:

Dense retrieval alone can miss exact lexical variants, rare abbreviations,
surface forms, and newly refreshed ontology terminology.

Design direction:

- Add a candidate union layer with provenance:
  - dictionary/FastHPOCR-style lexical candidates
  - current BioLORD multi-vector candidates
  - optional synthetic clinical phrase embedding candidates
- Deduplicate by HPO ID while preserving source channel and per-channel score.
- Feed the union into existing local/LLM mapping.

Expected benefit:

- Higher recall without forcing the LLM to invent candidates.
- Better handling of rare wording, abbreviations, and exact synonyms.

Risks:

- Candidate union can increase false positives.
- Needs an optional reranker or stronger candidate scoring to avoid precision
  loss.

### 3. Optional HPO-Specific Reranker

Problem:

Candidate generation can return semantically close but clinically wrong terms,
especially siblings, broad ancestors, and terms matching modifiers rather than
the true phenotype.

Design direction:

- Add an optional reranking phase over top-K candidates only.
- Inputs:
  - phrase
  - evidence span/chunk
  - assertion category
  - candidate label
  - definition
  - synonyms
  - matched component
  - parent labels or short ancestor path
- Support two backends:
  - lightweight LLM judge for experiments
  - later HPO-specific cross-encoder if benchmarks justify it

Expected benefit:

- Better precision in ambiguous mappings.
- Better abstention when no candidate is adequate.

Risks:

- Cost and latency.
- The repository has a regression test banning old reranker references; any
  new work must be clearly scoped as a new optional HPO-normalization reranker,
  not a revival of removed generic reranking.

### 4. Evidence-Aware Aggregated Confidence

Problem:

Average-only confidence can penalize a strong local match and over-reward many
weak matches.

Design direction:

Compute a richer aggregate:

- `max_score`
- `avg_score`
- `evidence_count`
- `top_evidence_chunk_idx`
- assertion bucket
- source channel if hybrid retrieval is added
- ontology coherence with neighboring predicted terms

Use a configurable score formula, for example:

```text
aggregate_score =
  0.55 * max_score +
  0.25 * avg_score +
  0.10 * evidence_count_bonus +
  0.10 * ontology_coherence_bonus
```

Expected benefit:

- Preserve high-confidence local evidence.
- Better ranking of repeated weak evidence versus single decisive evidence.

Risks:

- Needs calibration per embedding model.
- Must preserve exact old behavior behind defaults or profile flags until
  benchmarks prove the new strategy.

### 5. LLM Evidence Validation Before Phase 2

Problem:

The phase 1 schema asks for chunk IDs, evidence text, and optional offsets, but
invalid or weakly grounded records can still reach mapping.

Design direction:

Before phase 2, validate every extracted phenotype:

- referenced chunk IDs exist
- evidence text appears in the referenced chunk, or fuzzy match exceeds a
  threshold
- offsets are within bounds and match the evidence text
- category is one of the allowed values
- normal/suspected/family history categories are compatible with local
  assertion signals when available

Failure modes:

- repair offsets if evidence text matches
- drop offsets but keep chunk-level evidence if only the span is weak
- downgrade evidence precision in metadata
- exclude records that cannot be grounded

Expected benefit:

- Fewer hallucinated or misanchored mappings.
- Better frontend evidence highlighting.
- More honest span-versus-chunk provenance.

### 6. Enriched Mapping Payloads

Problem:

The mapping prompt often sees only candidate labels and scores. Hard HPO
normalization needs definitions, synonyms, and ontology context.

Design direction:

Add compact candidate enrichment:

- definition, truncated to a safe length
- top synonyms, preferably only the matched synonym plus a few alternates
- matched component: label, synonym, definition, synthetic phrase, dictionary
- parent labels or one-line ancestor path
- obsolete/replaced-by metadata if relevant

Prompt rule changes:

- Retrieval score is a hint.
- Definition and evidence-context semantic equivalence are primary.
- If multiple candidates are plausible, choose the most specific term supported
  by the evidence.
- If only a broad ancestor is supported, allow that but label it as broad.
- Return null if no candidate is supported.

Expected benefit:

- Better disambiguation among sibling terms.
- Better handling of abbreviations and indirect wording.

### 7. Keep Phase 1 Source-Faithful

Problem:

It is tempting to ask phase 1 to normalize text into HPO-like wording, but that
damages provenance.

Design direction:

- Phase 1 remains source-faithful.
- Phase 2 generates retrieval variants:
  - original phrase
  - unit-stripped phrase
  - abbreviation-expanded phrase
  - normalized spelling/morphology
  - language-specific translation only if configured

Expected benefit:

- Cleaner provenance.
- More controlled normalization.
- Easier debugging when mappings are wrong.

### 8. A/B Evaluation Gate

Every retrieval or prompt change should produce a comparable benchmark artifact.

Minimum metrics:

- strict micro/macro F1
- soft micro/macro F1
- partial micro/macro F1
- failed document count
- token input/output
- request count
- runtime
- percentage of predictions with chunk IDs
- percentage with valid evidence spans
- assertion accuracy by present/negated/uncertain/family history
- top error categories

Suggested benchmark sets:

- 570 German single-term retrieval benchmark
- single-word German subset
- GeneReviews PhenoBERT subset
- multi-finding clinical note fixtures
- targeted negation/family-history/uncertainty fixtures
- discontinuous mention fixtures, if available

## Concrete Next Implementation Candidates

### Candidate A: Full-Text Multi-Vector Parity (completed)

Smallest high-impact code change and the clearest bug fix from this report.
This is an incomplete implementation of multi-vector behavior in standard
full-text extraction, not a prompt change.

Likely files:

- `phentrieve/text_processing/hpo_extraction_orchestrator.py`
- `phentrieve/retrieval/dense_retriever.py`
- `phentrieve/retrieval/utils.py`
- `phentrieve/text_processing/full_text_service.py`
- `tests/unit/text_processing/`
- `tests/unit/retrieval/`

Acceptance criteria:

- Standard full-text extraction can use true multi-vector HPO-level aggregation.
- Direct query and standard full-text use equivalent aggregation semantics for
  multi-vector indexes.
- Single-vector behavior remains unchanged.
- Benchmarks include before/after results.
- LLM backend behavior remains unchanged except for shared helper reuse if that
  is intentionally added later.

2026-05-25 outcome: this candidate has been implemented and moved to completed
planning records. Future work should treat it as baseline infrastructure, not
as part of the failed PR #261 experiment.

### Candidate B: LLM Evidence Validator (attempted; restart smaller)

Most likely to improve trust and UI evidence quality.

Likely files:

- `phentrieve/llm/pipeline.py`
- `phentrieve/llm/types.py`
- `phentrieve/text_processing/full_text_service.py`
- `tests/unit/llm/`
- `tests/unit/text_processing/`

Acceptance criteria:

- Invalid chunk IDs are rejected or repaired.
- Evidence text is validated against chunk text.
- Offset precision is represented honestly.
- Tests cover negation, uncertainty, family history, and bad chunk IDs.

2026-05-25 outcome: attempted in PR #261 together with source-faithful Phase 1
changes and enriched Phase 2 mapping. The focused same-command A/B regressed
from F1 `0.736` on `main` to F1 `0.597` on PR #261, so this should be
restarted from `main` as a smaller gated change.

### Candidate C: Enriched Mapping Payload (attempted; restart smaller)

Best prompt-only or prompt-adjacent improvement.

Likely files:

- `phentrieve/llm/pipeline_phase2.py`
- `phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml`
- `phentrieve/llm/prompts/templates/two_phase/en_mapping_batch.yaml`
- `phentrieve/retrieval/details_enrichment.py`
- `tests/unit/llm/test_prompts.py`
- `tests/unit/llm/test_two_phase.py`

Acceptance criteria:

- Candidate payload includes compact definitions and matched synonyms.
- Prompt asks for evidence-supported specificity and abstention.
- Token growth is measured.
- A/B benchmark does not increase failed-doc count.

2026-05-25 outcome: candidate enrichment remains directionally useful, but the
PR #261 implementation did not sufficiently compensate for source-faithful
abbreviations and modifiers in Phase 2 query construction. A new attempt should
prove abbreviation/context recovery on focused fixtures before full benchmark
runs.

## Source Links

Primary literature and tools:

- PhenoTagger: https://pubmed.ncbi.nlm.nih.gov/33471061/
- PhenoRerank: https://pubmed.ncbi.nlm.nih.gov/35351638/
- FastHPOCR: https://pubmed.ncbi.nlm.nih.gov/38913850/
- PhenoBCBERT and PhenoGPT: https://pubmed.ncbi.nlm.nih.gov/38264716/
- GPT evaluation for phenotype concept recognition:
  https://pubmed.ncbi.nlm.nih.gov/38297371/
- Synthetic case reports and embedding-based HPO retrieval:
  https://pubmed.ncbi.nlm.nih.gov/39720417/
- Simplified retriever for phenotype normalization:
  https://pubmed.ncbi.nlm.nih.gov/40103736/
- RAG-HPO: https://pubmed.ncbi.nlm.nih.gov/40826123/
- Biomedical RAG systematic review/meta-analysis:
  https://pubmed.ncbi.nlm.nih.gov/39812777/
- DiscHPO continuous/discontinuous phenotype extraction and normalization:
  https://pubmed.ncbi.nlm.nih.gov/41191926/
- SapBERT: https://aclanthology.org/2021.naacl-main.334/
- MedCPT: https://arxiv.org/abs/2307.00589
- HPO 2024 update: https://academic.oup.com/nar/article/52/D1/D1333/7416384

Internal context:

- Multi-vector document creation:
  [`phentrieve/data_processing/multi_vector_document_creator.py`](../../phentrieve/data_processing/multi_vector_document_creator.py)
- Dense retriever:
  [`phentrieve/retrieval/dense_retriever.py`](../../phentrieve/retrieval/dense_retriever.py)
- Full-text extraction orchestrator:
  [`phentrieve/text_processing/hpo_extraction_orchestrator.py`](../../phentrieve/text_processing/hpo_extraction_orchestrator.py)
- Full-text aggregation helpers:
  [`phentrieve/text_processing/_hpo_extraction_helpers.py`](../../phentrieve/text_processing/_hpo_extraction_helpers.py)
- LLM phase 2 mapping:
  [`phentrieve/llm/pipeline_phase2.py`](../../phentrieve/llm/pipeline_phase2.py)
- LLM types:
  [`phentrieve/llm/types.py`](../../phentrieve/llm/types.py)
- Two-phase extraction prompt:
  [`phentrieve/llm/prompts/templates/two_phase/en.yaml`](../../phentrieve/llm/prompts/templates/two_phase/en.yaml)
- HPO v2026-02-16 benchmark comparison:
  [`2026-05-22-hpo-v2026-02-16-benchmark-comparison.md`](2026-05-22-hpo-v2026-02-16-benchmark-comparison.md)

## Bottom Line

Phentrieve's retrieval stack is already strong, especially with BioLORD
multi-vector indexes. The most important gap is consistency: full-text
extraction should use the same HPO-level multi-vector aggregation as direct
query. After that, the main gains should come from stronger evidence validation
and richer phase-2 mapping context, not from making phase 1 less faithful to
the source text.

The literature supports a hybrid architecture: dictionary/lexical recall,
dense semantic candidate generation, ontology-aware or cross-encoder reranking
for ambiguous candidates, and LLM mapping constrained to retrieved candidates
with explicit abstention. That architecture fits Phentrieve's current design
well and can be added incrementally behind benchmark gates.

2026-05-25 outcome: the consistency gap has been addressed by the full-text
multi-vector parity work. Evidence validation and enriched mapping are still
worth pursuing, but PR #261 showed that making Phase 1 more source-faithful can
hurt exact HPO mapping unless Phase 2 explicitly recovers abbreviations,
clinical expansions, and context-sensitive concepts. Treat that as the next
required design constraint.

---

# Update 2026-05-24: Code Verification, Current Best-Practice Citations, and Implementation Diffs

> Author: deep code-and-literature pass against the live tree at commit
> `ed4b266`. The original report's diagnoses are correct on every claim
> tested. This update documents *what changed in the code since the report*,
> *what the report under-stated*, and *exact diffs* for the P0/P1 items.
> All cited URLs verified live in this session.

## Scope of verification

Each claim in the original report was checked against the live source. Six
findings are confirmed, three are extended in scope, and one minor finepoint
(`run_standard_backend()`) does not appear in the current tree and was
likely renamed into `phentrieve/text_processing/full_text_service.py`.

| Original §  | Status   | Evidence in current tree                                                            |
| ----------- | -------- | ----------------------------------------------------------------------------------- |
| §1 parity   | CONFIRMED + BROADER | `phentrieve/text_processing/hpo_extraction_orchestrator.py:78-83` calls `retriever.query_batch(...)` unconditionally. **Also affects** `phentrieve/llm/tools.py:235` (ToolExecutor `_process_clinical_text`), so the LLM `process_clinical_text` tool inherits the bug. Two-phase phase 2A is correct because it goes through `ToolExecutor.query_batch_hpo_terms` (tools.py:100) which *does* branch to multi-vector. |
| §4 avg-only confidence | CONFIRMED | `_hpo_extraction_helpers.py:225` sets `"confidence": avg_score`; line 237 sorts by `(-avg_score, -count)`. `max_score` is recorded but unused for ranking. |
| §5 evidence validation | CONFIRMED MISSING | No validator exists between phase 1 and phase 2. `pipeline.py:1230-1233` propagates LLM-emitted `chunk_ids`, `evidence_text`, `start_char`, `end_char` straight into the phase-2 payload. The Pydantic schema enforces shape, not grounding. |
| §6 enriched payload | CONFIRMED + FIXABLE TODAY | `pipeline_phase2.py:90-125` (`compact_mapping_item`) emits only `id, term, retrieval_score, retrieval_query, matched_text, matched_component` per candidate. **However**, `phentrieve/retrieval/details_enrichment.py:51-154` (`enrich_results_with_details`) is already a working pure-function helper that returns `definition` and `synonyms` — it just isn't wired into the mapping prompt path. |
| §7 source-faithful phase 1 | CONFIRMED + COMPROMISED BY DEMOS | The system prompt in `phentrieve/llm/prompts/templates/two_phase/en.yaml:23` says "Preserve the wording from the note whenever possible" but its own few-shot example 2 (lines 75-87) rewrites *"lactate dehydrogenase was markedly elevated"* → `phrase: "elevated lactate dehydrogenase"`. The demonstrations actively teach normalization. See §"New finding A" below. |
| §3 reranker | NUANCED | An LLM-judge variant doubles cost without changing the candidate set the mapping LLM ultimately sees. Skip in favor of a real cross-encoder. DiscHPO's published winner is `cross-encoder/ms-marco-electra-base` (see §"Best-Practice Citations" below). |

## New findings the original report missed

### New finding A — Phase-1 few-shot examples contradict the system prompt

The Anthropic Claude 4 prompting guide explicitly warns that demonstrations
shape the model's *output distribution* and that Claude 4.x interprets prompts
"more literally and explicitly" than 4.6, so it "will not silently generalize
an instruction from one item to another." Min et al. (2022, cited in the
Prompting Guide) found that "the format you use... is much better than no
labels at all" — the *shape* of the examples dominates. Conclusion: any
few-shot output that violates a system-prompt rule effectively *overrides*
the rule.

`en.yaml` (current):

```yaml
# Example 2 — system rule says "Preserve the wording from the note"
# but the demo paraphrases.
- input: |
    Blood tests showed lactate dehydrogenase was markedly elevated, and
    urine output remained low.
output: |
    {"phenotypes": [
      {"phrase": "elevated lactate dehydrogenase", ...},
      {"phrase": "low urine output", ...}
    ]}
```

Source citation:
- Anthropic Claude 4 best-practices: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices
- Min et al. format-dominance summary: https://www.promptingguide.ai/techniques/fewshot

### New finding B — `hybrid_select_candidates` is dense-only, not hybrid

`pipeline_phase2.py:164-202` is named `hybrid_select_candidates` but only
implements *token-overlap filtering over dense results*. There is no
separate lexical channel. Any future hybrid work (report §2) must rename
this function or it will compound the naming confusion.

### New finding C — `PHENOTYPE_ABBREVIATIONS` is a one-element dict

`phentrieve/llm/pipeline_phase1.py:38-40`:

```python
PHENOTYPE_ABBREVIATIONS = {"xlid": "X-linked intellectual disability"}
```

This is the entire abbreviation table. FastHPOCR (PMID 38913850) reports
that morphologic/lexical variability is the single biggest gap for
dictionary methods and uses precomputed clusters. Phentrieve has one
abbreviation. Either expand the table or replace with a curated lexicon.

### New finding D — Three retrieval surfaces with three behaviors

The report frames the bug as "standard full-text" only. The live code has
four call sites and three different multi-vector behaviors:

| Call site                                          | Path                                                | Multi-vector aware? |
| -------------------------------------------------- | --------------------------------------------------- | ------------------- |
| Direct query CLI                                   | `DenseRetriever.query_multi_vector`                 | yes                 |
| Two-phase LLM phase 2A                             | `ToolExecutor.query_batch_hpo_terms`                | yes                 |
| Standard full-text extraction                      | `orchestrate_hpo_extraction` → `query_batch`        | **no**              |
| `process_clinical_text` MCP/LLM tool               | `orchestrate_hpo_extraction` → `query_batch`        | **no**              |

The right fix lives one layer down: make `query_batch` itself index-aware
(or rename it and split the public surface). Adding a fourth helper
("`query_batch_multi_vector`") without converging the existing callers is
how the inconsistency arose in the first place.

### New finding E — `pipeline.py` >1700 LOC blocks safe iteration

The knowledge graph confirms `phentrieve/llm/pipeline.py` is the densest
node in the LLM layer (imports types, phase1, phase2, providers,
prompts/loader, tools, retry, trace, preprocessing). `_retrieve_candidates`
alone is ~130 lines (lines 1115-1240) and re-implements grouping logic
that already exists in `phase2.downstream_dedupe_key`. Implementing the
report's §1+§5+§6 directly into this class adds ~200 more lines to a file
that's already past safe-review size. Extract `phase2a.py` and
`mapping_payload_builder.py` *before* adding features.

### New finding F — Security/safety block duplicated across every prompt YAML

`safety.py` exists and exposes `apply_safety_instruction()` but every YAML
in `phentrieve/llm/prompts/templates/**/*.yaml` re-bakes the security
block. If you change posture you must update ~12 files. Centralize via
loader, similar to how OpenAI/Anthropic cookbooks recommend a single
"system safety preamble" injected at render time.

## Best-Practice Citations (verified 2026-05-24)

### Prompt engineering (Claude 4.x, OpenAI strict, Gemini)

| Topic | Source | Key guidance |
| ----- | ------ | ------------ |
| Few-shot example design | Anthropic Claude 4 best-practices | Wrap examples in `<example>` / `<examples>` tags. Use 3-5 examples. Make them relevant, diverse, structured. Positive examples beat negative instructions. Claude 4.7 interprets prompts literally. |
| Citations / grounding | Anthropic Citations API doc | Returns 0-indexed character ranges; pointers are guaranteed valid. **Incompatible with Structured Outputs in the same call** — returns HTTP 400. |
| Faithfulness measurement | RAGAS docs | `Faithfulness = supported_claims / total_claims`. LLM-as-judge, not string match. |
| Structured outputs | OpenAI strict-mode docs | All fields must be in `required`; `minLength`, `maximum`, `pattern`, `$ref` are silently unsupported. |
| Structured outputs | Gemini structured-output doc | Supports OpenAPI v3.0.3 subset; ignores unsupported keys silently. |
| Untrusted-input tagging | Claude 4 best-practices + AWS Prescriptive | Wrap untrusted text in `<document_content>` / `<clinical_note>` tags; salt tag names per request for multi-tenant. |

URLs:
- Claude 4 best-practices: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices
- Anthropic Citations: https://platform.claude.com/docs/en/build-with-claude/citations
- OpenAI Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- Gemini Structured Outputs: https://ai.google.dev/gemini-api/docs/structured-output
- RAGAS Faithfulness: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/

### Retrieval, reranking, calibration (biomedical NER, 2024-2025)

| Topic | Source | Key guidance |
| ----- | ------ | ------------ |
| Hybrid lexical+dense fusion | Cormack et al. SIGIR 2009 RRF | `score = Σ 1/(k + rank_i)`, default `k=60`, optimum flat over `[20, 100]`. Prefer RRF over convex combo when score scales differ (BM25 unbounded, cosine in [-1,1]). |
| LangChain ensemble | EnsembleRetriever API | RRF with tunable `c` and per-retriever weights; works as drop-in over BM25 + Chroma. |
| HPO cross-encoder | DiscHPO (PMID 41191926) | `cross-encoder/ms-marco-electra-base` selected after benchmark; F1 0.723 on normalization. |
| HPO cross-encoder | PhenoRerank (PMID 35351638) | BlueBERT contrastive pretrained on HPO label/def/comment pairs; +18% precision, removes >80% false positives. |
| Cross-encoder API | sentence-transformers docs | `CrossEncoder.rank(query, documents, top_k=10)` is the idiomatic call. |
| ChromaDB multi-vector | Chroma docs | No native multi-vector primitive. Over-fetch `n_results ≈ K × m × 1.5..2`. Late interaction needs Qdrant or Vespa. |
| Score calibration | arXiv 2601.16907 (2026) | Fit `IsotonicRegression` on labeled `(score, is_correct)` pairs. Order-preserving so MRR/Recall@K invariant. |
| Multi-chunk evidence | ColBERT long-context (Vespa) | Per-window MaxSim then max-aggregate; smooth alternative `max_score × log(1 + n_chunks)`. |
| Diversity | Carbonell & Goldstein 1998 MMR | `MMR = argmax[λ·Rel - (1-λ)·max Sim]`; λ≈0.7 starting point. Useful to prevent sibling HPO terms in output list. |
| Multilingual clinical SOTA | BioLORD-2023-M (JAMIA Sep 2024) | Strongest published multilingual clinical sentence encoder for German+English. No clear successor as of mid-2026. |

URLs:
- DiscHPO: https://medinform.jmir.org/2025/1/e68558
- PhenoRerank PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11040548/
- BioLORD-2023 paper: https://academic.oup.com/jamia/article/31/9/1844/7614965
- sentence-transformers retrieve-and-rerank: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- LangChain EnsembleRetriever: https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html
- RRF intro: https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/
- MMR (Carbonell & Goldstein 1998): https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
- Calibrated similarity for embeddings: https://arxiv.org/html/2601.16907
- ChromaDB query docs: https://docs.trychroma.com/docs/querying-collections/query-and-get

## Implementation Diffs

The diffs below are illustrative — final code should be a tested PR with
benchmarks attached. All file paths and line numbers reference commit
`ed4b266`. The order matches the revised priority matrix at the end of
this section.

### P0-A — Multi-vector parity (one shared retrieval path)

**Why one path, not two helpers**: today there are four call sites with
three behaviors (New finding D). Adding a fifth named helper compounds the
problem. The cleanest fix puts the dispatch *inside* `DenseRetriever` so
the policy lives with the data.

`phentrieve/retrieval/dense_retriever.py` — add a new method that batches
the multi-vector path, mirroring `query_multi_vector` but vectorized:

```python
# phentrieve/retrieval/dense_retriever.py  (NEW METHOD, after query_multi_vector)

def query_batch_multi_vector(
    self,
    texts: list[str],
    n_results: int = 10,
    aggregation_strategy: str | AggregationStrategy = (
        AggregationStrategy.LABEL_SYNONYMS_MAX
    ),
    component_weights: dict[str, float] | None = None,
    custom_formula: str | None = None,
) -> list[list[dict[str, Any]]]:
    """Batched multi-vector retrieval with per-text HPO-level aggregation.

    Returns one ranked candidate list per input text. Each list contains
    aggregated HPO entries shaped like ``query_multi_vector`` output.
    """
    if not texts:
        return []
    raw_n = n_results * MULTI_VECTOR_RESULT_MULTIPLIER
    # Re-use the proven batch path; it already over-fetches by 3x internally
    # so the effective ChromaDB n_results = raw_n * 3 — fine, IVF is fast.
    raw_batch = self.query_batch(
        texts=texts, n_results=raw_n, include_similarities=True
    )
    return [
        aggregate_multi_vector_results(
            results=raw_one,
            strategy=aggregation_strategy,
            weights=component_weights,
            custom_formula=custom_formula,
            min_similarity=self.min_similarity,
        )[:n_results]
        for raw_one in raw_batch
    ]
```

`phentrieve/text_processing/hpo_extraction_orchestrator.py` — branch on
index type instead of calling `query_batch` blindly:

```diff
 if precomputed_query_results is not None:
     ...
     all_query_results = precomputed_query_results
 else:
-    logger.info(f"Batch querying {len(text_chunks)} chunks at once")
-    all_query_results = retriever.query_batch(
-        texts=text_chunks,
-        n_results=num_results_per_chunk,
-        include_similarities=True,
-    )
+    index_type = retriever.detect_index_type()
+    logger.info(
+        "Batch querying %d chunks (index_type=%s)",
+        len(text_chunks), index_type,
+    )
+    if index_type == "multi_vector":
+        # Multi-vector index: aggregate per HPO term before chunk dedup.
+        aggregated_per_chunk = retriever.query_batch_multi_vector(
+            texts=text_chunks,
+            n_results=num_results_per_chunk,
+        )
+        # Adapt aggregated [{hpo_id, label, similarity, ...}] back to the
+        # query_batch contract that process_chunk_matches expects.
+        all_query_results = [
+            _aggregated_to_query_batch_shape(items)
+            for items in aggregated_per_chunk
+        ]
+    else:
+        all_query_results = retriever.query_batch(
+            texts=text_chunks,
+            n_results=num_results_per_chunk,
+            include_similarities=True,
+        )
```

The adapter is small and keeps the downstream contract stable:

```python
# phentrieve/text_processing/_hpo_extraction_helpers.py  (NEW HELPER)

def _aggregated_to_query_batch_shape(
    aggregated: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert query_batch_multi_vector output into the wrapping shape
    that ``process_chunk_matches`` consumes (ids/metadatas/similarities
    nested one level deep, matching ChromaDB's native batch format)."""
    metadatas = [
        {"id": item["hpo_id"], "label": item.get("label", ""),
         "matched_component": item.get("matched_component"),
         "matched_text": item.get("matched_text")}
        for item in aggregated
    ]
    similarities = [float(item.get("similarity", 0.0)) for item in aggregated]
    ids = [item["hpo_id"] for item in aggregated]
    return {
        "ids": [ids],
        "metadatas": [metadatas],
        "similarities": [similarities],
        "documents": [[item.get("label", "") for item in aggregated]],
        "distances": [[1.0 - s for s in similarities]],
    }
```

**Tests required**:
- `tests/unit/retrieval/test_query_batch_multi_vector.py` — fakes a
  multi-vector ChromaDB collection, asserts aggregated shape matches.
- `tests/unit/text_processing/test_orchestrator_multi_vector.py` — asserts
  `process_chunk_matches` receives aggregated HPO IDs (not component IDs).
- `tests/integration/test_full_text_parity.py` — same 570-term German
  benchmark routed through both direct query and standard full-text; MRR
  must be within 0.01 of each other when both use the multi-vector index.

### P0-B — Phase-1 source-faithful few-shot examples

`phentrieve/llm/prompts/templates/two_phase/en.yaml` — rewrite the demos
so every `phrase`/`evidence_text` is a verbatim substring of the input.
This is the most expensive prompt change *per word edited* in the entire
roadmap.

```diff
 - input: |
     Extract all phenotype phrases from the following clinical text.
     ---
     Blood tests showed lactate dehydrogenase was markedly elevated, and
     urine output remained low.
     ---
   output: |
     {
       "phenotypes": [
-        {"phrase": "elevated lactate dehydrogenase", "category": "Abnormal",
-         "chunk_ids": [1],
-         "evidence_text": "lactate dehydrogenase was markedly elevated"},
-        {"phrase": "low urine output", "category": "Abnormal",
-         "chunk_ids": [1],
-         "evidence_text": "urine output remained low"}
+        {"phrase": "lactate dehydrogenase was markedly elevated",
+         "category": "Abnormal", "chunk_ids": [1],
+         "evidence_text": "lactate dehydrogenase was markedly elevated"},
+        {"phrase": "urine output remained low",
+         "category": "Abnormal", "chunk_ids": [1],
+         "evidence_text": "urine output remained low"}
       ]
     }
```

Also bump `version: "v3.1.0"` and add to the system prompt's "Extraction
rules":

```yaml
  - Every phrase must be a verbatim substring of the evidence_text.
  - Every evidence_text must be a verbatim substring of the referenced chunk.
  - Phase 2 will compute retrieval variants (canonicalized noun phrases,
    abbreviation expansion); your job is faithful extraction, not rewriting.
```

This pairs naturally with §"P1-B (retrieval query variants)" below — the
existing `prepare_retrieval_queries` helper in `pipeline_phase2.py:35-44`
can be extended to generate canonical noun-phrase variants (`"elevated
lactate dehydrogenase"` from `"lactate dehydrogenase was markedly
elevated"`) without dragging that responsibility into the LLM prompt.

### P1-A — Phase-1 evidence validator

New module `phentrieve/llm/evidence_validation.py`. Pure functions,
returns a (validated_items, dropped_items, repair_notes) triple so the
trace can record what was changed.

```python
# phentrieve/llm/evidence_validation.py  (NEW FILE)
"""Validate phase-1 LLM extractions against the source text.

Three-tier validation:
1. chunk_id existence    -> drop record
2. evidence_text in chunk -> exact / fuzzy / drop with downgrade
3. (start, end) bounds   -> repair from evidence_text if mismatched
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz  # already viable; widely used in clinical NLP

# Vendor-doc gap (per RAGAS): no canonical threshold exists.
# 90 is the community default for paraphrase-resilient evidence matching.
FUZZY_MATCH_RATIO_THRESHOLD = 90


@dataclass(frozen=True)
class EvidenceValidationReport:
    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]   # (item, reason) tuples
    repairs: list[dict[str, Any]]   # offset corrections, fuzzy fallbacks


def validate_phase1_evidence(
    extracted: list[dict[str, Any]],
    grounded_chunks: list[dict[str, Any]],
    *,
    fuzzy_threshold: int = FUZZY_MATCH_RATIO_THRESHOLD,
) -> EvidenceValidationReport:
    """Validate every extracted phenotype against the grounded chunks.

    grounded_chunks is the list of {chunk_id, text, start_char, end_char}
    that phase 1 saw. The validator does not mutate inputs.
    """
    by_id = {int(c["chunk_id"]): c for c in grounded_chunks}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    for item in extracted:
        chunk_ids = [int(cid) for cid in item.get("chunk_ids") or []]
        if not chunk_ids or any(cid not in by_id for cid in chunk_ids):
            dropped.append({"item": item, "reason": "missing_chunk_id"})
            continue

        evidence = (item.get("evidence_text") or item.get("phrase") or "").strip()
        if not evidence:
            dropped.append({"item": item, "reason": "empty_evidence"})
            continue

        # Take the first referenced chunk as the substring search target.
        # Multi-chunk phrases are rare in practice; if present, concatenate.
        haystack = " ".join(by_id[cid]["text"] for cid in chunk_ids)
        if _normalized_in(evidence, haystack):
            # Repair offsets if missing or out of bounds.
            repaired = _repair_offsets(item, haystack)
            if repaired is not item:
                repairs.append({"item": item, "kind": "offset_repair"})
            kept.append(repaired)
            continue

        # Fuzzy fallback for paraphrased evidence the LLM may have emitted.
        ratio = fuzz.partial_ratio(evidence.lower(), haystack.lower())
        if ratio >= fuzzy_threshold:
            # Keep at chunk-level granularity, drop span offsets honestly.
            downgraded = {**item, "start_char": None, "end_char": None}
            repairs.append({"item": item, "kind": "fuzzy_downgrade",
                            "ratio": ratio})
            kept.append(downgraded)
            continue

        dropped.append({"item": item, "reason": "evidence_not_in_chunk",
                        "ratio": ratio})

    return EvidenceValidationReport(kept=kept, dropped=dropped, repairs=repairs)


def _normalized_in(needle: str, haystack: str) -> bool:
    n = " ".join(needle.lower().split())
    h = " ".join(haystack.lower().split())
    return n in h


def _repair_offsets(
    item: dict[str, Any], haystack: str
) -> dict[str, Any]:
    """If offsets are missing or out of bounds, regenerate from evidence text."""
    evidence = item.get("evidence_text") or item.get("phrase") or ""
    sc, ec = item.get("start_char"), item.get("end_char")
    if sc is None or ec is None or not (0 <= sc < ec <= len(haystack)) \
            or haystack[sc:ec].strip().lower() != evidence.strip().lower():
        # Locate via case-insensitive find on normalized whitespace.
        idx = haystack.lower().find(evidence.lower())
        if idx >= 0:
            return {**item, "start_char": idx, "end_char": idx + len(evidence)}
        # Could not anchor — drop offsets, keep chunk-level grounding.
        return {**item, "start_char": None, "end_char": None}
    return item
```

Wire into `phentrieve/llm/pipeline.py` between phase-1 dedup and
"actionable" filter (around line 336):

```diff
 extracted = self._deduplicate_phase1_extractions(
     _expand_combined_phase1_extractions(extracted)
 )
+report = validate_phase1_evidence(
+    extracted=extracted,
+    grounded_chunks=grounded_chunks,
+)
+extracted = report.kept
+trace["phase1_evidence_validation"] = {
+    "kept": len(report.kept),
+    "dropped": [(d["reason"], d["item"].get("phrase")) for d in report.dropped],
+    "repairs": [(r["kind"], r["item"].get("phrase")) for r in report.repairs],
+}
+phase_counts["evidence_dropped"] = len(report.dropped)
+phase_counts["evidence_repaired"] = len(report.repairs)
 actionable = [
     item for item in extracted
     if _normalize_category(str(item["category"])) in ACTIONABLE_CATEGORIES
 ]
```

**Why a separate Pydantic validator and not `min_length` on the schema**:
the research brief above confirms OpenAI strict mode silently drops
`minLength` and Gemini ignores unsupported keys; the only provider-agnostic
contract is a post-generation validator running server-side. Citing the
OpenAI structured-outputs limitations page and Anthropic's "Citations is
incompatible with Structured Outputs" callout.

### P1-B — Enriched mapping payload (wire the existing helper)

`phentrieve/llm/pipeline_phase2.py` — extend `compact_mapping_item` to
add `definition` (truncated) and a single `matched_synonym`. The HPO
database connection is already pooled by `details_enrichment.get_shared_database`.

```diff
 from phentrieve.retrieval.details_enrichment import enrich_results_with_details

 def compact_mapping_item(
     item: dict[str, Any],
     *,
     item_id: str | None = None,
+    enrich_candidates: bool = True,
+    definition_char_limit: int = 240,
 ) -> dict[str, Any]:
     ...
+    if enrich_candidates and item["candidates"]:
+        enriched = {
+            row["hpo_id"]: row
+            for row in enrich_results_with_details(
+                [{"hpo_id": c["hpo_id"], "label": c.get("term_name", "")}
+                 for c in item["candidates"]]
+            )
+        }
+    else:
+        enriched = {}

     for candidate in item["candidates"]:
         compact_candidate = {
             "id": candidate["hpo_id"],
             "term": candidate["term_name"],
             "retrieval_score": candidate.get("score"),
         }
+        details = enriched.get(candidate["hpo_id"], {}) or {}
+        if details.get("definition"):
+            d = details["definition"].strip()
+            compact_candidate["definition"] = (
+                d if len(d) <= definition_char_limit
+                else d[:definition_char_limit].rsplit(" ", 1)[0] + "..."
+            )
+        if details.get("synonyms"):
+            # Show the synonym closest to the candidate's matched_text
+            # if one exists; otherwise the first synonym.
+            mt = (candidate.get("matched_text") or "").lower()
+            best = next(
+                (s for s in details["synonyms"] if s.lower() == mt),
+                details["synonyms"][0],
+            )
+            compact_candidate["matched_synonym"] = best
         if candidate.get("retrieval_query"):
             compact_candidate["retrieval_query"] = candidate.get("retrieval_query")
         ...
```

And update the mapping prompt
`phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml` to (a) bump
to v4.2.0, (b) advertise the new fields, (c) replace the few-shot example
so the LLM sees the new shape:

```diff
 version: "v4.1.0"
+version: "v4.2.0"
...
   You receive:
   - phrase
   - category
   - primary_chunk_text
   - neighbor_chunk_texts
-  - candidates with id, term, retrieval_score
+  - candidates with id, term, retrieval_score, optional definition (truncated),
+    optional matched_synonym, optional matched_component, optional matched_text

   Rules:
   - Choose only one candidate HPO identifier.
-  - Use the grounded context to disambiguate.
+  - Use the grounded context AND definitions/synonyms to disambiguate.
+  - When two candidates have similar retrieval scores, the candidate whose
+    definition or matched_synonym is closest in meaning to the evidence wins.
+  - Prefer the most specific term supported by the evidence; if only a broad
+    ancestor is supported, choose it but the caller will label the result broad.
   - Treat retrieval_score as a hint, not the only decision rule.
   - Prefer semantic equivalence over lexical similarity.
   - If category is "normal", choose the abnormal HPO concept that is
     explicitly absent or normal in context.
   - If no candidate is a clear match, return null.
   - Never invent an HPO id outside the candidates list.
 ...
 few_shot_examples:
   - input: |
       Map the following JSON payload to the best HPO candidate.
       Return JSON only.

-      Payload:
-      {"primary_chunk_text": "The child has frequent falls while walking.",
-       "neighbor_chunk_texts": [], "phrase": "frequent falls",
-       "category": "abnormal",
-       "candidates": [
-         {"id": "HP:0002355", "term": "Difficulty walking",
-          "retrieval_score": 0.93},
-         {"id": "HP:0002317", "term": "Unsteady gait",
-          "retrieval_score": 0.67}
-       ]}
-     output: '{"phrase": "frequent falls", "hpo_id": "HP:0002355"}'
+      {"primary_chunk_text": "The child has frequent falls while walking.",
+       "neighbor_chunk_texts": [], "phrase": "frequent falls",
+       "category": "abnormal",
+       "candidates": [
+         {"id": "HP:0002355", "term": "Difficulty walking",
+          "retrieval_score": 0.93,
+          "definition": "Reduced ability to walk (ambulate).",
+          "matched_synonym": "Walking difficulty"},
+         {"id": "HP:0002359", "term": "Frequent falls",
+          "retrieval_score": 0.91,
+          "definition": "Increased frequency of falls relative to peers.",
+          "matched_synonym": "Frequent falls"}
+       ]}
+     output: '{"phrase": "frequent falls", "hpo_id": "HP:0002359"}'
```

Note the corrected expected output: the existing example mapped *"frequent
falls"* → `HP:0002355` *Difficulty walking*, which is at best a parent
concept. The literature-supported answer is `HP:0002359` *Frequent falls*
when that candidate exists. Few-shot demos must not teach a known
incorrect mapping.

**Token-growth measurement** is mandatory before merge — append a metric
`phase2_mapping_prompt_tokens_per_request` to the existing benchmark
output and gate the PR on growth ≤ +25%.

### P2-A — Evidence-aware aggregated confidence

`phentrieve/text_processing/_hpo_extraction_helpers.py` — replace the
`avg_score` ranking with a configurable weighted formula. Keep
`avg_score` in the output for backward compatibility.

```diff
 def aggregate_and_rank(
     evidence_map: dict[str, list[dict[str, Any]]],
     min_confidence_for_aggregated: float,
     hpo_synonyms_cache: dict[str, list[str]],
     hpo_definitions_cache: dict[str, str | None],
     include_details: bool,
+    *,
+    score_formula: Callable[[float, float, int], float] | None = None,
+    min_aggregate_for_filter: str = "avg",  # "avg" | "aggregate"
 ) -> list[dict[str, Any]]:
     ...
+    formula = score_formula or default_aggregate_score
     for hpo_id, evidence_list in evidence_map.items():
         ...
         max_score = max(e["score"] for e in evidence_list)
         avg_score = total_score / len(evidence_list)
-        if avg_score < min_confidence_for_aggregated:
-            continue
+        aggregate = formula(max_score, avg_score, len(evidence_list))
+        filter_value = avg_score if min_aggregate_for_filter == "avg" \
+            else aggregate
+        if filter_value < min_confidence_for_aggregated:
+            continue
         ...
         term: dict[str, Any] = {
             "id": hpo_id,
             "name": evidence_list[0]["name"],
             "score": max_score,
             ...
             "avg_score": avg_score,
-            "confidence": avg_score,
+            "aggregate_score": aggregate,
+            "confidence": aggregate,
             ...
         }
         ...
-    aggregated_list.sort(key=lambda x: (-x["avg_score"], -x["count"]))
+    aggregated_list.sort(key=lambda x: (-x["confidence"], -x["count"]))
```

```python
# Default formula matches the report §4 recommendation but is overridable.
def default_aggregate_score(
    max_score: float, avg_score: float, evidence_count: int
) -> float:
    import math
    count_bonus = math.log1p(evidence_count) / math.log(10)  # ~0..1.5 for 1..30
    return 0.55 * max_score + 0.25 * avg_score + 0.10 * min(count_bonus, 1.0)
    # Reserved 0.10 for ontology_coherence (added in a later phase).
```

**Calibration step**: once this lands, fit `IsotonicRegression` on
`(aggregate, is_correct_top1)` over the 570-term German benchmark; persist
to `data/calibration/<model>__aggregate.pkl`; load lazily in the API
response builder to populate a separate `display_confidence` field.
Reference: arXiv 2601.16907 confirms isotonic preserves ranking metrics.

### P2-B — Extract `phase2a.py` from `pipeline.py`

Mechanical refactor; no behavior change. Move `_retrieve_candidates`,
`_build_grounded_context`, `_hybrid_select_candidates`, `_extract_first_result_list`
and the per-call dedupe key helpers into a new
`phentrieve/llm/phase2a.py`. Use the existing `pipeline_phase2.py` patterns
(module-level pure functions). Keep `TwoPhaseLLMPipeline.run` thin.

Acceptance: `pipeline.py` LOC drops from ~1700 to ~1300; no test changes.

### P3-A — Expand `PHENOTYPE_ABBREVIATIONS`

Move the dict to a data file so it can be edited without code changes:

```python
# phentrieve/llm/pipeline_phase1.py  (diff)
-PHENOTYPE_ABBREVIATIONS = {"xlid": "X-linked intellectual disability"}
+from phentrieve.llm.resources import load_phenotype_abbreviations
+PHENOTYPE_ABBREVIATIONS = load_phenotype_abbreviations()
```

```python
# phentrieve/llm/resources.py  (NEW)
import json
from importlib.resources import files

def load_phenotype_abbreviations() -> dict[str, str]:
    """Curated clinical abbreviations -> canonical phrase. Editable JSON."""
    raw = files("phentrieve.llm").joinpath(
        "default_lang_resources/phenotype_abbreviations_en.json"
    ).read_text(encoding="utf-8")
    return {k.lower(): v for k, v in json.loads(raw).items()}
```

```json
// phentrieve/llm/default_lang_resources/phenotype_abbreviations_en.json
{
  "xlid": "X-linked intellectual disability",
  "id":   "intellectual disability",
  "dd":   "developmental delay",
  "asd":  "autism spectrum disorder",
  "vsd":  "ventricular septal defect",
  "pda":  "patent ductus arteriosus",
  "dcm":  "dilated cardiomyopathy",
  "hcm":  "hypertrophic cardiomyopathy",
  "scd":  "sudden cardiac death",
  "fh":   "family history",
  "pws":  "Prader-Willi syndrome",
  "scid": "severe combined immunodeficiency",
  "cmt":  "Charcot-Marie-Tooth disease",
  "opmd": "oculopharyngeal muscular dystrophy",
  "hsp":  "hereditary spastic paraplegia"
}
```

Add a per-language file for German next (`phenotype_abbreviations_de.json`).
Benchmark before/after on the single-word subsets.

### P3-B — Centralize the security preamble

`phentrieve/llm/prompts/loader.py` — wrap the YAML system_prompt at render
time so the security block lives in `safety.py`:

```diff
 def render_system_prompt(self, *, language: str | None = None) -> str:
     base = self._template.format(language=language or self.language)
+    if self.inject_safety:
+        from phentrieve.llm.prompts.safety import build_safety_preamble
+        base = build_safety_preamble() + "\n\n" + base
     return base
```

Strip the duplicated security block from each YAML in
`phentrieve/llm/prompts/templates/**/*.yaml`. Add a unit test that
`render_system_prompt()` always begins with the safety preamble.

### Bonus — Hybrid lexical+dense candidate generator (foundation for §2)

Once P0-A lands, add a second channel using `rank_bm25` over HPO label +
synonym strings, fuse with dense via RRF. Recommended by RRF Cormack 2009
and adopted by OpenSearch / Azure AI Search / LangChain (citations in the
research brief above).

```python
# phentrieve/retrieval/lexical_channel.py  (NEW)
from rank_bm25 import BM25Okapi
import re

_TOKEN = re.compile(r"\w+", re.UNICODE)

class HPOLexicalIndex:
    def __init__(self, hpo_terms: list[dict]):
        # Each (label OR synonym) becomes one document keyed by hpo_id.
        self.docs: list[tuple[str, str]] = []
        for term in hpo_terms:
            for text in [term["label"]] + list(term.get("synonyms", [])):
                if text:
                    self.docs.append((term["id"], text))
        self._bm25 = BM25Okapi([self._tok(d[1]) for d in self.docs])

    @staticmethod
    def _tok(s: str) -> list[str]:
        return [t.lower() for t in _TOKEN.findall(s)]

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(self._tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        seen, ranked = set(), []
        for i in order:
            hpo_id, _text = self.docs[i]
            if hpo_id in seen:
                continue
            seen.add(hpo_id)
            ranked.append((hpo_id, float(scores[i])))
        return ranked


# phentrieve/retrieval/fusion.py  (NEW)
def reciprocal_rank_fusion(
    rank_lists: list[list[str]], *, k: int = 60
) -> list[tuple[str, float]]:
    """RRF over multiple ranked HPO-id lists. Returns fused ranking.

    Cormack, Clarke & Buettcher (SIGIR 2009). k=60 is the canonical
    default; optimum is flat over [20, 100].
    """
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank_idx, hpo_id in enumerate(ranked, start=1):
            scores[hpo_id] = scores.get(hpo_id, 0.0) + 1.0 / (k + rank_idx)
    return sorted(scores.items(), key=lambda kv: -kv[1])
```

Wire into `_retrieve_candidates` behind a config flag
`enable_lexical_channel`. Benchmark on the German subset first — RRF can
hurt MRR on already-strong dense retrievers; only ship it if the German
single-word benchmark improves Hit@10 measurably.

### Bonus — Optional cross-encoder reranker (DiscHPO pattern)

Use the DiscHPO-validated `cross-encoder/ms-marco-electra-base`. Latency
fits Phentrieve's budget: ~50-120 ms for top-20 pairs (research brief).

```python
# phentrieve/retrieval/cross_encoder.py  (NEW)
from functools import lru_cache
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-electra-base"  # DiscHPO winner


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str = DEFAULT_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank_candidates(
    *,
    phrase: str,
    evidence_text: str,
    candidates: list[dict],     # [{hpo_id, term_name, score, ...}]
    top_k: int = 10,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    if not candidates:
        return candidates
    enc = get_cross_encoder(model_name)
    # Use phrase + evidence as query, candidate label + matched_synonym as doc.
    query = f"{phrase} | {evidence_text}".strip(" |")
    docs = [
        f"{c['term_name']} | {c.get('matched_synonym', '')}".strip(" |")
        for c in candidates
    ]
    ranked = enc.rank(query, docs, top_k=top_k, return_documents=False)
    return [
        {**candidates[r["corpus_id"]], "rerank_score": float(r["score"])}
        for r in ranked
    ]
```

**Important**: DiscHPO uses MS-MARCO-electra which is a *generic* relevance
ranker, not HPO-tuned. PhenoRerank's BlueBERT-on-HPO model would in
principle outperform; if/when you publish a re-trained reranker, swap
`DEFAULT_MODEL` only. The interface stays stable.

## Revised priority matrix (supersedes original Roadmap ordering)

| Prio | Action                                              | Why now                                                    | Effort     | Risk    |
| ---- | --------------------------------------------------- | ---------------------------------------------------------- | ---------- | ------- |
| P0   | Make `query_batch` index-type-aware (P0-A diff)     | README advertises +28% MRR; full-text silently doesn't use it. Affects 2 call sites. | 2-3 days   | Low     |
| P0   | Rewrite phase-1 few-shot examples (P0-B diff)       | Demos override system rules; pins §7 of original report.   | 1 day      | Low     |
| P1   | Phase-1 evidence validator (P1-A diff)              | Frontend lies to users when LLM hallucinates spans.        | 2-3 days   | Low     |
| P1   | Enriched mapping payload (P1-B diff)                | Helper already exists; not wired in.                       | 1-2 days   | Medium  |
| P2   | Evidence-aware aggregate confidence (P2-A diff)     | Locks in P0 gain.                                          | 1 day      | Medium  |
| P2   | Refactor `phase2a.py` out of `pipeline.py` (P2-B)   | Pre-req for safely doing P1+P2.                            | 2 days     | Low     |
| P3   | Expand `PHENOTYPE_ABBREVIATIONS` (P3-A)             | Cheap recall win on German clinical text.                  | 1-2 days   | Low     |
| P3   | Centralize safety preamble (P3-B)                   | Reduces drift before adding more prompt variants.          | 0.5 days   | Low     |
| Bonus| Hybrid lexical+dense via RRF                        | Foundation for original report §2.                         | 3-5 days   | Medium  |
| Bonus| Cross-encoder reranker (DiscHPO model)              | Foundation for original report §3 (proper variant).        | 3-5 days   | Medium  |
| DEFER | LLM-as-judge reranker                              | Doubles cost; cross-encoder is the right tool.             | —          | —       |

## Bottom Line (Update)

The original report's diagnoses hold. The strongest first PR remains:
**P0-A + P0-B together** — both are small, low-risk, file-scoped, and
unblock everything else. They are also the two changes most likely to be
*invisible to users until they regress*, so they are the highest-priority
tests-and-benchmarks targets.

Do not implement P1+ inside the current 1700-line `pipeline.py`. Land
P2-B (the mechanical refactor) before P1 so each feature lands with a
reviewable footprint.

For hybrid retrieval and cross-encoder reranking (the bigger bets), the
literature converges on a specific stack: BM25 + dense fused via RRF
(`k=60`), top-20 reranked with `cross-encoder/ms-marco-electra-base`
(DiscHPO's published choice), BioLORD-2023-M kept as the bi-encoder.
There is currently no published *multilingual* clinical encoder that
beats BioLORD-2023-M for German+English term-level normalization.

## Update Source Index (verified 2026-05-24)

- Anthropic Claude 4 best-practices: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices
- Anthropic Citations API: https://platform.claude.com/docs/en/build-with-claude/citations
- OpenAI Structured Outputs (strict-mode limits): https://developers.openai.com/api/docs/guides/structured-outputs
- Gemini Structured Outputs (OpenAPI v3.0.3 subset): https://ai.google.dev/gemini-api/docs/structured-output
- RAGAS Faithfulness metric: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- Few-shot format-dominance (Min et al. 2022 summary): https://www.promptingguide.ai/techniques/fewshot
- DiscHPO (cross-encoder choice + F1): https://medinform.jmir.org/2025/1/e68558 — PMID 41191926
- PhenoRerank: https://pmc.ncbi.nlm.nih.gov/articles/PMC11040548/ — PMID 35351638
- BioLORD-2023 (JAMIA 2024): https://academic.oup.com/jamia/article/31/9/1844/7614965
- sentence-transformers retrieve & rerank: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- LangChain EnsembleRetriever (RRF): https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html
- RRF practical guide (k=60 default): https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/
- Carbonell & Goldstein 1998 (MMR): https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
- Calibrated similarity / isotonic regression (2026): https://arxiv.org/html/2601.16907
- ChromaDB query semantics: https://docs.trychroma.com/docs/querying-collections/query-and-get
- Vespa long-context ColBERT (per-window MaxSim): https://blog.vespa.ai/announcing-long-context-colbert-in-vespa/

## Files referenced in this update (live tree at `ed4b266`)

- `phentrieve/text_processing/hpo_extraction_orchestrator.py`
- `phentrieve/text_processing/_hpo_extraction_helpers.py`
- `phentrieve/retrieval/dense_retriever.py`
- `phentrieve/retrieval/aggregation.py`
- `phentrieve/retrieval/details_enrichment.py`
- `phentrieve/llm/pipeline.py`
- `phentrieve/llm/pipeline_phase1.py`
- `phentrieve/llm/pipeline_phase2.py`
- `phentrieve/llm/tools.py`
- `phentrieve/llm/types.py`
- `phentrieve/llm/prompts/safety.py`
- `phentrieve/llm/prompts/templates/two_phase/en.yaml`
- `phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml`
