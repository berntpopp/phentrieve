# Phentrieve MCP — Tool Guide

All tools are read-only and accept `response_mode` (`minimal | compact |
standard | full`, default `compact`).

## phentrieve_search_hpo_terms
Map a short phenotype phrase to ranked HPO candidates by dense retrieval. Use for
single phrases, not documents.
`phentrieve_search_hpo_terms(text="muscle weakness", num_results=10)`

## phentrieve_extract_hpo_terms
Deterministic retrieval-backed extraction from multi-sentence text (chunking +
assertion detection + aggregation). No LLM calls.
`phentrieve_extract_hpo_terms(text="The patient had seizures and ataxia.")`

## phentrieve_extract_hpo_terms_llm
LLM-assisted two-phase extraction for full abstracts, publication-style
annotation, and syndrome/eponym-heavy text. Uses only the server-configured LLM
target; clients cannot override provider/model. Subject to a daily quota in
hosted mode; set `allow_standard_fallback=true` to fall back to deterministic
extraction when the quota is exhausted.

## phentrieve_compare_hpo_terms
Ontology semantic similarity between two HPO ids (`formula`: `hybrid` or
`simple_resnik_like`). A missing id returns a `not_found` error envelope.
`phentrieve_compare_hpo_terms(term1_id="HP:0001250", term2_id="HP:0002133")`

## phentrieve_export_phenopacket
Serialize an annotation set to a GA4GH Phenopacket v2 JSON bundle. Hand it the
`aggregated_hpo_terms` from an extract call.
`phentrieve_export_phenopacket(case_id="C1", phenotypes=[{"hpo_id": "HP:0001250", "label": "Seizure", "assertion": "affirmed"}])`

## phentrieve_chunk_text
Chunk text without retrieval, for clients driving their own loop. Defaults to the
`simple` (paragraph + sentence) strategy.
`phentrieve_chunk_text(text="A. B. C.", strategy="simple")`

## phentrieve_get_capabilities
Server capabilities, limits, response modes, error codes, and citation contract,
with a stable `capabilities_version`. Pass `details=["sample_calls"]` to expand.

## phentrieve_diagnostics
Subsystem health (ontology data, embedding model, LLM backend, vector index) and
recent sanitized errors for troubleshooting.
