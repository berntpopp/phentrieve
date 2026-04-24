# Analysis Audit — 2026-04-23

**Scope:** remaining `.planning/analysis/` documents checked against the
current `main` worktree and reachable git history.

## Archived

- `2026-04-17-llm-multi-provider-adoption-research.md`
  - Archived because it is pre-implementation research for a capability now
    present on `main`.
  - Current code includes `OpenAIStructuredOutputProvider`,
    `AnthropicStructuredOutputProvider`, `OllamaStructuredOutputProvider`,
    pricing config, structured retries, and related benchmark support.
  - Merge evidence: `5a5b0b2`.

- `UNIFIED-OUTPUT-FORMAT-PHENOPACKETS-REVIEW.md`
  - Archived because it reviews an older unified-output-format design that is
    already archived and no longer matches the implemented Phenopacket export
    path on `main`.
  - The shipped path is better represented by the current Phenopacket code and
    by `PR-113-ISSUE-87-VERIFICATION-REPORT.md` plus later sidecar work.

## Kept In `analysis/`

- `2026-04-16-llm-lean-v1-comparative-review.md`
  - Kept as durable review history. Several recommendations remain relevant to
    current LLM pipeline evolution even though some items have since landed.

- `2026-04-16-llm-pipeline-expert-review.md`
  - Kept as durable review history for PR `#216`; it captures branch-era
    reasoning that still explains why the current grounded pipeline evolved the
    way it did.

- `2026-04-23-planning-audit.md`
  - Kept as current lifecycle audit record.

- `CODE-QUALITY-REVIEW-2026-04-09.md`
  - Kept because it is already a post-execution consolidated review and remains
    the best high-level record of the code-quality program and its results.

- `CODEBASE-MAINTAINABILITY-REVIEW-2026-04-15.md`
  - Kept because multiple findings still map to current code, including large
    modules and maintainability hotspots.

- `EXTRACTION-BENCHMARK-DEEP-ANALYSIS.md`
  - Kept as benchmark interpretation history; it documents retrieval-vs-ontology
    mismatch findings rather than a stale implementation checklist.

- `HPO-TERM-DETAILS-CRITICAL-REVIEW.md`
  - Kept because several findings were implemented and the file still serves as
    rationale for the resulting DRY cleanup in `hpo_database.py` and
    `details_enrichment.py`.

- `NEGATION-DETECTION-ANALYSIS.md`
  - Kept because the negation cues and German edge-case analysis still matches
    current assertion-detection behavior and historical fixes.

- `PR-113-ISSUE-87-VERIFICATION-REPORT.md`
  - Kept as durable verification history for the original Phenopacket export.

- `RERANKING-DIAGNOSIS-AND-FIX.md`
  - Kept because reranking is still present in the standard retrieval path, so
    the diagnosis is still potentially relevant.

- `pr-229-deep-review-report.md`
  - Kept as recent PR review history. Some concerns were fixed, and the
    remaining size/modularization concerns still partly apply.
