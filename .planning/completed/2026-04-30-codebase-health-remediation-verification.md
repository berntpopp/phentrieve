# Codebase Health Remediation Verification

Date: 2026-04-30

Branch: `codebase-health-remediation`

## Commit Range Reviewed

Full remediation task range:

- Base plan commit: `219d0c0`
- Verified head: `b70d8d1`
- Range: `219d0c0..b70d8d1`

Commits in the verified range:

- `24694fd` fix(api): centralize retrieval model policy
- `79ed491` fix(text): honor assertion detector preference
- `cf2e1a3` fix(frontend): avoid persisting raw clinical text by default
- `5acba8c` refactor(text): use focused HPO extraction helpers
- `605677f` refactor(api): move text processing logic out of router
- `97f09f4` fix(api): avoid blocking retriever initialization in async path
- `2d9f645` fix(llm): apply shared prompt injection safeguards
- `95cd528` refactor(frontend): split query interface responsibilities
- `821e875` fix(frontend): remove validator side effects and dead trust flag
- `5893f61` refactor(llm): split two phase pipeline internals
- `4d6090e` refactor(llm): split provider implementations
- `a1bfc81` test: align expectations with refactored service boundaries
- `b70d8d1` chore(docs): align setup docs with packaging metadata

## Commands Run

Focused checks during the final tasks:

- `uv run pytest tests/unit/api/test_text_processing_router_performance.py tests/unit/core/test_embeddings_real.py tests/integration/test_adaptive_rechunking_api.py -q --no-cov`
  - Passed: 20 passed.
- `make test`
  - Passed after aligning stale test patch targets and model-trust expectations:
    1648 passed, 35 skipped, coverage 75.88%.
- `uv run pytest tests/unit/test_project_metadata_consistency.py -q`
  - Failed before remediation as expected on stale `text_processing` Makefile extra
    and stale `--extra text` docs reference. The focused run also hits the
    repository-wide coverage threshold when run alone.
- `uv run pytest tests/unit/test_project_metadata_consistency.py -q --no-cov`
  - Passed after remediation: 2 passed.
- `make check`
  - Passed during Task 12 after formatting the metadata consistency test.

Final required verification:

- `make check`
  - Passed: Ruff format left 296 files unchanged; Ruff check reported all
    checks passed.
- `make typecheck-fast`
  - Passed: `dmypy` crashed once, the Makefile restarted it, and the fresh run
    reported `Success: no issues found in 145 source files`.
- `make test`
  - Passed: 1650 passed, 35 skipped, required 40% coverage threshold met with
    total coverage 75.88%.
- `make frontend-test-ci`
  - Passed: 31 frontend test files, 314 tests passed.
- `make frontend-build-ci`
  - Passed: Vite production build completed, 841 modules transformed.

## Remaining Known Risk

- The pytest coverage gate remains at 40%, so it is useful as a regression
  floor but not as a high-confidence coverage target.
- Slow and E2E tests are still excluded from the default `make test` path.
- The largest LLM, text-processing, and frontend modules are now split behind
  compatibility facades or focused subcomponents, but some files remain large
  enough to justify continued incremental decomposition.
- The API README still documents existing HPO setup commands rather than
  validating that setup workflow end to end.

## Updated Estimated Scorecard

| Area | Previous | Updated Estimate |
| --- | ---: | ---: |
| Architecture/modularization | 6.4 | 8.0 |
| Python core maintainability | 5.8 | 7.6 |
| API/backend design | 7.0 | 8.2 |
| Frontend maintainability | 6.4 | 7.8 |
| Security/privacy/safety | 6.8 | 8.4 |
| RAG/HPO extraction quality architecture | 6.2 | 7.7 |
| Tests/CI signal | 7.2 | 8.3 |
| Packaging/config/docs/developer experience | 6.3 | 8.0 |

Overall updated estimate: `8.0/10`.
