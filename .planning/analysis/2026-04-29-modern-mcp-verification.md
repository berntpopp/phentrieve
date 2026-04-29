# Modern MCP Verification - 2026-04-29

Branch: `feat/modern-mcp-implementation`

## Targeted MCP Checks

- `uv run --extra mcp pytest tests/unit/mcp tests/integration/test_mcp_http_protocol.py -q --no-cov`
  - Passed: `24 passed`
- `uv run --extra mcp pytest tests/unit/mcp tests/integration/test_mcp_http_protocol.py -q`
  - Test assertions passed: `24 passed`
  - Command exit: failed on global coverage threshold for narrow selection:
    `total of 18 is less than fail-under=40`

## API Schema and Router Checks

- `uv run pytest tests/unit/api/test_schemas.py tests/unit/api/test_text_processing_router.py -q`
  - Test assertions passed: `61 passed`
  - Command exit: failed on global coverage threshold for narrow selection:
    `total of 18 is less than fail-under=40`
- `uv run pytest tests/unit/api/test_schemas.py tests/unit/api/test_text_processing_router.py -q --no-cov`
  - Passed: `61 passed`

## Required Repository Checks

- `make check`
  - Passed: Ruff format reported unchanged files; Ruff check passed.
- `make typecheck-fast`
  - Initial run found typed facade helper issues in `api/mcp/facade.py`.
  - Fixed in commit `441824c` (`fix: type explicit mcp facade helpers`).
  - Rerun passed: `Success: no issues found in 125 source files`.
- `make test`
  - Initial run failed due local environment:
    - `openpyxl` missing from the active environment for script tests.
    - CUDA out-of-memory while xdist workers loaded embedding models on GPU.
  - Remediation:
    - `uv sync --all-extras`
    - `CUDA_VISIBLE_DEVICES='' make test`
  - Rerun passed: `1507 passed, 44 skipped`
  - Coverage passed: `71.63%`, above required `40%`.

## Manual MCP HTTP Smoke

- Started server:
  - `make mcp-serve-http`
  - Server reported: `http://127.0.0.1:8734/mcp`
- Attempted plan command:
  - `uv run --extra mcp mcp dev http://127.0.0.1:8734/mcp`
  - Result: installed MCP CLI `mcp dev` expects a local Python file, not a remote HTTP URL.
- Verified Streamable HTTP directly with JSON-RPC POST calls:
  - `initialize`: `200`, server name `phentrieve`
  - `notifications/initialized`: `202`
  - `tools/list`: `200`
    - `phentrieve.extract_hpo_terms`
    - `phentrieve.extract_hpo_terms_llm`
    - `phentrieve.search_hpo_terms`
    - `phentrieve.compare_hpo_terms`
    - `phentrieve.get_server_capabilities`
  - `resources/list`: `200`
    - `phentrieve://capabilities`
    - `phentrieve://hpo/languages`
    - `phentrieve://hpo/extraction-profiles`
    - `phentrieve://compliance/research-use`
  - `prompts/list`: `200`
    - `annotate_research_text`
    - `review_hpo_research_annotations`
    - `extract_research_case_phenotypes`
  - `tools/call` for `phentrieve.get_server_capabilities`: `200`
    - Returned extraction backends: `standard`, `llm`

## Posture Checks

- MCP tool, resource, prompt, and docs language uses research-only framing.
- No MCP surface claims diagnosis, treatment, triage, patient management, or
  clinical decision support suitability.
- Public demo guidance warns against identifiable patient data.
- HTTP docs describe Streamable HTTP at `/mcp` and warn against legacy SSE URLs.
