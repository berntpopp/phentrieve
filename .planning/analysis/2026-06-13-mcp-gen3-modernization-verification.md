# MCP Gen-3 Modernization — Verification

- Date: 2026-06-13
- Branch: `feat/mcp-gen3-modernization`
- Spec: `specs/2026-06-13-mcp-gen3-modernization-design.md`
- Plan: `completed/2026-06-13-mcp-gen3-modernization-plan.md`

## Outcome

`api/mcp/` rebuilt on the standalone FastMCP v3 framework to match the
maintainer's `*-link` MCP house style. HTTP-only transport; the broken stdio
entry point and the legacy `fastapi-mcp` bridge were removed.

## Delivered (vs. spec)

- Transport: Streamable HTTP only (`http_app` + combined lifespan); stdio,
  `cli.py`, `metadata.py`, and the `phentrieve-mcp` script removed; `fastapi-mcp`
  replaced by `fastmcp>=3.2`. (spec S3, S16)
- Foundation modules: `envelope` (run_mcp_tool + Family B + error model + ring),
  `shaping` (response_mode + budgets), `capabilities` (sha256 version),
  `annotations`, `next_commands`, `schemas`, `arg_help`, `middleware`. (S4-S8, S14)
- 8 flat-arg tools: search, extract, extract_llm, compare, **export_phenopacket
  (new)**, **chunk_text (new)**, get_capabilities, **diagnostics (new)**. (S9, S10)
- Resources: markdown `schema/overview` + `tool-guide`, JSON capabilities,
  languages, extraction-profiles, research-use. Server instructions + 3 prompts. (S11)
- Observability (Mid): request_id + elapsed_ms in `_meta`, structured logging,
  error ring surfaced by diagnostics. (S12)
- Safety: `unsafe_for_clinical_use` in every `_meta`, research-use framing,
  research-ack parity for extraction tools in hosted mode. (S13)

## Verification

| Gate | Result |
|------|--------|
| `make test` | 1730 passed, 43 skipped |
| `make typecheck-fast` | clean (157 files) |
| `make check` (format + lint) | clean |
| `make security-python` | 0 High / 0 Medium / 6 Low (all pre-existing; none in api/mcp) |
| `make ci-python-quality` | PASS, coverage 75.17% |
| New MCP tests | 62 passed (`tests/unit/mcp_server/` + integration) |
| Functional | 8 tools listed, Family B envelopes, not_found path, arg-validation, alias normalization, `mcp info` CLI, in-memory protocol round-trip |

## Notes / follow-ups

- Non-goals deferred: Prometheus/OTel, OAuth, ChatGPT search/fetch shim.
- Test dir renamed `tests/unit/mcp` -> `tests/unit/mcp_server` (the `mcp` name
  shadowed the SDK package under pytest prepend import mode).
- Python compat matrix (3.12/3.13) not re-run locally; default-interpreter PR
  gate passed and fastmcp 3.4 supports 3.10-3.13 (siblings run 3.12).
