# MCP Server Integration

Phentrieve exposes a Streamable HTTP MCP endpoint for research-use Human
Phenotype Ontology (HPO) annotation workflows. The endpoint is intended for
Claude, ChatGPT developer mode, and other current remote MCP clients.

Phentrieve MCP outputs are algorithmic research suggestions only. They are not
for diagnosis, treatment, triage, patient management, or clinical decision
support. Do not submit identifiable patient data to public demo instances.

Public MCP LLM extraction uses the same server-owned LLM target as public REST:
`gemini/gemini-3.1-flash-lite-preview`. MCP clients cannot select
`llm_model`, `llm_provider`, or `llm_base_url`; those settings are intentionally
server-controlled.

## Transport

| Mode | Endpoint | Status | Use Case |
|------|----------|--------|----------|
| Streamable HTTP | `/mcp` | Recommended | Claude, ChatGPT developer mode, remote MCP clients |

Legacy SSE endpoints are not supported. Use the single Streamable HTTP URL:

```text
https://your-domain.example/mcp
```

## ChatGPT Developer Mode

Create a custom app from the remote MCP server:

```text
https://your-domain.example/mcp
```

Use no authentication only for local or private deployments. For public
deployments, put the endpoint behind OAuth or an authenticated reverse proxy.
Phentrieve's current tools are read-only and annotated as such.

## Claude Code HTTP

```bash
claude mcp add --transport http phentrieve https://your-domain.example/mcp
```

For local development:

```bash
make mcp-serve-http
claude mcp add --transport http phentrieve http://127.0.0.1:8734/mcp
```

## Available MCP Tools

Tools are namespaced with an underscore prefix (`phentrieve_*`).

| Tool | Use When |
|------|----------|
| `phentrieve_search_hpo_terms` | Candidate HPO terms for a short phrase |
| `phentrieve_extract_hpo_terms` | Deterministic retrieval-backed HPO term suggestions for research text |
| `phentrieve_extract_hpo_terms_llm` | LLM-assisted full-text research annotation and grounded HPO mapping suggestions |
| `phentrieve_compare_hpo_terms` | Similarity between two HPO IDs |
| `phentrieve_export_phenopacket` | Serialize an annotation set to a GA4GH Phenopacket v2 bundle |
| `phentrieve_chunk_text` | Chunk text without retrieval (client-driven loops) |
| `phentrieve_get_capabilities` | Discover tools, response modes, limits, error codes, and the citation contract |
| `phentrieve_diagnostics` | Subsystem health and recent (sanitized) errors |

All tools are read-only from the perspective of user data and are annotated
`readOnlyHint=true`. Tool descriptions carry the research-only limitation and
public demo data warning.

## Response Envelope and Token Efficiency

Every tool returns a structured JSON envelope:

- **Success:** `{"success": true, <domain keys>, "_meta": {...}}`
- **Error:** `{"success": false, "error_code", "message", "retryable", "recovery_action", "_meta": {...}}`

`_meta` always carries `tool`, `request_id`, `elapsed_ms`,
`capabilities_version`, `unsafe_for_clinical_use: true`, and `next_commands`
(ready-to-call `{tool, arguments}` workflow hints). Error codes are stable
strings (`invalid_input`, `validation_failed`, `not_found`, `ambiguous_query`,
`llm_quota_exhausted`, `llm_unavailable`, `upstream_unavailable`,
`temporarily_unavailable`, `internal_error`).

Pass `response_mode` (`minimal | compact | standard | full`, default `compact`)
on any tool to control token cost. Over-budget list results are truncated and
report `_meta.truncated`. Common argument synonyms (e.g. `query`/`phrase` →
`text`, `limit` → `num_results`) are accepted and disclosed via
`_meta.argument_aliases_applied`. Call `phentrieve_get_capabilities` once, then
compare the returned `capabilities_version` to the value echoed in `_meta` to
skip re-fetching when unchanged.

## MCP Resources

| Resource | Contents |
|----------|----------|
| `phentrieve://schema/overview` | Server overview, envelope, verbosity, and safety (markdown) |
| `phentrieve://schema/tool-guide` | Per-tool usage guide with sample calls (markdown) |
| `phentrieve://capabilities` | Full capability descriptor with `capabilities_version` (JSON) |
| `phentrieve://hpo/languages` | Supported language codes |
| `phentrieve://hpo/extraction-profiles` | Standard and LLM extraction profile guidance |
| `phentrieve://compliance/research-use` | Intended use, non-intended uses, and public demo data notice |

## MCP Prompts

| Prompt | Purpose |
|--------|---------|
| `annotate_research_text` | Map supplied research text to HPO suggestions |
| `review_hpo_research_annotations` | Review Phentrieve-returned HPO annotations against text evidence |
| `extract_research_case_phenotypes` | Extract phenotype suggestions from synthetic or research-consented case-report-like text |

These prompts are short workflow templates. They do not expose benchmarking
comparison workflows as routine annotation prompts.

## Local Development

```bash
# Install MCP dependencies
make mcp-install

# Start the Streamable HTTP MCP endpoint
make mcp-serve-http

# View MCP configuration
make mcp-info
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MCP_HTTP` | `false` local, `true` Docker | Mount MCP at `/mcp` on the main API |
| `PHENTRIEVE_MCP_ENABLE_HTTP` | `false` | Alternative env var for enabling HTTP |
| `PHENTRIEVE_MCP_NAME` | `phentrieve` | MCP server name shown to clients |
| `PHENTRIEVE_MCP_HOST` | `127.0.0.1` | Host for standalone HTTP mode |
| `PHENTRIEVE_MCP_PORT` | `8734` | Port for standalone HTTP mode |

The public MCP tool schema does not expose provider, model, or base URL
selection for LLM extraction. Deployments that need different public LLM
behavior must change the server policy rather than accepting client-supplied
`llm_model`, `llm_provider`, or `llm_base_url` values.

## CLI Commands

```bash
# Start MCP server over Streamable HTTP (stdio has been removed)
phentrieve mcp serve --port 8734

# Display MCP configuration and tools
phentrieve mcp info
```

## Deployment

The default `docker-compose.yml` enables MCP HTTP:

```yaml
services:
  phentrieve_api:
    environment:
      - ENABLE_MCP_HTTP=true
```

To disable MCP in Docker:

```bash
ENABLE_MCP_HTTP=false docker-compose up
```

## Nginx Reverse Proxy

Proxy `/mcp` to the API container:

```nginx
location /mcp {
    proxy_pass http://phentrieve_api:8000/mcp;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

## Troubleshooting

### MCP not available in Docker

Verify the environment variable is set:

```bash
docker-compose exec phentrieve_api printenv | grep MCP
```

Check API logs for the MCP mount message:

```bash
docker-compose logs phentrieve_api | grep -i mcp
```

### Dependencies not installed locally

Install MCP optional dependencies:

```bash
uv sync --extra mcp
```

### Client cannot initialize

Confirm the client is configured for Streamable HTTP at `/mcp`, not a legacy
SSE URL such as `/sse` or `/mcp/messages/`.

## Security Considerations

- Put public MCP deployments behind OAuth or an authenticated reverse proxy.
- Do not submit identifiable patient data to public demo instances.
- Treat all submitted text and retrieval payloads as untrusted data. Phentrieve
  uses prompt boundaries, a server-owned LLM target, structured output, and
  backend validation as defense-in-depth prompt-injection controls.
- The current MCP tools are read-only and intended for research, benchmarking,
  education, and research data standardisation only.
- Consider rate limiting MCP endpoints to prevent abuse.
