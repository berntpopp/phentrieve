# MCP Server Integration

Phentrieve supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) to expose HPO term extraction capabilities to LLM clients like Claude Desktop.

## Transport Modes

| Mode | URL | Use Case |
|------|-----|----------|
| **stdio** | N/A (command-based) | Claude Desktop integration |
| **HTTP (same-domain)** | `https://phentrieve.example.com/mcp` | Production Docker deployment |
| **HTTP (standalone)** | `http://localhost:8735/mcp` | Local development/testing |

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `query_hpo_terms` | Search HPO terms by semantic similarity to clinical text |
| `process_clinical_text` | Extract HPO terms from clinical notes with chunking and assertion detection |
| `calculate_term_similarity` | Calculate semantic similarity between two HPO terms |

## Quick Start

### Claude Desktop (stdio)

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "phentrieve": {
      "command": "phentrieve",
      "args": ["mcp", "serve"],
      "env": {
        "PHENTRIEVE_DATA_ROOT_DIR": "/path/to/phentrieve/data"
      }
    }
  }
}
```

### Production Docker (same-domain HTTP)

The Docker image ships with MCP support enabled by default. The `/mcp` endpoint is available on the same domain as the API:

```
https://phentrieve.example.com/mcp
```

**Environment variable** (optional, enabled by default in Docker):
```bash
ENABLE_MCP_HTTP=true
```

### Local Development

```bash
# Install MCP dependencies
make mcp-install

# Option 1: stdio mode (for Claude Desktop)
make mcp-serve

# Option 2: HTTP mode (standalone)
make mcp-serve-http

# View MCP configuration
make mcp-info
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MCP_HTTP` | `false` (local) / `true` (Docker) | Mount MCP at `/mcp` on main API |
| `PHENTRIEVE_MCP_ENABLE_HTTP` | `false` | Alternative env var for enabling HTTP |
| `PHENTRIEVE_MCP_NAME` | `phentrieve` | MCP server name shown to clients |
| `PHENTRIEVE_MCP_HOST` | `127.0.0.1` | Host for standalone HTTP mode |
| `PHENTRIEVE_MCP_PORT` | `8734` | Port for standalone HTTP mode |

### CLI Commands

```bash
# Start MCP server (stdio mode, for Claude Desktop)
phentrieve mcp serve

# Start MCP server (HTTP mode, standalone)
phentrieve mcp serve --http --port 8735

# Display MCP configuration and tools
phentrieve mcp info
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Phentrieve API                        │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────┐ │
│  │ /api/v1/query  │  │ /api/v1/text   │  │ /api/v1/  │ │
│  │                │  │                │  │ similarity│ │
│  └───────┬────────┘  └───────┬────────┘  └─────┬─────┘ │
│          │                   │                 │       │
│          └───────────────────┼─────────────────┘       │
│                              │                          │
│                    ┌─────────▼─────────┐               │
│                    │   FastAPI-MCP     │               │
│                    │   (when enabled)  │               │
│                    └─────────┬─────────┘               │
│                              │                          │
│                    ┌─────────▼─────────┐               │
│                    │      /mcp         │               │
│                    │  (HTTP transport) │               │
│                    └───────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

## Deployment

### Docker Compose (Production)

The default `docker-compose.yml` enables MCP HTTP:

```yaml
services:
  phentrieve_api:
    environment:
      - ENABLE_MCP_HTTP=true  # Enabled by default
```

To disable MCP in Docker:
```bash
ENABLE_MCP_HTTP=false docker-compose up
```

### Nginx Reverse Proxy

If using Nginx Proxy Manager or similar, ensure the `/mcp` path is proxied to the API container:

```nginx
location /mcp {
    proxy_pass http://phentrieve_api:8000/mcp;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

## Troubleshooting

### MCP not available in Docker

Verify the environment variable is set:
```bash
docker-compose exec phentrieve_api printenv | grep MCP
```

Check API logs for MCP mount message:
```bash
docker-compose logs phentrieve_api | grep -i mcp
```

### Dependencies not installed (local development)

Install MCP optional dependencies:
```bash
uv sync --extra mcp
# or
make mcp-install
```

### stdio mode not working with Claude Desktop

1. Verify `phentrieve` is in your PATH
2. Check Claude Desktop logs for error messages
3. Test manually: `echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | phentrieve mcp serve`

## API Reference

See `/docs` endpoint for full OpenAPI documentation of the underlying FastAPI routes that power the MCP tools.

## Security Considerations

- MCP HTTP endpoint uses the same authentication as the main API (none by default)
- For production, place behind a reverse proxy with authentication
- Consider rate limiting MCP endpoints to prevent abuse
- The `/mcp` endpoint only exposes explicitly allowlisted operations
