# API Usage

This page shows the concrete API calls for full-text extraction, including the
LLM-backed path added for end-to-end validation.

## Base URL

Local API development runs on `http://localhost:8734`.

## Full-Text Extraction

### CLI example

```bash
phentrieve text process --extraction-backend llm note.txt
```

### Standard backend

```bash
curl -X POST "http://localhost:8734/api/v1/text/process" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The patient exhibits microcephaly and frequent seizures.",
    "extraction_backend": "standard"
  }'
```

### LLM backend

```bash
curl -X POST "http://localhost:8734/api/v1/text/process" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The patient exhibits microcephaly and frequent seizures.",
    "extraction_backend": "llm",
    "llm_model": "gemini-3.1-flash-lite-preview",
    "llm_mode": "two_phase"
  }'
```

The LLM response keeps the same top-level shape and includes metadata such as
`extraction_backend`, `llm_model`, and `llm_mode`.

## Production Environment

The FastAPI layer uses these environment variables for production LLM handling:

```bash
export PHENTRIEVE_ENV=production
export PHENTRIEVE_TRUSTED_PROXY_CIDRS="127.0.0.1/32,10.0.0.0/8"
export PHENTRIEVE_LLM_DAILY_LIMIT=3
export PHENTRIEVE_LLM_QUOTA_DB_PATH="../data/app/llm_quota.db"
```

- `PHENTRIEVE_ENV` controls whether the API is running in development or
  production mode.
- `PHENTRIEVE_TRUSTED_PROXY_CIDRS` defines which proxy networks are allowed to
  forward client IPs for quota tracking.
- `PHENTRIEVE_LLM_DAILY_LIMIT` sets the number of successful anonymous LLM API
  analyses allowed per UTC day.
- `PHENTRIEVE_LLM_QUOTA_DB_PATH` points to the SQLite database used for API
  quota persistence.

## API Documentation

When the API is running, the OpenAPI pages are available at:

- Swagger UI: `http://localhost:8734/docs`
- ReDoc: `http://localhost:8734/redoc`

## Profiles vs API

The HTTP API does not accept the CLI's `--profile` flag - request fields
are explicit. If you're moving a workflow from the CLI to the API, copy the
relevant fields from your profile into the request body. See
[Configuration Profiles](./configuration-profiles.md) for the profile schema.
