# Modern MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Modernize Phentrieve's HTTP MCP server so Claude, ChatGPT, and other current MCP consumers can reliably discover and use research-only HPO retrieval plus full-text LLM extraction.

**Architecture:** Keep the FastAPI REST API as the application backend, but make the MCP layer an explicit research-use product surface instead of a raw OpenAPI conversion. First correct the HTTP transport to Streamable HTTP, then add a small MCP facade that provides high-signal tool metadata, structured outputs, resources, prompts, compliance guardrails, and regression tests.

**Tech Stack:** Python 3.11+, FastAPI, fastapi-mcp 0.4.x for immediate HTTP transport compatibility, official MCP Python SDK/FastMCP as the target facade, Pydantic schemas, pytest, uv, Ruff, mypy.

---

## Current Implementation Review

### Runtime Surface

Phentrieve currently exposes MCP through `fastapi-mcp`:

- `api/mcp/server.py` creates a `FastApiMCP` instance from the FastAPI app.
- `MCP_ALLOWED_OPERATIONS` exposes three OpenAPI operation IDs:
  - `query_hpo_terms`
  - `process_clinical_text`
  - `calculate_term_similarity`
- `api/mcp/http_server.py` starts a standalone HTTP server.
- `api/main.py` mounts MCP into the main API when `ENABLE_MCP_HTTP=true` or `PHENTRIEVE_MCP_ENABLE_HTTP=true`.
- Docker enables `ENABLE_MCP_HTTP=true` by default, so production expects `/mcp` on the same API domain.

The generated `tools/list` output currently has these gaps:

- No `title`.
- No `annotations`, including no `readOnlyHint`.
- No `outputSchema`.
- No resources.
- No prompts.
- LLM full-text extraction is hidden behind `process_clinical_text` with `extraction_backend="llm"` instead of being advertised as a first-class capability.

### Transport Behavior

The code calls `mcp.mount()` in two HTTP paths:

- `api/mcp/http_server.py`
- `api/main.py`

With installed `fastapi-mcp 0.4.0`, `mount()` is deprecated and mounts SSE-style routes:

```text
GET  /mcp
POST /mcp/messages/
```

`mount_http()` mounts current Streamable HTTP-style routes:

```text
GET    /mcp
POST   /mcp
DELETE /mcp
```

Claude and ChatGPT current docs both steer remote HTTP servers toward Streamable HTTP. Keeping `/mcp` as SSE makes the endpoint look like HTTP in docs while behaving like legacy SSE.

### LLM Full-Text Extraction Path

The shared service supports the modern LLM backend:

- `phentrieve/text_processing/full_text_service.py`
- `run_full_text_service(extraction_backend="llm", ...)`
- Provider fields supported by the service include `llm_provider`, `llm_model`, `llm_base_url`, `llm_mode`, and `llm_internal_mode`.

The API request schema only exposes:

- `extraction_backend`
- `llm_model`
- `llm_mode`

The API router only forwards `llm_model` and `llm_mode` for LLM requests. It does not expose or forward `llm_provider`, `llm_base_url`, or `llm_internal_mode`.

Production quota fallback is controlled by the HTTP header `X-Phentrieve-Allow-Standard-Fallback: true`. That header is not a natural MCP tool argument, so tool consumers are unlikely to discover or use it.

### Docs and Consumer Setup

`docs/mcp-server.md` documents `/mcp` as HTTP but does not distinguish Streamable HTTP from SSE. It also still includes stdio as the first Claude Desktop path, even though this deployment is primarily HTTP.

`docs/prompts/claude-desktop-hpo-annotator.md` instructs users to call `process_clinical_text`, but current MCP best practice is to encode that selection guidance into the tool metadata itself instead of relying on a long external prompt.

---

## Desired MCP Product Surface

### Tools

Expose explicit, action-oriented tool names:

- `phentrieve.extract_hpo_terms`
  - Standard retrieval-backed HPO term suggestions for clinical or biomedical research text.
  - Use when the user needs deterministic local HPO mapping for research, benchmarking, education, or research data standardisation without LLM calls.
- `phentrieve.extract_hpo_terms_llm`
  - LLM-assisted full-text phenotype extraction suggestions with grounded HPO mapping for research text.
  - Use when the user asks for research-only full-document extraction, difficult prose, case-report-like research examples, or higher-recall annotation.
- `phentrieve.search_hpo_terms`
  - Semantic HPO search for a short phenotype phrase.
- `phentrieve.compare_hpo_terms`
  - Semantic or ontology-based similarity between two HPO IDs.
- `phentrieve.get_server_capabilities`
  - Read-only discovery of supported backends, languages, models, transports, quota mode, examples, and research-use limitations.

Every tool description must include this limitation in concise form:

> Research use only. Not for diagnosis, treatment, triage, patient management, clinical decision support, or use with identifiable patient data in public demo instances.

All tools are read-only from the perspective of user data and should advertise:

```json
{
  "readOnlyHint": true,
  "destructiveHint": false,
  "idempotentHint": true,
  "openWorldHint": false
}
```

### Resources

Expose MCP resources for context and autocomplete-friendly discovery:

- `phentrieve://capabilities`
- `phentrieve://hpo/languages`
- `phentrieve://hpo/models`
- `phentrieve://hpo/extraction-profiles`
- `phentrieve://compliance/research-use`
- `phentrieve://examples/research-note`
- `phentrieve://schema/extraction-response`

### Prompts

Expose user-invoked MCP prompts:

- `annotate_research_text`
- `review_hpo_research_annotations`
- `extract_research_case_phenotypes`

These prompts should be short workflow templates that tell the client when to call the MCP tools and how to summarize the returned HPO IDs for research use. They should not contain hidden behavioral instructions or prompt-injection-like content. They must explicitly state that Phentrieve outputs are algorithmic research suggestions and are not for diagnosis, treatment, triage, patient management, or clinical decision support.

Do not expose `compare_standard_vs_llm_extraction` as an MCP tool or prompt. That workflow belongs to Phentrieve's benchmarking/evaluation surface because it compares extraction backends and can be expensive, model-dependent, and analysis-oriented rather than a routine annotation workflow.

---

## File Structure

### Modify

- `api/mcp/server.py`
  - Add explicit mount helpers.
  - Add metadata patching while `fastapi-mcp` remains in use.
  - Keep `MCP_ALLOWED_OPERATIONS` until the explicit MCP facade replaces it.
- `api/mcp/http_server.py`
  - Use Streamable HTTP mounting.
- `api/main.py`
  - Use Streamable HTTP mounting for same-domain `/mcp`.
- `api/schemas/text_processing_schemas.py`
  - Add API-visible LLM provider fields.
  - Add a narrower MCP-oriented extraction request schema if needed.
- `api/routers/text_processing_router.py`
  - Forward LLM provider/base URL/internal mode.
  - Make fallback an API request option in addition to the existing header.
- `phentrieve/cli/mcp_commands.py`
  - Update `mcp info` and HTTP-first configuration examples.
- `docs/mcp-server.md`
  - Rewrite for modern HTTP-first setup.
- `docs/prompts/claude-desktop-hpo-annotator.md`
  - Update examples to use the modern HTTP MCP server and explicit LLM tool.
- `pyproject.toml`
  - Consider replacing or supplementing `fastapi-mcp` with official `mcp[cli]`.

### Create

- `api/mcp/tools.py`
  - Explicit tool schemas and wrapper functions for MCP.
- `api/mcp/resources.py`
  - Static and dynamic resource handlers.
- `api/mcp/prompts.py`
  - Prompt templates.
- `api/mcp/facade.py`
  - Official MCP SDK/FastMCP server construction.
- `tests/unit/mcp/test_mcp_transport.py`
  - Streamable HTTP route and mounting tests.
- `tests/unit/mcp/test_mcp_tool_metadata.py`
  - Tool names, descriptions, annotations, and schemas.
- `tests/unit/mcp/test_mcp_llm_tool.py`
  - LLM tool request mapping and fallback behavior.
- `tests/unit/mcp/test_mcp_resources_prompts.py`
  - Resource and prompt listing/read/get behavior.
- `tests/integration/test_mcp_http_protocol.py`
  - JSON-RPC initialize, tools/list, and tools/call smoke tests over `/mcp`.

---

## Task 1: Lock Down Current HTTP Transport Behavior

**Files:**

- Create: `tests/unit/mcp/test_mcp_transport.py`
- Modify: `api/mcp/server.py`
- Modify: `api/mcp/http_server.py`
- Modify: `api/main.py`

- [x] **Step 1: Write failing route-shape tests**

Create `tests/unit/mcp/test_mcp_transport.py`:

```python
from fastapi import FastAPI


def _mcp_routes(app: FastAPI) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None) or set()
        if path.startswith("/mcp"):
            for method in methods:
                routes.add((method, path))
    return routes


def test_mount_mcp_http_uses_streamable_http_routes() -> None:
    from api.mcp.server import create_mcp_server, mount_mcp_http

    app = FastAPI(title="MCP test")
    mcp = create_mcp_server(app)

    mount_mcp_http(mcp)

    assert ("GET", "/mcp") in _mcp_routes(app)
    assert ("POST", "/mcp") in _mcp_routes(app)
    assert ("DELETE", "/mcp") in _mcp_routes(app)
    assert ("POST", "/mcp/messages/") not in _mcp_routes(app)
```

- [x] **Step 2: Run the failing tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_transport.py -q
```

Expected: fail because `mount_mcp_http` does not exist.

- [x] **Step 3: Add explicit mount helpers**

In `api/mcp/server.py`, add:

```python
def mount_mcp_http(
    mcp: Any,
    *,
    mount_path: str = "/mcp",
) -> None:
    """Mount MCP using modern Streamable HTTP."""
    mcp.mount_http(mount_path=mount_path)
```

Update the module docstring examples from `mcp.mount()` to `mount_mcp_http(mcp)`.

- [x] **Step 4: Use the helper in standalone HTTP mode**

In `api/mcp/http_server.py`, replace:

```python
mcp.mount()
```

with:

```python
from api.mcp.server import create_mcp_server, mount_mcp_http

mount_mcp_http(mcp)
```

Keep the existing `create_mcp_server` import in the same import block.

- [x] **Step 5: Use the helper in same-domain API mode**

In `api/main.py`, replace:

```python
from api.mcp.server import create_mcp_server

mcp = create_mcp_server(target_app)
mcp.mount()  # Mounts at /mcp by default
logger.info("MCP server mounted at /mcp (ENABLE_MCP_HTTP=true)")
```

with:

```python
from api.mcp.server import create_mcp_server, mount_mcp_http

mcp = create_mcp_server(target_app)
mount_mcp_http(mcp)
logger.info("MCP Streamable HTTP server mounted at /mcp")
```

- [x] **Step 6: Run transport tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_transport.py -q
```

Expected: all tests pass.

- [x] **Step 7: Commit**

```bash
git add api/mcp/server.py api/mcp/http_server.py api/main.py tests/unit/mcp/test_mcp_transport.py
git commit -m "fix: mount mcp over streamable http"
```

---

## Task 2: Add HTTP-First MCP CLI Clarity

**Files:**

- Modify: `phentrieve/cli/mcp_commands.py`
- Modify: `tests/unit/mcp/test_mcp_server.py`

- [x] **Step 1: Add tests for HTTP-first info output**

Append to `tests/unit/mcp/test_mcp_server.py`:

```python
def test_mcp_info_prefers_http_configuration():
    from typer.testing import CliRunner

    from phentrieve.cli.mcp_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert '"type": "http"' in result.output
    assert "http://127.0.0.1:8734/mcp" in result.output
    assert "Streamable HTTP" in result.output
    assert "/sse" not in result.output
```

- [x] **Step 2: Run the failing CLI test**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_server.py::test_mcp_info_prefers_http_configuration -q
```

Expected: fail because `mcp info` still presents stdio-first configuration and does not label Streamable HTTP.

- [x] **Step 3: Update `mcp info` output**

In `phentrieve/cli/mcp_commands.py`, change the example panel to HTTP-first:

```json
{
  "mcpServers": {
    "phentrieve": {
      "type": "http",
      "url": "http://127.0.0.1:8734/mcp"
    }
  }
}
```

Also print:

```python
console.print("  HTTP Transport: Streamable HTTP")
console.print(f"  MCP URL: http://{settings.host}:{settings.port}/mcp")
```

- [x] **Step 4: Run MCP unit tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp -q
```

Expected: all MCP unit tests pass.

- [x] **Step 5: Commit**

```bash
git add phentrieve/cli/mcp_commands.py tests/unit/mcp/test_mcp_server.py
git commit -m "chore: clarify http-first mcp configuration"
```

---

## Task 3: Patch Current Tool Metadata While Keeping `fastapi-mcp`

**Files:**

- Create: `api/mcp/metadata.py`
- Create: `tests/unit/mcp/test_mcp_tool_metadata.py`
- Modify: `api/mcp/server.py`

- [x] **Step 1: Write metadata tests**

Create `tests/unit/mcp/test_mcp_tool_metadata.py`:

```python
def test_tools_have_modern_metadata() -> None:
    from api.main import create_app
    from api.mcp.server import create_mcp_server

    app = create_app()
    mcp = create_mcp_server(app)

    tools = {tool.name: tool for tool in mcp.tools}

    for tool_name in (
        "query_hpo_terms",
        "process_clinical_text",
        "calculate_term_similarity",
    ):
        tool = tools[tool_name]
        assert tool.title
        assert tool.description
        assert "Use this when" in tool.description
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False


def test_process_clinical_text_description_mentions_llm_backend() -> None:
    from api.main import create_app
    from api.mcp.server import create_mcp_server

    app = create_app()
    mcp = create_mcp_server(app)
    tools = {tool.name: tool for tool in mcp.tools}

    description = tools["process_clinical_text"].description

    assert "extraction_backend='llm'" in description
    assert "full-text LLM" in description
```

- [x] **Step 2: Run the failing metadata tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_tool_metadata.py -q
```

Expected: fail because generated tools have no titles or annotations.

- [x] **Step 3: Add metadata patch definitions**

Create `api/mcp/metadata.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from mcp import types


@dataclass(frozen=True)
class ToolMetadata:
    title: str
    description: str


READ_ONLY_ANNOTATIONS = types.ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


TOOL_METADATA: dict[str, ToolMetadata] = {
    "query_hpo_terms": ToolMetadata(
        title="Search HPO Terms",
        description=(
            "Use this when a user provides a short phenotype phrase or research "
            "text snippet and needs candidate Human Phenotype Ontology terms by "
            "semantic similarity. Do not use this for full research notes; use "
            "process_clinical_text instead. Research use only; not for diagnosis, "
            "treatment, triage, patient management, or clinical decision support."
        ),
    ),
    "process_clinical_text": ToolMetadata(
        title="Extract HPO Terms From Research Text",
        description=(
            "Use this when a user asks to extract HPO term suggestions from "
            "clinical or biomedical research text, synthetic examples, or "
            "case-report-like research text. Use "
            "extraction_backend='standard' for deterministic retrieval-backed "
            "extraction. Use extraction_backend='llm' for full-text LLM extraction "
            "when the request asks for LLM-assisted interpretation, higher recall, "
            "or document-level phenotype annotation. Research use only; not for "
            "diagnosis, treatment, triage, patient management, or clinical decision "
            "support. Do not submit identifiable patient data to public demo "
            "instances."
        ),
    ),
    "calculate_term_similarity": ToolMetadata(
        title="Compare HPO Terms",
        description=(
            "Use this when a user provides two HPO IDs and asks how similar or "
            "related the terms are. Do not use this to search for HPO terms from "
            "free text. Research use only; not for diagnosis, treatment, triage, "
            "patient management, or clinical decision support."
        ),
    ),
}


def apply_tool_metadata(tools: list[types.Tool]) -> list[types.Tool]:
    patched: list[types.Tool] = []
    for tool in tools:
        metadata = TOOL_METADATA.get(tool.name)
        if metadata is None:
            patched.append(tool)
            continue
        patched.append(
            tool.model_copy(
                update={
                    "title": metadata.title,
                    "description": metadata.description,
                    "annotations": READ_ONLY_ANNOTATIONS,
                }
            )
        )
    return patched
```

- [x] **Step 4: Apply metadata in server factory**

In `api/mcp/server.py`, after constructing `mcp`, add:

```python
from api.mcp.metadata import apply_tool_metadata

mcp.tools = apply_tool_metadata(mcp.tools)
```

- [x] **Step 5: Run metadata tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_tool_metadata.py -q
```

Expected: all tests pass.

- [x] **Step 6: Run MCP unit tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp -q
```

Expected: all MCP unit tests pass.

- [x] **Step 7: Commit**

```bash
git add api/mcp/server.py api/mcp/metadata.py tests/unit/mcp/test_mcp_tool_metadata.py
git commit -m "feat: improve mcp tool metadata"
```

---

## Task 4: Expose Full LLM Request Controls Through API and MCP

**Files:**

- Modify: `api/schemas/text_processing_schemas.py`
- Modify: `api/routers/text_processing_router.py`
- Modify: `tests/unit/api/test_schemas.py`
- Modify: `tests/unit/api/test_text_processing_router.py`
- Modify: `tests/unit/mcp/test_mcp_tool_metadata.py`

- [x] **Step 1: Add schema tests**

Append to `tests/unit/api/test_schemas.py`:

```python
def test_text_processing_request_accepts_llm_provider_fields() -> None:
    from api.schemas.text_processing_schemas import TextProcessingRequest

    request = TextProcessingRequest(
        text="Patient has seizures.",
        extraction_backend="llm",
        llm_model="openai/gpt-5.4-mini",
        llm_provider="openai",
        llm_base_url="https://api.openai.com/v1",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        allow_standard_fallback=True,
    )

    assert request.llm_provider == "openai"
    assert request.llm_base_url == "https://api.openai.com/v1"
    assert request.llm_internal_mode == "whole_document_grounded"
    assert request.allow_standard_fallback is True
```

- [x] **Step 2: Run the failing schema test**

Run:

```bash
uv run pytest tests/unit/api/test_schemas.py::test_text_processing_request_accepts_llm_provider_fields -q
```

Expected: fail because the fields do not exist.

- [x] **Step 3: Add request fields**

In `api/schemas/text_processing_schemas.py`, update `TextProcessingRequest`:

```python
    llm_provider: str | None = Field(
        default=None,
        description=(
            "Optional LLM provider name for full-text extraction. Examples: "
            "'openai', 'anthropic', 'gemini', 'ollama'. If omitted, server "
            "environment defaults are used."
        ),
    )
    llm_base_url: str | None = Field(
        default=None,
        description=(
            "Optional provider base URL for compatible LLM providers. Use this "
            "for local Ollama or OpenAI-compatible gateways."
        ),
    )
    llm_internal_mode: Literal[
        "whole_document_legacy", "whole_document_grounded"
    ] | None = Field(
        default="whole_document_grounded",
        description="Internal grounding mode for LLM full-text extraction.",
    )
    allow_standard_fallback: bool = Field(
        default=False,
        description=(
            "When true, production LLM quota exhaustion falls back to the standard "
            "backend instead of returning a quota error."
        ),
    )
```

- [x] **Step 4: Use request fallback flag in router**

In `api/routers/text_processing_router.py`, replace:

```python
allow_standard_fallback = (
    http_request.headers.get("x-phentrieve-allow-standard-fallback", "").lower()
    == "true"
)
```

with:

```python
allow_standard_fallback = request.allow_standard_fallback or (
    http_request.headers.get("x-phentrieve-allow-standard-fallback", "").lower()
    == "true"
)
```

- [x] **Step 5: Forward LLM fields into service**

In `_process_text_via_shared_service`, replace the LLM `service_kwargs.update` block with:

```python
        service_kwargs.update(
            {
                "language": request.language,
                "llm_provider": request.llm_provider,
                "llm_model": request.llm_model,
                "llm_base_url": request.llm_base_url,
                "llm_mode": request.llm_mode or "two_phase",
                "llm_internal_mode": (
                    request.llm_internal_mode or "whole_document_grounded"
                ),
            }
        )
```

- [x] **Step 6: Add router forwarding test**

Append to `tests/unit/api/test_text_processing_router.py`:

```python
@pytest.mark.asyncio
async def test_llm_request_forwards_provider_fields(monkeypatch) -> None:
    from api.routers import text_processing_router
    from api.schemas.text_processing_schemas import TextProcessingRequest

    captured: dict[str, object] = {}

    def fake_run_full_text_service(**kwargs):
        captured.update(kwargs)
        return {
            "meta": {"extraction_backend": "llm"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }

    async def fake_run_in_threadpool(func, **kwargs):
        return func(**kwargs)

    monkeypatch.setattr(
        text_processing_router,
        "run_full_text_service",
        fake_run_full_text_service,
    )
    monkeypatch.setattr(
        text_processing_router,
        "run_in_threadpool",
        fake_run_in_threadpool,
    )

    request = TextProcessingRequest(
        text="Patient has seizures.",
        extraction_backend="llm",
        llm_provider="openai",
        llm_model="openai/gpt-5.4-mini",
        llm_base_url="https://api.openai.com/v1",
        llm_internal_mode="whole_document_grounded",
    )

    await text_processing_router._process_text_via_shared_service(request)

    assert captured["llm_provider"] == "openai"
    assert captured["llm_model"] == "openai/gpt-5.4-mini"
    assert captured["llm_base_url"] == "https://api.openai.com/v1"
    assert captured["llm_internal_mode"] == "whole_document_grounded"
```

- [x] **Step 7: Run targeted tests**

Run:

```bash
uv run pytest tests/unit/api/test_schemas.py::test_text_processing_request_accepts_llm_provider_fields tests/unit/api/test_text_processing_router.py::test_llm_request_forwards_provider_fields -q
```

Expected: both tests pass.

- [x] **Step 8: Run MCP metadata test to verify fields appear in tool schema**

Add to `tests/unit/mcp/test_mcp_tool_metadata.py`:

```python
def test_process_clinical_text_schema_exposes_llm_controls() -> None:
    from api.main import create_app
    from api.mcp.server import create_mcp_server

    app = create_app()
    mcp = create_mcp_server(app)
    tools = {tool.name: tool for tool in mcp.tools}
    schema = tools["process_clinical_text"].inputSchema
    request_properties = schema["properties"]

    assert "llm_provider" in request_properties
    assert "llm_base_url" in request_properties
    assert "llm_internal_mode" in request_properties
    assert "allow_standard_fallback" in request_properties
```

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_tool_metadata.py::test_process_clinical_text_schema_exposes_llm_controls -q
```

Expected: pass.

- [x] **Step 9: Commit**

```bash
git add api/schemas/text_processing_schemas.py api/routers/text_processing_router.py tests/unit/api/test_schemas.py tests/unit/api/test_text_processing_router.py tests/unit/mcp/test_mcp_tool_metadata.py
git commit -m "feat: expose llm extraction controls to mcp"
```

---

## Task 5: Add an Explicit MCP Facade With First-Class Tools

**Files:**

- Create: `api/mcp/tools.py`
- Create: `api/mcp/facade.py`
- Create: `tests/unit/mcp/test_mcp_facade_tools.py`
- Modify: `pyproject.toml`

- [x] **Step 1: Add official MCP SDK dependency**

In `pyproject.toml`, extend the `mcp` optional dependency:

```toml
mcp = [
    "fastapi-mcp>=0.4.0,<1.0.0",
    "mcp[cli]>=1.22.0,<2.0.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-dotenv>=1.0.0",
    "pydantic-settings>=2.0.0",
]
```

- [x] **Step 2: Write facade tool tests**

Create `tests/unit/mcp/test_mcp_facade_tools.py`:

```python
def test_facade_registers_first_class_tools() -> None:
    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "phentrieve.extract_hpo_terms" in tool_names
    assert "phentrieve.extract_hpo_terms_llm" in tool_names
    assert "phentrieve.get_server_capabilities" in tool_names
```

- [x] **Step 3: Run failing facade test**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_facade_tools.py -q
```

Expected: fail because `api.mcp.facade` does not exist.

- [x] **Step 4: Add tool request and response models**

Create `api/mcp/tools.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractHpoTermsRequest(BaseModel):
    text: str = Field(
        description=(
            "Clinical or biomedical research text to map to HPO term suggestions. "
            "Do not submit identifiable patient data to public demo instances."
        )
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code. Use null to let Phentrieve detect it.",
    )
    include_details: bool = Field(
        default=True,
        description="Include HPO definitions and synonyms.",
    )
    include_chunk_positions: bool = Field(
        default=True,
        description="Include source character offsets for evidence chunks.",
    )
    num_results_per_chunk: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum HPO candidates per chunk.",
    )
    chunk_retrieval_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum chunk-level retrieval similarity.",
    )


class ExtractHpoTermsLlmRequest(ExtractHpoTermsRequest):
    llm_model: str = Field(
        description="LLM model name, optionally provider-prefixed.",
    )
    llm_provider: str | None = Field(
        default=None,
        description="Provider name such as openai, anthropic, gemini, or ollama.",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Optional provider base URL.",
    )
    llm_mode: Literal["two_phase"] = "two_phase"
    llm_internal_mode: Literal[
        "whole_document_legacy", "whole_document_grounded"
    ] = "whole_document_grounded"
    allow_standard_fallback: bool = Field(
        default=False,
        description="Fall back to standard extraction if production quota is exhausted.",
    )


class SearchHpoTermsRequest(BaseModel):
    text: str = Field(description="Phenotype phrase or short clinical snippet.")
    language: str | None = None
    num_results: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    include_details: bool = True


class CompareHpoTermsRequest(BaseModel):
    term1_id: str = Field(pattern=r"^HP:\d{7}$")
    term2_id: str = Field(pattern=r"^HP:\d{7}$")
    formula: Literal["hybrid", "simple_resnik_like"] = "hybrid"
```

- [x] **Step 5: Add the initial facade**

Create `api/mcp/facade.py`:

```python
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from api.mcp.tools import (
    CompareHpoTermsRequest,
    ExtractHpoTermsLlmRequest,
    ExtractHpoTermsRequest,
    SearchHpoTermsRequest,
)
from phentrieve.text_processing.full_text_service import run_full_text_service


def create_phentrieve_mcp() -> FastMCP:
    mcp = FastMCP(
        name="phentrieve",
        instructions=(
            "Phentrieve maps clinical or biomedical research text to Human "
            "Phenotype Ontology term suggestions for research, benchmarking, "
            "education, and research data standardisation only. It is not for "
            "diagnosis, treatment, triage, patient management, or clinical "
            "decision support. Use the LLM tool only when the user asks for "
            "research-only LLM-assisted full-text extraction or document-level "
            "phenotype annotation."
        ),
    )

    @mcp.tool(
        name="phentrieve.extract_hpo_terms",
        title="Extract HPO Terms",
    )
    def extract_hpo_terms(request: ExtractHpoTermsRequest) -> dict[str, Any]:
        """Use this when research text should be mapped to HPO term suggestions without LLM calls. Research use only; not for diagnosis, treatment, triage, patient management, or clinical decision support."""
        return run_full_text_service(
            text=request.text,
            extraction_backend="standard",
            language=request.language,
            include_details=request.include_details,
            include_positions=request.include_chunk_positions,
            num_results_per_chunk=request.num_results_per_chunk,
            chunk_retrieval_threshold=request.chunk_retrieval_threshold,
        )

    @mcp.tool(
        name="phentrieve.extract_hpo_terms_llm",
        title="Extract HPO Terms With LLM",
    )
    def extract_hpo_terms_llm(request: ExtractHpoTermsLlmRequest) -> dict[str, Any]:
        """Use this when research-only full-text LLM extraction should identify phenotype mentions and map them to grounded HPO term suggestions. Not for diagnosis, treatment, triage, patient management, or clinical decision support."""
        return run_full_text_service(
            text=request.text,
            extraction_backend="llm",
            language=request.language,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            llm_base_url=request.llm_base_url,
            llm_mode=request.llm_mode,
            llm_internal_mode=request.llm_internal_mode,
            include_details=request.include_details,
            include_positions=request.include_chunk_positions,
        )

    @mcp.tool(
        name="phentrieve.get_server_capabilities",
        title="Get Phentrieve Capabilities",
    )
    def get_server_capabilities() -> dict[str, Any]:
        """Use this when a client needs supported languages, backends, and examples."""
        return {
            "server": "phentrieve",
            "transports": ["streamable_http"],
            "extraction_backends": ["standard", "llm"],
            "llm_modes": ["two_phase"],
            "llm_internal_modes": [
                "whole_document_grounded",
                "whole_document_legacy",
            ],
            "languages": ["en", "de", "es", "fr", "nl"],
            "intended_use": (
                "Research, benchmarking, education, and research data "
                "standardisation only."
            ),
            "prohibited_uses": [
                "diagnosis",
                "treatment",
                "triage",
                "patient management",
                "clinical decision support",
                "identifiable patient data in public demo instances",
            ],
        }

    return mcp
```

- [x] **Step 6: Run facade registration test**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_facade_tools.py -q
```

Expected: pass for registration.

- [x] **Step 7: Commit**

```bash
git add pyproject.toml api/mcp/tools.py api/mcp/facade.py tests/unit/mcp/test_mcp_facade_tools.py
git commit -m "feat: add explicit phentrieve mcp facade"
```

---

## Task 6: Wire Facade Tools to Existing Backend Services

**Files:**

- Modify: `api/mcp/facade.py`
- Modify: `api/mcp/tools.py`
- Modify: `tests/unit/mcp/test_mcp_llm_tool.py`
- Modify: `tests/unit/mcp/test_mcp_facade_tools.py`

- [x] **Step 1: Add extraction wrapper tests**

Create `tests/unit/mcp/test_mcp_llm_tool.py`:

```python
def test_llm_tool_maps_request_to_full_text_service(monkeypatch) -> None:
    from api.mcp.facade import extract_hpo_terms_llm_impl
    from api.mcp.tools import ExtractHpoTermsLlmRequest

    captured: dict[str, object] = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return {
            "meta": {"extraction_backend": "llm"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }

    request = ExtractHpoTermsLlmRequest(
        text="Patient has seizures.",
        language="en",
        llm_provider="openai",
        llm_model="openai/gpt-5.4-mini",
        llm_base_url="https://api.openai.com/v1",
    )

    result = extract_hpo_terms_llm_impl(request, service=fake_service)

    assert result["meta"]["extraction_backend"] == "llm"
    assert captured["text"] == "Patient has seizures."
    assert captured["extraction_backend"] == "llm"
    assert captured["llm_provider"] == "openai"
    assert captured["llm_model"] == "openai/gpt-5.4-mini"
    assert captured["llm_base_url"] == "https://api.openai.com/v1"
    assert captured["llm_internal_mode"] == "whole_document_grounded"
```

- [x] **Step 2: Run failing wrapper test**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_llm_tool.py -q
```

Expected: fail because implementation helper does not exist.

- [x] **Step 3: Extract implementation helpers**

In `api/mcp/facade.py`, add module-level helper functions:

```python
def extract_hpo_terms_impl(
    request: ExtractHpoTermsRequest,
    *,
    service=run_full_text_service,
) -> dict[str, Any]:
    return service(
        text=request.text,
        extraction_backend="standard",
        language=request.language,
        include_details=request.include_details,
        include_positions=request.include_chunk_positions,
        num_results_per_chunk=request.num_results_per_chunk,
        chunk_retrieval_threshold=request.chunk_retrieval_threshold,
    )


def extract_hpo_terms_llm_impl(
    request: ExtractHpoTermsLlmRequest,
    *,
    service=run_full_text_service,
) -> dict[str, Any]:
    return service(
        text=request.text,
        extraction_backend="llm",
        language=request.language,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        llm_base_url=request.llm_base_url,
        llm_mode=request.llm_mode,
        llm_internal_mode=request.llm_internal_mode,
        include_details=request.include_details,
        include_positions=request.include_chunk_positions,
    )
```

Change the registered facade tools to call these helpers.

- [x] **Step 4: Wire search and compare helpers**

Add tests first in `tests/unit/mcp/test_mcp_facade_tools.py` using injected fake callables:

```python
def test_facade_registers_search_and_compare_tools() -> None:
    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "phentrieve.search_hpo_terms" in tool_names
    assert "phentrieve.compare_hpo_terms" in tool_names


def test_search_hpo_terms_impl_delegates() -> None:
    from api.mcp.facade import search_hpo_terms_impl
    from api.mcp.tools import SearchHpoTermsRequest

    captured: dict[str, object] = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return {"results": [{"hpo_id": "HP:0001250", "label": "Seizure"}]}

    result = search_hpo_terms_impl(
        SearchHpoTermsRequest(text="seizures", language="en"),
        search=fake_search,
    )

    assert result["results"][0]["hpo_id"] == "HP:0001250"
    assert captured["text"] == "seizures"
    assert captured["language"] == "en"


def test_compare_hpo_terms_impl_delegates() -> None:
    from api.mcp.facade import compare_hpo_terms_impl
    from api.mcp.tools import CompareHpoTermsRequest

    captured: dict[str, object] = {}

    def fake_compare(**kwargs):
        captured.update(kwargs)
        return {"similarity_score": 0.75}

    result = compare_hpo_terms_impl(
        CompareHpoTermsRequest(term1_id="HP:0001250", term2_id="HP:0001249"),
        compare=fake_compare,
    )

    assert result["similarity_score"] == 0.75
    assert captured["term1_id"] == "HP:0001250"
    assert captured["term2_id"] == "HP:0001249"
```

Implement `search_hpo_terms_impl` and `compare_hpo_terms_impl` with dependency injection:

```python
def search_hpo_terms_impl(
    request: SearchHpoTermsRequest,
    *,
    search: Any,
) -> dict[str, Any]:
    return search(
        text=request.text,
        language=request.language,
        num_results=request.num_results,
        similarity_threshold=request.similarity_threshold,
        include_details=request.include_details,
    )


def compare_hpo_terms_impl(
    request: CompareHpoTermsRequest,
    *,
    compare: Any,
) -> dict[str, Any]:
    return compare(
        term1_id=request.term1_id,
        term2_id=request.term2_id,
        formula=request.formula,
    )
```

Then add real `search_hpo_terms` and `compare_hpo_terms` facade tool registrations that call these helpers with existing backend functions. If the current code only exposes router-level logic, create small service helpers under `phentrieve/retrieval/api_helpers.py` and the current similarity module, then inject those helpers into the facade tools.

- [x] **Step 5: Run MCP facade tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_facade_tools.py tests/unit/mcp/test_mcp_llm_tool.py -q
```

Expected: all tests pass.

- [x] **Step 6: Commit**

```bash
git add api/mcp/facade.py api/mcp/tools.py tests/unit/mcp/test_mcp_facade_tools.py tests/unit/mcp/test_mcp_llm_tool.py
git commit -m "feat: wire explicit mcp tools to services"
```

---

## Task 7: Add MCP Resources and Prompts

**Files:**

- Create: `api/mcp/resources.py`
- Create: `api/mcp/prompts.py`
- Create: `tests/unit/mcp/test_mcp_resources_prompts.py`
- Modify: `api/mcp/facade.py`

- [x] **Step 1: Write resources and prompts tests**

Create `tests/unit/mcp/test_mcp_resources_prompts.py`:

```python
def test_resource_payloads_are_available() -> None:
    from api.mcp.resources import get_capabilities_resource, get_languages_resource

    capabilities = get_capabilities_resource()
    languages = get_languages_resource()

    assert capabilities["server"] == "phentrieve"
    assert "llm" in capabilities["extraction_backends"]
    assert "en" in languages["supported_languages"]
    assert "de" in languages["supported_languages"]


def test_prompt_templates_are_short_actionable_and_research_only() -> None:
    from api.mcp.prompts import annotate_research_text_prompt

    prompt = annotate_research_text_prompt(language="en")

    assert "phentrieve.extract_hpo_terms" in prompt
    assert "phentrieve.extract_hpo_terms_llm" in prompt
    assert "compare_standard_vs_llm_extraction" not in prompt
    assert "Research use only" in prompt
    assert "not for diagnosis" in prompt
    assert len(prompt) < 3000


def test_benchmark_comparison_is_not_an_mcp_prompt() -> None:
    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    prompt_names = set(mcp._prompt_manager._prompts.keys())
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "compare_standard_vs_llm_extraction" not in prompt_names
    assert "phentrieve.compare_standard_vs_llm_extraction" not in tool_names
```

- [x] **Step 2: Run failing tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_resources_prompts.py -q
```

Expected: fail because modules do not exist.

- [x] **Step 3: Add resources**

Create `api/mcp/resources.py`:

```python
from __future__ import annotations

from typing import Any


def get_capabilities_resource() -> dict[str, Any]:
    return {
        "server": "phentrieve",
        "domain": "research-use phenotype annotation",
        "intended_use": (
            "Research, benchmarking, education, and research data standardisation."
        ),
        "not_intended_for": [
            "diagnosis",
            "treatment",
            "triage",
            "patient management",
            "clinical decision support",
            "use with identifiable patient data in public demo instances",
        ],
        "ontology": "Human Phenotype Ontology",
        "transports": ["streamable_http"],
        "extraction_backends": ["standard", "llm"],
        "tools": [
            "phentrieve.extract_hpo_terms",
            "phentrieve.extract_hpo_terms_llm",
            "phentrieve.search_hpo_terms",
            "phentrieve.compare_hpo_terms",
        ],
    }


def get_languages_resource() -> dict[str, Any]:
    return {
        "supported_languages": ["en", "de", "es", "fr", "nl"],
        "default_language": "en",
        "language_parameter": "ISO 639-1 code",
    }


def get_extraction_profiles_resource() -> dict[str, Any]:
    return {
        "profiles": [
            {
                "name": "standard",
                "backend": "standard",
                "use_when": "deterministic local retrieval-backed extraction",
            },
            {
                "name": "llm_full_text",
                "backend": "llm",
                "use_when": "document-level extraction with LLM phrase identification",
            },
        ]
    }


def get_research_use_resource() -> dict[str, Any]:
    return {
        "intended_use": (
            "Phentrieve is open-source research software for mapping clinical "
            "or biomedical research text to Human Phenotype Ontology term "
            "suggestions for exploration, benchmarking, education, and research "
            "data standardisation."
        ),
        "not_intended_for": [
            "diagnosis",
            "treatment",
            "triage",
            "patient management",
            "clinical decision support",
        ],
        "public_demo_data_notice": (
            "Do not submit identifiable patient data to public demo instances."
        ),
    }
```

- [x] **Step 4: Add prompts**

Create `api/mcp/prompts.py`:

```python
from __future__ import annotations


def annotate_research_text_prompt(language: str = "en") -> str:
    return (
        "Research use only; not for diagnosis, treatment, triage, patient "
        "management, or clinical decision support. Map the supplied clinical or "
        "biomedical research text to Human Phenotype Ontology term suggestions. "
        f"Use language='{language}'. For ordinary deterministic extraction, call "
        "phentrieve.extract_hpo_terms. For full-text LLM-assisted extraction, "
        "call phentrieve.extract_hpo_terms_llm. Return HPO IDs, labels, assertion "
        "status, evidence spans, and a short research-use uncertainty note."
    )


def review_hpo_research_annotations_prompt() -> str:
    return (
        "Research use only; not for diagnosis, treatment, triage, patient "
        "management, or clinical decision support. Review the supplied HPO "
        "annotations against the research text evidence. "
        "Keep only HPO IDs returned by Phentrieve tools. Flag unsupported, "
        "negated, historical, or family-history-only phenotype suggestions."
    )


def extract_research_case_phenotypes_prompt(language: str = "en") -> str:
    return (
        "Research use only; not for diagnosis, treatment, triage, patient "
        "management, or clinical decision support. Extract phenotype suggestions "
        "from synthetic or research-consented case-report-like text. Exclude "
        "family history unless it is explicitly described as the research "
        "subject's phenotype. Prefer "
        f"phentrieve.extract_hpo_terms_llm with language='{language}' for long "
        "documents."
    )
```

- [x] **Step 5: Register resources and prompts in facade**

In `api/mcp/facade.py`, import:

```python
from api.mcp.prompts import (
    annotate_research_text_prompt,
    extract_research_case_phenotypes_prompt,
    review_hpo_research_annotations_prompt,
)
from api.mcp.resources import (
    get_capabilities_resource,
    get_extraction_profiles_resource,
    get_languages_resource,
    get_research_use_resource,
)
```

Inside `create_phentrieve_mcp`, add:

```python
    @mcp.resource("phentrieve://capabilities")
    def capabilities() -> dict[str, Any]:
        return get_capabilities_resource()

    @mcp.resource("phentrieve://hpo/languages")
    def languages() -> dict[str, Any]:
        return get_languages_resource()

    @mcp.resource("phentrieve://hpo/extraction-profiles")
    def extraction_profiles() -> dict[str, Any]:
        return get_extraction_profiles_resource()

    @mcp.resource("phentrieve://compliance/research-use")
    def research_use() -> dict[str, Any]:
        return get_research_use_resource()

    @mcp.prompt(name="annotate_research_text", title="Annotate Research Text")
    def annotate_research_text(language: str = "en") -> str:
        return annotate_research_text_prompt(language=language)

    @mcp.prompt(
        name="review_hpo_research_annotations",
        title="Review HPO Research Annotations",
    )
    def review_hpo_research_annotations() -> str:
        return review_hpo_research_annotations_prompt()

    @mcp.prompt(
        name="extract_research_case_phenotypes",
        title="Extract Research Case Phenotypes",
    )
    def extract_research_case_phenotypes(language: str = "en") -> str:
        return extract_research_case_phenotypes_prompt(language=language)
```

- [x] **Step 6: Run resource and prompt tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp/test_mcp_resources_prompts.py -q
```

Expected: all tests pass.

- [x] **Step 7: Commit**

```bash
git add api/mcp/facade.py api/mcp/resources.py api/mcp/prompts.py tests/unit/mcp/test_mcp_resources_prompts.py
git commit -m "feat: add mcp resources and prompts"
```

---

## Task 8: Switch HTTP Serving to the Explicit Facade

**Files:**

- Modify: `api/mcp/http_server.py`
- Modify: `api/main.py`
- Modify: `api/mcp/server.py`
- Create: `tests/integration/test_mcp_http_protocol.py`

- [x] **Step 1: Write HTTP protocol smoke test**

Create `tests/integration/test_mcp_http_protocol.py`:

```python
import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_mcp_http_tools_list_smoke(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MCP_HTTP", "true")

    from api.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        initialize = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0"},
                },
            },
        )
        assert initialize.status_code in {200, 202}

        tools = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )

    assert tools.status_code == 200
    payload = tools.json()
    tool_names = {tool["name"] for tool in payload["result"]["tools"]}
    assert "phentrieve.extract_hpo_terms" in tool_names
    assert "phentrieve.extract_hpo_terms_llm" in tool_names
```

- [x] **Step 2: Run failing smoke test**

Run:

```bash
uv run --extra mcp pytest tests/integration/test_mcp_http_protocol.py -q
```

Expected: fail until the facade is mounted over HTTP.

- [x] **Step 3: Add facade ASGI mounting helper**

In `api/mcp/server.py`, add:

```python
def mount_phentrieve_mcp_facade(
    app: FastAPI,
    *,
    mount_path: str = "/mcp",
) -> None:
    """Mount the explicit Phentrieve MCP facade over Streamable HTTP."""
    from api.mcp.facade import create_phentrieve_mcp

    facade = create_phentrieve_mcp()
    app.mount(mount_path, facade.streamable_http_app())
```

If the installed official MCP SDK exposes a different ASGI helper name, inspect it with:

```bash
uv run --extra mcp python - <<'PY'
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("x")
print([name for name in dir(mcp) if "http" in name.lower() or "app" in name.lower()])
PY
```

Use the SDK-provided Streamable HTTP ASGI app method shown by that command.

- [x] **Step 4: Use facade for HTTP**

In `api/main.py`, replace the `fastapi-mcp` mount path with:

```python
from api.mcp.server import mount_phentrieve_mcp_facade

mount_phentrieve_mcp_facade(target_app)
logger.info("Phentrieve MCP facade mounted at /mcp using Streamable HTTP")
```

In `api/mcp/http_server.py`, create a new FastAPI app for MCP-only serving:

```python
from fastapi import FastAPI

from api.mcp.server import mount_phentrieve_mcp_facade

mcp_app = FastAPI(title="Phentrieve MCP")
mount_phentrieve_mcp_facade(mcp_app)
uvicorn.run(mcp_app, host=settings.host, port=settings.port)
```

- [x] **Step 5: Keep OpenAPI conversion behind a compatibility name**

Rename the old `create_mcp_server` docstring to state it is legacy OpenAPI conversion. Keep it importable for compatibility and tests until a later release can remove it.

- [x] **Step 6: Run smoke test**

Run:

```bash
uv run --extra mcp pytest tests/integration/test_mcp_http_protocol.py -q
```

Expected: pass.

- [x] **Step 7: Run MCP suite**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp tests/integration/test_mcp_http_protocol.py -q
```

Expected: all tests pass.

- [x] **Step 8: Commit**

```bash
git add api/mcp/server.py api/mcp/http_server.py api/main.py tests/integration/test_mcp_http_protocol.py
git commit -m "feat: serve explicit mcp facade over http"
```

---

## Task 9: Update Documentation and Deployment Guidance

**Files:**

- Modify: `docs/mcp-server.md`
- Modify: `docs/prompts/claude-desktop-hpo-annotator.md`
- Modify: `docker-compose.yml`
- Modify: `api/Dockerfile`

- [x] **Step 1: Rewrite transport section**

In `docs/mcp-server.md`, make this table the source of truth:

```markdown
| Mode | Endpoint | Status | Use Case |
|------|----------|--------|----------|
| Streamable HTTP | `/mcp` | Recommended | Claude, ChatGPT developer mode, remote MCP clients |
| stdio | command-based | Local fallback | Local desktop clients only |
```

- [x] **Step 2: Add ChatGPT developer mode configuration**

Add:

```markdown
### ChatGPT Developer Mode

Create a custom app from the remote MCP server:

```text
https://your-domain.example/mcp
```

Use no authentication only for local/private deployments. For public deployments, put the endpoint behind OAuth or an authenticated reverse proxy before enabling write-capable tools. Phentrieve's current tools are read-only and annotated as such.
```
```

- [x] **Step 3: Add Claude HTTP configuration**

Add:

```markdown
### Claude Code HTTP

```bash
claude mcp add --transport http phentrieve https://your-domain.example/mcp
```

For local development:

```bash
make mcp-serve-http
claude mcp add --transport http phentrieve http://127.0.0.1:8734/mcp
```
```

- [x] **Step 4: Document first-class tools**

Replace the old three-tool table with:

```markdown
| Tool | Use When |
|------|----------|
| `phentrieve.extract_hpo_terms` | Deterministic retrieval-backed HPO term suggestions for research text |
| `phentrieve.extract_hpo_terms_llm` | LLM-assisted full-text research annotation and grounded HPO mapping suggestions |
| `phentrieve.search_hpo_terms` | Candidate HPO terms for a short phrase |
| `phentrieve.compare_hpo_terms` | Similarity between two HPO IDs |
| `phentrieve.get_server_capabilities` | Discover supported languages, models, backends, and research-use limitations |
```

- [x] **Step 5: Update Docker comments**

In `docker-compose.yml`, change the MCP comment to:

```yaml
# MCP Streamable HTTP endpoint
# URL: https://your-domain.com/mcp
```

In `api/Dockerfile`, update the MCP install comment to:

```dockerfile
# Install MCP dependencies for the Streamable HTTP endpoint at /mcp
```

- [x] **Step 6: Run docs checks available in repo**

Run:

```bash
make check
```

Expected: pass.

- [x] **Step 7: Commit**

```bash
git add docs/mcp-server.md docs/prompts/claude-desktop-hpo-annotator.md docker-compose.yml api/Dockerfile
git commit -m "docs: document modern http mcp usage"
```

---

## Task 10: Final Verification

**Files:**

- Verify only.

- [x] **Step 1: Run targeted MCP tests**

Run:

```bash
uv run --extra mcp pytest tests/unit/mcp tests/integration/test_mcp_http_protocol.py -q
```

Expected: all tests pass.

- [x] **Step 2: Run API schema and router tests**

Run:

```bash
uv run pytest tests/unit/api/test_schemas.py tests/unit/api/test_text_processing_router.py -q
```

Expected: all tests pass.

- [x] **Step 3: Run required repository checks**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected: all checks pass.

- [x] **Step 4: Manual MCP Inspector smoke test**

Run the HTTP MCP server:

```bash
make mcp-serve-http
```

In another terminal, run MCP Inspector according to the installed MCP tooling:

```bash
uv run --extra mcp mcp dev http://127.0.0.1:8734/mcp
```

Expected:

- Server initializes successfully.
- `tools/list` shows first-class `phentrieve.*` tools.
- `resources/list` shows Phentrieve resources.
- `prompts/list` shows annotation prompts.
- `phentrieve.get_server_capabilities` returns JSON with `standard` and `llm` backends.

- [x] **Step 5: Commit verification notes**

Create `.planning/archived/2026-04-29-modern-mcp-verification.md` with the exact commands run and outcomes.

Commit:

```bash
git add .planning/archived/2026-04-29-modern-mcp-verification.md
git commit -m "test: record modern mcp verification"
```

---

## Migration Strategy

### Phase 1: Compatibility Fix

Complete Tasks 1-4. This keeps `fastapi-mcp` as the serving mechanism but fixes the transport and tool metadata enough to unblock modern HTTP clients.

### Phase 2: Product-Grade MCP

Complete Tasks 5-8. This makes MCP an explicit Phentrieve interface with first-class LLM extraction, resources, prompts, and stable schemas.

### Phase 3: Docs and Release

Complete Tasks 9-10. This updates user-facing setup instructions and verifies behavior through unit, integration, and manual MCP client testing.

---

## Acceptance Criteria

- `/mcp` is Streamable HTTP, not legacy SSE.
- No SSE endpoint is mounted or documented.
- `tools/list` includes clear titles, action-oriented descriptions, and read-only annotations.
- Full-text LLM extraction is exposed as an explicit MCP tool.
- LLM provider, model, base URL, mode, internal mode, and fallback behavior are controllable through tool arguments.
- Tool outputs are structured JSON suitable for model follow-up turns.
- Resources expose capabilities, supported languages, profiles, and schemas.
- Prompts expose common annotation workflows.
- Tools, prompts, resources, docs, and server instructions consistently state the research-only intended use and avoid affirmative diagnostic, treatment, triage, patient-management, or clinical decision-support claims.
- Public-demo-facing MCP metadata warns against submitting identifiable patient data.
- Claude HTTP and ChatGPT developer mode setup are documented.
- MCP unit and integration tests cover transport, metadata, tool mapping, resources, prompts, and HTTP protocol smoke behavior.
- Required repo checks pass: `make check`, `make typecheck-fast`, and `make test`.

---

## Self-Review

Spec coverage:

- HTTP transport modernization is covered by Tasks 1, 2, 8, and 9.
- Tool discoverability is covered by Tasks 3, 5, and 6.
- LLM full-text functionality is covered by Tasks 4, 5, and 6.
- Resources and prompts are covered by Task 7.
- Consumer documentation is covered by Task 9.
- Verification is covered by Task 10.

Placeholder scan:

- The plan contains no `TBD`, no deferred code markers, and no empty test instructions.
- Task 6 defines the dependency-injected helper signatures before asking the implementer to bind them to existing service helpers.

Type consistency:

- LLM request fields use `str | None` and `Literal` types consistent with the rest of the API schema.
- Tool names consistently use the `phentrieve.*` prefix.
- The Streamable HTTP endpoint remains `/mcp` throughout the plan.
