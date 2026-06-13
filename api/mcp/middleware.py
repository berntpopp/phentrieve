"""FastMCP middleware that wraps argument-binding failures in the error envelope.

FastMCP validates call arguments with pydantic inside the tool's ``run()`` --
before the registered tool body executes -- so a wrong argument name/type or a
missing required argument raises a ``pydantic.ValidationError`` that never reaches
``run_mcp_tool``'s error boundary. This middleware catches it at ``on_call_tool``
and returns the standard ``validation_failed`` envelope (valid names + a
did-you-mean). It also normalizes curated argument aliases (e.g. ``query`` ->
``text``) before dispatch and discloses any rewrite under
``_meta.argument_aliases_applied``.

Patterned after ../hgnc-link/hgnc_link/mcp/middleware.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp.server.middleware.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult
from mcp.types import CallToolRequestParams, TextContent
from pydantic import ValidationError

from api.mcp.arg_help import (
    describe_constraints,
    did_you_mean,
    normalize_alias_args,
    tool_signature,
)
from api.mcp.envelope import build_arg_error_envelope

logger = logging.getLogger("phentrieve.mcp")


class ArgValidationMiddleware(Middleware):
    """Reshape argument-binding errors into the envelope and apply argument aliases."""

    def __init__(self) -> None:
        self._schema_cache: dict[str, dict[str, Any]] = {}

    async def _schema(
        self, context: MiddlewareContext[Any], name: str
    ) -> dict[str, Any]:
        if name not in self._schema_cache:
            fctx = context.fastmcp_context
            if fctx is None:
                raise RuntimeError("no fastmcp context")
            tool = await fctx.fastmcp.get_tool(name)
            self._schema_cache[name] = dict(getattr(tool, "parameters", None) or {})
        return self._schema_cache[name]

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        name = context.message.name
        try:
            schema = await self._schema(context, name)
        except Exception:  # registry miss: let core handle the call untouched
            return await call_next(context)

        valid = list(schema.get("properties", {}).keys())
        new_args, applied = normalize_alias_args(valid, context.message.arguments or {})
        context.message.arguments = new_args

        try:
            result = await call_next(context)
        except ValidationError as exc:
            return self._error_result(name, valid, schema, exc)

        if (
            applied
            and isinstance(result, ToolResult)
            and isinstance(result.structured_content, dict)
        ):
            meta = result.structured_content.setdefault("_meta", {})
            meta["argument_aliases_applied"] = [list(pair) for pair in applied]
        return result

    def _error_result(
        self,
        name: str,
        valid: list[str],
        schema: dict[str, Any],
        exc: ValidationError,
    ) -> ToolResult:
        first = exc.errors(include_url=False)[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "input"
        error_type = str(first.get("type", "value_error"))
        constraints = None
        if loc in valid and error_type not in ("missing", "missing_argument"):
            field_schema = schema.get("properties", {}).get(loc, {})
            constraints = describe_constraints(field_schema)
        suggestion = did_you_mean(loc, valid) if loc not in valid else None
        envelope = build_arg_error_envelope(
            tool_name=name,
            loc=loc,
            error_type=error_type,
            valid_params=valid,
            signature=tool_signature(name, schema),
            suggestion=suggestion,
            constraints=constraints,
        )
        logger.warning("mcp_arg_error tool=%s loc=%s type=%s", name, loc, error_type)
        return ToolResult(
            structured_content=envelope,
            content=[TextContent(type="text", text=json.dumps(envelope))],
        )
