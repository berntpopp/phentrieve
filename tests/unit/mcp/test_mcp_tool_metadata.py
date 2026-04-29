from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


@dataclass
class FakeTool:
    name: str
    title: str | None = None
    description: str | None = None
    annotations: object | None = None
    inputSchema: dict[str, object] | None = None

    def model_copy(self, *, update: dict[str, object]) -> "FakeTool":
        copied = replace(self)
        for key, value in update.items():
            setattr(copied, key, value)
        return copied


def _fake_mcp_module() -> object:
    class FakeFastApiMCP:
        def __init__(self, *_: object, **__: object) -> None:
            from api.schemas.text_processing_schemas import TextProcessingRequest

            self.tools = [
                FakeTool("query_hpo_terms"),
                FakeTool(
                    "process_clinical_text",
                    inputSchema=TextProcessingRequest.model_json_schema(),
                ),
                FakeTool("calculate_term_similarity"),
            ]

    return SimpleNamespace(FastApiMCP=FakeFastApiMCP)


def _fake_mcp_types_module() -> object:
    class ToolAnnotations:
        def __init__(
            self,
            *,
            readOnlyHint: bool,
            destructiveHint: bool,
            idempotentHint: bool,
            openWorldHint: bool,
        ) -> None:
            self.readOnlyHint = readOnlyHint
            self.destructiveHint = destructiveHint
            self.idempotentHint = idempotentHint
            self.openWorldHint = openWorldHint

    return SimpleNamespace(ToolAnnotations=ToolAnnotations)


def test_tools_have_modern_metadata() -> None:
    import sys

    from api.main import create_app

    app = create_app()
    modules_to_clear = [key for key in sys.modules if key.startswith("api.mcp")]
    for module_name in modules_to_clear:
        del sys.modules[module_name]

    with patch.dict(
        sys.modules,
        {
            "fastapi_mcp": _fake_mcp_module(),
            "mcp": MagicMock(types=_fake_mcp_types_module()),
        },
    ):
        from api.mcp.server import create_mcp_server

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
        assert "Research use only" in tool.description
        assert "Do not submit identifiable patient data" in tool.description
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False


def test_process_clinical_text_description_mentions_llm_backend() -> None:
    import sys

    from api.main import create_app

    app = create_app()
    modules_to_clear = [key for key in sys.modules if key.startswith("api.mcp")]
    for module_name in modules_to_clear:
        del sys.modules[module_name]

    with patch.dict(
        sys.modules,
        {
            "fastapi_mcp": _fake_mcp_module(),
            "mcp": MagicMock(types=_fake_mcp_types_module()),
        },
    ):
        from api.mcp.server import create_mcp_server

        mcp = create_mcp_server(app)
    tools = {tool.name: tool for tool in mcp.tools}

    description = tools["process_clinical_text"].description

    assert "extraction_backend='llm'" in description
    assert "full-text LLM" in description


def test_process_clinical_text_schema_exposes_llm_controls() -> None:
    import sys

    from api.main import create_app

    app = create_app()
    modules_to_clear = [key for key in sys.modules if key.startswith("api.mcp")]
    for module_name in modules_to_clear:
        del sys.modules[module_name]

    with patch.dict(
        sys.modules,
        {
            "fastapi_mcp": _fake_mcp_module(),
            "mcp": MagicMock(types=_fake_mcp_types_module()),
        },
    ):
        from api.mcp.server import create_mcp_server

        mcp = create_mcp_server(app)
    tools = {tool.name: tool for tool in mcp.tools}
    schema = tools["process_clinical_text"].inputSchema
    assert schema is not None
    request_properties = schema["properties"]

    assert "llm_provider" in request_properties
    assert "llm_base_url" in request_properties
    assert "llm_internal_mode" in request_properties
    assert "allow_standard_fallback" in request_properties
