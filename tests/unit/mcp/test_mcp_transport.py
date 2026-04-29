from fastapi import FastAPI


def _noop() -> None:
    return None


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
    import sys
    from unittest.mock import patch

    class FakeFastApiMCP:
        def __init__(self, app: FastAPI, **_: object) -> None:
            self.app = app

        def mount_http(self, *, mount_path: str = "/mcp") -> None:
            self.app.add_api_route(mount_path, _noop, methods=["GET"])
            self.app.add_api_route(mount_path, _noop, methods=["POST"])
            self.app.add_api_route(mount_path, _noop, methods=["DELETE"])

    modules_to_clear = [key for key in sys.modules if key.startswith("api.mcp")]
    for module_name in modules_to_clear:
        del sys.modules[module_name]

    fake_fastapi_mcp = type("FakeFastApiMcpModule", (), {"FastApiMCP": FakeFastApiMCP})
    with patch.dict(sys.modules, {"fastapi_mcp": fake_fastapi_mcp}):
        from api.mcp.server import create_mcp_server, mount_mcp_http

        app = FastAPI(title="MCP test")
        mcp = create_mcp_server(app)

        mount_mcp_http(mcp)

    assert ("GET", "/mcp") in _mcp_routes(app)
    assert ("POST", "/mcp") in _mcp_routes(app)
    assert ("DELETE", "/mcp") in _mcp_routes(app)
    assert ("POST", "/mcp/messages/") not in _mcp_routes(app)
