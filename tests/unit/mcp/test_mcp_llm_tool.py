def _ensure_external_mcp_sdk() -> None:
    import sys
    from pathlib import Path

    removed_paths = [
        path for path in sys.path if Path(path).as_posix().endswith("/tests/unit")
    ]
    for path in removed_paths:
        sys.path.remove(path)
    sys.modules.pop("mcp", None)

    try:
        import mcp.server.fastmcp  # noqa: F401
    finally:
        sys.path[:0] = removed_paths


def test_llm_tool_maps_request_to_full_text_service(monkeypatch) -> None:
    _ensure_external_mcp_sdk()

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
