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


def test_resource_payloads_are_available() -> None:
    from api.mcp.resources import get_capabilities_resource, get_languages_resource

    capabilities = get_capabilities_resource()
    languages = get_languages_resource()

    assert capabilities["server"] == "phentrieve"
    assert "llm" in capabilities["extraction_backends"]
    assert capabilities["recommended_backend_for_full_text"] == "llm"
    assert capabilities["default_llm_provider"]
    assert capabilities["default_llm_model"]
    assert capabilities["configured_llm_models"] == [capabilities["default_llm_model"]]
    assert "supported_llm_providers" not in capabilities
    assert "en" in languages["supported_languages"]
    assert "de" in languages["supported_languages"]
    assert "identifiable patient data" in " ".join(capabilities["not_intended_for"])


def test_server_capabilities_list_configured_llm_defaults() -> None:
    _ensure_external_mcp_sdk()

    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    result = mcp._tool_manager._tools["phentrieve.get_server_capabilities"].fn()

    assert result["recommended_backend_for_full_text"] == "llm"
    assert result["default_llm_provider"]
    assert result["default_llm_model"]
    assert result["configured_llm_models"] == [result["default_llm_model"]]
    assert "supported_llm_providers" not in result
    assert "credential_environment_variables" not in result


def test_prompt_templates_are_short_actionable_and_research_only() -> None:
    from api.mcp.prompts import annotate_research_text_prompt

    prompt = annotate_research_text_prompt(language="en")

    assert "phentrieve.extract_hpo_terms" in prompt
    assert "phentrieve.extract_hpo_terms_llm" in prompt
    assert "compare_standard_vs_llm_extraction" not in prompt
    assert "Research use only" in prompt
    assert "not for diagnosis" in prompt
    assert "Do not submit identifiable patient data" in prompt
    assert len(prompt) < 3000


def test_benchmark_comparison_is_not_an_mcp_prompt() -> None:
    _ensure_external_mcp_sdk()

    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    prompt_names = set(mcp._prompt_manager._prompts.keys())
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "compare_standard_vs_llm_extraction" not in prompt_names
    assert "phentrieve.compare_standard_vs_llm_extraction" not in tool_names
