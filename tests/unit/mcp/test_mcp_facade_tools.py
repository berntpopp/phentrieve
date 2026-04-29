import pytest


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


def test_facade_registers_first_class_tools() -> None:
    _ensure_external_mcp_sdk()

    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "phentrieve.extract_hpo_terms" in tool_names
    assert "phentrieve.extract_hpo_terms_llm" in tool_names
    assert "phentrieve.get_server_capabilities" in tool_names


def test_facade_registers_search_and_compare_tools() -> None:
    _ensure_external_mcp_sdk()

    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "phentrieve.search_hpo_terms" in tool_names
    assert "phentrieve.compare_hpo_terms" in tool_names


def test_search_hpo_terms_impl_delegates() -> None:
    _ensure_external_mcp_sdk()

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


@pytest.mark.asyncio
async def test_search_hpo_terms_service_uses_multi_vector(mocker) -> None:
    _ensure_external_mcp_sdk()

    from api.mcp.facade import _search_hpo_terms_service

    retriever = mocker.Mock()
    get_retriever = mocker.patch(
        "api.dependencies.get_dense_retriever_dependency",
        return_value=retriever,
    )
    execute_retrieval = mocker.patch(
        "api.mcp.facade.execute_hpo_retrieval_for_api",
        return_value={"results": []},
    )

    await _search_hpo_terms_service(
        text="seizures",
        language="en",
        num_results=3,
        similarity_threshold=0.1,
        include_details=False,
    )

    from api.mcp.facade import DEFAULT_MULTI_VECTOR

    assert get_retriever.call_args.kwargs["multi_vector"] is DEFAULT_MULTI_VECTOR
    assert execute_retrieval.call_args.kwargs["multi_vector"] is DEFAULT_MULTI_VECTOR


def test_compare_hpo_terms_impl_delegates() -> None:
    _ensure_external_mcp_sdk()

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
