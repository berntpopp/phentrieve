from collections.abc import Callable
from typing import Any

import pytest

import api.dependencies as dependencies
from phentrieve.config import DEFAULT_MODEL

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_dependency_caches():
    dependencies.LOADED_RETRIEVERS.clear()
    dependencies.LOADED_SBERT_MODELS.clear()
    yield
    dependencies.LOADED_RETRIEVERS.clear()
    dependencies.LOADED_SBERT_MODELS.clear()


@pytest.mark.asyncio
async def test_dense_retriever_cache_miss_constructs_retriever_in_threadpool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sbert_model = object()
    retriever = object()
    sbert_calls: list[dict[str, Any]] = []
    constructor_calls: list[dict[str, Any]] = []
    threadpool_calls: list[Callable[[], Any]] = []

    async def fake_get_sbert_model_dependency(**kwargs: Any) -> object:
        sbert_calls.append(kwargs)
        return sbert_model

    def fake_from_model_name(**kwargs: Any) -> object:
        constructor_calls.append(kwargs)
        return retriever

    async def fake_run_in_threadpool(func: Callable[[], Any]) -> object:
        threadpool_calls.append(func)
        return func()

    monkeypatch.setattr(
        dependencies,
        "get_sbert_model_dependency",
        fake_get_sbert_model_dependency,
    )
    monkeypatch.setattr(
        dependencies.DenseRetriever,
        "from_model_name",
        fake_from_model_name,
    )
    monkeypatch.setattr(dependencies, "run_in_threadpool", fake_run_in_threadpool)

    result = await dependencies.get_dense_retriever_dependency(
        DEFAULT_MODEL,
        multi_vector=False,
    )

    assert result is retriever
    assert len(threadpool_calls) == 1
    assert sbert_calls == [
        {
            "model_name_requested": DEFAULT_MODEL,
            "trust_remote_code": True,
        }
    ]
    assert constructor_calls == [
        {
            "model": sbert_model,
            "model_name": DEFAULT_MODEL,
            "multi_vector": False,
        }
    ]


@pytest.mark.asyncio
async def test_dense_retriever_cache_hit_does_not_use_threadpool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_retriever = object()
    cache_key = f"retriever_for_{DEFAULT_MODEL}_multi=True"
    dependencies.LOADED_RETRIEVERS[cache_key] = cached_retriever

    async def fail_get_sbert_model_dependency(**_kwargs: Any) -> object:
        raise AssertionError("cached retriever should not load an SBERT model")

    def fail_from_model_name(**_kwargs: Any) -> object:
        raise AssertionError("cached retriever should not be reconstructed")

    async def fail_run_in_threadpool(_func: Callable[[], Any]) -> object:
        raise AssertionError("cached retriever should not use the threadpool")

    monkeypatch.setattr(
        dependencies,
        "get_sbert_model_dependency",
        fail_get_sbert_model_dependency,
    )
    monkeypatch.setattr(
        dependencies.DenseRetriever, "from_model_name", fail_from_model_name
    )
    monkeypatch.setattr(dependencies, "run_in_threadpool", fail_run_in_threadpool)

    result = await dependencies.get_dense_retriever_dependency(DEFAULT_MODEL)

    assert result is cached_retriever
