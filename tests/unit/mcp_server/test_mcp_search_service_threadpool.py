"""search_hpo_terms_service must not run blocking retrieval on the event loop (#405)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

import api.dependencies as dependencies
from api.mcp import service_adapters

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_search_hpo_terms_service_runs_retrieval_in_threadpool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    retriever = MagicMock()

    async def fake_get_dense_retriever_dependency(**_kwargs: Any) -> MagicMock:
        return retriever

    def fake_execute_hpo_retrieval_for_api(**kwargs: Any) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
            captured["off_loop"] = False
        except RuntimeError:
            captured["off_loop"] = True
        captured.update(kwargs)
        return {"results": [{"hpo_id": "HP:0001250", "label": "Seizure"}]}

    monkeypatch.setattr(
        dependencies,
        "get_dense_retriever_dependency",
        fake_get_dense_retriever_dependency,
    )
    monkeypatch.setattr(
        service_adapters,
        "execute_hpo_retrieval_for_api",
        fake_execute_hpo_retrieval_for_api,
    )

    out = await service_adapters.search_hpo_terms_service(
        text="seizures",
        language="en",
        num_results=5,
        similarity_threshold=0.3,
        include_details=False,
    )

    assert captured["off_loop"] is True
    assert captured["retriever"] is retriever
    assert out["results"][0]["hpo_id"] == "HP:0001250"
