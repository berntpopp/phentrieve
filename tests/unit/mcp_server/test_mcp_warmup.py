"""D9 -- best-effort startup warmup loads the model/index and never raises."""

import asyncio

import pytest

from api.mcp import facade

pytestmark = pytest.mark.unit


def test_warmup_swallows_failures(monkeypatch):
    calls = {"n": 0}

    async def boom() -> None:
        calls["n"] += 1
        raise RuntimeError("model load failed")

    monkeypatch.setattr(facade, "_warmup_retriever", boom)
    # Must not raise even though the underlying load failed.
    asyncio.run(facade.warmup())
    assert calls["n"] == 1


def test_warmup_invokes_retriever_loader(monkeypatch):
    calls = {"n": 0}

    async def ok() -> None:
        calls["n"] += 1

    monkeypatch.setattr(facade, "_warmup_retriever", ok)
    asyncio.run(facade.warmup())
    assert calls["n"] == 1
