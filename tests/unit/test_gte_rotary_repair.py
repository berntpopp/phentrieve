"""Regression tests for the GTE non-persistent buffer repair.

GTE's pinned remote implementation registers ``position_ids`` and the rotary
``inv_freq`` / ``cos_cached`` / ``sin_cached`` tables with ``persistent=False``, so
none of them are stored in the checkpoint. Transformers materializes the model
without re-running the module ``__init__`` that computes them, leaving the buffers
pointing at uninitialized memory.

The failure is allocation-dependent and silent in the worst case: zeroed pages give
an all-zero cosine table (a no-op rotation, so finite but wrong embeddings), while
reused pages give NaN and fail the index build outright.
"""

import math
from typing import Any

import pytest
import torch

from phentrieve.embeddings import _repair_gte_multilingual_position_ids

pytestmark = pytest.mark.unit

GTE = "Alibaba-NLP/gte-multilingual-base"
DIM = 8
MAX_POS = 16


class _StubRotary(torch.nn.Module):
    """Mimics the pinned RotaryEmbedding: non-persistent, __init__-computed buffers."""

    def __init__(self, dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, inv_freq.device, torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: Any) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class _StubEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.position_embedding_type = "rope"
        self.register_buffer("position_ids", torch.arange(MAX_POS), persistent=False)
        self._init_rope(None)

    def _init_rope(self, config: Any) -> None:
        self.rotary_emb = _StubRotary(DIM, MAX_POS, base=10000.0)


class _StubAutoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = object()
        self.embeddings = _StubEmbeddings()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _StubTransformer:
    def __init__(self) -> None:
        self.auto_model = _StubAutoModel()


class _StubModel:
    def __init__(self) -> None:
        self._module = _StubTransformer()

    def __getitem__(self, index: int) -> _StubTransformer:
        assert index == 0
        return self._module


def _corrupt(model: _StubModel, fill: float) -> None:
    """Simulate uninitialized memory landing in the non-persistent buffers."""
    rotary = model[0].auto_model.embeddings.rotary_emb
    rotary.cos_cached = torch.full_like(rotary.cos_cached, fill)
    rotary.sin_cached = torch.full_like(rotary.sin_cached, fill)
    rotary.inv_freq = torch.full_like(rotary.inv_freq, fill)


@pytest.mark.parametrize(
    ("fill", "label"),
    [
        (0.0, "zeroed pages -> degenerate all-zero rotation, silently wrong"),
        (float("nan"), "reused pages -> NaN, index build fails"),
    ],
)
def test_repair_restores_rotary_tables(fill: float, label: str) -> None:
    model = _StubModel()
    _corrupt(model, fill)

    _repair_gte_multilingual_position_ids(GTE, model)  # type: ignore[arg-type]

    rotary = model[0].auto_model.embeddings.rotary_emb
    cos = rotary.cos_cached.float()
    sin = rotary.sin_cached.float()

    assert torch.isfinite(cos).all(), label
    assert torch.isfinite(sin).all(), label
    # A real cosine table reaches 1.0 at position 0; an all-zero table does not.
    assert math.isclose(float(cos.abs().max()), 1.0, abs_tol=1e-5), label
    # The rotary identity must hold for the tables to be a valid rotation.
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5), label
    expected_inv_freq = 1.0 / (10000.0 ** (torch.arange(0, DIM, 2).float() / DIM))
    assert torch.allclose(rotary.inv_freq.float(), expected_inv_freq, atol=1e-6), label


def test_repair_rebuilds_position_ids() -> None:
    model = _StubModel()
    embeddings = model[0].auto_model.embeddings
    embeddings.position_ids = torch.zeros_like(embeddings.position_ids)

    _repair_gte_multilingual_position_ids(GTE, model)  # type: ignore[arg-type]

    assert torch.equal(embeddings.position_ids, torch.arange(MAX_POS))


def test_repair_is_scoped_to_gte() -> None:
    """Other models must not be touched; only GTE ships these broken buffers."""
    model = _StubModel()
    _corrupt(model, float("nan"))

    _repair_gte_multilingual_position_ids("FremyCompany/BioLORD-2023-M", model)  # type: ignore[arg-type]

    assert torch.isnan(model[0].auto_model.embeddings.rotary_emb.cos_cached).all()


def test_repair_raises_when_tables_stay_degenerate(monkeypatch) -> None:
    """A rebuild that silently fails must abort rather than poison an index."""
    model = _StubModel()

    def _broken_init_rope(config: Any) -> None:
        rotary = model[0].auto_model.embeddings.rotary_emb
        rotary.cos_cached = torch.zeros_like(rotary.cos_cached)
        rotary.sin_cached = torch.zeros_like(rotary.sin_cached)

    monkeypatch.setattr(model[0].auto_model.embeddings, "_init_rope", _broken_init_rope)

    with pytest.raises(ValueError, match="degenerate"):
        _repair_gte_multilingual_position_ids(GTE, model)  # type: ignore[arg-type]
