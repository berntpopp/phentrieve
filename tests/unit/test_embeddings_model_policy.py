"""Tests for embedding model trust policy behavior."""

from unittest.mock import MagicMock

import pytest

from phentrieve.embeddings import clear_model_registry, load_embedding_model

pytestmark = pytest.mark.unit


def test_biolord_like_name_does_not_imply_trust_remote_code(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}
    fake_model = MagicMock()
    fake_model.device = "cpu"

    def fake_sentence_transformer(model_name: str, **kwargs):
        captured_kwargs["model_name"] = model_name
        captured_kwargs.update(kwargs)
        return fake_model

    monkeypatch.setattr(
        "phentrieve.embeddings.SentenceTransformer",
        fake_sentence_transformer,
    )
    clear_model_registry()

    load_embedding_model(
        model_name="attacker/BioLORD-remote-code",
        trust_remote_code=False,
        device="cpu",
        force_reload=True,
    )

    assert captured_kwargs["model_name"] == "attacker/BioLORD-remote-code"
    assert captured_kwargs.get("trust_remote_code") is not True
