"""The extraction benchmark must connect its retriever with the configured
``multi_vector`` mode so it can use the (multi-vector) bundle indexes that the
rest of the system defaults to. The per-chunk query path then auto-dispatches to
multi-vector aggregation via ``retriever.detect_index_type()``.
"""

from phentrieve.benchmark.extraction_benchmark import ExtractionConfig, HPOExtractor
from phentrieve.config import DEFAULT_MULTI_VECTOR


def test_config_defaults_multi_vector_to_project_default():
    assert ExtractionConfig().multi_vector == DEFAULT_MULTI_VECTOR


def _patch_heavy(monkeypatch, captured):
    import phentrieve.embeddings as emb
    import phentrieve.retrieval.dense_retriever as dr
    import phentrieve.text_processing.pipeline as pl

    monkeypatch.setattr(emb, "load_embedding_model", lambda name: object())
    monkeypatch.setattr(pl, "TextProcessingPipeline", lambda **kw: object())

    def fake_from_model_name(*, model, model_name, multi_vector=False, **kw):
        captured["multi_vector"] = multi_vector
        return object()

    monkeypatch.setattr(
        dr.DenseRetriever, "from_model_name", staticmethod(fake_from_model_name)
    )


def test_lazy_init_passes_multi_vector_true(monkeypatch):
    captured: dict[str, object] = {}
    _patch_heavy(monkeypatch, captured)
    HPOExtractor(ExtractionConfig(multi_vector=True))._lazy_init()
    assert captured["multi_vector"] is True


def test_lazy_init_passes_multi_vector_false(monkeypatch):
    captured: dict[str, object] = {}
    _patch_heavy(monkeypatch, captured)
    HPOExtractor(ExtractionConfig(multi_vector=False))._lazy_init()
    assert captured["multi_vector"] is False
