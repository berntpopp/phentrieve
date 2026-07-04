"""Unit tests for api.mcp.service_adapters domain logic."""

from __future__ import annotations

import json

import pytest

from api.mcp import service_adapters
from api.mcp.envelope import McpToolError
from api.mcp.service_adapters import (
    _coerce_export_phenotype,
    chunk_text_service,
    export_phenopacket_service,
    extract_hpo_terms_llm_service,
    extract_hpo_terms_service,
)
from api.schemas.phenopacket_schemas import ExportPhenotypeRequest
from phentrieve.config import DEFAULT_MODEL


def test_chunk_text_service_simple_strategy():
    out = chunk_text_service(
        text="First sentence. Second sentence.", language="en", strategy="simple"
    )
    assert out["chunk_count"] >= 1
    assert out["chunks"][0]["chunk_id"] == 1
    assert "text" in out["chunks"][0]


def test_chunk_text_service_unknown_strategy_raises_validation_failed():
    with pytest.raises(McpToolError) as ei:
        chunk_text_service(text="x", language="en", strategy="does-not-exist")
    # Unknown strategies are now rejected explicitly with the valid list (L4)
    # instead of silently falling back to the default config.
    assert ei.value.error_code == "validation_failed"
    assert "simple" in str(ei.value)


def _stub_pipeline(monkeypatch, captured):
    """Replace TextProcessingPipeline with a recording fake that yields chunks."""

    class _FakePipeline:
        def __init__(self, **kwargs):
            captured["sbert"] = kwargs.get("sbert_model_for_semantic_chunking")

        def process(self, text, include_positions=False):
            captured["processed"] = True
            return [{"text": "chunk one", "start_char": 0, "end_char": 9}]

    monkeypatch.setattr(service_adapters, "TextProcessingPipeline", _FakePipeline)


def test_chunk_text_model_dependent_strategy_lazy_loads_model(monkeypatch):
    """B2: a strategy whose resolved config contains a sliding_window stage must
    lazy-load the cached embedding singleton and thread it into the pipeline
    instead of hard-failing with invalid_input."""
    calls: list[str] = []
    captured: dict[str, object] = {}

    def fake_load(model_name):
        calls.append(model_name)
        return "FAKE_MODEL"

    monkeypatch.setattr("phentrieve.embeddings.load_embedding_model", fake_load)
    _stub_pipeline(monkeypatch, captured)

    out = chunk_text_service(
        text="some clinical text", language="en", strategy="sliding_window"
    )

    assert calls == [DEFAULT_MODEL]
    assert captured["sbert"] == "FAKE_MODEL"
    assert out["chunk_count"] == 1
    assert out["chunks"][0]["chunk_id"] == 1


def test_chunk_text_simple_strategy_does_not_load_model(monkeypatch):
    """B2: the model-free 'simple' strategy must not trigger a model load."""
    calls: list[str] = []
    captured: dict[str, object] = {}

    def fake_load(model_name):
        calls.append(model_name)
        return "FAKE_MODEL"

    monkeypatch.setattr("phentrieve.embeddings.load_embedding_model", fake_load)
    _stub_pipeline(monkeypatch, captured)

    chunk_text_service(text="some clinical text", language="en", strategy="simple")

    assert calls == []
    assert captured["sbert"] is None


def test_chunk_text_model_load_failure_is_temporarily_unavailable(monkeypatch):
    """B2: a genuine model-load failure must surface as temporarily_unavailable
    (retryable), not invalid_input (which blames the caller's strategy)."""

    def boom(model_name):
        raise RuntimeError("model weights unavailable")

    monkeypatch.setattr("phentrieve.embeddings.load_embedding_model", boom)

    with pytest.raises(McpToolError) as ei:
        chunk_text_service(text="x", language="en", strategy="sliding_window")
    assert ei.value.error_code == "temporarily_unavailable"


def test_diagnostics_reports_cold_then_loaded_subsystems(monkeypatch):
    """D3: diagnostics probes the live caches -- embedding_model/vector_index
    read 'cold' before any load and 'loaded' after a search has warmed them,
    instead of a constant 'lazy'."""
    from api import dependencies
    from phentrieve.config import DEFAULT_MODEL

    # Keep the ontology probe deterministic and fast.
    monkeypatch.setattr(service_adapters, "_probe_ontology_data", lambda: "ok")

    # Cold: empty caches.
    monkeypatch.setattr(dependencies, "LOADED_SBERT_MODELS", {}, raising=False)
    monkeypatch.setattr(dependencies, "LOADED_RETRIEVERS", {}, raising=False)
    monkeypatch.setattr(dependencies, "MODEL_LOADING_STATUS", {}, raising=False)
    monkeypatch.setattr("phentrieve.embeddings._MODEL_REGISTRY", {}, raising=False)

    cold = service_adapters.diagnostics_service()
    assert cold["subsystems"]["embedding_model"] == "cold"
    assert cold["subsystems"]["vector_index"] == "cold"
    assert cold["status"] == "ok"  # cold is not an error

    # Warm: a search has populated the dependency caches.
    monkeypatch.setattr(
        dependencies, "LOADED_SBERT_MODELS", {DEFAULT_MODEL: object()}, raising=False
    )
    monkeypatch.setattr(
        dependencies, "LOADED_RETRIEVERS", {("idx",): object()}, raising=False
    )

    warm = service_adapters.diagnostics_service()
    assert warm["subsystems"]["embedding_model"] == "loaded"
    assert warm["subsystems"]["vector_index"] == "loaded"


def test_diagnostics_reports_loading_state(monkeypatch):
    """D3: a model mid-load reports 'loading', not 'cold' or 'loaded'."""
    from api import dependencies
    from phentrieve.config import DEFAULT_MODEL

    monkeypatch.setattr(service_adapters, "_probe_ontology_data", lambda: "ok")
    monkeypatch.setattr(dependencies, "LOADED_SBERT_MODELS", {}, raising=False)
    monkeypatch.setattr(dependencies, "LOADED_RETRIEVERS", {}, raising=False)
    monkeypatch.setattr(
        dependencies, "MODEL_LOADING_STATUS", {DEFAULT_MODEL: "loading"}, raising=False
    )
    monkeypatch.setattr("phentrieve.embeddings._MODEL_REGISTRY", {}, raising=False)

    out = service_adapters.diagnostics_service()
    assert out["subsystems"]["embedding_model"] == "loading"


def test_export_phenopacket_service_round_trips_case_id():
    out = export_phenopacket_service(
        case_id="CASE-1",
        case_label="demo",
        input_text=None,
        subject=None,
        phenotypes=[
            {"hpo_id": "HP:0001250", "label": "Seizure", "assertion": "affirmed"},
            {
                "hpo_id": "HP:0001263",
                "label": "Global developmental delay",
                "assertion": "negated",
            },
        ],
        include_annotation_sidecar=True,
    )
    assert "phenopacket_json" in out
    packet = json.loads(out["phenopacket_json"])
    assert packet["id"] == "CASE-1"


def test_export_excludes_family_history_phenotypes():
    """LLM-1: a family-history mention is not a proband phenotypic feature and
    must not be folded into an affirmed feature on the subject."""
    out = export_phenopacket_service(
        case_id="CASE-2",
        case_label="demo",
        input_text=None,
        subject=None,
        phenotypes=[
            {"hpo_id": "HP:0001250", "label": "Seizure", "assertion": "affirmed"},
            {
                "hpo_id": "HP:0000365",
                "label": "Hearing loss",
                "assertion": "family_history",
            },
        ],
        include_annotation_sidecar=False,
    )
    packet = json.loads(out["phenopacket_json"])
    feature_ids = {feat["type"]["id"] for feat in packet.get("phenotypicFeatures", [])}
    assert "HP:0001250" in feature_ids
    assert "HP:0000365" not in feature_ids


def test_coerce_drops_family_history_experiencer_even_with_present_assertion():
    """B2: experiencer, not assertion, now carries family-ness (B1 split them
    into separate axes). A family-history mention must be dropped even when
    its assertion is a normal, non-family value like 'present'."""
    out = _coerce_export_phenotype(
        ExportPhenotypeRequest,
        {
            "hpo_id": "HP:0000365",
            "label": "Hearing loss",
            "assertion": "present",
            "experiencer": "family_history",
        },
        0,
    )
    assert out is None


def test_coerce_keeps_proband_experiencer():
    """Regression: a proband term (explicit or absent experiencer) is not
    dropped by the experiencer-based guard."""
    for experiencer in ("proband", None):
        p = {"hpo_id": "HP:0001250", "label": "Seizure", "assertion": "present"}
        if experiencer is not None:
            p["experiencer"] = experiencer
        out = _coerce_export_phenotype(ExportPhenotypeRequest, p, 0)
        assert out is not None
        assert out.hpo_id == "HP:0001250"


def test_export_excludes_family_history_via_experiencer_mixed_proband():
    """Integration-style: export_phenopacket_service over a mixed
    proband+family input must yield zero family terms in the subject's
    PhenotypicFeatures, keyed off experiencer rather than assertion."""
    out = export_phenopacket_service(
        case_id="CASE-3",
        case_label="demo",
        input_text=None,
        subject=None,
        phenotypes=[
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "assertion": "present",
                "experiencer": "proband",
            },
            {
                "hpo_id": "HP:0000365",
                "label": "Hearing loss",
                "assertion": "present",
                "experiencer": "family_history",
            },
        ],
        include_annotation_sidecar=False,
    )
    packet = json.loads(out["phenopacket_json"])
    feature_ids = {feat["type"]["id"] for feat in packet.get("phenotypicFeatures", [])}
    assert "HP:0001250" in feature_ids
    assert "HP:0000365" not in feature_ids


def test_absent_assertion_exports_as_negated_not_affirmed():
    out = _coerce_export_phenotype(
        ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": "absent"}, 0
    )
    assert out is not None and out.assertion_status == "negated"


def test_normal_assertion_exports_as_negated_excluded():
    # A normalcy verdict is a ruled-out abnormality -> excluded (negated).
    out = _coerce_export_phenotype(
        ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": "normal"}, 0
    )
    assert out is not None and out.assertion_status == "negated"


def test_uncertain_does_not_crash_and_is_not_excluded():
    out = _coerce_export_phenotype(
        ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": "uncertain"}, 0
    )
    # not excluded, and no ValidationError
    assert out.assertion_status == "affirmed"


def test_present_assertion_affirmed():
    out = _coerce_export_phenotype(
        ExportPhenotypeRequest, {"hpo_id": "HP:0001250", "assertion": "present"}, 0
    )
    assert out.assertion_status == "affirmed"


def test_extract_service_resolves_none_language():
    # language=None must be resolved to a concrete code before the chunking
    # pipeline runs (the conjunction chunker calls language.lower()).
    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

    extract_hpo_terms_service(
        text="The patient has seizures.",
        language=None,
        include_details=False,
        include_chunk_positions=False,
        num_results_per_chunk=5,
        chunk_retrieval_threshold=0.5,
        service=fake_service,
    )
    assert captured["language"] is not None
    assert isinstance(captured["language"], str) and captured["language"]


def test_extract_llm_fallback_passes_resolved_language(monkeypatch):
    import api.config as api_config

    monkeypatch.setattr(api_config, "PHENTRIEVE_ENV", "development", raising=False)
    seen = {}

    def flaky_service(**kwargs):
        if kwargs.get("extraction_backend") == "llm":
            raise RuntimeError("backend down")
        seen.update(kwargs)
        return {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

    extract_hpo_terms_llm_service(
        text="Seizures and ataxia.",
        language=None,
        include_details=True,
        include_chunk_positions=True,
        num_results_per_chunk=5,
        chunk_retrieval_threshold=0.5,
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        allow_standard_fallback=True,
        service=flaky_service,
    )
    assert seen.get("language")  # resolved, non-None, non-empty


def test_extract_service_uses_injected_service():
    captured = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        return {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

    out = extract_hpo_terms_service(
        text="hi",
        language="en",
        include_details=False,
        include_chunk_positions=False,
        num_results_per_chunk=5,
        chunk_retrieval_threshold=0.5,
        service=fake_service,
    )
    assert out["aggregated_hpo_terms"] == []
    assert captured["extraction_backend"] == "standard"
    assert captured["num_results_per_chunk"] == 5


def test_extract_llm_falls_back_on_backend_error_when_allowed(monkeypatch):
    import api.config as api_config

    monkeypatch.setattr(api_config, "PHENTRIEVE_ENV", "development", raising=False)

    def flaky_service(**kwargs):
        if kwargs.get("extraction_backend") == "llm":
            raise RuntimeError("backend down")
        return {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

    out = extract_hpo_terms_llm_service(
        text="hi",
        language="en",
        include_details=True,
        include_chunk_positions=True,
        num_results_per_chunk=5,
        chunk_retrieval_threshold=0.5,
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        allow_standard_fallback=True,
        service=flaky_service,
    )
    assert out["meta"]["fallback_reason"] == "llm_backend_error"


def test_extract_llm_reraises_backend_error_without_fallback(monkeypatch):
    import api.config as api_config

    monkeypatch.setattr(api_config, "PHENTRIEVE_ENV", "development", raising=False)

    def flaky_service(**kwargs):
        raise RuntimeError("backend down")

    with pytest.raises(RuntimeError):
        extract_hpo_terms_llm_service(
            text="hi",
            language="en",
            include_details=True,
            include_chunk_positions=True,
            num_results_per_chunk=5,
            chunk_retrieval_threshold=0.5,
            llm_mode="two_phase",
            llm_internal_mode="whole_document_grounded",
            allow_standard_fallback=False,
            service=flaky_service,
        )
