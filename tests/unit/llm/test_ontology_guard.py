"""Ontology guard tests (extraction contract v2 hardening).

The guard drops resolved terms KNOWN to sit outside Phenotypic abnormality
(HP:0000118) -- clinical modifiers / course / inheritance -- which are never
valid standalone phenotype annotations, while failing open on unknown ancestry
so a real finding is never dropped.
"""

from __future__ import annotations

import phentrieve.llm.ontology_guard as ontology_guard
from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.types import LLMPhenotype, LLMPipelineConfig


def _guard_with_map(monkeypatch, ancestors: dict[str, set[str]]) -> None:
    monkeypatch.setattr(
        ontology_guard,
        "_ancestors_map",
        lambda: {t: frozenset(a) for t, a in ancestors.items()},
    )


def test_flags_clinical_modifier_and_keeps_phenotype(monkeypatch):
    _guard_with_map(
        monkeypatch,
        {
            "HP:0012825": {"HP:0000001", "HP:0012823"},  # Mild -> Clinical modifier
            "HP:0001250": {"HP:0000001", "HP:0000118"},  # Seizure -> phenotype
        },
    )
    assert ontology_guard.is_non_phenotypic_abnormality("HP:0012825") is True
    assert ontology_guard.is_non_phenotypic_abnormality("HP:0001250") is False


def test_fails_open_on_unknown_or_root(monkeypatch):
    _guard_with_map(monkeypatch, {})
    # Unknown ancestry -> kept (never silently drop a real finding).
    assert ontology_guard.is_non_phenotypic_abnormality("HP:9999999") is False
    # The phenotype root itself is kept.
    assert ontology_guard.is_non_phenotypic_abnormality("HP:0000118") is False
    assert ontology_guard.is_non_phenotypic_abnormality("") is False


def test_empty_load_is_not_cached_and_recovers(monkeypatch, tmp_path):
    """Regression guard (Codex High): a fail-open empty/failed load must NOT be
    memoized -- otherwise the first call before hpo_data.db is present latches the
    guard OFF for the whole process. It must recover once the DB appears."""
    og = ontology_guard
    og._ANCESTORS_CACHE.clear()
    monkeypatch.setattr(og, "resolve_data_path", lambda *a, **k: tmp_path)

    db_file = tmp_path / "hpo_data.db"
    # 1) DB absent -> fail open, and DO NOT cache.
    assert og._ancestors_map() == {}
    assert og._ANCESTORS_CACHE == {}

    # 2) DB now present -> loads and caches by path.
    db_file.write_text("")
    monkeypatch.setattr(
        og,
        "_load_ancestors_map",
        lambda path: {"HP:0012825": frozenset({"HP:0012823"})},
    )
    assert og._ancestors_map() == {"HP:0012825": frozenset({"HP:0012823"})}
    assert str(db_file) in og._ANCESTORS_CACHE


def test_pipeline_drops_non_phenotypic_terms(monkeypatch):
    _guard_with_map(
        monkeypatch,
        {
            "HP:0012825": {"HP:0000001", "HP:0012823"},  # Mild (modifier) -> drop
            "HP:0001250": {"HP:0000001", "HP:0000118"},  # Seizure -> keep
        },
    )
    terms = [
        LLMPhenotype(term_id="HP:0001250", label="Seizure", evidence="seizures"),
        LLMPhenotype(term_id="HP:0012825", label="Mild", evidence="normal"),
    ]
    kept = TwoPhaseLLMPipeline._drop_non_phenotypic(terms)

    assert [t.term_id for t in kept] == ["HP:0001250"]


def test_pipeline_counts_guard_drops_and_traces_availability(monkeypatch):
    # Drops are counted and the guard state is traced, not only logged (#493).
    _guard_with_map(monkeypatch, {"HP:9000001": {"HP:0000001", "HP:0012823"}})

    class _Provider:
        last_usage: dict = {}
        last_request_count = 1

        def run_structured_prompt(self, *, response_model, **_kwargs):
            return response_model.model_validate(
                {"phenotypes": [{"phrase": "term-a", "category": "Abnormal"}]}
            )

    class _Retriever:
        def query_batch_hpo_terms(self, *, phrases, language, n_results):
            candidate = {"hpo_id": "HP:9000001", "term_name": "term-a", "score": 0.99}
            return [{"phrase": phrase, "candidates": [candidate]} for phrase in phrases]

    pipeline = TwoPhaseLLMPipeline(provider=_Provider(), tool_executor=_Retriever())
    result = pipeline.run(
        text="term-a",
        config=LLMPipelineConfig(model="fake", mode="two_phase", language="en"),
    )

    assert result.terms == []
    assert result.meta.phase_counts["ontology_guard_dropped_terms"] == 1
    assert result.meta.trace["ontology_guard"] == {
        "available": True,
        "dropped_terms": 1,
    }


def test_guard_reports_unavailable_graph(monkeypatch):
    _guard_with_map(monkeypatch, {})

    assert ontology_guard.guard_available() is False
