"""Ontology guard tests (extraction contract v2 hardening).

The guard drops resolved terms KNOWN to sit outside Phenotypic abnormality
(HP:0000118) -- clinical modifiers / course / inheritance -- which are never
valid standalone phenotype annotations, while failing open on unknown ancestry
so a real finding is never dropped.
"""

from __future__ import annotations

import phentrieve.llm.ontology_guard as ontology_guard
from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.types import LLMPhenotype


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
