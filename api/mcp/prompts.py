from __future__ import annotations

RESEARCH_USE_NOTICE = (
    "Research use only; not for diagnosis, treatment, triage, patient "
    "management, or clinical decision support. Do not submit identifiable "
    "patient data to public demo instances."
)


def annotate_research_text_prompt(language: str = "en") -> str:
    return (
        f"{RESEARCH_USE_NOTICE} Map the supplied clinical or biomedical research "
        "text to Human Phenotype Ontology term suggestions. "
        f"Use language='{language}'. Prefer phentrieve.extract_hpo_terms_llm "
        "for full abstracts, publication-style annotation, syndrome/eponym-heavy "
        "text, and review work where retrieval-only noise should be suppressed. "
        "Use phentrieve.extract_hpo_terms for quick deterministic screening. "
        "Return HPO IDs, labels, assertion "
        "status, evidence spans, and a short research-use uncertainty note."
    )


def review_hpo_research_annotations_prompt() -> str:
    return (
        f"{RESEARCH_USE_NOTICE} Review the supplied HPO annotations against the "
        "research text evidence. Keep only HPO IDs returned by Phentrieve tools. "
        "Flag unsupported, negated, historical, or family-history-only phenotype "
        "suggestions."
    )


def extract_research_case_phenotypes_prompt(language: str = "en") -> str:
    return (
        f"{RESEARCH_USE_NOTICE} Extract phenotype suggestions from synthetic or "
        "research-consented case-report-like text. Exclude family history unless "
        "it is explicitly described as the research subject's phenotype. Prefer "
        f"phentrieve.extract_hpo_terms_llm with language='{language}' for long "
        "documents."
    )
