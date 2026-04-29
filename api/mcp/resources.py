from __future__ import annotations

from typing import Any

PUBLIC_DEMO_DATA_NOTICE = (
    "Do not submit identifiable patient data to public demo instances."
)


def get_capabilities_resource() -> dict[str, Any]:
    return {
        "server": "phentrieve",
        "domain": "research-use phenotype annotation",
        "intended_use": (
            "Research, benchmarking, education, and research data standardisation."
        ),
        "not_intended_for": [
            "diagnosis",
            "treatment",
            "triage",
            "patient management",
            "clinical decision support",
            "use with identifiable patient data in public demo instances",
        ],
        "public_demo_data_notice": PUBLIC_DEMO_DATA_NOTICE,
        "ontology": "Human Phenotype Ontology",
        "transports": ["streamable_http"],
        "extraction_backends": ["standard", "llm"],
        "tools": [
            "phentrieve.extract_hpo_terms",
            "phentrieve.extract_hpo_terms_llm",
            "phentrieve.search_hpo_terms",
            "phentrieve.compare_hpo_terms",
        ],
    }


def get_languages_resource() -> dict[str, Any]:
    return {
        "supported_languages": ["en", "de", "es", "fr", "nl"],
        "default_language": "en",
        "language_parameter": "ISO 639-1 code",
        "public_demo_data_notice": PUBLIC_DEMO_DATA_NOTICE,
    }


def get_extraction_profiles_resource() -> dict[str, Any]:
    return {
        "profiles": [
            {
                "name": "standard",
                "backend": "standard",
                "use_when": "deterministic local retrieval-backed extraction",
            },
            {
                "name": "llm_full_text",
                "backend": "llm",
                "use_when": "document-level extraction with LLM phrase identification",
            },
        ],
        "public_demo_data_notice": PUBLIC_DEMO_DATA_NOTICE,
    }


def get_research_use_resource() -> dict[str, Any]:
    return {
        "intended_use": (
            "Phentrieve is open-source research software for mapping clinical "
            "or biomedical research text to Human Phenotype Ontology term "
            "suggestions for exploration, benchmarking, education, and research "
            "data standardisation."
        ),
        "not_intended_for": [
            "diagnosis",
            "treatment",
            "triage",
            "patient management",
            "clinical decision support",
        ],
        "public_demo_data_notice": PUBLIC_DEMO_DATA_NOTICE,
    }
