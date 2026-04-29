from __future__ import annotations

import os
from typing import Any

from phentrieve.llm.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_PROVIDER_NAME,
)

PUBLIC_DEMO_DATA_NOTICE = (
    "Do not submit identifiable patient data to public demo instances."
)


def get_llm_capability_defaults() -> dict[str, Any]:
    default_model = os.getenv("PHENTRIEVE_LLM_MODEL", DEFAULT_LLM_MODEL)
    return {
        "recommended_backend_for_full_text": "llm",
        "default_llm_provider": os.getenv(
            "PHENTRIEVE_LLM_PROVIDER", DEFAULT_PROVIDER_NAME
        ),
        "default_llm_model": default_model,
        "configured_llm_models": [default_model],
        "llm_guidance": (
            "Prefer phentrieve.extract_hpo_terms_llm for full abstracts, "
            "publication-style annotation, syndrome/eponym-heavy text, and review "
            "work where retrieval-only noise should be suppressed. Use standard "
            "extraction for quick deterministic screening."
        ),
    }


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
        **get_llm_capability_defaults(),
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
                "use_when": (
                    "quick deterministic local retrieval-backed screening; not "
                    "preferred for full abstracts when LLM access is configured"
                ),
            },
            {
                "name": "llm_full_text",
                "backend": "llm",
                "use_when": (
                    "preferred for full abstracts, publication-style annotation, "
                    "syndrome/eponym-heavy text, and document-level extraction "
                    "with LLM phrase identification"
                ),
            },
        ],
        **get_llm_capability_defaults(),
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
