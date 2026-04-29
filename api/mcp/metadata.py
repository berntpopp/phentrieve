from __future__ import annotations

from dataclasses import dataclass

from mcp import types

RESEARCH_USE_LIMITATION = (
    "Research use only. Not for diagnosis, treatment, triage, patient "
    "management, clinical decision support, or use with identifiable patient "
    "data in public demo instances. Do not submit identifiable patient data "
    "to public demo instances."
)


@dataclass(frozen=True)
class ToolMetadata:
    title: str
    description: str


READ_ONLY_ANNOTATIONS = types.ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


TOOL_METADATA: dict[str, ToolMetadata] = {
    "query_hpo_terms": ToolMetadata(
        title="Search HPO Terms",
        description=(
            "Use this when a user provides a short phenotype phrase or research "
            "text snippet and needs candidate Human Phenotype Ontology terms by "
            "semantic similarity. Do not use this for full research notes; use "
            f"process_clinical_text instead. {RESEARCH_USE_LIMITATION}"
        ),
    ),
    "process_clinical_text": ToolMetadata(
        title="Extract HPO Terms From Research Text",
        description=(
            "Use this when a user asks to extract HPO term suggestions from "
            "clinical or biomedical research text, synthetic examples, or "
            "case-report-like research text. Use extraction_backend='standard' "
            "for deterministic retrieval-backed extraction. Use "
            "extraction_backend='llm' for full-text LLM extraction when the "
            "request asks for LLM-assisted interpretation, higher recall, or "
            f"document-level phenotype annotation. {RESEARCH_USE_LIMITATION}"
        ),
    ),
    "calculate_term_similarity": ToolMetadata(
        title="Compare HPO Terms",
        description=(
            "Use this when a user provides two HPO IDs and asks how similar or "
            "related the terms are for research analysis. Do not use this to "
            f"search for HPO terms from free text. {RESEARCH_USE_LIMITATION}"
        ),
    ),
}


def apply_tool_metadata(tools: list[types.Tool]) -> list[types.Tool]:
    patched: list[types.Tool] = []
    for tool in tools:
        metadata = TOOL_METADATA.get(tool.name)
        if metadata is None:
            patched.append(tool)
            continue
        patched.append(
            tool.model_copy(
                update={
                    "title": metadata.title,
                    "description": metadata.description,
                    "annotations": READ_ONLY_ANNOTATIONS,
                }
            )
        )
    return patched
