"""
API router for LLM-based HPO annotation.

Provides endpoints for annotating clinical text using LLMs via LiteLLM,
with support for multiple annotation modes and post-processing steps.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas.llm_annotation_schemas import (
    AnnotationModeEnum,
    AssertionStatusEnum,
    AvailableModelsResponse,
    HPOAnnotationItem,
    LLMAnnotationRequest,
    LLMAnnotationResponse,
    PostProcessingStepEnum,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _check_litellm_available() -> None:
    """Check if LiteLLM is installed."""
    try:
        import litellm  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="LLM annotation service unavailable. LiteLLM is not installed.",
        )


@router.post(
    "/annotate",
    response_model=LLMAnnotationResponse,
    summary="Annotate clinical text with HPO terms using LLM",
    description="""
    Extract HPO annotations from clinical text using an LLM.

    **Annotation Modes:**
    - `direct`: LLM outputs HPO IDs directly from its training knowledge
    - `tool_term`: LLM extracts phrases, queries Phentrieve, selects best matches
    - `tool_text`: Phentrieve processes text, LLM validates and selects candidates

    **Post-Processing Steps:**
    - `validation`: Re-check annotations against original text
    - `refinement`: Upgrade to more specific HPO terms if supported
    - `assertion_review`: Validate negation detection accuracy

    **Authentication:**
    Set the appropriate environment variable for your chosen provider:
    - GitHub Models: `GITHUB_TOKEN`
    - Google Gemini: `GEMINI_API_KEY`
    - Anthropic: `ANTHROPIC_API_KEY`
    - OpenAI: `OPENAI_API_KEY`
    """,
)
async def annotate_text(request: LLMAnnotationRequest) -> LLMAnnotationResponse:
    """
    Annotate clinical text with HPO terms using an LLM.

    Args:
        request: The annotation request containing text and options.

    Returns:
        LLMAnnotationResponse with extracted HPO annotations.
    """
    _check_litellm_available()

    try:
        from phentrieve.llm import (
            AnnotationMode,
            LLMAnnotationPipeline,
            PostProcessingStep,
        )

        # Convert mode enum
        mode_map = {
            AnnotationModeEnum.DIRECT: AnnotationMode.DIRECT,
            AnnotationModeEnum.TOOL_TERM: AnnotationMode.TOOL_TERM,
            AnnotationModeEnum.TOOL_TEXT: AnnotationMode.TOOL_TEXT,
        }
        annotation_mode = mode_map[request.mode]

        # Convert post-processing steps
        postprocess_steps = None
        if request.postprocess:
            step_map = {
                PostProcessingStepEnum.VALIDATION: PostProcessingStep.VALIDATION,
                PostProcessingStepEnum.REFINEMENT: PostProcessingStep.REFINEMENT,
                PostProcessingStepEnum.ASSERTION_REVIEW: PostProcessingStep.ASSERTION_REVIEW,
            }
            postprocess_steps = [step_map[s] for s in request.postprocess]

        # Create pipeline and run
        pipeline = LLMAnnotationPipeline(
            model=request.model,
            temperature=request.temperature,
            validate_hpo_ids=request.validate_hpo_ids,
        )

        result = pipeline.run(
            text=request.text_content,
            mode=annotation_mode,
            language=request.language,
            postprocess=postprocess_steps,
        )

        # Enrich with details if requested
        if request.include_details:
            result = _enrich_annotations_with_details(result)

        # Convert annotations to response format
        annotations = [
            HPOAnnotationItem(
                hpo_id=ann.hpo_id,
                term_name=ann.term_name,
                assertion=AssertionStatusEnum(ann.assertion.value),
                confidence=ann.confidence,
                evidence_text=ann.evidence_text,
                definition=ann.definition,
                synonyms=ann.synonyms if ann.synonyms else None,
            )
            for ann in result.annotations
        ]

        # Generate phenopacket if requested
        phenopacket = None
        if request.output_format == "phenopacket":
            phenopacket = _generate_phenopacket(result)

        # Convert post-processing steps for response
        response_steps = [
            PostProcessingStepEnum(s.value) for s in result.post_processing_steps
        ]

        return LLMAnnotationResponse(
            annotations=annotations,
            input_text=result.input_text,
            language=result.language,
            model=result.model,
            mode=AnnotationModeEnum(result.mode.value),
            prompt_version=result.prompt_version,
            post_processing_steps=response_steps,
            processing_time_seconds=result.processing_time_seconds,
            phenopacket=phenopacket,
        )

    except ImportError as e:
        logger.error("LLM module import error: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"LLM annotation service unavailable: {e}",
        )
    except Exception as e:
        logger.exception("LLM annotation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Annotation failed: {str(e)}",
        )


@router.get(
    "/models",
    response_model=AvailableModelsResponse,
    summary="List available LLM models",
    description="Returns available model presets organized by provider.",
)
async def list_models() -> AvailableModelsResponse:
    """
    List available LLM models by provider.

    Returns:
        Dict of available models organized by provider name.
    """
    try:
        from phentrieve.llm import get_available_models

        models = get_available_models()
        return AvailableModelsResponse(models=models)

    except ImportError:
        # Return basic list if LLM module not fully available
        return AvailableModelsResponse(
            models={
                "github": ["github/gpt-4o", "github/gpt-4o-mini"],
                "gemini": ["gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash"],
                "anthropic": ["anthropic/claude-sonnet-4-20250514"],
                "openai": ["gpt-4o", "gpt-4o-mini"],
                "ollama": ["ollama/llama3.1"],
            }
        )


def _enrich_annotations_with_details(result: Any) -> Any:
    """Add HPO definitions and synonyms to annotations."""
    try:
        from phentrieve.config import get_config_value
        from phentrieve.data_processing.hpo_database import HPODatabase

        db_path: str | None = get_config_value("data", None, "hpo_database_path")
        if not db_path:
            return result

        db = HPODatabase(db_path)
        hpo_ids = [a.hpo_id for a in result.annotations]
        terms_by_id: dict[str, dict[str, Any]] = db.get_terms_by_ids(hpo_ids)

        for ann in result.annotations:
            if ann.hpo_id in terms_by_id:
                term_data = terms_by_id[ann.hpo_id]
                ann.definition = term_data.get("definition")
                ann.synonyms = term_data.get("synonyms", [])

    except Exception as e:
        logger.warning("Failed to enrich annotations: %s", e)

    return result


def _generate_phenopacket(result: Any) -> dict[str, Any] | None:
    """Generate phenopacket representation from result."""
    import json as json_module

    try:
        from phentrieve.phenopackets.utils import format_as_phenopacket_v2

        aggregated_results = []
        for ann in result.annotations:
            aggregated_results.append(
                {
                    "hpo_id": ann.hpo_id,
                    "term_name": ann.term_name,
                    "assertion": ann.assertion.value,
                    "score": ann.confidence,
                    "evidence_text": ann.evidence_text,
                }
            )

        # format_as_phenopacket_v2 returns a JSON string, parse it to dict
        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=aggregated_results,
            embedding_model=result.model,
            input_text=result.input_text,
        )
        parsed: dict[str, Any] = json_module.loads(phenopacket_json)
        return parsed

    except Exception as e:
        logger.warning("Failed to generate phenopacket: %s", e)
        return None
