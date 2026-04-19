from __future__ import annotations

import importlib

from phentrieve.llm import config as llm_config
from phentrieve.llm import types as llm_types
from phentrieve.llm.types import AnnotationMode, AssertionStatus, PostProcessingStep


def test_assertion_status_keeps_present_alias() -> None:
    assert AssertionStatus.PRESENT.value == "present"
    assert AssertionStatus.AFFIRMED is AssertionStatus.PRESENT


def test_post_processing_step_includes_expected_values() -> None:
    assert PostProcessingStep.VALIDATION.value == "validation"
    assert PostProcessingStep.COMBINED.value == "combined"


def test_annotation_mode_does_not_expose_retrieval_only() -> None:
    assert "retrieval_only" not in {mode.value for mode in AnnotationMode}


def test_llm_type_defaults_follow_default_provider_name(monkeypatch) -> None:
    with monkeypatch.context() as context:
        context.setattr(llm_config, "DEFAULT_PROVIDER_NAME", "ollama")
        reloaded = importlib.reload(llm_types)

        assert reloaded.LLMMeta.model_fields["llm_provider"].default == "ollama"
        assert reloaded.LLMPipelineConfig.model_fields["provider"].default == "ollama"

    importlib.reload(llm_types)
