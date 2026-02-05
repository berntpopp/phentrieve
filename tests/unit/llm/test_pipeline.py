"""Unit tests for LLM annotation pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from phentrieve.llm.types import (
    AnnotationMode,
    PostProcessingStep,
)


class TestLLMAnnotationPipeline:
    """Tests for LLMAnnotationPipeline."""

    @pytest.fixture
    def mock_litellm(self):
        """Create a mock LiteLLM module."""
        mock = MagicMock()
        mock.suppress_debug_info = False
        return mock

    @pytest.fixture
    def mock_provider_class(self, mock_litellm):
        """Create a mock LLMProvider class."""
        mock_provider = MagicMock()
        mock_provider.model = "github/gpt-4o"
        mock_provider.temperature = 0.0
        mock_provider.supports_tools.return_value = True

        with patch("phentrieve.llm.pipeline.LLMProvider", return_value=mock_provider):
            yield mock_provider

    def test_pipeline_creation(self, mock_provider_class):
        """Test creating a pipeline."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(
            model="github/gpt-4o",
            temperature=0.0,
        )
        assert pipeline.validate_hpo_ids is True

    def test_pipeline_custom_settings(self, mock_provider_class):
        """Test pipeline with custom settings."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(
            model="gemini/gemini-1.5-flash",
            temperature=0.5,
            max_tokens=2048,
            validate_hpo_ids=False,
        )
        assert pipeline.validate_hpo_ids is False

    def test_get_strategy_direct(self, mock_provider_class):
        """Test getting direct text strategy."""
        from phentrieve.llm.annotation.direct_text import DirectTextStrategy
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(model="github/gpt-4o")
        strategy = pipeline._get_strategy(AnnotationMode.DIRECT)

        assert isinstance(strategy, DirectTextStrategy)
        assert strategy.mode == AnnotationMode.DIRECT

    def test_get_strategy_tool_term(self, mock_provider_class):
        """Test getting tool-guided term search strategy."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(model="github/gpt-4o")
        strategy = pipeline._get_strategy(AnnotationMode.TOOL_TERM)

        assert isinstance(strategy, ToolGuidedStrategy)
        assert strategy.mode == AnnotationMode.TOOL_TERM

    def test_get_strategy_tool_text(self, mock_provider_class):
        """Test getting tool-guided text process strategy."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(model="github/gpt-4o")
        strategy = pipeline._get_strategy(AnnotationMode.TOOL_TEXT)

        assert isinstance(strategy, ToolGuidedStrategy)
        assert strategy.mode == AnnotationMode.TOOL_TEXT

    def test_strategy_caching(self, mock_provider_class):
        """Test that strategies are cached."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(model="github/gpt-4o")

        strategy1 = pipeline._get_strategy(AnnotationMode.DIRECT)
        strategy2 = pipeline._get_strategy(AnnotationMode.DIRECT)

        assert strategy1 is strategy2  # Same instance

    def test_get_postprocessor_validation(self, mock_provider_class):
        """Test getting validation postprocessor."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline
        from phentrieve.llm.postprocess.validation import ValidationPostProcessor

        pipeline = LLMAnnotationPipeline(model="github/gpt-4o")
        processor = pipeline._get_postprocessor(PostProcessingStep.VALIDATION)

        assert isinstance(processor, ValidationPostProcessor)

    def test_get_postprocessor_refinement(self, mock_provider_class):
        """Test getting refinement postprocessor."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline
        from phentrieve.llm.postprocess.refinement import RefinementPostProcessor

        pipeline = LLMAnnotationPipeline(model="github/gpt-4o")
        processor = pipeline._get_postprocessor(PostProcessingStep.REFINEMENT)

        assert isinstance(processor, RefinementPostProcessor)

    def test_get_postprocessor_assertion_review(self, mock_provider_class):
        """Test getting assertion review postprocessor."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline
        from phentrieve.llm.postprocess.assertion_review import (
            AssertionReviewPostProcessor,
        )

        pipeline = LLMAnnotationPipeline(model="github/gpt-4o")
        processor = pipeline._get_postprocessor(PostProcessingStep.ASSERTION_REVIEW)

        assert isinstance(processor, AssertionReviewPostProcessor)


class TestCreatePipelineHelper:
    """Tests for create_pipeline convenience function."""

    def test_create_pipeline_default(self):
        """Test creating pipeline with defaults."""
        with patch("phentrieve.llm.pipeline.LLMProvider"):
            from phentrieve.llm.pipeline import create_pipeline

            pipeline = create_pipeline()
            assert pipeline is not None

    def test_create_pipeline_custom(self):
        """Test creating pipeline with custom settings."""
        with patch("phentrieve.llm.pipeline.LLMProvider"):
            from phentrieve.llm.pipeline import create_pipeline

            pipeline = create_pipeline(
                model="gemini/gemini-1.5-pro",
                temperature=0.3,
            )
            assert pipeline is not None
