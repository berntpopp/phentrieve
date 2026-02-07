"""Unit tests for LLM workflow optimizations (caching, auto-routing)."""

from unittest.mock import MagicMock

from phentrieve.llm.prompts.loader import load_prompt_template
from phentrieve.llm.types import AnnotationMode, PostProcessingStep


class TestPromptTemplateCaching:
    """Tests for prompt template caching via @lru_cache."""

    def setup_method(self):
        """Clear cache before each test."""
        load_prompt_template.cache_clear()

    def teardown_method(self):
        """Clear cache after each test."""
        load_prompt_template.cache_clear()

    def test_same_args_returns_cached(self):
        """Verify that the same (mode, language, variant) returns the cached object."""
        t1 = load_prompt_template(PostProcessingStep.VALIDATION, "en")
        t2 = load_prompt_template(PostProcessingStep.VALIDATION, "en")
        assert t1 is t2  # same object identity = cached

    def test_different_args_not_cached(self):
        """Verify different args produce different template objects."""
        t1 = load_prompt_template(PostProcessingStep.VALIDATION, "en")
        t2 = load_prompt_template(PostProcessingStep.REFINEMENT, "en")
        assert t1 is not t2

    def test_cache_info_hits(self):
        """Verify cache_info reports hits."""
        load_prompt_template.cache_clear()
        load_prompt_template(PostProcessingStep.VALIDATION, "en")
        load_prompt_template(PostProcessingStep.VALIDATION, "en")
        info = load_prompt_template.cache_info()
        assert info.hits >= 1
        assert info.misses >= 1

    def test_cache_clear_works(self):
        """Verify cache_clear resets the cache."""
        load_prompt_template(PostProcessingStep.VALIDATION, "en")
        load_prompt_template.cache_clear()
        info = load_prompt_template.cache_info()
        assert info.hits == 0
        assert info.misses == 0


class TestPipelineAutoRouting:
    """Tests for automatic routing to combined post-processor."""

    def test_all_three_steps_routes_to_combined(self):
        """When all 3 individual steps are requested, route to combined."""
        from unittest.mock import patch

        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        with patch("phentrieve.llm.pipeline.LLMProvider"):
            pipeline = LLMAnnotationPipeline(model="test")

        # Create a mock result
        from phentrieve.llm.types import AnnotationResult, HPOAnnotation

        mock_result = AnnotationResult(
            annotations=[
                HPOAnnotation(hpo_id="HP:0001250", term_name="Seizure"),
            ],
            input_text="test",
            language="en",
            mode=AnnotationMode.TOOL_TEXT,
            model="test",
        )

        # Mock the combined processor
        mock_processor = MagicMock()
        mock_processor.process.return_value = (
            mock_result.annotations,
            MagicMock(
                llm_time_seconds=0,
                tool_time_seconds=0,
                timing_events=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                api_calls=0,
            ),
            MagicMock(),
        )

        with patch.object(
            pipeline, "_get_postprocessor", return_value=mock_processor
        ) as mock_get:
            steps = [
                PostProcessingStep.VALIDATION,
                PostProcessingStep.ASSERTION_REVIEW,
                PostProcessingStep.REFINEMENT,
            ]
            pipeline._run_postprocessing(mock_result, steps)

            # Should have been called with COMBINED, not the individual steps
            mock_get.assert_called_once_with(PostProcessingStep.COMBINED)

    def test_subset_of_steps_not_routed(self):
        """When only 1-2 steps are requested, don't route to combined."""
        from unittest.mock import patch

        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        with patch("phentrieve.llm.pipeline.LLMProvider"):
            pipeline = LLMAnnotationPipeline(model="test")

        from phentrieve.llm.types import AnnotationResult, HPOAnnotation

        mock_result = AnnotationResult(
            annotations=[
                HPOAnnotation(hpo_id="HP:0001250", term_name="Seizure"),
            ],
            input_text="test",
            language="en",
            mode=AnnotationMode.TOOL_TEXT,
            model="test",
        )

        mock_processor = MagicMock()
        mock_processor.process.return_value = (
            mock_result.annotations,
            MagicMock(
                llm_time_seconds=0,
                tool_time_seconds=0,
                timing_events=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                api_calls=0,
            ),
            MagicMock(),
        )

        with patch.object(
            pipeline, "_get_postprocessor", return_value=mock_processor
        ) as mock_get:
            steps = [PostProcessingStep.VALIDATION]
            pipeline._run_postprocessing(mock_result, steps)

            mock_get.assert_called_once_with(PostProcessingStep.VALIDATION)
