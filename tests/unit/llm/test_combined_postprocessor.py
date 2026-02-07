"""Unit tests for the CombinedPostProcessor."""

import json
from unittest.mock import MagicMock, patch

import pytest

from phentrieve.llm.postprocess.combined import CombinedPostProcessor
from phentrieve.llm.types import (
    AssertionStatus,
    HPOAnnotation,
    PostProcessingStep,
)


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.model = "github/gpt-4o"
    provider.temperature = 0.0
    return provider


@pytest.fixture
def sample_annotations():
    """Create sample annotations for testing."""
    return [
        HPOAnnotation(
            hpo_id="HP:0001250",
            term_name="Seizure",
            assertion=AssertionStatus.AFFIRMED,
            confidence=0.9,
            evidence_text="patient has seizures",
        ),
        HPOAnnotation(
            hpo_id="HP:0000478",
            term_name="Abnormality of the eye",
            assertion=AssertionStatus.AFFIRMED,
            confidence=0.8,
            evidence_text="no eye abnormalities",
        ),
        HPOAnnotation(
            hpo_id="HP:0001263",
            term_name="Global developmental delay",
            assertion=AssertionStatus.AFFIRMED,
            confidence=0.85,
            evidence_text="developmental delay",
        ),
    ]


class TestCombinedPostProcessor:
    """Tests for CombinedPostProcessor."""

    def test_step_is_combined(self, mock_provider):
        processor = CombinedPostProcessor(mock_provider)
        assert processor.step == PostProcessingStep.COMBINED

    def test_empty_annotations(self, mock_provider):
        processor = CombinedPostProcessor(mock_provider)
        result, token_usage, stats = processor.process([], "some text")
        assert result == []
        assert stats.annotations_in == 0
        assert stats.annotations_out == 0

    @patch("phentrieve.llm.postprocess.combined.load_prompt_template")
    def test_parse_validated_annotations(
        self, mock_load, mock_provider, sample_annotations
    ):
        """Test parsing validated annotations from combined response."""
        response_json = json.dumps(
            {
                "validated_annotations": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Seizure",
                        "assertion": "affirmed",
                        "confidence": 0.95,
                        "evidence_text": "patient has seizures",
                    }
                ],
                "removed_annotations": [
                    {
                        "hpo_id": "HP:0000478",
                        "term_name": "Abnormality of the eye",
                        "reason": "Text says 'no eye abnormalities' - false positive",
                    }
                ],
                "refined_annotations": [
                    {
                        "original_hpo_id": "HP:0001263",
                        "original_term_name": "Global developmental delay",
                        "refined_hpo_id": "HP:0001263",
                        "refined_term_name": "Global developmental delay",
                        "assertion": "affirmed",
                        "confidence": 0.9,
                        "evidence_text": "developmental delay",
                        "refinement_reason": "kept as is",
                    }
                ],
            }
        )

        mock_response = MagicMock()
        mock_response.content = f"```json\n{response_json}\n```"
        mock_response.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_provider.complete.return_value = mock_response

        mock_template = MagicMock()
        mock_template.get_messages.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        mock_load.return_value = mock_template

        processor = CombinedPostProcessor(mock_provider)
        result, token_usage, stats = processor.process(
            sample_annotations,
            "patient has seizures and developmental delay, no eye abnormalities",
        )

        assert stats.step == "combined"
        assert stats.annotations_in == 3
        assert stats.removed == 1
        assert stats.terms_refined == 1
        # 1 validated + 1 refined = 2 annotations out
        assert stats.annotations_out == 2
        assert len(result) == 2

    @patch("phentrieve.llm.postprocess.combined.load_prompt_template")
    def test_assertion_change_tracking(
        self, mock_load, mock_provider, sample_annotations
    ):
        """Test that assertion changes are tracked."""
        response_json = json.dumps(
            {
                "validated_annotations": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Seizure",
                        "assertion": "affirmed",
                        "confidence": 0.95,
                    },
                    {
                        "hpo_id": "HP:0000478",
                        "term_name": "Abnormality of the eye",
                        "assertion": "negated",  # Changed from affirmed
                        "confidence": 0.9,
                    },
                    {
                        "hpo_id": "HP:0001263",
                        "term_name": "Global developmental delay",
                        "assertion": "affirmed",
                        "confidence": 0.85,
                    },
                ],
                "removed_annotations": [],
                "refined_annotations": [],
            }
        )

        mock_response = MagicMock()
        mock_response.content = response_json
        mock_response.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_provider.complete.return_value = mock_response

        mock_template = MagicMock()
        mock_template.get_messages.return_value = [{"role": "user", "content": "test"}]
        mock_load.return_value = mock_template

        processor = CombinedPostProcessor(mock_provider)
        result, _, stats = processor.process(sample_annotations, "text")

        assert stats.assertions_changed == 1
        assert len(result) == 3

    @patch("phentrieve.llm.postprocess.combined.load_prompt_template")
    def test_unparseable_response_returns_originals(
        self, mock_load, mock_provider, sample_annotations
    ):
        """Test that unparseable response returns original annotations."""
        mock_response = MagicMock()
        mock_response.content = "not valid json"
        mock_response.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_provider.complete.return_value = mock_response

        mock_template = MagicMock()
        mock_template.get_messages.return_value = [{"role": "user", "content": "test"}]
        mock_load.return_value = mock_template

        processor = CombinedPostProcessor(mock_provider)
        result, _, stats = processor.process(sample_annotations, "text")

        assert result == sample_annotations
        assert stats.removed == 0

    def test_prompt_not_found_returns_originals(
        self, mock_provider, sample_annotations
    ):
        """Test graceful handling when prompt template is missing."""
        with patch(
            "phentrieve.llm.postprocess.combined.load_prompt_template",
            side_effect=FileNotFoundError("not found"),
        ):
            processor = CombinedPostProcessor(mock_provider)
            result, _, stats = processor.process(sample_annotations, "text")

            assert result == sample_annotations
            assert stats.annotations_in == 3
            assert stats.annotations_out == 3


class TestCombinedPostProcessorSubstats:
    """Tests for process_returning_substats."""

    @patch("phentrieve.llm.postprocess.combined.load_prompt_template")
    def test_substats_returned(self, mock_load, mock_provider, sample_annotations):
        """Test that substats are returned for each sub-step."""
        response_json = json.dumps(
            {
                "validated_annotations": [
                    {
                        "hpo_id": "HP:0001250",
                        "term_name": "Seizure",
                        "assertion": "negated",  # assertion changed
                        "confidence": 0.95,
                    }
                ],
                "removed_annotations": [
                    {"hpo_id": "HP:0000478", "term_name": "Eye", "reason": "FP"}
                ],
                "refined_annotations": [
                    {
                        "original_hpo_id": "HP:0001263",
                        "refined_hpo_id": "HP:0012758",
                        "refined_term_name": "Neurodevelopmental delay",
                        "assertion": "affirmed",
                        "confidence": 0.9,
                    }
                ],
            }
        )

        mock_response = MagicMock()
        mock_response.content = response_json
        mock_response.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_provider.complete.return_value = mock_response

        mock_template = MagicMock()
        mock_template.get_messages.return_value = [{"role": "user", "content": "test"}]
        mock_load.return_value = mock_template

        processor = CombinedPostProcessor(mock_provider)
        result, _, substats = processor.process_returning_substats(
            sample_annotations, "text"
        )

        assert len(substats) == 3
        assert substats[0].step == "validation"
        assert substats[0].removed == 1
        assert substats[1].step == "assertion_review"
        assert substats[1].assertions_changed == 1
        assert substats[2].step == "refinement"
        assert substats[2].terms_refined == 1
