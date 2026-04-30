"""Real unit tests for API Pydantic schemas (validation logic).

Tests cover:
- QueryRequest, QueryResponse (query_schemas.py)
- SimilarityRequest, SimilarityResponse (similarity_schemas.py)
- TextProcessingRequest, TextProcessingResponse (text_processing_schemas.py)
- ConfigInfoResponse (config_info_schemas.py)
"""

import pytest
from pydantic import ValidationError

from api.schemas.query_schemas import HPOResultItem, QueryRequest, QueryResponse

# NOTE: Obsolete similarity schemas removed - see similarity_router.py for current implementation
# from api.schemas.similarity_schemas import (
#     SimilarityPairResult,  # Does not exist
#     SimilarityRequest,  # Does not exist
#     SimilarityResponse,  # Does not exist
# )
from api.schemas.text_processing_schemas import (
    TextProcessingRequest,
)
from phentrieve.config import (
    DEFAULT_ASSERTION_PREFERENCE,
    DEFAULT_NUM_RESULTS,
    MIN_SIMILARITY_THRESHOLD,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Query Schemas (api/schemas/query_schemas.py)
# =============================================================================


class TestQueryRequest:
    """Test QueryRequest schema validation."""

    def test_minimal_valid_request(self):
        """Test minimal valid query request with defaults."""
        # Arrange & Act
        req = QueryRequest(text="patient has seizures")

        # Assert
        assert req.text == "patient has seizures"
        assert req.model_name is None  # Optional
        assert req.language is None  # Optional
        assert req.num_results == DEFAULT_NUM_RESULTS
        assert req.similarity_threshold == MIN_SIMILARITY_THRESHOLD
        assert req.detect_query_assertion is True  # Default
        assert req.query_assertion_preference == DEFAULT_ASSERTION_PREFERENCE

    def test_full_request_with_all_fields(self):
        """Test request with all optional fields specified."""
        # Arrange & Act
        req = QueryRequest(
            text="Patient zeigt Krampfanfälle",
            model_name="FremyCompany/BioLORD-2023-M",
            language="de",
            num_results=20,
            similarity_threshold=0.5,
            detect_query_assertion=False,
            query_assertion_language="de",
            query_assertion_preference="keyword",
        )

        # Assert
        assert req.text == "Patient zeigt Krampfanfälle"
        assert req.model_name == "FremyCompany/BioLORD-2023-M"
        assert req.language == "de"
        assert req.num_results == 20
        assert req.similarity_threshold == 0.5
        assert req.detect_query_assertion is False
        assert req.query_assertion_language == "de"
        assert req.query_assertion_preference == "keyword"

    def test_text_min_length_validation(self):
        """Test text must be non-empty (min_length=1)."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(text="")

        assert "String should have at least 1 character" in str(exc_info.value)

    def test_num_results_must_be_positive(self):
        """Test num_results must be > 0."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(text="test", num_results=0)

        assert "greater than 0" in str(exc_info.value)

    def test_num_results_max_limit(self):
        """Test num_results must be <= 50."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(text="test", num_results=51)

        assert "less than or equal to 50" in str(exc_info.value)

    def test_similarity_threshold_range(self):
        """Test similarity_threshold must be 0.0-1.0."""
        # Test minimum boundary
        req_min = QueryRequest(text="test", similarity_threshold=0.0)
        assert req_min.similarity_threshold == 0.0

        # Test maximum boundary
        req_max = QueryRequest(text="test", similarity_threshold=1.0)
        assert req_max.similarity_threshold == 1.0

        # Test below minimum
        with pytest.raises(ValidationError):
            QueryRequest(text="test", similarity_threshold=-0.1)

        # Test above maximum
        with pytest.raises(ValidationError):
            QueryRequest(text="test", similarity_threshold=1.1)

    def test_query_assertion_preference_literals(self):
        """Test query_assertion_preference only accepts valid strategies."""
        # Valid preferences
        req1 = QueryRequest(text="test", query_assertion_preference="dependency")
        assert req1.query_assertion_preference == "dependency"

        req2 = QueryRequest(text="test", query_assertion_preference="keyword")
        assert req2.query_assertion_preference == "keyword"

        req3 = QueryRequest(text="test", query_assertion_preference="any_negative")
        assert req3.query_assertion_preference == "any_negative"

        # Invalid preference
        with pytest.raises(ValidationError):
            QueryRequest(text="test", query_assertion_preference="invalid")


class TestHPOResultItem:
    """Test HPOResultItem schema."""

    def test_minimal_hpo_result(self):
        """Test HPO result with required fields only."""
        # Arrange & Act
        item = HPOResultItem(hpo_id="HP:0001250", label="Seizure")

        # Assert
        assert item.hpo_id == "HP:0001250"
        assert item.label == "Seizure"
        assert item.similarity is None  # Optional

    def test_full_hpo_result(self):
        """Test HPO result with all fields populated."""
        # Arrange & Act
        item = HPOResultItem(
            hpo_id="HP:0001250",
            label="Seizure",
            similarity=0.85,
        )

        # Assert
        assert item.hpo_id == "HP:0001250"
        assert item.label == "Seizure"
        assert item.similarity == 0.85


class TestQueryResponse:
    """Test QueryResponse schema."""

    def test_minimal_query_response(self):
        """Test minimal response structure."""
        # Arrange & Act
        resp = QueryResponse(
            query_text_received="patient has seizures",
            model_used_for_retrieval="FremyCompany/BioLORD-2023-M",
            results=[HPOResultItem(hpo_id="HP:0001250", label="Seizure")],
        )

        # Assert
        assert resp.query_text_received == "patient has seizures"
        assert resp.model_used_for_retrieval == "FremyCompany/BioLORD-2023-M"
        assert len(resp.results) == 1
        assert resp.language_detected is None
        assert resp.query_assertion_status is None

    def test_full_query_response(self):
        """Test response with language detection and assertion."""
        # Arrange & Act
        resp = QueryResponse(
            query_text_received="patient has seizures",
            language_detected="en",
            model_used_for_retrieval="FremyCompany/BioLORD-2023-M",
            query_assertion_status="negated",
            results=[
                HPOResultItem(
                    hpo_id="HP:0001250",
                    label="Seizure",
                    similarity=0.85,
                )
            ],
        )

        # Assert
        assert resp.language_detected == "en"
        assert resp.query_assertion_status == "negated"
        assert resp.results[0].similarity == 0.85

    def test_empty_results_allowed(self):
        """Test response with no results (valid case)."""
        # Arrange & Act
        resp = QueryResponse(
            query_text_received="nonsense text xyz",
            model_used_for_retrieval="test-model",
            results=[],
        )

        # Assert
        assert resp.results == []


class TestTextProcessingRequestDefaults:
    """Test TextProcessingRequest schema defaults that drive router behavior."""

    def test_semantic_model_name_defaults_to_none(self):
        """Omitted semantic model should stay unset so the router can follow retrieval."""
        req = TextProcessingRequest(text_content="patient has seizures")

        assert req.semantic_model_name is None
        assert req.retrieval_model_name is not None


# =============================================================================
# Similarity Schemas (api/schemas/similarity_schemas.py)
# =============================================================================
# NOTE: Obsolete similarity schema tests removed.
# The schemas referenced (SimilarityRequest, SimilarityResponse, SimilarityPairResult)
# no longer exist in the codebase.
# Current similarity API uses HPOTermSimilarityResponseAPI (see similarity_router.py)
# Tests for the actual similarity router are in test_similarity_router.py


# =============================================================================
# Text Processing Schemas (api/schemas/text_processing_schemas.py)
# =============================================================================


class TestTextProcessingRequest:
    """Test TextProcessingRequest schema validation."""

    def test_minimal_text_processing_request(self):
        """Test minimal request with defaults."""
        # Arrange & Act
        req = TextProcessingRequest(text_content="patient has seizures")

        # Assert
        assert req.text_content == "patient has seizures"
        assert req.language == "en"  # Defaults to DEFAULT_LANGUAGE
        assert req.chunking_strategy == "sliding_window_punct_conj_cleaned"  # Default

    def test_full_text_processing_request(self):
        """Test request with all fields."""
        # Arrange & Act
        req = TextProcessingRequest(
            text_content="Patient has seizures. No heart disease.",
            language="de",
            chunking_strategy="semantic",
        )

        # Assert
        assert req.text_content == "Patient has seizures. No heart disease."
        assert req.language == "de"
        assert req.chunking_strategy == "semantic"

    def test_text_processing_request_accepts_llm_backend(self):
        request = TextProcessingRequest.model_validate(
            {
                "text": "Patient had recurrent seizures.",
                "extraction_backend": "llm",
                "llm_mode": "two_phase",
            }
        )

        assert request.text == "Patient had recurrent seizures."
        assert request.text_content == "Patient had recurrent seizures."
        assert request.extraction_backend == "llm"
        assert request.llm_mode == "two_phase"

    @pytest.mark.parametrize("field", ["llm_model", "llm_provider", "llm_base_url"])
    def test_text_processing_request_forbids_public_llm_config_fields(
        self, field: str
    ) -> None:
        with pytest.raises(ValueError):
            TextProcessingRequest.model_validate(
                {
                    "text": "Patient has seizures.",
                    "extraction_backend": "llm",
                    field: "untrusted",
                }
            )

    def test_text_processing_request_accepts_text_content_extraction_backend_alias(
        self,
    ):
        request = TextProcessingRequest.model_validate(
            {
                "text_content": "Patient had recurrent seizures.",
            }
        )

        assert request.text == "Patient had recurrent seizures."
        assert request.text_content == "Patient had recurrent seizures."

    def test_window_size_validation(self):
        """Test window_size must be >= 1."""
        # Valid
        req = TextProcessingRequest(text_content="test", window_size=1)
        assert req.window_size == 1

        # Invalid (0)
        with pytest.raises(ValidationError):
            TextProcessingRequest(text_content="test", window_size=0)


# NOTE: Text processing schema tests commented out - schemas have changed significantly.
# Current schemas: ProcessedChunkAPI, AggregatedHPOTermAPI, TextProcessingResponseAPI
# See test_text_processing_router.py for actual router tests.

# class TestExtractedHPOTerm:  # OBSOLETE - schema doesn't exist
# class TestTextProcessingResponse:  # OBSOLETE - schema structure changed
