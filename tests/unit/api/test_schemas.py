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
from api.schemas.similarity_schemas import (
    SimilarityPairResult,
    SimilarityRequest,
    SimilarityResponse,
)
from api.schemas.text_processing_schemas import (
    ExtractedHPOTerm,
    TextProcessingRequest,
    TextProcessingResponse,
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
        assert req.num_results == 10  # Default
        assert req.similarity_threshold == 0.3  # Default
        assert req.enable_reranker is False  # Default
        assert req.detect_query_assertion is True  # Default

    def test_full_request_with_all_fields(self):
        """Test request with all optional fields specified."""
        # Arrange & Act
        req = QueryRequest(
            text="Patient zeigt Krampfanfälle",
            model_name="FremyCompany/BioLORD-2023-M",
            language="de",
            num_results=20,
            similarity_threshold=0.5,
            enable_reranker=True,
            reranker_model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            reranker_mode="cross-lingual",
            rerank_count=15,
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
        assert req.enable_reranker is True
        assert req.reranker_mode == "cross-lingual"
        assert req.rerank_count == 15

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

    def test_reranker_mode_literal_validation(self):
        """Test reranker_mode only accepts valid literals."""
        # Valid modes
        req1 = QueryRequest(text="test", reranker_mode="cross-lingual")
        assert req1.reranker_mode == "cross-lingual"

        req2 = QueryRequest(text="test", reranker_mode="monolingual")
        assert req2.reranker_mode == "monolingual"

        # Invalid mode
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(text="test", reranker_mode="invalid-mode")

        assert "Input should be 'cross-lingual' or 'monolingual'" in str(exc_info.value)

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

    def test_rerank_count_validation(self):
        """Test rerank_count must be 1-100."""
        # Valid boundary
        req_min = QueryRequest(text="test", rerank_count=1)
        assert req_min.rerank_count == 1

        req_max = QueryRequest(text="test", rerank_count=100)
        assert req_max.rerank_count == 100

        # Invalid (0)
        with pytest.raises(ValidationError):
            QueryRequest(text="test", rerank_count=0)

        # Invalid (101)
        with pytest.raises(ValidationError):
            QueryRequest(text="test", rerank_count=101)


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
        assert item.cross_encoder_score is None  # Optional
        assert item.original_rank is None  # Optional

    def test_full_hpo_result_with_reranking(self):
        """Test HPO result with all fields (after re-ranking)."""
        # Arrange & Act
        item = HPOResultItem(
            hpo_id="HP:0001250",
            label="Seizure",
            similarity=0.85,
            cross_encoder_score=0.92,
            original_rank=3,
        )

        # Assert
        assert item.hpo_id == "HP:0001250"
        assert item.label == "Seizure"
        assert item.similarity == 0.85
        assert item.cross_encoder_score == 0.92
        assert item.original_rank == 3


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
        assert resp.reranker_used is None
        assert resp.query_assertion_status is None

    def test_full_query_response_with_reranking(self):
        """Test response with language detection, reranking, and assertion."""
        # Arrange & Act
        resp = QueryResponse(
            query_text_received="patient has seizures",
            language_detected="en",
            model_used_for_retrieval="FremyCompany/BioLORD-2023-M",
            reranker_used="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            query_assertion_status="negated",
            results=[
                HPOResultItem(
                    hpo_id="HP:0001250",
                    label="Seizure",
                    similarity=0.85,
                    cross_encoder_score=0.92,
                    original_rank=2,
                )
            ],
        )

        # Assert
        assert resp.language_detected == "en"
        assert resp.reranker_used == "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        assert resp.query_assertion_status == "negated"
        assert resp.results[0].cross_encoder_score == 0.92

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


# =============================================================================
# Similarity Schemas (api/schemas/similarity_schemas.py)
# =============================================================================


class TestSimilarityRequest:
    """Test SimilarityRequest schema validation."""

    def test_minimal_similarity_request(self):
        """Test minimal request with defaults."""
        # Arrange & Act
        req = SimilarityRequest(hpo_id="HP:0001250")

        # Assert
        assert req.hpo_id == "HP:0001250"
        assert req.model_name is None  # Optional
        assert req.top_k == 10  # Default
        assert req.include_self is False  # Default

    def test_full_similarity_request(self):
        """Test request with all fields."""
        # Arrange & Act
        req = SimilarityRequest(
            hpo_id="HP:0001250",
            model_name="FremyCompany/BioLORD-2023-M",
            top_k=20,
            include_self=True,
        )

        # Assert
        assert req.hpo_id == "HP:0001250"
        assert req.model_name == "FremyCompany/BioLORD-2023-M"
        assert req.top_k == 20
        assert req.include_self is True

    def test_top_k_validation(self):
        """Test top_k must be 1-100."""
        # Valid boundaries
        req_min = SimilarityRequest(hpo_id="HP:0001250", top_k=1)
        assert req_min.top_k == 1

        req_max = SimilarityRequest(hpo_id="HP:0001250", top_k=100)
        assert req_max.top_k == 100

        # Invalid
        with pytest.raises(ValidationError):
            SimilarityRequest(hpo_id="HP:0001250", top_k=0)

        with pytest.raises(ValidationError):
            SimilarityRequest(hpo_id="HP:0001250", top_k=101)


class TestSimilarityPairResult:
    """Test SimilarityPairResult schema."""

    def test_similarity_pair_result(self):
        """Test similarity pair structure."""
        # Arrange & Act
        result = SimilarityPairResult(
            hpo_id="HP:0002066",
            label="Gait ataxia",
            similarity=0.78,
        )

        # Assert
        assert result.hpo_id == "HP:0002066"
        assert result.label == "Gait ataxia"
        assert result.similarity == 0.78


class TestSimilarityResponse:
    """Test SimilarityResponse schema."""

    def test_similarity_response(self):
        """Test similarity response structure."""
        # Arrange & Act
        resp = SimilarityResponse(
            query_hpo_id="HP:0001250",
            query_label="Seizure",
            model_used="FremyCompany/BioLORD-2023-M",
            similar_terms=[
                SimilarityPairResult(hpo_id="HP:0002066", label="Gait ataxia", similarity=0.78)
            ],
        )

        # Assert
        assert resp.query_hpo_id == "HP:0001250"
        assert resp.query_label == "Seizure"
        assert resp.model_used == "FremyCompany/BioLORD-2023-M"
        assert len(resp.similar_terms) == 1
        assert resp.similar_terms[0].hpo_id == "HP:0002066"

    def test_empty_similar_terms_allowed(self):
        """Test response with no similar terms (edge case)."""
        # Arrange & Act
        resp = SimilarityResponse(
            query_hpo_id="HP:0001250",
            query_label="Seizure",
            model_used="test-model",
            similar_terms=[],
        )

        # Assert
        assert resp.similar_terms == []


# =============================================================================
# Text Processing Schemas (api/schemas/text_processing_schemas.py)
# =============================================================================


class TestTextProcessingRequest:
    """Test TextProcessingRequest schema validation."""

    def test_minimal_text_processing_request(self):
        """Test minimal request with defaults."""
        # Arrange & Act
        req = TextProcessingRequest(text="patient has seizures")

        # Assert
        assert req.text == "patient has seizures"
        assert req.language is None  # Optional
        assert req.sentence_mode is False  # Default

    def test_full_text_processing_request(self):
        """Test request with all fields."""
        # Arrange & Act
        req = TextProcessingRequest(
            text="Patient has seizures. No heart disease.",
            language="en",
            sentence_mode=True,
        )

        # Assert
        assert req.text == "Patient has seizures. No heart disease."
        assert req.language == "en"
        assert req.sentence_mode is True

    def test_text_min_length_validation(self):
        """Test text must be non-empty."""
        # Act & Assert
        with pytest.raises(ValidationError):
            TextProcessingRequest(text="")


class TestExtractedHPOTerm:
    """Test ExtractedHPOTerm schema."""

    def test_extracted_hpo_term(self):
        """Test extracted HPO term structure."""
        # Arrange & Act
        term = ExtractedHPOTerm(
            segment_text="patient has seizures",
            hpo_id="HP:0001250",
            hpo_label="Seizure",
        )

        # Assert
        assert term.segment_text == "patient has seizures"
        assert term.hpo_id == "HP:0001250"
        assert term.hpo_label == "Seizure"


class TestTextProcessingResponse:
    """Test TextProcessingResponse schema."""

    def test_text_processing_response(self):
        """Test text processing response structure."""
        # Arrange & Act
        resp = TextProcessingResponse(
            original_text="patient has seizures",
            language_detected="en",
            processed_segments=[
                ExtractedHPOTerm(
                    segment_text="patient has seizures",
                    hpo_id="HP:0001250",
                    hpo_label="Seizure",
                )
            ],
        )

        # Assert
        assert resp.original_text == "patient has seizures"
        assert resp.language_detected == "en"
        assert len(resp.processed_segments) == 1
        assert resp.processed_segments[0].hpo_id == "HP:0001250"

    def test_empty_processed_segments_allowed(self):
        """Test response with no extracted terms (valid)."""
        # Arrange & Act
        resp = TextProcessingResponse(
            original_text="no medical terms here",
            language_detected="en",
            processed_segments=[],
        )

        # Assert
        assert resp.processed_segments == []
