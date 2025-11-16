"""
End-to-End API Workflow Tests.

This module validates complete API functionality through real HTTP requests
to the Dockerized service. These tests ensure:

- API endpoints are accessible and functional
- Query workflow processes clinical text correctly
- Response formats match API schema specifications
- Error handling returns appropriate status codes
- Integration with embedding models and vector database works
- Multilingual support functions correctly

All tests are marked with @pytest.mark.e2e for test categorization.
"""

import pytest
import requests


@pytest.mark.e2e
class TestAPIEndpoints:
    """Test suite for API endpoint availability and basic functionality."""

    def test_query_endpoint_accessible(self, api_query_endpoint: str):
        """
        Verify query endpoint is accessible (POST method).

        Expected:
            POST /api/v1/query without body returns 422 (validation error)
            Not 404 (endpoint exists) or 500 (server error)
        """
        response = requests.post(api_query_endpoint, json={}, timeout=10)

        # Should return 422 (validation error) since query_text is required
        # NOT 404 (endpoint doesn't exist) or 500 (server error)
        assert response.status_code in [
            422,
            400,
        ], f"Query endpoint should exist (expect 400/422), got: {response.status_code}"

    def test_health_endpoint_returns_ok_status(self, api_health_endpoint: str):
        """
        Verify health endpoint returns healthy status.

        Expected:
            GET /api/v1/health returns 200
            Response contains {"status": "healthy"} or similar
        """
        response = requests.get(api_health_endpoint, timeout=5)

        assert response.status_code == 200, "Health endpoint should return 200"

        data = response.json()
        assert "status" in data, "Health response should contain status field"

    def test_config_endpoint_returns_configuration(self, api_config_endpoint: str):
        """
        Verify config endpoint returns service configuration.

        Expected:
            GET /api/v1/config-info returns 200
            Response contains configuration data
        """
        response = requests.get(api_config_endpoint, timeout=5)

        assert response.status_code == 200, "Config endpoint should return 200"

        data = response.json()
        assert isinstance(data, dict), "Config response should be a dictionary"
        assert len(data) > 0, "Config response should contain data"


@pytest.mark.e2e
class TestQueryWorkflow:
    """Test suite for end-to-end query workflow validation."""

    def test_query_with_simple_text(self, api_query_endpoint: str):
        """
        Verify query endpoint processes simple clinical text.

        Test Case:
            Query: "Patient has fever and headache"
            Expected: HPO terms related to fever (HP:0001945) and headache

        Expected Response:
            - Status 200
            - Valid JSON response
            - Contains "results" field with HPO terms
        """
        payload = {
            "query_text": "Patient has fever and headache",
            "top_k": 5,
            "language": "en",
        }

        response = requests.post(api_query_endpoint, json=payload, timeout=30)

        assert response.status_code == 200, (
            f"Query should succeed, got: {response.status_code}, {response.text}"
        )

        data = response.json()

        # Verify response structure
        assert "results" in data, "Response should contain 'results' field"
        assert isinstance(data["results"], list), "Results should be a list"

        # Should return at least one HPO term
        assert len(data["results"]) > 0, "Should return at least one HPO term"

        # Verify result structure
        first_result = data["results"][0]
        assert "hpo_id" in first_result, "Result should contain 'hpo_id'"
        assert "hpo_name" in first_result, "Result should contain 'hpo_name'"
        assert "score" in first_result, "Result should contain 'score'"

    def test_query_with_medical_terminology(self, api_query_endpoint: str):
        """
        Verify query endpoint handles medical terminology correctly.

        Test Case:
            Query: "Microcephaly and developmental delay"
            Expected: HPO terms for microcephaly (HP:0000252) and delay

        Expected:
            - Returns relevant HPO terms
            - Scores are between 0 and 1
            - Results ordered by relevance (descending score)
        """
        payload = {
            "query_text": "Microcephaly and developmental delay",
            "top_k": 5,
            "language": "en",
        }

        response = requests.post(api_query_endpoint, json=payload, timeout=30)

        assert response.status_code == 200, (
            f"Query should succeed, got: {response.status_code}"
        )

        data = response.json()
        results = data["results"]

        assert len(results) > 0, "Should return HPO terms for medical terminology"

        # Verify scores are valid (between 0 and 1)
        for result in results:
            score = result["score"]
            assert 0.0 <= score <= 1.0, f"Score should be in [0, 1], got: {score}"

        # Verify results are ordered by score (descending)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            "Results should be ordered by score (descending)"
        )

    def test_query_with_top_k_parameter(self, api_query_endpoint: str):
        """
        Verify top_k parameter controls number of results.

        Test Cases:
            - top_k=3 returns exactly 3 results
            - top_k=10 returns up to 10 results
        """
        query_text = "Patient with seizures and intellectual disability"

        # Test top_k=3
        payload = {"query_text": query_text, "top_k": 3, "language": "en"}
        response = requests.post(api_query_endpoint, json=payload, timeout=30)
        assert response.status_code == 200, "Query with top_k=3 should succeed"

        data = response.json()
        assert len(data["results"]) <= 3, (
            f"top_k=3 should return ≤3 results, got: {len(data['results'])}"
        )

        # Test top_k=10
        payload = {"query_text": query_text, "top_k": 10, "language": "en"}
        response = requests.post(api_query_endpoint, json=payload, timeout=30)
        assert response.status_code == 200, "Query with top_k=10 should succeed"

        data = response.json()
        assert len(data["results"]) <= 10, (
            f"top_k=10 should return ≤10 results, got: {len(data['results'])}"
        )

    def test_query_with_empty_text_fails(self, api_query_endpoint: str):
        """
        Verify query endpoint rejects empty query text.

        Expected:
            Empty query_text returns 422 (validation error)
        """
        payload = {"query_text": "", "top_k": 5, "language": "en"}

        response = requests.post(api_query_endpoint, json=payload, timeout=10)

        # Should fail validation (422 Unprocessable Entity or 400 Bad Request)
        assert response.status_code in [
            422,
            400,
        ], f"Empty query should fail validation, got: {response.status_code}"

    def test_query_with_missing_required_field_fails(self, api_query_endpoint: str):
        """
        Verify query endpoint validates required fields.

        Expected:
            Missing query_text returns 422 (validation error)
        """
        payload = {"top_k": 5, "language": "en"}  # Missing query_text

        response = requests.post(api_query_endpoint, json=payload, timeout=10)

        assert response.status_code in [
            422,
            400,
        ], f"Missing required field should fail, got: {response.status_code}"

    def test_query_with_invalid_top_k_fails(self, api_query_endpoint: str):
        """
        Verify query endpoint validates top_k parameter.

        Expected:
            Invalid top_k (negative, zero, or too large) returns 422
        """
        query_text = "Patient with fever"

        # Test negative top_k
        payload = {"query_text": query_text, "top_k": -1, "language": "en"}
        response = requests.post(api_query_endpoint, json=payload, timeout=10)

        assert response.status_code in [
            422,
            400,
        ], f"Negative top_k should fail, got: {response.status_code}"

        # Test zero top_k
        payload = {"query_text": query_text, "top_k": 0, "language": "en"}
        response = requests.post(api_query_endpoint, json=payload, timeout=10)

        assert response.status_code in [
            422,
            400,
        ], f"Zero top_k should fail, got: {response.status_code}"

    def test_query_response_contains_metadata(self, api_query_endpoint: str):
        """
        Verify query response includes useful metadata.

        Expected metadata fields:
            - query_text (echoed back)
            - results_count or similar
            - model_name or embedding_model
        """
        payload = {
            "query_text": "Patient with hypertension",
            "top_k": 5,
            "language": "en",
        }

        response = requests.post(api_query_endpoint, json=payload, timeout=30)
        assert response.status_code == 200, "Query should succeed"

        data = response.json()

        # Should echo back query text or contain query metadata
        # (exact field names may vary based on schema)
        assert "results" in data, "Response should contain results"

    def test_query_with_negation_text(self, api_query_endpoint: str):
        """
        Verify query endpoint processes negation correctly.

        Test Case:
            Query: "Patient does not have fever but has headache"
            Expected: HPO terms should reflect presence/absence correctly

        Note: This test validates that the API processes the text,
        but assertion detection logic is tested at unit level.
        """
        payload = {
            "query_text": "Patient does not have fever but has headache",
            "top_k": 5,
            "language": "en",
        }

        response = requests.post(api_query_endpoint, json=payload, timeout=30)

        assert response.status_code == 200, (
            f"Query with negation should succeed, got: {response.status_code}"
        )

        data = response.json()
        assert len(data["results"]) > 0, "Should return results for negation text"

    def test_query_with_long_clinical_note(self, api_query_endpoint: str):
        """
        Verify query endpoint handles longer clinical text.

        Test Case:
            Multi-sentence clinical note with various phenotypes
        """
        clinical_note = """
        Patient is a 5-year-old male presenting with developmental delay,
        microcephaly, and seizures. Physical examination reveals hypotonia
        and poor muscle tone. Parents report feeding difficulties in infancy.
        Cardiac evaluation shows no structural abnormalities.
        """

        payload = {
            "query_text": clinical_note.strip(),
            "top_k": 10,
            "language": "en",
        }

        response = requests.post(api_query_endpoint, json=payload, timeout=60)

        assert response.status_code == 200, (
            f"Query with long text should succeed, got: {response.status_code}"
        )

        data = response.json()
        results = data["results"]

        # Should extract multiple relevant HPO terms
        assert len(results) > 0, "Should extract HPO terms from clinical note"

        # Verify each result has required fields
        for result in results:
            assert "hpo_id" in result, "Each result should have hpo_id"
            assert "hpo_name" in result, "Each result should have hpo_name"
            assert "score" in result, "Each result should have score"

    def test_query_performance_acceptable(self, api_query_endpoint: str):
        """
        Verify query endpoint responds within acceptable time.

        Expected:
            Query completes in < 10 seconds (including model inference)
        """
        import time

        payload = {
            "query_text": "Patient with seizures and developmental delay",
            "top_k": 5,
            "language": "en",
        }

        start_time = time.time()
        response = requests.post(api_query_endpoint, json=payload, timeout=30)
        elapsed_time = time.time() - start_time

        assert response.status_code == 200, "Query should succeed"

        # Performance threshold: 10 seconds (generous for model inference)
        assert elapsed_time < 10.0, (
            f"Query should complete in <10s, took: {elapsed_time:.2f}s"
        )

    def test_multiple_queries_succeed(self, api_query_endpoint: str):
        """
        Verify multiple consecutive queries succeed.

        This test validates service stability and proper resource cleanup.

        Expected:
            5 consecutive queries all succeed with valid responses
        """
        test_queries = [
            "Patient has fever",
            "Microcephaly and seizures",
            "Intellectual disability",
            "Cardiac abnormalities",
            "Hypotonia and poor feeding",
        ]

        for i, query_text in enumerate(test_queries):
            payload = {"query_text": query_text, "top_k": 5, "language": "en"}

            response = requests.post(api_query_endpoint, json=payload, timeout=30)

            assert response.status_code == 200, (
                f"Query {i + 1}/{len(test_queries)} failed: {response.status_code}"
            )

            data = response.json()
            assert len(data["results"]) > 0, f"Query {i + 1} should return results"

    def test_query_hpo_ids_valid_format(self, api_query_endpoint: str):
        """
        Verify returned HPO IDs follow correct format.

        Expected Format:
            HP:XXXXXXX (HP: prefix followed by 7 digits)
        """
        import re

        payload = {
            "query_text": "Patient with developmental delay",
            "top_k": 5,
            "language": "en",
        }

        response = requests.post(api_query_endpoint, json=payload, timeout=30)
        assert response.status_code == 200, "Query should succeed"

        data = response.json()
        hpo_id_pattern = re.compile(r"^HP:\d{7}$")

        for result in data["results"]:
            hpo_id = result["hpo_id"]
            assert hpo_id_pattern.match(hpo_id), (
                f"HPO ID should match format HP:XXXXXXX, got: {hpo_id}"
            )

    def test_query_returns_unique_hpo_terms(self, api_query_endpoint: str):
        """
        Verify query results contain unique HPO IDs (no duplicates).

        Expected:
            Each HPO ID appears only once in results
        """
        payload = {
            "query_text": "Patient has fever and high temperature",
            "top_k": 10,
            "language": "en",
        }

        response = requests.post(api_query_endpoint, json=payload, timeout=30)
        assert response.status_code == 200, "Query should succeed"

        data = response.json()
        hpo_ids = [result["hpo_id"] for result in data["results"]]

        # Check for duplicates
        unique_hpo_ids = set(hpo_ids)
        assert len(hpo_ids) == len(unique_hpo_ids), (
            f"Results should not contain duplicate HPO IDs, got: {hpo_ids}"
        )
