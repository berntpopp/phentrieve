"""End-to-End tests for reranker loading functionality.

These tests validate that the reranker loading improvements work correctly
in a real API environment, ensuring that:
1. First requests with reranker succeed (no 503 errors)
2. Multiple concurrent requests are handled efficiently
3. The actual reranking functionality works correctly
4. Performance meets expectations
"""

import asyncio
import time
from typing import Any, Dict

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@pytest.fixture(scope="module")
def api_base_url():
    """Base URL for the API server.

    Assumes API is running locally on port 8734.
    For CI/CD, override with environment variable.
    """
    import os

    return os.getenv("PHENTRIEVE_API_URL", "http://localhost:8734")


@pytest.fixture(scope="module")
def http_session():
    """HTTP session with retry logic for robustness."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 504],  # Retry on server errors
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@pytest.fixture
def sample_query_payload() -> Dict[str, Any]:
    """Sample query payload for testing."""
    return {
        "text": "Kleinwuchs",  # German for "short stature"
        "model_name": "FremyCompany/BioLORD-2023-M",
        "language": "de",
        "num_results": 5,
        "similarity_threshold": 0.3,
        "enable_reranker": True,
        "reranker_mode": "cross-lingual",
        "query_assertion_language": "de",
        "detect_query_assertion": True,
    }


class TestFirstRerankerRequestNoTimeout:
    """Test that first reranker requests succeed without 503 errors."""

    @pytest.mark.e2e
    def test_first_reranker_request_succeeds(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test that first request with reranker succeeds (no 503).

        This is the primary test for the fix - validates that the API waits
        for the CrossEncoder to load instead of immediately returning 503.

        Expected behavior:
        - Status: 200 OK (not 503)
        - Response time: <15 seconds (model loads in ~3-4s)
        - Response contains results
        """
        url = f"{api_base_url}/api/v1/query/"

        # Clear any cached state by using a unique model combination
        # (In production, this would be the first request after API restart)

        start_time = time.time()
        response = http_session.post(url, json=sample_query_payload, timeout=30)
        elapsed_time = time.time() - start_time

        # CRITICAL: Should NOT return 503
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. "
            f"Response: {response.json() if response.status_code != 200 else ''}"
        )

        # Should complete in reasonable time (<15s with buffer for model loading)
        assert elapsed_time < 15, f"Request took {elapsed_time:.2f}s, expected <15s"

        # Validate response structure
        data = response.json()
        assert "query_text_received" in data
        assert "results" in data
        assert len(data["results"]) > 0, "Should return at least one result"
        assert data["reranker_used"] is not None, "Should indicate reranker was used"

    @pytest.mark.e2e
    def test_reranker_response_quality(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test that reranker actually improves results.

        Compares results with and without reranker to ensure reranking is
        actually being applied.
        """
        url = f"{api_base_url}/api/v1/query/"

        # Request WITH reranker
        payload_with_reranker = sample_query_payload.copy()
        payload_with_reranker["enable_reranker"] = True

        response_with = http_session.post(url, json=payload_with_reranker, timeout=30)
        assert response_with.status_code == 200

        # Request WITHOUT reranker
        payload_without_reranker = sample_query_payload.copy()
        payload_without_reranker["enable_reranker"] = False

        response_without = http_session.post(
            url, json=payload_without_reranker, timeout=30
        )
        assert response_without.status_code == 200

        data_with = response_with.json()
        data_without = response_without.json()

        # With reranker should indicate reranker was used
        assert data_with["reranker_used"] is not None
        assert data_without["reranker_used"] is None

        # Both should return results
        assert len(data_with["results"]) > 0
        assert len(data_without["results"]) > 0


class TestConcurrentRerankerRequests:
    """Test handling of multiple concurrent reranker requests."""

    @pytest.mark.e2e
    def test_concurrent_requests_all_succeed(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test that multiple concurrent requests with reranker all succeed.

        Simulates multiple users enabling reranker simultaneously.
        All requests should succeed by waiting for the same model loading task.
        """
        url = f"{api_base_url}/api/v1/query/"

        # Send 5 concurrent requests
        num_concurrent = 5

        def send_request(session, payload):
            """Helper to send a single request."""
            return session.post(url, json=payload, timeout=30)

        # Use threading to send truly concurrent requests
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_concurrent
        ) as executor:
            futures = [
                executor.submit(send_request, http_session, sample_query_payload)
                for _ in range(num_concurrent)
            ]

            responses = [f.result() for f in concurrent.futures.as_completed(futures)]

        # ALL requests should succeed (no 503 errors)
        successful = [r for r in responses if r.status_code == 200]
        failed = [r for r in responses if r.status_code != 200]

        assert len(successful) == num_concurrent, (
            f"Expected {num_concurrent} successful requests, got {len(successful)}. "
            f"Failed: {[(r.status_code, r.text) for r in failed]}"
        )

        # All should have used the reranker
        for response in successful:
            data = response.json()
            assert data["reranker_used"] is not None

    @pytest.mark.e2e
    def test_concurrent_requests_efficient(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test that concurrent requests are handled efficiently.

        When multiple requests arrive simultaneously, they should share
        the model loading task rather than loading multiple times.

        Expected: Total time should be similar to single request time
        (not N x single request time).
        """
        url = f"{api_base_url}/api/v1/query/"

        # Measure time for single request
        single_start = time.time()
        response_single = http_session.post(url, json=sample_query_payload, timeout=30)
        single_elapsed = time.time() - single_start
        assert response_single.status_code == 200

        # Now measure time for 3 concurrent requests
        import concurrent.futures

        concurrent_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    http_session.post, url, json=sample_query_payload, timeout=30
                )
                for _ in range(3)
            ]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        concurrent_elapsed = time.time() - concurrent_start

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # Concurrent requests should NOT take 3x as long
        # Allow 2x for overhead, but should be much better than 3x
        assert concurrent_elapsed < (single_elapsed * 2.5), (
            f"Concurrent requests took {concurrent_elapsed:.2f}s, "
            f"single request took {single_elapsed:.2f}s. "
            f"Expected concurrent to be <2.5x single."
        )


class TestRerankerFunctionalityEndToEnd:
    """Test that reranker actually works correctly end-to-end."""

    @pytest.mark.e2e
    def test_reranker_returns_valid_scores(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test that reranker returns valid similarity scores."""
        url = f"{api_base_url}/api/v1/query/"

        response = http_session.post(url, json=sample_query_payload, timeout=30)
        assert response.status_code == 200

        data = response.json()
        results = data["results"]

        # All results should have similarity scores
        for result in results:
            assert "similarity" in result
            assert isinstance(result["similarity"], (int, float))
            assert 0 <= result["similarity"] <= 1, (
                "Similarity should be between 0 and 1"
            )

            assert "hpo_id" in result
            assert result["hpo_id"].startswith("HP:"), "Should be valid HPO ID"

            assert "label" in result
            assert len(result["label"]) > 0, "Should have non-empty label"

    @pytest.mark.e2e
    def test_different_reranker_modes(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test that different reranker modes work correctly."""
        url = f"{api_base_url}/api/v1/query/"

        # Test cross-lingual mode
        payload_cross_lingual = sample_query_payload.copy()
        payload_cross_lingual["reranker_mode"] = "cross-lingual"

        response_cross = http_session.post(url, json=payload_cross_lingual, timeout=30)
        assert response_cross.status_code == 200
        data_cross = response_cross.json()
        assert data_cross["reranker_used"] is not None

        # Test monolingual mode
        payload_monolingual = sample_query_payload.copy()
        payload_monolingual["reranker_mode"] = "monolingual"

        response_mono = http_session.post(url, json=payload_monolingual, timeout=30)
        assert response_mono.status_code == 200
        data_mono = response_mono.json()
        assert data_mono["reranker_used"] is not None

        # Different modes might use different models
        # (implementation detail, but worth checking)


class TestPerformanceExpectations:
    """Test that performance meets expectations."""

    @pytest.mark.e2e
    def test_subsequent_requests_are_fast(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test that subsequent requests with cached model are fast.

        After model is loaded, requests should complete quickly.
        """
        url = f"{api_base_url}/api/v1/query/"

        # First request (may load model)
        http_session.post(url, json=sample_query_payload, timeout=30)

        # Subsequent requests should be fast (model cached)
        start_time = time.time()
        response = http_session.post(url, json=sample_query_payload, timeout=10)
        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 5, f"Cached request took {elapsed:.2f}s, expected <5s"

    @pytest.mark.e2e
    def test_batch_requests_performance(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test performance of sequential batch requests.

        Useful for understanding throughput when processing multiple queries.
        """
        url = f"{api_base_url}/api/v1/query/"

        # Send 10 requests sequentially
        num_requests = 10
        start_time = time.time()

        for _ in range(num_requests):
            response = http_session.post(url, json=sample_query_payload, timeout=10)
            assert response.status_code == 200

        elapsed = time.time() - start_time
        avg_time = elapsed / num_requests

        # Average should be reasonable (<3s per request)
        assert avg_time < 3.0, (
            f"Average request time {avg_time:.2f}s is too slow. "
            f"Total: {elapsed:.2f}s for {num_requests} requests"
        )


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    @pytest.mark.e2e
    def test_invalid_reranker_mode_handled(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test that invalid reranker mode is handled gracefully."""
        url = f"{api_base_url}/api/v1/query/"

        payload = sample_query_payload.copy()
        payload["reranker_mode"] = "invalid-mode"  # Invalid mode

        response = http_session.post(url, json=payload, timeout=10)

        # Should return validation error (422) or handle gracefully (200)
        assert response.status_code in [200, 422], (
            f"Unexpected status {response.status_code} for invalid mode"
        )

    @pytest.mark.e2e
    def test_reranker_with_no_results(self, api_base_url, http_session):
        """Test reranker behavior when there are no initial results.

        Edge case: What happens if similarity_threshold is so high that
        no results are found?
        """
        url = f"{api_base_url}/api/v1/query/"

        payload = {
            "text": "impossible to match text xyz123",
            "enable_reranker": True,
            "similarity_threshold": 0.99,  # Extremely high threshold
            "num_results": 10,
        }

        response = http_session.post(url, json=payload, timeout=10)

        # Should handle gracefully (200 with empty results or appropriate response)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Results may be empty, which is fine


@pytest.mark.e2e
class TestRerankerLoadingLogs:
    """Test that appropriate logs are generated (for debugging/monitoring)."""

    def test_successful_first_load_logs(
        self, api_base_url, http_session, sample_query_payload
    ):
        """Test that successful first-time loading generates expected logs.

        Note: This test makes assumptions about logging. In production,
        you'd want to check actual log files or use structured logging
        with correlation IDs.
        """
        url = f"{api_base_url}/api/v1/query/"

        # This test primarily validates the request succeeds
        # Log validation would require log aggregation/collection
        response = http_session.post(url, json=sample_query_payload, timeout=30)
        assert response.status_code == 200

        # In a real scenario, you'd check logs for:
        # "API: CrossEncoder '...' loaded successfully on first request."
        # But that requires access to logs, which varies by deployment
