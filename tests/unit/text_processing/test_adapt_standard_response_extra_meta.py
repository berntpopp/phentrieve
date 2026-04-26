"""Tests for adapt_standard_response.extra_meta merging."""

from phentrieve.text_processing.full_text_service import adapt_standard_response


def test_extra_meta_none_preserves_existing_shape():
    response = adapt_standard_response(
        pipeline_result=[],
        extraction_result=([], []),
        extra_meta=None,
    )
    assert "meta" in response
    assert response["meta"]["extraction_backend"] == "standard"
    # No new keys introduced.
    assert "adaptive_rechunking" not in response["meta"]


def test_extra_meta_dict_merges_into_meta():
    extra = {"adaptive_rechunking": {"enabled": True, "trigger_count": 3}}
    response = adapt_standard_response(
        pipeline_result=[],
        extraction_result=([], []),
        extra_meta=extra,
    )
    assert response["meta"]["adaptive_rechunking"] == {
        "enabled": True,
        "trigger_count": 3,
    }
    # Existing meta keys preserved.
    assert response["meta"]["extraction_backend"] == "standard"


def test_extra_meta_does_not_overwrite_extraction_backend():
    """If extra_meta accidentally includes extraction_backend, the canonical
    value from the response wins (we set extraction_backend last)."""
    extra = {"extraction_backend": "wrong"}
    response = adapt_standard_response(
        pipeline_result=[],
        extraction_result=([], []),
        extra_meta=extra,
    )
    assert response["meta"]["extraction_backend"] == "standard"
