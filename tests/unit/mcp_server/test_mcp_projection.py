"""MCP-boundary projection: collapse the shared-service extraction schema to a
single score field and a single chunk-index scheme, normalize keys, drop empty
chunks, and always serialize hpo_matches (M4 / T1 / L5 / L7)."""

import pytest

from api.mcp.projection import (
    project_aggregated_terms_for_mcp,
    project_extract_payload,
    project_processed_chunks_for_mcp,
)

pytestmark = pytest.mark.unit

# Shape mirrors the live shared-service output (see 2026-06-14-mcp-baseline.md).
RAW_TERM = {
    "id": "HP:0001250",
    "name": "Seizure",
    "score": 0.91,
    "count": 2,
    "evidence_count": 2,
    "avg_score": 0.88,
    "confidence": 0.88,
    "chunks": [0, 2],
    "top_evidence_chunk_idx": 0,
    "assertion_status": "affirmed",
    "status": "affirmed",
    "rank": 1,
    "source_chunk_ids": [1, 3],
    "max_score_from_evidence": 0.91,
    "top_evidence_chunk_id": 1,
}


def test_single_score_field():
    out = project_aggregated_terms_for_mcp([RAW_TERM])[0]
    assert out["score"] == 0.91
    for dup in ("avg_score", "confidence", "max_score_from_evidence"):
        assert dup not in out


def test_synonyms_capped_with_truncation_count():
    """R3: an uncapped synonym list in the response is a token footgun; cap it to
    10 and report how many were dropped, without touching the (separate) list used
    for attribution matching."""
    term = {
        **RAW_TERM,
        "synonyms": [f"syn{i}" for i in range(15)],
    }
    out = project_aggregated_terms_for_mcp([term])[0]
    assert out["synonyms"] == [f"syn{i}" for i in range(10)]
    assert out["synonyms_truncated"] == 5


def test_synonyms_not_capped_when_within_limit():
    term = {**RAW_TERM, "synonyms": ["a", "b", "c"]}
    out = project_aggregated_terms_for_mcp([term])[0]
    assert out["synonyms"] == ["a", "b", "c"]
    assert "synonyms_truncated" not in out


def test_single_chunk_index_scheme():
    out = project_aggregated_terms_for_mcp([RAW_TERM])[0]
    assert out["chunk_ids"] == [1, 3]
    assert out["top_evidence_chunk_id"] == 1
    for dup in ("chunks", "top_evidence_chunk_idx", "source_chunk_ids"):
        assert dup not in out


def test_normalized_identity_keys():
    out = project_aggregated_terms_for_mcp([RAW_TERM])[0]
    assert out["hpo_id"] == "HP:0001250"
    assert out["label"] == "Seizure"
    assert out["assertion"] == "affirmed"
    assert out["rank"] == 1
    assert out["evidence_count"] == 2
    for old in ("id", "name", "status", "assertion_status", "count"):
        assert old not in out


def test_chunks_only_zero_based_fallback():
    raw = {
        k: v
        for k, v in RAW_TERM.items()
        if k not in ("source_chunk_ids", "top_evidence_chunk_id")
    }
    out = project_aggregated_terms_for_mcp([raw])[0]
    assert out["chunk_ids"] == [1, 3]  # 0-based [0,2] -> 1-based
    assert out["top_evidence_chunk_id"] == 1  # 0-based 0 -> 1


def test_text_attributions_and_details_passthrough():
    raw = {
        **RAW_TERM,
        "text_attributions": [{"start_char": 4, "end_char": 11}],
        "definition": "A seizure.",
        "synonyms": ["fit"],
    }
    out = project_aggregated_terms_for_mcp([raw])[0]
    assert out["text_attributions"] == [{"start_char": 4, "end_char": 11}]
    assert out["definition"] == "A seizure."
    assert out["synonyms"] == ["fit"]


RAW_CHUNKS = [
    {"chunk_id": 1, "text": "no fever", "status": "negated", "hpo_matches": []},
    {
        "chunk_id": 2,
        "text": "seizures",
        "status": "affirmed",
        "hpo_matches": [
            {
                "id": "HP:0001250",
                "name": "Seizure",
                "score": 0.91,
                "assertion_status": "affirmed",
            }
        ],
    },
]


def test_empty_match_chunk_dropped_by_default():
    out = project_processed_chunks_for_mcp(RAW_CHUNKS)
    assert [c["chunk_id"] for c in out] == [2]


def test_empty_match_chunk_kept_when_opted_in_with_empty_list():
    out = project_processed_chunks_for_mcp(RAW_CHUNKS, include_unmatched=True)
    assert [c["chunk_id"] for c in out] == [1, 2]
    assert out[0]["hpo_matches"] == []  # present, not omitted (L5)


def test_chunk_match_keys_normalized():
    out = project_processed_chunks_for_mcp(RAW_CHUNKS)
    m = out[0]["hpo_matches"][0]
    assert m["hpo_id"] == "HP:0001250"
    assert m["label"] == "Seizure"
    assert m["assertion"] == "affirmed"
    assert m["score"] == 0.91
    for old in ("id", "name", "assertion_status"):
        assert old not in m


def test_project_extract_payload_applies_both():
    payload = {
        "aggregated_hpo_terms": [RAW_TERM],
        "processed_chunks": RAW_CHUNKS,
        "meta": {},
    }
    out = project_extract_payload(payload, include_unmatched_chunks=False)
    assert out["aggregated_hpo_terms"][0]["hpo_id"] == "HP:0001250"
    assert [c["chunk_id"] for c in out["processed_chunks"]] == [2]
    assert out["meta"] == {}  # untouched
