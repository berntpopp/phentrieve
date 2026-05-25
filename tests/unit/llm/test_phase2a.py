from __future__ import annotations

import pytest

from phentrieve.llm.phase2a import retrieve_candidates

pytestmark = pytest.mark.unit


class FakeToolExecutor:
    def __init__(self, batch_results: list[dict[str, object]]) -> None:
        self.batch_results = list(batch_results)
        self.queries: list[dict[str, object]] = []

    def query_batch_hpo_terms(self, *, phrases, language, n_results):
        self.queries.append(
            {
                "phrases": list(phrases),
                "language": language,
                "n_results": n_results,
            }
        )
        return list(self.batch_results)


def test_retrieve_candidates_merges_query_variants_and_preserves_grounding() -> None:
    tool_executor = FakeToolExecutor(
        [
            {
                "phrase": "serum creatinine 11.2 mg/dL",
                "candidates": [
                    {
                        "hpo_id": "HP:0003259",
                        "term_name": "Elevated serum creatinine",
                        "score": 0.70,
                    }
                ],
            },
            {
                "phrase": "serum creatinine 11.2",
                "candidates": [
                    {
                        "hpo_id": "HP:0003259",
                        "term_name": "Elevated serum creatinine",
                        "score": 0.91,
                    }
                ],
            },
        ]
    )

    results = retrieve_candidates(
        actionable=[
            {
                "phrase": "serum creatinine 11.2 mg/dL",
                "category": "Abnormal",
                "chunk_ids": [2],
                "evidence_text": "serum creatinine 11.2 mg/dL",
                "start_char": 9,
                "end_char": 36,
            }
        ],
        grounded_chunks=[
            {"chunk_id": 1, "text": "Prior history."},
            {"chunk_id": 2, "text": "Labs show serum creatinine 11.2 mg/dL."},
            {"chunk_id": 3, "text": "Follow-up was arranged."},
        ],
        language="en",
        tool_executor=tool_executor,
        n_results_per_phrase=50,
        max_unique_candidates=10,
        min_unique_candidates=3,
        similarity_threshold=0.60,
    )

    assert tool_executor.queries == [
        {
            "phrases": ["serum creatinine 11.2 mg/dL", "serum creatinine 11.2"],
            "language": "en",
            "n_results": 50,
        }
    ]
    assert results == [
        {
            "phrase": "serum creatinine 11.2 mg/dL",
            "category": "Abnormal",
            "candidates": [
                {
                    "hpo_id": "HP:0003259",
                    "term_name": "Elevated serum creatinine",
                    "score": 0.91,
                    "retrieval_query": "serum creatinine 11.2",
                }
            ],
            "chunk_ids": [2],
            "evidence_text": "serum creatinine 11.2 mg/dL",
            "start_char": 9,
            "end_char": 36,
            "grounded_context": {
                "chunk_ids": [2],
                "primary_chunk_text": "Labs show serum creatinine 11.2 mg/dL.",
                "neighbor_chunk_texts": ["Prior history.", "Follow-up was arranged."],
            },
        }
    ]


def test_retrieve_candidates_adds_grounded_context_head_variants() -> None:
    tool_executor = FakeToolExecutor(
        [
            {
                "phrase": "unilateral",
                "candidates": [
                    {
                        "hpo_id": "HP:0012833",
                        "term_name": "Unilateral",
                        "score": 0.99,
                    }
                ],
            },
            {
                "phrase": "unilateral seizures",
                "candidates": [
                    {
                        "hpo_id": "HP:0006813",
                        "term_name": "Focal hemiclonic seizure",
                        "score": 0.88,
                    }
                ],
            },
        ]
    )

    results = retrieve_candidates(
        actionable=[
            {
                "phrase": "unilateral",
                "category": "Abnormal",
                "chunk_ids": [2],
                "evidence_text": "unilateral",
            }
        ],
        grounded_chunks=[
            {"chunk_id": 1, "text": "following seizures since infancy"},
            {"chunk_id": 2, "text": "initially unilateral prior to becoming GTC"},
        ],
        language="en",
        tool_executor=tool_executor,
        n_results_per_phrase=50,
        max_unique_candidates=10,
        min_unique_candidates=3,
        similarity_threshold=0.60,
    )

    assert tool_executor.queries == [
        {
            "phrases": ["unilateral", "unilateral seizures"],
            "language": "en",
            "n_results": 50,
        }
    ]
    assert results[0]["candidates"] == [
        {
            "hpo_id": "HP:0006813",
            "term_name": "Focal hemiclonic seizure",
            "score": 0.88,
            "retrieval_query": "unilateral seizures",
        }
    ]
