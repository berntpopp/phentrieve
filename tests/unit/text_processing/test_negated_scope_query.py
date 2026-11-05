"""Negated full-text chunks also query their scope without the cue (#472)."""

from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)

_RESULTS = {
    # The cue pulls the chunk embedding toward the wrong term.
    "Seizures were ruled out.": [("HP:0002121", "Absence seizure", 0.70)],
    "Seizures": [
        ("HP:0001250", "Seizure", 0.90),
        ("HP:0002121", "Absence seizure", 0.60),
    ],
}


class _FakeRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def detect_index_type(self):
        return "single_vector"

    def query_batch(self, texts, n_results=10, include_similarities=True):
        self.queries.extend(texts)
        return [
            {
                "ids": [[hpo_id for hpo_id, _, _ in _RESULTS[text]]],
                "metadatas": [
                    [{"id": i, "label": label} for i, label, _ in _RESULTS[text]]
                ],
                "similarities": [[score for _, _, score in _RESULTS[text]]],
            }
            for text in texts
        ]


def _run(status, scope_texts):
    retriever = _FakeRetriever()
    result = orchestrate_hpo_extraction(
        text_chunks=["Seizures were ruled out."],
        retriever=retriever,
        num_results_per_chunk=5,
        chunk_retrieval_threshold=0.0,
        assertion_statuses=[status],
        chunk_negated_scope_texts=[scope_texts],
    )
    return retriever, result


def test_negated_chunk_ranks_the_scope_term_first():
    retriever, result = _run("negated", ["Seizures"])

    assert retriever.queries == ["Seizures were ruled out.", "Seizures"]
    matches = result.chunk_results[0]["matches"]
    assert [m["id"] for m in matches] == ["HP:0001250", "HP:0002121"]
    # Absence seizure keeps its higher chunk score, not the scope score.
    assert matches[1]["score"] == 0.70


def test_affirmed_chunk_is_queried_once():
    retriever, result = _run("affirmed", ["Seizures"])

    assert retriever.queries == ["Seizures were ruled out."]
    assert [m["id"] for m in result.chunk_results[0]["matches"]] == ["HP:0002121"]
