import logging

import pytest

from phentrieve.retrieval.dense_retriever import DenseRetriever

pytestmark = pytest.mark.unit


def test_query_logs_text_length_without_raw_query(mocker, caplog):
    retriever = DenseRetriever.__new__(DenseRetriever)
    query_text = "Herr Bernt Popp ist dumm"
    mocker.patch.object(
        retriever,
        "query_batch",
        return_value=[
            {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        ],
    )

    with caplog.at_level(logging.INFO):
        retriever.query(query_text)

    assert query_text not in caplog.text
    assert f"text_chars={len(query_text)}" in caplog.text
