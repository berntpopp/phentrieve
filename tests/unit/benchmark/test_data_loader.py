from __future__ import annotations

import json

import pytest

from phentrieve.benchmark.data_loader import (
    _get_case_text,
    _validate_document_benchmark_payload,
    load_json_benchmark_data,
    load_phenobert_data,
)


def test_load_json_benchmark_data_uses_generic_list_error(tmp_path) -> None:
    test_file = tmp_path / "cases.json"
    test_file.write_text('{"unexpected": true}', encoding="utf-8")

    with pytest.raises(
        ValueError, match="Benchmark JSON objects must contain a 'documents' list."
    ):
        load_json_benchmark_data(test_file)


def test_validate_document_benchmark_payload_uses_generic_empty_error() -> None:
    with pytest.raises(ValueError, match="No benchmark documents found"):
        _validate_document_benchmark_payload({"documents": []}, source_name="sample")


def test_get_case_text_uses_generic_non_empty_text_error() -> None:
    with pytest.raises(
        ValueError, match="Each benchmark case must provide non-empty text."
    ):
        _get_case_text({})


def test_load_phenobert_data_accepts_raghpo_paper_datasets(tmp_path) -> None:
    annotations_dir = tmp_path / "CSC" / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "CSC_1.json").write_text(
        json.dumps(
            {
                "doc_id": "CSC_1",
                "full_text": "Patient has obesity.",
                "annotations": [
                    {
                        "hpo_id": "HP:0001513",
                        "label": "Obesity",
                        "assertion_status": "affirmed",
                        "evidence_spans": [],
                    },
                    {
                        "hpo_id": "HP:0001250",
                        "label": "Seizure",
                        "assertion_status": "family_history",
                        "evidence_spans": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = load_phenobert_data(tmp_path, dataset="CSC")

    assert payload["metadata"]["dataset_name"] == "phenobert_CSC"
    assert payload["metadata"]["source"] == "rag_hpo_paper"
    assert payload["metadata"]["dataset_namespace"] == "rag_hpo_paper"
    assert payload["documents"][0]["id"] == "CSC_1"
    assert payload["documents"][0]["source_dataset"] == "CSC"
    assert payload["documents"][0]["gold_hpo_terms"] == [
        {
            "id": "HP:0001513",
            "label": "Obesity",
            "assertion": "PRESENT",
            "evidence_spans": [],
        }
    ]
