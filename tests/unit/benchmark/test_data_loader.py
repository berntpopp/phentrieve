from __future__ import annotations

import pytest

from phentrieve.benchmark.data_loader import (
    _get_case_text,
    _validate_document_benchmark_payload,
    load_json_benchmark_data,
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
