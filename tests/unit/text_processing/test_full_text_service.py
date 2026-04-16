from types import SimpleNamespace

import pytest

from phentrieve.llm.types import LLMExtractionResult, LLMMeta, LLMPhenotype
from phentrieve.text_processing.full_text_service import (
    FullTextService,
    adapt_standard_response,
    run_llm_backend,
)


def test_full_text_service_uses_standard_backend(mocker):
    standard_backend = mocker.Mock(
        return_value={"meta": {"extraction_backend": "standard"}}
    )
    llm_backend = mocker.Mock()
    service = FullTextService(
        standard_backend=standard_backend,
        llm_backend=llm_backend,
    )

    result = service.process(text="clinical text", extraction_backend="standard")

    assert result["meta"]["extraction_backend"] == "standard"
    standard_backend.assert_called_once()
    llm_backend.assert_not_called()


def test_full_text_service_llm_response_can_return_empty_chunks(mocker):
    llm_backend = mocker.Mock(
        return_value={
            "meta": {"extraction_backend": "llm", "llm_mode": "two_phase"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [{"id": "HP:0001250", "name": "Seizure"}],
        }
    )
    service = FullTextService(standard_backend=mocker.Mock(), llm_backend=llm_backend)

    result = service.process(text="clinical text", extraction_backend="llm")

    assert result["processed_chunks"] == []
    assert result["meta"]["extraction_backend"] == "llm"


def test_run_llm_backend_surfaces_token_usage(mocker):
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    pipeline.run.return_value = LLMExtractionResult(
        terms=[
            LLMPhenotype(
                term_id="HP:0001250",
                label="Seizure",
                evidence="Patient had recurrent seizures.",
            )
        ],
        meta=LLMMeta(
            llm_model="gpt-4o-mini",
            llm_mode="two_phase",
            prompt_version="v9",
            token_input=12,
            token_output=34,
        ),
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.get_llm_provider",
        return_value=provider,
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.TwoPhaseLLMPipeline",
        return_value=pipeline,
    )

    result = run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gpt-4o-mini",
        llm_mode="two_phase",
    )

    assert result["meta"]["token_input"] == 12
    assert result["meta"]["token_output"] == 34
    assert result["meta"]["prompt_version"] == "v9"


def test_run_llm_backend_surfaces_evidence_records(mocker):
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    pipeline.run.return_value = LLMExtractionResult(
        terms=[
            LLMPhenotype(
                term_id="HP:0001250",
                label="Seizure",
                evidence="recurrent seizures",
                evidence_records=[
                    {
                        "phrase": "recurrent seizures",
                        "evidence_text": "recurrent seizures",
                        "chunk_ids": [1],
                        "match_method": "local",
                    }
                ],
            )
        ],
        meta=LLMMeta(
            llm_model="gpt-4o-mini",
            llm_mode="two_phase",
        ),
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.get_llm_provider",
        return_value=provider,
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.TwoPhaseLLMPipeline",
        return_value=pipeline,
    )

    result = run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gpt-4o-mini",
        llm_mode="two_phase",
    )

    assert result["aggregated_hpo_terms"][0]["evidence_records"][0]["chunk_ids"] == [1]


def test_run_llm_backend_builds_grounded_chunks_for_pipeline(mocker):
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    pipeline.run.return_value = LLMExtractionResult(
        terms=[],
        meta=LLMMeta(
            llm_model="gemini-2.5-flash",
            llm_mode="two_phase",
            prompt_version="v1",
        ),
    )

    mocker.patch(
        "phentrieve.text_processing.full_text_service.get_llm_provider",
        return_value=provider,
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.TwoPhaseLLMPipeline",
        return_value=pipeline,
    )

    run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        language="en",
    )

    grounded_chunks = pipeline.run.call_args.kwargs["grounded_chunks"]
    assert grounded_chunks
    assert grounded_chunks[0]["chunk_id"] == 1


def test_run_llm_backend_supports_grounded_internal_mode(mocker):
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    pipeline.run.return_value = LLMExtractionResult(
        terms=[],
        meta=LLMMeta(
            llm_model="gemini-2.5-flash",
            llm_mode="two_phase",
        ),
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.get_llm_provider",
        return_value=provider,
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.TwoPhaseLLMPipeline",
        return_value=pipeline,
    )

    run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
    )

    assert pipeline.run.call_args.kwargs["grounded_chunks"]


def test_run_llm_backend_uses_shared_preprocessing_for_grounded_mode(mocker):
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    preprocessed = SimpleNamespace(
        grounded_chunks=[
            {"chunk_id": 1, "text": "Chunk one.", "start_char": 0, "end_char": 10},
            {"chunk_id": 2, "text": "Chunk two.", "start_char": 11, "end_char": 21},
        ],
        extraction_groups=[
            {
                "group_id": 1,
                "chunk_ids": [1, 2],
                "text": "chunk_id=1: Chunk one.\nchunk_id=2: Chunk two.",
                "estimated_prompt_tokens": 17,
            }
        ],
    )
    pipeline.run.return_value = LLMExtractionResult(
        terms=[],
        meta=LLMMeta(
            llm_model="gemini-2.5-flash",
            llm_mode="two_phase",
        ),
    )

    preprocess = mocker.patch(
        "phentrieve.text_processing.full_text_service.preprocess_grounded_document",
        return_value=preprocessed,
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.get_llm_provider",
        return_value=provider,
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.TwoPhaseLLMPipeline",
        return_value=pipeline,
    )

    run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        language="en",
    )

    preprocess.assert_called_once()
    assert (
        pipeline.run.call_args.kwargs["grounded_chunks"] == preprocessed.grounded_chunks
    )
    assert (
        pipeline.run.call_args.kwargs["extraction_groups"]
        == preprocessed.extraction_groups
    )


def test_run_llm_backend_logs_group_preflight_details(mocker, caplog):
    provider = mocker.Mock()
    provider.count_tokens.return_value = {
        "prompt_tokens": 17,
        "completion_tokens": 0,
        "total_tokens": 17,
    }
    pipeline = mocker.Mock()
    preprocessed = SimpleNamespace(
        grounded_chunks=[
            {"chunk_id": 1, "text": "Chunk one.", "start_char": 0, "end_char": 10},
            {"chunk_id": 2, "text": "Chunk two.", "start_char": 11, "end_char": 21},
        ],
        extraction_groups=[
            {
                "group_id": 1,
                "chunk_ids": [1, 2],
                "text": "chunk_id=1: Chunk one.\nchunk_id=2: Chunk two.",
                "estimated_prompt_tokens": 17,
            }
        ],
    )
    pipeline.run.return_value = LLMExtractionResult(
        terms=[],
        meta=LLMMeta(
            llm_model="gemini-2.5-flash",
            llm_mode="two_phase",
        ),
    )

    mocker.patch(
        "phentrieve.text_processing.full_text_service.preprocess_grounded_document",
        return_value=preprocessed,
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.get_llm_provider",
        return_value=provider,
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.TwoPhaseLLMPipeline",
        return_value=pipeline,
    )

    with caplog.at_level("DEBUG"):
        run_llm_backend(
            text="Patient had recurrent seizures.",
            llm_model="gemini-2.5-flash",
            llm_mode="two_phase",
            llm_internal_mode="whole_document_grounded",
            language="en",
        )

    preflight_messages = [
        record.message
        for record in caplog.records
        if "LLM phase1 preflight" in record.message
    ]
    assert preflight_messages
    assert any("grounded_chunks=2" in message for message in preflight_messages)
    assert any("extraction_groups=1" in message for message in preflight_messages)
    assert any("prompt_tokens=17" in message for message in preflight_messages)
    assert any("total_tokens=17" in message for message in preflight_messages)


def test_adapt_standard_response_preserves_optional_term_fields():
    result = adapt_standard_response(
        [
            {
                "text": "chunk one",
                "status": "affirmed",
                "assertion_details": {"source": "pipeline"},
                "start_char": 0,
                "end_char": 9,
            }
        ],
        (
            [
                {
                    "id": "HP:0001250",
                    "name": "Seizure",
                    "confidence": 0.9,
                    "status": "affirmed",
                    "evidence_count": 1,
                    "chunks": [0],
                    "top_evidence_chunk_idx": 0,
                    "text_attributions": [
                        {
                            "chunk_idx": 0,
                            "start_char": 0,
                            "end_char": 7,
                            "matched_text_in_chunk": "seizure",
                        }
                    ],
                    "score": 0.95,
                    "reranker_score": 0.88,
                    "definition": "A seizure.",
                    "synonyms": ["convulsion"],
                }
            ],
            [
                {
                    "chunk_idx": 0,
                    "chunk_text": "chunk one",
                    "matches": [
                        {
                            "id": "HP:0001250",
                            "name": "Seizure",
                            "score": 0.95,
                            "assertion_status": "affirmed",
                        }
                    ],
                }
            ],
        ),
    )

    term = result["aggregated_hpo_terms"][0]
    assert term["definition"] == "A seizure."
    assert term["synonyms"] == ["convulsion"]
    assert term["reranker_score"] == 0.88
    assert term["source_chunk_ids"] == [1]
    assert term["top_evidence_chunk_id"] == 1
    assert term["text_attributions"][0]["chunk_id"] == 1


def test_full_text_service_normalizes_backend_and_rejects_unknown(mocker):
    standard_backend = mocker.Mock(return_value={"meta": {}})
    llm_backend = mocker.Mock(return_value={"meta": {}})
    service = FullTextService(
        standard_backend=standard_backend,
        llm_backend=llm_backend,
    )

    llm_result = service.process(text="clinical text", extraction_backend=" llm ")

    assert llm_result["meta"]["extraction_backend"] == "llm"
    llm_backend.assert_called_once()
    standard_backend.assert_not_called()

    with pytest.raises(ValueError, match="Unsupported extraction backend"):
        service.process(text="clinical text", extraction_backend="bogus")
