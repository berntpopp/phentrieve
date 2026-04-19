from types import SimpleNamespace

import pytest

from phentrieve.llm.types import (
    GroundedChunk,
    LLMExtractionResult,
    LLMMeta,
    LLMPhenotype,
)
from phentrieve.text_processing.full_text_service import (
    MAX_GROUNDED_PHASE1_INPUT_TOKENS,
    FullTextService,
    adapt_standard_response,
    preprocess_grounded_document,
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


def test_run_llm_backend_passes_provider_into_factory(mocker) -> None:
    factory = mocker.Mock()
    factory.return_value = SimpleNamespace(
        provider_name="ollama",
        model_name="qwen3.5:35b",
        base_url="http://localhost:11434",
    )
    pipeline = mocker.Mock()
    pipeline.run.return_value = LLMExtractionResult(
        terms=[],
        meta=LLMMeta(
            llm_provider="ollama",
            llm_model="qwen3.5:35b",
            llm_mode="two_phase",
        ),
    )

    run_llm_backend(
        text="Patient has seizures.",
        llm_provider="ollama",
        llm_model="qwen3.5:35b",
        llm_internal_mode="whole_document_legacy",
        provider_factory=factory,
        pipeline_factory=mocker.Mock(return_value=pipeline),
    )

    factory.assert_called_once_with(
        llm_model="qwen3.5:35b",
        llm_provider="ollama",
        llm_base_url=None,
    )
    assert pipeline.run.call_args.kwargs["config"].base_url == "http://localhost:11434"


def test_run_llm_backend_filters_factory_kwargs_and_uses_default_provider_name(
    mocker, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_factory(*, llm_model: str):
        captured["llm_model"] = llm_model
        return SimpleNamespace(
            model_name=llm_model,
            base_url="http://localhost:11434",
        )

    pipeline = mocker.Mock()
    pipeline.run.return_value = LLMExtractionResult(
        terms=[],
        meta=LLMMeta(
            llm_model="qwen3.5:35b",
            llm_mode="two_phase",
        ),
    )
    monkeypatch.setattr(
        "phentrieve.text_processing.full_text_service.DEFAULT_PROVIDER_NAME",
        "ollama",
        raising=False,
    )

    run_llm_backend(
        text="Patient has seizures.",
        llm_model="qwen3.5:35b",
        llm_internal_mode="whole_document_legacy",
        provider_factory=fake_factory,
        pipeline_factory=mocker.Mock(return_value=pipeline),
    )

    assert captured == {"llm_model": "qwen3.5:35b"}
    assert pipeline.run.call_args.kwargs["config"].provider == "ollama"
    assert pipeline.run.call_args.kwargs["config"].base_url == "http://localhost:11434"


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


def test_run_llm_backend_uses_default_llm_model(mocker, monkeypatch):
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    pipeline.run.return_value = LLMExtractionResult(
        terms=[],
        meta=LLMMeta(
            llm_model="gemini-3.1-flash-lite-preview",
            llm_mode="two_phase",
        ),
    )

    monkeypatch.delenv("PHENTRIEVE_LLM_MODEL", raising=False)
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
        llm_mode="two_phase",
        language="en",
    )

    assert (
        pipeline.run.call_args.kwargs["config"].model == "gemini-3.1-flash-lite-preview"
    )


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
    assert "extraction_groups" not in pipeline.run.call_args.kwargs


def test_run_llm_backend_does_not_retry_without_extraction_groups(mocker):
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    pipeline.run.side_effect = [
        TypeError("unexpected keyword argument 'extraction_groups'"),
        LLMExtractionResult(
            terms=[],
            meta=LLMMeta(
                llm_model="gemini-2.5-flash",
                llm_mode="two_phase",
            ),
        ),
    ]
    mocker.patch(
        "phentrieve.text_processing.full_text_service.get_llm_provider",
        return_value=provider,
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.TwoPhaseLLMPipeline",
        return_value=pipeline,
    )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        run_llm_backend(
            text="Patient had recurrent seizures.",
            llm_model="gemini-2.5-flash",
            llm_mode="two_phase",
            llm_internal_mode="whole_document_grounded",
            language="en",
        )

    assert pipeline.run.call_count == 1
    assert "extraction_groups" not in pipeline.run.call_args.kwargs


def test_run_llm_backend_logs_group_preflight_details(mocker, caplog):
    provider = mocker.Mock()
    provider.count_tokens.return_value = {
        "prompt_tokens": 99,
        "completion_tokens": 0,
        "total_tokens": 99,
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
                "chunk_ids": [1],
                "text": "chunk_id=1: Chunk one.",
                "estimated_prompt_tokens": 8,
            },
            {
                "group_id": 2,
                "chunk_ids": [2],
                "text": "chunk_id=2: Chunk two.",
                "estimated_prompt_tokens": 9,
            },
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
    assert any(
        "grouped_prompt_tokens_max=9" in message for message in preflight_messages
    )
    assert any(
        "grouped_prompt_tokens_total=17" in message for message in preflight_messages
    )
    assert any(
        "grouped_prompt_chars_total=44" in message for message in preflight_messages
    )
    assert any("grounded_chunks=2" in message for message in preflight_messages)
    assert any("extraction_groups=2" in message for message in preflight_messages)
    assert all("prompt_tokens=99" not in message for message in preflight_messages)


def test_preprocess_grounded_document_propagates_group_budget_value_error(mocker):
    provider = mocker.Mock()
    provider.count_tokens.return_value = {
        "prompt_tokens": MAX_GROUNDED_PHASE1_INPUT_TOKENS + 1,
        "completion_tokens": 0,
        "total_tokens": MAX_GROUNDED_PHASE1_INPUT_TOKENS + 1,
    }
    extraction_prompt = mocker.Mock()
    extraction_prompt.render_system_prompt.return_value = "system prompt"
    mocker.patch(
        "phentrieve.text_processing.full_text_service.build_grounded_chunks_from_text_pipeline",
        return_value=[
            GroundedChunk(
                chunk_id=1,
                text="Chunk one.",
                start_char=0,
                end_char=10,
                status="grounded",
            )
        ],
    )
    mocker.patch(
        "phentrieve.text_processing.full_text_service.build_extraction_groups",
        side_effect=ValueError(
            "Single grounded chunk exceeds max_prompt_tokens (chunk_id=1)"
        ),
    )

    with pytest.raises(ValueError, match="Single grounded chunk exceeds"):
        preprocess_grounded_document(
            text="clinical text",
            language="en",
            provider=provider,
            extraction_prompt=extraction_prompt,
            chunking_pipeline_config=None,
            assertion_config=None,
            retrieval_model_name="gemini-2.5-flash",
        )


def test_preprocess_grounded_document_skips_group_build_without_count_tokens(mocker):
    extraction_prompt = mocker.Mock()
    extraction_prompt.render_system_prompt.return_value = "system prompt"
    mocker.patch(
        "phentrieve.text_processing.full_text_service.build_grounded_chunks_from_text_pipeline",
        return_value=[
            GroundedChunk(
                chunk_id=1,
                text="Chunk one.",
                start_char=0,
                end_char=10,
                status="grounded",
            )
        ],
    )

    result = preprocess_grounded_document(
        text="clinical text",
        language="en",
        provider=SimpleNamespace(),
        extraction_prompt=extraction_prompt,
        chunking_pipeline_config=None,
        assertion_config=None,
        retrieval_model_name="gemini-2.5-flash",
    )

    assert result.extraction_groups == []


def test_preprocess_grounded_document_skips_group_build_when_whole_prompt_fits(mocker):
    provider = mocker.Mock()
    provider.count_tokens.return_value = {
        "prompt_tokens": 42,
        "completion_tokens": 0,
        "total_tokens": 42,
    }
    extraction_prompt = mocker.Mock()
    extraction_prompt.render_system_prompt.return_value = "system prompt"
    mocker.patch(
        "phentrieve.text_processing.full_text_service.build_grounded_chunks_from_text_pipeline",
        return_value=[
            GroundedChunk(
                chunk_id=1,
                text="Chunk one.",
                start_char=0,
                end_char=10,
                status="grounded",
            )
        ],
    )
    build_groups = mocker.patch(
        "phentrieve.text_processing.full_text_service.build_extraction_groups"
    )

    result = preprocess_grounded_document(
        text="clinical text",
        language="en",
        provider=provider,
        extraction_prompt=extraction_prompt,
        chunking_pipeline_config=None,
        assertion_config=None,
        retrieval_model_name="gemini-2.5-flash",
    )

    build_groups.assert_not_called()
    assert result.extraction_groups == []


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


def test_run_llm_backend_surfaces_grouped_observability(mocker):
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    preprocessed = SimpleNamespace(
        grounded_chunks=[
            {"chunk_id": 1, "text": "Chunk one.", "start_char": 0, "end_char": 10},
            {"chunk_id": 2, "text": "Chunk two.", "start_char": 11, "end_char": 21},
        ],
        extraction_groups=[
            {"group_id": 1, "chunk_ids": [1], "text": "Chunk one."},
            {"group_id": 2, "chunk_ids": [2], "text": "Chunk two."},
        ],
    )
    pipeline.run.return_value = LLMExtractionResult(
        terms=[],
        meta=LLMMeta(
            llm_model="gemini-2.5-flash",
            llm_mode="two_phase",
            request_count=3,
            phase_counts={
                "extracted_phrases": 1,
                "actionable_phrases": 2,
                "phase1_completed_groups": 1,
                "phase1_failed_groups": 1,
                "phase1_partial_failures": 1,
            },
            phase_request_counts={"phase1_requests": 2, "phase2b_llm_requests": 1},
            trace={
                "phase1": {
                    "extracted": [
                        {
                            "phrase": "seizures",
                            "category": "abnormal",
                            "chunk_ids": [1, 2],
                        }
                    ],
                    "groups": [
                        {"group_id": 1, "extracted_count": 1},
                        {"group_id": 2, "extracted_count": 1},
                    ],
                },
                "phase2b_llm": {
                    "resolved": [
                        {
                            "phrase": "seizures",
                            "selected_id": "HP:0001250",
                            "term_id": "HP:0001250",
                            "label": "Seizure",
                            "assertion": "present",
                            "category": "abnormal",
                            "match_method": "llm",
                            "local_fallback": False,
                            "chunk_ids": [1],
                        },
                        {
                            "phrase": "seizures",
                            "selected_id": "HP:0001250",
                            "term_id": "HP:0001250",
                            "label": "Seizure",
                            "assertion": "present",
                            "category": "abnormal",
                            "match_method": "llm",
                            "local_fallback": False,
                            "chunk_ids": [2],
                        },
                    ]
                },
            },
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

    result = run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
        language="en",
    )

    assert (
        pipeline.run.call_args.kwargs["extraction_groups"]
        == preprocessed.extraction_groups
    )
    assert result["meta"]["observability"] == {
        "request_count": 3,
        "extracted_phrases": 1,
        "actionable_phrases": 2,
        "phase2b_local_accept_count": 0,
        "phase2b_deferred_count": 0,
        "phase2b_no_candidate_skip_count": 0,
        "grounded_chunks": 2,
        "extraction_groups": 2,
        "failed_groups": 1,
        "deduplicated_phase1_mentions": 1,
        "deduplicated_unresolved_mappings": 1,
        "phase1_completed_groups": 1,
        "phase1_failed_groups": 1,
        "phase1_partial_failures": 1,
        "phase1_requests": 2,
        "phase2b_llm_requests": 1,
    }


def test_run_llm_backend_surfaces_phase2_routing_counts(mocker) -> None:
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    pipeline.run.return_value = LLMExtractionResult(
        terms=[],
        meta=LLMMeta(
            llm_model="gemini-2.5-flash",
            llm_mode="two_phase",
            phase_counts={
                "phase2b_local_accept_count": 3,
                "phase2b_deferred_count": 2,
                "phase2b_no_candidate_skip_count": 1,
            },
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
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
    )

    assert result["meta"]["observability"]["phase2b_local_accept_count"] == 3
    assert result["meta"]["observability"]["phase2b_deferred_count"] == 2
    assert result["meta"]["observability"]["phase2b_no_candidate_skip_count"] == 1
