# Grounded Whole-Note LLM CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mature the CLI/benchmark LLM full-text pipeline by keeping whole-note phase-1 reasoning while adding real chunk/span grounding, structured mapping, richer provenance, explicit failure handling, and logging.

**Architecture:** Add a shared grounding preprocessor ahead of the existing whole-note two-phase LLM pipeline. Phase 1 will emit anchored extractions tied to real chunks/spans, phase 2 will consume anchored context instead of reconstructed sentences, and the CLI/benchmark path will expose provenance through structured trace data and logs. Keep a legacy mode for regression comparison until grounded mode is validated.

**Tech Stack:** Python 3.10, Pydantic, Ruff, mypy, pytest, Typer CLI, existing Phentrieve text-processing pipeline, Gemini structured output support.

---

## File Map

### Core pipeline and types

- Modify: `phentrieve/llm/types.py`
  Add grounded extraction schema/types and richer internal phenotype provenance fields.
- Modify: `phentrieve/llm/pipeline.py`
  Add grounding preprocessing support, anchored phase-1 extraction, structured phase-2 mapping, richer dedup/provenance, explicit failure semantics, and logging.
- Modify: `phentrieve/llm/provider.py`
  Add token preflight support for Gemini if the SDK path is practical in this branch; otherwise add a provider hook for later use without blocking the rest of the work.

### Full-text service and CLI/benchmark wiring

- Modify: `phentrieve/text_processing/full_text_service.py`
  Wire grounded/legacy internal modes, pass chunking inputs through to the LLM backend, and surface richer metadata.
- Modify: `phentrieve/cli/text_commands.py`
  Add grounded-mode CLI plumbing, debug/provenance output behavior, and any internal flags needed for comparison.
- Modify: `phentrieve/benchmark/llm_benchmark.py`
  Record grounded provenance, explicit failures, and new trace fields in benchmark records.
- Modify: `phentrieve/benchmark/llm_cli.py`
  Expose benchmark-mode switches if needed and ensure artifacts persist new trace structure.

### Shared text-processing integration

- Modify: `phentrieve/text_processing/pipeline.py`
  Reuse existing position tracking cleanly from the LLM grounding preprocessor if helper extraction is needed.
- Modify: `phentrieve/text_processing/full_text_service.py`
  Reuse or factor standard-backend chunking setup so the LLM path can share it without copy/paste drift.

### Prompts

- Modify: `phentrieve/llm/prompts/templates/two_phase/en.yaml`
  Update phase-1 prompt to request anchors and grounded evidence while preserving source wording.
- Modify: `phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml`
  Convert single-item mapping prompt to match structured output and anchored local context.
- Modify: `phentrieve/llm/prompts/templates/two_phase/en_mapping_batch.yaml`
  Convert batch mapping prompt to match structured output and anchored local context.

### Tests

- Modify: `tests/unit/llm/test_pipeline.py`
  Replace `original_sentence` expectations with grounded context expectations, add failure/provenance/dedup coverage.
- Modify: `tests/unit/text_processing/test_full_text_service.py`
  Validate grounded metadata surfaced from the LLM backend.
- Modify: `tests/unit/cli/test_text_commands.py`
  Validate CLI wiring and debug/provenance options for the LLM backend.
- Modify: `tests/unit/test_llm_benchmark.py`
  Validate benchmark traces and explicit failure propagation.
- Add: `tests/integration/llm/test_grounded_pipeline_integration.py`
  Grounding integration tests for English and German.

## Task 1: Define Grounded Types And Legacy-Compatible Contracts

**Files:**
- Modify: `phentrieve/llm/types.py`
- Test: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write the failing type-contract tests**

```python
def test_grounded_extracted_phenotype_requires_chunk_ids() -> None:
    from pydantic import ValidationError
    from phentrieve.llm.types import LLMGroundedExtractedPhenotype

    with pytest.raises(ValidationError):
        LLMGroundedExtractedPhenotype(phrase="seizures", category="Abnormal")


def test_llm_phenotype_can_store_multiple_evidence_records() -> None:
    from phentrieve.llm.types import LLMPhenotype, LLMPhenotypeEvidence

    phenotype = LLMPhenotype(
        term_id="HP:0001250",
        label="Seizure",
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="recurrent seizures",
                evidence_text="recurrent seizures",
                chunk_ids=[2],
                start_char=14,
                end_char=32,
                match_method="local",
            )
        ],
    )

    assert phenotype.evidence_records[0].chunk_ids == [2]
```

- [ ] **Step 2: Run the unit tests to verify they fail**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k "grounded or evidence_records" -v`
Expected: FAIL because grounded extraction/evidence types do not exist yet.

- [ ] **Step 3: Add grounded extraction and evidence models in `phentrieve/llm/types.py`**

```python
class LLMGroundedExtractedPhenotype(BaseModel):
    phrase: str = Field(...)
    category: Literal["Abnormal", "Normal", "Suspected", "Family_History", "Other"]
    chunk_ids: list[int] = Field(min_length=1)
    evidence_text: str | None = None
    start_char: int | None = None
    end_char: int | None = None


class LLMGroundedExtractedPhenotypes(BaseModel):
    phenotypes: list[LLMGroundedExtractedPhenotype] = Field(default_factory=list)


class LLMPhenotypeEvidence(BaseModel):
    phrase: str
    evidence_text: str | None = None
    chunk_ids: list[int] = Field(default_factory=list)
    start_char: int | None = None
    end_char: int | None = None
    match_method: str = "unknown"


class LLMPhenotype(BaseModel):
    term_id: str
    label: str
    evidence: str | None = None
    assertion: str = AssertionStatus.PRESENT.value
    category: str | None = None
    evidence_records: list[LLMPhenotypeEvidence] = Field(default_factory=list)
```

- [ ] **Step 4: Update imports and call sites minimally to keep old code compiling**

```python
from phentrieve.llm.types import (
    LLMGroundedExtractedPhenotypes,
    LLMPhenotypeEvidence,
)
```

Use compatibility defaults so current tests outside the new work do not fail just from model shape changes.

- [ ] **Step 5: Run the focused tests to verify they pass**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k "grounded or evidence_records" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add phentrieve/llm/types.py tests/unit/llm/test_pipeline.py
git commit -m "feat: add grounded llm extraction types"
```

## Task 2: Add Grounding Preprocessor For The LLM Path

**Files:**
- Modify: `phentrieve/text_processing/full_text_service.py`
- Modify: `phentrieve/llm/pipeline.py`
- Test: `tests/unit/text_processing/test_full_text_service.py`
- Test: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write failing tests for grounded preprocessing**

```python
def test_run_llm_backend_builds_grounded_chunks_for_pipeline(mocker):
    provider = mocker.Mock()
    pipeline = mocker.Mock()
    pipeline.run.return_value = fake_llm_result()

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/text_processing/test_full_text_service.py -k grounded_chunks -v`
Expected: FAIL because `run_llm_backend()` does not build or pass grounded chunks.

- [ ] **Step 3: Factor chunk-building into a helper in `phentrieve/text_processing/full_text_service.py`**

```python
def _build_grounded_chunks(
    *,
    text: str,
    language: str,
    chunking_pipeline_config: list[dict[str, Any]] | None,
    assertion_config: dict[str, Any] | None,
    retrieval_model_name: str,
    include_positions: bool = True,
) -> list[dict[str, Any]]:
    text_pipeline = TextProcessingPipeline(
        language=language,
        chunking_pipeline_config=chunking_pipeline_config
        or get_default_chunk_pipeline_config(),
        assertion_config=assertion_config or {"disable": True},
        sbert_model_for_semantic_chunking=load_embedding_model(retrieval_model_name),
    )
    processed_chunks = text_pipeline.process(text, include_positions=include_positions)
    return [
        {
            "chunk_id": idx + 1,
            "text": chunk.get("text", ""),
            "start_char": chunk.get("start_char"),
            "end_char": chunk.get("end_char"),
            "status": _normalize_status(chunk.get("status")),
        }
        for idx, chunk in enumerate(processed_chunks)
    ]
```

- [ ] **Step 4: Pass grounded chunks into `TwoPhaseLLMPipeline.run()`**

```python
grounded_chunks = _build_grounded_chunks(
    text=text,
    language=kwargs.get("language") or DEFAULT_LANGUAGE,
    chunking_pipeline_config=kwargs.get("chunking_pipeline_config"),
    assertion_config={"disable": True},
    retrieval_model_name=kwargs.get("retrieval_model_name", DEFAULT_MODEL),
)

result = pipeline.run(
    text=text,
    grounded_chunks=grounded_chunks,
    config=LLMPipelineConfig(...),
)
```

- [ ] **Step 5: Update `TwoPhaseLLMPipeline.run()` signature without changing behavior yet**

```python
def run(
    self,
    *,
    text: str,
    grounded_chunks: list[dict[str, Any]] | None = None,
    config: LLMPipelineConfig,
) -> LLMExtractionResult:
    grounded_chunks = list(grounded_chunks or [])
```

- [ ] **Step 6: Run the focused tests**

Run: `uv run pytest tests/unit/text_processing/test_full_text_service.py -k grounded_chunks -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add phentrieve/text_processing/full_text_service.py phentrieve/llm/pipeline.py tests/unit/text_processing/test_full_text_service.py
git commit -m "feat: build grounded chunks for llm backend"
```

## Task 3: Convert Phase 1 To Anchored Structured Extraction

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `phentrieve/llm/prompts/templates/two_phase/en.yaml`
- Test: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write failing tests for anchored phase-1 extraction**

```python
def test_phase1_returns_chunk_ids_and_evidence_text():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "phenotypes": [
                        {
                            "phrase": "recurrent seizures",
                            "category": "Abnormal",
                            "chunk_ids": [1],
                            "evidence_text": "recurrent seizures",
                        }
                    ]
                }
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=FakeToolExecutor([]))

    result = pipeline._extract_phase1_phenotypes(
        text="Patient had recurrent seizures.",
        grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
        extraction_prompt=get_prompt(AnnotationMode.TWO_PHASE, "en"),
    )

    assert result[0][0]["chunk_ids"] == [1]
```

- [ ] **Step 2: Run the targeted tests**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k "chunk_ids and evidence_text" -v`
Expected: FAIL because phase 1 still returns tuple pairs only.

- [ ] **Step 3: Change `_extract_phase1_phenotypes()` to use grounded schema**

```python
response = self.provider.run_structured_prompt(
    system_prompt=extraction_prompt.render_system_prompt(),
    user_prompt=extraction_prompt.render_user_prompt(
        text,
        chunk_index=self._render_chunk_index(grounded_chunks),
    ),
    response_model=LLMGroundedExtractedPhenotypes,
    max_output_tokens=DEFAULT_PHASE1_MAX_OUTPUT_TOKENS,
)

parsed = []
for phenotype in response.phenotypes:
    parsed.append(
        {
            "phrase": phenotype.phrase.strip(),
            "category": _normalize_category(phenotype.category),
            "chunk_ids": list(phenotype.chunk_ids),
            "evidence_text": phenotype.evidence_text,
            "start_char": phenotype.start_char,
            "end_char": phenotype.end_char,
        }
    )
```

- [ ] **Step 4: Update phase-1 trace payloads and actionable filtering**

```python
actionable = [
    item
    for item in extracted
    if _normalize_category(item["category"]) in ACTIONABLE_CATEGORIES
]
```

- [ ] **Step 5: Update `en.yaml` prompt contract**

```yaml
version: "v3.0.0"
system_prompt: |
  You are an expert clinical geneticist. Extract source-faithful phenotype phrases
  from the full note. Each phenotype must reference one or more chunk_ids from the
  provided chunk index. Preserve source wording.
  ...
  Output JSON only:
  {
    "phenotypes": [
      {
        "phrase": "recurrent seizures",
        "category": "Abnormal",
        "chunk_ids": [3],
        "evidence_text": "recurrent seizures"
      }
    ]
  }
user_prompt_template: |
  Extract all phenotype phrases from the following clinical text.
  Chunk index:
  {chunk_index}

  ---
  {text}
  ---
```

- [ ] **Step 6: Run the phase-1 tests**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k "phase1 or chunk_ids or extracted" -v`
Expected: PASS after updating old expectations to the new trace structure.

- [ ] **Step 7: Commit**

```bash
git add phentrieve/llm/pipeline.py phentrieve/llm/prompts/templates/two_phase/en.yaml tests/unit/llm/test_pipeline.py
git commit -m "feat: add anchored phase1 extraction"
```

## Task 4: Remove Fake Sentence Reconstruction And Use Anchored Context

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Test: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write failing tests that no longer depend on `original_sentence`**

```python
def test_retrieval_uses_anchored_chunk_context_instead_of_original_sentence():
    item = {
        "phrase": "frequent falls",
        "category": "abnormal",
        "chunk_ids": [2],
        "evidence_text": "frequent falls",
    }
    grounded_chunks = [
        {"chunk_id": 1, "text": "The child walks independently."},
        {"chunk_id": 2, "text": "The child has frequent falls while walking."},
    ]
    pipeline = TwoPhaseLLMPipeline(provider=FakeProvider([]), tool_executor=FakeToolExecutor([]))

    context = pipeline._build_grounded_context(item=item, grounded_chunks=grounded_chunks)

    assert context["primary_chunk_text"] == "The child has frequent falls while walking."
    assert "original_sentence" not in context
```

- [ ] **Step 2: Run the focused test**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k grounded_context -v`
Expected: FAIL because helper/context shape does not exist.

- [ ] **Step 3: Add `_build_grounded_context()` and remove `_find_original_sentence()`**

```python
def _build_grounded_context(
    self,
    *,
    item: dict[str, Any],
    grounded_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in grounded_chunks}
    chunk_ids = [int(chunk_id) for chunk_id in item.get("chunk_ids", [])]
    primary = chunk_lookup.get(chunk_ids[0]) if chunk_ids else None
    neighbors = [
        chunk_lookup[cid]
        for cid in (chunk_ids[:1] + [chunk_ids[0] - 1, chunk_ids[0] + 1] if chunk_ids else [])
        if cid in chunk_lookup
    ]
    return {
        "chunk_ids": chunk_ids,
        "primary_chunk_text": primary.get("text", "") if primary else item.get("evidence_text", ""),
        "neighbor_chunk_texts": [chunk.get("text", "") for chunk in neighbors if chunk is not primary],
    }
```

- [ ] **Step 4: Thread grounded context into retrieval/mapping records**

```python
results.append(
    {
        "phrase": item["phrase"],
        "category": item["category"],
        "grounded_context": self._build_grounded_context(
            item=item,
            grounded_chunks=grounded_chunks,
        ),
        "candidates": ...,
    }
)
```

- [ ] **Step 5: Update tests that assert on `original_sentence`**

Replace assertions like:

```python
assert "original_sentence" in provider.calls[0][-1]["content"]
```

with:

```python
assert "primary_chunk_text" in provider.calls[0][-1]["content"]
```

- [ ] **Step 6: Run the pipeline unit tests**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add phentrieve/llm/pipeline.py tests/unit/llm/test_pipeline.py
git commit -m "refactor: replace sentence reconstruction with grounded context"
```

## Task 5: Move Phase 2 Mapping To Structured Output

**Files:**
- Modify: `phentrieve/llm/types.py`
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml`
- Modify: `phentrieve/llm/prompts/templates/two_phase/en_mapping_batch.yaml`
- Test: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Add failing tests for structured phase-2 mapping**

```python
def test_phase2_mapping_uses_structured_prompt():
    provider = FakeProvider(
        responses=[
            {
                "parsed": {
                    "mappings": [
                        {"phrase": "frequent falls", "hpo_id": "HP:0002355"}
                    ]
                }
            }
        ]
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=FakeToolExecutor([]))

    pipeline._run_mapping_batch(
        batch=[{
            "phrase": "frequent falls",
            "category": "abnormal",
            "grounded_context": {"primary_chunk_text": "The child has frequent falls while walking."},
            "candidates": [{"hpo_id": "HP:0002355", "term_name": "Difficulty walking"}],
        }],
        mapping_prompt=get_mapping_prompt("en"),
    )

    assert provider.structured_calls
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k structured_prompt -v`
Expected: FAIL because mapping still uses `provider.complete()`.

- [ ] **Step 3: Add mapping response models**

```python
class LLMMappingSelection(BaseModel):
    phrase: str
    hpo_id: str | None = None


class LLMBatchMappingSelections(BaseModel):
    mappings: list[LLMMappingSelection] = Field(default_factory=list)
```

- [ ] **Step 4: Replace `_run_mapping_batch()` to use `run_structured_prompt()`**

```python
response_model = (
    LLMBatchMappingSelections if len(batch) > 1 else LLMMappingSelection
)
response = self.provider.run_structured_prompt(
    system_prompt=mapping_prompt.render_system_prompt(),
    user_prompt=mapping_prompt.render_user_prompt(candidate_payload),
    response_model=response_model,
)
```

- [ ] **Step 5: Update mapping prompt templates to request structured anchored context**

```yaml
version: "v4.0.0"
system_prompt: |
  You map clinical phenotype phrases to HPO terms.
  You receive:
  - phrase
  - category
  - primary_chunk_text
  - neighbor_chunk_texts
  - candidates
  Output JSON only...
```

- [ ] **Step 6: Run the mapping tests**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k "mapping or structured_prompt" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add phentrieve/llm/types.py phentrieve/llm/pipeline.py phentrieve/llm/prompts/templates/two_phase/en_mapping.yaml phentrieve/llm/prompts/templates/two_phase/en_mapping_batch.yaml tests/unit/llm/test_pipeline.py
git commit -m "feat: use structured phase2 mapping with anchored context"
```

## Task 6: Preserve Rich Evidence, Assertions, And Dedup Semantics

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `phentrieve/text_processing/full_text_service.py`
- Test: `tests/unit/llm/test_pipeline.py`
- Test: `tests/unit/text_processing/test_full_text_service.py`

- [ ] **Step 1: Write failing dedup/evidence tests**

```python
def test_deduplicate_terms_keeps_assertion_variants():
    terms = [
        LLMPhenotype(term_id="HP:0001250", label="Seizure", assertion="present", category="abnormal"),
        LLMPhenotype(term_id="HP:0001250", label="Seizure", assertion="negated", category="normal"),
    ]

    deduped = TwoPhaseLLMPipeline._deduplicate_terms(terms)

    assert len(deduped) == 2
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k deduplicate_terms_keeps_assertion_variants -v`
Expected: FAIL because dedup uses only `term_id`.

- [ ] **Step 3: Change dedup key and preserve evidence records**

```python
@staticmethod
def _deduplicate_terms(terms: list[LLMPhenotype]) -> list[LLMPhenotype]:
    deduplicated: dict[tuple[str, str], LLMPhenotype] = {}
    for term in terms:
        key = (term.term_id, term.assertion)
        if key not in deduplicated:
            deduplicated[key] = term
            continue
        deduplicated[key].evidence_records.extend(term.evidence_records)
        if not deduplicated[key].evidence and term.evidence:
            deduplicated[key].evidence = term.evidence
    return list(deduplicated.values())
```

- [ ] **Step 4: Surface evidence records in service metadata/debug shape**

```python
"aggregated_hpo_terms": [
    {
        "id": term.term_id,
        "name": term.label,
        "evidence": term.evidence,
        "status": term.assertion,
        "evidence_records": [record.model_dump() for record in term.evidence_records],
    }
    for term in result.terms
]
```

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k "deduplicate or evidence_records" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add phentrieve/llm/pipeline.py phentrieve/text_processing/full_text_service.py tests/unit/llm/test_pipeline.py tests/unit/text_processing/test_full_text_service.py
git commit -m "feat: preserve llm evidence and assertion variants"
```

## Task 7: Add Explicit Failure Semantics And Logging

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Test: `tests/unit/llm/test_pipeline.py`
- Test: `tests/unit/test_llm_benchmark.py`

- [ ] **Step 1: Add failing tests for explicit phase-1 failure reporting**

```python
def test_phase1_failure_is_recorded_in_trace_not_silenced(caplog):
    provider = FailingStructuredProvider()
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=FakeToolExecutor([]))

    with pytest.raises(RuntimeError):
        pipeline.run(
            text="Patient had recurrent seizures.",
            grounded_chunks=[{"chunk_id": 1, "text": "Patient had recurrent seizures."}],
            config=LLMPipelineConfig(model="gemini-2.5-flash", mode="two_phase"),
        )
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k phase1_failure -v`
Expected: FAIL because phase 1 currently returns empty results.

- [ ] **Step 3: Add a typed pipeline exception and explicit trace/log fields**

```python
class LLMPipelinePhaseError(RuntimeError):
    def __init__(self, phase: str, message: str) -> None:
        super().__init__(message)
        self.phase = phase
```

```python
except Exception as exc:
    logger.exception("Phase 1 structured extraction failed")
    raise LLMPipelinePhaseError("phase1", "Structured extraction failed") from exc
```

- [ ] **Step 4: Catch and record document-level failures in benchmark mode**

```python
try:
    pipeline_result = pipeline.run(text=document["text"], grounded_chunks=..., config=config)
except LLMPipelinePhaseError as exc:
    result_record = {
        "case_index": index,
        "doc_id": document["id"],
        "status": "failed",
        "error_phase": exc.phase,
        "error_message": str(exc),
    }
    results.append(result_record)
    continue
```

- [ ] **Step 5: Add logging for chunking summary and anchor stats**

```python
logger.info(
    "LLM grounding summary: language=%s chunks=%d mode=%s",
    language,
    len(grounded_chunks),
    internal_mode,
)
logger.info(
    "Phase 1 anchor resolution: extracted=%d anchored=%d",
    len(extracted),
    sum(1 for item in extracted if item.get("chunk_ids")),
)
```

- [ ] **Step 6: Run the failure/logging tests**

Run: `uv run pytest tests/unit/llm/test_pipeline.py tests/unit/test_llm_benchmark.py -k "phase1_failure or failed or anchor" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add phentrieve/llm/pipeline.py phentrieve/benchmark/llm_benchmark.py tests/unit/llm/test_pipeline.py tests/unit/test_llm_benchmark.py
git commit -m "feat: surface llm pipeline failures and provenance logs"
```

## Task 8: Add Token Preflight And Internal Grounded/Legacy Mode Switch

**Files:**
- Modify: `phentrieve/llm/provider.py`
- Modify: `phentrieve/text_processing/full_text_service.py`
- Modify: `phentrieve/cli/text_commands.py`
- Modify: `phentrieve/benchmark/llm_cli.py`
- Test: `tests/unit/text_processing/test_full_text_service.py`
- Test: `tests/unit/cli/test_text_commands.py`

- [ ] **Step 1: Write failing tests for internal mode selection and token preflight**

```python
def test_run_llm_backend_supports_grounded_internal_mode(mocker):
    ...
    run_llm_backend(
        text="Patient had recurrent seizures.",
        llm_model="gemini-2.5-flash",
        llm_mode="two_phase",
        llm_internal_mode="whole_document_grounded",
    )
    assert pipeline.run.call_args.kwargs["grounded_chunks"]
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/unit/text_processing/test_full_text_service.py tests/unit/cli/test_text_commands.py -k "internal_mode or token_preflight" -v`
Expected: FAIL because internal mode and preflight do not exist.

- [ ] **Step 3: Add provider hook for token counting**

```python
class LLMProvider(ABC):
    ...
    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        raise NotImplementedError
```

Add a Gemini implementation if feasible with the current SDK path; otherwise return a clear unsupported error and wire the CLI mode switch first.

- [ ] **Step 4: Add internal mode switch in `run_llm_backend()`**

```python
llm_internal_mode = kwargs.get("llm_internal_mode", "whole_document_grounded")
if llm_internal_mode not in {"whole_document_legacy", "whole_document_grounded"}:
    raise ValueError(...)
```

- [ ] **Step 5: Add optional CLI/benchmark flags for internal comparison**

```python
llm_internal_mode: Annotated[
    Literal["whole_document_legacy", "whole_document_grounded"],
    typer.Option("--llm-internal-mode", help="Internal grounding mode for benchmarking"),
] = "whole_document_grounded"
```

- [ ] **Step 6: Add oversize warning path**

```python
if token_counts["total_tokens"] > MAX_GROUNDED_PHASE1_INPUT_TOKENS:
    logger.warning("LLM phase1 prompt exceeds configured token budget")
```

- [ ] **Step 7: Run the CLI/service tests**

Run: `uv run pytest tests/unit/text_processing/test_full_text_service.py tests/unit/cli/test_text_commands.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add phentrieve/llm/provider.py phentrieve/text_processing/full_text_service.py phentrieve/cli/text_commands.py phentrieve/benchmark/llm_cli.py tests/unit/text_processing/test_full_text_service.py tests/unit/cli/test_text_commands.py
git commit -m "feat: add grounded llm mode switch and token preflight"
```

## Task 9: Add Integration Coverage For Grounded English And German Paths

**Files:**
- Add: `tests/integration/llm/test_grounded_pipeline_integration.py`
- Modify: `tests/integration/test_benchmark_workflow.py`

- [ ] **Step 1: Write integration tests for grounded chunk provenance**

```python
@pytest.mark.integration
def test_grounded_llm_pipeline_preserves_english_chunk_provenance():
    ...
    assert result.meta.trace["phase1"]["extracted"][0]["chunk_ids"]


@pytest.mark.integration
def test_grounded_llm_pipeline_preserves_german_chunk_provenance():
    ...
    assert any(
        item["chunk_ids"]
        for item in result.meta.trace["phase1"]["extracted"]
    )
```

- [ ] **Step 2: Run the new tests and verify failures**

Run: `uv run pytest tests/integration/llm/test_grounded_pipeline_integration.py -n 0 -v`
Expected: FAIL until grounding/integration wiring is complete.

- [ ] **Step 3: Add a benchmark workflow assertion for new trace fields**

```python
assert prediction_record["trace"]["phase1"]["extracted"][0]["chunk_ids"] == [1]
```

- [ ] **Step 4: Run the integration tests**

Run: `uv run pytest tests/integration/llm/test_grounded_pipeline_integration.py tests/integration/test_benchmark_workflow.py -n 0 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/integration/llm/test_grounded_pipeline_integration.py tests/integration/test_benchmark_workflow.py
git commit -m "test: add grounded llm integration coverage"
```

## Task 10: Verify End-To-End CLI And Benchmark Behavior

**Files:**
- Modify if needed: `phentrieve/cli/text_commands.py`
- Modify if needed: `phentrieve/benchmark/llm_benchmark.py`
- Test: targeted verification only

- [ ] **Step 1: Run focused unit test suites**

Run: `uv run pytest tests/unit/llm/test_pipeline.py tests/unit/text_processing/test_full_text_service.py tests/unit/cli/test_text_commands.py tests/unit/test_llm_benchmark.py -v`
Expected: PASS

- [ ] **Step 2: Run the new integration suite single-threaded**

Run: `uv run pytest tests/integration/llm/test_grounded_pipeline_integration.py tests/integration/test_benchmark_workflow.py -n 0 -v`
Expected: PASS

- [ ] **Step 3: Run repository-required verification**

Run: `make check`
Expected: PASS

Run: `make typecheck-fast`
Expected: PASS

Run: `make test`
Expected: PASS

- [ ] **Step 4: Run one representative CLI smoke check locally**

Run: `uv run phentrieve text process --text "Patient had recurrent seizures and no skeletal anomalies." --extraction-backend llm --llm-model gemini-2.5-flash --llm-mode two_phase --llm-internal-mode whole_document_grounded --debug`
Expected: CLI output includes LLM results and logs show chunking summary plus grounded provenance.

- [ ] **Step 5: Run one representative benchmark smoke check**

Run: `uv run phentrieve benchmark llm --test-file tests/data/en/phenobert/tiny_extraction_test.json --llm-model gemini-2.5-flash --llm-mode two_phase --debug`
Expected: output JSON contains trace entries with chunk IDs or explicit failure metadata.

- [ ] **Step 6: Commit any final verification-driven fixes**

```bash
git add phentrieve/ tests/
git commit -m "chore: finalize grounded llm cli verification fixes"
```

## Self-Review

### Spec coverage

- Whole-note phase-1 reasoning preserved: Tasks 2, 3, 4
- Shared grounding preprocessor: Task 2
- Anchored phase-1 schema: Task 3
- Removal of fake sentence reconstruction: Task 4
- Structured phase-2 mapping: Task 5
- Richer evidence/assertion preservation: Task 6
- Logging and provenance: Task 7
- Internal grounded/legacy mode and token preflight: Task 8
- English/German validation and benchmark traces: Task 9
- Required repo verification: Task 10

### Placeholder scan

- No `TODO`, `TBD`, or deferred “write tests later” steps remain.
- Every task names exact files, commands, and expected outcomes.
- Later tasks use names introduced in earlier tasks: `LLMGroundedExtractedPhenotype`, `LLMPhenotypeEvidence`, `LLMPipelinePhaseError`, `whole_document_grounded`.

### Type consistency

- Grounded extraction uses `chunk_ids` consistently across tasks.
- Provenance uses `evidence_records` consistently.
- Internal mode names are consistent: `whole_document_legacy`, `whole_document_grounded`.
- Failure type is consistent: `LLMPipelinePhaseError`.

## Execution Handoff

Plan complete and saved to `.planning/active/2026-04-16-llm-cli-grounded-whole-note-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
