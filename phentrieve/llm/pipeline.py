from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

from phentrieve.llm.config import (
    DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_LANGUAGE,
    DEFAULT_LLM_MAPPING_BATCH_SIZE,
    DEFAULT_LOCAL_MATCH_THRESHOLD,
    DEFAULT_MAX_UNIQUE_CANDIDATES,
    DEFAULT_MIN_UNIQUE_CANDIDATES,
    DEFAULT_PHASE1_FALLBACK_ENABLED,
    DEFAULT_PHASE1_LARGE_GROUP_MAX_PROMPT_TOKENS,
    DEFAULT_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_PHASE1_SMALL_GROUP_MAX_CHUNKS,
    DEFAULT_QUERY_RESULTS_PER_PHRASE,
    DEFAULT_SIMILARITY_THRESHOLD,
    PRESENT_ASSERTION,
)
from phentrieve.llm.pipeline_phase1 import (
    ACTIONABLE_CATEGORIES,
)
from phentrieve.llm.pipeline_phase1 import (
    clean_text as _clean_text,
)
from phentrieve.llm.pipeline_phase1 import (
    expand_combined_phase1_extractions as _expand_combined_phase1_extractions,
)
from phentrieve.llm.pipeline_phase1 import (
    merge_optional_bounds as _merge_optional_bounds,
)
from phentrieve.llm.pipeline_phase1 import (
    normalize_category as _normalize_category,
)
from phentrieve.llm.pipeline_phase1 import (
    normalized_evidence_text as _normalized_evidence_text,
)
from phentrieve.llm.pipeline_phase1 import (
    phase1_extraction_dedup_key as _phase1_extraction_dedup_key,
)
from phentrieve.llm.pipeline_phase1 import (
    prefer_richer_text as _prefer_richer_text,
)
from phentrieve.llm.pipeline_phase1 import (
    render_group_chunk_index_text as _render_group_chunk_index_text,
)
from phentrieve.llm.pipeline_phase1 import (
    render_phase1_user_prompt as _render_phase1_user_prompt,
)
from phentrieve.llm.pipeline_phase1 import (
    sorted_chunk_ids as _sorted_chunk_ids,
)
from phentrieve.llm.pipeline_phase1 import (
    spans_overlap as _spans_overlap,
)
from phentrieve.llm.pipeline_phase1 import (
    tokenize as _tokenize,
)
from phentrieve.llm.pipeline_phase2 import (
    CATEGORY_TO_ASSERTION,
    build_grounded_context,
    deduplicate_terms,
    extract_first_result_list,
    hybrid_select_candidates,
    local_match,
    optional_float,
    phenotype_from_candidate,
    prepare_retrieval_queries,
    run_mapping_batch,
    select_candidate_id,
    select_candidate_ids,
)
from phentrieve.llm.pipeline_phase2 import (
    candidate_match_text as _candidate_match_text,
)
from phentrieve.llm.pipeline_phase2 import (
    downstream_dedupe_key as _downstream_dedupe_key,
)
from phentrieve.llm.pipeline_phase2 import (
    mapping_batch_item_id as _mapping_batch_item_id,
)
from phentrieve.llm.pipeline_retry import (
    LLMPipelinePhaseError,
)
from phentrieve.llm.pipeline_retry import (
    classify_phase1_failure as _classify_phase1_failure,
)
from phentrieve.llm.pipeline_retry import (
    group_source_chunk_ids as _group_source_chunk_ids,
)
from phentrieve.llm.pipeline_retry import (
    next_phase1_mode as _next_phase1_mode,
)
from phentrieve.llm.pipeline_trace import (
    annotate_mapping_trace_with_provenance,
    attach_phase1_failure_context,
    build_phase1_trace,
    build_phase2a_trace,
    build_phase2b_local_trace,
    phase1_extracted_trace_items,
    phase1_failure_class_from_groups,
)
from phentrieve.llm.pipeline_trace import (
    build_phase1_attempt_trace as _build_phase1_attempt_trace,
)
from phentrieve.llm.pipeline_trace import (
    sum_usage_dicts as _sum_usage_dicts,
)
from phentrieve.llm.preprocessing import build_extraction_groups
from phentrieve.llm.prompts.loader import (
    PromptTemplate,
    get_batch_mapping_prompt,
    get_mapping_prompt,
    get_prompt,
)
from phentrieve.llm.provider import ToolExecutor
from phentrieve.llm.types import (
    AnnotationMode,
    GroundedChunk,
    LLMBatchMappingSelections,
    LLMExtractedPhenotypes,
    LLMExtractionResult,
    LLMGroundedExtractedPhenotypes,
    LLMMappingSelection,
    LLMMeta,
    LLMPhenotype,
    LLMPipelineConfig,
    Phase1FailureClass,
    Phase1Mode,
)
from phentrieve.llm.utils import token_sort_similarity

logger = logging.getLogger(__name__)


class TwoPhaseLLMPipeline:
    def __init__(
        self,
        provider,
        *,
        tool_executor: ToolExecutor | None = None,
        n_results_per_phrase: int = DEFAULT_QUERY_RESULTS_PER_PHRASE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_unique_candidates: int = DEFAULT_MIN_UNIQUE_CANDIDATES,
        max_unique_candidates: int = DEFAULT_MAX_UNIQUE_CANDIDATES,
        local_match_threshold: float = DEFAULT_LOCAL_MATCH_THRESHOLD,
        mapping_batch_size: int = DEFAULT_LLM_MAPPING_BATCH_SIZE,
    ) -> None:
        self.provider = provider
        self.tool_executor = tool_executor or ToolExecutor()
        self.n_results_per_phrase = n_results_per_phrase
        self.similarity_threshold = similarity_threshold
        self.min_unique_candidates = min_unique_candidates
        self.max_unique_candidates = max_unique_candidates
        self.local_match_threshold = local_match_threshold
        self.mapping_batch_size = max(int(mapping_batch_size), 1)

    def run(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]] | None = None,
        extraction_groups: list[dict[str, Any]] | None = None,
        config: LLMPipelineConfig,
    ) -> LLMExtractionResult:
        grounded_chunks = list(grounded_chunks or [])
        extraction_groups = list(extraction_groups or [])
        mode = AnnotationMode(config.mode)
        if mode != AnnotationMode.TWO_PHASE:
            raise ValueError(
                f"Unsupported LLM mode: {config.mode!r}. Expected 'two_phase'."
            )

        language = config.language or DEFAULT_LLM_LANGUAGE
        extraction_prompt = get_prompt(mode, language)
        logger.debug(
            "Phase 1: extracting phenotype phrases (mode=%s, language=%s, text_chars=%d)",
            mode.value,
            language,
            len(text),
        )
        phase1_initial_mode = self._resolve_initial_phase1_mode(
            extraction_groups=extraction_groups
        )
        phase1_fallback_count = 0
        phase1_failure_class: Phase1FailureClass = None
        phase1_attempts_trace: list[dict[str, Any]] = []
        phase1_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        phase1_request_count = 0
        phase1_elapsed = 0.0
        phase1_carried_groups_trace: list[dict[str, Any]] = []
        phase1_carried_extracted: list[dict[str, Any]] = []
        extracted: list[dict[str, Any]] = []
        current_phase1_mode = phase1_initial_mode
        current_grounded_chunks = list(grounded_chunks)
        current_extraction_groups = list(extraction_groups)

        while True:
            try:
                (
                    extracted,
                    attempt_phase1_usage,
                    attempt_phase1_request_count,
                    attempt_phase1_elapsed,
                    attempt_phase1_groups_trace,
                ) = self._run_phase1_mode(
                    mode=current_phase1_mode,
                    text=text,
                    grounded_chunks=current_grounded_chunks,
                    extraction_groups=current_extraction_groups,
                    extraction_prompt=extraction_prompt,
                    capture_phase1_debug=config.capture_phase1_debug,
                )
                phase1_usage = _sum_usage_dicts(phase1_usage, attempt_phase1_usage)
                phase1_request_count += attempt_phase1_request_count
                phase1_elapsed = round(phase1_elapsed + attempt_phase1_elapsed, 6)
                phase1_final_mode = current_phase1_mode
                (
                    retry_grounded_chunks,
                    retry_failure_class,
                    retained_groups_trace,
                    terminal_failure_class,
                ) = self._phase1_partial_retry_chunks(
                    mode=current_phase1_mode,
                    grounded_chunks=current_grounded_chunks,
                    groups_trace=attempt_phase1_groups_trace,
                )
                if terminal_failure_class is not None:
                    terminal_exc = LLMPipelinePhaseError(
                        "phase1",
                        "Structured extraction failed for one or more extraction groups",
                        usage=attempt_phase1_usage,
                        request_count=attempt_phase1_request_count,
                        elapsed_seconds=attempt_phase1_elapsed,
                        groups_trace=attempt_phase1_groups_trace,
                    )
                    terminal_exc.failure_class = terminal_failure_class
                    raise terminal_exc
                if retry_grounded_chunks:
                    if phase1_failure_class is None:
                        phase1_failure_class = retry_failure_class
                    phase1_carried_extracted.extend(extracted)
                    phase1_carried_groups_trace.extend(retained_groups_trace)
                    phase1_attempts_trace.append(
                        _build_phase1_attempt_trace(
                            mode=current_phase1_mode,
                            status="partial",
                            groups_trace=attempt_phase1_groups_trace,
                            request_count=attempt_phase1_request_count,
                            elapsed_seconds=attempt_phase1_elapsed,
                            failure_class=retry_failure_class,
                        )
                    )
                    phase1_fallback_count += 1
                    current_phase1_mode = "grouped_small"
                    current_grounded_chunks = retry_grounded_chunks
                    current_extraction_groups = []
                    continue
                extracted = self._deduplicate_phase1_extractions(
                    phase1_carried_extracted + extracted
                )
                phase1_groups_trace = (
                    list(phase1_carried_groups_trace) + attempt_phase1_groups_trace
                )
                phase1_attempts_trace.append(
                    _build_phase1_attempt_trace(
                        mode=current_phase1_mode,
                        status="completed",
                        groups_trace=attempt_phase1_groups_trace,
                        request_count=attempt_phase1_request_count,
                        elapsed_seconds=attempt_phase1_elapsed,
                    )
                )
                break
            except LLMPipelinePhaseError as exc:
                failure_class = _classify_phase1_failure(exc)
                phase1_failure_class = failure_class
                phase1_usage = _sum_usage_dicts(phase1_usage, exc.usage)
                phase1_request_count += exc.request_count
                phase1_elapsed = round(
                    phase1_elapsed + float(exc.elapsed_seconds or 0.0), 6
                )
                phase1_groups_trace = list(phase1_carried_groups_trace) + list(
                    exc.groups_trace
                )
                phase1_attempts_trace.append(
                    _build_phase1_attempt_trace(
                        mode=current_phase1_mode,
                        status="failed",
                        groups_trace=exc.groups_trace,
                        request_count=exc.request_count,
                        elapsed_seconds=float(exc.elapsed_seconds or 0.0),
                        failure_class=failure_class,
                    )
                )
                next_mode = _next_phase1_mode(current_phase1_mode)
                if self._should_fallback_phase1(
                    failure_class=failure_class,
                    grounded_chunks=current_grounded_chunks,
                    next_mode=next_mode,
                ):
                    assert next_mode is not None
                    phase1_fallback_count += 1
                    current_phase1_mode = next_mode
                    continue
                self._attach_phase1_failure_context(
                    exc=exc,
                    initial_mode=phase1_initial_mode,
                    final_mode=current_phase1_mode,
                    fallback_count=phase1_fallback_count,
                    failure_class=failure_class,
                    final_groups_trace=phase1_groups_trace,
                    attempts_trace=phase1_attempts_trace,
                )
                raise
        if phase1_failure_class is None:
            phase1_failure_class = self._phase1_failure_class_from_groups(
                phase1_groups_trace
            )
        logger.debug(
            "Phase 1 complete: extracted=%d prompt_tokens=%s completion_tokens=%s",
            len(extracted),
            phase1_usage.get("prompt_tokens"),
            phase1_usage.get("completion_tokens"),
        )
        logger.info(
            "Phase 1 anchor resolution: extracted=%d anchored=%d",
            len(extracted),
            sum(1 for item in extracted if item.get("chunk_ids")),
        )
        extracted = self._deduplicate_phase1_extractions(
            _expand_combined_phase1_extractions(extracted)
        )
        actionable = [
            item
            for item in extracted
            if _normalize_category(str(item["category"])) in ACTIONABLE_CATEGORIES
        ]

        resolved_terms: list[LLMPhenotype] = []
        prompt_tokens_total = int(phase1_usage.get("prompt_tokens", 0) or 0)
        completion_tokens_total = int(phase1_usage.get("completion_tokens", 0) or 0)
        token_usage_total = _sum_usage_dicts(phase1_usage)
        request_count_total = phase1_request_count
        prompt_version = extraction_prompt.version
        phase_timings: dict[str, float] = {
            "phase1_seconds": phase1_elapsed,
            "phase2a_seconds": 0.0,
            "phase2b_local_seconds": 0.0,
            "phase2b_llm_seconds": 0.0,
        }
        phase_counts: dict[str, int] = {
            "extracted_phrases": len(extracted),
            "actionable_phrases": len(actionable),
            "candidate_sets": 0,
            "unresolved_phrases": 0,
            "local_matches": 0,
            "llm_mapped_phrases": 0,
            "local_fallbacks": 0,
            "phase1_fallbacks": phase1_fallback_count,
        }
        phase_request_counts: dict[str, int] = {
            "phase1_requests": phase1_request_count,
            "phase2b_llm_requests": 0,
        }
        trace: dict[str, Any] = {
            "phase1": build_phase1_trace(
                extracted=extracted,
                groups_trace=phase1_groups_trace,
                attempts_trace=phase1_attempts_trace,
                initial_mode=phase1_initial_mode,
                final_mode=phase1_final_mode,
                fallback_count=phase1_fallback_count,
                failure_class=phase1_failure_class,
            )
        }
        all_phase1_groups = [
            group
            for attempt in phase1_attempts_trace
            for group in attempt.get("groups", [])
        ]
        if all_phase1_groups:
            phase_counts["phase1_completed_groups"] = sum(
                1 for group in all_phase1_groups if group.get("status") == "completed"
            )
            phase_counts["phase1_failed_groups"] = sum(
                1 for group in all_phase1_groups if group.get("status") == "failed"
            )
            phase_counts["phase1_partial_failures"] = int(
                phase_counts["phase1_failed_groups"] > 0
                and phase_counts["phase1_completed_groups"] > 0
            )
        if actionable:
            logger.debug(
                "Phase 2A: retrieving candidates for actionable_phrases=%d",
                len(actionable),
            )
            phase2a_start = time.perf_counter()
            phrase_candidates = self._retrieve_candidates(
                actionable=actionable,
                grounded_chunks=grounded_chunks,
                language=language,
            )
            phase_timings["phase2a_seconds"] = round(
                time.perf_counter() - phase2a_start, 6
            )
            phase_counts["candidate_sets"] = len(phrase_candidates)
            trace["phase2a"] = build_phase2a_trace(phrase_candidates)
            logger.debug(
                "Phase 2A complete: candidate_sets=%d",
                len(phrase_candidates),
            )
            logger.debug(
                "Phase 2B-local: resolving local matches for candidate_sets=%d",
                len(phrase_candidates),
            )
            phase2b_local_start = time.perf_counter()
            (
                locally_resolved,
                unresolved,
                routing_counts,
            ) = self._route_phase2_candidates(
                phrase_candidates=phrase_candidates,
                language=language,
            )
            phase_timings["phase2b_local_seconds"] = round(
                time.perf_counter() - phase2b_local_start, 6
            )
            phase_counts["unresolved_phrases"] = len(unresolved)
            phase_counts["local_matches"] = len(locally_resolved)
            phase_counts.update(routing_counts)
            resolved_terms.extend(locally_resolved)
            trace["phase2b_local"] = build_phase2b_local_trace(
                locally_resolved=locally_resolved,
                unresolved=unresolved,
            )
            logger.debug(
                "Phase 2B-local complete: matched=%d unresolved=%d",
                len(locally_resolved),
                len(unresolved),
            )
            if unresolved:
                mapping_prompt = (
                    get_batch_mapping_prompt(language)
                    if len(unresolved) > 1 and self.mapping_batch_size > 1
                    else get_mapping_prompt(language)
                )
                prompt_version = f"{prompt_version}+{mapping_prompt.version}"
                logger.debug(
                    "Phase 2B-llm: mapping unresolved phrases=%d",
                    len(unresolved),
                )
                phase2b_llm_start = time.perf_counter()
                (
                    mapped_terms,
                    mapping_prompt_tokens,
                    mapping_completion_tokens,
                    mapping_token_usage,
                    mapping_request_count,
                    local_fallback_count,
                    mapping_trace,
                ) = self._resolve_with_mapping_prompt(
                    unresolved=unresolved,
                    mapping_prompt=mapping_prompt,
                )
                phase_timings["phase2b_llm_seconds"] = round(
                    time.perf_counter() - phase2b_llm_start, 6
                )
                prompt_tokens_total += mapping_prompt_tokens
                completion_tokens_total += mapping_completion_tokens
                token_usage_total = _sum_usage_dicts(
                    token_usage_total,
                    mapping_token_usage,
                )
                request_count_total += mapping_request_count
                phase_request_counts["phase2b_llm_requests"] = mapping_request_count
                resolved_terms.extend(mapped_terms)
                phase_counts["local_fallbacks"] = local_fallback_count
                phase_counts["llm_mapped_phrases"] = (
                    len(mapped_terms) - local_fallback_count
                )
                trace["phase2b_llm"] = {"resolved": mapping_trace}
                logger.debug(
                    "Phase 2B-llm complete: mapped=%d prompt_tokens=%d completion_tokens=%d",
                    phase_counts["llm_mapped_phrases"],
                    mapping_prompt_tokens,
                    mapping_completion_tokens,
                )

        return LLMExtractionResult(
            terms=self._deduplicate_terms(resolved_terms),
            meta=LLMMeta(
                llm_provider=config.provider,
                llm_model=config.model,
                llm_mode=config.mode,
                prompt_version=prompt_version,
                token_input=prompt_tokens_total,
                token_output=completion_tokens_total,
                token_count_source=getattr(
                    self.provider,
                    "token_count_source",
                    None,
                ),
                token_usage=token_usage_total,
                request_count=request_count_total,
                phase_timings=phase_timings,
                phase_counts=phase_counts,
                phase_request_counts=phase_request_counts,
                trace=trace,
            ),
        )

    def warmup(self, *, language: str) -> None:
        if hasattr(self.tool_executor, "warmup"):
            self.tool_executor.warmup(language=language)

    @staticmethod
    def _render_chunk_index(grounded_chunks: list[dict[str, Any]]) -> str:
        if not grounded_chunks:
            return "[]"
        return "\n".join(
            f"- chunk_id={chunk['chunk_id']}: {chunk.get('text', '')}"
            for chunk in grounded_chunks
        )

    def _extract_phase1_phenotypes(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_prompt,
        chunk_index_text: str | None = None,
        capture_debug: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int], int, dict[str, Any] | None]:
        user_prompt = _render_phase1_user_prompt(
            extraction_prompt=extraction_prompt,
            text=text,
            grounded_chunks=grounded_chunks,
            chunk_index_text=chunk_index_text,
        )
        max_output_tokens = (
            DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS
            if grounded_chunks or chunk_index_text is not None
            else DEFAULT_PHASE1_MAX_OUTPUT_TOKENS
        )
        logger.debug(
            "Phase 1 request shape: grounded=%s chunk_count=%d user_prompt_chars=%d max_output_tokens=%d",
            bool(grounded_chunks),
            len(grounded_chunks),
            len(user_prompt),
            max_output_tokens,
        )
        response_model = (
            LLMGroundedExtractedPhenotypes
            if grounded_chunks or chunk_index_text is not None
            else LLMExtractedPhenotypes
        )
        try:
            response = self.provider.run_structured_prompt(
                system_prompt=extraction_prompt.render_system_prompt(),
                user_prompt=user_prompt,
                response_model=response_model,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            logger.exception("Phase 1 structured extraction failed")
            raise LLMPipelinePhaseError(
                "phase1",
                "Structured extraction failed",
                usage=dict(getattr(self.provider, "last_usage", {}) or {}),
                request_count=int(getattr(self.provider, "last_request_count", 0) or 0),
            ) from exc

        usage = dict(getattr(self.provider, "last_usage", {}) or {})
        request_count = int(getattr(self.provider, "last_request_count", 0) or 0)
        parsed: list[dict[str, Any]] = []
        for phenotype in response.phenotypes:
            parsed.append(
                {
                    "phrase": phenotype.phrase.strip(),
                    "category": _normalize_category(phenotype.category),
                    "chunk_ids": list(getattr(phenotype, "chunk_ids", [])),
                    "evidence_text": getattr(phenotype, "evidence_text", None),
                    "start_char": getattr(phenotype, "start_char", None),
                    "end_char": getattr(phenotype, "end_char", None),
                }
            )
        debug_payload: dict[str, Any] | None = None
        if capture_debug:
            structured_payload = (
                dict(getattr(self.provider, "last_structured_payload", {}) or {})
                or None
            )
            debug_payload = {
                "source_text": chunk_index_text
                if chunk_index_text is not None
                else text,
                "user_prompt": user_prompt,
                "structured_response": structured_payload,
                "parsed_extracted": list(parsed),
            }
        return (
            parsed,
            usage,
            request_count,
            debug_payload,
        )

    def _extract_grouped_phase1_phenotypes(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_groups: list[dict[str, Any]],
        extraction_prompt,
        capture_debug: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int], int, float, list[dict[str, Any]]]:
        extracted: list[dict[str, Any]] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        total_request_count = 0
        phase1_groups_trace: list[dict[str, Any]] = []
        chunk_lookup = {chunk["chunk_id"]: chunk for chunk in grounded_chunks}
        phase1_started = time.perf_counter()
        indexed_groups = [
            {"group_index": index, **dict(group)}
            for index, group in enumerate(extraction_groups)
        ]
        indexed_results: list[
            tuple[
                int,
                dict[str, Any],
                list[dict[str, Any]] | None,
                dict[str, int] | None,
                int,
                float,
                dict[str, Any] | None,
                LLMPipelinePhaseError | None,
            ]
        ] = []
        max_workers = min(len(indexed_groups), 2) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_phase1_group,
                    extraction_group=group,
                    text=text,
                    extraction_prompt=extraction_prompt,
                    chunk_lookup=chunk_lookup,
                    capture_debug=capture_debug,
                ): group
                for group in indexed_groups
            }
            for future in as_completed(futures):
                group = futures[future]
                group_index = int(group.get("group_index", 0))
                try:
                    (
                        group_extracted,
                        group_usage,
                        group_request_count,
                        group_elapsed,
                        phase1_debug,
                    ) = future.result()
                    indexed_results.append(
                        (
                            group_index,
                            group,
                            group_extracted,
                            group_usage,
                            group_request_count,
                            group_elapsed,
                            phase1_debug,
                            None,
                        )
                    )
                except LLMPipelinePhaseError as exc:
                    indexed_results.append(
                        (
                            group_index,
                            group,
                            None,
                            exc.usage,
                            exc.request_count,
                            float(exc.elapsed_seconds or 0.0),
                            None,
                            exc,
                        )
                    )

        indexed_results.sort(key=lambda item: item[0])
        first_group_error: LLMPipelinePhaseError | None = None

        for (
            _group_index,
            group,
            group_extracted_opt,
            group_usage_opt,
            group_request_count,
            group_elapsed,
            phase1_debug,
            group_error,
        ) in indexed_results:
            group_chunk_ids = [int(chunk_id) for chunk_id in group.get("chunk_ids", [])]
            if group_error is not None:
                if first_group_error is None:
                    first_group_error = group_error
                prompt_tokens_total += int(
                    (group_usage_opt or {}).get("prompt_tokens", 0) or 0
                )
                completion_tokens_total += int(
                    (group_usage_opt or {}).get("completion_tokens", 0) or 0
                )
                total_request_count += group_request_count
                logger.warning(
                    "Continuing after grouped phase 1 failure: group_id=%s chunk_ids=%s error=%s",
                    group.get("group_id"),
                    group_chunk_ids,
                    str(group_error),
                )
                phase1_groups_trace.append(
                    {
                        "group_id": group.get("group_id"),
                        "chunk_ids": group_chunk_ids,
                        "source_chunk_ids": group_chunk_ids,
                        "status": "failed",
                        "error": str(group_error),
                        "error_type": type(group_error).__name__,
                        "failure_class": _classify_phase1_failure(group_error),
                        "extracted_count": 0,
                        "extracted": [],
                        "elapsed_seconds": group_elapsed,
                        **({"debug": phase1_debug} if phase1_debug else {}),
                    }
                )
                continue

            assert group_extracted_opt is not None
            assert group_usage_opt is not None
            group_extracted = group_extracted_opt
            group_usage = group_usage_opt
            prompt_tokens_total += int(group_usage.get("prompt_tokens", 0) or 0)
            completion_tokens_total += int(group_usage.get("completion_tokens", 0) or 0)
            total_request_count += group_request_count
            extracted.extend(group_extracted)
            group_trace_extracted = phase1_extracted_trace_items(group_extracted)
            phase1_groups_trace.append(
                {
                    "group_id": group.get("group_id"),
                    "chunk_ids": group_chunk_ids,
                    "source_chunk_ids": group_chunk_ids,
                    "status": "completed",
                    "extracted_count": len(group_trace_extracted),
                    "prompt_tokens": int(group_usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        group_usage.get("completion_tokens", 0) or 0
                    ),
                    "request_count": group_request_count,
                    "elapsed_seconds": group_elapsed,
                    "extracted": group_trace_extracted,
                    **({"debug": phase1_debug} if phase1_debug else {}),
                }
            )

        any_group_succeeded = any(
            group.get("status") == "completed" for group in phase1_groups_trace
        )
        if extraction_groups and not any_group_succeeded:
            raise LLMPipelinePhaseError(
                "phase1",
                "Structured extraction failed for all extraction groups",
                usage={
                    "prompt_tokens": prompt_tokens_total,
                    "completion_tokens": completion_tokens_total,
                    "total_tokens": prompt_tokens_total + completion_tokens_total,
                },
                request_count=total_request_count,
                elapsed_seconds=round(time.perf_counter() - phase1_started, 6),
                groups_trace=phase1_groups_trace,
            ) from first_group_error

        extracted = self._deduplicate_phase1_extractions(extracted)

        return (
            extracted,
            {
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
                "total_tokens": prompt_tokens_total + completion_tokens_total,
            },
            total_request_count,
            round(time.perf_counter() - phase1_started, 6),
            phase1_groups_trace,
        )

    def _run_phase1_group(
        self,
        *,
        extraction_group: dict[str, Any],
        text: str,
        extraction_prompt: PromptTemplate,
        chunk_lookup: dict[int, dict[str, Any]],
        capture_debug: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int], int, float, dict[str, Any] | None]:
        group_chunk_ids = [
            int(chunk_id) for chunk_id in extraction_group.get("chunk_ids", [])
        ]
        group_grounded_chunks = [
            chunk_lookup[chunk_id]
            for chunk_id in group_chunk_ids
            if chunk_id in chunk_lookup
        ]
        group_started = time.perf_counter()
        try:
            group_extracted, group_usage, group_request_count, phase1_debug = (
                self._extract_phase1_phenotypes(
                    text=text,
                    grounded_chunks=group_grounded_chunks,
                    chunk_index_text=_render_group_chunk_index_text(
                        group=extraction_group,
                        grounded_chunks=group_grounded_chunks,
                    ),
                    extraction_prompt=extraction_prompt,
                    capture_debug=capture_debug,
                )
            )
        except LLMPipelinePhaseError as exc:
            exc.elapsed_seconds = round(time.perf_counter() - group_started, 6)
            raise
        return (
            group_extracted,
            group_usage,
            group_request_count,
            round(time.perf_counter() - group_started, 6),
            phase1_debug if capture_debug else None,
        )

    def _resolve_initial_phase1_mode(
        self,
        *,
        extraction_groups: list[dict[str, Any]],
    ) -> Phase1Mode:
        return "grouped_large" if extraction_groups else "ungrouped"

    def _run_phase1_mode(
        self,
        *,
        mode: Phase1Mode,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_groups: list[dict[str, Any]],
        extraction_prompt: PromptTemplate,
        capture_phase1_debug: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int], int, float, list[dict[str, Any]]]:
        phase1_started = time.perf_counter()
        try:
            if mode == "ungrouped":
                extracted, phase1_usage, phase1_request_count, phase1_debug = (
                    self._extract_phase1_phenotypes(
                        text=text,
                        grounded_chunks=grounded_chunks,
                        extraction_prompt=extraction_prompt,
                        capture_debug=capture_phase1_debug,
                    )
                )
                groups_trace: list[dict[str, Any]] = []
                if capture_phase1_debug and phase1_debug is not None:
                    groups_trace = [
                        {
                            "group_id": 1,
                            "chunk_ids": [
                                int(chunk["chunk_id"])
                                for chunk in grounded_chunks
                                if "chunk_id" in chunk
                            ],
                            "source_chunk_ids": [
                                int(chunk["chunk_id"])
                                for chunk in grounded_chunks
                                if "chunk_id" in chunk
                            ],
                            "status": "completed",
                            "extracted_count": len(extracted),
                            "prompt_tokens": int(
                                phase1_usage.get("prompt_tokens", 0) or 0
                            ),
                            "completion_tokens": int(
                                phase1_usage.get("completion_tokens", 0) or 0
                            ),
                            "request_count": phase1_request_count,
                            "elapsed_seconds": round(
                                time.perf_counter() - phase1_started, 6
                            ),
                            "extracted": phase1_extracted_trace_items(extracted),
                            "debug": phase1_debug,
                        }
                    ]
                return (
                    extracted,
                    phase1_usage,
                    phase1_request_count,
                    round(time.perf_counter() - phase1_started, 6),
                    groups_trace,
                )

            active_extraction_groups = (
                extraction_groups
                if mode == "grouped_large" and extraction_groups
                else self._build_large_extraction_groups(
                    text=text,
                    grounded_chunks=grounded_chunks,
                    extraction_prompt=extraction_prompt,
                )
                if mode == "grouped_large"
                else self._build_small_extraction_groups(
                    grounded_chunks=grounded_chunks
                )
            )
            return self._extract_grouped_phase1_phenotypes(
                text=text,
                grounded_chunks=grounded_chunks,
                extraction_groups=active_extraction_groups,
                extraction_prompt=extraction_prompt,
                capture_debug=capture_phase1_debug,
            )
        except LLMPipelinePhaseError as exc:
            if exc.elapsed_seconds is None:
                exc.elapsed_seconds = round(time.perf_counter() - phase1_started, 6)
            raise
        except Exception as exc:
            raise LLMPipelinePhaseError(
                "phase1",
                f"Phase 1 grouped setup failed: {exc}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                request_count=0,
                elapsed_seconds=round(time.perf_counter() - phase1_started, 6),
                groups_trace=[],
            ) from exc

    def _phase1_partial_retry_chunks(
        self,
        *,
        mode: Phase1Mode,
        grounded_chunks: list[dict[str, Any]],
        groups_trace: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        Phase1FailureClass,
        list[dict[str, Any]],
        Phase1FailureClass,
    ]:
        if mode != "grouped_large":
            return [], None, list(groups_trace), None

        next_mode = _next_phase1_mode(mode)
        retry_chunk_ids: set[int] = set()
        retained_groups_trace: list[dict[str, Any]] = []
        retry_failure_class: Phase1FailureClass = None
        terminal_failure_class: Phase1FailureClass = None

        for group in groups_trace:
            if group.get("status") != "failed":
                retained_groups_trace.append(group)
                continue

            failure_class = group.get("failure_class")
            if self._should_fallback_phase1(
                failure_class=failure_class,
                grounded_chunks=grounded_chunks,
                next_mode=next_mode,
            ):
                if retry_failure_class is None:
                    retry_failure_class = failure_class
                retry_chunk_ids.update(_group_source_chunk_ids(group))
                continue
            retained_groups_trace.append(group)
            if terminal_failure_class is None:
                terminal_failure_class = failure_class

        if terminal_failure_class is not None:
            return [], None, list(groups_trace), terminal_failure_class
        if not retry_chunk_ids:
            return [], None, list(groups_trace), None

        retry_grounded_chunks = [
            chunk
            for chunk in grounded_chunks
            if int(chunk.get("chunk_id", 0)) in retry_chunk_ids
        ]
        return (
            retry_grounded_chunks,
            retry_failure_class,
            retained_groups_trace,
            None,
        )

    def _should_fallback_phase1(
        self,
        *,
        failure_class: Phase1FailureClass,
        grounded_chunks: list[dict[str, Any]],
        next_mode: Phase1Mode | None,
    ) -> bool:
        if not DEFAULT_PHASE1_FALLBACK_ENABLED:
            return False
        if next_mode is None or not grounded_chunks:
            return False
        return failure_class in {
            "provider_timeout",
            "structured_json_invalid",
            "structured_schema_validation_failed",
        }

    def _build_large_extraction_groups(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_prompt: PromptTemplate,
    ) -> list[dict[str, Any]]:
        if not grounded_chunks:
            return []
        grounded_chunk_models = [
            GroundedChunk(
                chunk_id=int(chunk["chunk_id"]),
                text=str(chunk.get("text", "")),
                start_char=chunk.get("start_char"),
                end_char=chunk.get("end_char"),
                status=str(chunk.get("status", "unknown")),
            )
            for chunk in grounded_chunks
        ]
        try:
            return [
                asdict(group)
                for group in build_extraction_groups(
                    grounded_chunks=grounded_chunk_models,
                    provider=self.provider,
                    system_prompt=extraction_prompt.render_system_prompt(),
                    max_prompt_tokens=max(
                        int(DEFAULT_PHASE1_LARGE_GROUP_MAX_PROMPT_TOKENS),
                        1,
                    ),
                )
            ]
        except (NotImplementedError, TypeError):
            pass

        return [
            {
                "group_id": 1,
                "chunk_ids": [int(chunk["chunk_id"]) for chunk in grounded_chunks],
                "text": self._render_chunk_index(grounded_chunks),
                "estimated_prompt_tokens": 0,
            }
        ]

    def _build_small_extraction_groups(
        self,
        *,
        grounded_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not grounded_chunks:
            return []

        max_chunks = max(int(DEFAULT_PHASE1_SMALL_GROUP_MAX_CHUNKS), 1)
        groups: list[dict[str, Any]] = []
        start_index = 0
        while start_index < len(grounded_chunks):
            group_chunks = grounded_chunks[start_index : start_index + max_chunks]
            groups.append(
                {
                    "group_id": len(groups) + 1,
                    "chunk_ids": [int(chunk["chunk_id"]) for chunk in group_chunks],
                    "text": self._render_chunk_index(group_chunks),
                    "estimated_prompt_tokens": 0,
                }
            )
            if start_index + max_chunks >= len(grounded_chunks):
                break
            start_index += max_chunks - 1 if max_chunks > 1 else 1
        return groups

    @staticmethod
    def _phase1_failure_class_from_groups(
        phase1_groups_trace: list[dict[str, Any]],
    ) -> Phase1FailureClass:
        return phase1_failure_class_from_groups(phase1_groups_trace)

    @staticmethod
    def _attach_phase1_failure_context(
        *,
        exc: LLMPipelinePhaseError,
        initial_mode: Phase1Mode,
        final_mode: Phase1Mode,
        fallback_count: int,
        failure_class: Phase1FailureClass,
        final_groups_trace: list[dict[str, Any]],
        attempts_trace: list[dict[str, Any]],
    ) -> None:
        attach_phase1_failure_context(
            exc=exc,
            initial_mode=initial_mode,
            final_mode=final_mode,
            fallback_count=fallback_count,
            failure_class=failure_class,
            final_groups_trace=final_groups_trace,
            attempts_trace=attempts_trace,
        )

    def _retrieve_candidates(
        self,
        *,
        actionable: list[dict[str, Any]],
        grounded_chunks: list[dict[str, Any]],
        language: str,
    ) -> list[dict[str, Any]]:
        unique_actionable: list[dict[str, Any]] = []
        actionable_groups: dict[
            tuple[str, str, tuple[str, ...]], list[dict[str, Any]]
        ] = {}
        for item in actionable:
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=False)
            actionable_groups.setdefault(dedupe_key, []).append(item)
            if len(actionable_groups[dedupe_key]) == 1:
                unique_actionable.append(item)

        expanded_queries: list[str] = []
        expanded_query_keys: list[tuple[str, str, tuple[str, ...]]] = []
        for item in unique_actionable:
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=False)
            query_variants = prepare_retrieval_queries(str(item["phrase"]))
            if not query_variants:
                query_variants = [str(item["phrase"])]
            for query in query_variants:
                expanded_queries.append(query)
                expanded_query_keys.append(dedupe_key)

        batched_variant_results = self.tool_executor.query_batch_hpo_terms(
            phrases=expanded_queries,
            language=language,
            n_results=self.n_results_per_phrase,
        )

        shared_results: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
        grouped_variant_results: dict[
            tuple[str, str, tuple[str, ...]], list[tuple[str, dict[str, Any]]]
        ] = {}
        for index, dedupe_key in enumerate(expanded_query_keys):
            query = expanded_queries[index]
            batch_result = (
                batched_variant_results[index]
                if index < len(batched_variant_results)
                else {}
            )
            grouped_variant_results.setdefault(dedupe_key, []).append(
                (query, batch_result)
            )

        for item in unique_actionable:
            phrase = str(item["phrase"])
            category = str(item["category"])
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=False)
            merged_candidates: dict[str, dict[str, Any]] = {}
            for query, batch_result in grouped_variant_results.get(dedupe_key, []):
                if "candidates" in batch_result:
                    candidates = [
                        {
                            **dict(candidate),
                            "retrieval_query": query,
                        }
                        for candidate in batch_result.get("candidates", [])
                        if isinstance(candidate, dict)
                    ]
                else:
                    metadatas = self._extract_first_result_list(
                        batch_result, "metadatas"
                    )
                    similarities = self._extract_first_result_list(
                        batch_result, "similarities"
                    )
                    candidates = [
                        {
                            **candidate,
                            "retrieval_query": query,
                        }
                        for candidate in self._hybrid_select_candidates(
                            phrase=query,
                            metadatas=metadatas,
                            similarities=similarities,
                        )
                    ]

                for candidate in candidates:
                    hpo_id = str(candidate.get("hpo_id", "")).strip()
                    if not hpo_id:
                        continue
                    existing = merged_candidates.get(hpo_id)
                    if existing is None or float(
                        candidate.get("score", 0.0) or 0.0
                    ) > float(existing.get("score", 0.0) or 0.0):
                        merged_candidates[hpo_id] = candidate

            merged = sorted(
                merged_candidates.values(),
                key=lambda candidate: float(candidate.get("score", 0.0) or 0.0),
                reverse=True,
            )[: self.n_results_per_phrase]
            shared_results[dedupe_key] = {
                "phrase": phrase,
                "category": category,
                "candidates": merged,
            }
            logger.debug(
                "Phase 2A candidate retrieval: phrase=%r candidates=%d",
                phrase,
                len(shared_results[dedupe_key]["candidates"]),
            )

        results: list[dict[str, Any]] = []
        for item in actionable:
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=False)
            shared_result = dict(shared_results.get(dedupe_key, {}))
            shared_result.setdefault("phrase", str(item["phrase"]))
            shared_result.setdefault("category", str(item["category"]))
            shared_result["chunk_ids"] = list(item.get("chunk_ids", []))
            shared_result["evidence_text"] = item.get("evidence_text")
            shared_result["start_char"] = item.get("start_char")
            shared_result["end_char"] = item.get("end_char")
            shared_result["grounded_context"] = self._build_grounded_context(
                item=item,
                grounded_chunks=grounded_chunks,
            )
            shared_result["candidates"] = list(shared_result.get("candidates", []))
            results.append(shared_result)
        return results

    @staticmethod
    def _extract_first_result_list(
        batch_result: dict[str, Any],
        key: str,
    ) -> list[Any]:
        return extract_first_result_list(batch_result, key)

    @staticmethod
    def _build_grounded_context(
        *,
        item: dict[str, Any],
        grounded_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return build_grounded_context(item=item, grounded_chunks=grounded_chunks)

    def _hybrid_select_candidates(
        self,
        *,
        phrase: str,
        metadatas: list[dict[str, Any]],
        similarities: list[float],
    ) -> list[dict[str, Any]]:
        return hybrid_select_candidates(
            phrase=phrase,
            metadatas=metadatas,
            similarities=similarities,
            max_unique_candidates=self.max_unique_candidates,
            min_unique_candidates=self.min_unique_candidates,
            similarity_threshold=self.similarity_threshold,
        )

    @staticmethod
    def _deduplicate_phase1_extractions(
        extracted: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deduplicated: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        for item in extracted:
            key = _phase1_extraction_dedup_key(item)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged = False
            for existing in deduplicated:
                if not TwoPhaseLLMPipeline._should_merge_phase1_extractions(
                    existing, item
                ):
                    continue
                TwoPhaseLLMPipeline._merge_phase1_extraction(existing, item)
                merged = True
                break
            if merged:
                continue
            deduplicated.append(dict(item))
        return deduplicated

    @staticmethod
    def _should_merge_phase1_extractions(
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> bool:
        if (
            str(existing.get("phrase", "")).strip().lower()
            != str(incoming.get("phrase", "")).strip().lower()
        ):
            return False
        if _normalize_category(
            str(existing.get("category", ""))
        ) != _normalize_category(str(incoming.get("category", ""))):
            return False

        existing_chunk_ids = set(_sorted_chunk_ids(existing.get("chunk_ids", [])))
        incoming_chunk_ids = set(_sorted_chunk_ids(incoming.get("chunk_ids", [])))
        if not existing_chunk_ids or not incoming_chunk_ids:
            return False
        if not (existing_chunk_ids & incoming_chunk_ids):
            return False

        if (
            existing.get("start_char") is not None
            and existing.get("end_char") is not None
            and incoming.get("start_char") is not None
            and incoming.get("end_char") is not None
            and not _spans_overlap(
                int(existing["start_char"]),
                int(existing["end_char"]),
                int(incoming["start_char"]),
                int(incoming["end_char"]),
            )
        ):
            return False

        existing_evidence = _normalized_evidence_text(existing)
        incoming_evidence = _normalized_evidence_text(incoming)
        if not existing_evidence or not incoming_evidence:
            return False
        return (
            existing_evidence == incoming_evidence
            or existing_evidence in incoming_evidence
            or incoming_evidence in existing_evidence
        )

    @staticmethod
    def _merge_phase1_extraction(
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> None:
        existing["chunk_ids"] = _sorted_chunk_ids(
            [*existing.get("chunk_ids", []), *incoming.get("chunk_ids", [])]
        )
        existing["evidence_text"] = _prefer_richer_text(
            existing.get("evidence_text"),
            incoming.get("evidence_text"),
        )
        existing["start_char"] = _merge_optional_bounds(
            existing.get("start_char"),
            incoming.get("start_char"),
            pick="min",
        )
        existing["end_char"] = _merge_optional_bounds(
            existing.get("end_char"),
            incoming.get("end_char"),
            pick="max",
        )

    def _resolve_local_matches(
        self,
        phrase_candidates: list[dict[str, Any]],
    ) -> tuple[list[LLMPhenotype], list[dict[str, Any]]]:
        resolved, unresolved, _ = self._route_phase2_candidates(
            phrase_candidates=phrase_candidates,
            language=DEFAULT_LLM_LANGUAGE,
        )
        return resolved, unresolved

    def _route_phase2_candidates(
        self,
        *,
        phrase_candidates: list[dict[str, Any]],
        language: str,
    ) -> tuple[list[LLMPhenotype], list[dict[str, Any]], dict[str, int]]:
        resolved: list[LLMPhenotype] = []
        unresolved: list[dict[str, Any]] = []
        counts = {
            "phase2b_local_accept_count": 0,
            "phase2b_deferred_count": 0,
            "phase2b_no_candidate_skip_count": 0,
        }

        for item in phrase_candidates:
            candidates = item.get("candidates", [])
            if not candidates:
                logger.warning(
                    "Phase 2B-local: phrase=%r has no candidates; skipping mapping",
                    item["phrase"],
                )
                counts["phase2b_no_candidate_skip_count"] += 1
                continue

            decision = self._decide_phase2_routing(item=item, language=language)
            if decision != "local":
                unresolved.append(item)
                counts["phase2b_deferred_count"] += 1
                logger.debug(
                    "Phase 2B-local: phrase=%r unresolved_candidates=%d",
                    item["phrase"],
                    len(candidates),
                )
                continue

            local_match = self._try_local_match(item["phrase"], candidates)
            if local_match is None:
                unresolved.append(item)
                counts["phase2b_deferred_count"] += 1
                logger.debug(
                    "Phase 2B-local: phrase=%r deferred_after_missing_local_match",
                    item["phrase"],
                )
                continue

            resolved.append(
                self._phenotype_from_candidate(item=item, candidate=local_match)
            )
            counts["phase2b_local_accept_count"] += 1
            logger.debug(
                "Phase 2B-local: phrase=%r matched=%s",
                item["phrase"],
                local_match["hpo_id"],
            )
        return resolved, unresolved, counts

    def _decide_phase2_routing(self, *, item: dict[str, Any], language: str) -> str:
        candidates = list(item.get("candidates", []))
        local_match = self._try_local_match(str(item["phrase"]), candidates)
        if local_match is None:
            return "defer"

        normalized_language = (language or "").strip().lower()
        phrase_clean = _clean_text(str(item["phrase"]))
        term_clean = _clean_text(_candidate_match_text(local_match))
        phrase_tokens = _tokenize(str(item["phrase"]))
        term_tokens = _tokenize(_candidate_match_text(local_match))
        match_score = float(local_match.get("score", 0.0) or 0.0)

        if normalized_language == "en":
            if phrase_clean == term_clean or (
                phrase_tokens and term_tokens and phrase_tokens == term_tokens
            ):
                return "local" if match_score >= 0.85 else "defer"
            if term_clean and re.search(rf"\b{re.escape(term_clean)}\b", phrase_clean):
                return "local" if match_score >= 0.85 else "defer"
            if token_sort_similarity(phrase_clean, term_clean) >= max(
                self.local_match_threshold, 90.0
            ):
                return "local" if match_score >= 0.85 else "defer"
            return "defer"

        if normalized_language == "de":
            if phrase_clean == term_clean or (
                phrase_tokens and term_tokens and phrase_tokens == term_tokens
            ):
                return "local"
            return "defer"

        return "defer"

    def _try_local_match(
        self,
        phrase: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        return local_match(
            phrase=phrase,
            candidates=candidates,
            local_match_threshold=self.local_match_threshold,
        )

    def _resolve_with_mapping_prompt(
        self,
        *,
        unresolved: list[dict[str, Any]],
        mapping_prompt,
    ) -> tuple[
        list[LLMPhenotype],
        int,
        int,
        dict[str, int],
        int,
        int,
        list[dict[str, Any]],
    ]:
        resolved: list[LLMPhenotype] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        token_usage_total: dict[str, int] = {}
        request_count_total = 0
        local_fallback_count = 0
        mapping_trace: list[dict[str, Any]] = []
        unresolved_groups: dict[
            tuple[str, str, tuple[str, ...]], list[dict[str, Any]]
        ] = {}
        unique_unresolved: list[dict[str, Any]] = []
        for item in unresolved:
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=True)
            unresolved_groups.setdefault(dedupe_key, []).append(item)
            if len(unresolved_groups[dedupe_key]) == 1:
                unique_unresolved.append(item)

        for start in range(0, len(unique_unresolved), self.mapping_batch_size):
            batch = unique_unresolved[start : start + self.mapping_batch_size]
            mapping_response, batch_usage = self._run_mapping_batch(
                batch=batch,
                mapping_prompt=mapping_prompt,
            )
            request_count_total += int(
                getattr(self.provider, "last_request_count", 0) or 0
            )
            token_usage_total = _sum_usage_dicts(token_usage_total, batch_usage)
            prompt_tokens_total += int(batch_usage.get("prompt_tokens", 0) or 0)
            completion_tokens_total += int(batch_usage.get("completion_tokens", 0) or 0)
            selected_ids = self._select_candidate_ids(
                mapping_response=mapping_response,
                batch=batch,
            )
            for index, item in enumerate(batch):
                dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=True)
                grouped_items = unresolved_groups[dedupe_key]
                selected_id = selected_ids.get(_mapping_batch_item_id(index))
                (
                    item_resolved,
                    item_local_fallback_count,
                    item_trace,
                ) = self._resolve_mapping_selection(item=item, selected_id=selected_id)
                resolved.extend(item_resolved)
                local_fallback_count += item_local_fallback_count
                mapping_trace.append(item_trace)
                for grouped_item in grouped_items[1:]:
                    (
                        grouped_resolved,
                        grouped_local_fallback_count,
                        grouped_item_trace,
                    ) = self._resolve_mapping_selection(
                        item=grouped_item,
                        selected_id=selected_id,
                    )
                    resolved.extend(grouped_resolved)
                    local_fallback_count += grouped_local_fallback_count
                    mapping_trace.append(
                        self._annotate_mapping_trace_with_provenance(
                            grouped_item_trace,
                            grouped_item,
                        )
                    )
        return (
            resolved,
            prompt_tokens_total,
            completion_tokens_total,
            token_usage_total,
            request_count_total,
            local_fallback_count,
            mapping_trace,
        )

    @staticmethod
    def _annotate_mapping_trace_with_provenance(
        trace_entry: dict[str, Any],
        item: dict[str, Any],
    ) -> dict[str, Any]:
        return annotate_mapping_trace_with_provenance(trace_entry, item)

    def _run_mapping_batch(
        self,
        *,
        batch: list[dict[str, Any]],
        mapping_prompt,
    ) -> tuple[LLMMappingSelection | LLMBatchMappingSelections, dict[str, int]]:
        return run_mapping_batch(
            provider=self.provider,
            batch=batch,
            mapping_prompt=mapping_prompt,
        )

    def _resolve_mapping_selection(
        self,
        *,
        item: dict[str, Any],
        selected_id: str | None,
    ) -> tuple[list[LLMPhenotype], int, dict[str, Any]]:
        if not selected_id:
            logger.warning(
                "Phase 2B-llm: phrase=%r produced no valid mapping; trying local fallback",
                item["phrase"],
            )
            fallback = self._try_local_match(item["phrase"], item["candidates"])
            if fallback is None:
                return (
                    [],
                    0,
                    {
                        "phrase": str(item["phrase"]),
                        "selected_id": None,
                        "term_id": None,
                        "label": None,
                        "assertion": CATEGORY_TO_ASSERTION.get(
                            _normalize_category(str(item.get("category", ""))),
                            PRESENT_ASSERTION,
                        ),
                        "category": _normalize_category(str(item.get("category", ""))),
                        "match_method": "llm",
                        "local_fallback": False,
                    },
                )
            logger.debug(
                "Phase 2B-llm: phrase=%r local fallback resolved=%s",
                item["phrase"],
                fallback["hpo_id"],
            )
            phenotype = self._phenotype_from_candidate(
                item=item,
                candidate=fallback,
                match_method="local",
            )
            return (
                [phenotype],
                1,
                {
                    "phrase": str(item["phrase"]),
                    "selected_id": None,
                    "term_id": phenotype.term_id,
                    "label": phenotype.label,
                    "assertion": phenotype.assertion,
                    "category": phenotype.category,
                    "match_method": "llm",
                    "local_fallback": True,
                },
            )

        candidate = next(
            (
                candidate
                for candidate in item["candidates"]
                if candidate["hpo_id"] == selected_id
            ),
            None,
        )
        if candidate is None:
            logger.warning(
                "Phase 2B-llm: phrase=%r selected_id=%s not in candidates; trying local fallback",
                item["phrase"],
                selected_id,
            )
            fallback = self._try_local_match(item["phrase"], item["candidates"])
            if fallback is None:
                return (
                    [],
                    0,
                    {
                        "phrase": str(item["phrase"]),
                        "selected_id": selected_id,
                        "term_id": None,
                        "label": None,
                        "assertion": CATEGORY_TO_ASSERTION.get(
                            _normalize_category(str(item.get("category", ""))),
                            PRESENT_ASSERTION,
                        ),
                        "category": _normalize_category(str(item.get("category", ""))),
                        "match_method": "llm",
                        "local_fallback": False,
                    },
                )
            logger.debug(
                "Phase 2B-llm: phrase=%r local fallback resolved=%s",
                item["phrase"],
                fallback["hpo_id"],
            )
            phenotype = self._phenotype_from_candidate(
                item=item,
                candidate=fallback,
                match_method="local",
            )
            return (
                [phenotype],
                1,
                {
                    "phrase": str(item["phrase"]),
                    "selected_id": selected_id,
                    "term_id": phenotype.term_id,
                    "label": phenotype.label,
                    "assertion": phenotype.assertion,
                    "category": phenotype.category,
                    "match_method": "llm",
                    "local_fallback": True,
                },
            )

        logger.debug(
            "Phase 2B-llm: phrase=%r mapped=%s",
            item["phrase"],
            selected_id,
        )
        phenotype = self._phenotype_from_candidate(
            item=item,
            candidate=candidate,
            match_method="llm",
        )
        return (
            [phenotype],
            0,
            {
                "phrase": str(item["phrase"]),
                "selected_id": selected_id,
                "term_id": phenotype.term_id,
                "label": phenotype.label,
                "assertion": phenotype.assertion,
                "category": phenotype.category,
                "match_method": "llm",
                "local_fallback": False,
            },
        )

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return optional_float(value)

    @staticmethod
    def _phenotype_from_candidate(
        *,
        item: dict[str, Any],
        candidate: dict[str, Any],
        match_method: str = "local",
    ) -> LLMPhenotype:
        return phenotype_from_candidate(
            item=item,
            candidate=candidate,
            match_method=match_method,
        )

    @staticmethod
    def _select_candidate_id(
        *,
        mapping_response: LLMMappingSelection,
        candidates: list[dict[str, Any]],
    ) -> str | None:
        return select_candidate_id(
            mapping_response=mapping_response,
            candidates=candidates,
        )

    @classmethod
    def _select_candidate_ids(
        cls,
        *,
        mapping_response: LLMMappingSelection | LLMBatchMappingSelections,
        batch: list[dict[str, Any]],
    ) -> dict[str, str | None]:
        return select_candidate_ids(mapping_response=mapping_response, batch=batch)

    @staticmethod
    def _deduplicate_terms(terms: list[LLMPhenotype]) -> list[LLMPhenotype]:
        return deduplicate_terms(terms)
