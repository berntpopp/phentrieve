from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from phentrieve.llm.config import (
    DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_LANGUAGE,
    DEFAULT_LLM_MAPPING_BATCH_SIZE,
    DEFAULT_LOCAL_MATCH_THRESHOLD,
    DEFAULT_MAX_UNIQUE_CANDIDATES,
    DEFAULT_MIN_UNIQUE_CANDIDATES,
    DEFAULT_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_QUERY_RESULTS_PER_PHRASE,
    DEFAULT_SIMILARITY_THRESHOLD,
    NEGATED_ASSERTION,
    PRESENT_ASSERTION,
    UNCERTAIN_ASSERTION,
)
from phentrieve.llm.prompts.loader import (
    PromptTemplate,
    get_batch_mapping_prompt,
    get_mapping_prompt,
    get_prompt,
)
from phentrieve.llm.provider import ToolExecutor
from phentrieve.llm.types import (
    AnnotationMode,
    LLMBatchMappingSelections,
    LLMExtractedPhenotypes,
    LLMExtractionResult,
    LLMGroundedExtractedPhenotypes,
    LLMMappingSelection,
    LLMMeta,
    LLMPhenotype,
    LLMPhenotypeEvidence,
    LLMPipelineConfig,
)
from phentrieve.llm.utils import token_sort_similarity

logger = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"\w+")
PHRASE_PARENTHESES_PATTERN = re.compile(r"\(.*?\)")

CATEGORY_TO_ASSERTION = {
    "abnormal": PRESENT_ASSERTION,
    "normal": NEGATED_ASSERTION,
    "suspected": UNCERTAIN_ASSERTION,
    "family_history": "family_history",
    "other": "other",
}

ACTIONABLE_CATEGORIES = frozenset({"abnormal", "normal", "suspected", "family_history"})


class LLMPipelinePhaseError(RuntimeError):
    def __init__(self, phase: str, message: str) -> None:
        super().__init__(message)
        self.phase = phase


def _normalize_category(category: str) -> str:
    normalized = category.strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "family_history": "family_history",
        "familyhistory": "family_history",
    }.get(normalized, normalized)


def _normalize_token(token: str) -> str:
    normalized = token.strip().lower()
    if len(normalized) > 3 and normalized.endswith("s"):
        return normalized[:-1]
    return normalized


def _tokenize(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in WORD_PATTERN.findall(text.lower())
        if _normalize_token(token)
    }


def _clean_text(text: str) -> str:
    text = PHRASE_PARENTHESES_PATTERN.sub("", text or "")
    text = text.replace("_", " ").replace("-", " ").lower().strip()
    return " ".join(text.split())


def _normalize_mapping_phrase_key(text: str) -> str:
    return " ".join(str(text or "").lower().replace("-", " ").split())


def _candidate_id_tuple(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(candidate.get("hpo_id", "")).strip()
                for candidate in item.get("candidates", [])
                if str(candidate.get("hpo_id", "")).strip()
            }
        )
    )


def _downstream_dedupe_key(
    item: dict[str, Any],
    *,
    include_candidate_ids: bool,
) -> tuple[str, str, tuple[str, ...]]:
    return (
        _normalize_mapping_phrase_key(str(item.get("phrase", ""))),
        _normalize_category(str(item.get("category", ""))),
        _candidate_id_tuple(item) if include_candidate_ids else (),
    )


def _render_phase1_user_prompt(
    *,
    extraction_prompt: PromptTemplate,
    text: str,
    grounded_chunks: list[dict[str, Any]],
    chunk_index_text: str | None = None,
) -> str:
    if chunk_index_text is not None:
        chunk_index = chunk_index_text or "[]"
        return (
            "Extract all phenotype phrases from the provided chunk index.\n\n"
            "Chunk index:\n"
            f"{chunk_index}\n"
        )
    if grounded_chunks:
        chunk_index = (
            "\n".join(
                f"- chunk_id={chunk['chunk_id']}: {chunk.get('text', '')}"
                for chunk in grounded_chunks
            )
            or "[]"
        )
        return (
            "Extract all phenotype phrases from the provided chunk index.\n\n"
            "Chunk index:\n"
            f"{chunk_index}\n"
        )
    return extraction_prompt.render_user_prompt(text, chunk_index="[]")


def _render_group_chunk_index_text(
    *, group: dict[str, Any], grounded_chunks: list[dict[str, Any]]
) -> str:
    group_text = group.get("text")
    if isinstance(group_text, str) and group_text.strip():
        return group_text
    return "\n".join(
        f"chunk_id={chunk['chunk_id']}: {chunk.get('text', '')}"
        for chunk in grounded_chunks
    )


def _normalize_grounded_text(text: Any) -> str:
    return str(text or "").strip()


def _mapping_batch_item_id(index: int) -> str:
    return f"item_{index + 1}"


def _compact_mapping_item(
    item: dict[str, Any],
    *,
    item_id: str | None = None,
) -> dict[str, Any]:
    grounded_context = dict(item.get("grounded_context", {}) or {})
    neighbor_texts = [
        _normalize_grounded_text(text)
        for text in grounded_context.get("neighbor_chunk_texts", [])
        if _normalize_grounded_text(text)
    ]
    compact_item: dict[str, Any] = {
        "primary_chunk_text": _normalize_grounded_text(
            grounded_context.get("primary_chunk_text")
        ),
        "neighbor_chunk_texts": neighbor_texts,
        "phrase": str(item["phrase"]).lower().replace("-", " ").strip(),
        "category": item["category"],
        "candidates": [
            {
                "id": candidate["hpo_id"],
                "term": candidate["term_name"],
                "retrieval_score": candidate.get("score"),
            }
            for candidate in item["candidates"]
        ],
    }
    if item_id is not None:
        compact_item["item_id"] = item_id
    return compact_item


def _phase1_extraction_dedup_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("phrase", "")).strip().lower(),
        _normalize_category(str(item.get("category", ""))),
        tuple(int(chunk_id) for chunk_id in item.get("chunk_ids", [])),
        item.get("evidence_text"),
        item.get("start_char"),
        item.get("end_char"),
    )


def _sorted_chunk_ids(chunk_ids: list[Any]) -> list[int]:
    return sorted({int(chunk_id) for chunk_id in chunk_ids})


def _normalized_evidence_text(item: dict[str, Any]) -> str:
    return _clean_text(str(item.get("evidence_text") or item.get("phrase") or ""))


def _prefer_richer_text(current: str | None, incoming: str | None) -> str | None:
    current_text = str(current or "").strip()
    incoming_text = str(incoming or "").strip()
    if not current_text:
        return incoming_text or None
    if not incoming_text:
        return current_text
    return incoming_text if len(incoming_text) > len(current_text) else current_text


def _merge_optional_bounds(
    current: int | None,
    incoming: int | None,
    *,
    pick: str,
) -> int | None:
    if current is None:
        return incoming
    if incoming is None:
        return current
    return min(current, incoming) if pick == "min" else max(current, incoming)


def _spans_overlap(
    start_a: int | None,
    end_a: int | None,
    start_b: int | None,
    end_b: int | None,
) -> bool:
    if None in (start_a, end_a, start_b, end_b):
        return False
    assert start_a is not None
    assert end_a is not None
    assert start_b is not None
    assert end_b is not None
    if end_a <= start_a or end_b <= start_b:
        return False
    return max(start_a, start_b) < min(end_a, end_b)


def _sum_usage_dicts(*usage_dicts: dict[str, int]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for usage in usage_dicts:
        for key, value in usage.items():
            totals[key] = int(totals.get(key, 0) or 0) + int(value or 0)
    return totals


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
        if extraction_groups:
            (
                extracted,
                phase1_usage,
                phase1_request_count,
                phase1_elapsed,
                phase1_groups_trace,
            ) = self._extract_grouped_phase1_phenotypes(
                text=text,
                grounded_chunks=grounded_chunks,
                extraction_groups=extraction_groups,
                extraction_prompt=extraction_prompt,
            )
        else:
            phase1_start = time.perf_counter()
            extracted, phase1_usage, phase1_request_count = (
                self._extract_phase1_phenotypes(
                    text=text,
                    grounded_chunks=grounded_chunks,
                    extraction_prompt=extraction_prompt,
                )
            )
            phase1_elapsed = round(time.perf_counter() - phase1_start, 6)
            phase1_groups_trace = []
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
        }
        phase_request_counts: dict[str, int] = {
            "phase1_requests": phase1_request_count,
            "phase2b_llm_requests": 0,
        }
        trace: dict[str, Any] = {
            "phase1": {
                "extracted": [
                    {
                        "phrase": str(item["phrase"]),
                        "category": _normalize_category(str(item["category"])),
                        "chunk_ids": list(item.get("chunk_ids", [])),
                        "evidence_text": item.get("evidence_text"),
                        "actionable": _normalize_category(str(item["category"]))
                        in ACTIONABLE_CATEGORIES,
                    }
                    for item in extracted
                ],
                "groups": phase1_groups_trace,
            }
        }
        if extraction_groups:
            phase_counts["phase1_completed_groups"] = sum(
                1 for group in phase1_groups_trace if group.get("status") == "completed"
            )
            phase_counts["phase1_failed_groups"] = sum(
                1 for group in phase1_groups_trace if group.get("status") == "failed"
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
            trace["phase2a"] = {
                "candidate_sets": [
                    {
                        "phrase": str(item.get("phrase", "")),
                        "category": _normalize_category(str(item.get("category", ""))),
                        "grounded_context": dict(item.get("grounded_context", {})),
                        "candidates": [
                            {
                                "term_id": str(candidate.get("hpo_id", "")),
                                "label": str(candidate.get("term_name", "")),
                                "score": float(candidate.get("score", 0.0) or 0.0),
                            }
                            for candidate in item.get("candidates", [])
                        ],
                    }
                    for item in phrase_candidates
                ]
            }
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
            trace["phase2b_local"] = {
                "resolved": [
                    {
                        "phrase": term.evidence,
                        "term_id": term.term_id,
                        "label": term.label,
                        "assertion": term.assertion,
                        "category": term.category,
                        "match_method": "local",
                    }
                    for term in locally_resolved
                ],
                "unresolved": [
                    {
                        "phrase": str(item.get("phrase", "")),
                        "category": _normalize_category(str(item.get("category", ""))),
                    }
                    for item in unresolved
                ],
            }
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
    ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
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
        except Exception:
            logger.exception("Phase 1 structured extraction failed")
            raise LLMPipelinePhaseError("phase1", "Structured extraction failed")

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
        return parsed, usage, request_count

    def _extract_grouped_phase1_phenotypes(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_groups: list[dict[str, Any]],
        extraction_prompt,
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
                    ) = future.result()
                    indexed_results.append(
                        (
                            group_index,
                            group,
                            group_extracted,
                            group_usage,
                            group_request_count,
                            group_elapsed,
                            None,
                        )
                    )
                except LLMPipelinePhaseError as exc:
                    indexed_results.append(
                        (
                            group_index,
                            group,
                            None,
                            None,
                            0,
                            group_elapsed,
                            exc,
                        )
                    )

        indexed_results.sort(key=lambda item: item[0])

        for (
            _group_index,
            group,
            group_extracted_opt,
            group_usage_opt,
            group_request_count,
            group_elapsed,
            group_error,
        ) in indexed_results:
            group_chunk_ids = [int(chunk_id) for chunk_id in group.get("chunk_ids", [])]
            if group_error is not None:
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
                        "extracted_count": 0,
                        "extracted": [],
                        "elapsed_seconds": group_elapsed,
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
            group_trace_extracted = [
                {
                    "phrase": str(item["phrase"]),
                    "category": _normalize_category(str(item["category"])),
                    "chunk_ids": list(item.get("chunk_ids", [])),
                    "evidence_text": item.get("evidence_text"),
                    "actionable": _normalize_category(str(item["category"]))
                    in ACTIONABLE_CATEGORIES,
                }
                for item in group_extracted
            ]
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
                }
            )

        any_group_succeeded = any(
            group.get("status") == "completed" for group in phase1_groups_trace
        )
        if extraction_groups and not any_group_succeeded:
            raise LLMPipelinePhaseError(
                "phase1",
                "Structured extraction failed for all extraction groups",
            )

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
    ) -> tuple[list[dict[str, Any]], dict[str, int], int, float]:
        group_chunk_ids = [
            int(chunk_id) for chunk_id in extraction_group.get("chunk_ids", [])
        ]
        group_grounded_chunks = [
            chunk_lookup[chunk_id]
            for chunk_id in group_chunk_ids
            if chunk_id in chunk_lookup
        ]
        group_started = time.perf_counter()
        group_extracted, group_usage, group_request_count = (
            self._extract_phase1_phenotypes(
                text=text,
                grounded_chunks=group_grounded_chunks,
                chunk_index_text=_render_group_chunk_index_text(
                    group=extraction_group,
                    grounded_chunks=group_grounded_chunks,
                ),
                extraction_prompt=extraction_prompt,
            )
        )
        return (
            group_extracted,
            group_usage,
            group_request_count,
            round(time.perf_counter() - group_started, 6),
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

        phrases = [str(item["phrase"]) for item in unique_actionable]
        raw_results = self.tool_executor.query_batch_hpo_terms(
            phrases=phrases,
            language=language,
            n_results=self.n_results_per_phrase,
        )

        shared_results: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
        for index, item in enumerate(unique_actionable):
            phrase = str(item["phrase"])
            category = str(item["category"])
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=False)
            if index < len(raw_results) and "candidates" in raw_results[index]:
                raw_result = dict(raw_results[index])
                raw_result.setdefault("phrase", phrase)
                raw_result.setdefault("category", category)
                shared_results[dedupe_key] = raw_result
                continue

            batch_result = raw_results[index] if index < len(raw_results) else {}
            metadatas = self._extract_first_result_list(batch_result, "metadatas")
            similarities = self._extract_first_result_list(batch_result, "similarities")
            shared_results[dedupe_key] = {
                "phrase": phrase,
                "category": category,
                "candidates": self._hybrid_select_candidates(
                    phrase=phrase,
                    metadatas=metadatas,
                    similarities=similarities,
                ),
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
        values = batch_result.get(key)
        if not isinstance(values, list) or not values:
            return []
        first = values[0]
        return first if isinstance(first, list) else []

    @staticmethod
    def _build_grounded_context(
        *,
        item: dict[str, Any],
        grounded_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        chunk_lookup = {chunk["chunk_id"]: chunk for chunk in grounded_chunks}
        chunk_ids = [int(chunk_id) for chunk_id in item.get("chunk_ids", [])]
        primary = chunk_lookup.get(chunk_ids[0]) if chunk_ids else None
        neighbor_ids = [chunk_ids[0] - 1, chunk_ids[0] + 1] if chunk_ids else []
        neighbors = [
            chunk_lookup[cid]
            for cid in neighbor_ids
            if cid in chunk_lookup and chunk_lookup[cid] is not primary
        ]
        return {
            "chunk_ids": chunk_ids,
            "primary_chunk_text": (
                primary.get("text", "")
                if primary
                else item.get("evidence_text", item.get("phrase", ""))
            ),
            "neighbor_chunk_texts": [chunk.get("text", "") for chunk in neighbors],
        }

    def _hybrid_select_candidates(
        self,
        *,
        phrase: str,
        metadatas: list[dict[str, Any]],
        similarities: list[float],
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        phrase_tokens = _tokenize(phrase)

        for index, metadata in enumerate(metadatas):
            if len(selected) >= self.max_unique_candidates:
                break

            hpo_id = str(metadata.get("hpo_id", "")).strip()
            if not hpo_id or hpo_id in seen_ids:
                continue

            label = str(metadata.get("label", "")).strip()
            similarity = (
                float(similarities[index]) if index < len(similarities) else 0.0
            )
            label_tokens = _tokenize(label)

            has_token_overlap = bool(phrase_tokens & label_tokens)
            meets_threshold = similarity >= self.similarity_threshold
            requires_fill = len(selected) < self.min_unique_candidates

            if has_token_overlap or meets_threshold or requires_fill:
                seen_ids.add(hpo_id)
                selected.append(
                    {
                        "hpo_id": hpo_id,
                        "term_name": label,
                        "score": similarity,
                    }
                )
        return selected

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
        term_clean = _clean_text(str(local_match["term_name"]))
        phrase_tokens = _tokenize(str(item["phrase"]))
        term_tokens = _tokenize(str(local_match["term_name"]))
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
        phrase_clean = _clean_text(phrase)
        phrase_tokens = _tokenize(phrase)

        for candidate in candidates:
            label_tokens = _tokenize(candidate["term_name"])
            if phrase_tokens and label_tokens and phrase_tokens == label_tokens:
                return candidate

        for candidate in candidates:
            if _clean_text(candidate["term_name"]) == phrase_clean:
                return candidate

        if len(phrase_tokens) > 1:
            for candidate in candidates:
                label_clean = _clean_text(candidate["term_name"])
                if label_clean and re.search(
                    rf"\b{re.escape(label_clean)}\b", phrase_clean
                ):
                    return candidate

        best_candidate: dict[str, Any] | None = None
        best_score = 0.0
        for candidate in candidates:
            score = token_sort_similarity(
                phrase_clean, _clean_text(candidate["term_name"])
            )
            if score >= self.local_match_threshold and score > best_score:
                best_candidate = candidate
                best_score = score
        return best_candidate

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
        if not trace_entry:
            return trace_entry
        annotated = dict(trace_entry)
        if item.get("evidence_text") is not None:
            annotated["evidence_text"] = item.get("evidence_text")
        if item.get("chunk_ids") is not None:
            annotated["chunk_ids"] = list(item.get("chunk_ids", []))
        if item.get("start_char") is not None:
            annotated["start_char"] = item.get("start_char")
        if item.get("end_char") is not None:
            annotated["end_char"] = item.get("end_char")
        return annotated

    def _run_mapping_batch(
        self,
        *,
        batch: list[dict[str, Any]],
        mapping_prompt,
    ) -> tuple[LLMMappingSelection | LLMBatchMappingSelections, dict[str, int]]:
        response_model: type[LLMMappingSelection] | type[LLMBatchMappingSelections]
        if len(batch) == 1:
            batch_mapping_prompt = get_mapping_prompt(mapping_prompt.language)
            candidate_payload = json.dumps(
                _compact_mapping_item(batch[0]),
                ensure_ascii=False,
            )
            response_model = LLMMappingSelection
        else:
            batch_mapping_prompt = get_batch_mapping_prompt(mapping_prompt.language)
            candidate_payload = json.dumps(
                {
                    "items": [
                        _compact_mapping_item(
                            item,
                            item_id=_mapping_batch_item_id(index),
                        )
                        for index, item in enumerate(batch)
                    ]
                },
                ensure_ascii=False,
            )
            response_model = LLMBatchMappingSelections
        response = self.provider.run_structured_prompt(
            system_prompt=batch_mapping_prompt.render_system_prompt(
                language=batch_mapping_prompt.language
            ),
            user_prompt=batch_mapping_prompt.render_user_prompt(
                candidate_payload,
                language=batch_mapping_prompt.language,
            ),
            response_model=response_model,
        )
        usage = dict(getattr(self.provider, "last_usage", {}) or {})
        return response, usage

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
    def _phenotype_from_candidate(
        *,
        item: dict[str, Any],
        candidate: dict[str, Any],
        match_method: str = "local",
    ) -> LLMPhenotype:
        evidence = LLMPhenotypeEvidence(
            phrase=str(item["phrase"]),
            evidence_text=item.get("evidence_text"),
            chunk_ids=list(item.get("chunk_ids", [])),
            start_char=item.get("start_char"),
            end_char=item.get("end_char"),
            match_method=match_method,
        )
        return LLMPhenotype(
            term_id=candidate["hpo_id"],
            label=candidate["term_name"],
            evidence=item["phrase"],
            assertion=CATEGORY_TO_ASSERTION.get(
                _normalize_category(str(item.get("category", ""))),
                PRESENT_ASSERTION,
            ),
            category=_normalize_category(str(item.get("category", ""))),
            evidence_records=[evidence],
        )

    @staticmethod
    def _select_candidate_id(
        *,
        mapping_response: LLMMappingSelection,
        candidates: list[dict[str, Any]],
    ) -> str | None:
        candidate_ids = {candidate["hpo_id"] for candidate in candidates}
        return (
            mapping_response.hpo_id
            if mapping_response.hpo_id in candidate_ids
            else None
        )

    @classmethod
    def _select_candidate_ids(
        cls,
        *,
        mapping_response: LLMMappingSelection | LLMBatchMappingSelections,
        batch: list[dict[str, Any]],
    ) -> dict[str, str | None]:
        if len(batch) == 1:
            item = batch[0]
            if not isinstance(mapping_response, LLMMappingSelection):
                return {_mapping_batch_item_id(0): None}
            return {
                _mapping_batch_item_id(0): cls._select_candidate_id(
                    mapping_response=mapping_response,
                    candidates=item["candidates"],
                )
            }

        selected: dict[str, str | None] = {
            _mapping_batch_item_id(index): None for index, _ in enumerate(batch)
        }
        if not isinstance(mapping_response, LLMBatchMappingSelections):
            return selected

        candidates_by_item_id = {
            _mapping_batch_item_id(index): {
                candidate["hpo_id"] for candidate in item["candidates"]
            }
            for index, item in enumerate(batch)
        }
        for mapping in mapping_response.mappings:
            item_id = mapping.item_id
            selected_id = mapping.hpo_id
            if (
                isinstance(selected_id, str)
                and item_id in candidates_by_item_id
                and selected_id in candidates_by_item_id[item_id]
            ):
                selected[item_id] = selected_id
        return selected

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
