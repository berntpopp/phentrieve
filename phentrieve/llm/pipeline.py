from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from phentrieve.llm.config import (
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
    get_batch_mapping_prompt,
    get_mapping_prompt,
    get_prompt,
)
from phentrieve.llm.provider import ToolExecutor
from phentrieve.llm.types import (
    AnnotationMode,
    LLMBatchMappingSelections,
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
        config: LLMPipelineConfig,
    ) -> LLMExtractionResult:
        grounded_chunks = list(grounded_chunks or [])
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
        phase1_start = time.perf_counter()
        extracted, phase1_usage = self._extract_phase1_phenotypes(
            text=text,
            grounded_chunks=grounded_chunks,
            extraction_prompt=extraction_prompt,
        )
        phase1_elapsed = round(time.perf_counter() - phase1_start, 6)
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
        request_count_total = int(getattr(self.provider, "last_request_count", 0) or 0)
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
            "phase1_requests": request_count_total,
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
                ]
            }
        }
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
            locally_resolved, unresolved = self._resolve_local_matches(
                phrase_candidates
            )
            phase_timings["phase2b_local_seconds"] = round(
                time.perf_counter() - phase2b_local_start, 6
            )
            phase_counts["unresolved_phrases"] = len(unresolved)
            phase_counts["local_matches"] = len(locally_resolved)
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
                request_count_total += mapping_request_count
                phase_request_counts["phase2b_llm_requests"] = mapping_request_count
                resolved_terms.extend(mapped_terms)
                phase_counts["llm_mapped_phrases"] = len(mapped_terms)
                phase_counts["local_fallbacks"] = local_fallback_count
                trace["phase2b_llm"] = {"resolved": mapping_trace}
                logger.debug(
                    "Phase 2B-llm complete: mapped=%d prompt_tokens=%d completion_tokens=%d",
                    len(mapped_terms),
                    mapping_prompt_tokens,
                    mapping_completion_tokens,
                )

        return LLMExtractionResult(
            terms=self._deduplicate_terms(resolved_terms),
            meta=LLMMeta(
                llm_model=config.model,
                llm_mode=config.mode,
                prompt_version=prompt_version,
                token_input=prompt_tokens_total,
                token_output=completion_tokens_total,
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
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        try:
            response = self.provider.run_structured_prompt(
                system_prompt=extraction_prompt.render_system_prompt(),
                user_prompt=extraction_prompt.render_user_prompt(
                    text,
                    chunk_index=self._render_chunk_index(grounded_chunks),
                ),
                response_model=LLMGroundedExtractedPhenotypes,
                max_output_tokens=DEFAULT_PHASE1_MAX_OUTPUT_TOKENS,
            )
        except Exception:
            logger.exception("Phase 1 structured extraction failed")
            raise LLMPipelinePhaseError("phase1", "Structured extraction failed")

        usage = dict(getattr(self.provider, "last_usage", {}) or {})
        parsed: list[dict[str, Any]] = []
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
        return parsed, usage

    def _retrieve_candidates(
        self,
        *,
        actionable: list[dict[str, Any]],
        grounded_chunks: list[dict[str, Any]],
        language: str,
    ) -> list[dict[str, Any]]:
        phrases = [str(item["phrase"]) for item in actionable]
        raw_results = self.tool_executor.query_batch_hpo_terms(
            phrases=phrases,
            language=language,
            n_results=self.n_results_per_phrase,
        )

        results: list[dict[str, Any]] = []
        for index, item in enumerate(actionable):
            phrase = str(item["phrase"])
            category = str(item["category"])
            if index < len(raw_results) and "candidates" in raw_results[index]:
                raw_result = dict(raw_results[index])
                raw_result.setdefault("phrase", phrase)
                raw_result.setdefault("category", category)
                raw_result.setdefault("chunk_ids", list(item.get("chunk_ids", [])))
                raw_result.setdefault("evidence_text", item.get("evidence_text"))
                raw_result.setdefault("start_char", item.get("start_char"))
                raw_result.setdefault("end_char", item.get("end_char"))
                raw_result.setdefault(
                    "grounded_context",
                    self._build_grounded_context(
                        item=item,
                        grounded_chunks=grounded_chunks,
                    ),
                )
                results.append(raw_result)
                continue

            batch_result = raw_results[index] if index < len(raw_results) else {}
            metadatas = self._extract_first_result_list(batch_result, "metadatas")
            similarities = self._extract_first_result_list(batch_result, "similarities")
            results.append(
                {
                    "phrase": phrase,
                    "category": category,
                    "chunk_ids": list(item.get("chunk_ids", [])),
                    "evidence_text": item.get("evidence_text"),
                    "start_char": item.get("start_char"),
                    "end_char": item.get("end_char"),
                    "grounded_context": self._build_grounded_context(
                        item=item,
                        grounded_chunks=grounded_chunks,
                    ),
                    "candidates": self._hybrid_select_candidates(
                        phrase=phrase,
                        metadatas=metadatas,
                        similarities=similarities,
                    ),
                }
            )
            logger.debug(
                "Phase 2A candidate retrieval: phrase=%r candidates=%d",
                phrase,
                len(results[-1]["candidates"]),
            )
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

    def _resolve_local_matches(
        self,
        phrase_candidates: list[dict[str, Any]],
    ) -> tuple[list[LLMPhenotype], list[dict[str, Any]]]:
        resolved: list[LLMPhenotype] = []
        unresolved: list[dict[str, Any]] = []

        for item in phrase_candidates:
            candidates = item.get("candidates", [])
            if not candidates:
                logger.warning(
                    "Phase 2B-local: phrase=%r has no candidates; skipping mapping",
                    item["phrase"],
                )
                continue
            local_match = self._try_local_match(item["phrase"], candidates)
            if local_match is None:
                unresolved.append(item)
                logger.debug(
                    "Phase 2B-local: phrase=%r unresolved_candidates=%d",
                    item["phrase"],
                    len(candidates),
                )
                continue

            resolved.append(
                LLMPhenotype(
                    term_id=local_match["hpo_id"],
                    label=local_match["term_name"],
                    evidence=item["phrase"],
                    assertion=CATEGORY_TO_ASSERTION.get(
                        _normalize_category(str(item.get("category", ""))),
                        PRESENT_ASSERTION,
                    ),
                    category=_normalize_category(str(item.get("category", ""))),
                )
            )
            logger.debug(
                "Phase 2B-local: phrase=%r matched=%s",
                item["phrase"],
                local_match["hpo_id"],
            )
        return resolved, unresolved

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
    ) -> tuple[list[LLMPhenotype], int, int, int, int, list[dict[str, Any]]]:
        resolved: list[LLMPhenotype] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        request_count_total = 0
        local_fallback_count = 0
        mapping_trace: list[dict[str, Any]] = []
        for start in range(0, len(unresolved), self.mapping_batch_size):
            batch = unresolved[start : start + self.mapping_batch_size]
            mapping_response, batch_usage = self._run_mapping_batch(
                batch=batch,
                mapping_prompt=mapping_prompt,
            )
            request_count_total += int(
                getattr(self.provider, "last_request_count", 0) or 0
            )
            prompt_tokens_total += int(batch_usage.get("prompt_tokens", 0) or 0)
            completion_tokens_total += int(batch_usage.get("completion_tokens", 0) or 0)
            selected_ids = self._select_candidate_ids(
                mapping_response=mapping_response,
                batch=batch,
            )
            for item in batch:
                (
                    item_resolved,
                    item_local_fallback_count,
                    item_trace,
                ) = self._resolve_mapping_selection(
                    item=item,
                    selected_id=selected_ids.get(str(item["phrase"])),
                )
                resolved.extend(item_resolved)
                local_fallback_count += item_local_fallback_count
                mapping_trace.append(item_trace)
        return (
            resolved,
            prompt_tokens_total,
            completion_tokens_total,
            request_count_total,
            local_fallback_count,
            mapping_trace,
        )

    def _run_mapping_batch(
        self,
        *,
        batch: list[dict[str, Any]],
        mapping_prompt,
    ) -> tuple[LLMMappingSelection | LLMBatchMappingSelections, dict[str, int]]:
        response_model: type[LLMMappingSelection] | type[LLMBatchMappingSelections]
        if len(batch) == 1:
            item = batch[0]
            normalized_phrase = str(item["phrase"]).lower().replace("-", " ").strip()
            candidate_payload = json.dumps(
                {
                    "phrase": normalized_phrase,
                    "category": item["category"],
                    "grounded_context": item.get("grounded_context", {}),
                    "candidates": [
                        {"id": candidate["hpo_id"], "term": candidate["term_name"]}
                        for candidate in item["candidates"]
                    ],
                },
                ensure_ascii=False,
            )
            response_model = LLMMappingSelection
        else:
            candidate_payload = json.dumps(
                {
                    "items": [
                        {
                            "phrase": str(item["phrase"])
                            .lower()
                            .replace("-", " ")
                            .strip(),
                            "category": item["category"],
                            "grounded_context": item.get("grounded_context", {}),
                            "candidates": [
                                {
                                    "id": candidate["hpo_id"],
                                    "term": candidate["term_name"],
                                }
                                for candidate in item["candidates"]
                            ],
                        }
                        for item in batch
                    ]
                },
                ensure_ascii=False,
            )
            response_model = LLMBatchMappingSelections
        response = self.provider.run_structured_prompt(
            system_prompt=mapping_prompt.render_system_prompt(),
            user_prompt=mapping_prompt.render_user_prompt(candidate_payload),
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
            phenotype = self._phenotype_from_candidate(item=item, candidate=fallback)
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
            phenotype = self._phenotype_from_candidate(item=item, candidate=fallback)
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
        phenotype = self._phenotype_from_candidate(item=item, candidate=candidate)
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
    ) -> LLMPhenotype:
        evidence = LLMPhenotypeEvidence(
            phrase=str(item["phrase"]),
            evidence_text=item.get("evidence_text"),
            chunk_ids=list(item.get("chunk_ids", [])),
            start_char=item.get("start_char"),
            end_char=item.get("end_char"),
            match_method="local",
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
                return {str(item["phrase"]): None}
            return {
                str(item["phrase"]): cls._select_candidate_id(
                    mapping_response=mapping_response,
                    candidates=item["candidates"],
                )
            }

        selected: dict[str, str | None] = {str(item["phrase"]): None for item in batch}
        if not isinstance(mapping_response, LLMBatchMappingSelections):
            return selected

        candidates_by_phrase = {
            str(item["phrase"]): {
                candidate["hpo_id"] for candidate in item["candidates"]
            }
            for item in batch
        }
        original_phrase_by_normalized = {
            _normalize_mapping_phrase_key(str(item["phrase"])): str(item["phrase"])
            for item in batch
        }
        for mapping in mapping_response.mappings:
            phrase = mapping.phrase
            original_phrase = original_phrase_by_normalized.get(
                _normalize_mapping_phrase_key(phrase)
            )
            selected_id = mapping.hpo_id
            if (
                isinstance(selected_id, str)
                and original_phrase in candidates_by_phrase
                and selected_id in candidates_by_phrase[original_phrase]
            ):
                selected[original_phrase] = selected_id
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
