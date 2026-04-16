from __future__ import annotations

import json
import logging
import re
from typing import Any

from phentrieve.llm.config import (
    DEFAULT_LLM_LANGUAGE,
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
from phentrieve.llm.prompts.loader import get_mapping_prompt, get_prompt
from phentrieve.llm.provider import ToolExecutor
from phentrieve.llm.types import (
    AnnotationMode,
    LLMExtractedPhenotypes,
    LLMExtractionResult,
    LLMMeta,
    LLMPhenotype,
    LLMPipelineConfig,
)
from phentrieve.llm.utils import extract_hpo_id, extract_json, token_sort_similarity

logger = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"\w+")
PHRASE_PARENTHESES_PATTERN = re.compile(r"\(.*?\)")

CATEGORY_TO_ASSERTION = {
    "abnormal": PRESENT_ASSERTION,
    "normal": NEGATED_ASSERTION,
    "suspected": UNCERTAIN_ASSERTION,
}

ACTIONABLE_CATEGORIES = frozenset({"abnormal"})


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
    ) -> None:
        self.provider = provider
        self.tool_executor = tool_executor or ToolExecutor()
        self.n_results_per_phrase = n_results_per_phrase
        self.similarity_threshold = similarity_threshold
        self.min_unique_candidates = min_unique_candidates
        self.max_unique_candidates = max_unique_candidates
        self.local_match_threshold = local_match_threshold

    def run(self, *, text: str, config: LLMPipelineConfig) -> LLMExtractionResult:
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
        extracted, phase1_usage = self._extract_phase1_phenotypes(
            text=text,
            extraction_prompt=extraction_prompt,
        )
        logger.debug(
            "Phase 1 complete: extracted=%d prompt_tokens=%s completion_tokens=%s",
            len(extracted),
            phase1_usage.get("prompt_tokens"),
            phase1_usage.get("completion_tokens"),
        )
        actionable = [
            (phrase, category)
            for phrase, category in extracted
            if category.lower() in ACTIONABLE_CATEGORIES
        ]

        resolved_terms: list[LLMPhenotype] = []
        prompt_tokens_total = int(phase1_usage.get("prompt_tokens", 0) or 0)
        completion_tokens_total = int(phase1_usage.get("completion_tokens", 0) or 0)
        prompt_version = extraction_prompt.version
        if actionable:
            logger.debug(
                "Phase 2A: retrieving candidates for actionable_phrases=%d",
                len(actionable),
            )
            phrase_candidates = self._retrieve_candidates(
                actionable=actionable,
                text=text,
                language=language,
            )
            logger.debug(
                "Phase 2A complete: candidate_sets=%d",
                len(phrase_candidates),
            )
            logger.debug(
                "Phase 2B-local: resolving local matches for candidate_sets=%d",
                len(phrase_candidates),
            )
            locally_resolved, unresolved = self._resolve_local_matches(
                phrase_candidates
            )
            resolved_terms.extend(locally_resolved)
            logger.debug(
                "Phase 2B-local complete: matched=%d unresolved=%d",
                len(locally_resolved),
                len(unresolved),
            )
            if unresolved:
                mapping_prompt = get_mapping_prompt(language)
                prompt_version = f"{prompt_version}+{mapping_prompt.version}"
                logger.debug(
                    "Phase 2B-llm: mapping unresolved phrases=%d",
                    len(unresolved),
                )
                mapped_terms, mapping_prompt_tokens, mapping_completion_tokens = (
                    self._resolve_with_mapping_prompt(
                        unresolved=unresolved,
                        mapping_prompt=mapping_prompt,
                    )
                )
                prompt_tokens_total += mapping_prompt_tokens
                completion_tokens_total += mapping_completion_tokens
                resolved_terms.extend(mapped_terms)
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
            ),
        )

    @staticmethod
    def _parse_phase1_response(response_text: str) -> list[tuple[str, str]]:
        json_data = extract_json(response_text)
        if not json_data:
            logger.warning(
                "Phase 1 response could not be parsed as JSON; treating extraction as empty."
            )
            return []

        phenotypes = json_data.get("phenotypes")
        if not isinstance(phenotypes, list):
            logger.warning(
                "Phase 1 response missing a valid phenotypes list; treating extraction as empty."
            )
            return []

        parsed: list[tuple[str, str]] = []
        for phenotype in phenotypes:
            if not isinstance(phenotype, dict):
                continue
            phrase = str(phenotype.get("phrase", "")).strip()
            category = str(phenotype.get("category", "")).strip()
            if phrase and category:
                parsed.append((phrase, category))
        return parsed

    def _extract_phase1_phenotypes(
        self,
        *,
        text: str,
        extraction_prompt,
    ) -> tuple[list[tuple[str, str]], dict[str, int]]:
        try:
            response = self.provider.run_structured_prompt(
                system_prompt=extraction_prompt.render_system_prompt(),
                user_prompt=extraction_prompt.render_user_prompt(text),
                response_model=LLMExtractedPhenotypes,
                max_output_tokens=DEFAULT_PHASE1_MAX_OUTPUT_TOKENS,
            )
        except Exception:
            logger.warning(
                "Phase 1 structured extraction failed; treating extraction as empty "
                "(finish_reason=%s).",
                getattr(self.provider, "last_finish_reason", None),
                exc_info=True,
            )
            return [], {}

        usage = dict(getattr(self.provider, "last_usage", {}) or {})
        parsed: list[tuple[str, str]] = []
        for phenotype in response.phenotypes:
            phrase = phenotype.phrase.strip()
            category = phenotype.category.strip()
            if phrase and category:
                parsed.append((phrase, category))
        return parsed, usage

    def _retrieve_candidates(
        self,
        *,
        actionable: list[tuple[str, str]],
        text: str,
        language: str,
    ) -> list[dict[str, Any]]:
        phrases = [phrase for phrase, _category in actionable]
        raw_results = self.tool_executor.query_batch_hpo_terms(
            phrases=phrases,
            language=language,
            n_results=self.n_results_per_phrase,
        )

        results: list[dict[str, Any]] = []
        for index, (phrase, category) in enumerate(actionable):
            if index < len(raw_results) and "candidates" in raw_results[index]:
                raw_result = dict(raw_results[index])
                raw_result.setdefault("phrase", phrase)
                raw_result.setdefault("category", category)
                raw_result.setdefault(
                    "original_sentence",
                    self._find_original_sentence(phrase, text),
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
                    "original_sentence": self._find_original_sentence(phrase, text),
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
    def _find_original_sentence(phrase: str, text: str) -> str:
        sentences = [
            sentence.strip() for sentence in text.split(".") if sentence.strip()
        ]
        if not sentences:
            return phrase

        phrase_tokens = _tokenize(phrase)
        best_sentence = phrase
        best_overlap = 0
        for sentence in sentences:
            overlap = len(phrase_tokens & _tokenize(sentence))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence
        return best_sentence

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
                        str(item.get("category", "")).lower(),
                        PRESENT_ASSERTION,
                    ),
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
    ) -> tuple[list[LLMPhenotype], int, int]:
        resolved: list[LLMPhenotype] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        for item in unresolved:
            normalized_phrase = str(item["phrase"]).lower().replace("-", " ").strip()
            candidate_payload = json.dumps(
                {
                    "phrase": normalized_phrase,
                    "category": item["category"],
                    "original_sentence": item["original_sentence"],
                    "candidates": [
                        {"id": candidate["hpo_id"], "term": candidate["term_name"]}
                        for candidate in item["candidates"]
                    ],
                },
                ensure_ascii=False,
            )
            response = self.provider.complete(
                mapping_prompt.get_messages(candidate_payload, include_examples=True)
            )
            prompt_tokens_total += int(response.usage.get("prompt_tokens", 0) or 0)
            completion_tokens_total += int(
                response.usage.get("completion_tokens", 0) or 0
            )
            selected_id = self._select_candidate_id(
                response_text=response.content or "",
                candidates=item["candidates"],
            )
            if not selected_id:
                logger.warning(
                    "Phase 2B-llm: phrase=%r produced no valid mapping; trying local fallback",
                    item["phrase"],
                )
                fallback = self._try_local_match(item["phrase"], item["candidates"])
                if fallback is not None:
                    resolved.append(
                        LLMPhenotype(
                            term_id=fallback["hpo_id"],
                            label=fallback["term_name"],
                            evidence=item["phrase"],
                            assertion=CATEGORY_TO_ASSERTION.get(
                                str(item.get("category", "")).lower(),
                                PRESENT_ASSERTION,
                            ),
                        )
                    )
                    logger.debug(
                        "Phase 2B-llm: phrase=%r local fallback resolved=%s",
                        item["phrase"],
                        fallback["hpo_id"],
                    )
                continue
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
                if fallback is not None:
                    resolved.append(
                        LLMPhenotype(
                            term_id=fallback["hpo_id"],
                            label=fallback["term_name"],
                            evidence=item["phrase"],
                            assertion=CATEGORY_TO_ASSERTION.get(
                                str(item.get("category", "")).lower(),
                                PRESENT_ASSERTION,
                            ),
                        )
                    )
                    logger.debug(
                        "Phase 2B-llm: phrase=%r local fallback resolved=%s",
                        item["phrase"],
                        fallback["hpo_id"],
                    )
                continue
            resolved.append(
                LLMPhenotype(
                    term_id=candidate["hpo_id"],
                    label=candidate["term_name"],
                    evidence=item["phrase"],
                    assertion=CATEGORY_TO_ASSERTION.get(
                        str(item.get("category", "")).lower(),
                        PRESENT_ASSERTION,
                    ),
                )
            )
            logger.debug(
                "Phase 2B-llm: phrase=%r mapped=%s",
                item["phrase"],
                selected_id,
            )
        return resolved, prompt_tokens_total, completion_tokens_total

    @staticmethod
    def _select_candidate_id(
        *,
        response_text: str,
        candidates: list[dict[str, Any]],
    ) -> str | None:
        candidate_ids = {candidate["hpo_id"] for candidate in candidates}
        json_data = extract_json(response_text)
        if json_data:
            for key in ("hpo_id", "id", "HPO_ID"):
                selected = json_data.get(key)
                if isinstance(selected, str) and selected in candidate_ids:
                    return selected

        fallback_id = extract_hpo_id(response_text)
        if fallback_id in candidate_ids:
            return fallback_id
        return None

    @staticmethod
    def _deduplicate_terms(terms: list[LLMPhenotype]) -> list[LLMPhenotype]:
        deduplicated: dict[str, LLMPhenotype] = {}
        for term in terms:
            deduplicated.setdefault(term.term_id, term)
        return list(deduplicated.values())
