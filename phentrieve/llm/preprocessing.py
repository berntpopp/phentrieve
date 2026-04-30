from __future__ import annotations

from typing import Any, Protocol

from phentrieve.llm.types import ExtractionGroup, GroundedChunk


class _TokenCountingProvider(Protocol):
    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        pass


def _normalize_status(status: Any) -> str:
    if hasattr(status, "value"):
        return str(status.value)
    if status is None:
        return "unknown"
    return str(status)


def _status_with_left_context(
    *,
    text: str,
    chunk: dict[str, Any],
    assertion_detector: Any,
    left_context_chars: int = 64,
) -> str:
    chunk_status = _normalize_status(chunk.get("status"))
    if chunk_status != "affirmed":
        return chunk_status

    start_char = chunk.get("start_char")
    end_char = chunk.get("end_char")
    if not isinstance(start_char, int) or not isinstance(end_char, int):
        return chunk_status

    context_start = max(0, start_char - left_context_chars)
    context_text = text[context_start:end_char]
    if context_text.strip() == str(chunk.get("text", "")).strip():
        return chunk_status

    try:
        context_status, _context_details = assertion_detector.detect(context_text)
    except Exception:
        return chunk_status

    normalized_context_status = _normalize_status(context_status)
    if normalized_context_status in {"negated", "normal"}:
        return normalized_context_status
    return chunk_status


def _render_group_text(chunks: list[GroundedChunk]) -> str:
    return "\n".join(f"chunk_id={chunk.chunk_id}: {chunk.text}" for chunk in chunks)


def build_grounded_chunks_from_text_pipeline(
    *,
    text: str,
    language: str,
    chunking_pipeline_config: list[dict[str, Any]] | None,
    assertion_config: dict[str, Any] | None,
    retrieval_model_name: str,
    include_positions: bool = True,
) -> list[GroundedChunk]:
    from phentrieve.config import get_default_chunk_pipeline_config
    from phentrieve.embeddings import load_embedding_model
    from phentrieve.text_processing.pipeline import TextProcessingPipeline

    text_pipeline = TextProcessingPipeline(
        language=language,
        chunking_pipeline_config=(
            get_default_chunk_pipeline_config()
            if chunking_pipeline_config is None
            else chunking_pipeline_config
        ),
        assertion_config=(
            {"disable": True} if assertion_config is None else assertion_config
        ),
        sbert_model_for_semantic_chunking=load_embedding_model(retrieval_model_name),
    )
    processed_chunks = text_pipeline.process(text, include_positions=include_positions)
    return [
        GroundedChunk(
            chunk_id=index + 1,
            text=str(chunk.get("text", "")),
            start_char=chunk.get("start_char"),
            end_char=chunk.get("end_char"),
            status=_status_with_left_context(
                text=text,
                chunk=chunk,
                assertion_detector=text_pipeline.assertion_detector,
            ),
        )
        for index, chunk in enumerate(processed_chunks)
    ]


def build_extraction_groups(
    *,
    grounded_chunks: list[GroundedChunk],
    provider: _TokenCountingProvider,
    system_prompt: str,
    max_prompt_tokens: int,
    neighbor_overlap: int = 1,
) -> list[ExtractionGroup]:
    if not grounded_chunks:
        return []

    ordered_chunks = list(grounded_chunks)
    groups: list[ExtractionGroup] = []
    start_index = 0
    overlap = max(int(neighbor_overlap), 0)
    prompt_token_cache: dict[tuple[int, int], int] = {}

    def _prompt_tokens_for_window(window_start: int, window_end: int) -> int:
        cache_key = (window_start, window_end)
        cached_tokens = prompt_token_cache.get(cache_key)
        if cached_tokens is not None:
            return cached_tokens

        candidate_chunks = ordered_chunks[window_start : window_end + 1]
        candidate_text = _render_group_text(candidate_chunks)
        token_counts = provider.count_tokens(
            system_prompt=system_prompt,
            user_prompt=candidate_text,
        )
        prompt_tokens = int(
            token_counts.get("prompt_tokens") or token_counts.get("total_tokens") or 0
        )
        prompt_token_cache[cache_key] = prompt_tokens
        return prompt_tokens

    while start_index < len(ordered_chunks):
        single_chunk_tokens = _prompt_tokens_for_window(start_index, start_index)
        if single_chunk_tokens > max_prompt_tokens:
            chunk = ordered_chunks[start_index]
            raise ValueError(
                "Single grounded chunk exceeds max_prompt_tokens "
                f"(chunk_id={chunk.chunk_id}, prompt_tokens={single_chunk_tokens}, "
                f"max_prompt_tokens={max_prompt_tokens})"
            )

        low = start_index
        high = len(ordered_chunks) - 1
        best_end = start_index
        best_tokens = single_chunk_tokens

        while low <= high:
            candidate_end = (low + high) // 2
            prompt_tokens = _prompt_tokens_for_window(start_index, candidate_end)
            if prompt_tokens <= max_prompt_tokens:
                best_end = candidate_end
                best_tokens = prompt_tokens
                low = candidate_end + 1
            else:
                high = candidate_end - 1

        group_chunks = ordered_chunks[start_index : best_end + 1]
        groups.append(
            ExtractionGroup(
                group_id=len(groups) + 1,
                chunk_ids=[chunk.chunk_id for chunk in group_chunks],
                text=_render_group_text(group_chunks),
                estimated_prompt_tokens=best_tokens,
            )
        )

        if best_end >= len(ordered_chunks) - 1:
            break

        next_start = max(best_end + 1 - overlap, start_index + 1)
        start_index = next_start

    return groups
