from __future__ import annotations

import re
from typing import Any

from phentrieve.llm.prompts.loader import PromptTemplate

WORD_PATTERN = re.compile(r"\w+")
PHRASE_PARENTHESES_PATTERN = re.compile(r"\(.*?\)")
UNIT_TOKEN_PATTERN = re.compile(
    r"\b(?:mg/dl|mg/dL|mg/l|mg/L|g/dl|g/dL|g/l|g/L|mmol/l|mmol/L|μmol/l|μmol/L|umol/l|umol/L)\b"
)

ACTIONABLE_CATEGORIES = frozenset({"abnormal", "normal", "suspected", "family_history"})
SHARED_HEAD_MODIFIERS = frozenset(
    {
        "brainstem",
        "cerebellar",
        "cerebral",
        "cortical",
        "pontine",
    }
)
SHARED_HEAD_NOUNS = frozenset(
    {
        "agenesis",
        "atrophy",
        "dysplasia",
        "hypoplasia",
        "malformation",
    }
)
SLASH_COMBINED_PHRASE_PATTERN = re.compile(r"^([^/\s][^/]*)/([^/\s][^/]*)$")
SHARED_HEAD_PHRASE_PATTERN = re.compile(
    r"^(?P<first>[a-z][a-z-]*)\s+(?P<second>[a-z][a-z-]*)\s+(?P<head>[a-z][a-z-]*)$",
    re.IGNORECASE,
)
PHENOTYPE_ABBREVIATIONS = {
    "xlid": "X-linked intellectual disability",
}


def normalize_category(category: str) -> str:
    normalized = category.strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "family_history": "family_history",
        "familyhistory": "family_history",
    }.get(normalized, normalized)


def normalize_token(token: str) -> str:
    normalized = token.strip().lower()
    if len(normalized) > 3 and normalized.endswith("s"):
        return normalized[:-1]
    return normalized


def tokenize(text: str) -> set[str]:
    return {
        normalize_token(token)
        for token in WORD_PATTERN.findall(text.lower())
        if normalize_token(token)
    }


def clean_text(text: str) -> str:
    text = PHRASE_PARENTHESES_PATTERN.sub("", text or "")
    text = text.replace("_", " ").replace("-", " ").lower().strip()
    return " ".join(text.split())


def render_phase1_user_prompt(
    *,
    extraction_prompt: PromptTemplate,
    text: str,
    grounded_chunks: list[dict[str, Any]],
    chunk_index_text: str | None = None,
) -> str:
    if chunk_index_text is not None:
        chunk_index = chunk_index_text or "[]"
        return extraction_prompt.render_user_prompt(
            "",
            chunk_index=chunk_index,
        )
    if grounded_chunks:
        chunk_index = (
            "\n".join(
                f"- chunk_id={chunk['chunk_id']}: {chunk.get('text', '')}"
                for chunk in grounded_chunks
            )
            or "[]"
        )
        return extraction_prompt.render_user_prompt(
            "",
            chunk_index=chunk_index,
        )
    return extraction_prompt.render_user_prompt(text, chunk_index="[]")


_CHUNK_INDEX_LINE_PATTERN = re.compile(r"^-?\s*chunk_id=(\d+):\s*(.*)$")


def parse_chunk_index_lines(chunk_index_text: str) -> list[tuple[list[int], str]]:
    rows: list[tuple[list[int], str]] = []
    for raw_line in chunk_index_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _CHUNK_INDEX_LINE_PATTERN.match(line)
        if not match:
            continue
        rows.append(([int(match.group(1))], match.group(2).strip()))
    return rows


def has_unmatched_open_paren(text: str) -> bool:
    return text.count("(") > text.count(")")


def looks_like_uppercase_fragment(text: str) -> bool:
    token = text.strip()
    return bool(token) and token.replace("-", "").replace("/", "").isupper()


def should_merge_chunk_rows(current_text: str, next_text: str) -> bool:
    current = current_text.strip()
    following = next_text.strip()
    if not current or not following:
        return False
    if re.search(r"[.!?]$", current):
        return False
    following_word_count = len(WORD_PATTERN.findall(following))
    continuation_start = bool(re.match(r"^[0-9(\[\]±<>/%-]", following))
    if has_unmatched_open_paren(current) and (
        continuation_start or following_word_count <= 3
    ):
        return True
    if looks_like_uppercase_fragment(current) and following_word_count <= 3:
        return True
    if current[-1].isdigit() and following_word_count <= 3:
        return True
    return continuation_start


def merge_chunk_index_rows(
    rows: list[tuple[list[int], str]],
) -> list[tuple[list[int], str]]:
    if not rows:
        return []
    merged: list[tuple[list[int], str]] = []
    current_ids, current_text = list(rows[0][0]), rows[0][1]
    for next_ids, next_text in rows[1:]:
        if should_merge_chunk_rows(current_text, next_text):
            current_ids.extend(next_ids)
            current_text = f"{current_text.rstrip()} {next_text.lstrip()}".strip()
            continue
        merged.append((current_ids, current_text))
        current_ids, current_text = list(next_ids), next_text
    merged.append((current_ids, current_text))
    return merged


def render_chunk_index_rows(rows: list[tuple[list[int], str]]) -> str:
    rendered_lines: list[str] = []
    for chunk_ids, text in rows:
        chunk_key = "chunk_id" if len(chunk_ids) == 1 else "chunk_ids"
        rendered_lines.append(
            f"{chunk_key}={','.join(str(chunk_id) for chunk_id in chunk_ids)}: {text}"
        )
    return "\n".join(rendered_lines)


def render_group_chunk_index_text(
    *, group: dict[str, Any], grounded_chunks: list[dict[str, Any]]
) -> str:
    group_text = group.get("text")
    raw_chunk_index = (
        group_text
        if isinstance(group_text, str) and group_text.strip()
        else "\n".join(
            f"chunk_id={chunk['chunk_id']}: {chunk.get('text', '')}"
            for chunk in grounded_chunks
        )
    )
    parsed_rows = parse_chunk_index_lines(raw_chunk_index)
    if not parsed_rows:
        return raw_chunk_index
    return render_chunk_index_rows(merge_chunk_index_rows(parsed_rows))


def phase1_extraction_dedup_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("phrase", "")).strip().lower(),
        normalize_category(str(item.get("category", ""))),
        tuple(int(chunk_id) for chunk_id in item.get("chunk_ids", [])),
        item.get("evidence_text"),
        item.get("start_char"),
        item.get("end_char"),
    )


def sorted_chunk_ids(chunk_ids: list[Any]) -> list[int]:
    return sorted({int(chunk_id) for chunk_id in chunk_ids})


def normalized_evidence_text(item: dict[str, Any]) -> str:
    return clean_text(str(item.get("evidence_text") or item.get("phrase") or ""))


def prefer_richer_text(current: str | None, incoming: str | None) -> str | None:
    current_text = str(current or "").strip()
    incoming_text = str(incoming or "").strip()
    if not current_text:
        return incoming_text or None
    if not incoming_text:
        return current_text
    return incoming_text if len(incoming_text) > len(current_text) else current_text


def merge_optional_bounds(
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


def _clone_split_phase1_item(item: dict[str, Any], phrase: str) -> dict[str, Any]:
    split_item = dict(item)
    split_item["phrase"] = phrase
    split_item["evidence_text"] = phrase
    split_item["start_char"] = None
    split_item["end_char"] = None
    return split_item


def _split_slash_combined_phrase(phrase: str) -> list[str]:
    match = SLASH_COMBINED_PHRASE_PATTERN.match(phrase.strip())
    if not match:
        return []
    left = match.group(1).strip()
    right = match.group(2).strip()
    if not left or not right:
        return []
    return [left, right]


def _split_shared_head_phrase(phrase: str) -> list[str]:
    match = SHARED_HEAD_PHRASE_PATTERN.match(clean_text(phrase))
    if not match:
        return []
    first = match.group("first")
    second = match.group("second")
    head = match.group("head")
    if (
        first not in SHARED_HEAD_MODIFIERS
        or second not in SHARED_HEAD_MODIFIERS
        or head not in SHARED_HEAD_NOUNS
    ):
        return []
    return [f"{first} {head}", f"{second} {head}"]


def split_combined_phase1_phrase(phrase: str) -> list[str]:
    """Split clear combined phenotype mentions into standalone source phrases."""
    expanded_abbreviation = PHENOTYPE_ABBREVIATIONS.get(phrase.strip().lower())
    if expanded_abbreviation:
        return [expanded_abbreviation]
    slash_split = _split_slash_combined_phrase(phrase)
    if slash_split:
        return slash_split
    return _split_shared_head_phrase(phrase)


def expand_combined_phase1_extractions(
    extracted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for item in extracted:
        phrase = str(item.get("phrase", "")).strip()
        expanded_abbreviation = PHENOTYPE_ABBREVIATIONS.get(phrase.lower())
        if expanded_abbreviation:
            expanded_item = dict(item)
            expanded_item["phrase"] = expanded_abbreviation
            expanded.append(expanded_item)
            continue
        split_phrases = split_combined_phase1_phrase(phrase)
        if not split_phrases:
            expanded.append(dict(item))
            continue
        expanded.extend(
            _clone_split_phase1_item(item, split_phrase)
            for split_phrase in split_phrases
        )
    return expanded


def spans_overlap(
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
