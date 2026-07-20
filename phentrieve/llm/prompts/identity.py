from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from phentrieve.llm.config import DEFAULT_LLM_LANGUAGE
from phentrieve.llm.prompts.loader import (
    MAPPING_BATCH_VARIANT,
    resolve_prompt_template,
)

PROMPT_IDENTITY_SCHEMA = "phentrieve-prompt-bundle/v1"
PROMPT_BEHAVIOR_VERSION = "1"

_DOCUMENT_SENTINEL = "<PHENTRIEVE_DOCUMENT_SENTINEL>"
_CHUNK_INDEX_SENTINEL = "<PHENTRIEVE_CHUNK_INDEX_SENTINEL>"
_TWO_PHASE_COMPONENTS = (
    ("extraction", None),
    ("mapping", "mapping"),
    ("batch_mapping", MAPPING_BATCH_VARIANT),
)


@dataclass(frozen=True)
class PromptComponentIdentity:
    name: str
    version: str
    sha256: str


@dataclass(frozen=True)
class PromptBundleIdentity:
    schema_version: str
    mode: str
    language: str
    prompt_behavior_version: str
    components: tuple[PromptComponentIdentity, ...]
    sha256: str


def _normalize_newlines(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(value, list):
        return [_normalize_newlines(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_newlines(item) for key, item in value.items()}
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize_newlines(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def build_prompt_bundle_identity(
    mode: str,
    language: str,
    prompt_dir: Path | None = None,
) -> PromptBundleIdentity:
    if mode != "two_phase":
        raise ValueError(f"Unsupported prompt bundle mode: {mode!r}")

    components: list[PromptComponentIdentity] = []
    for component_name, variant in _TWO_PHASE_COMPONENTS:
        resolved_language = (
            language if component_name == "extraction" else DEFAULT_LLM_LANGUAGE
        )
        template = resolve_prompt_template(
            mode, resolved_language, variant=variant, prompt_dir=prompt_dir
        )
        if component_name != "extraction":
            template = replace(template, language=language)
        emitted_examples = [
            {"input": example.get("input", ""), "output": example.get("output", "")}
            for example in template.few_shot_examples
        ]
        if component_name == "extraction":
            resolved_system_template = template.render_system_prompt()
            resolved_user_template = template.render_user_prompt(
                _DOCUMENT_SENTINEL,
                chunk_index=_CHUNK_INDEX_SENTINEL,
            )
        else:
            resolved_system_template = template.render_system_prompt(language=language)
            resolved_user_template = template.render_user_prompt(
                _DOCUMENT_SENTINEL, language=language
            )
        effective_content = {
            "component_name": component_name,
            "component_version": template.version,
            "few_shot_examples": emitted_examples,
            "language": language,
            "mode": mode,
            "resolved_system_template": resolved_system_template,
            "resolved_user_template": resolved_user_template,
        }
        components.append(
            PromptComponentIdentity(
                name=component_name,
                version=template.version,
                sha256=_sha256(effective_content),
            )
        )

    component_tuple = tuple(components)
    bundle_content = {
        "components": [
            {"name": item.name, "version": item.version, "sha256": item.sha256}
            for item in component_tuple
        ],
        "language": language,
        "mode": mode,
        "prompt_behavior_version": PROMPT_BEHAVIOR_VERSION,
        "schema_version": PROMPT_IDENTITY_SCHEMA,
    }
    return PromptBundleIdentity(
        schema_version=PROMPT_IDENTITY_SCHEMA,
        mode=mode,
        language=language,
        prompt_behavior_version=PROMPT_BEHAVIOR_VERSION,
        components=component_tuple,
        sha256=_sha256(bundle_content),
    )
