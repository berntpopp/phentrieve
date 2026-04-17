from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from phentrieve.llm.config import (
    DEFAULT_LLM_LANGUAGE,
    DEFAULT_TOOL_QUERY_RESULTS,
    PROMPT_VARIANT_MAPPING,
)
from phentrieve.llm.types import AnnotationMode, PostProcessingStep

logger = logging.getLogger(__name__)

PACKAGE_TEMPLATES_DIR = Path(__file__).parent / "templates"
USER_TEMPLATES_DIR = Path.home() / ".phentrieve" / "prompts"

TOOL_TERM_VARIANT = "term_search"
TOOL_TEXT_VARIANT = "text_process"
MAPPING_BATCH_VARIANT = "mapping_batch"
PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass(slots=True)
class PromptTemplate:
    system_prompt: str
    user_prompt_template: str
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    version: str = "v1"
    mode: str = ""
    language: str = DEFAULT_LLM_LANGUAGE
    source_path: str = ""

    def _render_prompt_text(self, template: str, **kwargs: Any) -> str:
        prompt_kwargs = {"tool_query_results": DEFAULT_TOOL_QUERY_RESULTS, **kwargs}
        return PLACEHOLDER_PATTERN.sub(
            lambda match: str(prompt_kwargs.get(match.group(1), match.group(0))),
            template,
        )

    def render_system_prompt(self, **kwargs: Any) -> str:
        return self._render_prompt_text(self.system_prompt, **kwargs)

    def render_user_prompt(self, text: str, **kwargs: Any) -> str:
        return self._render_prompt_text(self.user_prompt_template, text=text, **kwargs)

    def get_messages(
        self,
        text: str,
        *,
        include_examples: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.render_system_prompt(**kwargs)}]

        if include_examples:
            for example in self.few_shot_examples:
                messages.append({"role": "user", "content": example.get("input", "")})
                messages.append(
                    {"role": "assistant", "content": example.get("output", "")}
                )

        messages.append(
            {"role": "user", "content": self.render_user_prompt(text, **kwargs)}
        )
        return messages


def _mode_to_dir(mode: AnnotationMode | PostProcessingStep | str) -> str:
    if isinstance(mode, AnnotationMode):
        if mode == AnnotationMode.DIRECT:
            return "direct_text"
        if mode in {AnnotationMode.TOOL_TERM, AnnotationMode.TOOL_TEXT}:
            return "tool_guided"
        if mode == AnnotationMode.TWO_PHASE:
            return "two_phase"
    if isinstance(mode, PostProcessingStep):
        return "postprocess"
    return str(mode)


def _build_search_paths(
    *,
    mode_dir: str,
    language: str,
    variant: str | None,
) -> list[tuple[Path, str]]:
    filename = f"{language}_{variant}.yaml" if variant else f"{language}.yaml"
    paths: list[tuple[Path, str]] = [
        (USER_TEMPLATES_DIR / mode_dir / filename, language),
        (PACKAGE_TEMPLATES_DIR / mode_dir / filename, language),
    ]

    if language != DEFAULT_LLM_LANGUAGE:
        fallback_filename = (
            f"{DEFAULT_LLM_LANGUAGE}_{variant}.yaml"
            if variant
            else f"{DEFAULT_LLM_LANGUAGE}.yaml"
        )
        paths.extend(
            [
                (
                    USER_TEMPLATES_DIR / mode_dir / fallback_filename,
                    DEFAULT_LLM_LANGUAGE,
                ),
                (
                    PACKAGE_TEMPLATES_DIR / mode_dir / fallback_filename,
                    DEFAULT_LLM_LANGUAGE,
                ),
            ]
        )
    return paths


@lru_cache(maxsize=64)
def load_prompt_template(
    mode: AnnotationMode | PostProcessingStep | str,
    language: str = DEFAULT_LLM_LANGUAGE,
    variant: str | None = None,
) -> PromptTemplate:
    if variant is None and isinstance(mode, PostProcessingStep):
        variant = mode.value

    mode_dir = _mode_to_dir(mode)
    search_paths = _build_search_paths(
        mode_dir=mode_dir,
        language=language,
        variant=variant,
    )

    for path, resolved_language in search_paths:
        if path.exists():
            logger.debug("Loading prompt template from %s", path)
            return _load_yaml_template(
                path=path,
                mode=mode_dir,
                requested_language=language,
                resolved_language=resolved_language,
            )

    searched = ", ".join(str(path) for path, _language in search_paths)
    raise FileNotFoundError(
        f"No prompt template found for mode={mode_dir!r}, language={language!r}. "
        f"Searched: {searched}"
    )


def _load_yaml_template(
    *,
    path: Path,
    mode: str,
    requested_language: str,
    resolved_language: str,
) -> PromptTemplate:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return PromptTemplate(
        system_prompt=str(data.get("system_prompt", "")),
        user_prompt_template=str(data.get("user_prompt_template", "{text}")),
        few_shot_examples=list(data.get("few_shot_examples", [])),
        version=str(data.get("version", "v1")),
        mode=mode,
        language=resolved_language,
        source_path=str(path.resolve()),
    )


def get_prompt(
    mode: AnnotationMode | str,
    language: str = DEFAULT_LLM_LANGUAGE,
) -> PromptTemplate:
    if isinstance(mode, str):
        mode = AnnotationMode(mode)

    if mode == AnnotationMode.TOOL_TERM:
        return load_prompt_template(mode, language, variant=TOOL_TERM_VARIANT)
    if mode == AnnotationMode.TOOL_TEXT:
        return load_prompt_template(mode, language, variant=TOOL_TEXT_VARIANT)
    return load_prompt_template(mode, language)


def get_mapping_prompt(language: str = DEFAULT_LLM_LANGUAGE) -> PromptTemplate:
    template = load_prompt_template(
        AnnotationMode.TWO_PHASE,
        DEFAULT_LLM_LANGUAGE,
        variant=PROMPT_VARIANT_MAPPING,
    )
    return replace(template, language=language)


def get_batch_mapping_prompt(language: str = DEFAULT_LLM_LANGUAGE) -> PromptTemplate:
    template = load_prompt_template(
        AnnotationMode.TWO_PHASE,
        DEFAULT_LLM_LANGUAGE,
        variant=MAPPING_BATCH_VARIANT,
    )
    return replace(template, language=language)


def list_available_prompts() -> dict[str, list[str]]:
    available: dict[str, list[str]] = {}
    active_mode_dirs = {"direct_text", "tool_guided", "two_phase", "postprocess"}

    for templates_dir in [PACKAGE_TEMPLATES_DIR, USER_TEMPLATES_DIR]:
        if not templates_dir.exists():
            continue

        for mode_dir in templates_dir.iterdir():
            if not mode_dir.is_dir() or mode_dir.name not in active_mode_dirs:
                continue

            mode_name = mode_dir.name
            if mode_name not in available:
                available[mode_name] = []

            for yaml_file in mode_dir.glob("*.yaml"):
                language = yaml_file.stem.split("_")[0]
                if language not in available[mode_name]:
                    available[mode_name].append(language)

    return available
