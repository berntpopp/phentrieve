"""
Prompt template loading and rendering.

This module provides functions to load prompt templates from YAML files,
with support for user overrides and language-specific variants.

Template search order:
1. User override: ~/.phentrieve/prompts/{mode}/{language}.yaml
2. Package default: phentrieve/llm/prompts/templates/{mode}/{language}.yaml
3. Fallback to English if language-specific template not found
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from phentrieve.llm.types import AnnotationMode, PostProcessingStep

logger = logging.getLogger(__name__)

# Package templates directory
PACKAGE_TEMPLATES_DIR = Path(__file__).parent / "templates"

# User override directory
USER_TEMPLATES_DIR = Path.home() / ".phentrieve" / "prompts"


@dataclass
class PromptTemplate:
    """
    A loaded prompt template with metadata.

    Attributes:
        system_prompt: The system prompt instructing the LLM.
        user_prompt_template: Template for the user message (with {text} placeholder).
        few_shot_examples: Optional list of example input/output pairs.
        version: Template version for reproducibility tracking.
        mode: The annotation mode this template is for.
        language: The language code (e.g., "en", "de").
        source_path: Where this template was loaded from.
    """

    system_prompt: str
    user_prompt_template: str
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    version: str = "v1.0.0"
    mode: str = ""
    language: str = "en"
    source_path: str = ""

    def render_user_prompt(self, text: str, **kwargs: Any) -> str:
        """
        Render the user prompt template with the given text.

        Args:
            text: The clinical text to annotate.
            **kwargs: Additional template variables.

        Returns:
            The rendered user prompt string.
        """
        return self.user_prompt_template.format(text=text, **kwargs)

    def get_messages(
        self,
        text: str,
        include_examples: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """
        Build the complete message list for the LLM.

        Args:
            text: The clinical text to annotate.
            include_examples: Whether to include few-shot examples.
            **kwargs: Additional template variables.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add few-shot examples if available and requested
        if include_examples and self.few_shot_examples:
            for example in self.few_shot_examples:
                messages.append({"role": "user", "content": example.get("input", "")})
                messages.append(
                    {"role": "assistant", "content": example.get("output", "")}
                )

        # Add the actual user request
        messages.append(
            {"role": "user", "content": self.render_user_prompt(text, **kwargs)}
        )

        return messages


def _mode_to_dir(mode: AnnotationMode | PostProcessingStep | str) -> str:
    """Convert mode enum to directory name."""
    if isinstance(mode, AnnotationMode):
        if mode == AnnotationMode.DIRECT:
            return "direct_text"
        elif mode == AnnotationMode.TOOL_TERM:
            return "tool_guided"
        elif mode == AnnotationMode.TOOL_TEXT:
            return "tool_guided"
    elif isinstance(mode, PostProcessingStep):
        return "postprocess"
    return str(mode)


def load_prompt_template(
    mode: AnnotationMode | PostProcessingStep | str,
    language: str = "en",
    variant: str | None = None,
) -> PromptTemplate:
    """
    Load a prompt template for the given mode and language.

    Args:
        mode: The annotation mode or post-processing step.
        language: Language code (e.g., "en", "de").
        variant: Optional variant name (e.g., "term_search" for tool_guided mode).

    Returns:
        The loaded PromptTemplate.

    Raises:
        FileNotFoundError: If no template is found for the mode/language.
    """
    mode_dir = _mode_to_dir(mode)

    # Auto-derive variant from PostProcessingStep value (e.g., "validation", "refinement")
    if variant is None and isinstance(mode, PostProcessingStep):
        variant = mode.value

    # Build filename based on variant
    if variant:
        filename = f"{language}_{variant}.yaml"
    else:
        filename = f"{language}.yaml"

    # Search for template
    search_paths = [
        USER_TEMPLATES_DIR / mode_dir / filename,
        PACKAGE_TEMPLATES_DIR / mode_dir / filename,
    ]

    # Fallback to English if language-specific not found
    if language != "en":
        if variant:
            search_paths.extend(
                [
                    USER_TEMPLATES_DIR / mode_dir / f"en_{variant}.yaml",
                    PACKAGE_TEMPLATES_DIR / mode_dir / f"en_{variant}.yaml",
                ]
            )
        else:
            search_paths.extend(
                [
                    USER_TEMPLATES_DIR / mode_dir / "en.yaml",
                    PACKAGE_TEMPLATES_DIR / mode_dir / "en.yaml",
                ]
            )

    for path in search_paths:
        if path.exists():
            logger.debug("Loading prompt template from: %s", path)
            return _load_yaml_template(path, mode_dir, language)

    raise FileNotFoundError(
        f"No prompt template found for mode='{mode_dir}', language='{language}'. "
        f"Searched: {[str(p) for p in search_paths]}"
    )


def _load_yaml_template(path: Path, mode: str, language: str) -> PromptTemplate:
    """Load and parse a YAML template file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return PromptTemplate(
        system_prompt=data.get("system_prompt", ""),
        user_prompt_template=data.get("user_prompt_template", "{text}"),
        few_shot_examples=data.get("few_shot_examples", []),
        version=data.get("version", "v1.0.0"),
        mode=mode,
        language=language,
        source_path=str(path),
    )


def get_prompt(
    mode: AnnotationMode | str,
    language: str = "en",
) -> PromptTemplate:
    """
    Convenience function to get a prompt template.

    This is the primary entry point for getting prompts. It handles
    mode-specific variants automatically.

    Args:
        mode: The annotation mode.
        language: Language code.

    Returns:
        The appropriate PromptTemplate.
    """
    if isinstance(mode, str):
        mode = AnnotationMode(mode)

    # Tool-guided modes have variants
    if mode == AnnotationMode.TOOL_TERM:
        return load_prompt_template(mode, language, variant="term_search")
    elif mode == AnnotationMode.TOOL_TEXT:
        return load_prompt_template(mode, language, variant="text_process")
    else:
        return load_prompt_template(mode, language)


def list_available_prompts() -> dict[str, list[str]]:
    """
    List all available prompt templates.

    Returns:
        Dict mapping mode names to lists of available languages.
    """
    available: dict[str, list[str]] = {}

    for templates_dir in [PACKAGE_TEMPLATES_DIR, USER_TEMPLATES_DIR]:
        if not templates_dir.exists():
            continue

        for mode_dir in templates_dir.iterdir():
            if not mode_dir.is_dir():
                continue

            mode_name = mode_dir.name
            if mode_name not in available:
                available[mode_name] = []

            for yaml_file in mode_dir.glob("*.yaml"):
                # Extract language from filename
                lang = yaml_file.stem.split("_")[0]
                if lang not in available[mode_name]:
                    available[mode_name].append(lang)

    return available
