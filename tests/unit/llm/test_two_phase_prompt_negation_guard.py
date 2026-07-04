"""Guard test: extraction two_phase prompts must keep the negation-scope rule.

This test protects the LLM extraction contract from a silent regression where
a future localized (or edited) `two_phase` extraction prompt drops either:

  1. the negation-scope rule that tells the model a negation cue (e.g. "X
     without Y") negates only the noun phrase it directly modifies, not the
     whole sentence, and
  2. a `negated_qualifier` few-shot example that demonstrates the rule with a
     real (non-null) value.

Only true *extraction* templates are checked -- the `*_mapping.yaml` and
`*_mapping_batch.yaml` templates are a distinct prompt family (mapping
extracted phrases to HPO terms) and never contained this rule, so they are
excluded by name.
"""

from __future__ import annotations

import re
from pathlib import Path

from phentrieve.llm.prompts.loader import PACKAGE_TEMPLATES_DIR

TWO_PHASE_TEMPLATES_DIR = PACKAGE_TEMPLATES_DIR / "two_phase"

# Excluded suffixes: mapping-phase templates, not extraction-phase templates.
_MAPPING_SUFFIXES = ("_mapping.yaml", "_mapping_batch.yaml")

# Distinctive negation-scope rule text (en.yaml:28-30). Robust to minor
# reformatting but specific enough that a template lacking the guidance
# would fail the check.
_NEGATION_SCOPE_MARKERS = (
    "negates ONLY",
    "X without Y",
)

# A negated_qualifier few-shot must show a real (non-null, non-empty) value,
# not just "negated_qualifier": null.
_NEGATED_QUALIFIER_VALUE_PATTERN = re.compile(r'"negated_qualifier"\s*:\s*"[^"]+"')


def _is_extraction_template(path: Path) -> bool:
    return not path.name.endswith(_MAPPING_SUFFIXES)


def _extraction_templates() -> list[Path]:
    return sorted(
        path
        for path in TWO_PHASE_TEMPLATES_DIR.glob("*.yaml")
        if _is_extraction_template(path)
    )


def test_extraction_template_selection_is_nonempty_and_excludes_mapping() -> None:
    templates = _extraction_templates()

    assert templates, (
        "No extraction two_phase templates found -- the glob or exclusion "
        "filter is likely broken, which would make this guard vacuous."
    )
    for path in templates:
        assert not path.name.endswith(_MAPPING_SUFFIXES)


def test_extraction_templates_keep_negation_scope_rule() -> None:
    templates = _extraction_templates()
    assert templates, "expected at least one extraction two_phase template"

    for path in templates:
        text = path.read_text(encoding="utf-8")
        for marker in _NEGATION_SCOPE_MARKERS:
            assert marker in text, (
                f"{path.name} is missing negation-scope guidance "
                f"(expected marker {marker!r})"
            )


def test_extraction_templates_keep_negated_qualifier_few_shot() -> None:
    templates = _extraction_templates()
    assert templates, "expected at least one extraction two_phase template"

    for path in templates:
        text = path.read_text(encoding="utf-8")
        assert _NEGATED_QUALIFIER_VALUE_PATTERN.search(text), (
            f"{path.name} is missing a negated_qualifier few-shot with a non-null value"
        )
