"""Consistency checks for packaging metadata and setup documentation."""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
MAKEFILE = REPO_ROOT / "Makefile"
DOC_FILES = [
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "docs/development/dev-environment.md",
    REPO_ROOT / "api/README.md",
]


def _optional_dependency_names() -> set[str]:
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return set(metadata["project"]["optional-dependencies"])


def test_makefile_uv_sync_extras_exist_in_pyproject() -> None:
    extras = _optional_dependency_names()
    makefile = MAKEFILE.read_text(encoding="utf-8")

    referenced_extras = set(re.findall(r"uv sync --extra ([A-Za-z0-9_-]+)", makefile))

    assert referenced_extras <= extras


def test_setup_docs_do_not_reference_removed_optional_extras() -> None:
    stale_extra_patterns = {
        "text": re.compile(r"uv sync --extra text\b"),
        "text_processing": re.compile(r"\btext_processing\b"),
    }
    violations: list[str] = []

    for path in DOC_FILES:
        content = path.read_text(encoding="utf-8")
        for stale_extra, pattern in stale_extra_patterns.items():
            if pattern.search(content):
                rel_path = path.relative_to(REPO_ROOT)
                violations.append(
                    f"{rel_path} references removed extra {stale_extra!r}"
                )

    assert violations == []
