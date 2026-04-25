"""Guard against reintroducing the removed second-stage ranking feature."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


_BANNED_TERMS = (
    "rerank",
    "reranker",
    "reranked",
    "reranking",
    "cross-encoder",
    "cross_encoder",
    "cross encoder",
    "CrossEncoder",
)


def test_removed_second_stage_ranking_has_no_tracked_references() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    git = shutil.which("git")
    assert git is not None
    tracked_files = subprocess.run(  # noqa: S603 - fixed git executable and arguments
        [git, "ls-files"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()

    offenders: list[str] = []
    pattern = re.compile("|".join(re.escape(term) for term in _BANNED_TERMS), re.I)
    self_path = Path(__file__).resolve()

    for relative_path in tracked_files:
        if relative_path.startswith(".planning/"):
            continue
        path = repo_root / relative_path
        if path.resolve() == self_path or not path.is_file():
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{relative_path}:{line_number}: {line.strip()}")

    assert offenders == []
