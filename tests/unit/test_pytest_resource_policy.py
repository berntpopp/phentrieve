"""Policy checks for resource-safe local and CI pytest targets."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pytest_addopts() -> list[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]
    assert isinstance(addopts, list)
    return [str(option) for option in addopts]


def _makefile_text() -> str:
    return (REPO_ROOT / "Makefile").read_text("utf-8")


def _make_target_recipe(target: str) -> str:
    makefile = _makefile_text()
    match = re.search(
        rf"^{re.escape(target)}:.*\n(?P<recipe>(?:\t.*\n)+)",
        makefile,
        re.MULTILINE,
    )
    assert match is not None, f"Makefile target {target!r} not found"
    return match.group("recipe")


def test_pytest_defaults_do_not_force_xdist_or_coverage() -> None:
    addopts = _pytest_addopts()

    assert "-v" not in addopts
    assert "-n" not in addopts
    assert "--numprocesses" not in addopts
    assert not any(option.startswith("--cov") for option in addopts)


def test_make_test_caps_parallelism_and_disables_coverage() -> None:
    makefile = _makefile_text()
    recipe = _make_target_recipe("test")

    assert "PYTEST_PATHS ?= tests/" in makefile
    assert "PYTEST_WORKERS ?=" in makefile
    assert "PYTEST_DIST ?= loadscope" in makefile
    assert "PYTEST_OUTPUT ?= -q" in makefile
    assert "pytest $(PYTEST_PATHS)" in recipe
    assert "$(PYTEST_OUTPUT)" in recipe
    assert " -v " not in recipe
    assert "-n $(PYTEST_WORKERS)" in recipe
    assert "--dist $(PYTEST_DIST)" in recipe
    assert "--no-cov" in recipe


def test_make_test_ci_uses_separate_worker_cap_and_no_html_coverage() -> None:
    makefile = _makefile_text()
    recipe = _make_target_recipe("test-ci")

    assert "PYTEST_PATHS ?= tests/" in makefile
    assert "PYTEST_CI_WORKERS ?=" in makefile
    assert "PYTEST_DIST ?= loadscope" in makefile
    assert "PYTEST_OUTPUT ?= -q" in makefile
    assert "PYTEST_COV_FAIL_UNDER ?= 40" in makefile
    assert "PYTEST_COV_TERM_REPORT ?= term:skip-covered" in makefile
    assert "pytest $(PYTEST_PATHS)" in recipe
    assert "$(PYTEST_OUTPUT)" in recipe
    assert " -v " not in recipe
    assert "-n $(PYTEST_CI_WORKERS)" in recipe
    assert "--dist $(PYTEST_DIST)" in recipe
    assert "--cov-report=xml" in recipe
    assert "--cov-report=$(PYTEST_COV_TERM_REPORT)" in recipe
    assert "--cov-fail-under=$(PYTEST_COV_FAIL_UNDER)" in recipe
    assert "html" not in recipe


def test_local_python_ci_reuses_existing_environment_by_default() -> None:
    makefile = _makefile_text()
    ci_python = _make_target_recipe("ci-python-quality")
    ci_python_clean = _make_target_recipe("ci-python-quality-clean")

    assert "python-deps" in ci_python
    assert "uv sync --locked --all-extras --dev" not in ci_python
    assert "ci-python-quality-clean" in makefile
    assert "python-install-ci" in ci_python_clean


def test_default_typecheck_fast_does_not_use_mypy_daemon() -> None:
    recipe = _make_target_recipe("typecheck-fast")

    assert "dmypy" not in recipe
    assert "uv run mypy phentrieve/ api/" in recipe


def test_ci_quick_uses_project_tool_wrappers() -> None:
    recipe = _make_target_recipe("ci-quick")

    assert "$(MAKE) format-check" in recipe
    assert "$(MAKE) lint" in recipe
    assert "ruff format" not in recipe
    assert "ruff check" not in recipe
