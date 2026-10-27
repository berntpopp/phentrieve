"""Supply-chain policy checks for GitHub Actions workflows and Dockerfiles."""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
DOCKERFILES = [REPO_ROOT / "api" / "Dockerfile", REPO_ROOT / "frontend" / "Dockerfile"]

_USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)", re.MULTILINE)
_SHA_PIN = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_actions_are_pinned_to_commit_shas(workflow: Path) -> None:
    """Mutable tags (v4, main) let an upstream retag run new code in CI."""
    text = workflow.read_text(encoding="utf-8")
    unpinned = [
        ref
        for ref in _USES.findall(text)
        if not ref.startswith("./") and not _SHA_PIN.match(ref)
    ]
    assert not unpinned, f"{workflow.name}: unpinned actions {unpinned}"


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_workflows_default_to_read_only_token(workflow: Path) -> None:
    text = workflow.read_text(encoding="utf-8")
    assert re.search(r"^permissions:\n  contents: read$", text, re.MULTILINE), (
        f"{workflow.name} must set top-level `permissions: contents: read`"
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.parent.name)
def test_dockerfile_base_images_are_digest_pinned(dockerfile: Path) -> None:
    text = dockerfile.read_text(encoding="utf-8")
    args = dict(re.findall(r"^ARG (\w+)=(\S+)$", text, re.MULTILINE))
    for image in re.findall(r"^FROM (\S+)", text, re.MULTILINE):
        resolved = re.sub(r"\$\{(\w+)\}", lambda m: args.get(m.group(1), ""), image)
        # Internal stage references (FROM python-builder AS ...) carry no registry.
        if ":" not in resolved:
            continue
        assert "@sha256:" in resolved, f"{dockerfile}: unpinned base image {image}"


def test_api_image_installs_dependencies_from_uv_lock() -> None:
    """The API image must ship the exact versions CI tested, not a fresh resolve."""
    text = (REPO_ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY pyproject.toml uv.lock ./" in text
    assert "uv sync --frozen" in text


def _api_dockerfile_stages() -> dict[str, str]:
    """Split api/Dockerfile into its named build stages."""
    text = (REPO_ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    parts = re.split(r"^FROM \S+ AS (\S+)\s*$", text, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_api_image_installs_the_project_for_the_cli() -> None:
    """The runtime image must ship the `phentrieve` console script and dist-info (#403)."""
    stages = _api_dockerfile_stages()
    assert "uv build --wheel" in stages["project-wheel"]
    runtime = stages["runtime"]
    assert "from=project-wheel" in runtime
    assert re.search(r"uv pip install .*--no-deps .*/dist/\*\.whl", runtime)


def test_api_image_has_no_unlocked_package_installs() -> None:
    """Every package in the API image comes from uv.lock (#421).

    The only `uv pip install` allowed is the project's own wheel, with --no-deps.
    """
    text = (REPO_ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    installs = re.findall(r"uv pip install[^\n]*", text)
    assert installs, "expected the project wheel install"
    for line in installs:
        assert "--no-deps" in line and "/dist/*.whl" in line, line
    assert "spacy-models/releases" not in text


def test_api_extra_locks_runtime_language_packages() -> None:
    """langdetect and the German spaCy model are declared and locked (#421)."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    api_extra = pyproject["project"]["optional-dependencies"]["api"]
    assert any(dep.startswith("langdetect>=") for dep in api_extra)
    assert any(
        dep.startswith("de_core_news_sm @ https://github.com/explosion/spacy-models/")
        for dep in api_extra
    )
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text("utf-8"))
    locked = {package["name"] for package in lock["package"]}
    assert {"langdetect", "de-core-news-sm"} <= locked


def _ci_workflow() -> dict[str, Any]:
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    return yaml.safe_load(text)


def test_ci_runs_the_built_api_image() -> None:
    """Building and scanning is not enough; the image must start its CLI (#409)."""
    steps = _ci_workflow()["jobs"]["docker-build-test"]["steps"]
    smoke = [
        step["run"]
        for step in steps
        if step.get("if") == "matrix.service == 'api'"
        and "phentrieve-api:ci" in step.get("run", "")
    ]
    modules = ("api.main", "langdetect", "de_core_news_sm", "en_core_web_sm")
    assert any(
        "phentrieve --version" in run and all(module in run for module in modules)
        for run in smoke
    ), smoke


def test_ci_runs_full_python_matrix_on_dependency_changes() -> None:
    """A uv.lock / pyproject.toml change must test 3.12 and 3.13 too (#424)."""
    changes = _ci_workflow()["jobs"]["changes"]
    filter_step = next(step for step in changes["steps"] if step.get("id") == "filter")
    filters = yaml.safe_load(filter_step["with"]["filters"])
    assert set(filters["deps"]) == {"uv.lock", "pyproject.toml"}
    versions_expr = changes["outputs"]["python-versions"]
    assert "steps.filter.outputs.deps == 'true'" in versions_expr
    assert '"3.12", "3.13"' in versions_expr
