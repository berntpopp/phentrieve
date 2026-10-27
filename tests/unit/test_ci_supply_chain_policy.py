"""Supply-chain policy checks for GitHub Actions workflows and Dockerfiles."""

import re
from pathlib import Path

import pytest

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
