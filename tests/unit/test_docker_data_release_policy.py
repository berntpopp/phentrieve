"""Policy checks for the Docker image's embedded HPO data release."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_URL = (
    "https://github.com/berntpopp/phentrieve-data/releases/download/"
    "hpo-v2026-06-23-r1/"
    "phentrieve-data-v2026-06-23-biolord-multivec.tar.gz"
)


def test_docker_builds_default_to_the_current_verified_data_release() -> None:
    """Standalone and published API images must embed the same verified bundle."""
    api_dockerfile = (REPO_ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    publish_workflow = (
        REPO_ROOT / ".github" / "workflows" / "docker-publish.yml"
    ).read_text(encoding="utf-8")

    assert f'ARG BUNDLE_URL="{DEFAULT_BUNDLE_URL}"' in api_dockerfile
    assert f"BUNDLE_URL={DEFAULT_BUNDLE_URL}" in publish_workflow
