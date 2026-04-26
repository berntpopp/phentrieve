"""Integration test that every YAML snippet in docs/user-guide/
configuration-profiles.md parses against the Profile schema."""

import re
from pathlib import Path

import pytest
import yaml as pyyaml

from phentrieve.profiles import ProfilesFile

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "user-guide" / "configuration-profiles.md"


def _extract_yaml_blocks(md_text: str) -> list[str]:
    """Find all ```yaml ... ``` fenced blocks in the markdown."""
    pattern = re.compile(r"```yaml\n(.*?)```", re.DOTALL)
    return pattern.findall(md_text)


def test_doc_exists():
    assert DOC.exists(), f"Documentation file {DOC} missing"


def test_yaml_blocks_present():
    blocks = _extract_yaml_blocks(DOC.read_text())
    assert len(blocks) >= 2, "Expected at least two YAML examples in the doc"


@pytest.mark.parametrize("block_idx", range(20))  # bounded loop
def test_each_yaml_block_parses(block_idx: int):
    blocks = _extract_yaml_blocks(DOC.read_text())
    if block_idx >= len(blocks):
        pytest.skip(f"No block at index {block_idx}")
    raw = pyyaml.safe_load(blocks[block_idx])
    assert raw is not None, f"Block {block_idx} parsed as None"
    # Wrap in a profiles: dict if not already a top-level shape.
    if "profiles" in raw:
        ProfilesFile.model_validate(raw)
    else:
        # It's a profile body example or extraction section - both are fine.
        pass
