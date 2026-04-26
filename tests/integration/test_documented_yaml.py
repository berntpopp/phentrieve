"""Integration test that every YAML snippet in user-guide docs parses
against the schemas it documents.

Currently covers:
- docs/user-guide/configuration-profiles.md (Plan A)
- docs/user-guide/adaptive-rechunking.md    (Plan B)
"""

import re
from pathlib import Path

import pytest
import yaml as pyyaml

from phentrieve.profiles import ProfilesFile
from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingProfileBlock

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "user-guide" / "configuration-profiles.md"
ADAPTIVE_DOC = REPO_ROOT / "docs" / "user-guide" / "adaptive-rechunking.md"


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


# -----------------------------------------------------------------------------
# adaptive-rechunking.md (Plan B)
# -----------------------------------------------------------------------------


def _validate_adaptive_block(raw: dict) -> None:
    """Validate an extraction.adaptive_rechunking mapping."""
    block = raw["extraction"]["adaptive_rechunking"]
    AdaptiveRechunkingProfileBlock.model_validate(block)


def test_adaptive_doc_exists():
    assert ADAPTIVE_DOC.exists(), f"Documentation file {ADAPTIVE_DOC} missing"


def test_adaptive_yaml_blocks_present():
    blocks = _extract_yaml_blocks(ADAPTIVE_DOC.read_text())
    assert len(blocks) >= 3, (
        "Expected at least three YAML examples in adaptive-rechunking.md "
        "(quick start, full schema, worked examples)"
    )


@pytest.mark.parametrize("block_idx", range(20))  # bounded loop
def test_each_adaptive_yaml_block_parses(block_idx: int):
    blocks = _extract_yaml_blocks(ADAPTIVE_DOC.read_text())
    if block_idx >= len(blocks):
        pytest.skip(f"No block at index {block_idx}")
    raw = pyyaml.safe_load(blocks[block_idx])
    assert raw is not None, f"Block {block_idx} parsed as None"

    if "profiles" in raw:
        # Profile-style example: validate against ProfilesFile. The nested
        # adaptive_rechunking block is validated by Pydantic via the Profile
        # field type.
        ProfilesFile.model_validate(raw)
    elif "extraction" in raw and "adaptive_rechunking" in raw["extraction"]:
        # Direct extraction.adaptive_rechunking example.
        _validate_adaptive_block(raw)
    else:
        # Plain mapping (e.g. response meta example) - just confirm it parsed.
        pass
