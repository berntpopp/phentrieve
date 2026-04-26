"""Tests for AdaptiveRechunkingConfig and AdaptiveRechunkingProfileBlock."""

import dataclasses

import pytest


def test_default_config_disabled():
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig

    cfg = AdaptiveRechunkingConfig()
    assert cfg.enabled is False  # opt-in
    assert cfg.quality_threshold == 0.55
    assert cfg.margin_threshold == 0.03
    assert cfg.use_ontology_coherence is False
    assert cfg.max_depth == 2
    assert cfg.min_chunk_chars == 30
    assert cfg.max_sentences_per_subchunk == 3
    assert cfg.overlap_sentences == 1
    assert cfg.score_improvement_gate == 0.05


def test_config_is_frozen():
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingConfig

    cfg = AdaptiveRechunkingConfig(enabled=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.enabled = False  # type: ignore[misc]


def test_profile_block_pydantic_extra_forbid():
    from pydantic import ValidationError

    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingProfileBlock

    with pytest.raises(ValidationError):
        AdaptiveRechunkingProfileBlock(unknown_knob=123)


def test_profile_block_optional_fields():
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingProfileBlock

    block = AdaptiveRechunkingProfileBlock(enabled=True)
    assert block.enabled is True
    assert block.quality_threshold is None  # optional
    assert block.max_depth is None


def test_profile_block_all_fields():
    from phentrieve.retrieval.adaptive_rechunker import AdaptiveRechunkingProfileBlock

    block = AdaptiveRechunkingProfileBlock(
        enabled=True,
        quality_threshold=0.6,
        margin_threshold=0.02,
        max_depth=1,
        min_chunk_chars=40,
        max_sentences_per_subchunk=2,
        overlap_sentences=0,
        score_improvement_gate=0.1,
        use_ontology_coherence=False,
    )
    assert block.quality_threshold == 0.6
    assert block.max_depth == 1


def test_resolve_precedence_cli_over_profile_over_yaml_over_default():
    from phentrieve.retrieval.adaptive_rechunker import (
        AdaptiveRechunkingProfileBlock,
        adaptive_config_from_profile_block,
    )

    block = AdaptiveRechunkingProfileBlock(
        enabled=True, quality_threshold=0.7, margin_threshold=0.04
    )
    yaml_block = {
        "quality_threshold": 0.5,  # masked by profile
        "min_chunk_chars": 50,  # masked by no other source -> takes effect
    }
    cli_overrides = {"margin_threshold": 0.01}  # masks profile

    cfg = adaptive_config_from_profile_block(
        block=block,
        yaml_block=yaml_block,
        cli_overrides=cli_overrides,
    )

    # CLI wins
    assert cfg.margin_threshold == 0.01
    # Profile wins over YAML
    assert cfg.quality_threshold == 0.7
    assert cfg.enabled is True
    # YAML wins over default
    assert cfg.min_chunk_chars == 50
    # Default for unset
    assert cfg.max_depth == 2
    assert cfg.score_improvement_gate == 0.05


def test_resolve_with_no_inputs_returns_defaults():
    from phentrieve.retrieval.adaptive_rechunker import (
        AdaptiveRechunkingConfig,
        adaptive_config_from_profile_block,
    )

    cfg = adaptive_config_from_profile_block(
        block=None, yaml_block=None, cli_overrides=None
    )
    assert cfg == AdaptiveRechunkingConfig()
