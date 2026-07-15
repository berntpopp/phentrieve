"""Tests for immutable HPO data release specifications."""

from dataclasses import replace

import pytest

from phentrieve.data_processing.release_contract import (
    DATA_RELEASE_MODELS,
    DataReleaseSpec,
    ModelReleaseSpec,
)

pytestmark = pytest.mark.unit

HPO_SHA256 = "3b646565695329aa399e937883c68d5d424d0df5eaab2f22baa0e08d44fdbe87"
SOURCE_SHA = "4c59ae3294c3f6acb449f235de0a0eba1a4bb875"


def make_spec(**overrides) -> DataReleaseSpec:
    values = {
        "release_tag": "hpo-v2026-06-23-r1",
        "hpo_version": "v2026-06-23",
        "hpo_release_date": "2026-06-23",
        "hpo_source_url": (
            "https://github.com/obophenotype/human-phenotype-ontology/"
            "releases/download/v2026-06-23/hp.json"
        ),
        "hpo_sha256": HPO_SHA256,
        "phentrieve_version": "0.26.1",
        "source_commit": SOURCE_SHA,
        "lockfile_sha256": "a" * 64,
        "models": DATA_RELEASE_MODELS,
        "active_terms": 19836,
        "multivector_documents": 63428,
    }
    values.update(overrides)
    return DataReleaseSpec(**values)


def test_hpo_v2026_06_23_release_contract_has_complete_matrix():
    """The release contract fixes source bytes, models, and document counts."""
    spec = make_spec()

    assert spec.hpo_version == "v2026-06-23"
    assert spec.hpo_sha256 == HPO_SHA256
    assert len(spec.models) == 9
    assert spec.expected_document_count("single_vector") == 19836
    assert spec.expected_document_count("multi_vector") == 63428
    assert [model.slug for model in spec.models] == [
        "biolord",
        "bge-m3",
        "labse",
        "mpnet-multi",
        "minilm-multi",
        "gte-multi",
        "jina-de",
        "tsystems-ende",
        "distiluse-multi",
    ]
    assert {model.slug for model in spec.models if model.trust_remote_code} == {
        "gte-multi",
        "jina-de",
    }
    assert {
        model.slug: model.code_revision for model in spec.models if model.code_revision
    } == {
        "gte-multi": "40ced75c3017eb27626c9d4ea981bde21a2662f4",
        "jina-de": "f3ec4cf7de7e561007f27c9efc7148b0bd713f81",
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"release_tag": "data-v2026-06-23"}, "release_tag"),
        ({"hpo_sha256": "not-a-sha256"}, "hpo_sha256"),
        ({"source_commit": "abc123"}, "source_commit"),
        ({"lockfile_sha256": "abc123"}, "lockfile_sha256"),
        ({"hpo_release_date": "23-06-2026"}, "hpo_release_date"),
        ({"active_terms": 0}, "active_terms"),
        ({"multivector_documents": 19836}, "multivector_documents"),
    ],
)
def test_release_contract_rejects_invalid_provenance(changes, message):
    """A publishable data release cannot contain mutable or malformed identity."""
    with pytest.raises(ValueError, match=message):
        make_spec(**changes)


def test_release_contract_rejects_duplicate_model_slugs():
    """Two differently named models must not map to the same asset filename."""
    duplicate = replace(DATA_RELEASE_MODELS[0], name="example/duplicate")

    with pytest.raises(ValueError, match="duplicate model slug"):
        make_spec(models=(DATA_RELEASE_MODELS[0], duplicate))


def test_model_spec_rejects_unpinned_model_revision():
    """A model reference needs its immutable Hugging Face commit SHA."""
    with pytest.raises(ValueError, match="revision"):
        ModelReleaseSpec(
            name="example/model",
            slug="example",
            revision="main",
        )


def test_model_spec_rejects_non_boolean_custom_code_policy():
    with pytest.raises(ValueError, match="trust_remote_code"):
        ModelReleaseSpec(
            name="example/model",
            slug="example",
            revision="a" * 40,
            trust_remote_code="true",  # type: ignore[arg-type]
        )


def test_model_spec_requires_a_pinned_custom_code_revision():
    with pytest.raises(ValueError, match="code_revision"):
        ModelReleaseSpec(
            name="example/model",
            slug="example",
            revision="a" * 40,
            trust_remote_code=True,
        )


def test_release_contract_serializes_and_loads_json(tmp_path):
    """The committed data-repository spec round-trips without loss."""
    spec = make_spec()
    path = tmp_path / "hpo-v2026-06-23-r1.json"

    spec.save(path)

    assert DataReleaseSpec.load(path) == spec
