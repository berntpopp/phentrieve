"""Safety: recommended_citation is emitted in all response modes and embeds the
HPO release version so a pasted citation is reproducible."""

import pytest

from api.mcp.resources import hpo_release_version, recommended_citation
from api.mcp.tools.retrieval import _maybe_citation

pytestmark = pytest.mark.unit


def _expected_citation() -> str:
    version = hpo_release_version()
    release = f" {version}" if version and version != "unknown" else ""
    return (
        f"Human Phenotype Ontology (HPO{release}), https://hpo.jax.org/ "
        "(consulted via Phentrieve; research use only)."
    )


def test_citation_is_verbatim_contract():
    # Exact equality, not a hostname substring check: the citation is a verbatim
    # contract, and `"host" in url`-style checks trip CodeQL
    # py/incomplete-url-substring-sanitization (a false positive here).
    assert recommended_citation() == _expected_citation()


def test_citation_embeds_known_hpo_version():
    version = hpo_release_version()
    if version != "unknown":
        assert version in recommended_citation()


@pytest.mark.parametrize("mode", ["minimal", "compact", "standard", "full"])
def test_citation_emitted_in_all_modes(mode):
    meta: dict = {}
    _maybe_citation(meta, mode)
    assert meta["recommended_citation"]
    assert "Human Phenotype Ontology" in meta["recommended_citation"]
