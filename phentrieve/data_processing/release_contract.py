"""Immutable specifications for reproducible HPO data releases."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_MODEL_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_RELEASE_TAG_PATTERN = re.compile(r"^hpo-v(\d{4}-\d{2}-\d{2})-r([1-9]\d*)$")
_INDEX_DOCUMENT_COUNTS = {
    "single_vector": 19836,
    "multi_vector": 63428,
}


@dataclass(frozen=True)
class ModelReleaseSpec:
    """An embedding model fixed to an immutable Hugging Face revision."""

    name: str
    slug: str
    revision: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model name must not be empty")
        if not _MODEL_SLUG_PATTERN.fullmatch(self.slug):
            raise ValueError(f"invalid model slug: {self.slug!r}")
        if not _GIT_SHA_PATTERN.fullmatch(self.revision):
            raise ValueError(f"invalid model revision: {self.revision!r}")


DATA_RELEASE_MODELS: tuple[ModelReleaseSpec, ...] = (
    ModelReleaseSpec(
        name="FremyCompany/BioLORD-2023-M",
        slug="biolord",
        revision="4ea2ea2c89ef63365f7fcd91406a0cd8ac36d2e2",
    ),
    ModelReleaseSpec(
        name="BAAI/bge-m3",
        slug="bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
    ),
    ModelReleaseSpec(
        name="sentence-transformers/LaBSE",
        slug="labse",
        revision="836121a0533e5664b21c7aacc5d22951f2b8b25b",
    ),
    ModelReleaseSpec(
        name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        slug="mpnet-multi",
        revision="4328cf26390c98c5e3c738b4460a05b95f4911f5",
    ),
    ModelReleaseSpec(
        name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        slug="minilm-multi",
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    ),
    ModelReleaseSpec(
        name="Alibaba-NLP/gte-multilingual-base",
        slug="gte-multi",
        revision="9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
    ),
    ModelReleaseSpec(
        name="jinaai/jina-embeddings-v2-base-de",
        slug="jina-de",
        revision="3f9eede875721714945b6a99a3198299243cf2be",
    ),
    ModelReleaseSpec(
        name="T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
        slug="tsystems-ende",
        revision="73fdad86ea7ac68989712ce2007ab43ae89a2ad7",
    ),
    ModelReleaseSpec(
        name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        slug="distiluse-multi",
        revision="bfe45d0732ca50787611c0fe107ba278c7f3f889",
    ),
)


@dataclass(frozen=True)
class DataReleaseSpec:
    """The immutable inputs and expected output shape of one data release."""

    release_tag: str
    hpo_version: str
    hpo_release_date: str
    hpo_source_url: str
    hpo_sha256: str
    phentrieve_version: str
    source_commit: str
    lockfile_sha256: str
    models: tuple[ModelReleaseSpec, ...]
    active_terms: int
    multivector_documents: int

    def __post_init__(self) -> None:
        match = _RELEASE_TAG_PATTERN.fullmatch(self.release_tag)
        if match is None:
            raise ValueError(f"invalid release_tag: {self.release_tag!r}")

        release_date = match.group(1)
        if self.hpo_version != f"v{release_date}":
            raise ValueError("hpo_version must match release_tag")
        if self.hpo_release_date != release_date:
            raise ValueError("hpo_release_date must match release_tag")
        try:
            date.fromisoformat(self.hpo_release_date)
        except ValueError as error:
            raise ValueError(
                f"invalid hpo_release_date: {self.hpo_release_date!r}"
            ) from error

        for field_name, value in (
            ("hpo_sha256", self.hpo_sha256),
            ("lockfile_sha256", self.lockfile_sha256),
        ):
            if not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"invalid {field_name}: {value!r}")
        if not _GIT_SHA_PATTERN.fullmatch(self.source_commit):
            raise ValueError(f"invalid source_commit: {self.source_commit!r}")
        if not self.hpo_source_url.startswith("https://"):
            raise ValueError("hpo_source_url must use HTTPS")
        if not self.phentrieve_version:
            raise ValueError("phentrieve_version must not be empty")
        if self.active_terms <= 0:
            raise ValueError("active_terms must be positive")
        if self.multivector_documents <= self.active_terms:
            raise ValueError("multivector_documents must exceed active_terms")
        if not self.models:
            raise ValueError("models must not be empty")

        model_slugs = [model.slug for model in self.models]
        if len(set(model_slugs)) != len(model_slugs):
            raise ValueError("duplicate model slug")

    def expected_document_count(self, index_type: str) -> int:
        """Return the document count a release collection must contain."""
        if index_type == "single_vector":
            return self.active_terms
        if index_type == "multi_vector":
            return self.multivector_documents
        raise ValueError(f"unsupported index type: {index_type!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the specification in the data-repository JSON shape."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataReleaseSpec:
        """Deserialize a committed data-release JSON specification."""
        values = dict(data)
        raw_models = values.get("models", [])
        values["models"] = tuple(
            model if isinstance(model, ModelReleaseSpec) else ModelReleaseSpec(**model)
            for model in raw_models
        )
        return cls(**values)

    def save(self, path: Path) -> None:
        """Write a canonical, readable release specification."""
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> DataReleaseSpec:
        """Load a release specification from JSON."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
