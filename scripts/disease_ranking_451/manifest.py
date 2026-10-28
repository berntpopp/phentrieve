"""Write manifest.json: pinned releases, code commits, generator, package versions, SHA256 of inputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(root: Path) -> str:
    """One digest over every file below root (relative path + content digest, sorted)."""
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(f"{p.relative_to(root)}\t{sha256(p)}\n".encode())
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed git argv, no shell
        ["git", "-C", str(repo), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main(data: Path, varlens: Path) -> None:
    inputs = {
        "hpo-v2026-09-01": data / "hpo-v2026-09-01",
        "phentrieve-data-v2026-09-01": data / "phentrieve-data-v2026-09-01",
        "phenopacket-store-0.1.27": data / "phenopacket-store-0.1.27",
        "hpo-translations-735ab35": data / "hpo-translations-735ab35",
        "rarebench-6f054e0": data / "rarebench-6f054e0",
    }
    files = {
        name: {p.name: sha256(p) for p in sorted(d.iterdir()) if p.is_file()}
        for name, d in inputs.items()
    }
    manifest = {
        "issue": "berntpopp/phentrieve#451",
        "releases": {
            "hpo": "v2026-09-01",
            "phentrieve_data": "berntpopp/phentrieve-data hpo-v2026-09-01-r1 (biolord-multivec)",
            "phenopacket_store": "0.1.27",
            "hpo_translations": "obophenotype/hpo-translations@735ab350d1b1f8ea9e8833f6505c8c973df4280b",
            "rarebench": "huggingface chenxz/RareBench@6f054e04071953ef2c1779b279074245f2ab398c",
        },
        "code": {
            "phentrieve_commit": git(HERE, "rev-parse", "HEAD"),
            "phentrieve_library_base": "main@02f8fea (library code unchanged)",
            "varlens_commit_reference": git(varlens, "rev-parse", "HEAD"),
            "varlens_hpo_db_builder_version": 6,
        },
        "generator": {
            "provider": "gemini",
            "model": "gemini-3.7-flash",
            "version": "3.7-flash-08-2026",
            "temperature": 0.0,
            "seed": "451 + attempt",
        },
        "packages": {
            p: version(p)
            for p in (
                "phentrieve",
                "numpy",
                "scipy",
                "chromadb",
                "sentence-transformers",
                "google-genai",
            )
        },
        "python": sys.version.split()[0],
        "inputs_sha256": files,
        "derived_sha256": {
            "cases.jsonl": sha256(HERE / "cases.jsonl"),
            "letters/": tree_sha256(HERE / "letters"),
            # Cache rebuilt by run.sh (not committed: >1 MB); compare after a rerun.
            **{
                f"results/extractions_{lang}.jsonl": sha256(
                    HERE / "results" / f"extractions_{lang}.jsonl"
                )
                for lang in ("de", "en")
            },
            "conformance/varlens_reference.json": sha256(
                HERE / "conformance" / "varlens_reference.json"
            ),
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
