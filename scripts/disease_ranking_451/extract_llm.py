"""Exploratory arm: Phentrieve's LLM full-text backend on every letter (issue #451, post hoc).

Same call as `phentrieve text process --extraction-backend llm -l <lang>` with its defaults
(two_phase, whole_document_grounded, DEFAULT_LLM_MODEL), run from the directory set up by
run.sh. Writes results/extractions_<lang>_llm.jsonl; letters already there are skipped.

    python extract_llm.py <lang> [limit]
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from phentrieve.llm.config import DEFAULT_LLM_MODEL
from phentrieve.text_processing.full_text_service import run_full_text_service

HERE = Path(__file__).resolve().parent


def main(lang: str, limit: int | None = None) -> None:
    tag = __import__("os").environ.get(
        "EXTRACT_TAG", ""
    )  # e.g. "_negfix" for PR #453 code
    out_path = HERE / "results" / f"extractions_{lang}_llm{tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {
            json.loads(x)["case"]
            for x in out_path.read_text(encoding="utf-8").splitlines()
        }
    letters = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((HERE / "letters" / lang).glob("*.json"))
    ]
    todo = [x for x in letters if x["case"] not in done][:limit]
    lock = threading.Lock()

    def run(letter: dict) -> None:
        start = time.perf_counter()
        try:
            res = run_full_text_service(
                text=letter["text"], extraction_backend="llm", language=lang
            )
        except (
            Exception
        ) as exc:  # keep going; a failed letter is retried on the next run
            print(
                "FAILED", letter["case"], type(exc).__name__, str(exc)[:200], flush=True
            )
            return
        if not res[
            "aggregated_hpo_terms"
        ]:  # retrieval glitch (ChromaDB race): retry next run
            print("FAILED", letter["case"], "no terms", flush=True)
            return
        meta = res.get("meta", {})
        record = {
            "case": letter["case"],
            "lang": lang,
            "model": meta.get("llm_model", DEFAULT_LLM_MODEL),
            "seconds": round(time.perf_counter() - start, 3),
            "tokens": [meta.get("token_input"), meta.get("token_output")],
            "terms": [
                [t["id"], t.get("status"), bool(t.get("excluded"))]
                for t in res["aggregated_hpo_terms"]
            ],
        }
        with lock, out_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")

    if (
        todo
    ):  # one call first, so the index client is initialised before threads share it
        run(todo[0])
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(run, todo[1:]))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
