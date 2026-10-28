"""Run Phentrieve's default `text process` pipeline on every letter (issue #451).

Same call path and parameters as `phentrieve text process -l <lang>` with the repo's
phentrieve.yaml (standard backend, BioLORD multi-vector, sliding_window_punct_conj_cleaned,
chunk threshold 0.7, 10 results per chunk, aggregated confidence 0.75, dependency assertion).
The models and the index load once per language instead of once per letter.

Run from a directory holding a copy of phentrieve.yaml and a `data` link to the pinned
bundle (run.sh sets this up). Writes results/extractions_<lang>.jsonl; letters already
present there are skipped.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from phentrieve.cli.utils import resolve_chunking_pipeline_config
from phentrieve.config import DEFAULT_CHUNKING_STRATEGY
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing.full_text_service import run_full_text_service
from phentrieve.text_processing.pipeline import TextProcessingPipeline

HERE = Path(__file__).resolve().parent
MODEL = "FremyCompany/BioLORD-2023-M"
SETTINGS = {
    "num_results_per_chunk": 10,
    "chunk_retrieval_threshold": 0.7,
    "top_term_per_chunk": False,
    "min_confidence_for_aggregated": 0.75,
}
ASSERTION = {"disable": False, "preference": "dependency"}


def main(lang: str) -> None:
    out_path = HERE / "results" / f"extractions_{lang}.jsonl"
    done = set()
    if out_path.exists():
        done = {
            json.loads(line)["case"]
            for line in out_path.read_text(encoding="utf-8").splitlines()
        }
    letters = sorted((HERE / "letters" / lang).glob("*.json"))
    model = load_embedding_model(MODEL)
    pipeline = TextProcessingPipeline(
        language=lang,
        chunking_pipeline_config=resolve_chunking_pipeline_config(
            None, DEFAULT_CHUNKING_STRATEGY
        ),
        assertion_config=ASSERTION,
        sbert_model_for_semantic_chunking=model,
    )
    retriever = DenseRetriever.from_model_name(
        model=model, model_name=MODEL, multi_vector=True
    )
    with out_path.open("a", encoding="utf-8") as out:
        for path in letters:
            letter = json.loads(path.read_text(encoding="utf-8"))
            if letter["case"] in done:
                continue
            start = time.perf_counter()
            res = run_full_text_service(
                text=letter["text"],
                extraction_backend="standard",
                language=lang,
                text_pipeline=pipeline,
                retriever=retriever,
                retrieval_model_name=MODEL,
                include_positions=True,
                **SETTINGS,
            )
            record = {
                "case": letter["case"],
                "lang": lang,
                "seconds": round(time.perf_counter() - start, 3),
                "terms": [
                    [t["id"], t["status"], t["score"]]
                    for t in res["aggregated_hpo_terms"]
                ],
                "chunks": [
                    [
                        [m["id"], m["score"], m["assertion_status"]]
                        for m in c["hpo_matches"]
                    ]
                    for c in res["processed_chunks"]
                ],
            }
            out.write(json.dumps(record) + "\n")
            out.flush()


if __name__ == "__main__":
    main(sys.argv[1])
