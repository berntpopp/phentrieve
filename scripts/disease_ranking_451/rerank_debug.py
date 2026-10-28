"""Debug one rerank case (exploratory, issue #451): print the exact prompt, optionally ask why.

    RERANK_PROMPT=v2 RERANK_TOP_N=50 python rerank_debug.py <hpo_dir> <case_id> <de|en> [--ask MODEL]

Without --ask it only prints the system prompt and the user prompt that rerank.py would send.
With --ask (e.g. cc-claude-opus-5-5) it sends the same prompt with an extra instruction to
explain, for every candidate it moves, why; the reply is printed. Nothing is written to results/.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank  # noqa: E402

EXPLAIN = (
    "\n\nBefore the JSON, explain in plain prose for each candidate you move away from its ranker "
    "rank (and for the ranker's #1 if you demote it): which evidence in the letter or the candidate "
    "block made you move it. Then give the JSON."
)


def main(hpo_dir: Path, case_id: str, lang: str, model: str | None) -> None:
    captured: list[tuple[str, str]] = []
    orig = rerank.load_jsonl
    real = rerank.RESULTS
    rerank.load_jsonl = lambda p: (
        [c for c in orig(p) if c.get("id") == case_id]
        if p.name == "cases.jsonl"
        else orig(real / p.name if p.name.startswith("extractions") else p)
    )
    rerank.ask = lambda m, system, prompt: (
        captured.append((system, prompt)) or '{"ranking": []}'
    )
    rerank.LANGS = (lang,)
    with tempfile.TemporaryDirectory() as tmp:
        rerank.RESULTS = Path(tmp)
        for split in ("dev", "test"):
            rerank.main(hpo_dir, split, "debug")
    if not captured:
        sys.exit(f"{case_id}: not found or no usable terms")
    system, prompt = captured[0]
    print("=== SYSTEM ===\n" + system + "\n=== USER ===\n" + prompt)
    if model:
        rerank.ask = rerank.__dict__["_ask_orig"]
        print("\n=== REPLY ===\n" + rerank.ask(model, system + EXPLAIN, prompt))


if __name__ == "__main__":
    rerank._ask_orig = rerank.ask  # keep the real call for --ask
    ask_model = sys.argv[sys.argv.index("--ask") + 1] if "--ask" in sys.argv else None
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3], ask_model)
