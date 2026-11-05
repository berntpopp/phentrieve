"""Generate one German and one English referral letter per case (issue #451).

Cached in letters/<lang>/<case>.json; an existing letter is never regenerated. A letter that
names the gene, the disease ID or the disease label is regenerated with seed + 1 (3 attempts).
Needs GEMINI_API_KEY in the environment.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from phentrieve.llm.providers.gemini import GeminiStructuredOutputProvider

HERE = Path(__file__).resolve().parent
MODEL = "gemini-3.7-flash"
SEED, ATTEMPTS = 451, 3
LANGUAGES = {"de": "German", "en": "English"}

SYSTEM = """You write realistic clinical correspondence for a research test set.
Write one referral letter from one physician to a colleague (about 150-300 words) about the
patient described below, in {language}.

Rules:
- Describe each finding the way clinicians write it: paraphrase, use clinical shorthand,
  measurements or descriptions. Never copy a listed finding name word for word.
- Mention every listed observed finding. Write every listed absent finding as a natural
  negation (for example "no ...", "... was normal", "... unremarkable").
- Add realistic distractors that are not listed: normal examination findings, family
  history, current medication, social details. Every distractor must be normal or
  non-medical: do not add any symptom, sign, test result or finding that is abnormal and
  not listed, and do not add tests or findings that hint at a particular condition.
- Never name a diagnosis, syndrome, disease, gene, variant or inheritance pattern, and do not
  speculate about one. The letter asks the colleague for an assessment.
- Output only the letter text: no subject line with a diagnosis, no markdown, no placeholders
  in brackets; invent plausible names of places and people only if needed, never real ones."""


def _fmt(case: dict) -> str:
    lines = [
        f"Sex: {case['sex'] or 'unknown'}",
        f"Age at last encounter (ISO 8601): {case['age'] or 'unknown'}",
        f"Disease onset: {case['onset'] or 'unknown'}",
        "Observed findings:",
        *(f"- {t['label']}" for t in case["observed"] if t["label"]),
        "Absent findings (examined and not present):",
        *(f"- {t['label']}" for t in case["excluded"] if t["label"]),
    ]
    return "\n".join(lines)


def mentions(case: dict, text: str) -> list[str]:
    """Gene symbol, disease ID or disease label (or its head before , or ;) found in the text."""
    hits = []
    if re.search(rf"(?<![\w-]){re.escape(case['gene'])}(?![\w-])", text):
        hits.append(case["gene"])
    label = case["disease_label"]
    for needle in {case["disease"], label, re.split(r"[,;]", label)[0].strip()}:
        if needle and needle.lower() in text.lower():
            hits.append(needle)
    return hits


def _path(lang: str, case: dict) -> Path:
    return HERE / "letters" / lang / (re.sub(r"[^\w.-]", "_", case["id"]) + ".json")


def generate(lang: str, case: dict) -> str:
    path = _path(lang, case)
    if path.exists():
        return "cached"
    record: dict = {}
    for attempt in range(ATTEMPTS):
        llm = GeminiStructuredOutputProvider(
            model_name=MODEL, temperature=0.0, seed=SEED + attempt, max_tokens=16384
        )
        resp = llm.complete(
            [
                {"role": "system", "content": SYSTEM.format(language=LANGUAGES[lang])},
                {"role": "user", "content": _fmt(case)},
            ]
        )
        text = (resp.content or "").strip()
        hits = mentions(case, text)
        record = {
            "case": case["id"],
            "lang": lang,
            "model": MODEL,
            "temperature": 0.0,
            "seed": SEED + attempt,
            "attempt": attempt + 1,
            "finish_reason": resp.finish_reason,
            "usage": resp.usage,
            "mentions": hits,
            "text": text,
        }
        if text and not hits and "MAX_TOKENS" not in str(resp.finish_reason):
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return f"attempt {record['attempt']}" + (
        " FLAGGED" if record["mentions"] or not record["text"] else ""
    )


def main(limit: int | None = None) -> None:
    cases = [
        json.loads(line)
        for line in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # PoC scope (PROTOCOL.md deviation 2026-10-28): cited test cases are not used.
    cases = [c for c in cases if not (c["split"] == "test" and c["cited"])]
    jobs = [(lang, c) for c in cases[:limit] for lang in LANGUAGES]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (lang, case), status in zip(
            jobs, pool.map(lambda j: generate(*j), jobs), strict=True
        ):
            if status != "cached":
                print(lang, case["id"], status, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
