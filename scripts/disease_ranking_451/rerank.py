"""Exploratory step 3 (post hoc, RUO): LLM rerank of the scorer's top diseases (issue #451).

    python rerank.py <hpo_dir> <dev|test|all> <model>
    model: gemini-3.1-flash-lite (Phentrieve's default LLM), claude-opus-5-5 (Anthropic API) or
    cc-claude-opus-5-5 (same model through local Claude Code, `claude -p`)

Pipeline per letter:
1. Phentrieve LLM full-text terms (results/extractions_<lang>_llm.jsonl);
2. VarLens v2 scorer, top-N diseases;
3. an LLM reorders those N candidates, given the letter, the extracted terms and each
   candidate's most frequent annotated phenotypes.

The true disease keeps its scorer rank when it is outside the top N; inside, it gets its
reranked position (a missing or invalid answer falls back to scorer order). Cached in
results/rerank_<model>_top<N>_<lang>.jsonl (RERANK_TOP_N, default 20).
"""

from __future__ import annotations

import json
import re
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scorer  # noqa: E402
from analysis import LANGS, RESULTS, load_jsonl  # noqa: E402

from phentrieve.llm.providers.gemini import GeminiStructuredOutputProvider  # noqa: E402

SEED, N_PHENO = 451, 10
TOP_N = int(__import__("os").environ.get("RERANK_TOP_N", "20"))  # candidate pool size


def ask(model: str, system: str, prompt: str) -> str:
    """One reranking call. Gemini: temperature 0 + seed. Claude 5.x accepts no temperature/seed.
    "cc-<model>": headless local Claude Code (`claude -p`) on the user's Claude Code login, no
    tools, no MCP or settings, run from an empty directory so no project context is loaded."""
    if model.startswith("cc-"):
        import os
        import subprocess
        import tempfile

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
        }
        with tempfile.TemporaryDirectory() as cwd:
            done = subprocess.run(  # noqa: S603 - fixed argv, prompt on stdin
                [  # noqa: S607 - the user's local Claude Code CLI on PATH
                    "claude",
                    "-p",
                    "--model",
                    model[3:],
                    "--system-prompt",
                    system,
                    "--tools",
                    "",
                    "--strict-mcp-config",
                    "--setting-sources",
                    "",
                    "--no-session-persistence",
                    "--output-format",
                    "json",
                    "--effort",
                    "medium",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env,
                timeout=900,
                check=False,
            )
        out = json.loads(done.stdout or "{}")
        if done.returncode != 0 or out.get("is_error"):
            raise RuntimeError(
                f"claude -p failed: {(out.get('result') or done.stderr)[:200]}"
            )
        return out.get("result", "")
    if model.startswith("gemini"):
        llm = GeminiStructuredOutputProvider(
            model_name=model, temperature=0.0, seed=SEED, max_tokens=8192
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return llm.complete(msgs).content or ""
    import os

    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY") or os.environ["CLAUDE_API_KEY"]
    )
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},  # rerouted on a refusal instead of failing
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("refusal")
    return "".join(b.text for b in resp.content if b.type == "text")


SYSTEM = """You are a clinical geneticist supporting research on rare-disease phenotype matching.
You get a referral letter, the HPO terms extracted from it, and candidate diseases ranked by a
phenotype-similarity score, each with its most frequent annotated features.
Reorder ALL candidates from most to least likely diagnosis for this patient. Weigh the
characteristic features, the age of onset and the explicitly absent findings.
Answer with JSON only: {"ranking": ["<disease id>", ...]} containing every candidate id once."""


def candidates_text(ix: scorer.PhenotypeIndex, order: list[int], s: np.ndarray) -> str:
    lines = []
    for i, d in enumerate(order, 1):
        a, b = ix.dis_ptr[d], ix.dis_ptr[d + 1]
        top = sorted(range(a, b), key=lambda j: -ix.ann_f[j])[:N_PHENO]
        feats = "; ".join(ix.labels[ix.ann_term[j]] for j in top)
        lines.append(
            f"{i}. {ix.disease_ids[d]} | {ix.disease_labels[d]} | score {s[d]:.3f} | features: {feats}"
        )
    return "\n".join(lines)


PROMPT = __import__("os").environ.get("RERANK_PROMPT", "v1")  # prompt / input version

SYSTEM_V2 = """You are a clinical geneticist supporting research (not clinical care) on rare-disease
phenotype matching. A phenotype-similarity ranker has scored every disease against the patient's
findings; you see its top candidates in score order. Your job is to reorder them using what the
ranker cannot weigh well: the clinical picture as a whole, age at onset, inheritance pattern
versus the family history, and findings the letter explicitly reports as absent.

Rules:
- Treat the ranker's order as a strong prior. Move a candidate up only for specific evidence:
  characteristic features the patient has (marked [match]), a fitting onset or inheritance
  pattern, or a decisive absent finding for the competitor.
- Candidates from the same disease series (e.g. "... type 2" vs "... type 6") or the same
  disease under two identifiers often cannot be told apart from the phenotype; without a
  distinguishing feature keep them in the ranker's order.
- A finding missing from the letter is weak evidence; case reports are incomplete.
Answer with JSON only, no prose: {"ranking": ["<disease id>", ...]} with every candidate id exactly once."""

FREQ_WORDS = (
    (0.8, "very frequent"),
    (0.3, "frequent"),
    (0.05, "occasional"),
    (0.0, "rare"),
)


def disease_meta(hpo_dir: Path) -> dict[str, dict[str, set[str]]]:
    """Inheritance (aspect I) and onset (aspect C) labels per disease from phenotype.hpoa."""
    labels, _, _ = scorer.parse_hp_json(hpo_dir / "hp.json")
    meta: dict[str, dict[str, set[str]]] = {}
    for line in (hpo_dir / "phenotype.hpoa").read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if (
            line.startswith("#")
            or len(cols) < 11
            or cols[10] not in ("I", "C")
            or cols[2] == "NOT"
        ):
            continue
        key = "inheritance" if cols[10] == "I" else "onset"
        meta.setdefault(cols[0], {}).setdefault(key, set()).add(
            labels.get(cols[3], cols[3])
        )
    return meta


def freq_word(f: float) -> str:
    if abs(f - scorer.MISSING_F) < 1e-6:
        return "frequency unknown"
    return next(w for t, w in FREQ_WORDS if f >= t)


def candidates_text_v2(ix, order, s, present: list[int], meta) -> str:
    """Per candidate: inheritance, onset, the features matching the patient, then its most
    characteristic other features (frequency x specificity)."""
    anc = {t: set(ix.ancestors(t).tolist()) for t in present}
    lines = []
    for i, d in enumerate(order, 1):
        a, b = ix.dis_ptr[d], ix.dis_ptr[d + 1]
        ann = list(range(a, b))
        matched: dict[int, str] = {}
        for p in (
            present
        ):  # VarLens explain rule: best disease term for p with IC(MICA)/IC(p) >= 0.5
            best, arg = 0.0, -1
            for j in ann:
                t = ix.ann_term[j]
                t_anc = set(ix.ancestors(t).tolist())
                if t not in anc[p] and p not in t_anc:
                    continue  # same lineage only: siblings sharing a broad parent are no match
                common = anc[p] & t_anc
                v = max((ix.ic[c] for c in common), default=0.0) / ix.self_ic[p]
                if v > best:
                    best, arg = v, j
            if arg >= 0 and best >= 0.5:
                matched.setdefault(arg, ix.labels[p])
        other = sorted(
            (j for j in ann if j not in matched),
            key=lambda j: -ix.ann_f[j] * ix.self_ic[ix.ann_term[j]],
        )[:N_PHENO]
        m = meta.get(ix.disease_ids[d], {})
        lines.append(
            f"{i}. {ix.disease_ids[d]} | {ix.disease_labels[d]} | ranker score {s[d]:.3f}\n"
            f"   inheritance: {', '.join(sorted(m.get('inheritance', []))) or 'not annotated'}; "
            f"onset: {', '.join(sorted(m.get('onset', []))) or 'not annotated'}\n"
            f"   [match] {'; '.join(f'{ix.labels[ix.ann_term[j]]} ({freq_word(ix.ann_f[j])}) <- patient: {pl}' for j, pl in matched.items()) or 'none'}\n"
            f"   other characteristic features: {'; '.join(f'{ix.labels[ix.ann_term[j]]} ({freq_word(ix.ann_f[j])})' for j in other)}"
        )
    return "\n".join(lines)


def parse_ranking(reply: str) -> list[str]:
    """The first JSON object in the reply that has a "ranking" list."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", reply):
        try:
            obj, _ = dec.raw_decode(reply[m.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("ranking"), list):
            return [str(x) for x in obj["ranking"]]
    return []


SYSTEM_V3 = """You are a clinical geneticist supporting research (not clinical care) on rare-disease
phenotype matching. A phenotype-similarity ranker has scored all ~9,000 gene-linked diseases
against the patient's findings; you see its top candidates with their ranker rank and score.
The ranker is usually right. Change its order only for specific, stated evidence:
(a) a very frequent or obligatory feature of a candidate is explicitly absent in the letter
    (listed under "conflicts");
(b) onset, sex or inheritance pattern is incompatible with the letter;
(c) the letter describes a characteristic or pathognomonic combination of features that one
    candidate explains clearly better (see its [match] list);
(d) a gene, biomarker or test result named in the letter favours a candidate.
Candidates that share a group id are subtypes of one disease series, allelic disorders of the
same gene, or duplicate entries: do not reorder them among themselves unless the letter gives a
distinguishing feature or gene. Annotations are authoritative; do not demote a candidate because
it is rare, newly described or unfamiliar to you. Missing findings are weak evidence; case
reports are incomplete.
Answer with JSON only:
{"ranking": ["<disease id>", ... every candidate id exactly once, most likely first],
 "changes": [{"id": "<disease id>", "from": <ranker rank>, "to": <new rank>, "reason": "<short>"}]}"""

SYSTEM_V4 = """You are a clinical geneticist supporting research (not clinical care) on rare-disease
phenotype matching. A phenotype-similarity ranker has scored all ~9,000 gene-linked diseases
against the patient's findings; you see its top candidates with their ranker rank and score.
The ranker's order is a good prior. Reorder where the letter gives specific evidence:
(a) a very frequent or obligatory feature of a candidate is explicitly absent in the letter;
(b) onset, sex or inheritance pattern is incompatible with the letter;
(c) the letter describes a characteristic combination of features that one candidate explains
    clearly better (see its [match] list);
(d) a gene, biomarker or test result named in the letter favours a candidate.
The finding lists were extracted from the letter by a machine and contain errors: findings the
letter denies (often in lists such as "no A, B or C") can appear as patient findings and in
[match], and a finding can wrongly appear as absent. The letter is authoritative: ignore matches
and conflicts that the letter contradicts.
Candidates that share a group id are subtypes of one disease series, allelic disorders of the
same gene, or the same disease under OMIM and ORPHA ids. Keep entries of the same disease next to
each other. Order group members by onset, inheritance or distinguishing features when the letter
gives them; otherwise keep the ranker's order. Annotations are authoritative; do not demote a
candidate because it is rare, newly described or unfamiliar to you. Missing findings are weak
evidence; case reports are incomplete.
Answer with JSON only:
{"ranking": ["<disease id>", ... every candidate id exactly once, most likely first],
 "changes": [{"id": "<disease id>", "from": <ranker rank>, "to": <new rank>, "reason": "<short>"}]}"""

# v5: v4 without "keep entries of the same disease next to each other", which v4 read as
# "move whole groups as blocks" (96/111 German replies)
SYSTEM_V5 = """You are a clinical geneticist supporting research (not clinical care) on rare-disease
phenotype matching. A phenotype-similarity ranker has scored all ~9,000 gene-linked diseases
against the patient's findings; you see its top candidates with their ranker rank and score.
The ranker's order is a good prior. Reorder where the letter gives specific evidence:
(a) a very frequent or obligatory feature of a candidate is explicitly absent in the letter;
(b) onset, sex or inheritance pattern is incompatible with the letter;
(c) the letter describes a characteristic combination of features that one candidate explains
    clearly better (see its [match] list);
(d) a gene, biomarker or test result named in the letter favours a candidate.
The finding lists were extracted from the letter by a machine and contain errors: findings the
letter denies (often in lists such as "no A, B or C") can appear as patient findings and in
[match], and a finding can wrongly appear as absent. The letter is authoritative: ignore matches
and conflicts that the letter contradicts.
Candidates that share a group id are subtypes of one disease series, allelic disorders of the
same gene, or the same disease under OMIM and ORPHA ids. A group is not a block: judge every
candidate on its own evidence and never move a group as a whole. Within a group, keep the ranker's
order unless the letter gives onset, inheritance or a distinguishing feature. Annotations are authoritative; do not demote a
candidate because it is rare, newly described or unfamiliar to you. Missing findings are weak
evidence; case reports are incomplete.
Answer with JSON only:
{"ranking": ["<disease id>", ... every candidate id exactly once, most likely first],
 "changes": [{"id": "<disease id>", "from": <ranker rank>, "to": <new rank>, "reason": "<short>"}]}"""

PASSES = int(
    __import__("os").environ.get("RERANK_PASSES", "1")
)  # shuffled consensus passes


def series_stem(label: str) -> str:
    """Disease label without a trailing subtype designator ("..., type IID", "... 2", "... 1H")."""
    stem = re.sub(
        r"[,;]?\s*(type|form|group)?\s*\b([0-9]+[a-z]?|[ivx]+[a-z]?)$",
        "",
        label.lower().strip(),
    )
    return stem.strip(" ,;")


def candidates_text_v3(
    ix, order, shown, s, present, excluded, meta, gene_of, max_genes: int | None = None
) -> str:
    """v2 block plus causal genes, group ids (shared gene or series stem), conflicts with absent
    findings, and pool-generic features dropped. `shown` is the presentation order.
    max_genes (v4): a shared gene links two candidates only when both have at most this many
    genes, so multi-gene ORPHA umbrella entries no longer chain unrelated groups together."""
    anc = {t: set(ix.ancestors(t).tolist()) for t in present}
    parent = {d: d for d in order}

    def find(d: int) -> int:
        while parent[d] != d:
            d = parent[d]
        return d

    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            ga, gb = gene_of.get(a, set()), gene_of.get(b, set())
            if max_genes is not None and max(len(ga), len(gb)) > max_genes:
                ga = gb = set()
            if ga & gb or series_stem(ix.disease_labels[a]) == series_stem(
                ix.disease_labels[b]
            ):
                parent[find(b)] = find(a)
    roots = list(dict.fromkeys(find(d) for d in order))
    group = {d: f"G{roots.index(find(d)) + 1}" for d in order}
    feat_count: dict[int, int] = {}
    for d in order:
        for t in set(ix.ann_term[ix.dis_ptr[d] : ix.dis_ptr[d + 1]].tolist()):
            feat_count[t] = feat_count.get(t, 0) + 1
    generic = {t for t, c in feat_count.items() if c >= 0.2 * len(order)}
    rank_of = {d: i + 1 for i, d in enumerate(order)}
    lines = []
    for i, d in enumerate(shown, 1):
        a, b = ix.dis_ptr[d], ix.dis_ptr[d + 1]
        ann = list(range(a, b))
        matched: dict[int, str] = {}
        for p in present:
            best, arg = 0.0, -1
            for j in ann:
                t = ix.ann_term[j]
                t_anc = set(ix.ancestors(t).tolist())
                if t not in anc[p] and p not in t_anc:
                    continue
                v = max((ix.ic[c] for c in anc[p] & t_anc), default=0.0) / ix.self_ic[p]
                if v > best:
                    best, arg = v, j
            if arg >= 0 and best >= 0.5:
                matched.setdefault(arg, ix.labels[p])
        conflicts = sorted(
            {
                f"{ix.labels[ix.ann_term[j]]} ({freq_word(ix.ann_f[j])}) vs absent: {ix.labels[e]}"
                for j in ann
                for e in excluded
                if e in set(ix.ancestors(ix.ann_term[j]).tolist())
            }
        )
        other = sorted(
            (j for j in ann if j not in matched and ix.ann_term[j] not in generic),
            key=lambda j: (
                -ix.ann_f[j] * ix.self_ic[ix.ann_term[j]],
                -ix.self_ic[ix.ann_term[j]],
            ),
        )[:N_PHENO]
        m = meta.get(ix.disease_ids[d], {})
        genes = ", ".join(sorted(gene_of.get(d, set()))) or "?"
        lines.append(
            f"{i}. {ix.disease_ids[d]} | {ix.disease_labels[d]} | gene {genes} | group {group[d]} | "
            f"ranker rank {rank_of[d]}, score {s[d]:.3f}\n"
            f"   inheritance: {', '.join(sorted(m.get('inheritance', []))) or 'not annotated'}; "
            f"onset: {', '.join(sorted(m.get('onset', []))) or 'not annotated'}\n"
            f"   [match] {'; '.join(f'{ix.labels[ix.ann_term[j]]} ({freq_word(ix.ann_f[j])}) <- patient: {pl}' for j, pl in matched.items()) or 'none'}\n"
            f"   conflicts: {'; '.join(conflicts[:8]) or 'none'}\n"
            f"   other distinctive features: {'; '.join(f'{ix.labels[ix.ann_term[j]]} ({freq_word(ix.ann_f[j])})' for j in other) or 'none'}"
        )
    return "\n".join(lines)


def main(hpo_dir: Path, split: str, model: str) -> None:
    ix = scorer.build_index(hpo_dir)
    meta = disease_meta(hpo_dir) if PROMPT != "v1" else {}
    gene_of: dict[int, set[str]] = {}
    for g, ds in ix.gene_diseases.items():
        for d in ds:
            gene_of.setdefault(d, set()).add(g)
    cases = [
        c
        for c in load_jsonl(HERE / "cases.jsonl")
        if not (c["split"] == "test" and c["cited"])
    ]
    if split != "all":
        cases = [c for c in cases if c["split"] == split]
    for lang in LANGS:
        ext = {
            e["case"]: e for e in load_jsonl(RESULTS / f"extractions_{lang}_llm.jsonl")
        }
        suffix = ("" if PROMPT == "v1" else f"_{PROMPT}") + (
            f"_p{PASSES}" if PASSES > 1 else ""
        )
        out_path = RESULTS / f"rerank_{model}_top{TOP_N}{suffix}_{lang}.jsonl"
        done = {r["case"] for r in load_jsonl(out_path)} if out_path.exists() else set()
        lock = threading.Lock()

        def run(
            case: dict,
            lang: str = lang,
            ext: dict = ext,
            out_path: Path = out_path,
            lock: threading.Lock = lock,
        ) -> None:
            terms = [
                t[0] for t in ext[case["id"]]["terms"] if t[1] == "present" and not t[2]
            ]
            s = scorer.score(ix, dict.fromkeys(terms, 1.0))
            true = ix.disease_ids.index(case["disease"])
            if s is None:
                rec = {
                    "case": case["id"],
                    "lang": lang,
                    "scorer_rank": (len(ix.disease_ids) + 1) / 2,
                    "rerank_rank": None,
                }
            else:
                order = [int(d) for d in np.argsort(-s, kind="stable")[:TOP_N]]
                letter = json.loads(
                    (
                        HERE
                        / "letters"
                        / lang
                        / (re.sub(r"[^\w.-]", "_", case["id"]) + ".json")
                    ).read_text()
                )
                labels = [ix.labels[ix.index[t]] for t in terms if t in ix.index]
                if PROMPT == "v1":
                    system = SYSTEM
                    prompt = (
                        f"Referral letter:\n{letter['text']}\n\nExtracted HPO terms: {'; '.join(labels)}\n\n"
                        f"Candidates:\n{candidates_text(ix, order, s)}"
                    )
                else:
                    system = SYSTEM_V2
                    present = [
                        ix.index[t] for t in dict.fromkeys(terms) if t in ix.index
                    ]
                    absent = [
                        ix.labels[ix.index[t[0]]]
                        for t in ext[case["id"]]["terms"]
                        if t[2] and t[0] in ix.index
                    ]
                    prompt = (
                        f"Referral letter:\n{letter['text']}\n\n"
                        f"Patient findings (HPO): {'; '.join(labels)}\n"
                        f"Explicitly absent findings: {'; '.join(absent) or 'none recorded'}\n\n"
                        f"Candidates in ranker order:\n{candidates_text_v2(ix, order, s, present, meta)}"
                    )
                if PROMPT in ("v3", "v4", "v5"):
                    system = {"v3": SYSTEM_V3, "v4": SYSTEM_V4, "v5": SYSTEM_V5}[PROMPT]
                    excluded = [
                        ix.index[t[0]]
                        for t in ext[case["id"]]["terms"]
                        if t[2] and t[0] in ix.index
                    ]
                    head = prompt.split("Candidates in ranker order:")[0]
                    if PROMPT in ("v4", "v5"):
                        # an absent term that is an ancestor of a present one is an extraction
                        # error ("Vascular dilatation" absent vs aortic aneurysm present)
                        pres_anc = {
                            a for p in present for a in ix.ancestors(p).tolist()
                        }
                        excluded = [e for e in excluded if e not in pres_anc]
                        head = (
                            f"Referral letter:\n{letter['text']}\n\n"
                            f"Patient findings (machine-extracted, check against the letter): {'; '.join(labels)}\n"
                            f"Absent findings (machine-extracted): {'; '.join(ix.labels[e] for e in dict.fromkeys(excluded)) or 'none recorded'}\n\n"
                        )
                    prompts = []
                    for k in range(PASSES):
                        shown = list(order)
                        if (
                            k > 0
                        ):  # shuffled presentation; the ranker rank stays visible
                            np.random.default_rng(
                                [SEED, k, zlib.crc32(case["id"].encode())]
                            ).shuffle(shown)
                        prompts.append(
                            head
                            + "Candidates (presentation order is arbitrary; see ranker rank):\n"
                            + candidates_text_v3(
                                ix,
                                order,
                                shown,
                                s,
                                present,
                                excluded,
                                meta,
                                gene_of,
                                max_genes=None if PROMPT == "v3" else 2,
                            )
                        )
                else:
                    prompts = [prompt]
                passes: list[list[str]] = []
                replies: list[str] = []
                for pr in prompts:
                    try:
                        replies.append(ask(model, system, pr))
                        passes.append(parse_ranking(replies[-1]))
                    except (
                        Exception
                    ) as exc:  # this pass falls back to scorer order, logged
                        print(
                            "FAILED",
                            case["id"],
                            lang,
                            type(exc).__name__,
                            str(exc)[:120],
                            flush=True,
                        )
                        passes.append([])
                ids = [ix.disease_ids[d] for d in order]
                # Borda consensus: mean position over passes (missing ids keep their ranker position)
                pos = {
                    x: np.mean(
                        [
                            (p.index(x) if x in p else len(p) + ids.index(x))
                            for p in passes
                        ]
                    )
                    for x in ids
                }
                ranking = sorted(ids, key=lambda x: (pos[x], ids.index(x)))
                valid = [
                    ix.disease_ids.index(x)
                    for x in dict.fromkeys(ranking)
                    if x in ix.disease_ids and ix.disease_ids.index(x) in order
                ]
                final = valid + [
                    d for d in order if d not in valid
                ]  # missing candidates keep scorer order
                rec = {
                    "case": case["id"],
                    "lang": lang,
                    "scorer_rank": scorer.avg_rank(s, true),
                    "rerank_rank": final.index(true) + 1 if true in final else None,
                    "valid_ids": len(valid),
                    "scorer_order": [ix.disease_ids[d] for d in order],
                    "final": [ix.disease_ids[d] for d in final],
                    "passes": passes,
                    "replies": replies,  # raw model text, for debugging
                }
            with lock, out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        todo = [c for c in cases if c["id"] not in done]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run, todo))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
