"""Sanity check, not an endpoint: RareBench gold HPO profiles through the scorer (issue #451).

    python -I rarebench.py <hpo_dir> <rarebench_data_dir>

A case counts when one of its diagnoses is a gene-linked disease of the index; its rank is
the best rank over those diagnoses. Writes results/rarebench_sanity.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: own code only
import numpy as np  # noqa: E402
import scorer  # noqa: E402
from analysis import RESULTS, write_csv  # noqa: E402


def main(hpo_dir: Path, data_dir: Path) -> None:
    ix = scorer.build_index(hpo_dir)
    pos = {d: i for i, d in enumerate(ix.disease_ids)}
    rows = []
    for cohort in ("MME", "LIRICAL", "HMS", "RAMEDIS"):
        lines = (data_dir / f"{cohort}.jsonl").read_text(encoding="utf-8").splitlines()
        ranks = []
        for line in lines:
            case = json.loads(line)
            targets = [pos[d] for d in case["RareDisease"] if d in pos]
            s = scorer.score(ix, dict.fromkeys(case["Phenotype"], 1.0))
            if targets and s is not None:
                ranks.append(min(scorer.avg_rank(s, t) for t in targets))
        r = np.array(ranks)
        rows.append(
            {
                "cohort": cohort,
                "cases": len(lines),
                "scored": len(r),
                "hit1": float((r <= 1).mean()),
                "hit10": float((r <= 10).mean()),
                "median_rank": float(np.median(r)),
            }
        )
        print(rows[-1])
    write_csv(RESULTS / "rarebench_sanity.csv", rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
