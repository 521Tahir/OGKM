"""
exp037 summary - all six models, both splits, arm-dedicated adapters.

Emits two tables:
  A. onto_v3 @ dedicated  -  onto_v3 @ mixed   (how much the harness bias cost)
  B. onto_v3 @ dedicated  vs rag @ dedicated   (the matched verdict)

A metric counts as a WIN only when the paired t-test AND Wilcoxon both reject at
0.05. Several cells in this experiment pass t and fail Wilcoxon - a few large
differences rather than a consistent shift - and reporting those as wins would
overstate the result.

  python -m experiments.exp037_dedicated_adapter.summary --split test
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scipy import stats

OUT = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs"
METRICS = ["bertf1", "meteor", "comet", "keytok", "judge"]
NICE = {"bertf1": "BERT-F1", "meteor": "METEOR", "comet": "COMET",
        "keytok": "Key-tok", "judge": "Resp.Acc"}
MODELS = ["llama3.2-1b", "llama3.2-3b", "mistral-7b",
          "llama3.1-8b", "qwen3-8b", "deepseek-r1-8b"]


def load(d: Path, split: str, model: str, arm: str) -> dict:
    p = d / f"{split}__{model}__{arm}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["record_id"]] = r
    return out


def cmp(lhs: dict, rhs: dict) -> dict:
    ids = sorted(set(lhs) & set(rhs))
    res = {}
    for m in METRICS:
        x = [lhs[i][m] for i in ids
             if isinstance(lhs[i].get(m), (int, float))
             and isinstance(rhs[i].get(m), (int, float))]
        y = [rhs[i][m] for i in ids
             if isinstance(lhs[i].get(m), (int, float))
             and isinstance(rhs[i].get(m), (int, float))]
        if not x:
            res[m] = None
            continue
        d = st.mean(x) - st.mean(y)
        pt = stats.ttest_rel(x, y).pvalue
        try:
            pw = stats.wilcoxon(x, y).pvalue
        except ValueError:
            pw = 1.0
        robust = pt < 0.05 and pw < 0.05
        res[m] = (d, pt, pw, robust, len(x))
    return res


def table(title: str, rows: dict, note: str) -> None:
    print(f"\n### {title}\n")
    print("| Model | " + " | ".join(NICE[m] for m in METRICS) + " | robust |")
    print("|---" * (len(METRICS) + 2) + "|")
    tally = [0, 0, 0]
    for mdl, r in rows.items():
        cells, nrob = [], 0
        for m in METRICS:
            if r.get(m) is None:
                cells.append("—")
                continue
            d, pt, pw, robust, _ = r[m]
            mark = "**" if robust else ""
            cells.append(f"{mark}{d:+.4f}{mark}")
            if robust:
                nrob += 1
                tally[0 if d > 0 else 2] += 1
            else:
                tally[1] += 1
        print(f"| {mdl} | " + " | ".join(cells) + f" | {nrob}/5 |")
    print(f"\n{note}")
    print(f"robust gains {tally[0]} / not significant {tally[1]} / "
          f"robust losses {tally[2]}  (of {sum(tally)} cells)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    mixed, ded = OUT / "scores", OUT / "scores_ded"

    A, B = {}, {}
    for m in MODELS:
        om, od = load(mixed, a.split, m, "onto_v3"), load(ded, a.split, m, "onto_v3")
        rd = load(ded, a.split, m, "rag")
        if od and om:
            A[m] = cmp(od, om)
        if od and rd:
            B[m] = cmp(od, rd)

    print(f"# exp037 - arm-dedicated adapters, split = {a.split}")
    table("A. Cost of the shared-adapter bias:  onto_v3 dedicated - onto_v3 mixed",
          A, "Positive = the dedicated adapter improved the ontology arm. "
             "Bold = significant on t AND Wilcoxon.")
    table("B. Matched verdict:  onto_v3 dedicated - rag dedicated",
          B, "Positive = ontology beats its OWN-adapter RAG control. "
             "Bold = significant on t AND Wilcoxon.")


if __name__ == "__main__":
    main()
