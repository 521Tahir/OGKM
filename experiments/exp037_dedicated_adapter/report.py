"""
exp037 report - does an ARM-DEDICATED adapter change the ontology verdict?

Compares, for one model, four cells:

    onto_v3 @ mixed adapter    (the eleven-approach baseline)
    rag     @ mixed adapter    (the incumbent winner)
    onto_v3 @ onto_v3 adapter  (NEW: weights tuned on ontology contexts)
    rag     @ rag adapter      (NEW: matched control, so the fight stays fair)

The headline question is the last-vs-last comparison. Comparing a dedicated
ontology adapter against a MIXED rag adapter would be rigged; both arms get
arm-dedicated weights or neither does.

Paired t-test and Wilcoxon over the record_ids present in every cell, so all
four columns describe exactly the same items.

  python -m experiments.exp037_dedicated_adapter.report --model llama3.2-1b --split test
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

METRICS = ["bertf1", "meteor", "comet", "keytok", "judge"]
NICE = {"bertf1": "BERT-F1", "meteor": "METEOR", "comet": "COMET",
        "keytok": "Key-token", "judge": "Resp. Acc."}


def load(score_dir: Path, split: str, model: str, arm: str) -> dict:
    p = score_dir / f"{split}__{model}__{arm}.jsonl"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-1b")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    out = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs"
    mixed_dir, ded_dir = out / "scores", out / "scores_ded"

    cells = {
        "onto_v3 @ mixed": load(mixed_dir, a.split, a.model, "onto_v3"),
        "rag @ mixed": load(mixed_dir, a.split, a.model, "rag"),
        "onto_v3 @ DEDICATED": load(ded_dir, a.split, a.model, "onto_v3"),
        "rag @ DEDICATED": load(ded_dir, a.split, a.model, "rag"),
    }
    present = {k: v for k, v in cells.items() if v}
    missing = [k for k, v in cells.items() if not v]
    if missing:
        print(f"note: not yet scored -> {', '.join(missing)}\n")
    if len(present) < 2:
        raise SystemExit("need at least two scored cells to compare")

    ids = set.intersection(*(set(v) for v in present.values()))
    ids = sorted(ids)
    print(f"{a.model} / {a.split} - {len(ids):,} items common to "
          f"{len(present)} cells\n")

    hdr = f"{'cell':22s}" + "".join(f"{NICE[m]:>12s}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    means = {}
    for name, d in present.items():
        row = {}
        for m in METRICS:
            vals = [d[i][m] for i in ids if isinstance(d[i].get(m), (int, float))]
            row[m] = st.mean(vals) if vals else float("nan")
        means[name] = row
        print(f"{name:22s}" + "".join(f"{row[m]:12.4f}" for m in METRICS))

    # ── the two comparisons that matter ───────────────────────────────────
    try:
        from scipy import stats
    except ImportError:
        print("\n(scipy unavailable - means only)")
        return

    def compare(lhs: str, rhs: str) -> None:
        if lhs not in present or rhs not in present:
            return
        print(f"\n{lhs}  vs  {rhs}")
        wins = 0
        for m in METRICS:
            x = [present[lhs][i][m] for i in ids
                 if isinstance(present[lhs][i].get(m), (int, float))
                 and isinstance(present[rhs][i].get(m), (int, float))]
            y = [present[rhs][i][m] for i in ids
                 if isinstance(present[lhs][i].get(m), (int, float))
                 and isinstance(present[rhs][i].get(m), (int, float))]
            if not x:
                continue
            d = st.mean(x) - st.mean(y)
            _, p_t = stats.ttest_rel(x, y)
            try:
                _, p_w = stats.wilcoxon(x, y)
            except ValueError:
                p_w = float("nan")
            sig = "*" if p_t < 0.05 else " "
            verdict = ("ONTO WINS" if d > 0 and p_t < 0.05 else
                       "onto ahead (ns)" if d > 0 else
                       "RAG WINS" if p_t < 0.05 else "rag ahead (ns)")
            wins += 1 if d > 0 else 0
            print(f"  {NICE[m]:<11s} {d:+.4f}  t p={p_t:.4f}{sig}  "
                  f"wilcoxon p={p_w:.4f}   {verdict}")
        print(f"  -> ontology nominally ahead on {wins}/5 metrics")

    compare("onto_v3 @ mixed", "rag @ mixed")
    compare("onto_v3 @ DEDICATED", "rag @ DEDICATED")
    compare("onto_v3 @ DEDICATED", "onto_v3 @ mixed")


if __name__ == "__main__":
    main()
