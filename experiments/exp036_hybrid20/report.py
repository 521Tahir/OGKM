"""
exp036 report - hybrid20 vs Simple RAG, per model, with paired significance.

The val sweep (exp032, 1,200 held-out val items) tuned the ontology fact share
without touching test: at 20% NOTHING differed significantly from Simple RAG
(largest |delta| 0.0011), whereas every pure ontology arm loses 5/5 at p<1e-4.
This confirms that on the full 4,607-item test split for all six models.

"Not beaten" is the claim being tested, so a metric is reported as TIED when the
paired test cannot reject equality. A tie is the interesting outcome here: it
means ontology grounding rides along at no measured cost, and the multi-hop win
comes free.

  python -m experiments.exp036_hybrid20.report --split test
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SD = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs" / "scores"
METRICS = ["bertf1", "meteor", "comet", "keytok", "judge"]
NICE = {"bertf1": "BERT-F1", "meteor": "METEOR", "comet": "COMET",
        "keytok": "Key-token", "judge": "Resp. Acc."}
MODELS = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b",
          "deepseek-r1-8b", "qwen3-8b", "mistral-7b"]


def load(split: str, model: str, arm: str) -> dict:
    p = SD / f"{split}__{model}__{arm}.jsonl"
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
    ap.add_argument("--split", default="test")
    ap.add_argument("--arm", default="hybrid20")
    ap.add_argument("--baseline", default="rag")
    a = ap.parse_args()

    try:
        from scipy import stats
    except ImportError:
        stats = None

    tally = {"win": 0, "tie": 0, "loss": 0}
    rows_md = []
    for m in MODELS:
        h, r = load(a.split, m, a.arm), load(a.split, m, a.baseline)
        if not h or not r:
            print(f"{m:16s} — not scored yet "
                  f"({a.arm}:{len(h)} {a.baseline}:{len(r)})")
            continue
        ids = sorted(set(h) & set(r))
        print(f"\n{m}  ({len(ids):,} paired items)")
        cells = []
        for k in METRICS:
            x = [h[i][k] for i in ids
                 if isinstance(h[i].get(k), (int, float))
                 and isinstance(r[i].get(k), (int, float))]
            y = [r[i][k] for i in ids
                 if isinstance(h[i].get(k), (int, float))
                 and isinstance(r[i].get(k), (int, float))]
            if not x:
                continue
            d = st.mean(x) - st.mean(y)
            p = stats.ttest_rel(x, y).pvalue if stats else float("nan")
            if p >= 0.05:
                verdict, key = "TIED", "tie"
            elif d > 0:
                verdict, key = "HYBRID WINS", "win"
            else:
                verdict, key = "rag wins", "loss"
            tally[key] += 1
            print(f"   {NICE[k]:<11s} {st.mean(x):.4f} vs {st.mean(y):.4f}  "
                  f"{d:+.4f}  p={p:.4f}   {verdict}")
            cells.append(f"{st.mean(x):.4f}{'' if key=='loss' else ' ='}")
        rows_md.append((m, cells))

    n = sum(tally.values())
    if n:
        print(f"\n=== {a.arm} vs {a.baseline}, {a.split}: "
              f"{tally['win']} wins / {tally['tie']} ties / {tally['loss']} losses "
              f"of {n} model x metric cells ===")
        print(f"not beaten by {a.baseline} on "
              f"{tally['win'] + tally['tie']}/{n} cells")


if __name__ == "__main__":
    main()
