"""
Goal 1 — does ontology grounding help more, less, or the same as models get smaller?

Compares Simple RAG against the best ontology config (onto_c) across the full
parameter ladder: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B. The 1B/3B checkpoints
come from the ungated unsloth mirrors (meta-llama/Llama-3.2-* is gated for this
account); weights are identical to Meta's release.

Hypothesis worth testing: a smaller model has less parametric knowledge to fall
back on, so explicit grounding should matter MORE, not less. If the ontology's
advantage grows as capacity shrinks, that is a substantive argument for it.

  python -m experiments.exp029_scale.report_scale --split bench
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from src.evaluation.stats_tests import paired_ttest
from src.utils.proctitle import set_proctitle

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
BENCH = ROOT / "experiments" / "exp025_multihop" / "outputs" / "jeju_bench_200.jsonl"
LADDER = [("llama3.2-1b", "Llama-3.2-1B", 1.24), ("llama3.2-3b", "Llama-3.2-3B", 3.21),
          ("llama3.1-8b", "Llama-3.1-8B", 8.03)]
ARMS = [("rag", "Simple RAG"), ("onto_c", "Ontology"), ("base", "Vanilla (no FT)")]


def _mean(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return round(statistics.mean(v), 4) if v else None


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bench")
    a = ap.parse_args()

    meta = ({json.loads(l)["record_id"]: json.loads(l) for l in open(BENCH, encoding="utf-8")}
            if a.split == "bench" else {})
    cells = {}
    for p in sorted(C.SCORE_DIR.glob(f"{a.split}__*.jsonl")):
        parts = p.stem.split("__")
        cells[(parts[1], parts[2])] = {json.loads(l)["record_id"]: json.loads(l)
                                       for l in open(p, encoding="utf-8")}
    have = [(mk, name, b) for mk, name, b in LADDER if (mk, "rag") in cells]
    if not have:
        raise SystemExit("no ladder cells scored yet")

    subsets = ({"all": lambda r: True,
                "multihop": lambda r: meta[r]["kind"] == "multihop",
                "single": lambda r: meta[r]["kind"] == "single"} if meta
               else {"all": lambda r: True})

    L = [f"# Scale ladder — does grounding matter more at small capacity? (`{a.split}`)", "",
         "Simple RAG vs the cleaned-ontology arm (`onto_c`) across 1B / 3B / 8B, same adapter "
         "recipe, same prompts, same decoding at every scale.", ""]
    payload = {}
    for subset, pred in subsets.items():
        L += [f"## {subset}", "",
              "| Model | Params (B) | Condition | " +
              " | ".join(C.METRIC_LABELS[k] for k in C.METRICS) + " |",
              "|" + "---|" * (len(C.METRICS) + 3)]
        for mk, name, nb in have:
            for arm, alab in ARMS:
                rows = [r for rid, r in cells.get((mk, arm), {}).items() if pred(rid)]
                if not rows:
                    continue
                d = {k: _mean(rows, k) for k in C.METRICS}
                payload[f"{subset}__{mk}__{arm}"] = d
                L.append(f"| {name if arm == 'rag' else ''} | {nb if arm == 'rag' else ''} | "
                         f"{alab} | " + " | ".join(f"{d[k]:.4f}" if isinstance(d[k], float)
                                                   else "n/a" for k in C.METRICS) + " |")
        L.append("")

        L += ["### Ontology advantage by scale", "",
              "| Model | Params (B) | mean Δ vs RAG | metrics won | significant |",
              "|---|---|---|---|---|"]
        for mk, name, nb in have:
            r, o = cells.get((mk, "rag")), cells.get((mk, "onto_c"))
            if not (r and o):
                continue
            ids = [i for i in r if i in o and pred(i)]
            deltas, sig = [], 0
            for k in C.METRICS:
                pairs = [(r[i][k], o[i][k]) for i in ids
                         if isinstance(r[i].get(k), (int, float))
                         and isinstance(o[i].get(k), (int, float))]
                if len(pairs) < 2:
                    continue
                t = paired_ttest([x for x, _ in pairs], [y for _, y in pairs],
                                 "rag", "onto_c", metric=k)
                deltas.append(t.delta)
                sig += int(t.significant and t.delta > 0)
            if deltas:
                payload[f"{subset}__{mk}__delta"] = round(statistics.mean(deltas), 4)
                L.append(f"| {name} | {nb} | {statistics.mean(deltas):+.4f} | "
                         f"{sum(1 for d in deltas if d > 0)}/{len(deltas)} | {sig}/{len(deltas)} |")
        L.append("")

    # the actual question: does the gap grow as capacity shrinks?
    gaps = [(nb, payload.get(f"multihop__{mk}__delta") or payload.get(f"all__{mk}__delta"))
            for mk, _, nb in have]
    gaps = [(b, g) for b, g in gaps if g is not None]
    if len(gaps) >= 2:
        trend = ("grows as models shrink" if gaps[0][1] > gaps[-1][1]
                 else "grows with model size" if gaps[0][1] < gaps[-1][1] else "flat")
        L += ["## Reading", "",
              f"Ontology advantage by parameter count: " +
              ", ".join(f"{b}B → {g:+.4f}" for b, g in gaps) + ".", "",
              f"The advantage **{trend}**.", ""]
    (OUT / f"scale_{a.split}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / f"scale_{a.split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {OUT}/scale_{a.split}.md")


if __name__ == "__main__":
    main()
