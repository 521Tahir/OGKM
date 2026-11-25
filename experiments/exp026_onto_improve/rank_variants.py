"""Rank every ontology variant against Simple RAG on the benchmark, so the config
carried forward is chosen on measured metrics rather than on a proxy."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from src.evaluation.stats_tests import paired_ttest
from src.utils.proctitle import set_proctitle

BENCH = ROOT / "experiments" / "exp025_multihop" / "outputs" / "jeju_bench_200.jsonl"
OUT = Path(__file__).resolve().parent / "outputs"
# Only arms that are actually scored on this split are ranked; `hybrid` has
# contexts on bench but was only generated on the test split.
ARMS = ["rag", "onto", "onto_c", "onto_v3", "onto_v4", "onto_v5", "hybrid"]


def main() -> None:
    set_poc = set_proctitle(); _ = set_poc
    meta = {json.loads(l)["record_id"]: json.loads(l) for l in open(BENCH, encoding="utf-8")}
    cells = {}
    for p in sorted(C.SCORE_DIR.glob("bench__*.jsonl")):
        parts = p.stem.split("__")
        cells[(parts[1], parts[2])] = {json.loads(l)["record_id"]: json.loads(l)
                                       for l in open(p, encoding="utf-8")}
    models = [m["key"] for m in C.MODELS if (m["key"], "rag") in cells]
    arms = [a for a in ARMS if any(k[1] == a for k in cells)]
    subsets = {"all": lambda r: True,
               "multihop": lambda r: meta[r]["kind"] == "multihop",
               "single": lambda r: meta[r]["kind"] == "single"}

    report, L = {}, ["# Ontology variant ranking on the 200-item benchmark", ""]
    for subset, pred in subsets.items():
        L += [f"## {subset}", "",
              "| Arm | " + " | ".join(C.METRIC_LABELS[k] for k in C.METRICS) +
              " | mean Δ vs RAG | metrics won |", "|" + "---|" * (len(C.METRICS) + 3)]
        macro = {}
        for arm in arms:
            vals = {}
            for k in C.METRICS:
                per_model = []
                for mk in models:
                    rows = [r for rid, r in cells.get((mk, arm), {}).items() if pred(rid)]
                    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
                    if v:
                        per_model.append(statistics.mean(v))
                vals[k] = round(statistics.mean(per_model), 4) if per_model else None
            macro[arm] = vals
        for arm in arms:
            v = macro[arm]
            if arm == "rag":
                d, won = "—", "—"
            else:
                deltas = [v[k] - macro["rag"][k] for k in C.METRICS
                          if isinstance(v[k], float) and isinstance(macro["rag"][k], float)]
                d = f"{statistics.mean(deltas):+.4f}"
                won = f"{sum(1 for x in deltas if x > 0)}/{len(deltas)}"
            L.append(f"| {arm} | " + " | ".join(f"{v[k]:.4f}" if isinstance(v[k], float) else "n/a"
                                                for k in C.METRICS) + f" | {d} | {won} |")
        report[subset] = macro
        L.append("")

    # per-model significance for the best arm on multihop
    def _gap(a):
        vals = [report["multihop"][a][k] - report["multihop"]["rag"][k]
                for k in C.METRICS
                if isinstance(report["multihop"][a].get(k), float)
                and isinstance(report["multihop"]["rag"].get(k), float)]
        return statistics.mean(vals) if vals else float("-inf")

    cands = [a for a in arms if a != "rag" and _gap(a) != float("-inf")]
    best = max(cands, key=_gap) if cands else "onto"
    L += [f"## Best variant on multi-hop: **{best}** — per-model paired tests", "",
          "| Model | Metric | RAG | " + best + " | Δ | p |", "|---|---|---|---|---|---|"]
    for mk in models:
        r, o = cells.get((mk, "rag")), cells.get((mk, best))
        if not (r and o):
            continue
        ids = [i for i in r if i in o and meta[i]["kind"] == "multihop"]
        for k in C.METRICS:
            pairs = [(r[i][k], o[i][k]) for i in ids
                     if isinstance(r[i].get(k), (int, float)) and isinstance(o[i].get(k), (int, float))]
            if len(pairs) < 2:
                continue
            t = paired_ttest([x for x, _ in pairs], [y for _, y in pairs], "rag", best, metric=k)
            L.append(f"| {C.MODEL_BY_KEY[mk]['name']} | {C.METRIC_LABELS[k]} | {t.mean_a:.4f} | "
                     f"{t.mean_b:.4f} | {t.delta:+.4f}{' *' if t.significant else ''} | "
                     f"{t.p_value:.2e} |")
    (OUT / "variant_ranking.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "variant_ranking.json").write_text(
        json.dumps({"macro": report, "best_multihop": best}, indent=2), encoding="utf-8")
    print("\n".join(L))
    print(f"\nBEST ONTOLOGY VARIANT (multi-hop): {best}")


if __name__ == "__main__":
    main()
