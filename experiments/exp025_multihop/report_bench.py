"""
exp025 report — RAG vs Ontology on the 200-item Jeju benchmark, split by hop count.

The question this answers: does structured ontology grounding help *more* on
multi-hop questions (where the answer requires chaining two facts) than on
single-hop lookups? exp024 showed RAG ahead on free-form itinerary generation;
multi-hop is where a knowledge graph should have the advantage, if anywhere.
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
from src.evaluation.stats_tests import paired_ttest, wilcoxon_signed_rank
from src.utils.proctitle import set_proctitle

BENCH = Path(__file__).resolve().parent / "outputs" / "jeju_bench_200.jsonl"
OUT = Path(__file__).resolve().parent / "outputs"
ARM_LABEL = {"rag": "Simple RAG", "onto": "Ontology v1", "onto_c": "Ontology (cleaned)",
             "onto_v3": "Ontology v3", "onto_v4": "Ontology v4", "onto_v5": "Ontology v5",
             "base": "Vanilla (no FT)"}
ARM_ORDER = ["rag", "onto", "onto_c", "onto_v3", "onto_v4", "onto_v5", "base"]


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def _mean(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return round(statistics.mean(v), 4) if v else None


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bench")
    a = ap.parse_args()

    meta = {json.loads(l)["record_id"]: json.loads(l) for l in open(BENCH, encoding="utf-8")}
    cells: dict[tuple[str, str], dict] = {}
    for p in sorted(C.SCORE_DIR.glob(f"{a.split}__*.jsonl")):
        parts = p.stem.split("__")
        cells[(parts[1], parts[2])] = {json.loads(l)["record_id"]: json.loads(l)
                                       for l in open(p, encoding="utf-8")}
    if not cells:
        raise SystemExit("no scored bench cells yet")

    models = [m["key"] for m in C.MODELS if any(k[0] == m["key"] for k in cells)]
    arms = [x for x in ARM_ORDER if any(k[1] == x for k in cells)]
    subsets = {"all": lambda r: True,
               "multihop": lambda r: meta[r]["kind"] == "multihop",
               "single": lambda r: meta[r]["kind"] == "single"}

    L = ["# exp025 — RAG vs Ontology on 200 Jeju questions (100 multi-hop / 100 single-hop)", "",
         "- Benchmark grounded in the curated Jeju gazetteer (69 real places, real",
         "  coordinates); every answer computed by haversine, so gold is auditable.",
         "- Deliberately NOT grounded in the ontology (59.9% of its type-checkable",
         "  relations are implausible) nor in the LLM-generated corpus — either choice",
         "  would hand one arm the answer by construction.",
         "- Multi-hop items chain two gazetteer facts and never name the bridge entity.",
         "- Same five metrics, same adapters, same greedy decoding as exp024.", ""]

    agg = {}
    for subset, pred in subsets.items():
        L += [f"## {subset}", "",
              "| Model | Condition | " + " | ".join(C.METRIC_LABELS[k] for k in C.METRICS) + " | n |",
              "|" + "---|" * (len(C.METRICS) + 3)]
        for model in models:
            for arm in arms:
                rows = [r for rid, r in cells.get((model, arm), {}).items() if pred(rid)]
                if not rows:
                    continue
                d = {k: _mean(rows, k) for k in C.METRICS}
                agg[f"{subset}__{model}__{arm}"] = dict(d, n=len(rows))
                L.append(f"| {C.MODEL_BY_KEY[model]['name']} | {ARM_LABEL[arm]} | " +
                         " | ".join(_fmt(d[k]) for k in C.METRICS) + f" | {len(rows)} |")
        # macro across models
        L.append("")
        L.append("| **Macro** | Condition | " + " | ".join(C.METRIC_LABELS[k] for k in C.METRICS) + " | |")
        L.append("|" + "---|" * (len(C.METRICS) + 3))
        for arm in arms:
            vals = {k: [agg[f"{subset}__{m}__{arm}"][k] for m in models
                        if f"{subset}__{m}__{arm}" in agg
                        and isinstance(agg[f"{subset}__{m}__{arm}"][k], (int, float))]
                    for k in C.METRICS}
            row = {k: round(statistics.mean(v), 4) if v else None for k, v in vals.items()}
            agg[f"{subset}__MACRO__{arm}"] = row
            L.append(f"| | {ARM_LABEL[arm]} | " + " | ".join(_fmt(row[k]) for k in C.METRICS) + " | |")
        L.append("")

    # ── ontology - rag, paired, per subset ────────────────────────────────
    L += ["## Ontology − RAG (paired per item)", "",
          "| Subset | Model | Arm | Metric | RAG | Ontology | Δ | p (t-test) | Onto win-rate |",
          "|---|---|---|---|---|---|---|---|---|"]
    tests = {}
    for subset, pred in subsets.items():
        for model in models:
            if (model, "rag") not in cells:
                continue
            for oarm in [x for x in ("onto", "onto_c", "onto_v3", "onto_v4", "onto_v5")
                         if (model, x) in cells]:
                r, o = cells[(model, "rag")], cells[(model, oarm)]
                ids = [i for i in r if i in o and pred(i)]
                for k in C.METRICS:
                    pairs = [(r[i][k], o[i][k]) for i in ids
                             if isinstance(r[i].get(k), (int, float))
                             and isinstance(o[i].get(k), (int, float))]
                    if len(pairs) < 2:
                        continue
                    xa = [x for x, _ in pairs]
                    xb = [y for _, y in pairs]
                    t = paired_ttest(xa, xb, "rag", oarm, metric=k)
                    w = wilcoxon_signed_rank(xa, xb, "rag", oarm, metric=k)
                    wr = sum(1 for x, y in pairs if y > x) / len(pairs)
                    tests[f"{subset}__{model}__{oarm}__{k}"] = {
                        "mean_rag": round(t.mean_a, 4), "mean_onto": round(t.mean_b, 4),
                        "delta": round(t.delta, 4), "p_ttest": t.p_value,
                        "p_wilcoxon": w.p_value, "significant": bool(t.significant),
                        "onto_win_rate": round(wr, 4), "n": len(pairs)}
                    if subset != "all":
                        L.append(f"| {subset} | {C.MODEL_BY_KEY[model]['name']} | {ARM_LABEL[oarm]} | "
                                 f"{C.METRIC_LABELS[k]} | {t.mean_a:.4f} | {t.mean_b:.4f} | "
                                 f"{t.delta:+.4f}{' *' if t.significant else ''} | "
                                 f"{t.p_value:.2e} | {wr:.3f} |")
    L += ["", "`*` = significant at α=0.05 (paired t-test).", ""]

    (OUT / "bench_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "bench_results.json").write_text(json.dumps(
        {"aggregates": agg, "paired_tests": tests}, indent=2, default=str), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {OUT}/bench_results.{{md,json}}")


if __name__ == "__main__":
    main()
