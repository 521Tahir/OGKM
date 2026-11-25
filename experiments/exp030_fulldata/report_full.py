"""
Full-dataset comparison: Simple RAG vs every ontology arm, all models,
on the complete 4,607-item test split.

This is the headline table for the free-form generation task. It deliberately
renders whichever arms are actually scored rather than a fixed list, so it stays
correct as arms are added or regenerated.

  python -m experiments.exp030_fulldata.report_full
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from src.evaluation.stats_tests import paired_ttest, wilcoxon_signed_rank
from src.utils.proctitle import set_proctitle

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
S = C.SCORE_DIR
K = C.METRICS
MODELS = [("llama3.2-1b", "Llama-3.2-1B", 1.24), ("llama3.2-3b", "Llama-3.2-3B", 3.21),
          ("llama3.1-8b", "Llama-3.1-8B", 8.03), ("deepseek-r1-8b", "DeepSeek-R1-8B", 8.03),
          ("qwen3-8b", "Qwen3-8B", 8.19), ("mistral-7b", "Mistral-7B", 7.25)]
ARM_LABEL = {"rag": "Simple RAG", "onto": "Ontology v1", "onto_c": "Ontology (cleaned)",
             "onto_v3": "Ontology v3 (entity cards)", "onto_v4": "Ontology v4 (traversal)",
             "onto_v5": "Ontology v5 (traversal+routing)",
             "hybrid": "Onto-RAG hybrid (ours)", "base": "Vanilla (no fine-tune)"}
ARM_ORDER = ["rag", "hybrid", "onto", "onto_c", "onto_v3", "onto_v4", "onto_v5", "base"]


def load(split: str, m: str, a: str):
    p = S / f"{split}__{m}__{a}.jsonl"
    if not p.exists():
        return None
    return {json.loads(l)["record_id"]: json.loads(l) for l in open(p, encoding="utf-8")}


def mean(d, ids, k):
    v = [d[i][k] for i in ids if isinstance(d[i].get(k), (int, float))]
    return round(st.mean(v), 4) if v else None


def f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    cells = {}
    for mk, _, _ in MODELS:
        for arm in ARM_ORDER:
            d = load(a.split, mk, arm)
            if d:
                cells[(mk, arm)] = d
    if not cells:
        raise SystemExit("no scored cells")
    arms = [x for x in ARM_ORDER if any(k[1] == x for k in cells)]
    n_items = len(next(iter(cells.values())))

    L = [f"# Full-dataset results — Simple RAG vs Ontology (`{a.split}` split)", "",
         f"**{n_items:,} items per cell** · {len([m for m in MODELS if (m[0],'rag') in cells])} models "
         f"· {len(arms)} conditions", "",
         "Gold reference is the dataset's own `Output` field, held out from training and "
         "independent of both retrievers. Identical base weights, adapter, prompt template "
         "and greedy decoding in every arm — only the injected context differs.", "",
         "## Per-model results", ""]

    agg = {}
    for mk, mn, nb in MODELS:
        if (mk, "rag") not in cells:
            continue
        ids = list(cells[(mk, "rag")])
        L += [f"### {mn} ({nb}B)", "",
              "| Condition | " + " | ".join(C.METRIC_LABELS[k] for k in K) + " |",
              "|" + "---|" * (len(K) + 1)]
        for arm in arms:
            if (mk, arm) not in cells:
                continue
            v = [mean(cells[(mk, arm)], ids, k) for k in K]
            agg[f"{mk}__{arm}"] = v
            lab = ARM_LABEL[arm]
            if arm == "rag":
                lab = f"**{lab}**"
            L.append(f"| {lab} | " + " | ".join(f(x) for x in v) + " |")
        r = agg.get(f"{mk}__rag")
        for arm in arms:
            if arm in ("rag", "base") or f"{mk}__{arm}" not in agg:
                continue
            o = agg[f"{mk}__{arm}"]
            L.append(f"| *Δ {ARM_LABEL[arm]} − RAG* | " +
                     " | ".join(f"{y-x:+.4f}" if x is not None and y is not None else "n/a"
                                for x, y in zip(r, o)) + " |")
        L.append("")

    # macro
    L += ["## Macro-average across models", "",
          "| Condition | " + " | ".join(C.METRIC_LABELS[k] for k in K) + " | metrics beating RAG |",
          "|" + "---|" * (len(K) + 2)]
    macro = {}
    for arm in arms:
        cols = []
        for j, _ in enumerate(K):
            v = [agg[f"{m}__{arm}"][j] for m, _, _ in MODELS
                 if f"{m}__{arm}" in agg and agg[f"{m}__{arm}"][j] is not None]
            cols.append(round(st.mean(v), 4) if v else None)
        macro[arm] = cols
        won = ("—" if arm == "rag" else
               f"{sum(1 for x, y in zip(macro['rag'], cols) if x is not None and y is not None and y > x)}/{len(K)}")
        L.append(f"| {ARM_LABEL[arm]} | " + " | ".join(f(x) for x in cols) + f" | {won} |")
    L.append("")

    # significance vs RAG, per model, for each ontology arm
    L += ["## Paired significance vs Simple RAG", "",
          "| Model | Arm | Metric | RAG | Arm | Δ | p (t-test) | arm win-rate |",
          "|---|---|---|---|---|---|---|---|"]
    tests = {}
    for mk, mn, _ in MODELS:
        if (mk, "rag") not in cells:
            continue
        r = cells[(mk, "rag")]
        for arm in arms:
            if arm in ("rag", "base") or (mk, arm) not in cells:
                continue
            o = cells[(mk, arm)]
            ids = [i for i in r if i in o]
            for k in K:
                pairs = [(r[i][k], o[i][k]) for i in ids
                         if isinstance(r[i].get(k), (int, float)) and isinstance(o[i].get(k), (int, float))]
                if len(pairs) < 2:
                    continue
                xa = [x for x, _ in pairs]
                xb = [y for _, y in pairs]
                t = paired_ttest(xa, xb, "rag", arm, metric=k)
                w = wilcoxon_signed_rank(xa, xb, "rag", arm, metric=k)
                wr = sum(1 for x, y in pairs if y > x) / len(pairs)
                tests[f"{mk}__{arm}__{k}"] = {
                    "mean_rag": round(t.mean_a, 4), "mean_arm": round(t.mean_b, 4),
                    "delta": round(t.delta, 4), "p_ttest": t.p_value,
                    "p_wilcoxon": w.p_value, "significant": bool(t.significant),
                    "win_rate": round(wr, 4), "n": len(pairs)}
                L.append(f"| {mn} | {ARM_LABEL[arm]} | {C.METRIC_LABELS[k]} | {t.mean_a:.4f} | "
                         f"{t.mean_b:.4f} | {t.delta:+.4f}{' *' if t.significant else ''} | "
                         f"{t.p_value:.2e} | {wr:.3f} |")
    L += ["", "`*` = significant at α=0.05 (paired t-test over all items).", ""]

    (OUT / f"fulldata_{a.split}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / f"fulldata_{a.split}.json").write_text(
        json.dumps({"split": a.split, "n_items": n_items, "per_model": agg,
                    "macro": macro, "tests": tests}, indent=2), encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"... wrote {OUT}/fulldata_{a.split}.md  ({len(L)} lines)")


if __name__ == "__main__":
    main()
