"""
Aggregate the per-item scores into the comparison tables.

Emits, under outputs/reports/:
  results.json          every aggregate + paired test, machine readable
  results.md            main table + per-model deltas + significance
  results.tex           LaTeX booktabs version of the main table

Paired tests are item-level (same 4,607 test questions in both arms), using the
project's existing paired_ttest / wilcoxon_signed_rank helpers.

Usage:  python -m experiments.exp024_rag_vs_ontology.report
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from . import config as C

sys.path.insert(0, str(C.ROOT))
from src.evaluation.stats_tests import paired_ttest, wilcoxon_signed_rank
from src.utils.proctitle import set_proctitle

ARM_LABEL = {"rag": "Simple RAG", "onto": "Ontology v1",
             "onto_c": "Ontology (cleaned)", "onto_v2": "Ontology v2 (dense triples)",
             "onto_v3": "Ontology v3 (entity cards)", "onto_v4": "Ontology v4 (graph traversal)",
             "onto_v5": "Ontology v5 (traversal + routing)",
             "hybrid": "Onto-RAG hybrid", "hybrid10": "Onto-RAG hybrid (10% facts)",
             "hybrid15": "Onto-RAG hybrid (15% facts)", "hybrid20": "Onto-RAG hybrid (20% facts)",
             "none": "Fine-tuned, no context", "base": "Vanilla LLM (no FT)"}


def _load(split: str) -> dict[tuple[str, str], dict[int, dict]]:
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    for p in sorted(C.SCORE_DIR.glob(f"{split}__*.jsonl")):
        model, arm = p.stem.split("__")[1], p.stem.split("__")[2]
        cells[(model, arm)] = {json.loads(l)["record_id"]: json.loads(l)
                               for l in open(p, encoding="utf-8")}
    return cells


def _mean(rows: list[dict], k: str) -> float | None:
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return round(statistics.mean(v), 4) if v else None


def _composite(agg: dict) -> float | None:
    num = wsum = 0.0
    for k, w in C.COMPOSITE_WEIGHTS.items():
        if isinstance(agg.get(k), (int, float)):
            num += w * agg[k]; wsum += w
    return round(num / wsum, 4) if wsum else None


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def _paired_metrics(r: dict, o: dict, onto_arm: str) -> dict:
    """Item-level paired comparison of one ontology arm against the RAG arm."""
    ids = [i for i in r if i in o]
    per_metric = {}
    for k in C.METRICS:
        pairs = [(r[i][k], o[i][k]) for i in ids
                 if isinstance(r[i].get(k), (int, float)) and isinstance(o[i].get(k), (int, float))]
        if len(pairs) < 2:
            continue
        xa = [x for x, _ in pairs]
        xb = [y for _, y in pairs]
        t = paired_ttest(xa, xb, "rag", onto_arm, metric=k)
        w = wilcoxon_signed_rank(xa, xb, "rag", onto_arm, metric=k)
        per_metric[k] = {
            "n_paired": len(xa), "mean_rag": round(t.mean_a, 4), "mean_onto": round(t.mean_b, 4),
            "delta": round(t.delta, 4),
            "delta_pct": round(100 * t.delta / t.mean_a, 2) if t.mean_a else None,
            "t": round(t.statistic, 3), "p_ttest": t.p_value, "p_wilcoxon": w.p_value,
            "significant": bool(t.significant),
            "onto_win_rate": round(sum(1 for x, y in pairs if y > x) / len(pairs), 4),
        }
    return per_metric


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    cells = _load(a.split)
    if not cells:
        raise SystemExit("no score files found — run score.py first")

    models = [m["key"] for m in C.MODELS if any(k[0] == m["key"] for k in cells)]
    arms = [x for x in C.ALL_ARMS if any(k[1] == x for k in cells)]

    # ── aggregates ────────────────────────────────────────────────────────
    agg: dict[str, dict] = {}
    for (model, arm), rows in cells.items():
        vals = list(rows.values())
        d = {k: _mean(vals, k) for k in C.METRICS}
        d["composite"] = _composite(d)
        d["n"] = len(vals)
        agg[f"{model}__{arm}"] = d

    # ── paired tests: onto vs rag, per model per metric ───────────────────
    tests: dict[str, dict] = {}
    for model in models:
        if (model, "rag") not in cells:
            continue
        for onto_arm in [x for x in ("onto", "onto_c", "onto_v2", "onto_v3", "onto_v4",
                                     "onto_v5", "hybrid") if (model, x) in cells]:
            pm = _paired_metrics(cells[(model, "rag")], cells[(model, onto_arm)], onto_arm)
            if pm:
                tests[f"{model}__{onto_arm}"] = pm

    # ── markdown ──────────────────────────────────────────────────────────
    L = [f"# exp024 — Simple RAG vs Ontology grounding ({a.split} split)", "",
         f"- Test items per cell: **{agg[list(agg)[0]]['n']:,}**",
         f"- Gold reference: the dataset's own `Output` field (identical for both arms)",
         f"- Ontology: v{C.ONTO_VERSION} · RAG: {C.EMBED_MODEL}, top-{C.RAG_TOP_K} over the train split",
         f"- Adaptation: LoRA r={C.LORA['r']}, alpha={C.LORA['lora_alpha']}, "
         f"{C.TRAIN['epochs']} epoch on 36,847 train examples — ONE adapter per model, "
         f"tuned on a 50/50 mix of RAG and ontology contexts, serving both arms",
         f"- Decoding: greedy, max_new_tokens={C.MAX_NEW_TOKENS}", "",
         "## Main table", ""]
    hdr = "| Model | Condition | " + " | ".join(C.METRIC_LABELS[k] for k in C.METRICS) + " | Composite |"
    L += [hdr, "|" + "---|" * (len(C.METRICS) + 3)]
    for model in models:
        name = C.MODEL_BY_KEY[model]["name"]
        for arm in arms:
            d = agg.get(f"{model}__{arm}")
            if not d:
                continue
            label = ARM_LABEL[arm]
            if arm == "onto":
                label = f"**{label}**"
            L.append(f"| {name} | {label} | " +
                     " | ".join(_fmt(d[k]) for k in C.METRICS) + f" | {_fmt(d['composite'])} |")

    L += ["", "## Ontology − RAG (paired, per item)", "",
          "| Model | Metric | RAG | Ontology | Δ | Δ% | p (t-test) | p (Wilcoxon) | Ontology win-rate |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, per_metric in tests.items():
        model, onto_arm = key.split("__")
        for k, v in per_metric.items():
            star = " *" if v["significant"] else ""
            L.append(f"| {C.MODEL_BY_KEY[model]['name']} ({ARM_LABEL[onto_arm]}) | {C.METRIC_LABELS[k]} | "
                     f"{v['mean_rag']:.4f} | {v['mean_onto']:.4f} | {v['delta']:+.4f}{star} | "
                     f"{v['delta_pct']:+.2f}% | {v['p_ttest']:.2e} | {v['p_wilcoxon']:.2e} | "
                     f"{v['onto_win_rate']:.3f} |")
    L += ["", "`*` = significant at α=0.05 (paired t-test).", ""]

    # macro-average across models
    L += ["## Macro-average across models", "",
          "| Condition | " + " | ".join(C.METRIC_LABELS[k] for k in C.METRICS) + " | Composite |",
          "|" + "---|" * (len(C.METRICS) + 2)]
    macro = {}
    for arm in arms:
        vals = {k: [agg[f"{m}__{arm}"][k] for m in models
                    if f"{m}__{arm}" in agg and isinstance(agg[f"{m}__{arm}"][k], (int, float))]
                for k in C.METRICS}
        row = {k: round(statistics.mean(v), 4) if v else None for k, v in vals.items()}
        row["composite"] = _composite(row)
        macro[arm] = row
        L.append(f"| {ARM_LABEL[arm]} | " + " | ".join(_fmt(row[k]) for k in C.METRICS) +
                 f" | {_fmt(row['composite'])} |")

    (C.REPORT_DIR / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── LaTeX ─────────────────────────────────────────────────────────────
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Simple RAG vs.\ ontology grounding on the full Jeju test split "
         rf"({agg[list(agg)[0]]['n']:,} items), LoRA fine-tuned per arm.}}",
         r"\label{tab:rag_vs_onto}",
         r"\begin{tabular}{ll" + "r" * (len(C.METRICS) + 1) + "}", r"\toprule",
         "Model & Condition & " + " & ".join(C.METRIC_LABELS[k] for k in C.METRICS) +
         r" & Composite \\", r"\midrule"]
    for model in models:
        for arm in arms:
            d = agg.get(f"{model}__{arm}")
            if not d:
                continue
            cellv = " & ".join(_fmt(d[k]) for k in C.METRICS)
            T.append(f"{C.MODEL_BY_KEY[model]['name']} & {ARM_LABEL[arm]} & {cellv} & "
                     f"{_fmt(d['composite'])} " + r"\\")
        T.append(r"\midrule")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (C.REPORT_DIR / "results.tex").write_text("\n".join(T) + "\n", encoding="utf-8")

    (C.REPORT_DIR / "results.json").write_text(json.dumps(
        {"split": a.split, "aggregates": agg, "macro": macro, "paired_tests": tests,
         "config": {"ontology_version": C.ONTO_VERSION, "embed_model": C.EMBED_MODEL,
                    "rag_top_k": C.RAG_TOP_K, "lora": C.LORA, "train": C.TRAIN,
                    "max_new_tokens": C.MAX_NEW_TOKENS,
                    "composite_weights": C.COMPOSITE_WEIGHTS}},
        indent=2, default=str), encoding="utf-8")

    print("\n".join(L))
    print(f"\nwrote {C.REPORT_DIR}/results.{{md,tex,json}}")


if __name__ == "__main__":
    main()
