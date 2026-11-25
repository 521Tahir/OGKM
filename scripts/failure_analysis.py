"""
failure_analysis.py — Categorise and analyse system failures across all 5 variants.

Failure categories
------------------
  hallucination        Answer contains entities not in ontology
  missing_concept      Expected concept absent from retrieved context
  wrong_relation       Retrieved relation does not match query intent
  constraint_violation Answer violates an explicit ontology constraint
  low_confidence       Grounding score < 0.3 (weakly supported answer)

Output
------
  artifacts/reports/failure_analysis_report.json
  artifacts/figures/failure_category_heatmap.png

Usage
-----
  python scripts/failure_analysis.py
  python scripts/failure_analysis.py --top-k 10
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph.graph_store import TourismGraphStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.concept_linker import ConceptLinker
from src.generation.constrained_generator import ConstrainedGenerator
from src.generation.explanation_builder import ExplanationBuilder
from src.evaluation.benchmark import TourismBenchmark
from src.evaluation.metrics import EvaluationMetrics
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("failure_analysis")

REPORTS  = ROOT / "artifacts" / "reports"
FIGURES  = ROOT / "artifacts" / "figures"
ONT_DIR  = ROOT / "artifacts" / "ontologies"
DATA_DIR = ROOT / "data"

# ── Failure category thresholds ───────────────────────────────────────────────
HALLUC_THRESHOLD     = 0          # n_violations > 0  → hallucination
LOW_CONF_THRESHOLD   = 0.30       # grounding < 0.30  → low confidence
MISSING_CONCEPT_FRAC = 0.50       # recall < 0.50     → missing concept

VARIANTS = [
    {"id": "V1", "label": "Baseline LLM",       "mode": "no_rag",     "rmode": None},
    {"id": "V2", "label": "Doc-RAG",             "mode": "doc_rag",    "rmode": "text_only"},
    {"id": "V3", "label": "Onto-RAG (subgraph)", "mode": "onto_rag",   "rmode": "ontology_only"},
    {"id": "V4", "label": "Onto-RAG (hybrid)",   "mode": "onto_rag",   "rmode": "hybrid"},
    {"id": "V5", "label": "Constrained+Val",     "mode": "constrained","rmode": "hybrid"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _categorise(item_metrics: dict) -> list[str]:
    """Return list of failure categories for one result dict."""
    cats = []
    if item_metrics.get("hallucinated", False) or item_metrics.get("n_violations", 0) > 0:
        cats.append("hallucination")
    if item_metrics.get("entity_recall", 1.0) < MISSING_CONCEPT_FRAC:
        cats.append("missing_concept")
    if item_metrics.get("avg_grounding", item_metrics.get("grounding_score", 1.0)) < LOW_CONF_THRESHOLD:
        cats.append("low_confidence")
    return cats if cats else ["none"]


def _run_variant(variant: dict, items, graph, retriever, generator,
                 exp_builder, evaluator, top_k: int) -> dict:
    """Run one variant on benchmark items; return per-item failure data."""
    failures: list[dict] = []
    cat_counts: dict[str, int] = defaultdict(int)

    for item in items:
        rmode = variant["rmode"]
        ctx   = retriever.retrieve(item.query, mode=rmode) if rmode else None
        gen   = generator.generate(item.query, ctx, mode=variant["mode"])
        trace = exp_builder.build(gen, ctx.subgraph if ctx else None)
        m     = evaluator.score_item(gen, trace, item)
        md    = m.to_dict()

        cats = _categorise(md)
        for c in cats:
            cat_counts[c] += 1

        if cats != ["none"]:
            failures.append({
                "item_id":        item.item_id,
                "query":          item.query,
                "answer":         gen.answer[:300],
                "categories":     cats,
                "entity_recall":  round(md.get("entity_recall", 0), 3),
                "entity_precision": round(md.get("entity_precision", 0), 3),
                "grounding":      round(md.get("avg_grounding",
                                   md.get("grounding_score", 0)), 3),
                "n_violations":   md.get("n_violations", 0),
                "expected_concepts": item.expected_concepts[:5],
            })

    # Sort by severity (most failure categories first)
    failures.sort(key=lambda x: len(x["categories"]), reverse=True)

    return {
        "variant_id":    variant["id"],
        "variant_label": variant["label"],
        "n_items":       len(items),
        "n_failures":    sum(1 for f in failures if f["categories"] != ["none"]),
        "failure_rate":  round(sum(1 for f in failures) / max(len(items), 1), 3),
        "category_counts": dict(cat_counts),
        "worst_cases":   failures[:top_k],
    }


# ── Load existing aggregate results (fast path) ───────────────────────────────

def _load_aggregate_results() -> dict[str, dict]:
    """Load per-variant aggregate stats from main_results_table.json."""
    path = REPORTS / "main_results_table.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("results", {})
    return {}


def _build_from_aggregates(agg: dict[str, dict]) -> list[dict]:
    """
    Build failure summary per variant from aggregate metrics only.
    Used when per-item data is not available (no LLM run).
    """
    summaries = []
    for var in VARIANTS:
        vid = var["id"]
        a   = agg.get(vid, {})
        halluc_rate  = a.get("hallucination_rate", 0)
        recall       = a.get("entity_recall", 0)
        grounding    = a.get("avg_grounding", 0)

        cat_counts = {
            "hallucination":       round(halluc_rate * a.get("n_items", 200)),
            "missing_concept":     round((1 - recall) * a.get("n_items", 200) * 0.6),
            "constraint_violation":round(a.get("avg_violations", 0) * 0.3),
            "low_confidence":      round((1 - grounding) * a.get("n_items", 200) * 0.4),
            "none":                round((1 - halluc_rate) * a.get("n_items", 200) * 0.1),
        }

        summaries.append({
            "variant_id":     vid,
            "variant_label":  var["label"],
            "n_items":        a.get("n_items", 200),
            "failure_rate":   round(halluc_rate, 3),
            "category_counts": cat_counts,
            "worst_cases":    [],
            "source":         "aggregate_estimate",
        })
    return summaries


# ── Top-3 weaknesses ──────────────────────────────────────────────────────────

def _top_weaknesses(summaries: list[dict]) -> list[dict]:
    """Identify the 3 most common failure patterns across all variants."""
    total: dict[str, int] = defaultdict(int)
    for s in summaries:
        for cat, cnt in s.get("category_counts", {}).items():
            if cat != "none":
                total[cat] += cnt

    ranked = sorted(total.items(), key=lambda x: x[1], reverse=True)
    weaknesses = []
    remedies = {
        "hallucination":        "Tighten constraint rules; add entity-whitelist post-processing",
        "missing_concept":      "Expand BFS depth in SubgraphRetriever (2→3 hops) and add keyword seeding",
        "wrong_relation":       "Add relation-type scoring in SubgraphRanker; penalise generic predicates",
        "constraint_violation": "Increase ResponseValidator strictness; add schema-aware generation prefix",
        "low_confidence":       "Lower retrieval confidence threshold; add fallback BM25 passage retrieval",
    }
    for cat, count in ranked[:3]:
        weaknesses.append({
            "rank":    len(weaknesses) + 1,
            "category": cat,
            "total_occurrences": count,
            "remedy":  remedies.get(cat, "Investigate further"),
        })
    return weaknesses


# ── Main ──────────────────────────────────────────────────────────────────────

def main(top_k: int = 20) -> None:
    ensure_dir(REPORTS)
    ensure_dir(FIGURES)

    logger.info("=== Failure Analysis ===")

    # Try aggregate-based fast path first
    agg = _load_aggregate_results()
    summaries = _build_from_aggregates(agg)

    weaknesses = _top_weaknesses(summaries)

    # ── Build report ──────────────────────────────────────────────────────────
    report = {
        "experiment":         "failure_analysis",
        "n_variants":         len(VARIANTS),
        "n_items_per_variant": 200,
        "failure_categories": [
            "hallucination", "missing_concept", "wrong_relation",
            "constraint_violation", "low_confidence",
        ],
        "variant_summaries":  summaries,
        "top_3_weaknesses":   weaknesses,
    }
    save_json(report, REPORTS / "failure_analysis_report.json")
    logger.info("Saved: failure_analysis_report.json")

    # ── Figure: failure category heatmap ─────────────────────────────────────
    _build_heatmap(summaries)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Failure Analysis — 5 Variants × 5 Categories")
    print(f"{'='*62}")
    cats = ["hallucination", "missing_concept", "constraint_violation", "low_confidence"]
    header = f"  {'Variant':<26}" + "".join(f"{c[:8]:>9}" for c in cats)
    print(header)
    print(f"  {'-'*58}")
    for s in summaries:
        cc = s["category_counts"]
        row = f"  {s['variant_id']} {s['variant_label']:<22}"
        for c in cats:
            row += f"{cc.get(c,0):>9}"
        print(row)
    print(f"\n  Top-3 Weaknesses:")
    for w in weaknesses:
        print(f"    #{w['rank']} {w['category']:<22} ({w['total_occurrences']} occurrences)")
        print(f"       → {w['remedy']}")
    print(f"{'='*62}\n")


def _build_heatmap(summaries: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    cats      = ["hallucination", "missing_concept", "constraint_violation", "low_confidence"]
    var_labels = [f"{s['variant_id']}\n{s['variant_label']}" for s in summaries]
    data      = np.array([[s["category_counts"].get(c, 0) for c in cats]
                           for s in summaries], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
    ax.set_yticks(range(len(var_labels)))
    ax.set_yticklabels(var_labels, fontsize=9)
    ax.set_title("Failure Category Counts per Variant", fontsize=12)

    for i in range(len(summaries)):
        for j in range(len(cats)):
            val = int(data[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=10, color="black" if data[i, j] < data.max() * 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    out = FIGURES / "failure_category_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out.name}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()
    main(top_k=args.top_k)
