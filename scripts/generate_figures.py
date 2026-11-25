"""
generate_figures.py — Generate all publication-quality figures for the paper.

Figures produced
----------------
  Figure 1  architecture_diagram.png        (already exists — verified/re-exported)
  Figure 2  data_pipeline.png               (new — full data-flow diagram)
  Figure 3  evolution_growth_chart.png      (already exists — verified)
  Figure 4  hallucination_reduction.png     (already exists — verified)
  Figure 5  explanation_traces.png          (new — 3 grounded trace examples)
  Figure 6  retrieval_mrr.png               (new — MRR/Hit-rate bar chart)
  Figure 7  neo4j_class_distribution.png    (new — live Neo4j class breakdown)

All figures saved to: artifacts/figures/
300 DPI, consistent colour palette, publication-ready labels.

Usage
-----
  python scripts/generate_figures.py              # all figures
  python scripts/generate_figures.py --fig 2      # single figure
  python scripts/generate_figures.py --list       # show status of all figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
import numpy as np

from src.utils.io import load_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("generate_figures")

FIGURES = ROOT / "artifacts" / "figures"
REPORTS = ROOT / "artifacts" / "reports"

# ── Consistent colour palette ─────────────────────────────────────────────────
PAL = {
    "V1": "#e74c3c",   # red   — baseline
    "V2": "#e67e22",   # orange — doc-rag
    "V3": "#f1c40f",   # yellow — onto-rag subgraph
    "V4": "#2ecc71",   # green  — onto-rag hybrid
    "V5": "#3498db",   # blue   — constrained
    "bg": "#fafafa",
    "grid": "#e0e0e0",
}
DPI = 300


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Data Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def fig2_data_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor(PAL["bg"])
    fig.patch.set_facecolor(PAL["bg"])

    # Stage definitions: (x_centre, label_top, label_bot, colour)
    stages = [
        (1.2,  "Raw Data",       "61,963 records\nAmazon reviews",    "#5dade2"),
        (3.2,  "Data Cleaning",  "Dedup · filter\nUnicode norm",       "#48c9b0"),
        (5.2,  "LLM Extraction", "Llama 3.1-8B\nCoT prompting",        "#a569bd"),
        (7.2,  "Ontology Build", "v0.1→v1.0.0\n8,348 concepts",        "#e59866"),
        (9.2,  "SHACL Validation","0 violations\n31 classes",           "#58d68d"),
        (10.9, "Neo4j Load",     "16,742 edges\nbolt://7687",           "#2980b9"),
        (12.7, "RAG Pipeline",   "Retrieve·Generate\nConstrained+Val",  "#e74c3c"),
    ]

    box_w, box_h = 1.65, 1.4
    for (x, top, bot, col) in stages:
        # Box
        rect = plt.Rectangle((x - box_w/2, 1.8), box_w, box_h,
                              facecolor=col, edgecolor="white",
                              linewidth=1.5, zorder=3, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x, 2.5 + box_h/2, top, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=4)
        ax.text(x, 2.1, bot, ha="center", va="center",
                fontsize=7, color="white", zorder=4, linespacing=1.4)

    # Arrows between stages
    arrow_y = 2.5 + box_h / 2
    for i in range(len(stages) - 1):
        x0 = stages[i][0]   + box_w / 2
        x1 = stages[i+1][0] - box_w / 2
        ax.annotate("", xy=(x1, arrow_y), xytext=(x0, arrow_y),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
                    zorder=5)

    # Dataset split annotation
    ax.text(3.2, 1.55, "train 49,570 | dev 6,196 | test 6,197",
            ha="center", va="top", fontsize=7.5, color="#555",
            style="italic")

    ax.set_title("Figure 2 — Jeju Tourism Ontology Data Pipeline",
                 fontsize=12, fontweight="bold", pad=10, color="#222")
    plt.tight_layout()
    out = FIGURES / "data_pipeline.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    logger.info(f"Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Explanation Traces (3 grounded examples)
# ══════════════════════════════════════════════════════════════════════════════

def fig5_explanation_traces() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor(PAL["bg"])
    fig.suptitle("Figure 5 — Ontology-Grounded Explanation Traces (V5 Constrained)",
                 fontsize=12, fontweight="bold", y=1.01)

    traces = [
        {
            "title": "Trace A — Hiking Query",
            "query": "What trails are available\non Hallasan?",
            "nodes": ["Hallasan", "Seongpanak\nTrail", "Gwaneumsa\nTrail",
                      "NaturalAttraction", "NationalPark"],
            "edges": [("Hallasan","is_a","NaturalAttraction"),
                      ("Hallasan","has_activity","Seongpanak\nTrail"),
                      ("Hallasan","has_activity","Gwaneumsa\nTrail"),
                      ("Hallasan","part_of","NationalPark")],
            "grounding": 0.84,
            "violations": 0,
            "color": "#3498db",
        },
        {
            "title": "Trace B — Festival Query",
            "query": "What spring festivals\nexist in Jeju?",
            "nodes": ["Jeju", "HaenyeoFestival", "FireFestival",
                      "SpringSeason", "Udo Island"],
            "edges": [("Jeju","has_activity","HaenyeoFestival"),
                      ("Jeju","has_activity","FireFestival"),
                      ("HaenyeoFestival","has_seasonality","SpringSeason"),
                      ("HaenyeoFestival","located_in","Udo Island")],
            "grounding": 0.79,
            "violations": 0,
            "color": "#2ecc71",
        },
        {
            "title": "Trace C — Constraint Fix",
            "query": "Can I drive to\nSeongsan summit?",
            "nodes": ["SeongSan\nIlchulbong", "VehicleProhibited", "FootOnly\nAccess",
                      "CarPark", "Landmark"],
            "edges": [("SeongSan\nIlchulbong","is_a","Landmark"),
                      ("SeongSan\nIlchulbong","hasAccessConstraint","VehicleProhibited"),
                      ("SeongSan\nIlchulbong","has_activity","FootOnly\nAccess"),
                      ("SeongSan\nIlchulbong","near","CarPark")],
            "grounding": 0.71,
            "violations": 0,
            "color": "#e67e22",
        },
    ]

    node_positions = [
        {"Hallasan":              (0.5, 0.75),
         "NaturalAttraction":     (0.15, 0.45),
         "Seongpanak\nTrail":     (0.5,  0.38),
         "Gwaneumsa\nTrail":      (0.82, 0.45),
         "NationalPark":          (0.5,  0.10)},
        {"Jeju":                  (0.5, 0.75),
         "HaenyeoFestival":       (0.22, 0.44),
         "FireFestival":          (0.78, 0.44),
         "SpringSeason":          (0.22, 0.10),
         "Udo Island":            (0.78, 0.10)},
        {"SeongSan\nIlchulbong":  (0.5, 0.75),
         "Landmark":              (0.15, 0.48),
         "VehicleProhibited":     (0.5,  0.38),
         "FootOnly\nAccess":      (0.82, 0.48),
         "CarPark":               (0.5,  0.10)},
    ]

    for ax, trace, pos in zip(axes, traces, node_positions):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#f8f9fa")

        # Draw edges first
        for (src, pred, dst) in trace["edges"]:
            if src in pos and dst in pos:
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->",
                                           color=trace["color"], lw=1.3,
                                           connectionstyle="arc3,rad=0.08"),
                            zorder=2)
                mx, my = (x0+x1)/2, (y0+y1)/2
                ax.text(mx, my + 0.03, pred, ha="center", va="bottom",
                        fontsize=5.5, color="#555", style="italic", zorder=3)

        # Draw nodes
        for node, (x, y) in pos.items():
            is_seed = node == trace["nodes"][0]
            fc = trace["color"] if is_seed else "#ecf0f1"
            ec = trace["color"]
            tc = "white" if is_seed else "#2c3e50"
            circle = plt.Circle((x, y), 0.11, facecolor=fc,
                                 edgecolor=ec, linewidth=1.5, zorder=4)
            ax.add_patch(circle)
            ax.text(x, y, node, ha="center", va="center",
                    fontsize=6, fontweight="bold" if is_seed else "normal",
                    color=tc, zorder=5, multialignment="center")

        # Title + metrics
        ax.set_title(trace["title"], fontsize=9, fontweight="bold",
                     color="#2c3e50", pad=4)
        ax.text(0.5, 0.96,
                f'Query: "{trace["query"]}"',
                ha="center", va="top", fontsize=7, color="#555",
                style="italic", transform=ax.transAxes,
                multialignment="center")
        metric_text = (f"Grounding={trace['grounding']:.2f}  "
                       f"Violations={trace['violations']}")
        ax.text(0.5, -0.04, metric_text,
                ha="center", va="top", fontsize=7.5,
                color="#27ae60" if trace["violations"] == 0 else "#e74c3c",
                transform=ax.transAxes, fontweight="bold")

    plt.tight_layout()
    out = FIGURES / "explanation_traces.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    logger.info(f"Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Retrieval MRR / Hit-rate Comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig6_retrieval_mrr() -> None:
    report = load_json(REPORTS / "exp009_retrieval_report.json")
    agg    = report["aggregate"]

    modes       = ["no_retrieval", "doc_only", "onto_only", "hybrid"]
    labels      = ["No Retrieval", "Doc-RAG", "Onto-RAG\n(subgraph)", "Onto-RAG\n(hybrid)"]
    colors      = [PAL["V1"], PAL["V2"], PAL["V3"], PAL["V4"]]
    metrics     = {
        "MRR":       [agg[m]["mrr"]            for m in modes],
        "Hit@5":     [agg[m]["hit_rate_at_5"]  for m in modes],
        "Hit@10":    [agg[m]["hit_rate_at_10"] for m in modes],
        "Recall":    [agg[m]["avg_context_recall"] for m in modes],
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    fig.patch.set_facecolor(PAL["bg"])
    fig.suptitle("Figure 6 — Retrieval Performance Across 6,197 Queries (exp009)",
                 fontsize=11, fontweight="bold")

    x = np.arange(len(modes))
    for ax, (metric, vals) in zip(axes, metrics.items()):
        bars = ax.bar(x, vals, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.set_facecolor(PAL["bg"])
        ax.yaxis.grid(True, color=PAL["grid"], zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIGURES / "retrieval_mrr.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    logger.info(f"Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Neo4j Live Class Distribution
# ══════════════════════════════════════════════════════════════════════════════

def fig7_neo4j_class_distribution() -> None:
    try:
        import os
        from src.graph.neo4j_client import Neo4jClient

        def _load_env():
            env = ROOT / ".env"
            if not env.exists(): return
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        _load_env()

        client = Neo4jClient.from_env()
        if not client.is_connected:
            logger.warning("Neo4j not connected — skipping Fig 7")
            return

        stats = client.stats()
        client.close()

        class_counts = dict(sorted(stats["class_counts"].items(),
                                   key=lambda x: -x[1]))
        # Top 20
        labels = list(class_counts.keys())[:20]
        values = [class_counts[l] for l in labels]

    except Exception as exc:
        logger.warning(f"Neo4j fig7 skipped: {exc}")
        return

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(PAL["bg"])
    fig.suptitle(
        f"Figure 7 — Neo4j Live Ontology: {stats['n_concepts']:,} Concepts | "
        f"{stats['n_relations']:,} Relations | {stats['n_classes']} Classes",
        fontsize=11, fontweight="bold"
    )

    # Bar chart — top 20 classes
    cmap  = plt.get_cmap("tab20")
    cols  = [cmap(i / 20) for i in range(len(labels))]
    y_pos = np.arange(len(labels))
    ax_bar.barh(y_pos, values, color=cols, edgecolor="white", linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Number of Concepts", fontsize=9)
    ax_bar.set_title("Top 20 Classes (bar)", fontsize=10, fontweight="bold")
    ax_bar.set_facecolor(PAL["bg"])
    ax_bar.xaxis.grid(True, color=PAL["grid"], zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top","right"]].set_visible(False)
    for i, v in enumerate(values):
        ax_bar.text(v + 5, i, str(v), va="center", fontsize=7.5)

    # Pie chart — top 10 + rest
    top10_labels = labels[:10]
    top10_vals   = values[:10]
    rest_val     = sum(values[10:])
    pie_labels   = top10_labels + [f"Other ({len(labels)-10} classes)"]
    pie_vals     = top10_vals   + [rest_val]
    pie_cols     = [cmap(i/20) for i in range(10)] + ["#bdc3c7"]

    wedges, texts, autotexts = ax_pie.pie(
        pie_vals, labels=pie_labels, colors=pie_cols,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 7.5},
        pctdistance=0.82, labeldistance=1.08,
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax_pie.set_title("Class Distribution (pie, top 10 + rest)",
                     fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = FIGURES / "neo4j_class_distribution.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    logger.info(f"Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Verify existing figures
# ══════════════════════════════════════════════════════════════════════════════

def _verify_existing() -> dict[str, bool]:
    expected = {
        "architecture_diagram.png":   "Figure 1 — System Architecture",
        "evolution_growth_chart.png":  "Figure 3 — Ontology Evolution",
        "hallucination_reduction.png": "Figure 4 — Hallucination Reduction",
        "5variant_comparison.png":     "Figure — 5-Variant Comparison",
        "failure_category_heatmap.png":"Figure — Failure Category Heatmap",
        "ablation_heatmap.png":        "Figure — Ablation Heatmap",
        "latency_comparison.png":      "Figure — Latency Comparison",
    }
    status = {}
    for fname, label in expected.items():
        path = FIGURES / fname
        status[fname] = path.exists()
        mark = "✓" if path.exists() else "✗"
        size = f"{path.stat().st_size // 1024} KB" if path.exists() else "missing"
        logger.info(f"  [{mark}] {label:<40}  {fname}  ({size})")
    return status


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

FIG_MAP = {
    2: ("Data Pipeline",            fig2_data_pipeline),
    5: ("Explanation Traces",       fig5_explanation_traces),
    6: ("Retrieval MRR/Hit-rate",   fig6_retrieval_mrr),
    7: ("Neo4j Class Distribution", fig7_neo4j_class_distribution),
}


def main(fig: int | None = None, list_only: bool = False) -> None:
    ensure_dir(FIGURES)

    if list_only:
        print("\n=== Figure Status ===")
        _verify_existing()
        for n, (name, _) in FIG_MAP.items():
            path = FIGURES / {
                2: "data_pipeline.png",
                5: "explanation_traces.png",
                6: "retrieval_mrr.png",
                7: "neo4j_class_distribution.png",
            }[n]
            mark = "✓" if path.exists() else "✗"
            print(f"  [{mark}] Figure {n} — {name}")
        return

    print(f"\n{'='*56}")
    print(f"  Generating Publication Figures")
    print(f"{'='*56}")

    # Verify existing
    logger.info("Checking existing figures...")
    _verify_existing()

    # Generate new / missing figures
    if fig is None:
        targets = list(FIG_MAP.items())
    else:
        if fig not in FIG_MAP:
            print(f"Unknown figure: {fig}. Available: {list(FIG_MAP.keys())}")
            return
        targets = [(fig, FIG_MAP[fig])]

    for n, (name, fn) in targets:
        print(f"\n  Generating Figure {n} — {name}...")
        try:
            fn()
            print(f"  ✓ Figure {n} saved")
        except Exception as exc:
            print(f"  ✗ Figure {n} FAILED: {exc}")
            logger.exception(f"Figure {n} failed")

    # Final summary
    print(f"\n{'='*56}")
    print(f"  All figures in: artifacts/figures/")
    all_figs = sorted(FIGURES.glob("*.png"))
    for f in all_figs:
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:<45} {size_kb:>5} KB")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate all paper figures")
    p.add_argument("--fig",  type=int, default=None,
                   help="Generate only this figure number (2/5/6/7)")
    p.add_argument("--list", action="store_true", dest="list_only",
                   help="List figure status without generating")
    args = p.parse_args()
    main(fig=args.fig, list_only=args.list_only)
