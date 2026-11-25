"""Charts drawn for the deck and the video, in the deck's own palette.

The project's figure set answers research questions; a business audience is
being asked a different one, so these two charts are drawn from the same
evidence file the slides read.

    python -m wbs.task5_demo.make_deck_figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "wbs" / "deliverables" / "presentation"

NAVY, TEAL, AMBER, RED, GREY = "#0B1F3A", "#00A3A1", "#F2A900", "#C83A3A", "#9AA7B4"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#C9D3DF",
                     "axes.labelcolor": NAVY, "text.color": NAVY,
                     "xtick.color": "#5B6B7E", "ytick.color": "#5B6B7E",
                     "axes.spines.top": False, "axes.spines.right": False})


def bars(ax, labels, series, colors, title, fmt="{:.1f}", ylabel=""):
    n, w = len(series), 0.8 / len(series)
    for i, (name, vals) in enumerate(series):
        xs = [j - 0.4 + w * (i + 0.5) for j in range(len(labels))]
        ax.bar(xs, vals, w * 0.92, label=name, color=colors[i], zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8.5,
                    color=NAVY)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=12, color=NAVY, pad=10, weight="bold")
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(axis="y", color="#E7ECF2", zorder=0)
    ax.set_axisbelow(True)
    ax.margins(y=0.18)


def main() -> None:
    ev = json.loads((OUT / "demo_evidence.json").read_text(encoding="utf-8"))
    V, G, P, SZ = ev["variants"], ev["gated"], ev["pretty"], ev["sizes"]
    short = [P[m].replace("Llama-3.2-", "").replace("Llama-3.1-", "") for m in SZ]

    # ── what the graph buys, at each model size ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    series = [("model alone", "V1_llm"), ("+ retrieval", "V2_rag"),
              ("+ ontology (deployed)", None)]
    cols = [GREY, AMBER, TEAL]
    bars(axes[0], short,
         [(lab, [V[m][k]["ungrounded"] if k else G[m]["ungrounded"] for m in SZ])
          for lab, k in series], cols,
         "Unsupported claims  (lower is better)", "{:.2f}", "% of claims")
    bars(axes[1], short,
         [(lab, [V[m][k]["clean"] if k else G[m]["clean"] for m in SZ])
          for lab, k in series], cols,
         "Fully grounded answers  (higher is better)", "{:.0f}", "% of answers")
    bars(axes[2], short,
         [(lab, [V[m][k]["token_f1"] if k else G[m]["token_f1"] for m in SZ])
          for lab, k in series], cols,
         "Answer quality vs reference  (higher is better)", "{:.1f}", "token F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("What the ontology buys, at three model sizes  ·  347 held-out questions",
                 fontsize=14, color=NAVY, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p1 = OUT / "deck_fig_what_the_graph_buys.png"
    fig.savefig(p1, dpi=200, facecolor="white")
    plt.close(fig)

    # ── cost of changing what the system knows ───────────────────────────
    er = ev["edit_vs_retrain"]
    names = sorted(er["train_hours"], key=lambda k: er["train_hours"][k])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    xs = list(range(len(names) + 1))
    vals = [er["train_hours"][n] for n in names] + [er["edit_gpu_hours"]]
    labels = [n.split("__")[0].replace("llama3.2-", "").replace("llama3.1-", "").upper() +
              "\nfine-tune" for n in names] + ["ontology\nedit"]
    colors = [GREY] * len(names) + [TEAL]
    ax.bar(xs, vals, 0.6, color=colors, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.12, f"{v:g} GPU-h", ha="center", fontsize=11,
                color=NAVY, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("GPU-hours to change what the system knows", fontsize=10)
    ax.grid(axis="y", color="#E7ECF2", zorder=0)
    ax.set_axisbelow(True)
    ax.margins(y=0.22)
    ax.annotate(f"{er['triples_added']} facts added\nrules enforceable "
                f"{er['rules_enforceable_before']} → {er['rules_enforceable_after']}\n"
                f"model weights unchanged",
                xy=(xs[-1], 0.35), xytext=(xs[-1], max(vals) * 0.52),
                fontsize=12, color=TEAL, weight="bold", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8,
                                shrinkA=6, shrinkB=2))
    fig.suptitle("The same correction, two ways", fontsize=14, color=NAVY,
                 weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p2 = OUT / "deck_fig_cost_of_a_change.png"
    fig.savefig(p2, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {p1.name}, {p2.name}")


if __name__ == "__main__":
    main()
