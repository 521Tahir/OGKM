"""
Combine per-model temporal-metric reports into one baseline comparison
(Case A without ontology vs Case B with ontology) and a presentation figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.proctitle import set_proctitle

REPORTS = ROOT / "artifacts" / "reports"
FIGURES = ROOT / "artifacts" / "figures"

MODELS = [
    ("Llama-3.1-8B", "temporal_metrics_full50.json"),
    ("DeepSeek-R1-8B", "temporal_metrics_deepseek.json"),
    ("Qwen3-8B", "temporal_metrics_qwen3.json"),
    ("Mistral-7B", "temporal_metrics_mistral.json"),
]
METRICS = [("BERT-F1", "bertf1"), ("METEOR", "meteor"), ("COMET", "comet"),
           ("Key-token", "keytok"), ("LLM-judge", "judge"), ("TTFT(ms)", "ttft"), ("TPOT(ms)", "tpot")]
HIGHER = {"bertf1", "meteor", "comet", "keytok", "judge"}


def main() -> None:
    set_proctitle()
    data = {}
    for name, fn in MODELS:
        p = REPORTS / fn
        if p.exists():
            data[name] = json.loads(p.read_text())["aggregate"]
        else:
            print(f"  (missing {fn} for {name})")

    # ── markdown + console table ───────────────────────────────────────────────
    hdr = "| Model | Case | " + " | ".join(m for m, _ in METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    lines = ["# Baseline comparison — comments.txt (Case A vs Case B), 50 questions, ontology v1.2.0",
             "", hdr, sep]
    for name in data:
        agg = data[name]
        for case in ("A", "B"):
            vals = []
            for _, k in METRICS:
                v = agg[k][case]
                vals.append(f"{v:.3f}" if isinstance(v, (int, float)) else "n/a")
            label = "A (no onto)" if case == "A" else "**B (with onto)**"
            lines.append(f"| {name} | {label} | " + " | ".join(vals) + " |")
    md = "\n".join(lines)
    (REPORTS / "baseline_models_comparison.md").write_text(md, encoding="utf-8")

    print("\n" + "=" * 100)
    print("  FINAL BASELINE TABLE — vanilla LLMs, Case A (no ontology) vs Case B (with ontology)")
    print("=" * 100)
    print(f"  {'Model':<16}{'Case':<14}" + "".join(f"{m:>11}" for m, _ in METRICS))
    print("  " + "-" * 96)
    G, Y, RST = "\033[92m", "\033[93m", "\033[0m"
    for name in data:
        agg = data[name]
        for case in ("A", "B"):
            cells = ""
            for _, k in METRICS:
                v = agg[k][case]
                cells += f"{v:>11.3f}" if isinstance(v, (int, float)) else f"{'n/a':>11}"
            lab = "A (no onto)" if case == "A" else "B (with onto)"
            col = G if case == "B" else ""
            print(f"  {col}{name:<16}{lab:<14}{cells}{RST}")
        print("  " + "-" * 96)

    # ── figure: small multiples (one subplot per metric, 3 models × A/B) ───────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(data.keys())
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()
        w = 0.36
        for ax_i, (label, key) in enumerate(METRICS):
            ax = axes[ax_i]
            A = [data[n][key]["A"] or 0 for n in names]
            B = [data[n][key]["B"] or 0 for n in names]
            x = range(len(names))
            ax.bar([i - w/2 for i in x], A, w, label="Case A", color="#c0392b")
            ax.bar([i + w/2 for i in x], B, w, label="Case B", color="#27ae60")
            ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
            ax.set_title(label + ("  (↓ better)" if key not in HIGHER else "  (↑ better)"), fontsize=10)
            if key in HIGHER:
                ax.set_ylim(0, 1.1)
            for i, (va, vb) in enumerate(zip(A, B)):
                fmt = "{:.0f}" if key in ("ttft", "tpot") else "{:.2f}"
                ax.text(i - w/2, va, fmt.format(va), ha="center", va="bottom", fontsize=7)
                ax.text(i + w/2, vb, fmt.format(vb), ha="center", va="bottom", fontsize=7, fontweight="bold")
            if ax_i == 0:
                ax.legend(fontsize=8)
        axes[-1].axis("off")
        fig.suptitle("comments.txt — Vanilla LLM baselines: Case A (no ontology) vs Case B (with temporal ontology)\n"
                     "50 Jeju time-dependent questions · ontology v1.2.0",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = FIGURES / "baseline_models_comparison.png"
        fig.savefig(out, dpi=180); plt.close(fig)
        print(f"\n  Figure  → {out.relative_to(ROOT)}")
    except Exception as e:
        print(f"  figure skipped: {e}")
    print(f"  Markdown→ artifacts/reports/baseline_models_comparison.md")
    print("=" * 100)


if __name__ == "__main__":
    main()
