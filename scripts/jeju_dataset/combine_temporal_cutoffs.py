"""
Combine per-model temporal-metric reports (1000-Q run) into SEPARATE Case A vs
Case B tables for each query count: 100, 200, 500, 1000 — across all 4 models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.proctitle import set_proctitle

REPORTS = ROOT / "artifacts" / "reports"
MODELS = [
    ("Llama-3.1-8B", "temporal_metrics_t1000_llama.json"),
    ("DeepSeek-R1-8B", "temporal_metrics_t1000_deepseek.json"),
    ("Qwen3-8B", "temporal_metrics_t1000_qwen3.json"),
    ("Mistral-7B", "temporal_metrics_t1000_mistral.json"),
]
METRICS = [("BERT-F1", "bertf1"), ("METEOR", "meteor"), ("COMET", "comet"),
           ("Key-token", "keytok"), ("LLM-judge", "judge"), ("TTFT(ms)", "ttft"), ("TPOT(ms)", "tpot")]
CUTOFFS = ["100", "200", "500", "1000"]
HIGHER = {"bertf1", "meteor", "comet", "keytok", "judge"}


def fmt(v, key):
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:.1f}" if key in ("ttft", "tpot") else f"{v:.3f}"


def main() -> None:
    set_proctitle()
    data = {}
    for name, fn in MODELS:
        p = REPORTS / fn
        if p.exists():
            data[name] = json.loads(p.read_text()).get("by_cutoff", {})
        else:
            print(f"  (missing {fn})")

    G, RST = "\033[92m", "\033[0m"
    md = ["# Temporal Case A vs Case B — scaled benchmark (comments.txt), by query count\n"]
    for n in CUTOFFS:
        present = {m: agg for m, agg in data.items() if n in agg}
        if not present:
            continue
        title = f"Query count = {n}"
        print("\n" + "=" * 104)
        print(f"  {title}  —  Case A (no ontology) vs Case B (with temporal ontology), 4 models")
        print("=" * 104)
        print(f"  {'Model':<16}{'Case':<15}" + "".join(f"{m:>11}" for m, _ in METRICS))
        print("  " + "-" * 100)
        md += [f"## {title}", "",
               "| Model | Case | " + " | ".join(m for m, _ in METRICS) + " |",
               "|" + "---|" * (len(METRICS) + 2)]
        for name, agg in present.items():
            c = agg[n]
            for case in ("A", "B"):
                cells = "".join(f"{fmt(c[k][case], k):>11}" for _, k in METRICS)
                lab = "A (no onto)" if case == "A" else "B (with onto)"
                col = G if case == "B" else ""
                print(f"  {col}{name:<16}{lab:<15}{cells}{RST}")
                mdcells = " | ".join(fmt(c[k][case], k) for _, k in METRICS)
                md.append(f"| {name} | {'A (no onto)' if case=='A' else '**B (with onto)**'} | {mdcells} |")
            print("  " + "-" * 100)
        md.append("")
    (REPORTS / "temporal_cutoffs_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n  Markdown → artifacts/reports/temporal_cutoffs_comparison.md")


if __name__ == "__main__":
    main()
