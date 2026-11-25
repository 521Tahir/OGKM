"""
Experiment 015 — LLM-Powered Comparative Mode Benchmark

Runs all four pipeline modes (no_rag / doc_rag / onto_rag / constrained)
on 15 curated questions using the real Llama 3.1-8B model and compares:
  - Answer quality (entity recall, precision)
  - Hallucination rate
  - Constraint violations
  - Grounding score
  - Latency

Usage:
    LOAD_LLM=1 CUDA_VISIBLE_DEVICES=5,6 python experiments/exp015_llm_benchmark/run_llm_benchmark.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import save_json, ensure_dir
from src.utils.logging_utils import setup_logging, get_logger

setup_logging(log_level="WARNING")
logger = get_logger("exp015")

REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
ensure_dir(REPORTS_DIR)
ensure_dir(FIGURES_DIR)

MODES = ["no_rag", "doc_rag", "onto_rag", "constrained"]

# 15 curated questions across 5 topic categories (3 each)
BENCHMARK_QUESTIONS = [
    # Beaches & Nature
    {"id": "B1", "category": "Beaches",   "question": "What are the best beaches to visit in Jeju Island?"},
    {"id": "B2", "category": "Beaches",   "question": "Which beaches in Jeju are good for families with children?"},
    {"id": "B3", "category": "Beaches",   "question": "Is Hyeopjae Beach good for swimming?"},
    # Hiking & Mountains
    {"id": "H1", "category": "Hiking",    "question": "How do I hike Hallasan? What trails are available?"},
    {"id": "H2", "category": "Hiking",    "question": "What is the difficulty level of Eorimok trail on Hallasan?"},
    {"id": "H3", "category": "Hiking",    "question": "What are popular outdoor activities in Jeju besides hiking?"},
    # Food & Cuisine
    {"id": "F1", "category": "Food",      "question": "What is Jeju black pork and where can I eat it?"},
    {"id": "F2", "category": "Food",      "question": "What are the must-try traditional foods in Jeju Island?"},
    {"id": "F3", "category": "Food",      "question": "Where can I find the best haenyeo-caught seafood in Jeju?"},
    # Accommodation
    {"id": "A1", "category": "Accommodation", "question": "What types of accommodation are available in Jeju?"},
    {"id": "A2", "category": "Accommodation", "question": "Are there luxury resorts near Jeju's beaches?"},
    {"id": "A3", "category": "Accommodation", "question": "What is a good area to stay for exploring Jeju City?"},
    # Culture & Attractions
    {"id": "C1", "category": "Culture",   "question": "What cultural attractions should I visit in Jeju?"},
    {"id": "C2", "category": "Culture",   "question": "What is Jeju Haenyeo and where can I learn about them?"},
    {"id": "C3", "category": "Culture",   "question": "What is Seongsan Ilchulbong and why is it famous?"},
]

# Jeju-specific entity set for grounding/recall checks
JEJU_ENTITIES = {
    "hallasan", "hyeopjae", "hamdeok", "jungmun", "seongsan", "ilchulbong",
    "eorimok", "haenyeo", "black pork", "jeju", "olle", "osulloc",
    "manjanggul", "cheonjiyeon", "jusangjeolli", "loveland", "samseonghyeol",
    "gwandeokjeong", "yongduam", "bijarim", "songaksan",
}


def _entity_scores(answer: str) -> tuple[float, float]:
    """Simple entity recall and precision against JEJU_ENTITIES."""
    words = set(answer.lower().split())
    hits  = sum(1 for e in JEJU_ENTITIES if e in answer.lower())
    recall    = round(hits / len(JEJU_ENTITIES), 3)
    precision = round(hits / max(1, len(words)) * 10, 3)  # scaled
    return recall, min(1.0, precision)


def run_benchmark(sys_obj, max_tokens_override: int = 300) -> list[dict]:
    """Run all questions × modes, return raw result records."""
    results = []
    total   = len(BENCHMARK_QUESTIONS) * len(MODES)
    done    = 0

    # Patch generator max_new_tokens for speed
    if sys_obj.llm is not None:
        sys_obj.llm.max_new_tokens = max_tokens_override

    for item in BENCHMARK_QUESTIONS:
        for mode in MODES:
            done += 1
            print(f"  [{done:>2}/{total}] [{mode:>11}] {item['id']}: {item['question'][:55]}...", flush=True)

            t0 = time.perf_counter()
            try:
                r = sys_obj.query(item["question"], mode)
            except Exception as e:
                logger.warning(f"Query failed: {e}")
                r = {"answer": "", "grounding": 0.0, "violations": [], "concepts": [], "elapsed_ms": 0}

            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            answer  = r.get("answer", "")
            recall, precision = _entity_scores(answer)

            # Hallucination proxy: answer contains entities NOT in ontology context
            concepts_used = set(c.lower() for c in r.get("concepts", []))
            hallucinated  = any(
                e in answer.lower() and e not in concepts_used
                for e in JEJU_ENTITIES
            ) if mode != "no_rag" else False

            results.append({
                "id":          item["id"],
                "category":    item["category"],
                "question":    item["question"],
                "mode":        mode,
                "answer":      answer,
                "answer_len":  len(answer.split()),
                "grounding":   r.get("grounding", 0.0),
                "violations":  len(r.get("violations", [])),
                "n_concepts":  len(r.get("concepts", [])),
                "entity_recall":    recall,
                "entity_precision": precision,
                "hallucinated":     hallucinated,
                "elapsed_ms":  elapsed,
            })

    return results


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-mode metrics."""
    from collections import defaultdict
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        buckets[r["mode"]].append(r)

    summary = {}
    for mode, items in buckets.items():
        n = len(items)
        summary[mode] = {
            "n": n,
            "entity_recall":    round(sum(i["entity_recall"] for i in items) / n, 3),
            "entity_precision": round(sum(i["entity_precision"] for i in items) / n, 3),
            "grounding":        round(sum(i["grounding"] for i in items) / n, 3),
            "violations_avg":   round(sum(i["violations"] for i in items) / n, 2),
            "hallucination_rate": round(sum(1 for i in items if i["hallucinated"]) / n, 3),
            "avg_answer_len":   round(sum(i["answer_len"] for i in items) / n, 1),
            "avg_latency_ms":   round(sum(i["elapsed_ms"] for i in items) / n, 1),
        }
    return summary


def print_table(summary: dict) -> None:
    """Print a markdown-style comparison table."""
    print("\n" + "=" * 85)
    print("  COMPARATIVE BENCHMARK — Jeju Tourism RAG  (Llama 3.1-8B, v0.7.0 ontology)")
    print("=" * 85)
    header = f"{'Mode':<14} {'Recall':>8} {'Precision':>10} {'Grounding':>10} {'Violations':>11} {'Halluc%':>8} {'Len':>5} {'ms':>7}"
    print(header)
    print("-" * 85)
    for mode in MODES:
        m = summary.get(mode, {})
        print(
            f"{mode:<14} "
            f"{m.get('entity_recall',0):>8.3f} "
            f"{m.get('entity_precision',0):>10.3f} "
            f"{m.get('grounding',0):>10.3f} "
            f"{m.get('violations_avg',0):>11.2f} "
            f"{m.get('hallucination_rate',0)*100:>7.1f}% "
            f"{m.get('avg_answer_len',0):>5.0f} "
            f"{m.get('avg_latency_ms',0):>7.0f}"
        )
    print("=" * 85)


def plot_results(summary: dict, results: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    modes  = MODES
    colors = ["#607D8B", "#4CAF50", "#2196F3", "#FF5722"]

    # ── Figure 1: 4-metric bar chart ─────────────────────────────────────────
    metrics = ["entity_recall", "entity_precision", "grounding", "hallucination_rate"]
    labels  = ["Entity Recall", "Entity Precision", "Grounding Score", "Hallucination Rate"]
    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (mode, col) in enumerate(zip(modes, colors)):
        vals = [summary[mode].get(m, 0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=mode, color=col, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Jeju Tourism RAG — Mode Comparison (Llama 3.1-8B + Ontology v0.7.0)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout()
    out1 = str(FIGURES_DIR / "exp015_mode_comparison.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}")

    # ── Figure 2: Latency vs grounding scatter per mode ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, col in zip(modes, colors):
        items = [r for r in results if r["mode"] == mode]
        xs = [r["elapsed_ms"] / 1000 for r in items]
        ys = [r["grounding"] for r in items]
        ax.scatter(xs, ys, color=col, label=mode, alpha=0.7, s=60)
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Grounding Score")
    ax.set_title("Latency vs Grounding Score by Mode")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out2 = str(FIGURES_DIR / "exp015_latency_vs_grounding.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}")

    # ── Figure 3: Per-category grounding heatmap ──────────────────────────────
    from collections import defaultdict
    cats = sorted({r["category"] for r in results})
    data = []
    for mode in modes:
        row = []
        for cat in cats:
            vals = [r["grounding"] for r in results if r["mode"] == mode and r["category"] == cat]
            row.append(round(sum(vals) / max(1, len(vals)), 2))
        data.append(row)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(modes))); ax.set_yticklabels(modes, fontsize=9)
    for i in range(len(modes)):
        for j in range(len(cats)):
            ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if data[i][j] < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Grounding Score")
    ax.set_title("Grounding Score by Mode × Category")
    plt.tight_layout()
    out3 = str(FIGURES_DIR / "exp015_grounding_heatmap.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}")


def main() -> None:
    print("\n=== Exp015: LLM-Powered Comparative Benchmark ===")
    print(f"  Questions: {len(BENCHMARK_QUESTIONS)} | Modes: {len(MODES)} | Total runs: {len(BENCHMARK_QUESTIONS)*len(MODES)}")

    os.environ["LOAD_LLM"] = "1"

    from src.cli.interactive import TourismSystem
    print("\nLoading system...", flush=True)
    sys_obj = TourismSystem()
    sys_obj.load()
    print(f"  Model: {sys_obj.model_name}  |  Concepts: {sys_obj.n_concepts:,}\n")

    print("Running benchmark...")
    t_start = time.perf_counter()
    results = run_benchmark(sys_obj, max_tokens_override=300)
    t_total = round(time.perf_counter() - t_start, 1)

    summary = aggregate(results)
    print_table(summary)
    print(f"\n  Total time: {t_total:.0f}s ({t_total/60:.1f} min)")

    print("\nGenerating plots...")
    plot_results(summary, results)

    report = {
        "experiment":      "exp015_llm_benchmark",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "ontology_version": "0.7.0",
        "model":           sys_obj.model_name,
        "n_questions":     len(BENCHMARK_QUESTIONS),
        "modes":           MODES,
        "total_time_s":    t_total,
        "summary":         summary,
        "results":         results,
    }
    out = REPORTS_DIR / "exp015_llm_benchmark.json"
    save_json(report, out)
    print(f"\n  Report saved: {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
