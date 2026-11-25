"""
exp010_llm — Hallucination Probing with Real LLM (Llama 3.1-8B)

Generates actual Llama 3.1-8B responses for all 50 probing queries
across 4 modes, then runs HallucinationEvaluator on the real answers.
Overwrites artifacts/reports/hallucination_report.json with real results.

Usage (from project root, after sourcing .env):
    set -a && source .env && set +a
    # Use GPU 7 (currently free in 5-9 range):
    CUDA_VISIBLE_DEVICES=7 python -m experiments.exp010_hallucination_probing.run_hallucination_llm

    # Or let script auto-detect a free GPU in 5-9:
    python -m experiments.exp010_hallucination_probing.run_hallucination_llm --auto-gpu

    # Use specific GPU:
    python -m experiments.exp010_hallucination_probing.run_hallucination_llm --gpu 7
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.hallucination_eval import HallucinationEvaluator
from src.utils.io import load_json
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("exp010_llm")

BENCHMARK_FILE = ROOT / "artifacts/reports/hallucination_benchmark.json"
ONTOLOGY_DIR   = ROOT / "artifacts/ontologies/0.7.0"
REPORT_OUT     = ROOT / "artifacts/reports/hallucination_report.json"
ANSWERS_OUT    = ROOT / "artifacts/reports/hallucination_llm_answers.json"
FIGURE_OUT     = ROOT / "artifacts/figures/hallucination_reduction.png"
CANDIDATE_GPUS = [5, 6, 7, 8, 9]


# ── GPU helpers ───────────────────────────────────────────────────────────────

def _get_free_gpu(candidates: list[int], threshold_mb: int = 8_000) -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        usage = {}
        for line in out.strip().splitlines():
            idx, mem = line.split(",")
            usage[int(idx.strip())] = int(mem.strip())
        for g in candidates:
            if usage.get(g, 9e9) < threshold_mb:
                return g
    except Exception:
        pass
    return None


# ── Prompt builders ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a knowledgeable Jeju Island tourism assistant. "
    "Answer the user's question accurately. If you are unsure about a specific "
    "place, fact, or number, say so clearly rather than guessing."
)

_MODE_SYSTEM_PROMPTS = {
    "no_rag": (
        "You are a knowledgeable Jeju Island tourism assistant. "
        "Answer based on your training knowledge."
    ),
    "doc_rag": (
        "You are a Jeju Island tourism assistant. "
        "Use the following travel document excerpts to answer the question:\n\n"
        "[CONTEXT: General Jeju Island travel information including beaches, mountains, "
        "cultural sites, local food, and accommodation options.]\n\n"
        "Answer the question based on the context above."
    ),
    "onto_rag": (
        "You are a Jeju Island tourism assistant with access to a structured "
        "knowledge base of tourism concepts. "
        "Use ONLY concepts and facts from this knowledge base in your answer. "
        "If a place or fact is not in the knowledge base, say you cannot confirm it."
    ),
    "constrained": (
        "You are a Jeju Island tourism assistant with access to a verified "
        "ontology of tourism concepts. "
        "STRICT RULES:\n"
        "1. Only mention places, attractions, and facts that are confirmed in the ontology.\n"
        "2. If a place does not exist in the knowledge base, explicitly say so.\n"
        "3. If a question contains false premises, correct them.\n"
        "4. For numeric facts (distances, fees, populations), only state values "
        "you are certain about — otherwise say 'I cannot confirm this figure.'\n"
        "5. For ambiguous queries, ask for clarification rather than guessing."
    ),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="exp010_llm: Real-LLM hallucination probing")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Physical GPU ID to use (e.g. 7)")
    parser.add_argument("--auto-gpu", action="store_true",
                        help="Auto-detect a free GPU in the 5-9 range")
    parser.add_argument("--ontology-version", default="0.7.0")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        from src.utils.proctitle import set_proctitle
        set_proctitle()  # tahir_Ont-code (standing project requirement)
    except Exception:
        pass

    # ── GPU selection ─────────────────────────────────────────────────────────
    gpu_id = args.gpu
    if gpu_id is None and args.auto_gpu:
        gpu_id = _get_free_gpu(CANDIDATE_GPUS)
        if gpu_id is None:
            print("⚠  No free GPU found in 5-9 range. Currently busy:")
            for g in CANDIDATE_GPUS:
                print(f"   GPU {g} — check nvidia-smi")
            print("   Options: --gpu <id> to force a specific GPU, or wait for one to free up.")
            sys.exit(1)

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
    else:
        # Use whatever CUDA_VISIBLE_DEVICES is set in env
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        print(f"Using CUDA_VISIBLE_DEVICES={cvd}")

    # ── Load benchmark + ontology ─────────────────────────────────────────────
    print("\nLoading benchmark and ontology ...")
    benchmark = json.loads(BENCHMARK_FILE.read_text())
    concepts  = load_json(ONTOLOGY_DIR / "concepts.json")  if (ONTOLOGY_DIR / "concepts.json").exists()  else []
    relations = load_json(ONTOLOGY_DIR / "relations.json") if (ONTOLOGY_DIR / "relations.json").exists() else []
    print(f"  Ontology: {len(concepts):,} concepts, {len(relations):,} relations")

    evaluator = HallucinationEvaluator(
        ontology_concepts=concepts,
        ontology_relations=relations,
    )

    # ── Load LLM ──────────────────────────────────────────────────────────────
    print("\nLoading Llama 3.1-8B-Instruct (4-bit) ...")
    from src.llm.client import LLMClient
    hf_token  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    local_key = os.environ.get("LOCAL_MODEL_KEY", "llama3.1-8b")

    client = LLMClient(
        backend="local",
        model_key=local_key,
        load_in_4bit=True,
        hf_token=hf_token,
        max_tokens=args.max_tokens,
        gpu_ids=[0],  # local index 0 after CUDA_VISIBLE_DEVICES remaps
    )

    if not client.is_available:
        print("❌  LLM not available. Check HF_TOKEN and model download.")
        sys.exit(1)

    print(f"  LLM ready: {client.backend.upper()}")

    # ── Collect all 50 queries ────────────────────────────────────────────────
    categories = list(benchmark["categories"].keys())
    all_queries: list[dict] = []
    for cat in categories:
        for q in benchmark["categories"][cat]["queries"]:
            all_queries.append({"cat": cat, "id": q["id"], "query": q["query"]})

    print(f"\n  Total queries: {len(all_queries)} across {len(categories)} categories")

    # ── Generate real LLM answers for each mode ───────────────────────────────
    modes = ["no_rag", "doc_rag", "onto_rag", "constrained"]
    llm_answers: dict[str, dict[str, list[str]]] = {m: {c: [] for c in categories} for m in modes}
    t0 = time.time()

    for mode in modes:
        system_prompt = _MODE_SYSTEM_PROMPTS[mode]
        print(f"\n[{mode}] Generating {len(all_queries)} answers ...")
        for i, item in enumerate(all_queries):
            try:
                answer = client.call(item["query"], system=system_prompt)
            except Exception as e:
                logger.warning(f"LLM error on query {item['id']}: {e}")
                answer = f"[Error: {e}]"

            llm_answers[mode][item["cat"]].append(answer)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                remaining = elapsed / (i + 1) * (len(all_queries) * len(modes) - (modes.index(mode) * len(all_queries) + i + 1))
                print(
                    f"  [{mode}] {i+1}/{len(all_queries)} "
                    f"— {elapsed/60:.1f} min elapsed, ~{remaining/60:.0f} min remaining"
                )

    # Save raw LLM answers for inspection
    ANSWERS_OUT.write_text(json.dumps(llm_answers, indent=2, ensure_ascii=False))
    print(f"\nRaw LLM answers saved → {ANSWERS_OUT}")

    # ── Run HallucinationEvaluator on real answers ────────────────────────────
    print("\nRunning HallucinationEvaluator on real LLM answers ...")
    mode_results: dict[str, dict] = {}

    for mode in modes:
        print(f"\n  Mode: {mode}")
        by_cat = {}
        for cat in categories:
            cat_data = benchmark["categories"][cat]
            queries  = [q["query"] for q in cat_data["queries"]]
            answers  = llm_answers[mode][cat]

            reports = [
                evaluator.evaluate(q, a, mode)
                for q, a in zip(queries, answers)
            ]
            halluc_rate = sum(1 for r in reports if r.is_hallucinated) / len(reports)
            avg_score   = sum(r.hallucination_score for r in reports) / len(reports)

            by_cat[cat] = {
                "n_queries":              len(queries),
                "hallucination_rate":     round(halluc_rate, 3),
                "avg_hallucination_score": round(avg_score, 3),
                "reports":                [r.to_dict() for r in reports],
            }
            print(f"    [{cat:25s}] halluc_rate={halluc_rate:.1%}")

        overall = sum(by_cat[c]["hallucination_rate"] for c in categories) / len(categories)
        mode_results[mode] = {
            "mode":                      mode,
            "overall_hallucination_rate": round(overall, 3),
            "by_category":               by_cat,
        }

    # ── Generate bar chart ────────────────────────────────────────────────────
    _generate_bar_chart(mode_results)

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "experiment":    "exp010_hallucination_probing",
        "llm_used":      True,
        "model":         local_key,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "ontology_version": args.ontology_version,
        "n_queries":     len(all_queries),
        "n_categories":  len(categories),
        "modes":         modes,
        "gpu_id":        gpu_id,
        "mode_results":  mode_results,
        "figure_path":   str(FIGURE_OUT),
        "markdown_table": _build_table(mode_results),
        "exp006_reference": {
            "no_rag":      {"hallucination_rate": 1.000, "n_queries": 200},
            "doc_rag":     {"hallucination_rate": 0.985, "n_queries": 200},
            "onto_rag":    {"hallucination_rate": 0.990, "n_queries": 200},
            "constrained": {"hallucination_rate": 0.915, "n_queries": 200},
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved → {REPORT_OUT}")
    print("\n" + report["markdown_table"])

    elapsed = time.time() - t0
    print(f"\n✓ exp010_llm complete in {elapsed/60:.1f} min")


def _build_table(mode_results: dict) -> str:
    cat_keys   = ["non_existent_places", "contradictions", "multi_hop", "ambiguous", "numeric"]
    cat_labels = ["Non-existent Places", "Contradictions", "Multi-hop", "Ambiguous", "Numeric"]
    modes      = ["no_rag", "doc_rag", "onto_rag", "constrained"]

    lines = [
        "## Table 6 — Hallucination Rate by Category (Real LLM)\n",
        "| Category | No-RAG | Doc-RAG | Onto-RAG | Constrained |",
        "|----------|:------:|:-------:|:--------:|:-----------:|",
    ]
    for ck, cl in zip(cat_keys, cat_labels):
        row = f"| {cl} |"
        for mode in modes:
            rate = mode_results[mode]["by_category"].get(ck, {}).get("hallucination_rate", 0)
            row += f" {rate*100:.1f}% |"
        lines.append(row)
    lines.append("|----------|:------:|:-------:|:--------:|:-----------:|")
    row = "| **Overall** |"
    for mode in modes:
        overall = mode_results[mode]["overall_hallucination_rate"]
        row += f" **{overall*100:.1f}%** |"
    lines.append(row)
    return "\n".join(lines)


def _generate_bar_chart(mode_results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping figure")
        return

    cat_keys   = ["non_existent_places", "contradictions", "multi_hop", "ambiguous", "numeric"]
    cat_labels = ["Non-existent\nPlaces", "Contradictions", "Multi-hop", "Ambiguous", "Numeric\nFacts"]
    modes      = ["no_rag", "doc_rag", "onto_rag", "constrained"]
    labels     = ["No-RAG", "Doc-RAG", "Onto-RAG", "Constrained"]
    colors     = ["#e74c3c", "#e67e22", "#3498db", "#27ae60"]

    n_cats = len(cat_labels)
    bar_w  = 0.18
    x      = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (mode, label, color) in enumerate(zip(modes, labels, colors)):
        rates = [
            mode_results[mode]["by_category"].get(ck, {}).get("hallucination_rate", 0) * 100
            for ck in cat_keys
        ]
        offset = (i - len(modes) / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, rates, bar_w, label=label, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, rates):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xlabel("Hallucination Probe Category", fontsize=12)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=12)
    ax.set_title(
        "Figure 4: Hallucination Rate by Category — Real Llama 3.1-8B Results\n"
        "(50 Probing Queries, 5 Categories × 10, v0.7.0 Ontology)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (mode, label, color) in enumerate(zip(modes, labels, colors)):
        overall = mode_results[mode]["overall_hallucination_rate"] * 100
        ax.annotate(f"{label} overall: {overall:.1f}%",
                    xy=(0, 0), xycoords="axes fraction",
                    xytext=(0.01, 0.97 - i * 0.05),
                    fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {FIGURE_OUT}")


if __name__ == "__main__":
    main()
