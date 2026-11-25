"""
Experiment 021 — Temporal Ontology Demo (Case A vs Case B)

Implements the comparative demo from comments.txt: a vanilla (non-fine-tuned)
local LLM answers time-dependent Jeju questions under two conditions —

  Case A (Without Ontology): the LLM alone.
  Case B (With Ontology):    the same LLM + injected temporal-ontology facts.

Each answer is scored two ways:
  - deterministic key-token check (gold month/season strings present), and
  - LLM-as-judge against the gold answer (correct / partial / incorrect).

Outputs:
  artifacts/reports/temporal_demo_report.json   — per-question + aggregates
  artifacts/reports/temporal_demo_report.md     — comparison tables
  artifacts/figures/temporal_demo_comparison.png — Case A vs B accuracy bar chart

Usage:
  python experiments/exp021_temporal_demo/run_temporal_demo.py
  python experiments/exp021_temporal_demo/run_temporal_demo.py --limit 5
  python experiments/exp021_temporal_demo/run_temporal_demo.py --no-judge   # token scoring only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm.client import LLMClient
from src.retrieval.temporal_retriever import TemporalRetriever
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("exp021")

BENCHMARK = ROOT / "data" / "temporal" / "temporal_qa_benchmark.json"
REPORTS = ROOT / "artifacts" / "reports"
FIGURES = ROOT / "artifacts" / "figures"

SYSTEM = "You are a concise tourism assistant for Jeju Island, South Korea. Answer in 1-2 sentences."

CASE_A_PROMPT = "Question: {q}\nAnswer concisely."

CASE_B_PROMPT = (
    "{context}\n\n"
    "Using ONLY the authoritative temporal facts above when relevant, answer the question. "
    "Be specific about months or seasons.\n\n"
    "Question: {q}\nAnswer concisely."
)

JUDGE_PROMPT = (
    "You are grading whether a candidate answer matches the reference answer for a "
    "time-dependent question. Focus on whether the months/season are correct.\n\n"
    "Question: {q}\n"
    "Reference (correct) answer: {gold}\n"
    "Candidate answer: {ans}\n\n"
    "Respond with ONLY one word: CORRECT (months/season match), PARTIAL (partially right), "
    "or INCORRECT (wrong or missing). Verdict:"
)


def token_score(answer: str, key_tokens: list[str]) -> float:
    """1.0 if all gold key tokens appear, 0.5 if some, 0.0 if none."""
    a = answer.lower()
    hits = sum(1 for t in key_tokens if t.lower() in a)
    if hits == len(key_tokens):
        return 1.0
    return 0.5 if hits > 0 else 0.0


def judge(llm: LLMClient, q: str, gold: str, ans: str) -> float:
    """LLM-as-judge → 1.0 / 0.5 / 0.0."""
    try:
        verdict = llm.call(JUDGE_PROMPT.format(q=q, gold=gold, ans=ans),
                           system="You are a strict grader.", max_tokens=8, temperature=0.0)
    except Exception as e:  # pragma: no cover
        logger.warning("judge failed: %s", e)
        return -1.0
    v = verdict.strip().upper()
    if "CORRECT" in v and "INCORRECT" not in v:
        return 1.0
    if "PARTIAL" in v:
        return 0.5
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal ontology demo (Case A vs B)")
    ap.add_argument("--version", default="1.2.0", help="ontology version with temporal layer")
    ap.add_argument("--limit", type=int, default=0, help="limit number of questions")
    ap.add_argument("--top-k", type=int, default=5, help="temporal facts to inject (Case B)")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge (token scoring only)")
    ap.add_argument("--max-tokens", type=int, default=160)
    args = ap.parse_args()

    ensure_dir(REPORTS); ensure_dir(FIGURES)

    questions = load_json(BENCHMARK)["questions"]
    if args.limit:
        questions = questions[: args.limit]

    retriever = TemporalRetriever.from_version(args.version)
    llm = LLMClient(backend="local", max_tokens=args.max_tokens, temperature=0.0)
    if not llm.is_available:
        logger.error("No local LLM backend available. Set HF_TOKEN + LOCAL_MODEL_KEY in .env.")
        sys.exit(1)
    logger.info("LLM backend: %s", llm.backend)

    results = []
    for i, item in enumerate(questions, 1):
        q, gold = item["question"], item["gold_answer"]
        logger.info("[%d/%d] %s", i, len(questions), q)

        # Case A — LLM alone
        ans_a = llm.call(CASE_A_PROMPT.format(q=q), system=SYSTEM).strip()
        # Case B — LLM + temporal ontology
        tr = retriever.retrieve(q, top_k=args.top_k)
        ans_b = llm.call(CASE_B_PROMPT.format(context=tr.context, q=q), system=SYSTEM).strip()

        rec = {
            "id": item["id"], "category": item["category"], "question": q, "gold_answer": gold,
            "case_a_answer": ans_a, "case_b_answer": ans_b,
            "case_a_token": token_score(ans_a, item["key_tokens"]),
            "case_b_token": token_score(ans_b, item["key_tokens"]),
            "injected_facts": [f.to_sentence() for f in tr.facts],
        }
        if not args.no_judge:
            rec["case_a_judge"] = judge(llm, q, gold, ans_a)
            rec["case_b_judge"] = judge(llm, q, gold, ans_b)
        results.append(rec)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def acc(key: str) -> float:
        vals = [r[key] for r in results if r.get(key, -1) >= 0]
        return sum(vals) / len(vals) if vals else 0.0

    metric_keys = ["case_a_token", "case_b_token"]
    if not args.no_judge:
        metric_keys += ["case_a_judge", "case_b_judge"]
    overall = {k: round(acc(k), 4) for k in metric_keys}

    by_cat: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for k in metric_keys:
            if r.get(k, -1) >= 0:
                by_cat[r["category"]][k].append(r[k])
    by_category = {
        cat: {k: round(sum(v) / len(v), 3) for k, v in d.items()}
        for cat, d in by_cat.items()
    }

    report = {
        "experiment": "exp021_temporal_demo",
        "date": datetime.now().isoformat(),
        "ontology_version": args.version,
        "llm_backend": llm.backend,
        "n_questions": len(results),
        "judge": not args.no_judge,
        "overall": overall,
        "by_category": by_category,
        "results": results,
    }
    save_json(report, REPORTS / "temporal_demo_report.json")
    (REPORTS / "temporal_demo_report.md").write_text(_markdown(report), encoding="utf-8")
    _plot(report)

    _print_console(report)


def _markdown(rep: dict) -> str:
    o = rep["overall"]
    L = [
        "# Temporal Ontology Demo — Case A (no ontology) vs Case B (with ontology)",
        "", f"**Date:** {rep['date']}  ",
        f"**LLM:** {rep['llm_backend']} (vanilla) · **Ontology:** v{rep['ontology_version']} · "
        f"**Questions:** {rep['n_questions']}",
        "", "## Overall accuracy", "",
        "| Scorer | Case A (no onto) | Case B (with onto) | Gain |",
        "|--------|:----------------:|:------------------:|:----:|",
    ]
    if "case_a_judge" in o:
        L.append(f"| LLM-judge | {o['case_a_judge']:.3f} | {o['case_b_judge']:.3f} | "
                 f"{o['case_b_judge']-o['case_a_judge']:+.3f} |")
    L.append(f"| Key-token | {o['case_a_token']:.3f} | {o['case_b_token']:.3f} | "
             f"{o['case_b_token']-o['case_a_token']:+.3f} |")
    L += ["", "## By category (key-token accuracy)", "",
          "| Category | Case A | Case B |", "|----------|:------:|:------:|"]
    for cat, d in sorted(rep["by_category"].items()):
        L.append(f"| {cat} | {d.get('case_a_token',0):.2f} | {d.get('case_b_token',0):.2f} |")
    L += ["", "## Example answers", ""]
    for r in rep["results"][:5]:
        L += [f"**Q{r['id']}: {r['question']}**  ",
              f"- Gold: {r['gold_answer']}  ",
              f"- Case A: {r['case_a_answer']}  ",
              f"- Case B: {r['case_b_answer']}  ", ""]
    return "\n".join(L)


def _plot(rep: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        logger.warning("matplotlib unavailable, skipping plot: %s", e)
        return
    o = rep["overall"]
    scorers, a_vals, b_vals = [], [], []
    if "case_a_judge" in o:
        scorers.append("LLM-judge"); a_vals.append(o["case_a_judge"]); b_vals.append(o["case_b_judge"])
    scorers.append("Key-token"); a_vals.append(o["case_a_token"]); b_vals.append(o["case_b_token"])
    x = range(len(scorers)); w = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar([i - w/2 for i in x], a_vals, w, label="Case A — no ontology", color="#c0392b")
    plt.bar([i + w/2 for i in x], b_vals, w, label="Case B — with temporal ontology", color="#27ae60")
    plt.xticks(list(x), scorers); plt.ylim(0, 1.05); plt.ylabel("Accuracy")
    plt.title("Temporal QA accuracy: vanilla LLM with vs without ontology")
    for i, v in enumerate(a_vals): plt.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(b_vals): plt.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.legend(); plt.tight_layout()
    out = FIGURES / "temporal_demo_comparison.png"
    plt.savefig(out, dpi=200); plt.close()
    logger.info("Figure → %s", out)


def _print_console(rep: dict) -> None:
    o = rep["overall"]
    G, R, B, RST, C = "\033[92m", "\033[91m", "\033[1m", "\033[0m", "\033[96m"
    print("\n" + "=" * 72)
    print(f"{B}TEMPORAL ONTOLOGY DEMO — Case A (no ontology) vs Case B (with ontology){RST}")
    print("=" * 72)
    print(f"  LLM: {rep['llm_backend']} (vanilla) · ontology v{rep['ontology_version']} · "
          f"{rep['n_questions']} questions\n")
    hdr = f"  {'Scorer':<14}{'Case A (no onto)':>18}{'Case B (with onto)':>20}{'Gain':>10}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = [("LLM-judge", "case_a_judge", "case_b_judge")] if "case_a_judge" in o else []
    rows.append(("Key-token", "case_a_token", "case_b_token"))
    for name, ka, kb in rows:
        a, b = o[ka], o[kb]
        gain = b - a
        col = G if gain > 0 else (R if gain < 0 else "")
        print(f"  {name:<14}{a:>18.3f}{b:>20.3f}{col}{gain:>+10.3f}{RST}")
    print(f"\n  {C}Report → artifacts/reports/temporal_demo_report.md{RST}")
    print(f"  {C}Figure → artifacts/figures/temporal_demo_comparison.png{RST}")
    print("=" * 72)


if __name__ == "__main__":
    main()
