"""
Experiment 022 — Rich metric comparison for comments.txt (Case A vs Case B).

For each tough time-dependent Jeju question, generates an answer with a vanilla
LLM under two conditions — Case A (no ontology) and Case B (temporal ontology
injected) — and scores both against the gold answer with:

  Text quality : BERT-F1 (BERTScore), METEOR, COMET
  Correctness  : key-token match, LLM-as-judge
  Latency      : TTFT (time-to-first-token, ms), TPOT (time-per-output-token, ms)

Outputs a per-question table and an aggregate Case A vs Case B table.

Usage:
  CUDA_VISIBLE_DEVICES=6 python experiments/exp022_temporal_metrics/run_temporal_metrics.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.proctitle import set_proctitle
from src.retrieval.temporal_retriever import TemporalRetriever
from src.utils.io import load_json, save_json, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger("exp022")

REPORTS = ROOT / "artifacts" / "reports"
FIGURES = ROOT / "artifacts" / "figures"

SYSTEM = "You are a concise tourism assistant for Jeju Island, South Korea. Answer in 1-2 sentences."
CASE_A = "Question: {q}\nAnswer concisely."
CASE_B = ("{ctx}\n\nUsing ONLY the authoritative temporal facts above when relevant, answer the "
          "question. Be specific about months or seasons.\n\nQuestion: {q}\nAnswer concisely.")
JUDGE = ("Grade if the candidate matches the reference for a time-dependent question (months/season).\n"
         "Question: {q}\nReference: {gold}\nCandidate: {ans}\n"
         "Reply ONE word: CORRECT, PARTIAL, or INCORRECT. Verdict:")


def key_token(ans, toks):
    a = ans.lower(); h = sum(1 for t in toks if t.lower() in a)
    return 1.0 if h == len(toks) else (0.5 if h else 0.0)


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--qset", default="data/temporal/tough_qa_10.json",
                    help="question set JSON (e.g. data/temporal/temporal_qa_benchmark.json)")
    ap.add_argument("--tag", default="tough10", help="output tag for report/figure")
    ap.add_argument("--model-key", default="llama3.1-8b",
                    help="llama3.1-8b | deepseek-r1-8b | qwen3-8b | ...")
    args = ap.parse_args()
    ensure_dir(REPORTS); ensure_dir(FIGURES)

    questions = load_json(ROOT / args.qset)["questions"]
    retr = TemporalRetriever.from_version("1.2.0")

    from src.llm.client import LLMClient
    llm = LLMClient(backend="local", model_key=args.model_key, max_tokens=args.max_tokens, temperature=0.0)
    if not llm.is_available:
        logger.error("LLM unavailable"); sys.exit(1)
    client = llm._client
    # Reasoning models (DeepSeek-R1, Qwen3) emit long <think>; skip it for concise QA.
    is_reasoning = any(k in args.model_key.lower() for k in ("r1", "deepseek", "qwen3"))
    APREFIX = "<think>\n\n</think>\n\n" if is_reasoning else ""

    def strip_think(t):
        import re
        return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).replace("</think>", "").strip()

    def gen(prompt):
        """stream_call to capture TTFT/TPOT; suppress token printing."""
        with contextlib.redirect_stdout(io.StringIO()):
            ans = client.stream_call(prompt, system=SYSTEM, max_tokens=args.max_tokens,
                                     temperature=0.0, assistant_prefix=APREFIX)
        m = client._last_metrics
        return strip_think(ans), m.get("ttft_ms"), m.get("tpot_ms")

    rows = []
    for i, item in enumerate(questions, 1):
        q, gold = item["question"], item["gold_answer"]
        logger.info("[%d/%d] %s", i, len(questions), q)
        a_ans, a_ttft, a_tpot = gen(CASE_A.format(q=q))
        ctx = retr.retrieve(q, top_k=5).context
        b_ans, b_ttft, b_tpot = gen(CASE_B.format(ctx=ctx, q=q))
        rows.append({"id": item["id"], "question": q, "gold": gold,
                     "A_ans": a_ans, "B_ans": b_ans,
                     "A_ttft": a_ttft, "A_tpot": a_tpot, "B_ttft": b_ttft, "B_tpot": b_tpot,
                     "A_keytok": key_token(a_ans, item["key_tokens"]),
                     "B_keytok": key_token(b_ans, item["key_tokens"]),
                     "key_tokens": item["key_tokens"]})

    # ── LLM-as-judge (uses the same model) ─────────────────────────────────────
    def judge(q, gold, ans):
        out = client.call_batch([JUDGE.format(q=q, gold=gold, ans=ans)],
                                system="You are a strict grader.", max_tokens=12,
                                temperature=0.0, assistant_prefix=APREFIX)
        v = strip_think(out[0]).upper()
        return 1.0 if ("CORRECT" in v and "INCORRECT" not in v) else (0.5 if "PARTIAL" in v else 0.0)
    for r in rows:
        r["A_judge"] = judge(r["question"], r["gold"], r["A_ans"])
        r["B_judge"] = judge(r["question"], r["gold"], r["B_ans"])

    # ── Text metrics: BERT-F1, METEOR, COMET ───────────────────────────────────
    cands_A = [r["A_ans"] for r in rows]; cands_B = [r["B_ans"] for r in rows]
    refs = [r["gold"] for r in rows]; srcs = [r["question"] for r in rows]

    # BERTScore
    try:
        from bert_score import score as bert_score
        _, _, fA = bert_score(cands_A, refs, lang="en", verbose=False, rescale_with_baseline=False)
        _, _, fB = bert_score(cands_B, refs, lang="en", verbose=False, rescale_with_baseline=False)
        for r, a, b in zip(rows, fA.tolist(), fB.tolist()):
            r["A_bertf1"], r["B_bertf1"] = round(a, 4), round(b, 4)
    except Exception as e:
        logger.warning("BERTScore failed: %s", e)
        for r in rows: r["A_bertf1"] = r["B_bertf1"] = None

    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        for r in rows:
            r["A_meteor"] = round(meteor_score([r["gold"].split()], r["A_ans"].split()), 4)
            r["B_meteor"] = round(meteor_score([r["gold"].split()], r["B_ans"].split()), 4)
    except Exception as e:
        logger.warning("METEOR failed: %s", e)
        for r in rows: r["A_meteor"] = r["B_meteor"] = None

    # COMET (reference-based: src=question, mt=answer, ref=gold)
    try:
        from comet import download_model, load_from_checkpoint
        ckpt = download_model("Unbabel/wmt22-comet-da")
        cmodel = load_from_checkpoint(ckpt)
        dA = [{"src": s, "mt": m, "ref": rf} for s, m, rf in zip(srcs, cands_A, refs)]
        dB = [{"src": s, "mt": m, "ref": rf} for s, m, rf in zip(srcs, cands_B, refs)]
        oA = cmodel.predict(dA, batch_size=8, gpus=1, progress_bar=False)
        oB = cmodel.predict(dB, batch_size=8, gpus=1, progress_bar=False)
        for r, a, b in zip(rows, oA.scores, oB.scores):
            r["A_comet"], r["B_comet"] = round(a, 4), round(b, 4)
    except Exception as e:
        logger.warning("COMET failed: %s", e)
        for r in rows: r["A_comet"] = r["B_comet"] = None

    # ── aggregate ───────────────────────────────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None
    metrics = ["bertf1", "meteor", "comet", "keytok", "judge", "ttft", "tpot"]
    agg = {m: {"A": avg(f"A_{m}"), "B": avg(f"B_{m}")} for m in metrics}

    # nested cutoffs (100/200/500/1000) so one run reports multiple query counts
    def avg_sub(sub, key):
        vals = [r[key] for r in sub if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None
    by_cutoff = {}
    for n in [100, 200, 500, 1000]:
        if n <= len(rows):
            sub = rows[:n]
            by_cutoff[str(n)] = {m: {"A": avg_sub(sub, f"A_{m}"), "B": avg_sub(sub, f"B_{m}")} for m in metrics}

    report = {"experiment": "exp022_temporal_metrics", "date": datetime.now().isoformat(),
              "n_questions": len(rows), "llm_backend": llm.backend, "qset": args.qset,
              "model": args.model_key, "aggregate": agg, "by_cutoff": by_cutoff, "rows": rows}
    save_json(report, REPORTS / f"temporal_metrics_{args.tag}.json")
    _print(report)
    _plot(report, FIGURES / f"temporal_metrics_{args.tag}.png")


def _plot(rep, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        logger.warning("matplotlib unavailable: %s", e); return
    a = rep["aggregate"]
    qual = [("BERT-F1", "bertf1"), ("METEOR", "meteor"), ("COMET", "comet"),
            ("Key-token", "keytok"), ("LLM-judge", "judge")]
    lat = [("TTFT (ms)", "ttft"), ("TPOT (ms)", "tpot")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [3, 1.2]})

    # quality panel (0-1, higher better)
    labels = [l for l, _ in qual]
    A = [a[k]["A"] or 0 for _, k in qual]; B = [a[k]["B"] or 0 for _, k in qual]
    x = range(len(labels)); w = 0.38
    ax1.bar([i - w/2 for i in x], A, w, label="Case A — no ontology", color="#c0392b")
    ax1.bar([i + w/2 for i in x], B, w, label="Case B — with ontology", color="#27ae60")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels); ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Score (higher better)")
    ax1.set_title(f"Answer quality — Case A vs Case B  ({rep['n_questions']} questions)")
    for i, (va, vb) in enumerate(zip(A, B)):
        ax1.text(i - w/2, va + 0.015, f"{va:.2f}", ha="center", fontsize=8)
        ax1.text(i + w/2, vb + 0.015, f"{vb:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax1.legend(loc="lower right")

    # latency panel (ms, lower better)
    llabels = [l for l, _ in lat]
    LA = [a[k]["A"] or 0 for _, k in lat]; LB = [a[k]["B"] or 0 for _, k in lat]
    lx = range(len(llabels))
    ax2.bar([i - w/2 for i in lx], LA, w, label="Case A", color="#c0392b")
    ax2.bar([i + w/2 for i in lx], LB, w, label="Case B", color="#27ae60")
    ax2.set_xticks(list(lx)); ax2.set_xticklabels(llabels); ax2.set_ylabel("ms (lower better)")
    ax2.set_title("Latency")
    for i, (va, vb) in enumerate(zip(LA, LB)):
        ax2.text(i - w/2, va, f"{va:.0f}", ha="center", va="bottom", fontsize=8)
        ax2.text(i + w/2, vb, f"{vb:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.suptitle("Temporal Ontology Demo (comments.txt): vanilla LLM with vs without ontology",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200); plt.close(fig)
    logger.info("Figure → %s", out_path)


def _print(rep):
    G, B, RST, C, Y = "\033[92m", "\033[1m", "\033[0m", "\033[96m", "\033[93m"
    a = rep["aggregate"]
    labels = {"bertf1": "BERT-F1", "meteor": "METEOR", "comet": "COMET",
              "keytok": "Key-token", "judge": "LLM-judge", "ttft": "TTFT (ms)", "tpot": "TPOT (ms)"}
    higher_better = {"bertf1", "meteor", "comet", "keytok", "judge"}
    print(f"\n{B}{'='*70}{RST}")
    print(f"{B}  comments.txt METRICS — Case A (no onto) vs Case B (with onto){RST}")
    print(f"{B}{'='*70}{RST}")
    print(f"  LLM: {rep['llm_backend']} (vanilla) · {rep['n_questions']} tough questions · ontology v1.2.0\n")
    print(f"  {'Metric':<14}{'Case A':>12}{'Case B':>12}{'Better':>10}")
    print("  " + "-" * 46)
    for m, lab in labels.items():
        va, vb = a[m]["A"], a[m]["B"]
        if va is None or vb is None:
            print(f"  {lab:<14}{'n/a':>12}{'n/a':>12}{'—':>10}"); continue
        if m in higher_better:
            better = "B" if vb > va else ("A" if va > vb else "=")
        else:  # latency: lower better
            better = "B" if vb < va else ("A" if va < vb else "=")
        col = G if better == "B" else (Y if better == "A" else "")
        print(f"  {lab:<14}{va:>12.3f}{vb:>12.3f}{col}{better:>10}{RST}")
    print(f"\n  {C}Report → artifacts/reports/temporal_metrics_report.json{RST}")
    print(f"{B}{'='*70}{RST}")


if __name__ == "__main__":
    main()
