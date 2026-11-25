"""
Experiment 023 — Scaled ontology-RAG evaluation (Case A vs Case B) on the Jeju
test split, with all comments.txt metrics, reported at 100/200/500/1000 queries.

  Case A (no ontology): vanilla LLM answers the question alone.
  Case B (with ontology): LLM + Hybrid ontology v1.1.0 subgraph context (RAG).
  Gold = the dataset's reference Output.

Metrics (per query, aggregated over nested subsets): BERT-F1, METEOR, COMET,
Key-token (gold ontology entities present), LLM-judge, TTFT (ms), TPOT (ms).

Generates 1000 once per model; reports nested cutoffs. Run 4 models in parallel
(one GPU each). Reasoning models (DeepSeek-R1, Qwen3) use empty-think for concise
answers. proctitle = tahir_Ont-code.

Usage:
  CUDA_VISIBLE_DEVICES=6 python experiments/exp023_ontology_rag_scale/run_rag_scale.py \
      --model-key llama3.1-8b --tag llama --n 1000
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.proctitle import set_proctitle
from src.graph.graph_store import TourismGraphStore
from src.retrieval.subgraph_retriever import SubgraphRetriever
from src.utils.io import load_json, save_json, ensure_dir, load_jsonl
from src.utils.logging_utils import get_logger

logger = get_logger("exp023")

TEST = ROOT / "data" / "splits" / "test.jsonl"
ONT = ROOT / "artifacts" / "ontologies" / "1.1.0"
REPORTS = ROOT / "artifacts" / "reports"
CUTOFFS = [100, 200, 500, 1000]

SYSTEM = "You are a helpful Jeju Island (South Korea) tourism assistant. Answer directly; do not mention any provided facts, context, or ontology."
CASE_A = "Question: {q}\nAnswer directly and specifically."
CASE_B = ("Relevant Jeju facts:\n{ctx}\n\nUse the facts above where helpful. Answer the question "
          "directly and specifically. Do NOT mention these facts or any ontology in your answer.\n\n"
          "Question: {q}\nAnswer:")
JUDGE = ("Judge if the candidate answer is consistent and correct with respect to the reference "
         "answer for the question.\nQuestion: {q}\nReference: {gold}\nCandidate: {ans}\n"
         "Reply ONE word: CORRECT, PARTIAL, or INCORRECT. Verdict:")


def derive_key_tokens(gold: str, concept_terms: list[str]) -> list[str]:
    """Up to 5 distinctive ontology entities present in the gold answer."""
    g = gold.lower()
    hits = [t for t in concept_terms if re.search(rf"(?<![a-z]){re.escape(t.lower())}(?![a-z])", g)]
    hits = sorted(set(hits), key=len, reverse=True)
    return hits[:5]


def key_token_score(ans: str, toks: list[str]) -> float:
    if not toks:
        return None
    a = ans.lower()
    return sum(1 for t in toks if t.lower() in a) / len(toks)


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="llama3.1-8b")
    ap.add_argument("--tag", default="llama")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()
    ensure_dir(REPORTS)

    records = load_jsonl(TEST)[: args.n]
    logger.info("loaded %d test questions", len(records))

    # hybrid ontology → graph → retriever
    concepts = load_json(ONT / "concepts.json")
    relations = load_json(ONT / "relations.json")
    graph = TourismGraphStore(); graph.load(concepts, relations)
    retr = SubgraphRetriever(graph, use_ranker=True, max_seeds=5, max_top_seeds=3)
    concept_terms = [c["name"] for c in concepts if len(c.get("name", "")) >= 4
                     and c.get("concept_class") != "OntologyClass"]
    logger.info("ontology v1.1.0: %d concepts, %d relations", len(concepts), len(relations))

    from src.llm.client import LLMClient
    llm = LLMClient(backend="local", model_key=args.model_key, max_tokens=args.max_tokens, temperature=0.0)
    if not llm.is_available:
        logger.error("LLM unavailable"); sys.exit(1)
    client = llm._client
    is_reasoning = any(k in args.model_key.lower() for k in ("r1", "deepseek", "qwen3"))
    APREFIX = "<think>\n\n</think>\n\n" if is_reasoning else ""

    def strip_think(t):
        return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).replace("</think>", "").strip()

    def gen(prompt):
        with contextlib.redirect_stdout(io.StringIO()):
            ans = client.stream_call(prompt, system=SYSTEM, max_tokens=args.max_tokens,
                                     temperature=0.0, assistant_prefix=APREFIX)
        m = client._last_metrics
        return strip_think(ans), m.get("ttft_ms"), m.get("tpot_ms")

    rows = []
    for i, rec in enumerate(records, 1):
        q, gold = rec["Input"].strip(), rec["Output"].strip()
        a_ans, a_ttft, a_tpot = gen(CASE_A.format(q=q))
        try:
            ctx = retr.retrieve(q).to_text(max_relations=25)
        except Exception:
            ctx = "[no ontology context]"
        b_ans, b_ttft, b_tpot = gen(CASE_B.format(ctx=ctx[:1600], q=q))
        kt = derive_key_tokens(gold, concept_terms)
        rows.append({"id": rec.get("record_id"), "q": q, "gold": gold,
                     "A_ans": a_ans, "B_ans": b_ans, "key_tokens": kt,
                     "A_ttft": a_ttft, "A_tpot": a_tpot, "B_ttft": b_ttft, "B_tpot": b_tpot,
                     "A_keytok": key_token_score(a_ans, kt), "B_keytok": key_token_score(b_ans, kt)})
        if i % 100 == 0:
            logger.info("generated %d/%d", i, len(records))

    # ── LLM-judge ───────────────────────────────────────────────────────────────
    logger.info("judging ...")
    for j, r in enumerate(rows, 1):
        for case in ("A", "B"):
            out = client.call_batch([JUDGE.format(q=r["q"], gold=r["gold"][:600], ans=r[f"{case}_ans"])],
                                    system="You are a strict grader.", max_tokens=12,
                                    temperature=0.0, assistant_prefix=APREFIX)
            v = strip_think(out[0]).upper()
            r[f"{case}_judge"] = 1.0 if ("CORRECT" in v and "INCORRECT" not in v) else (0.5 if "PARTIAL" in v else 0.0)
        if j % 200 == 0:
            logger.info("judged %d/%d", j, len(rows))

    # ── text metrics (BERT-F1, METEOR, COMET) over ALL rows once ───────────────
    cA = [r["A_ans"] for r in rows]; cB = [r["B_ans"] for r in rows]
    refs = [r["gold"] for r in rows]; srcs = [r["q"] for r in rows]
    try:
        from bert_score import score as bsc
        _, _, fA = bsc(cA, refs, lang="en", verbose=False); _, _, fB = bsc(cB, refs, lang="en", verbose=False)
        for r, a, b in zip(rows, fA.tolist(), fB.tolist()): r["A_bertf1"], r["B_bertf1"] = a, b
    except Exception as e:
        logger.warning("BERT failed: %s", e)
        for r in rows: r["A_bertf1"] = r["B_bertf1"] = None
    try:
        from nltk.translate.meteor_score import meteor_score
        for r in rows:
            r["A_meteor"] = meteor_score([r["gold"].split()], r["A_ans"].split())
            r["B_meteor"] = meteor_score([r["gold"].split()], r["B_ans"].split())
    except Exception as e:
        logger.warning("METEOR failed: %s", e)
        for r in rows: r["A_meteor"] = r["B_meteor"] = None
    try:
        from comet import download_model, load_from_checkpoint
        cm = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        dA = [{"src": s, "mt": m, "ref": rf} for s, m, rf in zip(srcs, cA, refs)]
        dB = [{"src": s, "mt": m, "ref": rf} for s, m, rf in zip(srcs, cB, refs)]
        oA = cm.predict(dA, batch_size=16, gpus=1, progress_bar=False)
        oB = cm.predict(dB, batch_size=16, gpus=1, progress_bar=False)
        for r, a, b in zip(rows, oA.scores, oB.scores): r["A_comet"], r["B_comet"] = a, b
    except Exception as e:
        logger.warning("COMET failed: %s", e)
        for r in rows: r["A_comet"] = r["B_comet"] = None

    # ── aggregate at nested cutoffs ────────────────────────────────────────────
    metrics = ["bertf1", "meteor", "comet", "keytok", "judge", "ttft", "tpot"]
    def avg(subset, key):
        vals = [r[key] for r in subset if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None
    by_cutoff = {}
    for n in CUTOFFS:
        if n > len(rows):
            continue
        sub = rows[:n]
        by_cutoff[str(n)] = {m: {"A": avg(sub, f"A_{m}"), "B": avg(sub, f"B_{m}")} for m in metrics}

    report = {"experiment": "exp023_ontology_rag_scale", "date": datetime.now().isoformat(),
              "model": args.model_key, "n_total": len(rows), "ontology": "1.1.0",
              "by_cutoff": by_cutoff, "rows": rows}
    save_json(report, REPORTS / f"rag_scale_{args.tag}.json")
    logger.info("DONE → rag_scale_%s.json | cutoffs: %s", args.tag, list(by_cutoff.keys()))
    print(f"[exp023] {args.model_key}: {len(rows)} done, cutoffs {list(by_cutoff.keys())}")


if __name__ == "__main__":
    main()
