"""
Goal 2 — qualitative analysis of where Ontology grounding beats Simple RAG and
where it does not.

Quantitative tables say which arm wins on average; they do not say why. This
script pairs every item across the two arms, ranks by score gap, classifies the
failure/success mode, and writes both a machine-readable dump and a readable
markdown report with full side-by-side cases.

  python -m experiments.exp027_qualitative.analyse --split bench --onto-arm onto_v5
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp024_rag_vs_ontology import config as C
from src.utils.proctitle import set_proctitle
from streamlit_app.evaluation.metrics import derive_key_tokens

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
BOILER = {"morning", "afternoon", "evening", "night", "breakfast", "lunch", "dinner",
          "snack", "start", "arrival", "departure", "explore", "visit", "enjoy",
          "taking", "head", "return", "afterward", "optional", "day", "continue",
          "end", "check", "conclude", "begin", "stop", "travel", "activity",
          "arrive", "take", "city", "hike"}


def gold_entities(ref: str) -> list[str]:
    return [t for t in derive_key_tokens(ref) if t.lower() not in BOILER]


def hit(text: str, ents: list[str]) -> set[str]:
    low = (text or "").lower()
    return {e for e in ents if e.lower() in low}


def classify(ctx_r: str, ctx_o: str, ans_r: str, ans_o: str, ents: list[str]) -> str:
    """Why did one arm beat the other on this item?"""
    if not ents:
        return "no_gold_entities"
    cr, co = hit(ctx_r, ents), hit(ctx_o, ents)
    ar, ao = hit(ans_r, ents), hit(ans_o, ents)
    if co - cr and ao - ar:
        return "onto_retrieved_and_used"        # ontology surfaced facts RAG missed
    if co - cr and not (ao - ar):
        return "onto_retrieved_not_used"        # had the fact, failed to use it
    if cr - co and ar - ao:
        return "rag_retrieved_and_used"
    if cr - co and not (ar - ao):
        return "rag_retrieved_not_used"
    if not cr and not co:
        return "neither_retrieved"              # both blind; answers are guesses
    return "both_retrieved_similar"


def main() -> None:
    set_proctitle()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bench")
    ap.add_argument("--onto-arm", default="onto")
    ap.add_argument("--metric", default="judge")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    ctx = {}
    for arm in ("rag", a.onto_arm):
        p = C.CTX_DIR / f"{a.split}_{arm}.jsonl"
        ctx[arm] = {json.loads(l)["record_id"]: json.loads(l)["context"]
                    for l in open(p, encoding="utf-8")} if p.exists() else {}

    models = [m["key"] for m in C.MODELS]
    rows = []
    for mk in models:
        gr = C.GEN_DIR / f"{a.split}__{mk}__rag.jsonl"
        go = C.GEN_DIR / f"{a.split}__{mk}__{a.onto_arm}.jsonl"
        sr = C.SCORE_DIR / f"{a.split}__{mk}__rag.jsonl"
        so = C.SCORE_DIR / f"{a.split}__{mk}__{a.onto_arm}.jsonl"
        if not (gr.exists() and go.exists() and sr.exists() and so.exists()):
            continue
        G = {arm: {json.loads(l)["record_id"]: json.loads(l) for l in open(p, encoding="utf-8")}
             for arm, p in (("rag", gr), ("onto", go))}
        S = {arm: {json.loads(l)["record_id"]: json.loads(l) for l in open(p, encoding="utf-8")}
             for arm, p in (("rag", sr), ("onto", so))}
        for rid in G["rag"]:
            if rid not in G["onto"] or rid not in S["rag"] or rid not in S["onto"]:
                continue
            ref = G["rag"][rid]["reference"]
            ents = gold_entities(ref)
            ar, ao = G["rag"][rid]["answer"], G["onto"][rid]["answer"]
            cr = ctx["rag"].get(rid, "")
            co = ctx[a.onto_arm].get(rid, "")
            scores = {k: (S["rag"][rid].get(k), S["onto"][rid].get(k)) for k in C.METRICS}
            m = a.metric
            if not all(isinstance(x, (int, float)) for x in scores[m]):
                continue
            rows.append({
                "record_id": rid, "model": mk, "question": G["rag"][rid]["Input"],
                "reference": ref, "gold_entities": ents,
                "answer_rag": ar, "answer_onto": ao,
                "ctx_rag": cr, "ctx_onto": co,
                "scores": scores, "gap": scores[m][1] - scores[m][0],
                "ent_ctx_rag": sorted(hit(cr, ents)), "ent_ctx_onto": sorted(hit(co, ents)),
                "ent_ans_rag": sorted(hit(ar, ents)), "ent_ans_onto": sorted(hit(ao, ents)),
                "mode": classify(cr, co, ar, ao, ents)})
    if not rows:
        raise SystemExit("no paired cells found — generate and score first")

    modes = Counter(r["mode"] for r in rows)
    onto_win = sum(1 for r in rows if r["gap"] > 0)
    rag_win = sum(1 for r in rows if r["gap"] < 0)
    tie = sum(1 for r in rows if r["gap"] == 0)

    # per-mode average gap: which mechanism actually moves the score
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r["gap"])

    (OUT / f"qualitative_{a.split}_{a.onto_arm}.json").write_text(
        json.dumps({"n": len(rows), "modes": dict(modes),
                    "onto_win": onto_win, "rag_win": rag_win, "tie": tie,
                    "mode_mean_gap": {k: round(statistics.mean(v), 4)
                                      for k, v in by_mode.items()},
                    "rows": rows}, indent=1, default=str), encoding="utf-8")

    print(f"paired items: {len(rows):,}  (ontology better {onto_win}, "
          f"RAG better {rag_win}, tie {tie})  metric={a.metric}")
    print("\nmechanism breakdown:")
    for k, c in modes.most_common():
        print(f"  {k:28s} {c:6,}  ({100*c/len(rows):5.1f}%)  mean {a.metric} gap "
              f"{statistics.mean(by_mode[k]):+.4f}")
    print("\ntop ontology wins:")
    for r in sorted(rows, key=lambda x: -x["gap"])[:5]:
        print(f"  [{r['mode']}] {r['model']} gap {r['gap']:+.2f} :: {r['question'][:80]}")
    print("\ntop ontology losses:")
    for r in sorted(rows, key=lambda x: x["gap"])[:5]:
        print(f"  [{r['mode']}] {r['model']} gap {r['gap']:+.2f} :: {r['question'][:80]}")


if __name__ == "__main__":
    main()
